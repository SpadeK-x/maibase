import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from mvp_event_encoder import (
    EncodedChartDataset,
    MVPEventEncoder,
    PreencodedChartDataset,
    apply_zero_feature_mask,
    collate_encoded_charts,
    project_feature_tensor,
)
from mvp_mlp_model import EventMLPEncoder, EventPooler, MLPConfig, make_padding_mask
from probe_structural_signals import load_records
from train_mvp_mlp import (
    DEFAULT_LABEL_CLASSES,
    TrainConfig,
    build_prediction_rows,
    compute_class_weight,
    discover_samples,
    export_prediction_rows,
    fit_train_normalizer,
    print_confusion_and_metrics,
    print_dataset_distributions,
    set_seed,
    split_dataset,
)
from train_mvp_mlp_late_fusion import (
    PROBE_PRESETS,
    ProbeFeatureNormalizer,
    build_criterion,
    build_probe_rows,
    resolve_probe_features,
)


def build_segment_ids_from_records(
    records: Sequence[Dict[str, object]],
    segment_seconds: float,
    max_segments: int,
) -> torch.Tensor:
    if segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be positive.")
    if max_segments < 1:
        raise ValueError("max_segments must be at least 1.")

    raw_ids: List[int] = []
    for record in records:
        event_time = float(record.get("event_time", 0.0))
        raw_ids.append(int(max(0.0, event_time) // segment_seconds))

    if not raw_ids:
        return torch.zeros(0, dtype=torch.long)

    base_id = raw_ids[0]
    normalized = [min(segment_id - base_id, max_segments - 1) for segment_id in raw_ids]
    return torch.tensor(normalized, dtype=torch.long)


def build_segment_rows(
    samples: Sequence[Dict[str, object]],
    segment_events_dir: Path,
    segment_seconds: float,
    max_segments: int,
) -> List[torch.Tensor]:
    rows: List[torch.Tensor] = []
    for sample in samples:
        chart = str((sample.get("meta") or {}).get("chart", ""))
        if not chart:
            chart = Path(str(sample["path"])).stem
        records = load_records(segment_events_dir, chart)
        rows.append(build_segment_ids_from_records(records, segment_seconds, max_segments))
    return rows


class SegmentFusionChartDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        probe_rows: Sequence[Sequence[float]],
        segment_rows: Sequence[torch.Tensor],
        probe_normalizer: ProbeFeatureNormalizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.probe_rows = [list(row) for row in probe_rows]
        self.segment_rows = [row.clone().to(torch.long) for row in segment_rows]
        self.probe_normalizer = probe_normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        encoded = self.base_dataset[index]
        probe = self.probe_normalizer.transform_row(self.probe_rows[index])
        segment_ids = self.segment_rows[index]
        effective_len = min(int(encoded.length), int(segment_ids.numel()))
        events = encoded.events[:effective_len]
        segment_ids = segment_ids[:effective_len]
        meta = dict(encoded.meta or {})
        meta["segment_count"] = int(segment_ids.max().item()) + 1 if segment_ids.numel() > 0 else 0
        return events, effective_len, encoded.label, probe, segment_ids, meta


def collate_segment_fusion_batch(batch):
    encoded_batch = []
    probes: List[torch.Tensor] = []
    segment_rows: List[torch.Tensor] = []

    for events, length, label, probe, segment_ids, meta in batch:
        encoded_batch.append(type("Obj", (), {"events": events, "length": length, "label": label, "meta": meta})())
        probes.append(probe)
        segment_rows.append(segment_ids)

    batch_x, lengths, labels, metas = collate_encoded_charts(encoded_batch)
    max_len = batch_x.size(1)
    padded_segments = torch.full((len(segment_rows), max_len), -1, dtype=torch.long)
    for index, segment_ids in enumerate(segment_rows):
        size = min(max_len, int(segment_ids.numel()))
        if size > 0:
            padded_segments[index, :size] = segment_ids[:size]
    probe_x = torch.stack(probes, dim=0)
    return batch_x, lengths, labels, probe_x, padded_segments, metas


@dataclass
class SegmentFusionConfig:
    input_dim: int = MVPEventEncoder.input_dim
    probe_dim: int = len(PROBE_PRESETS["v1"])
    num_classes: int = 4
    pooling: str = "mean_max"
    classifier_hidden_dim: int = 128
    dropout: float = 0.4
    event_hidden_dim: int = 128
    event_embed_dim: int = 64
    segment_ff_dim: int = 128
    segment_num_heads: int = 4
    segment_num_layers: int = 2
    segment_dropout: float = 0.2
    max_segments: int = 64
    segment_branch_scale: float = 1.0


class SegmentFusionClassifier(nn.Module):
    def __init__(self, config: SegmentFusionConfig) -> None:
        super().__init__()
        self.config = config
        mlp_config = MLPConfig(
            input_dim=config.input_dim,
            event_hidden_dim=config.event_hidden_dim,
            event_embed_dim=config.event_embed_dim,
            classifier_hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pooling=config.pooling,
        )
        self.event_encoder = EventMLPEncoder(mlp_config)
        self.event_pooler = EventPooler(config.pooling)
        self.segment_pooler = EventPooler(config.pooling)
        self.segment_pos_embedding = nn.Embedding(config.max_segments, config.event_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.event_embed_dim,
            nhead=config.segment_num_heads,
            dim_feedforward=config.segment_ff_dim,
            dropout=config.segment_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.segment_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.segment_num_layers)

        pooled_dim = config.event_embed_dim * (2 if config.pooling == "mean_max" else 1)
        fused_dim = pooled_dim + pooled_dim + config.probe_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    def build_segment_sequence(
        self,
        event_embeddings: torch.Tensor,
        mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_segments: List[torch.Tensor] = []
        segment_masks: List[torch.Tensor] = []
        embed_dim = event_embeddings.size(-1)

        for batch_index in range(event_embeddings.size(0)):
            valid_mask = mask[batch_index]
            valid_embeddings = event_embeddings[batch_index][valid_mask]
            valid_segment_ids = segment_ids[batch_index][valid_mask]
            nonnegative_mask = valid_segment_ids >= 0
            valid_embeddings = valid_embeddings[nonnegative_mask]
            valid_segment_ids = valid_segment_ids[nonnegative_mask]

            if valid_embeddings.size(0) == 0 or valid_segment_ids.numel() == 0:
                segment_embed = torch.zeros(1, embed_dim, dtype=event_embeddings.dtype, device=event_embeddings.device)
                batch_segments.append(segment_embed)
                segment_masks.append(torch.tensor([True], dtype=torch.bool, device=event_embeddings.device))
                continue

            segment_count = int(valid_segment_ids.max().item()) + 1
            segment_sum = torch.zeros(
                segment_count,
                embed_dim,
                dtype=event_embeddings.dtype,
                device=event_embeddings.device,
            )
            segment_den = torch.zeros(segment_count, 1, dtype=event_embeddings.dtype, device=event_embeddings.device)
            ones = torch.ones(valid_embeddings.size(0), 1, dtype=event_embeddings.dtype, device=event_embeddings.device)
            segment_sum.index_add_(0, valid_segment_ids, valid_embeddings)
            segment_den.index_add_(0, valid_segment_ids, ones)
            segment_mean = segment_sum / segment_den.clamp_min(1.0)

            batch_segments.append(segment_mean)
            segment_masks.append(torch.ones(segment_count, dtype=torch.bool, device=event_embeddings.device))

        max_segments = max(segment.size(0) for segment in batch_segments)
        padded_segments = event_embeddings.new_zeros((len(batch_segments), max_segments, embed_dim))
        padded_mask = torch.zeros((len(batch_segments), max_segments), dtype=torch.bool, device=event_embeddings.device)

        for batch_index, segment_embed in enumerate(batch_segments):
            size = segment_embed.size(0)
            padded_segments[batch_index, :size] = segment_embed
            padded_mask[batch_index, :size] = segment_masks[batch_index]

        return padded_segments, padded_mask

    def forward(
        self,
        x: torch.Tensor,
        probe_x: torch.Tensor,
        segment_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, F], got {tuple(x.shape)}")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected feature dim {self.config.input_dim}, got {x.size(-1)}")
        if probe_x.dim() != 2 or probe_x.size(-1) != self.config.probe_dim:
            raise ValueError(f"Expected probe_x shape [B, {self.config.probe_dim}], got {tuple(probe_x.shape)}")
        if segment_ids.dim() != 2 or segment_ids.size(0) != x.size(0) or segment_ids.size(1) != x.size(1):
            raise ValueError(f"Expected segment_ids shape {(x.size(0), x.size(1))}, got {tuple(segment_ids.shape)}")

        event_embeddings = self.event_encoder(x)
        global_embedding = self.event_pooler(event_embeddings, mask)

        segment_embeddings, segment_mask = self.build_segment_sequence(event_embeddings, mask, segment_ids)
        positions = torch.arange(segment_embeddings.size(1), device=x.device).unsqueeze(0)
        positions = positions.clamp_max(self.config.max_segments - 1)
        segment_embeddings = segment_embeddings + self.segment_pos_embedding(positions)
        transformed_segments = self.segment_encoder(
            segment_embeddings,
            src_key_padding_mask=~segment_mask,
        )
        segment_chart_embedding = self.segment_pooler(transformed_segments, segment_mask)
        segment_chart_embedding = segment_chart_embedding * float(self.config.segment_branch_scale)

        fused_embedding = torch.cat([global_embedding, segment_chart_embedding, probe_x], dim=-1)
        logits = self.classifier(fused_embedding)
        return logits, fused_embedding, transformed_segments


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for batch_x, lengths, labels, probe_x, segment_ids, _ in loader:
        if labels is None:
            raise ValueError("Labels are required for training/evaluation.")
        batch_x = project_feature_tensor(batch_x, model.config.input_dim)
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x = apply_zero_feature_mask(batch_x, None)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probe_x = probe_x.to(device, non_blocking=True)
        segment_ids = segment_ids.to(device, non_blocking=True)
        mask = make_padding_mask(lengths, batch_x.size(1))

        logits, _, _ = model(batch_x, probe_x, segment_ids, mask)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item()
        total_acc += (preds == labels).float().mean().item()
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_acc / steps


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch_x, lengths, labels, probe_x, segment_ids, metas in loader:
            if labels is None:
                continue
            batch_x = project_feature_tensor(batch_x, model.config.input_dim)
            batch_x = batch_x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probe_x = probe_x.to(device, non_blocking=True)
            segment_ids = segment_ids.to(device, non_blocking=True)
            mask = make_padding_mask(lengths, batch_x.size(1))

            logits, _, _ = model(batch_x, probe_x, segment_ids, mask)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    if not all_preds:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), []
    return torch.cat(all_preds), torch.cat(all_labels), all_metas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a segment-level late-fusion MVP classifier with local sequence modeling."
    )
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--probe-events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--segment-events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=Path("segment_fusion_predictions_test.csv"))
    parser.add_argument("--misclassified-output", type=Path)
    parser.add_argument("--probe-preset", type=str, default="v1", choices=sorted(PROBE_PRESETS.keys()))
    parser.add_argument("--probe-features", nargs="*", default=None)
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--event-hidden-dim", type=int, default=128)
    parser.add_argument("--event-embed-dim", type=int, default=64)
    parser.add_argument("--segment-ff-dim", type=int, default=128)
    parser.add_argument("--segment-num-heads", type=int, default=4)
    parser.add_argument("--segment-num-layers", type=int, default=2)
    parser.add_argument("--segment-dropout", type=float, default=0.2)
    parser.add_argument("--segment-branch-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    probe_features = resolve_probe_features(args.probe_preset, args.probe_features or [])
    if not probe_features:
        raise ValueError("At least one probe feature is required.")

    config = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pooling=args.pooling,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {data_dir}")

    probe_rows = build_probe_rows(samples, args.probe_events_dir, probe_features)
    segment_rows = build_segment_rows(samples, args.segment_events_dir, args.segment_seconds, args.max_segments)

    if args.encoded_dir:
        base_dataset = PreencodedChartDataset(samples)
        total_size = len(base_dataset)
        train_size = int(total_size * config.train_ratio)
        eval_size = int(total_size * config.eval_ratio)
        test_size = total_size - train_size - eval_size
        train_subset, eval_subset, test_subset = random_split(
            base_dataset,
            [train_size, eval_size, test_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
        fake_dataset = EncodedChartDataset(samples, encoder=MVPEventEncoder())
        print_dataset_distributions(fake_dataset, train_subset, eval_subset, test_subset)
        class_weight = compute_class_weight(fake_dataset, train_subset)
    else:
        encoder = MVPEventEncoder()
        base_dataset = EncodedChartDataset(samples, encoder=encoder)
        train_subset, eval_subset, test_subset = split_dataset(base_dataset, config)
        print_dataset_distributions(base_dataset, train_subset, eval_subset, test_subset)
        fit_train_normalizer(base_dataset, train_subset)
        class_weight = compute_class_weight(base_dataset, train_subset)

    probe_normalizer = ProbeFeatureNormalizer()
    probe_normalizer.fit(probe_rows, train_subset.indices)
    fusion_dataset = SegmentFusionChartDataset(base_dataset, probe_rows, segment_rows, probe_normalizer)
    train_dataset = torch.utils.data.Subset(fusion_dataset, train_subset.indices)
    eval_dataset = torch.utils.data.Subset(fusion_dataset, eval_subset.indices)
    test_dataset = torch.utils.data.Subset(fusion_dataset, test_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_segment_fusion_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_segment_fusion_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_segment_fusion_batch,
    )

    model = SegmentFusionClassifier(
        SegmentFusionConfig(
            input_dim=MVPEventEncoder.input_dim,
            probe_dim=len(probe_features),
            num_classes=config.num_classes,
            pooling=config.pooling,
            event_hidden_dim=args.event_hidden_dim,
            event_embed_dim=args.event_embed_dim,
            segment_ff_dim=args.segment_ff_dim,
            segment_num_heads=args.segment_num_heads,
            segment_num_layers=args.segment_num_layers,
            segment_dropout=args.segment_dropout,
            max_segments=args.max_segments,
            segment_branch_scale=args.segment_branch_scale,
        )
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = build_criterion(args.loss_type, class_weight, config.device, args.focal_gamma)
    best_eval_loss = float("inf")
    best_state = None

    print("probe_preset", args.probe_preset)
    print("probe_features", probe_features)
    print("loss_type", args.loss_type)
    print("segment_seconds", args.segment_seconds)
    print("max_segments", args.max_segments)
    print("segment_num_layers", args.segment_num_layers)
    print("segment_num_heads", args.segment_num_heads)
    print("segment_branch_scale", args.segment_branch_scale)
    print("class_weight", class_weight.tolist())

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, config.device)
        with torch.no_grad():
            eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, None, config.device)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}"
        )
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, config.device)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    preds, labels, metas = collect_predictions(model, test_loader, config.device)
    print_confusion_and_metrics(preds, labels, DEFAULT_LABEL_CLASSES)

    prediction_rows = build_prediction_rows(preds, labels, metas, DEFAULT_LABEL_CLASSES)
    exported = export_prediction_rows(prediction_rows, args.predictions_output, misclassified_only=False)
    print(f"predictions_exported={exported}")
    print(f"predictions_output={args.predictions_output}")
    if args.misclassified_output:
        mis_exported = export_prediction_rows(prediction_rows, args.misclassified_output, misclassified_only=True)
        print(f"misclassified_exported={mis_exported}")
        print(f"misclassified_output={args.misclassified_output}")

    if args.save_model:
        payload = {
            "state_dict": model.state_dict(),
            "probe_preset": args.probe_preset,
            "probe_features": list(probe_features),
            "probe_mean": probe_normalizer.mean,
            "probe_std": probe_normalizer.std,
            "pooling": config.pooling,
            "input_dim": model.config.input_dim,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "segment_seconds": args.segment_seconds,
            "max_segments": args.max_segments,
            "segment_num_layers": args.segment_num_layers,
            "segment_num_heads": args.segment_num_heads,
            "segment_branch_scale": args.segment_branch_scale,
        }
        torch.save(payload, args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
