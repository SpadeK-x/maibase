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
from probe_structural_signals import compute_probe_metrics as compute_structural_probe_metrics, load_records
from probe_technical_signals import compute_technical_probe_metrics
from train_mvp_mlp import (
    DEFAULT_LABEL_CLASSES,
    build_prediction_rows,
    compute_class_weight,
    discover_samples,
    export_prediction_rows,
    fit_train_normalizer,
    print_confusion_and_metrics,
    print_dataset_distributions,
    set_seed,
    split_dataset,
    TrainConfig,
)


PROBE_PRESETS = {
    "v1": [
        "busy_density_mean",
        "busy_density_p90",
        "outer_move_ge_0_25_ratio",
        "span_jump_p90",
        "slide_conflict_when_busy_ratio",
        "busy_outer_move_p90",
    ],
    "v2": [
        "busy_density_mean",
        "busy_density_p90",
        "outer_move_ge_0_25_ratio",
        "outer_move_ge_0_375_ratio",
        "outer_move_p90",
        "span_jump_p90",
        "span_jump_ge_0_5_ratio",
        "slide_conflict_when_busy_ratio",
        "busy_outer_move_mean",
        "busy_outer_move_p90",
    ],
    "technical_v1": [
        "interval_cv",
        "interval_entropy",
        "rhythm_switch_ratio",
        "short_long_alternation_ratio",
        "low_density_rhythm_switch_ratio",
        "nonburst_rhythm_switch_ratio",
        "slide_density_mean",
        "slide_density_p90",
        "slide_outer_move_p90",
        "slide_compound_ratio",
        "slide_span_p90",
        "long_slide_ratio",
        "slide_conflict_active_ratio",
    ],
    "hybrid_v1": [
        "busy_density_mean",
        "busy_density_p90",
        "outer_move_ge_0_25_ratio",
        "span_jump_p90",
        "slide_conflict_when_busy_ratio",
        "busy_outer_move_p90",
        "rhythm_switch_ratio",
        "low_density_rhythm_switch_ratio",
        "slide_density_p90",
        "slide_outer_move_p90",
        "slide_span_p90",
        "slide_conflict_active_ratio",
    ],
}

DEFAULT_PROBE_FEATURES = PROBE_PRESETS["v1"]


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def resolve_probe_features(
    probe_preset: str,
    override_probe_features: Sequence[str],
) -> List[str]:
    if override_probe_features:
        return list(override_probe_features)
    if probe_preset not in PROBE_PRESETS:
        raise KeyError(f"Unknown probe preset: {probe_preset}")
    return list(PROBE_PRESETS[probe_preset])


def build_criterion(
    loss_type: str,
    class_weight: torch.Tensor,
    device: str,
    focal_gamma: float,
) -> nn.Module:
    class_weight = class_weight.to(device)
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weight)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, class_weight=class_weight)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


@dataclass
class ProbeNormalizerState:
    mean: torch.Tensor
    std: torch.Tensor


class ProbeFeatureNormalizer:
    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, feature_rows: Sequence[Sequence[float]], train_indices: Sequence[int]) -> None:
        values = torch.tensor([feature_rows[index] for index in train_indices], dtype=torch.float32)
        if values.numel() == 0:
            raise ValueError("ProbeFeatureNormalizer received no train values.")
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean = mean
        self.std = std

    def transform_row(self, row: Sequence[float]) -> torch.Tensor:
        values = torch.tensor(row, dtype=torch.float32)
        if self.mean is None or self.std is None:
            return values
        return (values - self.mean) / self.std

    def export_state(self) -> ProbeNormalizerState:
        if self.mean is None or self.std is None:
            raise ValueError("Probe normalizer has not been fitted.")
        return ProbeNormalizerState(self.mean.clone(), self.std.clone())


class LateFusionChartDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        probe_rows: Sequence[Sequence[float]],
        probe_normalizer: ProbeFeatureNormalizer,
    ) -> None:
        self.base_dataset = base_dataset
        self.probe_rows = [list(row) for row in probe_rows]
        self.probe_normalizer = probe_normalizer

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        encoded = self.base_dataset[index]
        probe = self.probe_normalizer.transform_row(self.probe_rows[index])
        return encoded.events, encoded.length, encoded.label, probe, dict(encoded.meta or {})


def collate_late_fusion_batch(batch):
    encoded_batch = []
    probes: List[torch.Tensor] = []
    for events, length, label, probe, meta in batch:
        encoded_batch.append(type("Obj", (), {"events": events, "length": length, "label": label, "meta": meta})())
        probes.append(probe)
    batch_x, lengths, labels, metas = collate_encoded_charts(encoded_batch)
    probe_x = torch.stack(probes, dim=0)
    return batch_x, lengths, labels, probe_x, metas


@dataclass
class LateFusionConfig:
    input_dim: int = MVPEventEncoder.input_dim
    probe_dim: int = len(DEFAULT_PROBE_FEATURES)
    num_classes: int = 4
    pooling: str = "mean_max"
    classifier_hidden_dim: int = 128
    dropout: float = 0.4


class LateFusionMLPClassifier(nn.Module):
    def __init__(self, config: LateFusionConfig) -> None:
        super().__init__()
        self.config = config
        mlp_config = MLPConfig(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            pooling=config.pooling,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout=config.dropout,
        )
        self.event_encoder = EventMLPEncoder(mlp_config)
        self.pooler = EventPooler(config.pooling)
        pooled_dim = mlp_config.event_embed_dim * (2 if config.pooling == "mean_max" else 1)
        fused_dim = pooled_dim + config.probe_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        probe_x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, T, F], got {tuple(x.shape)}")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected feature dim {self.config.input_dim}, got {x.size(-1)}")
        if probe_x.dim() != 2:
            raise ValueError(f"Expected probe_x to have shape [B, P], got {tuple(probe_x.shape)}")
        if probe_x.size(-1) != self.config.probe_dim:
            raise ValueError(f"Expected probe dim {self.config.probe_dim}, got {probe_x.size(-1)}")

        event_embeddings = self.event_encoder(x)
        chart_embedding = self.pooler(event_embeddings, mask)
        fused_embedding = torch.cat([chart_embedding, probe_x], dim=-1)
        logits = self.classifier(fused_embedding)
        return logits, fused_embedding, event_embeddings


def build_probe_rows(
    samples: Sequence[Dict[str, object]],
    probe_events_dir: Path,
    probe_feature_names: Sequence[str],
) -> List[List[float]]:
    rows: List[List[float]] = []
    for sample in samples:
        chart = str((sample.get("meta") or {}).get("chart", ""))
        if not chart:
            chart = Path(str(sample["path"])).stem
        records = load_records(probe_events_dir, chart)
        metrics = compute_structural_probe_metrics(records)
        metrics.update(compute_technical_probe_metrics(records))
        rows.append([float(metrics[name]) for name in probe_feature_names])
    return rows


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

    for batch_x, lengths, labels, probe_x, _ in loader:
        if labels is None:
            raise ValueError("Labels are required for training/evaluation.")
        batch_x = project_feature_tensor(batch_x, model.config.input_dim)
        batch_x = batch_x.to(device, non_blocking=True)
        batch_x = apply_zero_feature_mask(batch_x, None)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probe_x = probe_x.to(device, non_blocking=True)
        mask = make_padding_mask(lengths, batch_x.size(1))

        logits, _, _ = model(batch_x, probe_x, mask)
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
        for batch_x, lengths, labels, probe_x, metas in loader:
            if labels is None:
                continue
            batch_x = project_feature_tensor(batch_x, model.config.input_dim)
            batch_x = batch_x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probe_x = probe_x.to(device, non_blocking=True)
            mask = make_padding_mask(lengths, batch_x.size(1))

            logits, _, _ = model(batch_x, probe_x, mask)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    return torch.cat(all_preds), torch.cat(all_labels), all_metas


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MVP MLP classifier with late-fusion chart-level probe features.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--probe-events-dir", type=Path, default=Path("events_all"))
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
    parser.add_argument("--predictions-output", type=Path, default=Path("late_fusion_predictions_test.csv"))
    parser.add_argument("--misclassified-output", type=Path)
    parser.add_argument("--probe-preset", type=str, default="v1", choices=sorted(PROBE_PRESETS.keys()))
    parser.add_argument("--probe-features", nargs="*", default=None)
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
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
    fusion_dataset = LateFusionChartDataset(base_dataset, probe_rows, probe_normalizer)
    train_dataset = torch.utils.data.Subset(fusion_dataset, train_subset.indices)
    eval_dataset = torch.utils.data.Subset(fusion_dataset, eval_subset.indices)
    test_dataset = torch.utils.data.Subset(fusion_dataset, test_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_late_fusion_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_late_fusion_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_late_fusion_batch,
    )

    model = LateFusionMLPClassifier(
        LateFusionConfig(
            input_dim=MVPEventEncoder.input_dim,
            probe_dim=len(probe_features),
            num_classes=config.num_classes,
            pooling=config.pooling,
        )
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = build_criterion(args.loss_type, class_weight, config.device, args.focal_gamma)
    best_eval_loss = float("inf")
    best_state = None

    print("probe_preset", args.probe_preset)
    print("probe_features", probe_features)
    print("loss_type", args.loss_type)
    if args.loss_type == "focal":
        print("focal_gamma", args.focal_gamma)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

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
        }
        torch.save(payload, args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
