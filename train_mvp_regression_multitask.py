import argparse
import csv
import math
import random
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
    collate_encoded_charts,
    project_feature_tensor,
)
from mvp_mlp_model import EventMLPEncoder, EventPooler, MLPConfig, make_padding_mask
from train_mvp_mlp import DEFAULT_LABEL_CLASSES, print_confusion_and_metrics


@dataclass
class MultiTaskConfig:
    batch_size: int = 128
    eval_batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    seed: int = 42
    pooling: str = "mean_max"
    input_dim: int = MVPEventEncoder.input_dim
    num_classes: int = 4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = torch.cuda.is_available()
    cls_loss_weight: float = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_level_map(labels_csv: Path) -> Dict[str, float]:
    level_map: Dict[str, float] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chart = row.get("chart")
            level = row.get("level")
            if not chart or level is None:
                continue
            try:
                numeric = float(level)
            except ValueError:
                continue
            if 13.0 <= numeric < 15.0:
                level_map[chart] = numeric
    return level_map


def bucket_level(raw_level: float) -> Optional[int]:
    if raw_level < 13.0 or raw_level >= 15.0:
        return None
    integer_part = int(raw_level)
    frac = round(raw_level - integer_part, 1)
    if integer_part == 13:
        return 0 if frac <= 0.5 + 1e-6 else 1
    if integer_part == 14:
        return 2 if frac <= 0.5 + 1e-6 else 3
    return None


def discover_samples(data_dir: Path, labels_csv: Path, file_suffixes: Sequence[str]) -> List[Dict[str, object]]:
    level_map = load_raw_level_map(labels_csv)
    samples: List[Dict[str, object]] = []
    paths: List[Path] = []
    for suffix in file_suffixes:
        paths.extend(data_dir.glob(f"*{suffix}"))
    for data_path in sorted(paths):
        chart_name = data_path.stem
        if chart_name not in level_map:
            continue
        raw_level = float(level_map[chart_name])
        bucket = bucket_level(raw_level)
        if bucket is None:
            continue
        samples.append(
            {
                "path": str(data_path),
                "target": raw_level,
                "label": bucket,
                "meta": {"chart": chart_name},
            }
        )
    return samples


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        raw_targets: Sequence[float],
        class_targets: Sequence[int],
        target_mean: float,
        target_std: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.raw_targets = list(raw_targets)
        self.class_targets = list(class_targets)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        encoded = self.base_dataset[index]
        raw_target = float(self.raw_targets[index])
        class_target = int(self.class_targets[index])
        norm_target = (raw_target - self.target_mean) / self.target_std
        meta = dict(encoded.meta or {})
        meta["target_level"] = raw_target
        return encoded.events, encoded.length, norm_target, class_target, meta


def collate_multitask_batch(batch):
    encoded_batch = []
    reg_targets: List[float] = []
    cls_targets: List[int] = []
    metas: List[Dict[str, object]] = []
    for events, length, reg_target, cls_target, meta in batch:
        encoded_batch.append(type("Obj", (), {"events": events, "length": length, "label": 0, "meta": meta})())
        reg_targets.append(float(reg_target))
        cls_targets.append(int(cls_target))
        metas.append(meta)
    batch_x, lengths, _, _ = collate_encoded_charts(encoded_batch)
    reg_tensor = torch.tensor(reg_targets, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_targets, dtype=torch.long)
    return batch_x, lengths, reg_tensor, cls_tensor, metas


def fit_train_normalizer(dataset: EncodedChartDataset, train_subset) -> None:
    train_paths = [Path(dataset.samples[i]["path"]) for i in train_subset.indices]
    dataset.encoder.fit_normalizer_from_paths(train_paths)


def build_base_datasets(samples: Sequence[Dict[str, object]], use_encoded: bool):
    stripped = [{"path": s["path"], "label": 0, "meta": s.get("meta")} for s in samples]
    if use_encoded:
        base_dataset = PreencodedChartDataset(stripped)
        fake_dataset = EncodedChartDataset(stripped, encoder=MVPEventEncoder())
        return base_dataset, fake_dataset
    encoder = MVPEventEncoder()
    base_dataset = EncodedChartDataset(stripped, encoder=encoder)
    return base_dataset, base_dataset


def split_dataset(dataset: Dataset, config: MultiTaskConfig):
    total_size = len(dataset)
    train_size = int(total_size * config.train_ratio)
    eval_size = int(total_size * config.eval_ratio)
    test_size = total_size - train_size - eval_size
    return random_split(
        dataset,
        [train_size, eval_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )


def fit_target_normalizer(targets: Sequence[float], train_indices: Sequence[int]) -> Tuple[float, float]:
    values = torch.tensor([float(targets[i]) for i in train_indices], dtype=torch.float32)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if std < 1e-6:
        std = 1.0
    return mean, std


class MVPRegressionMultiTaskModel(nn.Module):
    def __init__(self, input_dim: int, pooling: str = "mean_max", num_classes: int = 4) -> None:
        super().__init__()
        config = MLPConfig(input_dim=input_dim, pooling=pooling, num_classes=num_classes)
        self.event_encoder = EventMLPEncoder(config)
        self.pooler = EventPooler(config.pooling)
        pooled_dim = config.event_embed_dim * (2 if pooling == "mean_max" else 1)
        self.reg_head = nn.Sequential(
            nn.Linear(pooled_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(pooled_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        event_embeddings = self.event_encoder(x)
        chart_embedding = self.pooler(event_embeddings, mask)
        reg_pred = self.reg_head(chart_embedding).squeeze(-1)
        cls_logits = self.cls_head(chart_embedding)
        return reg_pred, cls_logits, chart_embedding, event_embeddings


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    reg_criterion: nn.Module,
    cls_criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    cls_loss_weight: float,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_reg_mae = 0.0
    total_cls_acc = 0.0
    steps = 0

    for batch_x, lengths, reg_targets, cls_targets, _ in loader:
        batch_x = project_feature_tensor(batch_x, model.event_encoder.config.input_dim)
        batch_x = batch_x.to(device)
        lengths = lengths.to(device)
        reg_targets = reg_targets.to(device)
        cls_targets = cls_targets.to(device)
        mask = make_padding_mask(lengths, batch_x.size(1))

        reg_pred, cls_logits, _, _ = model(batch_x, mask)
        reg_loss = reg_criterion(reg_pred, reg_targets)
        cls_loss = cls_criterion(cls_logits, cls_targets)
        loss = reg_loss + cls_loss_weight * cls_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_reg_mae += torch.mean(torch.abs(reg_pred - reg_targets)).item()
        total_cls_acc += (cls_logits.argmax(dim=-1) == cls_targets).float().mean().item()
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
    return total_loss / steps, total_reg_mae / steps, total_cls_acc / steps


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_mean: float,
    target_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    model.eval()
    all_reg_preds: List[torch.Tensor] = []
    all_reg_targets: List[torch.Tensor] = []
    all_cls_preds: List[torch.Tensor] = []
    all_cls_targets: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch_x, lengths, reg_targets, cls_targets, metas in loader:
            batch_x = project_feature_tensor(batch_x, model.event_encoder.config.input_dim)
            batch_x = batch_x.to(device)
            lengths = lengths.to(device)
            mask = make_padding_mask(lengths, batch_x.size(1))
            reg_pred, cls_logits, _, _ = model(batch_x, mask)
            all_reg_preds.append(reg_pred.cpu() * target_std + target_mean)
            all_reg_targets.append(reg_targets.cpu() * target_std + target_mean)
            all_cls_preds.append(cls_logits.argmax(dim=-1).cpu())
            all_cls_targets.append(cls_targets.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    return (
        torch.cat(all_reg_preds),
        torch.cat(all_reg_targets),
        torch.cat(all_cls_preds),
        torch.cat(all_cls_targets),
        all_metas,
    )


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(torch.mean((preds - targets) ** 2).item())


def export_predictions(
    reg_preds: torch.Tensor,
    reg_targets: torch.Tensor,
    cls_preds: torch.Tensor,
    cls_targets: torch.Tensor,
    metas: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chart",
                "true_level",
                "pred_level",
                "true_label_from_level",
                "pred_label_from_level",
                "pred_label_from_cls_head",
                "path",
            ],
        )
        writer.writeheader()
        for reg_pred, reg_target, cls_pred, cls_target, meta in zip(
            reg_preds.tolist(),
            reg_targets.tolist(),
            cls_preds.tolist(),
            cls_targets.tolist(),
            metas,
        ):
            writer.writerow(
                {
                    "chart": str(meta.get("chart", "")),
                    "true_level": f"{reg_target:.4f}",
                    "pred_level": f"{reg_pred:.4f}",
                    "true_label_from_level": DEFAULT_LABEL_CLASSES[int(cls_target)],
                    "pred_label_from_level": DEFAULT_LABEL_CLASSES[bucket_level(float(reg_pred)) or 0],
                    "pred_label_from_cls_head": DEFAULT_LABEL_CLASSES[int(cls_pred)],
                    "path": str(meta.get("path", "")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MVP MLP multitask regressor/classifier for raw chart level prediction.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=Path("multitask_predictions_test.csv"))
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    config = MultiTaskConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pooling=args.pooling,
        seed=args.seed,
        num_workers=args.num_workers,
        cls_loss_weight=args.cls_loss_weight,
    )
    set_seed(config.seed)

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No multitask samples found in {data_dir}")

    use_encoded = args.encoded_dir is not None
    base_dataset, fit_dataset = build_base_datasets(samples, use_encoded=use_encoded)
    initial_train_subset, _, _ = split_dataset(base_dataset, config)
    target_mean, target_std = fit_target_normalizer(
        [float(sample["target"]) for sample in samples],
        initial_train_subset.indices,
    )
    print(f"target_mean={target_mean:.4f} target_std={target_std:.4f}")

    if not use_encoded:
        fit_train_normalizer(fit_dataset, initial_train_subset)

    multitask_dataset = MultiTaskDataset(
        base_dataset,
        [float(sample["target"]) for sample in samples],
        [int(sample["label"]) for sample in samples],
        target_mean=target_mean,
        target_std=target_std,
    )
    train_subset, eval_subset, test_subset = split_dataset(multitask_dataset, config)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_multitask_batch,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_multitask_batch,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_multitask_batch,
    )

    model = MVPRegressionMultiTaskModel(
        input_dim=config.input_dim,
        pooling=config.pooling,
        num_classes=config.num_classes,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.CrossEntropyLoss()

    best_eval_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss, train_reg_mae, train_cls_acc = run_epoch(
            model,
            train_loader,
            reg_criterion,
            cls_criterion,
            optimizer,
            config.device,
            config.cls_loss_weight,
        )
        with torch.no_grad():
            eval_loss, eval_reg_mae, eval_cls_acc = run_epoch(
                model,
                eval_loader,
                reg_criterion,
                cls_criterion,
                None,
                config.device,
                config.cls_loss_weight,
            )

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_reg_mae={train_reg_mae:.4f} train_cls_acc={train_cls_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_reg_mae={eval_reg_mae:.4f} eval_cls_acc={eval_cls_acc:.4f}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    reg_preds, reg_targets, cls_preds, cls_targets, metas = collect_predictions(
        model,
        test_loader,
        config.device,
        target_mean=target_mean,
        target_std=target_std,
    )
    print(f"test_reg_mae={torch.mean(torch.abs(reg_preds - reg_targets)).item():.4f}")
    print(f"test_reg_rmse={rmse(reg_preds, reg_targets):.4f}")

    level_bucket_preds: List[int] = []
    level_bucket_targets: List[int] = []
    for pred, target in zip(reg_preds.tolist(), reg_targets.tolist()):
        pred_bucket = bucket_level(float(pred))
        target_bucket = bucket_level(float(target))
        if pred_bucket is None or target_bucket is None:
            continue
        level_bucket_preds.append(pred_bucket)
        level_bucket_targets.append(target_bucket)

    print("regression_bucket_confusion")
    print_confusion_and_metrics(
        torch.tensor(level_bucket_preds, dtype=torch.long),
        torch.tensor(level_bucket_targets, dtype=torch.long),
        DEFAULT_LABEL_CLASSES,
    )

    print("classification_head_confusion")
    print_confusion_and_metrics(cls_preds, cls_targets, DEFAULT_LABEL_CLASSES)

    export_predictions(
        reg_preds,
        reg_targets,
        cls_preds,
        cls_targets,
        metas,
        args.predictions_output,
    )
    print(f"predictions_output={args.predictions_output}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
