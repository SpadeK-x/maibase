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
from mvp_mlp_model import build_model, make_padding_mask
from train_mvp_mlp import DEFAULT_LABEL_CLASSES, print_confusion_and_metrics


@dataclass
class RegressionConfig:
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
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = torch.cuda.is_available()


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


def discover_regression_samples(data_dir: Path, labels_csv: Path, file_suffixes: Sequence[str]) -> List[Dict[str, object]]:
    level_map = load_raw_level_map(labels_csv)
    samples: List[Dict[str, object]] = []
    paths: List[Path] = []
    for suffix in file_suffixes:
        paths.extend(data_dir.glob(f"*{suffix}"))
    for data_path in sorted(paths):
        chart_name = data_path.stem
        if chart_name not in level_map:
            continue
        samples.append(
            {
                "path": str(data_path),
                "target": float(level_map[chart_name]),
                "meta": {"chart": chart_name},
            }
        )
    return samples


class RegressionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        targets: Sequence[float],
        target_mean: float,
        target_std: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.targets = list(targets)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        encoded = self.base_dataset[index]
        meta = dict(encoded.meta or {})
        raw_target = float(self.targets[index])
        meta["target_level"] = raw_target
        normalized_target = (raw_target - self.target_mean) / self.target_std
        return encoded.events, encoded.length, normalized_target, meta


def collate_regression_batch(batch):
    encoded_batch = []
    targets: List[float] = []
    metas: List[Dict[str, object]] = []
    for events, length, target, meta in batch:
        encoded_batch.append(type("Obj", (), {"events": events, "length": length, "label": 0, "meta": meta})())
        targets.append(float(target))
        metas.append(meta)
    batch_x, lengths, _, _ = collate_encoded_charts(encoded_batch)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    return batch_x, lengths, target_tensor, metas


def fit_train_normalizer(dataset: EncodedChartDataset, train_subset) -> None:
    train_paths = [Path(dataset.samples[i]["path"]) for i in train_subset.indices]
    dataset.encoder.fit_normalizer_from_paths(train_paths)


def build_regression_datasets(samples: Sequence[Dict[str, object]], use_encoded: bool):
    if use_encoded:
        base_dataset = PreencodedChartDataset(
            [{"path": sample["path"], "label": 0, "meta": sample.get("meta")} for sample in samples]
        )
        fake_dataset = EncodedChartDataset(
            [{"path": sample["path"], "label": 0, "meta": sample.get("meta")} for sample in samples],
            encoder=MVPEventEncoder(),
        )
        return base_dataset, fake_dataset

    encoder = MVPEventEncoder()
    base_dataset = EncodedChartDataset(
        [{"path": sample["path"], "label": 0, "meta": sample.get("meta")} for sample in samples],
        encoder=encoder,
    )
    return base_dataset, base_dataset


def split_dataset(dataset: Dataset, config: RegressionConfig):
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
    total_mae = 0.0
    steps = 0

    for batch_x, lengths, targets, _ in loader:
        batch_x = project_feature_tensor(batch_x, model.config.input_dim)
        batch_x = batch_x.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        mask = make_padding_mask(lengths, batch_x.size(1))

        preds, _, _ = model(batch_x, mask)
        preds = preds.squeeze(-1)
        loss = criterion(preds, targets)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mae += torch.mean(torch.abs(preds - targets)).item()
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_mae / steps


def collect_regression_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_mean: float,
    target_std: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch_x, lengths, targets, metas in loader:
            batch_x = project_feature_tensor(batch_x, model.config.input_dim)
            batch_x = batch_x.to(device)
            lengths = lengths.to(device)
            mask = make_padding_mask(lengths, batch_x.size(1))
            preds, _, _ = model(batch_x, mask)
            preds = preds.squeeze(-1).cpu() * target_std + target_mean
            raw_targets = targets.cpu() * target_std + target_mean
            all_preds.append(preds)
            all_targets.append(raw_targets)
            all_metas.extend(dict(meta or {}) for meta in metas)

    return torch.cat(all_preds), torch.cat(all_targets), all_metas


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return math.sqrt(torch.mean((preds - targets) ** 2).item())


def export_regression_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metas: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chart", "true_level", "pred_level", "true_label", "pred_label", "path"],
        )
        writer.writeheader()
        for pred, target, meta in zip(preds.tolist(), targets.tolist(), metas):
            true_bucket = bucket_level(float(target))
            pred_bucket = bucket_level(float(pred))
            writer.writerow(
                {
                    "chart": str(meta.get("chart", "")),
                    "true_level": f"{target:.4f}",
                    "pred_level": f"{pred:.4f}",
                    "true_label": DEFAULT_LABEL_CLASSES[true_bucket] if true_bucket is not None else "",
                    "pred_label": DEFAULT_LABEL_CLASSES[pred_bucket] if pred_bucket is not None else "",
                    "path": str(meta.get("path", "")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MVP MLP regressor for raw chart level prediction.")
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
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--predictions-output", type=Path, default=Path("regression_predictions_test.csv"))
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    config = RegressionConfig(
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
    samples = discover_regression_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No regression samples found in {data_dir}")

    use_encoded = args.encoded_dir is not None
    base_dataset, fit_dataset = build_regression_datasets(samples, use_encoded=use_encoded)
    initial_train_subset, _, _ = split_dataset(base_dataset, config)
    target_mean, target_std = fit_target_normalizer(
        [float(sample["target"]) for sample in samples],
        initial_train_subset.indices,
    )
    print(f"target_mean={target_mean:.4f} target_std={target_std:.4f}")

    if not use_encoded:
        fit_train_normalizer(fit_dataset, initial_train_subset)

    regression_dataset = RegressionDataset(
        base_dataset,
        [float(sample["target"]) for sample in samples],
        target_mean=target_mean,
        target_std=target_std,
    )
    train_subset, eval_subset, test_subset = split_dataset(regression_dataset, config)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_regression_batch,
    )
    eval_loader = DataLoader(
        eval_subset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_regression_batch,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_regression_batch,
    )

    model = build_model(input_dim=config.input_dim, num_classes=1, pooling=config.pooling).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_eval_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss, train_mae = run_epoch(model, train_loader, criterion, optimizer, config.device)
        with torch.no_grad():
            eval_loss, eval_mae = run_epoch(model, eval_loader, criterion, None, config.device)

        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
            f"eval_loss={eval_loss:.4f} eval_mae={eval_mae:.4f}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    preds, targets, metas = collect_regression_predictions(
        model,
        test_loader,
        config.device,
        target_mean=target_mean,
        target_std=target_std,
    )
    mae_value = torch.mean(torch.abs(preds - targets)).item()
    rmse_value = rmse(preds, targets)
    print(f"test_mae={mae_value:.4f}")
    print(f"test_rmse={rmse_value:.4f}")

    pred_buckets = []
    true_buckets = []
    for pred, target in zip(preds.tolist(), targets.tolist()):
        pred_bucket = bucket_level(float(pred))
        true_bucket = bucket_level(float(target))
        if pred_bucket is None or true_bucket is None:
            continue
        pred_buckets.append(pred_bucket)
        true_buckets.append(true_bucket)
    print_confusion_and_metrics(
        torch.tensor(pred_buckets, dtype=torch.long),
        torch.tensor(true_buckets, dtype=torch.long),
        DEFAULT_LABEL_CLASSES,
    )

    export_regression_predictions(preds, targets, metas, args.predictions_output)
    print(f"predictions_output={args.predictions_output}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
