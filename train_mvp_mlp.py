import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mvp_event_encoder import (
    EncodedChartDataset,
    MVPEventEncoder,
    PreencodedChartDataset,
    apply_zero_feature_mask,
    collate_encoded_charts,
    get_numeric_feature_indices,
    project_feature_tensor,
)
from mvp_mlp_model import build_model, make_padding_mask


DEFAULT_LABEL_CLASSES = ["13", "13+", "14", "14+"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(DEFAULT_LABEL_CLASSES)}
ABLATION_PRESETS = {
    "baseline_21_fields": ["slide_conflict_load", "hand_span_pressure"],
}


@dataclass
class TrainConfig:
    batch_size: int = 128
    eval_batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    seed: int = 42
    pooling: str = "mean"
    input_dim: int = MVPEventEncoder.input_dim
    num_classes: int = 4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = torch.cuda.is_available()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bucket_level(raw_level: str) -> Optional[int]:
    """
    Buckets raw level values into the 4-class scheme:
      13.0~13.5 -> 13
      13.6~13.9 -> 13+
      14.0~14.5 -> 14
      14.6~14.9 -> 14+
    """

    try:
        level = float(raw_level)
    except ValueError:
        return None

    if level < 13.0 or level >= 15.0:
        return None

    integer_part = int(level)
    frac = round(level - integer_part, 1)

    if integer_part == 13:
        if frac <= 0.5 + 1e-6:
            return LABEL_TO_INDEX["13"]
        return LABEL_TO_INDEX["13+"]

    if integer_part == 14:
        if frac <= 0.5 + 1e-6:
            return LABEL_TO_INDEX["14"]
        return LABEL_TO_INDEX["14+"]

    return None


def load_label_map(labels_csv: Path) -> Dict[str, int]:
    label_map: Dict[str, int] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chart = row.get("chart")
            level = row.get("level")
            if not chart or level is None:
                continue
            label = bucket_level(level)
            if label is None:
                continue
            label_map[chart] = label
    return label_map


def discover_samples(data_dir: Path, labels_csv: Path, file_suffixes: Sequence[str]) -> List[Dict[str, object]]:
    label_map = load_label_map(labels_csv)
    samples: List[Dict[str, object]] = []

    paths: List[Path] = []
    for suffix in file_suffixes:
        paths.extend(data_dir.glob(f"*{suffix}"))
    paths = sorted(paths)
    for data_path in paths:
        chart_name = data_path.stem
        if chart_name not in label_map:
            continue
        samples.append(
            {
                "path": str(data_path),
                "label": label_map[chart_name],
                "meta": {"chart": chart_name},
            }
        )

    return samples


def count_label_distribution(samples: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = {name: 0 for name in DEFAULT_LABEL_CLASSES}
    for sample in samples:
        label = int(sample["label"])
        counts[DEFAULT_LABEL_CLASSES[label]] += 1
    return counts


def count_subset_distribution(dataset: EncodedChartDataset, indices: Sequence[int]) -> Dict[str, int]:
    samples = [dataset.samples[i] for i in indices]
    return count_label_distribution(samples)


def print_label_distribution(title: str, counts: Dict[str, int]) -> None:
    total = sum(counts.values())
    formatted = " ".join(f"{name}={count}" for name, count in counts.items())
    print(f"{title} total={total} {formatted}")


def split_dataset(dataset: EncodedChartDataset, config: TrainConfig):
    total_size = len(dataset)
    train_size = int(total_size * config.train_ratio)
    eval_size = int(total_size * config.eval_ratio)
    test_size = total_size - train_size - eval_size
    return random_split(
        dataset,
        [train_size, eval_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )


def build_loaders(dataset: EncodedChartDataset, config: TrainConfig):
    train_set, eval_set, test_set = split_dataset(dataset, config)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )
    return train_loader, eval_loader, test_loader


def fit_train_normalizer(dataset: EncodedChartDataset, train_subset) -> None:
    train_paths = [Path(dataset.samples[i]["path"]) for i in train_subset.indices]
    dataset.encoder.fit_normalizer_from_paths(train_paths)


def print_dataset_distributions(dataset: EncodedChartDataset, train_subset, eval_subset, test_subset) -> None:
    print_label_distribution("all", count_label_distribution(dataset.samples))
    print_label_distribution("train", count_subset_distribution(dataset, train_subset.indices))
    print_label_distribution("eval", count_subset_distribution(dataset, eval_subset.indices))
    print_label_distribution("test", count_subset_distribution(dataset, test_subset.indices))


def compute_class_weight(dataset: EncodedChartDataset, train_subset) -> torch.Tensor:
    counts = count_subset_distribution(dataset, train_subset.indices)
    raw = torch.tensor([counts[name] for name in DEFAULT_LABEL_CLASSES], dtype=torch.float32)
    total = raw.sum()
    weights = total / (len(DEFAULT_LABEL_CLASSES) * raw.clamp_min(1.0))
    return weights


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    zero_feature_indices: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for batch_x, lengths, labels, _ in loader:
        if labels is None:
            raise ValueError("Labels are required for training/evaluation.")

        batch_x = project_feature_tensor(batch_x, config.input_dim)
        batch_x = batch_x.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        batch_x = apply_zero_feature_mask(batch_x, zero_feature_indices)
        mask = make_padding_mask(lengths, batch_x.size(1))

        logits, _, _ = model(batch_x, mask)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, labels)
        steps += 1

    if steps == 0:
        return 0.0, 0.0
    return total_loss / steps, total_acc / steps


def train_model(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: TrainConfig,
    class_weight: Optional[torch.Tensor] = None,
    zero_feature_indices: Optional[torch.Tensor] = None,
):
    model = build_model(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        pooling=config.pooling,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    if class_weight is not None:
        class_weight = class_weight.to(config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    best_eval_loss = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            zero_feature_indices=zero_feature_indices,
        )
        with torch.no_grad():
            eval_loss, eval_acc = run_epoch(
                model,
                eval_loader,
                criterion,
                None,
                config.device,
                zero_feature_indices=zero_feature_indices,
            )

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
    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    zero_feature_indices: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return run_epoch(model, loader, criterion, None, config.device, zero_feature_indices=zero_feature_indices)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    zero_feature_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch_x, lengths, labels, metas in loader:
            if labels is None:
                continue

            batch_x = project_feature_tensor(batch_x, model.config.input_dim)
            batch_x = batch_x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_x = apply_zero_feature_mask(batch_x, zero_feature_indices)
            mask = make_padding_mask(lengths, batch_x.size(1))

            logits, _, _ = model(batch_x, mask)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    if not all_preds:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), []
    return torch.cat(all_preds), torch.cat(all_labels), all_metas


def resolve_chart_name(meta: Dict[str, object]) -> str:
    chart = meta.get("chart")
    if chart:
        return str(chart)
    path_value = meta.get("path")
    if path_value:
        return Path(str(path_value)).stem
    return ""


def build_prediction_rows(
    preds: torch.Tensor,
    labels: torch.Tensor,
    metas: Sequence[Dict[str, object]],
    class_names: Sequence[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for pred, label, meta in zip(preds.tolist(), labels.tolist(), metas):
        rows.append(
            {
                "chart": resolve_chart_name(meta),
                "true_label": class_names[int(label)],
                "pred_label": class_names[int(pred)],
                "path": str(meta.get("path", "")),
            }
        )
    return rows


def export_prediction_rows(rows: Sequence[Dict[str, str]], output_path: Path, misclassified_only: bool = True) -> int:
    export_rows = [
        row for row in rows
        if (row["true_label"] != row["pred_label"]) or not misclassified_only
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chart", "true_label", "pred_label", "path"])
        writer.writeheader()
        writer.writerows(export_rows)
    return len(export_rows)


def parse_disable_numeric_fields(
    preset: Optional[str],
    disable_numeric_fields: Sequence[str],
) -> List[str]:
    disabled: List[str] = []
    if preset:
        disabled.extend(ABLATION_PRESETS[preset])
    disabled.extend(disable_numeric_fields)
    result: List[str] = []
    seen = set()
    for name in disabled:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def print_confusion_and_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
) -> None:
    num_classes = len(class_names)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for y_true, y_pred in zip(labels, preds):
        cm[int(y_true), int(y_pred)] += 1

    print("confusion_matrix")
    print(cm)

    print("per_class_recall")
    for i, name in enumerate(class_names):
        tp = cm[i, i].item()
        total_true = cm[i].sum().item()
        recall = tp / total_true if total_true > 0 else 0.0
        print(f"{name}: recall={recall:.4f} ({tp}/{total_true})")

    print("per_class_precision")
    for i, name in enumerate(class_names):
        tp = cm[i, i].item()
        total_pred = cm[:, i].sum().item()
        precision = tp / total_pred if total_pred > 0 else 0.0
        print(f"{name}: precision={precision:.4f} ({tp}/{total_pred})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP MLP classifier on event-json features.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"), help="CSV file with chart and level columns.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-model", type=Path, help="Optional output path for the trained model state_dict.")
    parser.add_argument(
        "--ablation-preset",
        choices=sorted(ABLATION_PRESETS.keys()),
        help="Optional preset for disabling selected new numeric fields.",
    )
    parser.add_argument(
        "--disable-numeric-fields",
        nargs="*",
        default=[],
        help="Optional numeric field names to zero out during train/eval/test.",
    )
    parser.add_argument(
        "--misclassified-output",
        type=Path,
        help="Optional CSV path to export misclassified test samples with chart/true/pred labels.",
    )
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    config = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        pooling=args.pooling,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    set_seed(config.seed)
    disabled_numeric_fields = parse_disable_numeric_fields(args.ablation_preset, args.disable_numeric_fields)
    zero_feature_indices = None
    if disabled_numeric_fields:
        zero_feature_indices = torch.tensor(
            get_numeric_feature_indices(disabled_numeric_fields),
            dtype=torch.long,
            device=config.device,
        )
        print("disabled_numeric_fields", disabled_numeric_fields)

    events_source = args.encoded_dir if args.encoded_dir else args.events_dir
    file_suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(events_source, args.labels_csv, file_suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {events_source}")

    if args.encoded_dir:
        dataset = PreencodedChartDataset(samples)
        total_size = len(dataset)
        train_size = int(total_size * config.train_ratio)
        eval_size = int(total_size * config.eval_ratio)
        test_size = total_size - train_size - eval_size
        train_set, eval_set, test_set = random_split(
            dataset,
            [train_size, eval_size, test_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
        fake_dataset = EncodedChartDataset(samples, encoder=MVPEventEncoder())
        print_dataset_distributions(fake_dataset, train_set, eval_set, test_set)
        class_weight = compute_class_weight(fake_dataset, train_set)
    else:
        encoder = MVPEventEncoder()
        dataset = EncodedChartDataset(samples, encoder=encoder)
        train_set, eval_set, test_set = split_dataset(dataset, config)
        print_dataset_distributions(dataset, train_set, eval_set, test_set)
        fit_train_normalizer(dataset, train_set)
        class_weight = compute_class_weight(dataset, train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )

    print("class_weight", class_weight.tolist())
    model = train_model(
        train_loader,
        eval_loader,
        config,
        class_weight=class_weight,
        zero_feature_indices=zero_feature_indices,
    )
    test_loss, test_acc = evaluate_model(model, test_loader, config, zero_feature_indices=zero_feature_indices)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    preds, labels, metas = collect_predictions(
        model,
        test_loader,
        config.device,
        zero_feature_indices=zero_feature_indices,
    )
    print_confusion_and_metrics(preds, labels, DEFAULT_LABEL_CLASSES)

    if args.misclassified_output:
        prediction_rows = build_prediction_rows(preds, labels, metas, DEFAULT_LABEL_CLASSES)
        exported = export_prediction_rows(prediction_rows, args.misclassified_output, misclassified_only=True)
        print(f"misclassified_exported={exported}")
        print(f"misclassified_output={args.misclassified_output}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
