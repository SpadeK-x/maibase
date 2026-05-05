import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mvp_event_encoder import EncodedChartDataset, collate_encoded_charts
from mvp_mlp_model import build_model, make_padding_mask


DEFAULT_LABEL_CLASSES = ["13", "13+", "14", "14+"]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(DEFAULT_LABEL_CLASSES)}


@dataclass
class TrainConfig:
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    seed: int = 42
    pooling: str = "mean"
    input_dim: int = 84
    num_classes: int = 4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


def discover_samples(events_dir: Path, labels_csv: Path) -> List[Dict[str, object]]:
    label_map = load_label_map(labels_csv)
    samples: List[Dict[str, object]] = []

    for json_path in sorted(events_dir.glob("*.json")):
        chart_name = json_path.stem
        if chart_name not in label_map:
            continue
        samples.append(
            {
                "path": str(json_path),
                "label": label_map[chart_name],
                "meta": {"chart": chart_name},
            }
        )

    return samples


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
        collate_fn=collate_encoded_charts,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_encoded_charts,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_encoded_charts,
    )
    return train_loader, eval_loader, test_loader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


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

    for batch_x, lengths, labels, _ in loader:
        if labels is None:
            raise ValueError("Labels are required for training/evaluation.")

        batch_x = batch_x.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
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


def train_model(train_loader: DataLoader, eval_loader: DataLoader, config: TrainConfig):
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
    criterion = nn.CrossEntropyLoss()

    best_eval_loss = float("inf")
    best_state = None

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
    return model


def evaluate_model(model: nn.Module, loader: DataLoader, config: TrainConfig) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return run_epoch(model, loader, criterion, None, config.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP MLP classifier on event-json features.")
    parser.add_argument("--events-dir", type=Path, required=True, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"), help="CSV file with chart and level columns.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "mean_max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=Path, help="Optional output path for the trained model state_dict.")
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        pooling=args.pooling,
        seed=args.seed,
    )
    set_seed(config.seed)

    samples = discover_samples(args.events_dir, args.labels_csv)
    if not samples:
        raise ValueError(f"No labeled event-json samples found in {args.events_dir}")

    dataset = EncodedChartDataset(samples)
    train_loader, eval_loader, test_loader = build_loaders(dataset, config)

    model = train_model(train_loader, eval_loader, config)
    test_loss, test_acc = evaluate_model(model, test_loader, config)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
