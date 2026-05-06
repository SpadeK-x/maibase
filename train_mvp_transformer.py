import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from mvp_event_encoder import EncodedChartDataset, MVPEventEncoder, PreencodedChartDataset, collate_encoded_charts
from mvp_event_encoder import get_numeric_feature_indices
from mvp_mlp_model import make_padding_mask
from mvp_transformer_model import build_transformer_model
from train_mvp_mlp import (
    ABLATION_PRESETS,
    TrainConfig,
    compute_class_weight,
    discover_samples,
    evaluate_model as _unused_evaluate_model,
    fit_train_normalizer,
    parse_disable_numeric_fields,
    print_dataset_distributions,
    set_seed,
    split_dataset,
)


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

        batch_x = batch_x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if zero_feature_indices is not None and zero_feature_indices.numel() > 0:
            batch_x[:, :, zero_feature_indices] = 0.0
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    zero_feature_indices: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    with torch.no_grad():
        return run_epoch(model, loader, criterion, None, device, zero_feature_indices=zero_feature_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MVP Transformer classifier on event features.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "cls_mean"])
    parser.add_argument("--save-model", type=Path)
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
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    config = TrainConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {data_dir}")

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

    model = build_transformer_model(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        model_dim=args.model_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(config.device))

    print("class_weight", class_weight.tolist())

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
        eval_loss, eval_acc = evaluate_model(
            model,
            eval_loader,
            criterion,
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

    test_loss, test_acc = evaluate_model(
        model,
        test_loader,
        criterion,
        config.device,
        zero_feature_indices=zero_feature_indices,
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
