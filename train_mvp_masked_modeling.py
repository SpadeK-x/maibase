import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from mvp_event_encoder import (
    EncodedChartDataset,
    MVPEventEncoder,
    NumericNormalizerState,
    PreencodedChartDataset,
    collate_encoded_charts,
    project_feature_tensor,
)
from mvp_masked_event_model import (
    build_masked_event_model,
    build_masked_targets,
    compute_masked_modeling_loss,
)
from mvp_mlp_model import make_padding_mask
from train_mvp_mlp import TrainConfig, discover_samples, fit_train_normalizer, print_dataset_distributions, set_seed, split_dataset


def create_masked_positions(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    random_mask = torch.rand(valid_mask.shape, device=valid_mask.device) < mask_ratio
    masked_positions = valid_mask & random_mask
    for row_idx in range(valid_mask.size(0)):
        if valid_mask[row_idx].any() and not masked_positions[row_idx].any():
            valid_indices = torch.nonzero(valid_mask[row_idx], as_tuple=False).squeeze(-1)
            choice = valid_indices[torch.randint(0, valid_indices.numel(), (1,), device=valid_mask.device)]
            masked_positions[row_idx, choice] = True
    return masked_positions


def init_dataset(
    events_dir: Optional[Path],
    encoded_dir: Optional[Path],
    labels_csv: Path,
    seed: int,
) -> Tuple[torch.utils.data.Dataset, object, object, object, Optional[NumericNormalizerState]]:
    data_dir = encoded_dir if encoded_dir else events_dir
    suffixes = [".pt"] if encoded_dir else [".json"]
    samples = discover_samples(data_dir, labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {data_dir}")

    config = TrainConfig(seed=seed)
    if encoded_dir:
        dataset = PreencodedChartDataset(samples)
        total_size = len(dataset)
        train_size = int(total_size * config.train_ratio)
        eval_size = int(total_size * config.eval_ratio)
        test_size = total_size - train_size - eval_size
        train_set, eval_set, test_set = random_split(
            dataset,
            [train_size, eval_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )
        fake_dataset = EncodedChartDataset(samples, encoder=MVPEventEncoder())
        print_dataset_distributions(fake_dataset, train_set, eval_set, test_set)
        return dataset, train_set, eval_set, test_set, None

    encoder = MVPEventEncoder()
    dataset = EncodedChartDataset(samples, encoder=encoder)
    train_set, eval_set, test_set = split_dataset(dataset, config)
    print_dataset_distributions(dataset, train_set, eval_set, test_set)
    fit_train_normalizer(dataset, train_set)
    normalizer_state = dataset.encoder.export_normalizer_state()
    return dataset, train_set, eval_set, test_set, normalizer_state


def make_loader(subset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_encoded_charts,
    )


def run_epoch(
    model,
    loader: DataLoader,
    device: str,
    mask_ratio: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    metric_sums: Dict[str, float] = {}
    steps = 0

    for batch_x, lengths, _labels, _metas in loader:
        batch_x = project_feature_tensor(batch_x, model.config.input_dim)
        batch_x = batch_x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        valid_mask = make_padding_mask(lengths, batch_x.size(1))
        masked_positions = create_masked_positions(valid_mask, mask_ratio)
        targets = build_masked_targets(batch_x)

        outputs = model(batch_x, valid_mask, masked_positions=masked_positions)
        loss, loss_metrics = compute_masked_modeling_loss(outputs, targets, masked_positions)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        for key, value in loss_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        steps += 1

    if steps == 0:
        return 0.0, {}
    avg_metrics = {key: value / steps for key, value in metric_sums.items()}
    return total_loss / steps, avg_metrics


def build_checkpoint(
    model,
    args,
    normalizer_state: Optional[NumericNormalizerState],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.config.input_dim,
            "model_dim": model.config.model_dim,
            "ff_dim": model.config.ff_dim,
            "num_heads": model.config.num_heads,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
            "pooling": model.config.pooling,
        },
        "train_args": vars(args),
    }
    if normalizer_state is not None:
        payload["numeric_mean"] = normalizer_state.mean.clone()
        payload["numeric_std"] = normalizer_state.std.clone()
    return payload


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    ordered_keys = ["total", "event_type", "event_trait", "slide_shape", "outer_slot1", "outer_slot2", "numeric", "inner_mask"]
    parts = [f"{prefix}_{key}={metrics[key]:.4f}" for key in ordered_keys if key in metrics]
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a masked event modeling transformer on MVP event sequences.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--save-model", type=Path, default=Path("mvp_masked_model.pt"))
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")
    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("--mask-ratio must be in (0, 1)")

    set_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = torch.cuda.is_available()

    _dataset, train_set, eval_set, test_set, normalizer_state = init_dataset(
        args.events_dir,
        args.encoded_dir,
        args.labels_csv,
        args.seed,
    )
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, pin_memory)
    eval_loader = make_loader(eval_set, args.eval_batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_set, args.eval_batch_size, False, args.num_workers, pin_memory)

    model = build_masked_event_model(
        input_dim=84,
        model_dim=args.model_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_eval_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        train_loss, train_metrics = run_epoch(model, train_loader, device, args.mask_ratio, optimizer=optimizer)
        with torch.no_grad():
            eval_loss, eval_metrics = run_epoch(model, eval_loader, device, args.mask_ratio, optimizer=None)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} "
            f"{format_metrics('train', train_metrics)} {format_metrics('eval', eval_metrics)}"
        )
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_loss, test_metrics = run_epoch(model, test_loader, device, args.mask_ratio, optimizer=None)
    print(f"test_loss={test_loss:.4f} {format_metrics('test', test_metrics)}")

    checkpoint = build_checkpoint(model, args, normalizer_state)
    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.save_model)
    print(f"saved_model={args.save_model}")


if __name__ == "__main__":
    main()
