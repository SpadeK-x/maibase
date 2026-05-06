import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
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
from train_mvp_mlp import (
    ABLATION_PRESETS,
    DEFAULT_LABEL_CLASSES,
    TrainConfig,
    build_prediction_rows,
    collect_predictions,
    discover_samples,
    export_prediction_rows,
    parse_disable_numeric_fields,
    print_confusion_and_metrics,
    print_dataset_distributions,
    set_seed,
)


TARGET_CONFUSION_PAIRS = [
    ("13+", "13"),
    ("13+", "14"),
    ("14", "13+"),
]


def infer_input_dim_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("event_encoder.net.0.weight")
    if weight is None or weight.ndim != 2:
        raise KeyError("Could not infer input dim from state_dict key `event_encoder.net.0.weight`.")
    return int(weight.size(1))


def project_features_for_model(batch_x: torch.Tensor, target_input_dim: int) -> torch.Tensor:
    return project_feature_tensor(batch_x, target_input_dim)


def export_pair_subsets(rows: Sequence[Dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for true_label, pred_label in TARGET_CONFUSION_PAIRS:
        subset = [row for row in rows if row["true_label"] == true_label and row["pred_label"] == pred_label]
        output_path = output_dir / f"misclassified_{true_label.replace('+', 'plus')}_to_{pred_label.replace('+', 'plus')}.csv"
        export_prediction_rows(subset, output_path, misclassified_only=False)
        print(f"{true_label}->{pred_label} count={len(subset)} output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved MVP MLP checkpoint and export misclassified test samples.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("mvp_mlp.pth"))
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ablation-preset",
        choices=sorted(ABLATION_PRESETS.keys()),
        help="Optional preset for disabling selected numeric fields during evaluation.",
    )
    parser.add_argument(
        "--disable-numeric-fields",
        nargs="*",
        default=[],
        help="Optional numeric field names to zero out during evaluation.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("predictions_test.csv"),
        help="CSV path for all test predictions.",
    )
    parser.add_argument(
        "--misclassified-output",
        type=Path,
        default=Path("misclassified_test.csv"),
        help="CSV path for misclassified test samples.",
    )
    parser.add_argument(
        "--pair-output-dir",
        type=Path,
        default=Path("misclassified_pairs"),
        help="Directory for focused confusion-pair CSV exports.",
    )
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TrainConfig(
        eval_batch_size=args.eval_batch_size,
        pooling=args.pooling,
        seed=args.seed,
        num_workers=args.num_workers,
        device=device,
        pin_memory=torch.cuda.is_available(),
    )
    set_seed(config.seed)

    disabled_numeric_fields = parse_disable_numeric_fields(args.ablation_preset, args.disable_numeric_fields)
    zero_feature_indices: Optional[torch.Tensor] = None
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
    else:
        dataset = EncodedChartDataset(samples, encoder=MVPEventEncoder())

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

    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_encoded_charts,
    )

    state_dict = torch.load(args.model_path, map_location=config.device)
    input_dim = infer_input_dim_from_state_dict(state_dict)
    print("model_input_dim", input_dim)
    model = build_model(input_dim=input_dim, num_classes=config.num_classes, pooling=config.pooling).to(config.device)
    model.load_state_dict(state_dict)

    model.eval()
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_metas: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_x, lengths, labels, metas in test_loader:
            batch_x = project_features_for_model(batch_x, input_dim)
            batch_x = batch_x.to(config.device, non_blocking=True)
            lengths = lengths.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            batch_x = apply_zero_feature_mask(batch_x, zero_feature_indices)
            logits, _, _ = model(batch_x, make_padding_mask(lengths, batch_x.size(1)))
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metas = all_metas
    print_confusion_and_metrics(preds, labels, DEFAULT_LABEL_CLASSES)

    rows = build_prediction_rows(preds, labels, metas, DEFAULT_LABEL_CLASSES)
    total_exported = export_prediction_rows(rows, args.predictions_output, misclassified_only=False)
    mis_exported = export_prediction_rows(rows, args.misclassified_output, misclassified_only=True)
    print(f"predictions_exported={total_exported} output={args.predictions_output}")
    print(f"misclassified_exported={mis_exported} output={args.misclassified_output}")
    export_pair_subsets(rows, args.pair_output_dir)


if __name__ == "__main__":
    main()
