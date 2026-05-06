import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from evaluate_saved_mlp import infer_input_dim_from_state_dict, project_features_for_model
from mvp_event_encoder import EncodedChartDataset, MVPEventEncoder, PreencodedChartDataset, collate_encoded_charts
from mvp_mlp_model import build_model, make_padding_mask
from train_mvp_mlp import DEFAULT_LABEL_CLASSES, discover_samples, print_confusion_and_metrics, print_dataset_distributions, set_seed, TrainConfig


@dataclass
class ThresholdSearchResult:
    thresholds: Tuple[float, float, float]
    score: float
    preds: torch.Tensor
    labels: torch.Tensor


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


def bucket_level_with_thresholds(level: float, thresholds: Tuple[float, float, float]) -> int:
    t1, t2, t3 = thresholds
    if level < t1:
        return 0
    if level < t2:
        return 1
    if level < t3:
        return 2
    return 3


def fit_linear_calibration(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    x = preds.to(torch.float64)
    y = targets.to(torch.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum().item()
    if denom < 1e-12:
        return 1.0, 0.0
    a = (((x - x_mean) * (y - y_mean)).sum().item()) / denom
    b = (y_mean.item() - a * x_mean.item())
    return float(a), float(b)


def fit_target_normalizer(level_map: Dict[str, float], samples: Sequence[Dict[str, object]], train_indices: Sequence[int]) -> Tuple[float, float]:
    values = []
    for index in train_indices:
        chart = str((samples[index].get("meta") or {}).get("chart", ""))
        values.append(float(level_map[chart]))
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    if std < 1e-6:
        std = 1.0
    return mean, std


def apply_linear_calibration(preds: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return preds * a + b


def accuracy_score(pred_buckets: torch.Tensor, true_buckets: torch.Tensor) -> float:
    return (pred_buckets == true_buckets).float().mean().item()


def collect_regression_outputs(
    model: torch.nn.Module,
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
            batch_x = batch_x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            mask = make_padding_mask(lengths, batch_x.size(1))
            preds, _, _ = model(batch_x, mask)
            denorm_preds = preds.squeeze(-1).cpu() * target_std + target_mean
            all_preds.append(denorm_preds)
            all_targets.append(targets.cpu())
            all_metas.extend(dict(meta or {}) for meta in metas)

    return torch.cat(all_preds), torch.cat(all_targets), all_metas


def build_regression_eval_dataset(samples: Sequence[Dict[str, object]], level_map: Dict[str, float], use_encoded: bool):
    stripped = []
    for sample in samples:
        chart = str((sample.get("meta") or {}).get("chart", ""))
        stripped.append(
            {
                "path": sample["path"],
                "label": 0,
                "meta": {"chart": chart, "target_level": level_map[chart]},
            }
        )
    if use_encoded:
        return PreencodedChartDataset(stripped)
    return EncodedChartDataset(stripped, encoder=MVPEventEncoder())


def collate_level_batch(batch):
    batch_x, lengths, _, metas = collate_encoded_charts(batch)
    levels = torch.tensor([float(meta["target_level"]) for meta in metas], dtype=torch.float32)
    return batch_x, lengths, levels, metas


def search_thresholds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    step: float = 0.02,
) -> ThresholdSearchResult:
    target_buckets = torch.tensor(
        [bucket_level_with_thresholds(float(x), (13.6, 14.0, 14.6)) for x in targets.tolist()],
        dtype=torch.long,
    )
    best: Optional[ThresholdSearchResult] = None

    t1_values = torch.arange(13.3, 13.9 + 1e-9, step).tolist()
    t2_values = torch.arange(13.8, 14.3 + 1e-9, step).tolist()
    t3_values = torch.arange(14.3, 14.9 + 1e-9, step).tolist()

    for t1 in t1_values:
        for t2 in t2_values:
            if t2 <= t1:
                continue
            for t3 in t3_values:
                if t3 <= t2:
                    continue
                pred_buckets = torch.tensor(
                    [bucket_level_with_thresholds(float(x), (t1, t2, t3)) for x in preds.tolist()],
                    dtype=torch.long,
                )
                score = accuracy_score(pred_buckets, target_buckets)
                if best is None or score > best.score:
                    best = ThresholdSearchResult(
                        thresholds=(float(t1), float(t2), float(t3)),
                        score=score,
                        preds=pred_buckets,
                        labels=target_buckets,
                    )
    if best is None:
        raise RuntimeError("Threshold search failed to produce a result.")
    return best


def export_predictions(
    output_path: Path,
    metas: Sequence[Dict[str, object]],
    true_levels: torch.Tensor,
    pred_levels: torch.Tensor,
    pred_buckets: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chart", "true_level", "pred_level", "true_label", "pred_label", "path"],
        )
        writer.writeheader()
        for meta, true_level, pred_level, pred_bucket in zip(
            metas,
            true_levels.tolist(),
            pred_levels.tolist(),
            pred_buckets.tolist(),
        ):
            true_bucket = bucket_level_with_thresholds(float(true_level), (13.6, 14.0, 14.6))
            writer.writerow(
                {
                    "chart": str(meta.get("chart", "")),
                    "true_level": f"{true_level:.4f}",
                    "pred_level": f"{pred_level:.4f}",
                    "true_label": DEFAULT_LABEL_CLASSES[true_bucket],
                    "pred_label": DEFAULT_LABEL_CLASSES[int(pred_bucket)],
                    "path": str(meta.get("path", "")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search best bucket thresholds for regression predictions on eval split.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--model-path", type=Path, required=True, help="Saved regression checkpoint path.")
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--use-linear-calibration", action="store_true")
    parser.add_argument("--export-test-csv", type=Path, default=Path("regression_thresholded_test.csv"))
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    config = TrainConfig(
        eval_batch_size=args.eval_batch_size,
        pooling=args.pooling,
        seed=args.seed,
        num_workers=args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pin_memory=torch.cuda.is_available(),
    )
    set_seed(config.seed)

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    base_samples = discover_samples(data_dir, args.labels_csv, suffixes)
    level_map = load_raw_level_map(args.labels_csv)
    dataset = build_regression_eval_dataset(base_samples, level_map, use_encoded=args.encoded_dir is not None)

    total_size = len(dataset)
    train_size = int(total_size * config.train_ratio)
    eval_size = int(total_size * config.eval_ratio)
    test_size = total_size - train_size - eval_size
    train_set, eval_set, test_set = random_split(
        dataset,
        [train_size, eval_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    fake_dataset = EncodedChartDataset(base_samples, encoder=MVPEventEncoder())
    print_dataset_distributions(fake_dataset, train_set, eval_set, test_set)

    eval_loader = DataLoader(
        eval_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_level_batch,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_level_batch,
    )

    target_mean, target_std = fit_target_normalizer(level_map, base_samples, train_set.indices)
    print(f"target_mean={target_mean:.4f} target_std={target_std:.4f}")

    state_dict = torch.load(args.model_path, map_location=config.device)
    input_dim = infer_input_dim_from_state_dict(state_dict)
    print("model_input_dim", input_dim)
    model = build_model(input_dim=input_dim, num_classes=1, pooling=config.pooling).to(config.device)
    model.load_state_dict(state_dict)

    eval_preds, eval_targets, _ = collect_regression_outputs(
        model,
        eval_loader,
        config.device,
        target_mean=target_mean,
        target_std=target_std,
    )
    test_preds, test_targets, test_metas = collect_regression_outputs(
        model,
        test_loader,
        config.device,
        target_mean=target_mean,
        target_std=target_std,
    )

    if args.use_linear_calibration:
        a, b = fit_linear_calibration(eval_preds, eval_targets)
        print(f"linear_calibration a={a:.6f} b={b:.6f}")
        eval_preds = apply_linear_calibration(eval_preds, a, b)
        test_preds = apply_linear_calibration(test_preds, a, b)

    eval_result = search_thresholds(eval_preds, eval_targets, step=args.threshold_step)
    print(
        "best_eval_thresholds",
        {
            "t1": round(eval_result.thresholds[0], 4),
            "t2": round(eval_result.thresholds[1], 4),
            "t3": round(eval_result.thresholds[2], 4),
            "eval_accuracy": round(eval_result.score, 4),
        },
    )

    print("eval_confusion")
    print_confusion_and_metrics(eval_result.preds, eval_result.labels, DEFAULT_LABEL_CLASSES)

    test_true_buckets = torch.tensor(
        [bucket_level_with_thresholds(float(x), (13.6, 14.0, 14.6)) for x in test_targets.tolist()],
        dtype=torch.long,
    )
    test_pred_buckets = torch.tensor(
        [bucket_level_with_thresholds(float(x), eval_result.thresholds) for x in test_preds.tolist()],
        dtype=torch.long,
    )
    print("test_confusion")
    print_confusion_and_metrics(test_pred_buckets, test_true_buckets, DEFAULT_LABEL_CLASSES)

    export_predictions(args.export_test_csv, test_metas, test_targets, test_preds, test_pred_buckets)
    print(f"export_test_csv={args.export_test_csv}")


if __name__ == "__main__":
    main()
