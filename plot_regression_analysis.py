import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(rows: List[Dict[str, str]], key: str) -> List[float]:
    return [float(row[key]) for row in rows]


def level_bucket(level: float) -> str:
    if level < 13.6:
        return "13.0-13.5"
    if level < 14.0:
        return "13.6-13.9"
    if level < 14.6:
        return "14.0-14.5"
    return "14.6-14.9"


def make_summary(rows: List[Dict[str, str]]) -> None:
    true_levels = to_float(rows, "true_level")
    pred_levels = to_float(rows, "pred_level")
    abs_err = [abs(p - t) for p, t in zip(pred_levels, true_levels)]
    print(f"num_samples={len(rows)}")
    print(f"mean_true_level={sum(true_levels) / len(true_levels):.4f}")
    print(f"mean_pred_level={sum(pred_levels) / len(pred_levels):.4f}")
    print(f"mean_abs_error={sum(abs_err) / len(abs_err):.4f}")

    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_bucket[level_bucket(float(row['true_level']))].append(abs(float(row["pred_level"]) - float(row["true_level"])))
    print("bucket_mae")
    for bucket in ["13.0-13.5", "13.6-13.9", "14.0-14.5", "14.6-14.9"]:
        values = by_bucket.get(bucket, [])
        if not values:
            continue
        print(f"{bucket}: {sum(values) / len(values):.4f} n={len(values)}")

    label_pairs = Counter((row["true_label"], row["pred_label"]) for row in rows)
    print("top_label_pairs")
    for pair, count in label_pairs.most_common(10):
        print(pair, count)


def plot(rows: List[Dict[str, str]], output_path: Path) -> None:
    true_levels = to_float(rows, "true_level")
    pred_levels = to_float(rows, "pred_level")
    abs_err = [abs(p - t) for p, t in zip(pred_levels, true_levels)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(true_levels, pred_levels, alpha=0.7, s=20)
    min_v = min(min(true_levels), min(pred_levels))
    max_v = max(max(true_levels), max(pred_levels))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_title("Pred vs True Level")
    ax.set_xlabel("True Level")
    ax.set_ylabel("Pred Level")

    ax = axes[0, 1]
    residuals = [p - t for p, t in zip(pred_levels, true_levels)]
    ax.hist(residuals, bins=20)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Pred - True")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.hist(pred_levels, bins=20, alpha=0.7, label="pred")
    ax.hist(true_levels, bins=20, alpha=0.5, label="true")
    ax.set_title("Level Distribution")
    ax.set_xlabel("Level")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 1]
    bucket_names = ["13.0-13.5", "13.6-13.9", "14.0-14.5", "14.6-14.9"]
    bucket_mae = []
    for bucket in bucket_names:
        values = [
            err for err, row in zip(abs_err, rows)
            if level_bucket(float(row["true_level"])) == bucket
        ]
        bucket_mae.append(sum(values) / len(values) if values else 0.0)
    ax.bar(bucket_names, bucket_mae)
    ax.set_title("MAE by True-Level Bucket")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple diagnostics for regression_predictions_test.csv.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("regression_predictions_test.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regression_analysis.png"),
    )
    args = parser.parse_args()

    rows = load_rows(args.predictions_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.predictions_csv}")
    make_summary(rows)
    plot(rows, args.output)
    print(f"saved_plot={args.output}")


if __name__ == "__main__":
    main()
