import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CLASS_NAMES = ["13", "13+", "14", "14+"]
KEY_PAIRS = [("13+", "13"), ("13+", "14"), ("14", "13+")]


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_rows(rows: Sequence[Dict[str, str]], true_key: str, pred_key: str) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for row in rows:
        true_label = row.get(true_key, "")
        pred_label = row.get(pred_key, "")
        if true_label not in CLASS_NAMES or pred_label not in CLASS_NAMES:
            continue
        result.append(
            {
                "chart": row.get("chart", ""),
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )
    return result


def compute_confusion(rows: Sequence[Dict[str, str]]) -> List[List[int]]:
    index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        matrix[index[row["true_label"]]][index[row["pred_label"]]] += 1
    return matrix


def compute_metrics(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    matrix = compute_confusion(rows)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    accuracy = (correct / total) if total else 0.0

    metrics: Dict[str, object] = {
        "num_samples": total,
        "accuracy": accuracy,
        "confusion_matrix": matrix,
    }

    for i, class_name in enumerate(CLASS_NAMES):
        row_total = sum(matrix[i])
        col_total = sum(matrix[r][i] for r in range(len(CLASS_NAMES)))
        recall = matrix[i][i] / row_total if row_total else 0.0
        precision = matrix[i][i] / col_total if col_total else 0.0
        metrics[f"{class_name}_recall"] = recall
        metrics[f"{class_name}_precision"] = precision

    for true_label, pred_label in KEY_PAIRS:
        count = sum(1 for row in rows if row["true_label"] == true_label and row["pred_label"] == pred_label)
        metrics[f"{true_label}_to_{pred_label}"] = count

    return metrics


def method_specs(output_root: Path) -> List[Tuple[str, Path, str, str]]:
    return [
        ("baseline_ce", output_root / "baseline_ce" / "predictions_test.csv", "true_label", "pred_label"),
        ("regression_official_thresholds", output_root / "regression" / "regression_predictions_test.csv", "true_label", "pred_label"),
        ("regression_searched_thresholds", output_root / "regression" / "regression_thresholded_test.csv", "true_label", "pred_label"),
        ("multitask_cls_head", output_root / "multitask" / "multitask_predictions_test.csv", "true_label_from_level", "pred_label_from_cls_head"),
        ("multitask_regression_bucket", output_root / "multitask" / "multitask_predictions_test.csv", "true_label_from_level", "pred_label_from_level"),
    ]


def write_summary_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "method",
        "num_samples",
        "accuracy",
        "13_recall",
        "13+_recall",
        "14_recall",
        "14+_recall",
        "13_precision",
        "13+_precision",
        "14_precision",
        "14+_precision",
        "13+_to_13",
        "13+_to_14",
        "14_to_13+",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_markdown(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    lines = [
        "| method | acc | 13 r | 13+ r | 14 r | 14+ r | 13+->13 | 13+->14 | 14->13+ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        accuracy = float(row["accuracy"])
        recall_13 = float(row["13_recall"])
        recall_13plus = float(row["13+_recall"])
        recall_14 = float(row["14_recall"])
        recall_14plus = float(row["14+_recall"])
        pair_1 = int(row["13+_to_13"])
        pair_2 = int(row["13+_to_14"])
        pair_3 = int(row["14_to_13+"])
        lines.append(
            f"| {row['method']} | {accuracy:.4f} | {recall_13:.4f} | {recall_13plus:.4f} | "
            f"{recall_14:.4f} | {recall_14plus:.4f} | {pair_1} | {pair_2} | {pair_3} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_text(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    parts: List[str] = []
    for row in rows:
        parts.append(f"[{row['method']}]")
        matrix = row["confusion_matrix"]
        for line in matrix:
            parts.append(" ".join(str(value) for value in line))
        parts.append("")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize target-design experiment outputs into one comparison table.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=Path("experiment_summary"))
    args = parser.parse_args()

    summary_rows: List[Dict[str, object]] = []
    for method, csv_path, true_key, pred_key in method_specs(args.output_root):
        if not csv_path.exists():
            print(f"skip_missing={csv_path}")
            continue
        rows = normalize_rows(load_rows(csv_path), true_key=true_key, pred_key=pred_key)
        metrics = compute_metrics(rows)
        metrics["method"] = method
        summary_rows.append(metrics)

    if not summary_rows:
        raise FileNotFoundError(f"No experiment outputs found under {args.output_root}")

    summary_dir = args.summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary_rows, summary_dir / "target_design_comparison.csv")
    write_summary_markdown(summary_rows, summary_dir / "target_design_comparison.md")
    write_confusion_text(summary_rows, summary_dir / "target_design_confusions.txt")
    print(f"summary_csv={summary_dir / 'target_design_comparison.csv'}")
    print(f"summary_md={summary_dir / 'target_design_comparison.md'}")
    print(f"summary_confusions={summary_dir / 'target_design_confusions.txt'}")


if __name__ == "__main__":
    main()
