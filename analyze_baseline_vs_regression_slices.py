import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_confusion_pairs import CORE_METRICS, chart_metrics, load_event_records, load_level_map, mean


SLICE_ORDER = [
    "baseline_wrong_regression_right",
    "baseline_right_regression_wrong",
    "both_right",
    "both_wrong",
    "fixed_13plus_boundary",
    "fixed_13plus_to_14",
    "fixed_13plus_to_13",
    "broken_14_to_13plus",
]


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_baseline_predictions(path: Path) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for row in load_csv_rows(path):
        chart = row.get("chart", "")
        if not chart:
            continue
        result[chart] = {
            "chart": chart,
            "true_label": row.get("true_label", ""),
            "baseline_pred_label": row.get("pred_label", ""),
            "path": row.get("path", ""),
        }
    return result


def load_regression_predictions(path: Path) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for row in load_csv_rows(path):
        chart = row.get("chart", "")
        if not chart:
            continue
        pred_level_text = row.get("pred_level", "")
        true_level_text = row.get("true_level", "")
        result[chart] = {
            "chart": chart,
            "true_label": row.get("true_label", ""),
            "regression_pred_label": row.get("pred_label", ""),
            "true_level": float(true_level_text) if true_level_text else 0.0,
            "pred_level": float(pred_level_text) if pred_level_text else 0.0,
            "path": row.get("path", ""),
        }
    return result


def enrich_records(
    baseline_map: Dict[str, Dict[str, object]],
    regression_map: Dict[str, Dict[str, object]],
    events_dir: Path,
    labels_map: Dict[str, float],
) -> List[Dict[str, object]]:
    charts = sorted(set(baseline_map) & set(regression_map))
    metrics_cache: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, object]] = []

    for chart in charts:
        base_row = baseline_map[chart]
        reg_row = regression_map[chart]
        metrics = metrics_cache.get(chart)
        if metrics is None:
            metrics = chart_metrics(load_event_records(events_dir, chart))
            metrics_cache[chart] = metrics

        true_label = str(base_row["true_label"])
        baseline_pred = str(base_row["baseline_pred_label"])
        regression_pred = str(reg_row["regression_pred_label"])
        true_level = float(reg_row["true_level"])
        pred_level = float(reg_row["pred_level"])
        row: Dict[str, object] = {
            "chart": chart,
            "true_label": true_label,
            "raw_level": labels_map.get(chart, true_level),
            "true_level": true_level,
            "baseline_pred_label": baseline_pred,
            "regression_pred_label": regression_pred,
            "regression_pred_level": pred_level,
            "baseline_correct": baseline_pred == true_label,
            "regression_correct": regression_pred == true_label,
            "path": reg_row.get("path") or base_row.get("path", ""),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def aggregate_mean(rows: Sequence[Dict[str, object]], keys: Iterable[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in keys:
        result[key] = mean([float(row[key]) for row in rows])
    return result


def level_hist(rows: Sequence[Dict[str, object]]) -> str:
    counts: Dict[float, int] = {}
    for row in rows:
        level = round(float(row["raw_level"]), 1)
        counts[level] = counts.get(level, 0) + 1
    return " ".join(f"{level:.1f}:{counts[level]}" for level in sorted(counts))


def filter_rows(rows: Sequence[Dict[str, object]], predicate) -> List[Dict[str, object]]:
    return [row for row in rows if predicate(row)]


def build_slices(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    slices = {
        "baseline_wrong_regression_right": filter_rows(rows, lambda r: (not r["baseline_correct"]) and r["regression_correct"]),
        "baseline_right_regression_wrong": filter_rows(rows, lambda r: r["baseline_correct"] and (not r["regression_correct"])),
        "both_right": filter_rows(rows, lambda r: r["baseline_correct"] and r["regression_correct"]),
        "both_wrong": filter_rows(rows, lambda r: (not r["baseline_correct"]) and (not r["regression_correct"])),
        "fixed_13plus_boundary": filter_rows(
            rows,
            lambda r: r["true_label"] == "13+" and (not r["baseline_correct"]) and r["regression_correct"],
        ),
        "fixed_13plus_to_14": filter_rows(
            rows,
            lambda r: r["true_label"] == "13+" and r["baseline_pred_label"] == "14" and r["regression_correct"],
        ),
        "fixed_13plus_to_13": filter_rows(
            rows,
            lambda r: r["true_label"] == "13+" and r["baseline_pred_label"] == "13" and r["regression_correct"],
        ),
        "broken_14_to_13plus": filter_rows(
            rows,
            lambda r: r["true_label"] == "14" and r["baseline_correct"] and r["regression_pred_label"] == "13+",
        ),
    }
    return slices


def compute_reference(rows: Sequence[Dict[str, object]], true_label: str) -> List[Dict[str, object]]:
    return [
        row for row in rows
        if row["true_label"] == true_label and row["baseline_correct"] and row["regression_correct"]
    ]


def print_slice_summary(name: str, rows: Sequence[Dict[str, object]], reference: Optional[Sequence[Dict[str, object]]]) -> None:
    print(f"[{name}]")
    print(f"count={len(rows)}")
    if not rows:
        print()
        return

    print(f"raw_level_mean={mean([float(row['raw_level']) for row in rows]):.4f}")
    print(f"raw_level_hist={level_hist(rows)}")
    print(
        "baseline_preds="
        + " ".join(
            f"{label}:{sum(1 for row in rows if row['baseline_pred_label'] == label)}"
            for label in ["13", "13+", "14", "14+"]
        )
    )
    print(
        "regression_preds="
        + " ".join(
            f"{label}:{sum(1 for row in rows if row['regression_pred_label'] == label)}"
            for label in ["13", "13+", "14", "14+"]
        )
    )
    print(f"mean_regression_pred_level={mean([float(row['regression_pred_level']) for row in rows]):.4f}")

    row_means = aggregate_mean(rows, CORE_METRICS)
    ref_means = aggregate_mean(reference, CORE_METRICS) if reference else {}
    if reference is not None:
        print(f"reference_count={len(reference)}")
    else:
        print("reference_count=0")
    for metric in CORE_METRICS:
        value = row_means[metric]
        if reference:
            ref = ref_means[metric]
            print(f"{metric}={value:.4f} ref={ref:.4f} delta={value - ref:+.4f}")
        else:
            print(f"{metric}={value:.4f}")

    representatives = sorted(
        rows,
        key=lambda row: (
            abs(float(row["regression_pred_level"]) - float(row["raw_level"])),
            str(row["chart"]),
        ),
    )
    print(
        "representative_charts="
        + ", ".join(
            f"{row['chart']}({row['raw_level']:.1f},{row['baseline_pred_label']}->{row['regression_pred_label']},{row['regression_pred_level']:.3f})"
            for row in representatives[:8]
        )
    )
    print()


def export_slice_csv(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chart",
        "true_label",
        "raw_level",
        "baseline_pred_label",
        "regression_pred_label",
        "regression_pred_level",
    ] + CORE_METRICS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_overview(output_path: Path, slices: Dict[str, List[Dict[str, object]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["slice", "count", "raw_level_mean", "mean_regression_pred_level"] + CORE_METRICS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name in SLICE_ORDER:
            rows = slices.get(name, [])
            row = {
                "slice": name,
                "count": len(rows),
                "raw_level_mean": mean([float(item["raw_level"]) for item in rows]) if rows else 0.0,
                "mean_regression_pred_level": mean([float(item["regression_pred_level"]) for item in rows]) if rows else 0.0,
            }
            if rows:
                row.update(aggregate_mean(rows, CORE_METRICS))
            else:
                row.update({metric: 0.0 for metric in CORE_METRICS})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze what regression fixed and broke relative to baseline CE.")
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--regression-predictions", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_vs_regression_analysis"))
    args = parser.parse_args()

    baseline_map = load_baseline_predictions(args.baseline_predictions)
    regression_map = load_regression_predictions(args.regression_predictions)
    labels_map = load_level_map(args.labels_csv)
    rows = enrich_records(baseline_map, regression_map, args.events_dir, labels_map)
    slices = build_slices(rows)

    reference_13plus = compute_reference(rows, "13+")
    reference_14 = compute_reference(rows, "14")

    for name in SLICE_ORDER:
        slice_rows = slices.get(name, [])
        if name in {"fixed_13plus_boundary", "fixed_13plus_to_14", "fixed_13plus_to_13"}:
            reference = reference_13plus
        elif name == "broken_14_to_13plus":
            reference = reference_14
        else:
            reference = None
        print_slice_summary(name, slice_rows, reference)
        export_slice_csv(args.output_dir / f"{name}.csv", slice_rows)

    export_overview(args.output_dir / "slice_overview.csv", slices)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
