import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CORE_METRICS = [
    "num_events",
    "chart_span_seconds",
    "events_per_second",
    "mean_density_500ms",
    "p90_density_500ms",
    "touch_ratio",
    "compound_ratio",
    "cross_zone_ratio",
    "inner_add_ge2_ratio",
    "inner_count_ge2_ratio",
    "slide_active_ratio",
    "hold_active_ratio",
    "slide_conflict_ratio",
    "mean_outer_move_dist",
    "mean_slide_span",
    "max_slide_span",
]


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_level_map(labels_csv: Path) -> Dict[str, float]:
    level_map: Dict[str, float] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chart = row.get("chart")
            level = row.get("level")
            if not chart or level is None:
                continue
            try:
                level_map[chart] = float(level)
            except ValueError:
                continue
    return level_map


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def load_event_records(events_dir: Path, chart: str) -> List[Dict[str, object]]:
    path = events_dir / f"{chart}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def density_from_log(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(math.expm1(float(value)))
    return 0.0


def chart_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    num_events = len(records)
    if num_events == 0:
        return {name: 0.0 for name in CORE_METRICS}

    event_times = [float(record.get("event_time", 0.0)) for record in records]
    densities = [density_from_log(record.get("local_density_500ms", 0.0)) for record in records]
    outer_moves = [float(record.get("outer_move_dist", 0.0)) for record in records]
    slide_spans = [float(record.get("slide_span", 0.0)) for record in records]
    inner_adds = [float(record.get("inner_add_count", 0.0)) for record in records]
    inner_counts = [float(record.get("inner_count", 0.0)) for record in records]
    event_types = [str(record.get("event_type", "")) for record in records]

    chart_span = max(event_times) - min(event_times) if num_events >= 2 else 0.0
    duration = max(chart_span, 1e-6)

    metrics = {
        "num_events": float(num_events),
        "chart_span_seconds": chart_span,
        "events_per_second": num_events / duration,
        "mean_density_500ms": mean(densities),
        "p90_density_500ms": percentile(densities, 0.9),
        "touch_ratio": ratio(sum(1 for value in event_types if value == "touch"), num_events),
        "compound_ratio": ratio(sum(1 for value in event_types if value == "compound"), num_events),
        "cross_zone_ratio": ratio(sum(int(record.get("cross_zone_flag", 0)) for record in records), num_events),
        "inner_add_ge2_ratio": ratio(sum(1 for value in inner_adds if value >= 2.0), num_events),
        "inner_count_ge2_ratio": ratio(sum(1 for value in inner_counts if value >= 2.0), num_events),
        "slide_active_ratio": ratio(sum(int(record.get("slide_active", 0)) for record in records), num_events),
        "hold_active_ratio": ratio(sum(int(record.get("hold_active", 0)) for record in records), num_events),
        "slide_conflict_ratio": ratio(sum(int(record.get("slide_conflict_flag", 0)) for record in records), num_events),
        "mean_outer_move_dist": mean(outer_moves),
        "mean_slide_span": mean(slide_spans),
        "max_slide_span": max(slide_spans) if slide_spans else 0.0,
    }
    return metrics


def format_level_hist(levels: Sequence[float]) -> str:
    counter = Counter(round(level, 1) for level in levels)
    parts = [f"{level:.1f}:{counter[level]}" for level in sorted(counter)]
    return " ".join(parts)


def aggregate_mean(rows: Sequence[Dict[str, object]], keys: Iterable[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in keys:
        result[key] = mean([float(row[key]) for row in rows])
    return result


def enrich_rows(
    rows: Sequence[Dict[str, str]],
    events_dir: Path,
    level_map: Dict[str, float],
    metrics_cache: Dict[str, Dict[str, float]],
) -> List[Dict[str, float]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        chart = row["chart"]
        metrics = metrics_cache.get(chart)
        if metrics is None:
            metrics = chart_metrics(load_event_records(events_dir, chart))
            metrics_cache[chart] = metrics
        enriched_row: Dict[str, object] = {
            "chart": chart,
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "raw_level": level_map.get(chart, 0.0),
        }
        enriched_row.update(metrics)
        enriched.append(enriched_row)
    return enriched


def export_enriched_csv(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chart", "true_label", "pred_label", "raw_level"] + CORE_METRICS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_pair_summary(
    pair_name: str,
    pair_rows: Sequence[Dict[str, object]],
    correct_rows: Sequence[Dict[str, object]],
) -> None:
    levels = [float(row["raw_level"]) for row in pair_rows]
    pair_mean = aggregate_mean(pair_rows, CORE_METRICS)
    correct_mean = aggregate_mean(correct_rows, CORE_METRICS) if correct_rows else {}

    print(f"[{pair_name}]")
    print(f"count={len(pair_rows)}")
    print(f"raw_level_mean={mean(levels):.4f}")
    print(f"raw_level_hist={format_level_hist(levels)}")
    if correct_rows:
        print(f"reference_correct_count={len(correct_rows)}")
    else:
        print("reference_correct_count=0")

    for key in CORE_METRICS:
        value = pair_mean[key]
        if correct_rows:
            ref = correct_mean[key]
            delta = value - ref
            print(f"{key}={value:.4f} ref={ref:.4f} delta={delta:+.4f}")
        else:
            print(f"{key}={value:.4f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize key confusion pairs using event-json statistics.")
    parser.add_argument("--pairs-dir", type=Path, default=Path("misclassified_pairs"))
    parser.add_argument("--predictions-csv", type=Path, default=Path("predictions_test.csv"))
    parser.add_argument("--events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("confusion_pair_analysis"))
    args = parser.parse_args()

    predictions_rows = load_csv_rows(args.predictions_csv)
    level_map = load_level_map(args.labels_csv)
    metrics_cache: Dict[str, Dict[str, float]] = {}

    pair_paths = sorted(args.pairs_dir.glob("misclassified_*.csv"))
    if not pair_paths:
        raise FileNotFoundError(f"No pair csv files found in {args.pairs_dir}")

    for pair_path in pair_paths:
        raw_rows = load_csv_rows(pair_path)
        if not raw_rows:
            continue

        pair_rows = enrich_rows(raw_rows, args.events_dir, level_map, metrics_cache)
        true_label = str(raw_rows[0]["true_label"])
        correct_reference = [
            row for row in predictions_rows
            if row["true_label"] == true_label and row["pred_label"] == true_label
        ]
        correct_rows = enrich_rows(correct_reference, args.events_dir, level_map, metrics_cache)

        pair_name = pair_path.stem.replace("misclassified_", "")
        print_pair_summary(pair_name, pair_rows, correct_rows)
        export_enriched_csv(pair_rows, args.output_dir / f"{pair_name}.csv")


if __name__ == "__main__":
    main()
