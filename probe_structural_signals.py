import argparse
import csv
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


PROBE_METRICS = [
    "outer_move_p90",
    "outer_move_p95",
    "outer_move_ge_0_25_ratio",
    "outer_move_ge_0_375_ratio",
    "outer_move_ge_0_5_ratio",
    "busy_ratio",
    "busy_outer_move_mean",
    "busy_outer_move_p90",
    "busy_compound_ratio",
    "busy_cross_zone_ratio",
    "busy_inner_add_ge2_ratio",
    "busy_density_mean",
    "busy_density_p90",
    "dual_outer_ratio",
    "dual_outer_span_mean",
    "dual_outer_span_p90",
    "dual_outer_wide_ratio",
    "span_jump_mean",
    "span_jump_p90",
    "span_jump_ge_0_5_ratio",
    "active_to_tap_ratio",
    "active_to_compound_ratio",
    "slide_conflict_when_busy_ratio",
    "hold_only_ratio",
    "slide_only_ratio",
    "compound_with_slide_ratio",
]


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


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


def load_records(events_dir: Path, chart: str) -> List[Dict[str, object]]:
    path = events_dir / f"{chart}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def pair_span(outer_idx: Sequence[int]) -> float:
    if len(outer_idx) < 2:
        return 0.0
    a = int(outer_idx[0])
    b = int(outer_idx[1])
    if a == 0 or b == 0:
        return 0.0
    diff = abs(a - b)
    return min(diff, 8 - diff) / 4.0


def compute_probe_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    outer_moves = [float(record.get("outer_move_dist", 0.0)) for record in records]
    busy_records = [
        record for record in records
        if int(record.get("hold_active", 0)) == 1 or int(record.get("slide_active", 0)) == 1
    ]
    busy_outer_moves = [float(record.get("outer_move_dist", 0.0)) for record in busy_records]
    busy_density = [float(math.expm1(float(record.get("local_density_500ms", 0.0)))) for record in busy_records]

    dual_outer_records = []
    dual_outer_spans = []
    prev_dual_span = None
    span_jumps: List[float] = []
    active_to_tap_count = 0
    active_to_compound_count = 0
    transition_count = 0

    prev_busy = False
    for record in records:
        outer_idx = record.get("outer_idx", [0, 0])
        if isinstance(outer_idx, list) and len(outer_idx) == 2 and int(outer_idx[0]) and int(outer_idx[1]):
            dual_outer_records.append(record)
            current_span = pair_span(outer_idx)
            dual_outer_spans.append(current_span)
            if prev_dual_span is not None:
                span_jumps.append(abs(current_span - prev_dual_span))
            prev_dual_span = current_span

        current_busy = int(record.get("hold_active", 0)) == 1 or int(record.get("slide_active", 0)) == 1
        if prev_busy and not current_busy:
            event_type = str(record.get("event_type", ""))
            transition_count += 1
            if event_type == "tap":
                active_to_tap_count += 1
            if event_type == "compound":
                active_to_compound_count += 1
        prev_busy = current_busy

    busy_count = len(busy_records)
    dual_count = len(dual_outer_records)

    metrics = {
        "outer_move_p90": percentile(outer_moves, 0.9),
        "outer_move_p95": percentile(outer_moves, 0.95),
        "outer_move_ge_0_25_ratio": safe_ratio(sum(1 for value in outer_moves if value >= 0.25), len(outer_moves)),
        "outer_move_ge_0_375_ratio": safe_ratio(sum(1 for value in outer_moves if value >= 0.375), len(outer_moves)),
        "outer_move_ge_0_5_ratio": safe_ratio(sum(1 for value in outer_moves if value >= 0.5), len(outer_moves)),
        "busy_ratio": safe_ratio(busy_count, len(records)),
        "busy_outer_move_mean": mean(busy_outer_moves),
        "busy_outer_move_p90": percentile(busy_outer_moves, 0.9),
        "busy_compound_ratio": safe_ratio(sum(1 for r in busy_records if r.get("event_type") == "compound"), busy_count),
        "busy_cross_zone_ratio": safe_ratio(sum(int(r.get("cross_zone_flag", 0)) for r in busy_records), busy_count),
        "busy_inner_add_ge2_ratio": safe_ratio(sum(1 for r in busy_records if float(r.get("inner_add_count", 0.0)) >= 2.0), busy_count),
        "busy_density_mean": mean(busy_density),
        "busy_density_p90": percentile(busy_density, 0.9),
        "dual_outer_ratio": safe_ratio(dual_count, len(records)),
        "dual_outer_span_mean": mean(dual_outer_spans),
        "dual_outer_span_p90": percentile(dual_outer_spans, 0.9),
        "dual_outer_wide_ratio": safe_ratio(sum(1 for value in dual_outer_spans if value >= 0.5), dual_count),
        "span_jump_mean": mean(span_jumps),
        "span_jump_p90": percentile(span_jumps, 0.9),
        "span_jump_ge_0_5_ratio": safe_ratio(sum(1 for value in span_jumps if value >= 0.5), len(span_jumps)),
        "active_to_tap_ratio": safe_ratio(active_to_tap_count, transition_count),
        "active_to_compound_ratio": safe_ratio(active_to_compound_count, transition_count),
        "slide_conflict_when_busy_ratio": safe_ratio(sum(int(r.get("slide_conflict_flag", 0)) for r in busy_records), busy_count),
        "hold_only_ratio": safe_ratio(
            sum(
                1 for r in records
                if int(r.get("hold_active", 0)) == 1 and int(r.get("slide_active", 0)) == 0
            ),
            len(records),
        ),
        "slide_only_ratio": safe_ratio(
            sum(
                1 for r in records
                if int(r.get("slide_active", 0)) == 1 and int(r.get("hold_active", 0)) == 0
            ),
            len(records),
        ),
        "compound_with_slide_ratio": safe_ratio(
            sum(
                1 for r in records
                if r.get("event_type") == "compound" and int(r.get("slide_active", 0)) == 1
            ),
            len(records),
        ),
    }
    return metrics


def aggregate(rows: Sequence[Dict[str, float]], metrics: Iterable[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for metric in metrics:
        result[metric] = mean([float(row[metric]) for row in rows])
    return result


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def effect_size(group_a: Sequence[Dict[str, float]], group_b: Sequence[Dict[str, float]], metric: str) -> float:
    values_a = [float(row[metric]) for row in group_a]
    values_b = [float(row[metric]) for row in group_b]
    mean_a = mean(values_a)
    mean_b = mean(values_b)
    std_a = sample_std(values_a)
    std_b = sample_std(values_b)
    pooled = math.sqrt((std_a * std_a + std_b * std_b) / 2.0)
    if pooled < 1e-8:
        return 0.0
    return (mean_a - mean_b) / pooled


def enrich_slice(rows: Sequence[Dict[str, str]], events_dir: Path) -> List[Dict[str, float]]:
    result: List[Dict[str, float]] = []
    for row in rows:
        chart = row["chart"]
        metrics = compute_probe_metrics(load_records(events_dir, chart))
        enriched: Dict[str, float] = {
            "chart": chart,
            "raw_level": float(row.get("raw_level", 0.0)),
        }
        for key, value in metrics.items():
            enriched[key] = float(value)
        result.append(enriched)
    return result


def load_slice(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return load_csv_rows(path)


def export_group_summary(output_path: Path, group_stats: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "count", "raw_level_mean"] + PROBE_METRICS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in group_stats:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def export_ranked_features(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "effect_size", "fixed_mean", "broken_mean", "stable_13plus_mean", "stable_14_mean"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe candidate structural signals for fixed 13+ vs broken 14 slices.")
    parser.add_argument("--slice-dir", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("feature_probe"))
    args = parser.parse_args()

    fixed_13plus = enrich_slice(load_slice(args.slice_dir / "fixed_13plus_boundary.csv"), args.events_dir)
    broken_14 = enrich_slice(load_slice(args.slice_dir / "broken_14_to_13plus.csv"), args.events_dir)
    stable_13plus = enrich_slice(load_slice(args.slice_dir / "both_right.csv"), args.events_dir)
    stable_13plus = [row for row, raw in zip(stable_13plus, load_slice(args.slice_dir / "both_right.csv")) if raw.get("true_label") == "13+"]
    stable_14 = enrich_slice(load_slice(args.slice_dir / "both_right.csv"), args.events_dir)
    stable_14 = [row for row, raw in zip(stable_14, load_slice(args.slice_dir / "both_right.csv")) if raw.get("true_label") == "14"]

    groups = [
        ("fixed_13plus_boundary", fixed_13plus),
        ("broken_14_to_13plus", broken_14),
        ("stable_13plus", stable_13plus),
        ("stable_14", stable_14),
    ]

    group_stats: List[Dict[str, object]] = []
    for name, rows in groups:
        stats: Dict[str, object] = {
            "group": name,
            "count": len(rows),
            "raw_level_mean": mean([float(row["raw_level"]) for row in rows]) if rows else 0.0,
        }
        if rows:
            stats.update(aggregate(rows, PROBE_METRICS))
        else:
            stats.update({metric: 0.0 for metric in PROBE_METRICS})
        group_stats.append(stats)

    ranked_rows: List[Dict[str, object]] = []
    fixed_mean = aggregate(fixed_13plus, PROBE_METRICS)
    broken_mean = aggregate(broken_14, PROBE_METRICS)
    stable_13plus_mean = aggregate(stable_13plus, PROBE_METRICS)
    stable_14_mean = aggregate(stable_14, PROBE_METRICS)

    for metric in PROBE_METRICS:
        ranked_rows.append(
            {
                "metric": metric,
                "effect_size": effect_size(fixed_13plus, broken_14, metric),
                "fixed_mean": fixed_mean[metric],
                "broken_mean": broken_mean[metric],
                "stable_13plus_mean": stable_13plus_mean[metric],
                "stable_14_mean": stable_14_mean[metric],
            }
        )

    ranked_rows.sort(key=lambda row: abs(float(row["effect_size"])), reverse=True)

    for row in ranked_rows[:15]:
        metric = row["metric"]
        print(
            f"{metric}: effect={float(row['effect_size']):+.4f} "
            f"fixed={float(row['fixed_mean']):.4f} broken={float(row['broken_mean']):.4f} "
            f"stable13+={float(row['stable_13plus_mean']):.4f} stable14={float(row['stable_14_mean']):.4f}"
        )

    export_group_summary(args.output_dir / "group_probe_summary.csv", group_stats)
    export_ranked_features(args.output_dir / "ranked_probe_features.csv", ranked_rows)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
