import argparse
import csv
from pathlib import Path
from typing import Dict, List

from analyze_confusion_pairs import chart_metrics, load_event_records, load_level_map
from probe_structural_signals import PROBE_METRICS, compute_probe_metrics
from probe_technical_signals import TECHNICAL_PROBE_METRICS, compute_technical_probe_metrics
from train_mvp_mlp import DEFAULT_LABEL_CLASSES, discover_samples


STYLE_FEATURE_COLUMNS = [
    "bucket_label",
    "raw_level",
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
] + PROBE_METRICS + TECHNICAL_PROBE_METRICS


def export_rows(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chart"] + STYLE_FEATURE_COLUMNS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export interpretable chart-level style features from event JSON.")
    parser.add_argument("--events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("chart_style_features.csv"))
    args = parser.parse_args()

    samples = discover_samples(args.events_dir, args.labels_csv, [".json"])
    if not samples:
        raise ValueError(f"No labeled event-json samples found in {args.events_dir}")

    level_map = load_level_map(args.labels_csv)
    rows: List[Dict[str, object]] = []
    for sample in samples:
        chart = str(sample["meta"]["chart"])
        label_idx = int(sample["label"])
        records = load_event_records(args.events_dir, chart)
        core = chart_metrics(records)
        structural = compute_probe_metrics(records)
        technical = compute_technical_probe_metrics(records)

        row: Dict[str, object] = {
            "chart": chart,
            "bucket_label": DEFAULT_LABEL_CLASSES[label_idx],
            "raw_level": round(float(level_map.get(chart, 0.0)), 4),
        }
        for name in STYLE_FEATURE_COLUMNS[2:]:
            if name in core:
                row[name] = round(float(core[name]), 6)
            elif name in structural:
                row[name] = round(float(structural[name]), 6)
            elif name in technical:
                row[name] = round(float(technical[name]), 6)
            else:
                raise KeyError(f"Unknown style feature column: {name}")
        rows.append(row)

    export_rows(rows, args.output_path)
    print(f"num_charts={len(rows)}")
    print(f"num_style_features={len(STYLE_FEATURE_COLUMNS) - 2}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
