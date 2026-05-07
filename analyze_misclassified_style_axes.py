import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TARGET_PAIRS = [
    ("13+", "13"),
    ("13+", "14"),
    ("14", "13+"),
]

PREDICTION_RESERVED_COLUMNS = {"chart", "true_label", "pred_label", "path"}
ASSIGNMENT_RESERVED_COLUMNS = {
    "chart",
    "raw_level",
    "bucket_label",
    "cluster_id",
    "distance_to_centroid",
    "pca_x",
    "pca_y",
}


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def detect_axis_names(assignment_rows: Sequence[Dict[str, str]]) -> List[str]:
    if not assignment_rows:
        raise ValueError("No assignment rows found.")
    return [name for name in assignment_rows[0].keys() if name not in ASSIGNMENT_RESERVED_COLUMNS]


def join_rows(
    prediction_rows: Sequence[Dict[str, str]],
    assignment_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    assignment_map = {row["chart"]: row for row in assignment_rows}
    joined: List[Dict[str, object]] = []
    for row in prediction_rows:
        chart = row.get("chart", "")
        if not chart or chart not in assignment_map:
            continue
        merged: Dict[str, object] = dict(row)
        merged.update(assignment_map[chart])
        joined.append(merged)
    return joined


def export_rows(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(
    rows: Sequence[Dict[str, object]],
    true_label: str,
    pred_label: str = "",
    require_correct: bool = False,
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for row in rows:
        if str(row.get("true_label", "")) != true_label:
            continue
        row_pred = str(row.get("pred_label", ""))
        if pred_label and row_pred != pred_label:
            continue
        if require_correct and row_pred != true_label:
            continue
        result.append(row)
    return result


def cluster_distribution(rows: Sequence[Dict[str, object]]) -> str:
    counts = Counter(int(parse_float(row, "cluster_id")) for row in rows)
    return " ".join(f"c{cluster_id}={counts[cluster_id]}" for cluster_id in sorted(counts))


def top_axis_deltas(
    pair_rows: Sequence[Dict[str, object]],
    ref_rows: Sequence[Dict[str, object]],
    axis_names: Sequence[str],
) -> Tuple[str, str, Dict[str, float]]:
    deltas: Dict[str, float] = {}
    for axis_name in axis_names:
        deltas[axis_name] = mean([parse_float(row, axis_name) for row in pair_rows]) - mean(
            [parse_float(row, axis_name) for row in ref_rows]
        )
    positive = sorted(deltas.items(), key=lambda item: item[1], reverse=True)
    negative = sorted(deltas.items(), key=lambda item: item[1])
    pos_text = ", ".join(f"{name}={value:+.6f}" for name, value in positive[:3])
    neg_text = ", ".join(f"{name}={value:+.6f}" for name, value in negative[:3])
    return pos_text, neg_text, deltas


def build_summary_rows(joined_rows: Sequence[Dict[str, object]], axis_names: Sequence[str]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for true_label, pred_label in TARGET_PAIRS:
        pair_rows = filter_rows(joined_rows, true_label=true_label, pred_label=pred_label, require_correct=False)
        ref_rows = filter_rows(joined_rows, true_label=true_label, require_correct=True)
        if not pair_rows or not ref_rows:
            continue

        pos_text, neg_text, deltas = top_axis_deltas(pair_rows, ref_rows, axis_names)
        row: Dict[str, object] = {
            "pair_name": f"{true_label}->{pred_label}",
            "count": len(pair_rows),
            "reference_count": len(ref_rows),
            "pair_raw_level_mean": round(mean([parse_float(item, "raw_level") for item in pair_rows]), 6),
            "reference_raw_level_mean": round(mean([parse_float(item, "raw_level") for item in ref_rows]), 6),
            "pair_cluster_distribution": cluster_distribution(pair_rows),
            "reference_cluster_distribution": cluster_distribution(ref_rows),
            "top_positive_axes": pos_text,
            "top_negative_axes": neg_text,
            "examples": ", ".join(
                f"{item['chart']}(c{int(parse_float(item, 'cluster_id'))})" for item in pair_rows[:8]
            ),
        }
        for axis_name in axis_names:
            row[f"pair_{axis_name}"] = round(mean([parse_float(item, axis_name) for item in pair_rows]), 6)
            row[f"reference_{axis_name}"] = round(mean([parse_float(item, axis_name) for item in ref_rows]), 6)
            row[f"delta_{axis_name}"] = round(deltas[axis_name], 6)
        summary_rows.append(row)
    return summary_rows


def build_pair_exports(
    joined_rows: Sequence[Dict[str, object]],
    axis_names: Sequence[str],
) -> Dict[str, List[Dict[str, object]]]:
    result: Dict[str, List[Dict[str, object]]] = {}
    for true_label, pred_label in TARGET_PAIRS:
        pair_name = f"{true_label.replace('+', 'plus')}_to_{pred_label.replace('+', 'plus')}"
        pair_rows = filter_rows(joined_rows, true_label=true_label, pred_label=pred_label, require_correct=False)
        export_rows_list: List[Dict[str, object]] = []
        for row in pair_rows:
            export_row: Dict[str, object] = {
                "chart": row["chart"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "raw_level": round(parse_float(row, "raw_level"), 4),
                "cluster_id": int(parse_float(row, "cluster_id")),
                "distance_to_centroid": round(parse_float(row, "distance_to_centroid"), 6),
            }
            for axis_name in axis_names:
                export_row[axis_name] = round(parse_float(row, axis_name), 6)
            export_rows_list.append(export_row)
        result[pair_name] = export_rows_list
    return result


def build_ranked_delta_rows(summary_rows: Sequence[Dict[str, object]], axis_names: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for summary in summary_rows:
        pair_name = str(summary["pair_name"])
        for axis_name in axis_names:
            rows.append(
                {
                    "pair_name": pair_name,
                    "axis_name": axis_name,
                    "delta": float(summary[f"delta_{axis_name}"]),
                    "pair_mean": float(summary[f"pair_{axis_name}"]),
                    "reference_mean": float(summary[f"reference_{axis_name}"]),
                }
            )
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze misclassification pairs against interpretable style axes.")
    parser.add_argument("--predictions-csv", type=Path, default=Path("predictions_test.csv"))
    parser.add_argument(
        "--assignments-csv",
        type=Path,
        default=Path("cluster_style_axes_k8/cluster_assignments.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("style_axis_pair_analysis"))
    args = parser.parse_args()

    prediction_rows = load_rows(args.predictions_csv)
    assignment_rows = load_rows(args.assignments_csv)
    axis_names = detect_axis_names(assignment_rows)
    joined_rows = join_rows(prediction_rows, assignment_rows)

    summary_rows = build_summary_rows(joined_rows, axis_names)
    ranked_delta_rows = build_ranked_delta_rows(summary_rows, axis_names)
    pair_exports = build_pair_exports(joined_rows, axis_names)

    export_rows(summary_rows, args.output_dir / "pair_summary.csv")
    export_rows(ranked_delta_rows, args.output_dir / "ranked_axis_deltas.csv")
    for pair_name, rows in pair_exports.items():
        export_rows(rows, args.output_dir / f"{pair_name}.csv")

    print(f"joined_rows={len(joined_rows)}")
    print(f"num_axes={len(axis_names)}")
    print(f"output_dir={args.output_dir}")
    for row in summary_rows:
        print(
            f"{row['pair_name']}: count={row['count']} "
            f"pos=[{row['top_positive_axes']}] "
            f"neg=[{row['top_negative_axes']}]"
        )


if __name__ == "__main__":
    main()
