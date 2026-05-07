import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence


RESERVED_COLUMNS = {
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


def detect_axis_names(rows: Sequence[Dict[str, str]]) -> List[str]:
    if not rows:
        raise ValueError("No rows found in cluster assignment CSV.")
    return [name for name in rows[0].keys() if name not in RESERVED_COLUMNS]


def parse_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def export_rows(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rank_rows(rows: Sequence[Dict[str, str]], axis_name: str, top_n: int, reverse: bool) -> List[Dict[str, object]]:
    ordered = sorted(rows, key=lambda row: parse_float(row, axis_name), reverse=reverse)
    result: List[Dict[str, object]] = []
    for rank, row in enumerate(ordered[:top_n], start=1):
        result.append(
            {
                "rank": rank,
                "chart": row["chart"],
                "raw_level": round(parse_float(row, "raw_level"), 4),
                "bucket_label": row["bucket_label"],
                "cluster_id": int(parse_float(row, "cluster_id")),
                "axis_name": axis_name,
                "axis_score": round(parse_float(row, axis_name), 6),
                "distance_to_centroid": round(parse_float(row, "distance_to_centroid"), 6),
                "pca_x": round(parse_float(row, "pca_x"), 6),
                "pca_y": round(parse_float(row, "pca_y"), 6),
            }
        )
    return result


def summarize_axis(rows: Sequence[Dict[str, str]], axis_name: str, top_n: int) -> Dict[str, object]:
    top_rows = sorted(rows, key=lambda row: parse_float(row, axis_name), reverse=True)[:top_n]
    bottom_rows = sorted(rows, key=lambda row: parse_float(row, axis_name))[:top_n]
    top_labels: Dict[str, int] = {}
    bottom_labels: Dict[str, int] = {}
    for row in top_rows:
        label = row["bucket_label"]
        top_labels[label] = top_labels.get(label, 0) + 1
    for row in bottom_rows:
        label = row["bucket_label"]
        bottom_labels[label] = bottom_labels.get(label, 0) + 1

    def format_counts(counts: Dict[str, int]) -> str:
        return " ".join(f"{label}={counts.get(label, 0)}" for label in ["13", "13+", "14", "14+"])

    return {
        "axis_name": axis_name,
        "top_mean": round(sum(parse_float(row, axis_name) for row in top_rows) / max(len(top_rows), 1), 6),
        "bottom_mean": round(sum(parse_float(row, axis_name) for row in bottom_rows) / max(len(bottom_rows), 1), 6),
        "top_label_distribution": format_counts(top_labels),
        "bottom_label_distribution": format_counts(bottom_labels),
        "top_examples": ", ".join(f"{row['chart']}({parse_float(row, axis_name):.3f})" for row in top_rows[:8]),
        "bottom_examples": ", ".join(f"{row['chart']}({parse_float(row, axis_name):.3f})" for row in bottom_rows[:8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank charts by interpretable style-axis scores.")
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("style_axis_rankings"))
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    rows = load_rows(args.assignments_csv)
    axis_names = detect_axis_names(rows)
    summary_rows: List[Dict[str, object]] = []

    for axis_name in axis_names:
        top_rows = build_rank_rows(rows, axis_name, args.top_n, reverse=True)
        bottom_rows = build_rank_rows(rows, axis_name, args.top_n, reverse=False)
        export_rows(top_rows, args.output_dir / f"{axis_name}_top.csv")
        export_rows(bottom_rows, args.output_dir / f"{axis_name}_bottom.csv")
        summary_rows.append(summarize_axis(rows, axis_name, args.top_n))

    export_rows(summary_rows, args.output_dir / "axis_summary.csv")
    print(f"num_axes={len(axis_names)}")
    print(f"top_n={args.top_n}")
    print(f"output_dir={args.output_dir}")
    for row in summary_rows:
        print(
            f"{row['axis_name']}: top_mean={row['top_mean']} bottom_mean={row['bottom_mean']} "
            f"top_labels=[{row['top_label_distribution']}]"
        )


if __name__ == "__main__":
    main()
