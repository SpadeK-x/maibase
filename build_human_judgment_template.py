import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_TARGET_PAIRS = [
    ("13+", "14"),
    ("13+", "13"),
    ("14", "13+"),
    ("14", "14+"),
    ("14+", "14"),
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


def priority_group(true_label: str, pred_label: str) -> str:
    if true_label == "13+" and pred_label == "14":
        return "13plus_border_push_up"
    if true_label == "13+" and pred_label == "13":
        return "technical_13plus_drop_down"
    if true_label == "14" and pred_label == "13+":
        return "low_density_14_drop_down"
    if true_label == "14" and pred_label == "14+":
        return "14_push_up"
    if true_label == "14+" and pred_label == "14":
        return "14plus_drop_down"
    return "other"


def collect_rows(
    predictions_rows: Sequence[Dict[str, str]],
    level_map: Dict[str, float],
    target_pairs: Sequence[Tuple[str, str]],
    source_method: str,
) -> List[Dict[str, str]]:
    target_set = set(target_pairs)
    collected: List[Dict[str, str]] = []
    for row in predictions_rows:
        pair = (row.get("true_label", ""), row.get("pred_label", ""))
        if pair not in target_set:
            continue
        chart = row.get("chart", "")
        collected.append(
            {
                "chart": chart,
                "raw_level": f"{level_map.get(chart, 0.0):.1f}" if chart in level_map else "",
                "true_label": pair[0],
                "pred_label": pair[1],
                "priority_group": priority_group(pair[0], pair[1]),
                "source_method": source_method,
                "human_bucket": "",
                "human_level_view": "",
                "accept_model_push": "",
                "notes": "",
            }
        )
    return collected


def dedupe_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    result: List[Dict[str, str]] = []
    for row in rows:
        key = (row["chart"], row["true_label"], row["pred_label"], row["source_method"])
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-judgment template for disputed charts.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--source-method", type=str, default="late_fusion_v1_ce")
    parser.add_argument("--output", type=Path, default=Path("human_judgment_round1.csv"))
    args = parser.parse_args()

    predictions_rows = load_csv_rows(args.predictions_csv)
    level_map = load_level_map(args.labels_csv)
    rows = collect_rows(
        predictions_rows,
        level_map,
        target_pairs=DEFAULT_TARGET_PAIRS,
        source_method=args.source_method,
    )
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: (row["priority_group"], row["true_label"], row["chart"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chart",
        "raw_level",
        "true_label",
        "pred_label",
        "priority_group",
        "source_method",
        "human_bucket",
        "human_level_view",
        "accept_model_push",
        "notes",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"written={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
