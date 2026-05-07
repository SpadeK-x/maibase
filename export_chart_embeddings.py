import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mvp_event_encoder import (
    EncodedChartDataset,
    EVENT_TRAIT_VOCAB,
    EVENT_TYPE_VOCAB,
    LEGACY_EXTRA_NUMERIC_FIELDS,
    MVPEventEncoder,
    NUMERIC_FIELD_ORDER,
    OUTER_SLOT_VOCAB_SIZE,
    PreencodedChartDataset,
    SLIDE_SHAPE_GROUP_VOCAB,
    project_feature_tensor,
)
from parse_events_mvp import INNER_REGIONS
from train_mvp_mlp import DEFAULT_LABEL_CLASSES, discover_samples


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
                level_map[chart] = float(level)
            except ValueError:
                continue
    return level_map


def resolve_chart_name(meta: Dict[str, object]) -> str:
    chart = meta.get("chart")
    if chart:
        return str(chart)
    path_value = meta.get("path")
    if path_value:
        return Path(str(path_value)).stem
    return ""


def build_event_feature_names(target_input_dim: int) -> List[str]:
    names: List[str] = []
    names.extend(f"event_type__{name}" for name in EVENT_TYPE_VOCAB)
    names.extend(f"event_trait__{name}" for name in EVENT_TRAIT_VOCAB)
    names.extend(f"slide_shape_group__{name}" for name in SLIDE_SHAPE_GROUP_VOCAB)
    names.extend(f"outer_idx_slot1__{idx}" for idx in range(OUTER_SLOT_VOCAB_SIZE))
    names.extend(f"outer_idx_slot2__{idx}" for idx in range(OUTER_SLOT_VOCAB_SIZE))
    names.extend(NUMERIC_FIELD_ORDER)
    if target_input_dim == 86:
        names.extend(LEGACY_EXTRA_NUMERIC_FIELDS)
    names.extend(f"inner_mask__{name}" for name in INNER_REGIONS)
    if len(names) != target_input_dim:
        raise ValueError(f"Unsupported target input dim for feature naming: {target_input_dim}")
    return names


def summarize_chart_events(events: torch.Tensor, target_input_dim: int) -> Tuple[torch.Tensor, List[str]]:
    feature_names = build_event_feature_names(target_input_dim)
    if events.dim() != 2:
        raise ValueError(f"Expected events shape [T, F], got {tuple(events.shape)}")
    if events.size(-1) != target_input_dim:
        events = project_feature_tensor(events.unsqueeze(0), target_input_dim).squeeze(0)
    if events.size(-1) != len(feature_names):
        raise ValueError(
            f"Feature name count mismatch: expected {len(feature_names)}, got feature dim {events.size(-1)}"
        )

    if events.size(0) == 0:
        mean_vec = torch.zeros(events.size(-1), dtype=torch.float32)
        max_vec = torch.zeros(events.size(-1), dtype=torch.float32)
        std_vec = torch.zeros(events.size(-1), dtype=torch.float32)
    else:
        mean_vec = events.mean(dim=0)
        max_vec = events.max(dim=0).values
        std_vec = events.std(dim=0, unbiased=False)

    summary = torch.cat(
        [
            mean_vec,
            max_vec,
            std_vec,
            torch.tensor([math.log1p(float(events.size(0)))], dtype=torch.float32),
        ],
        dim=0,
    )
    summary_names = (
        [f"mean__{name}" for name in feature_names]
        + [f"max__{name}" for name in feature_names]
        + [f"std__{name}" for name in feature_names]
        + ["chart_num_events_log1p"]
    )
    return summary, summary_names


def build_rows(
    charts: Sequence[str],
    bucket_labels: Sequence[int],
    raw_levels: Sequence[float],
    embeddings: torch.Tensor,
    feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, chart in enumerate(charts):
        row: Dict[str, object] = {
            "chart": chart,
            "bucket_label": DEFAULT_LABEL_CLASSES[int(bucket_labels[index])],
            "raw_level": round(float(raw_levels[index]), 4),
        }
        for feature_idx, feature_name in enumerate(feature_names):
            row[feature_name] = round(float(embeddings[index, feature_idx].item()), 6)
        rows.append(row)
    return rows


def export_csv(rows: Sequence[Dict[str, object]], feature_names: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chart", "bucket_label", "raw_level"] + list(feature_names)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chart-level baseline embeddings from MVP event features.")
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("chart_embeddings.pt"),
        help="Path to save tensor payload with chart embeddings.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV export for chart embeddings.",
    )
    parser.add_argument(
        "--target-input-dim",
        type=int,
        default=84,
        choices=[84, 86],
        help="Project event features to this input dim before chart-level pooling.",
    )
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {data_dir}")

    if args.encoded_dir:
        dataset = PreencodedChartDataset(samples)
    else:
        dataset = EncodedChartDataset(samples, encoder=MVPEventEncoder())

    raw_level_map = load_raw_level_map(args.labels_csv)
    embedding_rows: List[torch.Tensor] = []
    chart_names: List[str] = []
    bucket_labels: List[int] = []
    raw_levels: List[float] = []
    feature_names: Optional[List[str]] = None

    for index in range(len(dataset)):
        chart = dataset[index]
        chart_name = resolve_chart_name(chart.meta or {})
        if not chart_name:
            raise ValueError(f"Could not resolve chart name for sample index {index}")
        if chart.label is None:
            raise ValueError(f"Missing label for chart {chart_name}")

        embedding, current_feature_names = summarize_chart_events(chart.events, args.target_input_dim)
        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise ValueError("Inconsistent embedding feature names across charts.")

        embedding_rows.append(embedding)
        chart_names.append(chart_name)
        bucket_labels.append(int(chart.label))
        raw_levels.append(float(raw_level_map.get(chart_name, 0.0)))

    if feature_names is None:
        raise ValueError("No feature names produced.")

    embeddings = torch.stack(embedding_rows, dim=0)
    payload = {
        "charts": chart_names,
        "bucket_labels": torch.tensor(bucket_labels, dtype=torch.long),
        "bucket_label_names": list(DEFAULT_LABEL_CLASSES),
        "raw_levels": torch.tensor(raw_levels, dtype=torch.float32),
        "embeddings": embeddings,
        "feature_names": feature_names,
        "target_input_dim": args.target_input_dim,
        "source_dir": str(data_dir),
        "source_type": "encoded" if args.encoded_dir else "events",
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"num_charts={len(chart_names)}")
    print(f"embedding_dim={embeddings.size(1)}")
    print(f"output_path={args.output_path}")

    if args.csv_output:
        rows = build_rows(chart_names, bucket_labels, raw_levels, embeddings, feature_names)
        export_csv(rows, feature_names, args.csv_output)
        print(f"csv_output={args.csv_output}")


if __name__ == "__main__":
    main()
