import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from mvp_event_encoder import MVPEventEncoder
from mvp_event_encoder import NUMERIC_BLOCK_END, NUMERIC_BLOCK_START, STANDARDIZED_NUMERIC_INDICES
from train_mvp_mlp import (
    discover_samples,
    print_dataset_distributions,
    set_seed,
    split_dataset,
    TrainConfig,
    EncodedChartDataset,
)

BOS_PATTERN_TOKEN = "__bos__"


def convert_json_name_to_pt_name(json_path: Path) -> str:
    return f"{json_path.stem}.pt"


def build_pattern_token(record: Dict[str, object]) -> str:
    outer_idx = record.get("outer_idx") or [0, 0]
    outer_count = sum(1 for value in outer_idx if int(value) != 0)
    inner_count = int(record.get("inner_count", 0))
    if inner_count >= 2:
        inner_bucket = "2_plus"
    else:
        inner_bucket = str(inner_count)
    return "|".join(
        [
            str(record.get("event_type", "none")),
            f"o{outer_count}",
            f"i{inner_bucket}",
            f"sa{int(record.get('slide_active', 0))}",
            f"sg{record.get('slide_shape_group', 'none')}",
            f"sc{int(record.get('slide_conflict_flag', 0))}",
        ]
    )


def build_pattern_key(tokens: Sequence[str], index: int) -> Tuple[str, str, str]:
    left2 = tokens[index - 2] if index - 2 >= 0 else BOS_PATTERN_TOKEN
    left1 = tokens[index - 1] if index - 1 >= 0 else BOS_PATTERN_TOKEN
    current = tokens[index]
    return left2, left1, current


def fit_pattern_stats(paths: Sequence[Path], encoder: MVPEventEncoder) -> Tuple[Counter, int]:
    pattern_counter: Counter = Counter()
    total_patterns = 0

    for path in paths:
        records = encoder.load_records_from_json(path)
        tokens = [build_pattern_token(record) for record in records]
        for index in range(len(tokens)):
            pattern_counter[build_pattern_key(tokens, index)] += 1
            total_patterns += 1

    return pattern_counter, total_patterns


def attach_pattern_novelty(
    records: Sequence[Dict[str, object]],
    pattern_counter: Counter,
    total_patterns: int,
) -> List[Dict[str, object]]:
    if not records:
        return []

    tokens = [build_pattern_token(record) for record in records]
    augmented: List[Dict[str, object]] = []
    denom = total_patterns + 1

    for index, record in enumerate(records):
        pattern = build_pattern_key(tokens, index)
        freq = pattern_counter.get(pattern, 0)
        novelty = -math.log((freq + 1) / denom)
        updated = dict(record)
        updated["pattern_novelty_local"] = round(novelty, 6)
        augmented.append(updated)

    return augmented


def fit_normalizer_from_augmented_paths(
    paths: Sequence[Path],
    encoder: MVPEventEncoder,
    pattern_counter: Counter,
    total_patterns: int,
) -> None:
    total_count = 0
    sum_vec = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float64)
    sumsq_vec = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float64)

    for path in paths:
        records = encoder.load_records_from_json(path)
        records = attach_pattern_novelty(records, pattern_counter, total_patterns)
        raw_events = encoder.encode_records(records, apply_normalization=False)
        if raw_events.numel() == 0:
            continue
        numeric_block = raw_events[:, NUMERIC_BLOCK_START:NUMERIC_BLOCK_END]
        selected = numeric_block[:, STANDARDIZED_NUMERIC_INDICES].to(torch.float64)
        total_count += selected.size(0)
        sum_vec += selected.sum(dim=0)
        sumsq_vec += (selected * selected).sum(dim=0)

    if total_count == 0:
        encoder.numeric_mean = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float32)
        encoder.numeric_std = torch.ones(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float32)
        return

    mean = sum_vec / total_count
    var = (sumsq_vec / total_count) - mean * mean
    var = torch.clamp(var, min=1e-8)
    encoder.numeric_mean = mean.to(torch.float32)
    encoder.numeric_std = torch.sqrt(var).to(torch.float32)


def write_preencoded_samples(
    samples: List[Dict[str, object]],
    encoder: MVPEventEncoder,
    output_dir: Path,
    pattern_counter: Counter,
    total_patterns: int,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []

    for sample in samples:
        json_path = Path(sample["path"])
        records = encoder.load_records_from_json(json_path)
        records = attach_pattern_novelty(records, pattern_counter, total_patterns)
        events = encoder.encode_records(records, apply_normalization=True)
        encoded = encoder.encode_json_file(json_path, label=sample.get("label"), apply_normalization=False)
        encoded.events = events
        encoded.length = events.size(0)
        output_path = output_dir / convert_json_name_to_pt_name(json_path)
        payload_meta = dict(sample.get("meta") or {})
        payload_meta.update(encoded.meta or {})
        payload = {
            "events": encoded.events,
            "length": encoded.length,
            "label": encoded.label,
            "meta": payload_meta,
        }
        torch.save(payload, output_path)
        manifest.append(
            {
                "path": str(output_path),
                "label": sample.get("label"),
                "meta": dict(payload_meta),
            }
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-encode event JSON files into tensor `.pt` samples.")
    parser.add_argument("--events-dir", type=Path, required=True, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store encoded `.pt` files.")
    parser.add_argument("--manifest-path", type=Path, help="Optional path to save manifest `.pt`.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(seed=args.seed)
    set_seed(config.seed)

    samples = discover_samples(args.events_dir, args.labels_csv, [".json"])
    if not samples:
        raise ValueError(f"No labeled event-json samples found in {args.events_dir}")

    encoder = MVPEventEncoder()
    dataset = EncodedChartDataset(samples, encoder=encoder)
    train_set, eval_set, test_set = split_dataset(dataset, config)
    print_dataset_distributions(dataset, train_set, eval_set, test_set)
    train_paths = [Path(dataset.samples[i]["path"]) for i in train_set.indices]
    pattern_counter, total_patterns = fit_pattern_stats(train_paths, encoder)
    fit_normalizer_from_augmented_paths(train_paths, encoder, pattern_counter, total_patterns)

    manifest = write_preencoded_samples(samples, encoder, args.output_dir, pattern_counter, total_patterns)

    if args.manifest_path:
        payload = {
            "samples": manifest,
            "numeric_mean": encoder.numeric_mean,
            "numeric_std": encoder.numeric_std,
            "pattern_total": total_patterns,
        }
        torch.save(payload, args.manifest_path)
        print(f"saved_manifest={args.manifest_path}")

    print(f"written={len(manifest)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
