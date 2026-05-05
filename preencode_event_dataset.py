import argparse
from pathlib import Path
from typing import Dict, List

import torch

from mvp_event_encoder import MVPEventEncoder
from train_mvp_mlp import (
    discover_samples,
    fit_train_normalizer,
    print_dataset_distributions,
    set_seed,
    split_dataset,
    TrainConfig,
    EncodedChartDataset,
)


def convert_json_name_to_pt_name(json_path: Path) -> str:
    return f"{json_path.stem}.pt"


def write_preencoded_samples(
    samples: List[Dict[str, object]],
    encoder: MVPEventEncoder,
    output_dir: Path,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, object]] = []

    for sample in samples:
        json_path = Path(sample["path"])
        encoded = encoder.encode_json_file(json_path, label=sample.get("label"), apply_normalization=True)
        output_path = output_dir / convert_json_name_to_pt_name(json_path)
        payload = {
            "events": encoded.events,
            "length": encoded.length,
            "label": encoded.label,
            "meta": encoded.meta or sample.get("meta") or {},
        }
        torch.save(payload, output_path)
        manifest.append(
            {
                "path": str(output_path),
                "label": sample.get("label"),
                "meta": payload["meta"],
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
    fit_train_normalizer(dataset, train_set)

    manifest = write_preencoded_samples(samples, encoder, args.output_dir)

    if args.manifest_path:
        payload = {
            "samples": manifest,
            "numeric_mean": encoder.numeric_mean,
            "numeric_std": encoder.numeric_std,
        }
        torch.save(payload, args.manifest_path)
        print(f"saved_manifest={args.manifest_path}")

    print(f"written={len(manifest)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
