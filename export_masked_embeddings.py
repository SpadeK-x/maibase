import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from mvp_event_encoder import EncodedChartDataset, MVPEventEncoder, NumericNormalizerState, PreencodedChartDataset, project_feature_tensor
from mvp_masked_event_model import build_masked_event_model
from mvp_mlp_model import make_padding_mask
from train_mvp_mlp import DEFAULT_LABEL_CLASSES
from train_mvp_mlp import discover_samples


def resolve_chart_name(meta: Dict[str, object]) -> str:
    chart = meta.get("chart")
    if chart:
        return str(chart)
    path_value = meta.get("path")
    if path_value:
        return Path(str(path_value)).stem
    return ""


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


def load_model(checkpoint_path: Path, device: str):
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["model_config"]
    model = build_masked_event_model(
        input_dim=int(config["input_dim"]),
        model_dim=int(config["model_dim"]),
        ff_dim=int(config["ff_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        pooling=str(config["pooling"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def maybe_build_encoder(payload: Dict[str, object]) -> MVPEventEncoder:
    encoder = MVPEventEncoder()
    mean = payload.get("numeric_mean")
    std = payload.get("numeric_std")
    if mean is not None and std is not None:
        encoder.load_normalizer_state(NumericNormalizerState(mean=torch.as_tensor(mean), std=torch.as_tensor(std)))
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Export chart embeddings from a masked event modeling checkpoint.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--events-dir", type=Path, help="Directory containing per-chart event JSON files.")
    parser.add_argument("--encoded-dir", type=Path, help="Directory containing pre-encoded per-chart `.pt` files.")
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("masked_chart_embeddings.pt"))
    args = parser.parse_args()

    if not args.events_dir and not args.encoded_dir:
        raise ValueError("Either --events-dir or --encoded-dir must be provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_model(args.model_path, device)

    data_dir = args.encoded_dir if args.encoded_dir else args.events_dir
    suffixes = [".pt"] if args.encoded_dir else [".json"]
    samples = discover_samples(data_dir, args.labels_csv, suffixes)
    if not samples:
        raise ValueError(f"No labeled samples found in {data_dir}")
    raw_level_map = load_raw_level_map(args.labels_csv)

    if args.encoded_dir:
        dataset = PreencodedChartDataset(samples)
    else:
        dataset = EncodedChartDataset(samples, encoder=maybe_build_encoder(payload))

    chart_names: List[str] = []
    bucket_labels: List[int] = []
    raw_levels: List[float] = []
    embeddings: List[torch.Tensor] = []
    for index in range(len(dataset)):
        chart = dataset[index]
        chart_name = resolve_chart_name(chart.meta or {})
        if not chart_name:
            raise ValueError(f"Could not resolve chart name for sample index {index}")

        events = project_feature_tensor(chart.events.unsqueeze(0), model.config.input_dim).to(device)
        lengths = torch.tensor([chart.length], dtype=torch.long, device=device)
        valid_mask = make_padding_mask(lengths, events.size(1))
        with torch.no_grad():
            chart_embedding, _seq_embeddings, _token_embeddings = model.encode(events, valid_mask, masked_positions=None)
        chart_names.append(chart_name)
        bucket_labels.append(int(chart.label) if chart.label is not None else -1)
        raw_levels.append(float(raw_level_map.get(chart_name, 0.0)))
        embeddings.append(chart_embedding.squeeze(0).cpu())

    output_payload = {
        "charts": chart_names,
        "bucket_labels": torch.tensor(bucket_labels, dtype=torch.long),
        "bucket_label_names": list(DEFAULT_LABEL_CLASSES),
        "raw_levels": torch.tensor(raw_levels, dtype=torch.float32),
        "embeddings": torch.stack(embeddings, dim=0),
        "model_path": str(args.model_path),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output_path)
    print(f"num_charts={len(chart_names)}")
    print(f"embedding_dim={output_payload['embeddings'].size(1)}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
