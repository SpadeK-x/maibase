import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


EVENT_TYPE_VOCAB = ["tap", "hold", "slide", "touch", "compound"]
EVENT_TYPE_INDEX = {name: idx for idx, name in enumerate(EVENT_TYPE_VOCAB)}

EVENT_TRAIT_VOCAB = ["none", "b", "x", "bx"]
EVENT_TRAIT_INDEX = {name: idx for idx, name in enumerate(EVENT_TRAIT_VOCAB)}

SLIDE_SHAPE_GROUP_VOCAB = ["none", "line", "curve", "turn", "special"]
SLIDE_SHAPE_GROUP_INDEX = {name: idx for idx, name in enumerate(SLIDE_SHAPE_GROUP_VOCAB)}

OUTER_SLOT_VOCAB_SIZE = 9  # 0..8
INNER_MASK_DIM = 34

EVENT_TYPE_DIM = len(EVENT_TYPE_VOCAB)  # 5
EVENT_TRAIT_DIM = len(EVENT_TRAIT_VOCAB)  # 4
SLIDE_SHAPE_GROUP_DIM = len(SLIDE_SHAPE_GROUP_VOCAB)  # 5
OUTER_IDX_DIM = OUTER_SLOT_VOCAB_SIZE * 2  # 18
MAINLINE_NUMERIC_DIM = 18
LEGACY_EXTRA_NUMERIC_FIELDS = ["slide_conflict_load", "hand_span_pressure"]
NUMERIC_DIM = MAINLINE_NUMERIC_DIM
INPUT_DIM = EVENT_TYPE_DIM + EVENT_TRAIT_DIM + SLIDE_SHAPE_GROUP_DIM + OUTER_IDX_DIM + NUMERIC_DIM + INNER_MASK_DIM

NUMERIC_BLOCK_START = EVENT_TYPE_DIM + EVENT_TRAIT_DIM + SLIDE_SHAPE_GROUP_DIM + OUTER_IDX_DIM
NUMERIC_BLOCK_END = NUMERIC_BLOCK_START + NUMERIC_DIM

NUMERIC_FIELD_ORDER = [
    "delta_time",
    "outer_active",
    "outer_pos_sin_1",
    "outer_pos_cos_1",
    "outer_pos_sin_2",
    "outer_pos_cos_2",
    "inner_count",
    "cross_zone_flag",
    "outer_move_dist",
    "inner_add_count",
    "inner_remove_count",
    "hold_active",
    "hold_remaining_time",
    "slide_active",
    "slide_remaining_time",
    "slide_span",
    "slide_conflict_flag",
    "local_density_500ms",
]
ALL_NUMERIC_FIELD_ORDER = NUMERIC_FIELD_ORDER + LEGACY_EXTRA_NUMERIC_FIELDS

STANDARDIZED_NUMERIC_FIELDS = [
    "delta_time",
    "inner_count",
    "outer_move_dist",
    "inner_add_count",
    "inner_remove_count",
    "hold_remaining_time",
    "slide_remaining_time",
    "slide_span",
    "local_density_500ms",
]
STANDARDIZED_NUMERIC_INDICES = [NUMERIC_FIELD_ORDER.index(name) for name in STANDARDIZED_NUMERIC_FIELDS]


def one_hot(index: int, size: int) -> List[float]:
    vec = [0.0] * size
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def require_list(value, name: str, expected_len: int) -> Sequence[float]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(f"Expected `{name}` to be a list of length {expected_len}, got {value!r}")
    return value


def normalize_numeric(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric value, got {value!r}")


def get_numeric_feature_indices(field_names: Sequence[str]) -> List[int]:
    indices: List[int] = []
    for name in field_names:
        if name not in ALL_NUMERIC_FIELD_ORDER:
            raise KeyError(f"Unknown numeric field: {name}")
        indices.append(NUMERIC_BLOCK_START + ALL_NUMERIC_FIELD_ORDER.index(name))
    return indices


def project_feature_tensor(batch_x: torch.Tensor, target_input_dim: int) -> torch.Tensor:
    source_dim = int(batch_x.size(-1))
    if source_dim == target_input_dim:
        return batch_x

    cat_end = NUMERIC_BLOCK_START
    if source_dim == 89 and target_input_dim == 84:
        numeric = batch_x[:, :, cat_end:cat_end + 18]
        inner = batch_x[:, :, -34:]
        return torch.cat([batch_x[:, :, :cat_end], numeric, inner], dim=-1)

    if source_dim == 89 and target_input_dim == 86:
        numeric_base = batch_x[:, :, cat_end:cat_end + 18]
        slide_conflict_load = batch_x[:, :, cat_end + 20:cat_end + 21]
        hand_span_pressure = batch_x[:, :, cat_end + 21:cat_end + 22]
        inner = batch_x[:, :, -34:]
        return torch.cat(
            [batch_x[:, :, :cat_end], numeric_base, slide_conflict_load, hand_span_pressure, inner],
            dim=-1,
        )

    if source_dim == 86 and target_input_dim == 84:
        numeric = batch_x[:, :, cat_end:cat_end + 18]
        inner = batch_x[:, :, -34:]
        return torch.cat([batch_x[:, :, :cat_end], numeric, inner], dim=-1)

    raise ValueError(f"Unsupported feature projection: source_dim={source_dim} target_input_dim={target_input_dim}")


def apply_zero_feature_mask(batch_x: torch.Tensor, zero_feature_indices: Optional[torch.Tensor]) -> torch.Tensor:
    if zero_feature_indices is None or zero_feature_indices.numel() == 0:
        return batch_x
    active_indices = zero_feature_indices[zero_feature_indices < batch_x.size(-1)]
    if active_indices.numel() == 0:
        return batch_x
    batch_x[:, :, active_indices] = 0.0
    return batch_x


@dataclass
class EncodedChart:
    events: torch.Tensor  # [T, INPUT_DIM]
    length: int
    label: Optional[int] = None
    meta: Optional[Dict[str, object]] = None


@dataclass
class NumericNormalizerState:
    mean: torch.Tensor
    std: torch.Tensor


class MVPEventEncoder:
    """
    Encodes one parsed event record into the fixed input vector for the MVP event models.
    """

    input_dim: int = INPUT_DIM

    def __init__(self) -> None:
        self.numeric_mean: Optional[torch.Tensor] = None
        self.numeric_std: Optional[torch.Tensor] = None

    def encode_event(self, record: Dict[str, object], apply_normalization: bool = True) -> List[float]:
        event_type = str(record["event_type"])
        event_trait = str(record["event_trait"])
        slide_shape_group = str(record["slide_shape_group"])

        if event_type not in EVENT_TYPE_INDEX:
            raise KeyError(f"Unknown event_type: {event_type}")
        if event_trait not in EVENT_TRAIT_INDEX:
            raise KeyError(f"Unknown event_trait: {event_trait}")
        if slide_shape_group not in SLIDE_SHAPE_GROUP_INDEX:
            raise KeyError(f"Unknown slide_shape_group: {slide_shape_group}")

        outer_idx = require_list(record["outer_idx"], "outer_idx", 2)
        outer_pos_sin = require_list(record["outer_pos_sin"], "outer_pos_sin", 2)
        outer_pos_cos = require_list(record["outer_pos_cos"], "outer_pos_cos", 2)
        inner_mask = require_list(record["inner_mask"], "inner_mask", INNER_MASK_DIM)

        vector: List[float] = []

        # 1. one-hot blocks
        vector.extend(one_hot(EVENT_TYPE_INDEX[event_type], EVENT_TYPE_DIM))
        vector.extend(one_hot(EVENT_TRAIT_INDEX[event_trait], EVENT_TRAIT_DIM))
        vector.extend(one_hot(SLIDE_SHAPE_GROUP_INDEX[slide_shape_group], SLIDE_SHAPE_GROUP_DIM))
        vector.extend(one_hot(int(outer_idx[0]), OUTER_SLOT_VOCAB_SIZE))
        vector.extend(one_hot(int(outer_idx[1]), OUTER_SLOT_VOCAB_SIZE))

        # 2. numeric block (18 dims, strict 21-field mainline order)
        vector.append(normalize_numeric(record["delta_time"]))
        vector.append(normalize_numeric(record["outer_active"]))
        vector.append(normalize_numeric(outer_pos_sin[0]))
        vector.append(normalize_numeric(outer_pos_cos[0]))
        vector.append(normalize_numeric(outer_pos_sin[1]))
        vector.append(normalize_numeric(outer_pos_cos[1]))
        vector.append(normalize_numeric(record["inner_count"]))
        vector.append(normalize_numeric(record["cross_zone_flag"]))
        vector.append(normalize_numeric(record["outer_move_dist"]))
        vector.append(normalize_numeric(record["inner_add_count"]))
        vector.append(normalize_numeric(record["inner_remove_count"]))
        vector.append(normalize_numeric(record["hold_active"]))
        vector.append(normalize_numeric(record["hold_remaining_time"]))
        vector.append(normalize_numeric(record["slide_active"]))
        vector.append(normalize_numeric(record["slide_remaining_time"]))
        vector.append(normalize_numeric(record["slide_span"]))
        vector.append(normalize_numeric(record["slide_conflict_flag"]))
        vector.append(normalize_numeric(record["local_density_500ms"]))

        # 3. inner multi-hot block
        vector.extend(normalize_numeric(x) for x in inner_mask)

        if len(vector) != self.input_dim:
            raise ValueError(f"Encoded event dim mismatch: expected {self.input_dim}, got {len(vector)}")
        if apply_normalization:
            vector = self.apply_numeric_normalization(vector)
        return vector

    def apply_numeric_normalization(self, vector: List[float]) -> List[float]:
        if self.numeric_mean is None or self.numeric_std is None:
            return vector
        normalized = list(vector)
        for local_idx, field_idx in enumerate(STANDARDIZED_NUMERIC_INDICES):
            absolute_idx = NUMERIC_BLOCK_START + field_idx
            normalized[absolute_idx] = (
                normalized[absolute_idx] - float(self.numeric_mean[local_idx].item())
            ) / float(self.numeric_std[local_idx].item())
        return normalized

    def encode_records(self, records: Sequence[Dict[str, object]], apply_normalization: bool = True) -> torch.Tensor:
        rows = [self.encode_event(record, apply_normalization=apply_normalization) for record in records]
        if not rows:
            return torch.zeros(0, self.input_dim, dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def load_records_from_json(self, path: Path) -> List[Dict[str, object]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON list in {path}")
        return data

    def encode_json_file(self, path: Path, label: Optional[int] = None, apply_normalization: bool = True) -> EncodedChart:
        records = self.load_records_from_json(path)
        events = self.encode_records(records, apply_normalization=apply_normalization)
        meta = {"path": str(path), "num_events": len(records)}
        return EncodedChart(events=events, length=events.size(0), label=label, meta=meta)

    def fit_normalizer_from_paths(self, paths: Sequence[Path]) -> None:
        total_count = 0
        sum_vec = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float64)
        sumsq_vec = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float64)

        for path in paths:
            records = self.load_records_from_json(path)
            raw_events = self.encode_records(records, apply_normalization=False)
            if raw_events.numel() == 0:
                continue
            numeric_block = raw_events[:, NUMERIC_BLOCK_START:NUMERIC_BLOCK_END]
            selected = numeric_block[:, STANDARDIZED_NUMERIC_INDICES].to(torch.float64)
            total_count += selected.size(0)
            sum_vec += selected.sum(dim=0)
            sumsq_vec += (selected * selected).sum(dim=0)

        if total_count == 0:
            self.numeric_mean = torch.zeros(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float32)
            self.numeric_std = torch.ones(len(STANDARDIZED_NUMERIC_INDICES), dtype=torch.float32)
            return

        mean = sum_vec / total_count
        var = (sumsq_vec / total_count) - mean * mean
        var = torch.clamp(var, min=1e-8)
        std = torch.sqrt(var)
        self.numeric_mean = mean.to(torch.float32)
        self.numeric_std = std.to(torch.float32)

    def export_normalizer_state(self) -> NumericNormalizerState:
        if self.numeric_mean is None or self.numeric_std is None:
            raise ValueError("Normalizer has not been fitted yet.")
        return NumericNormalizerState(
            mean=self.numeric_mean.clone(),
            std=self.numeric_std.clone(),
        )

    def load_normalizer_state(self, state: NumericNormalizerState) -> None:
        self.numeric_mean = state.mean.clone().to(torch.float32)
        self.numeric_std = state.std.clone().to(torch.float32)


class EncodedChartDataset(Dataset):
    """
    Minimal dataset wrapper around already-parsed event JSON files.

    Each sample is:
        {
            "path": Path to event-json file,
            "label": optional class index,
            "meta": optional free-form metadata
        }
    """

    def __init__(self, samples: Sequence[Dict[str, object]], encoder: Optional[MVPEventEncoder] = None) -> None:
        self.samples = list(samples)
        self.encoder = encoder or MVPEventEncoder()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedChart:
        sample = self.samples[index]
        path = Path(sample["path"])
        label = sample.get("label")
        encoded = self.encoder.encode_json_file(path, label=label)
        if sample.get("meta") is not None:
            encoded.meta = dict(sample["meta"])
            encoded.meta["path"] = str(path)
            encoded.meta["num_events"] = encoded.length
        return encoded


class PreencodedChartDataset(Dataset):
    """
    Dataset backed by precomputed `.pt` files.

    Each file is expected to contain:
      {
        "events": FloatTensor [T, INPUT_DIM],
        "length": int,
        "label": int,
        "meta": dict
      }
    """

    def __init__(self, samples: Sequence[Dict[str, object]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EncodedChart:
        sample = self.samples[index]
        path = Path(sample["path"])
        payload = torch.load(path, map_location="cpu")
        events = payload["events"].to(torch.float32)
        length = int(payload["length"])
        label = payload.get("label")
        meta = dict(sample.get("meta") or {})
        meta.update(dict(payload.get("meta") or {}))
        meta["path"] = str(path)
        return EncodedChart(events=events, length=length, label=label, meta=meta)


def collate_encoded_charts(batch: Sequence[EncodedChart]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[Dict[str, object]]]:
    if not batch:
        raise ValueError("Batch is empty")

    batch_size = len(batch)
    max_len = max(item.length for item in batch)
    feature_dim = batch[0].events.size(-1) if batch[0].length > 0 else INPUT_DIM

    padded = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    labels: List[int] = []
    metas: List[Dict[str, object]] = []
    has_all_labels = True

    for i, item in enumerate(batch):
        lengths[i] = item.length
        if item.length > 0:
            padded[i, :item.length] = item.events
        if item.label is None:
            has_all_labels = False
        else:
            labels.append(int(item.label))
        metas.append(item.meta or {})

    label_tensor = torch.tensor(labels, dtype=torch.long) if has_all_labels else None
    return padded, lengths, label_tensor, metas


def smoke_test(json_path: Optional[str] = None) -> None:
    encoder = MVPEventEncoder()
    if json_path is None:
        fake_record = {
            "event_type": "tap",
            "event_trait": "b",
            "slide_shape_group": "none",
            "outer_idx": [1, 8],
            "outer_pos_sin": [0.707107, -0.0],
            "outer_pos_cos": [0.707107, 1.0],
            "inner_mask": [0] * INNER_MASK_DIM,
            "delta_time": 0.1,
            "outer_active": 1,
            "inner_count": 0,
            "cross_zone_flag": 0,
            "outer_move_dist": 0.25,
            "inner_add_count": 0,
            "inner_remove_count": 0,
            "hold_active": 0,
            "hold_remaining_time": 0.0,
            "slide_active": 0,
            "slide_remaining_time": 0.0,
            "slide_span": 0.0,
            "slide_conflict_flag": 0,
            "local_density_500ms": 0.69,
            "slide_conflict_load": 0.0,
            "hand_span_pressure": 0.0,
        }
        x = encoder.encode_records([fake_record])
        print("encoded_shape:", tuple(x.shape))
        print("feature_dim:", x.size(-1))
        return

    chart = encoder.encode_json_file(Path(json_path))
    print("events_shape:", tuple(chart.events.shape))
    print("num_events:", chart.length)


if __name__ == "__main__":
    smoke_test()
