import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Set


OUTER_REGIONS = {str(i) for i in range(1, 9)}
INNER_REGIONS = (
    [f"A{i}" for i in range(1, 9)]
    + [f"B{i}" for i in range(1, 9)]
    + [f"C{i}" for i in range(1, 3)]
    + [f"D{i}" for i in range(1, 9)]
    + [f"E{i}" for i in range(1, 9)]
)
INNER_INDEX = {name: idx for idx, name in enumerate(INNER_REGIONS)}
ALL_REGIONS = sorted(list(OUTER_REGIONS) + list(INNER_REGIONS), key=len, reverse=True)
REGION_PATTERN = "|".join(re.escape(x) for x in ALL_REGIONS)

TRAIT_PATTERN = r"[bxf]*"
RAW_SHAPE_PATTERN = r"(pp|qq|p|q|h|-|<|>|\^|v|V|w|s|z)"
TIME_RATIO_PATTERN = r"\[(?P<a>\d+):(?P<b>\d+)\]"
TIME_HASH_PATTERN = r"\[(?P<x>\d+(?:\.\d+)?)##(?P<y>\d+(?:\.\d+)?)\]"

PREFIX_REGEX = re.compile(
    rf"^\s*(?P<region>{REGION_PATTERN})(?P<trait>{TRAIT_PATTERN})(?P<rest>.*)$"
)
BPM_REGEX = re.compile(r"\((?P<bpm>\d+(?:\.\d+)?)\)")
GRID_REGEX = re.compile(r"^\{(?P<grid>\d+)\}")
TOKEN_SPLIT_REGEX = re.compile(r"([,/])")


# Manual slide maps. Fill these out as domain knowledge is finalized.
SLIDE_SHAPE_GROUP_MAP: Dict[str, str] = {
    "-": "line",
    "<": "curve",
    ">": "curve",
    "p": "turn",
    "q": "turn",
    "pp": "turn",
    "qq": "turn",
    "^": "curve",
    "v": "curve",
    "V": "curve",
    "w": "special",
    "s": "curve",
    "z": "curve",
}
SLIDE_DIRECTION_MAP: Dict[str, str] = {
    "-": "none",
    "<": "ccw",
    ">": "cw",
    "p": "cw",
    "q": "ccw",
    "pp": "cw",
    "qq": "ccw",
    "^": "none",
    "v": "none",
    "V": "none",
    "w": "none",
    "s": "none",
    "z": "none",
}
SLIDE_SHAPE_SPAN_MAP: Dict[str, float] = {
    "-": 0.0,   # 直线
    "<": 1.0,   # 简单曲线
    ">": 1.0,
    "^": 1.0,
    "v": 1.0,
    "V": 1.0,
    "s": 1.0,
    "z": 1.0,
    "w": 1.0,
    "p": 2.0,   # 简单转向
    "q": 2.0,
    "pp": 3.0,  # 更复杂转向
    "qq": 3.0,
}


def shortest_ring_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 8 - diff)


def trait_union_to_label(has_b: bool, has_x: bool) -> str:
    if has_b and has_x:
        return "bx"
    if has_b:
        return "b"
    if has_x:
        return "x"
    return "none"


@dataclass
class AtomicObject:
    object_type: str
    start_time: float
    trait_b: bool
    trait_x: bool
    outer_positions: List[int] = field(default_factory=list)
    inner_regions: List[str] = field(default_factory=list)
    duration: float = 0.0
    raw_shape: str = ""
    slide_start: Optional[int] = None
    slide_end: Optional[int] = None
    slide_path_span: float = 0.0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class EventGroup:
    event_time: float
    atomic_objects: List[AtomicObject] = field(default_factory=list)

    @property
    def outer_set(self) -> List[int]:
        values = sorted({x for obj in self.atomic_objects for x in obj.outer_positions})
        return values[:2]

    @property
    def inner_set(self) -> List[str]:
        return sorted({x for obj in self.atomic_objects for x in obj.inner_regions}, key=lambda x: INNER_INDEX[x])

    @property
    def object_types(self) -> Set[str]:
        return {obj.object_type for obj in self.atomic_objects}

    @property
    def trait_label(self) -> str:
        has_b = any(obj.trait_b for obj in self.atomic_objects)
        has_x = any(obj.trait_x for obj in self.atomic_objects)
        return trait_union_to_label(has_b, has_x)

    @property
    def new_holds(self) -> List[AtomicObject]:
        return [obj for obj in self.atomic_objects if obj.object_type == "hold"]

    @property
    def new_slides(self) -> List[AtomicObject]:
        return [obj for obj in self.atomic_objects if obj.object_type == "slide"]


class SimaiMVPParser:
    def __init__(self) -> None:
        self.current_bpm = 120.0
        self.current_time = 0.0

    def parse_chart_section(self, lines: Sequence[str]) -> List[Dict[str, object]]:
        atomic_objects: List[AtomicObject] = []
        self.current_time = 0.0
        for line in lines:
            atomic_objects.extend(self.parse_line(line.strip()))
        event_groups = self.merge_event_groups(atomic_objects)
        return self.generate_records(event_groups)

    def parse_line(self, line: str) -> List[AtomicObject]:
        if not line:
            return []

        self._consume_inline_bpms_at_line_start(line)

        while True:
            bpm_match = BPM_REGEX.match(line)
            if not bpm_match:
                break
            self.current_bpm = float(bpm_match.group("bpm"))
            line = line[bpm_match.end():]

        grid_match = GRID_REGEX.match(line)
        if not grid_match:
            return []
        grid = int(grid_match.group("grid"))
        step = 60.0 / self.current_bpm * 4.0 / grid
        body = line[grid_match.end():]

        tokens = TOKEN_SPLIT_REGEX.split(body)
        objects: List[AtomicObject] = []
        cursor_time = self.current_time

        for idx in range(0, len(tokens), 2):
            raw_token = tokens[idx].strip()
            delimiter = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if raw_token:
                token_time, token_payload = self.extract_inline_bpm(raw_token)
                cursor_time = token_time if token_time is not None else cursor_time
                objects.extend(self.parse_event_token(token_payload, cursor_time))
            if delimiter == ",":
                cursor_time += step

        self.current_time = cursor_time
        return objects

    def _consume_inline_bpms_at_line_start(self, line: str) -> None:
        # no-op placeholder for readability; start-of-line BPM is handled in parse_line
        return

    def extract_inline_bpm(self, token: str) -> tuple[Optional[float], str]:
        bpm_match = BPM_REGEX.match(token)
        if not bpm_match:
            return None, token
        self.current_bpm = float(bpm_match.group("bpm"))
        return None, token[bpm_match.end():].strip()

    def parse_event_token(self, token: str, event_time: float) -> List[AtomicObject]:
        parts = [part.strip() for part in token.split("/") if part.strip()]
        objects: List[AtomicObject] = []
        for part in parts:
            objects.extend(self.parse_atomic_part(part, event_time))
        return objects

    def parse_atomic_part(self, part: str, event_time: float) -> List[AtomicObject]:
        bpm_prefix = BPM_REGEX.match(part)
        if bpm_prefix:
            self.current_bpm = float(bpm_prefix.group("bpm"))
            part = part[bpm_prefix.end():].strip()

        if not part:
            return []

        match = PREFIX_REGEX.match(part)
        if not match:
            return []

        region = match.group("region")
        trait_text = match.group("trait") or ""
        rest = (match.group("rest") or "").strip()
        trait_b = "b" in trait_text
        trait_x = "x" in trait_text

        outer_positions = [int(region)] if region in OUTER_REGIONS else []
        inner_regions = [region] if region in INNER_REGIONS else []

        if not rest:
            return [
                AtomicObject(
                    object_type="tap" if outer_positions else "touch",
                    start_time=event_time,
                    trait_b=trait_b,
                    trait_x=trait_x,
                    outer_positions=outer_positions,
                    inner_regions=inner_regions,
                )
            ]

        parsed = self.parse_hold_or_slide(rest, outer_positions, inner_regions, event_time, trait_b, trait_x)
        if parsed is not None:
            return [parsed]

        if not self.looks_like_slide_or_hold(rest):
            return [
                AtomicObject(
                    object_type="tap" if outer_positions else "touch",
                    start_time=event_time,
                    trait_b=trait_b,
                    trait_x=trait_x,
                    outer_positions=outer_positions,
                    inner_regions=inner_regions,
                )
            ]

        return [
            AtomicObject(
                object_type="tap" if outer_positions else "touch",
                start_time=event_time,
                trait_b=trait_b,
                trait_x=trait_x,
                outer_positions=outer_positions,
                inner_regions=inner_regions,
            )
        ]

    def looks_like_slide_or_hold(self, rest: str) -> bool:
        return bool(re.search(TIME_RATIO_PATTERN, rest) or re.search(TIME_HASH_PATTERN, rest))

    def parse_hold_or_slide(
        self,
        rest: str,
        outer_positions: List[int],
        inner_regions: List[str],
        event_time: float,
        trait_b: bool,
        trait_x: bool,
    ) -> Optional[AtomicObject]:
        hold_match = re.match(rf"^(?P<shape>h)(?P<tail>.*)$", rest)
        if hold_match:
            tail = hold_match.group("tail").strip()
            duration = self.parse_duration(tail)
            return AtomicObject(
                object_type="hold",
                start_time=event_time,
                trait_b=trait_b,
                trait_x=trait_x,
                outer_positions=outer_positions,
                inner_regions=inner_regions,
                duration=duration,
                raw_shape="h",
            )

        slide_shape = self.detect_slide_shape(rest)
        if slide_shape is None:
            return None
        duration = self.parse_duration(rest)
        slide_start = outer_positions[0] if outer_positions else None
        slide_end = self.extract_slide_end(rest)
        slide_path_span = self.extract_slide_path_span(rest)
        return AtomicObject(
            object_type="slide",
            start_time=event_time,
            trait_b=trait_b,
            trait_x=trait_x,
            outer_positions=outer_positions,
            inner_regions=inner_regions,
            duration=duration,
            raw_shape=slide_shape,
            slide_start=slide_start,
            slide_end=slide_end,
            slide_path_span=slide_path_span,
        )

    def detect_slide_shape(self, text: str) -> Optional[str]:
        if "[" not in text:
            return None
        head = text.split("[", 1)[0]
        for shape in ("pp", "qq", "p", "q", "h", "-", "<", ">", "^", "v", "V", "w", "s", "z"):
            if shape in head:
                return shape
        return None

    def parse_duration(self, text: str) -> float:
        ratio_match = re.search(TIME_RATIO_PATTERN, text)
        if ratio_match:
            a = float(ratio_match.group("a"))
            b = float(ratio_match.group("b"))
            return 60.0 / self.current_bpm * 4.0 * (b / a)

        hash_match = re.search(TIME_HASH_PATTERN, text)
        if hash_match:
            x = float(hash_match.group("x"))
            y = float(hash_match.group("y"))
            # Stable MVP heuristic for custom timing expressions.
            return max(x, y)

        return 0.0

    def extract_slide_end(self, text: str) -> Optional[int]:
        ratio_index = text.find("[")
        body = text[:ratio_index] if ratio_index != -1 else text
        body = body.split("*")[0]
        trailing = re.findall(r"\d", body)
        if not trailing:
            return None
        return int(trailing[-1])

    def extract_slide_path_span(self, text: str) -> float:
        ratio_index = text.find("[")
        body = text[:ratio_index] if ratio_index != -1 else text
        body = body.split("*")[0]
        digits = re.findall(r"\d", body)
        if len(digits) < 1:
            return 0.0
        return float(len(digits))

    def merge_event_groups(self, atomic_objects: Sequence[AtomicObject]) -> List[EventGroup]:
        grouped: Dict[float, EventGroup] = {}
        for obj in sorted(atomic_objects, key=lambda x: x.start_time):
            group = grouped.setdefault(obj.start_time, EventGroup(event_time=obj.start_time))
            group.atomic_objects.append(obj)
        return [grouped[t] for t in sorted(grouped)]

    def generate_records(self, event_groups: Sequence[EventGroup]) -> List[Dict[str, object]]:
        active_holds: List[AtomicObject] = []
        active_slides: List[AtomicObject] = []
        records: List[Dict[str, object]] = []
        previous_time: Optional[float] = None
        previous_inner_mask = [0] * len(INNER_REGIONS)
        previous_outer = [0, 0]
        raw_delta_history: List[float] = []

        for index, group in enumerate(event_groups):
            current_time = group.event_time
            active_holds = [obj for obj in active_holds if obj.end_time > current_time]
            active_slides = [obj for obj in active_slides if obj.end_time > current_time]
            active_slides_before_new = list(active_slides)
            active_holds.extend(group.new_holds)
            active_slides.extend(group.new_slides)

            outer_idx = self.build_outer_idx(group.outer_set)
            inner_mask = self.build_inner_mask(group.inner_set)

            delta_time_raw = 0.0 if previous_time is None else current_time - previous_time
            event_type = self.build_event_type(group.object_types, group.outer_set, group.inner_set)
            event_trait = group.trait_label
            hold_active = 1 if active_holds else 0
            slide_active = 1 if active_slides else 0

            dominant_slide = self.choose_dominant_slide(active_slides)
            rhythm_irregularity_local = self.rhythm_irregularity_local(delta_time_raw, raw_delta_history)
            burst_compactness = self.burst_compactness(event_groups, index)
            slide_conflict_load = self.slide_conflict_load(group, active_slides_before_new)
            hand_span_pressure = self.hand_span_pressure(group, active_holds, active_slides)

            record = {
                "event_index": index,
                "event_time": round(current_time, 6),
                "delta_time": self.safe_log1p(delta_time_raw),
                "event_type": event_type,
                "event_trait": event_trait,
                "outer_active": 1 if group.outer_set else 0,
                "outer_idx": outer_idx,
                "outer_pos_sin": [self.outer_sin(x) for x in outer_idx],
                "outer_pos_cos": [self.outer_cos(x) for x in outer_idx],
                "inner_mask": inner_mask,
                "inner_count": sum(inner_mask),
                "cross_zone_flag": 1 if group.outer_set and group.inner_set else 0,
                "outer_move_dist": self.compute_outer_move_dist(previous_outer, outer_idx),
                "inner_add_count": self.count_bit_adds(previous_inner_mask, inner_mask),
                "inner_remove_count": self.count_bit_removes(previous_inner_mask, inner_mask),
                "hold_active": hold_active,
                "hold_remaining_time": self.safe_log1p(self.max_remaining_time(active_holds, current_time)),
                "slide_active": slide_active,
                "slide_remaining_time": self.safe_log1p(self.max_remaining_time(active_slides, current_time)),
                "slide_shape_group": self.slide_shape_group(dominant_slide),
                "slide_direction": self.slide_direction(dominant_slide),
                "slide_span": self.slide_span(dominant_slide),
                "slide_conflict_flag": self.slide_conflict_flag(group, active_slides_before_new),
                "local_density_500ms": self.safe_log1p(self.local_density(event_groups, current_time)),
                "rhythm_irregularity_local": round(rhythm_irregularity_local, 6),
                "burst_compactness": round(burst_compactness, 6),
                "slide_conflict_load": round(slide_conflict_load, 6),
                "hand_span_pressure": round(hand_span_pressure, 6),
                "pattern_novelty_local": 0.0,
            }
            records.append(record)

            previous_time = current_time
            previous_inner_mask = inner_mask
            previous_outer = outer_idx
            raw_delta_history.append(delta_time_raw)

        return records

    def build_event_type(self, object_types: Set[str], outer_set: Sequence[int], inner_set: Sequence[str]) -> str:
        if len(object_types) > 1:
            return "compound"
        if object_types == {"touch"}:
            return "touch"
        if object_types == {"hold"}:
            return "hold"
        if object_types == {"slide"}:
            return "slide"
        if object_types == {"tap"}:
            if inner_set and not outer_set:
                return "touch"
            return "tap"
        return "compound"

    def build_outer_idx(self, outer_set: Sequence[int]) -> List[int]:
        values = list(sorted(set(outer_set)))[:2]
        while len(values) < 2:
            values.append(0)
        return values

    def build_inner_mask(self, inner_set: Sequence[str]) -> List[int]:
        mask = [0] * len(INNER_REGIONS)
        for name in inner_set:
            mask[INNER_INDEX[name]] = 1
        return mask

    def outer_sin(self, idx: int) -> float:
        if idx == 0:
            return 0.0
        return round(math.sin(2.0 * math.pi * idx / 8.0), 6)

    def outer_cos(self, idx: int) -> float:
        if idx == 0:
            return 0.0
        return round(math.cos(2.0 * math.pi * idx / 8.0), 6)

    def compute_outer_move_dist(self, prev_outer: Sequence[int], curr_outer: Sequence[int]) -> float:
        prev_vals = [x for x in prev_outer if x]
        curr_vals = [x for x in curr_outer if x]
        if not prev_vals or not curr_vals:
            return 0.0
        if len(prev_vals) == 1 and len(curr_vals) == 1:
            return shortest_ring_distance(prev_vals[0], curr_vals[0]) / 8.0
        if len(prev_vals) == 1:
            best = min(shortest_ring_distance(prev_vals[0], c) for c in curr_vals)
            return best / 8.0
        if len(curr_vals) == 1:
            best = min(shortest_ring_distance(p, curr_vals[0]) for p in prev_vals)
            return best / 8.0
        d1 = shortest_ring_distance(prev_vals[0], curr_vals[0]) + shortest_ring_distance(prev_vals[1], curr_vals[1])
        d2 = shortest_ring_distance(prev_vals[0], curr_vals[1]) + shortest_ring_distance(prev_vals[1], curr_vals[0])
        return min(d1, d2) / 8.0

    def count_bit_adds(self, prev_mask: Sequence[int], curr_mask: Sequence[int]) -> int:
        return sum(1 for p, c in zip(prev_mask, curr_mask) if p == 0 and c == 1)

    def count_bit_removes(self, prev_mask: Sequence[int], curr_mask: Sequence[int]) -> int:
        return sum(1 for p, c in zip(prev_mask, curr_mask) if p == 1 and c == 0)

    def max_remaining_time(self, objects: Sequence[AtomicObject], current_time: float) -> float:
        if not objects:
            return 0.0
        return max(max(0.0, obj.end_time - current_time) for obj in objects)

    def choose_dominant_slide(self, active_slides: Sequence[AtomicObject]) -> Optional[AtomicObject]:
        if not active_slides:
            return None
        return max(active_slides, key=lambda x: x.start_time)

    def slide_shape_group(self, slide: Optional[AtomicObject]) -> str:
        if slide is None:
            return "none"
        return SLIDE_SHAPE_GROUP_MAP.get(slide.raw_shape, "none")

    def slide_direction(self, slide: Optional[AtomicObject]) -> str:
        if slide is None:
            return "none"
        return SLIDE_DIRECTION_MAP.get(slide.raw_shape, "none")

    def slide_span(self, slide: Optional[AtomicObject]) -> float:
        if slide is None or slide.slide_start is None or slide.slide_end is None:
            return 0.0
        positional = shortest_ring_distance(slide.slide_start, slide.slide_end)
        shape_span = SLIDE_SHAPE_SPAN_MAP.get(slide.raw_shape, 0.0)
        path_span = slide.slide_path_span
        return float(max(positional, path_span) + shape_span)

    def slide_conflict_flag(self, group: EventGroup, active_slides_before_new: Sequence[AtomicObject]) -> int:
        if not active_slides_before_new:
            return 0
        extra_inputs = [obj for obj in group.atomic_objects if obj.object_type in {"tap", "touch", "hold", "slide"}]
        return 1 if extra_inputs else 0

    def rhythm_irregularity_local(self, delta_time_raw: float, raw_delta_history: Sequence[float]) -> float:
        if delta_time_raw <= 0:
            return 0.0
        recent_non_zero = [value for value in reversed(raw_delta_history) if value > 0][:4]
        if not recent_non_zero:
            return 0.0
        med = float(median(recent_non_zero))
        eps = 1e-6
        return abs(math.log(delta_time_raw + eps) - math.log(med + eps))

    def burst_compactness(self, event_groups: Sequence[EventGroup], current_index: int) -> float:
        start_index = max(0, current_index - 4)
        window = event_groups[start_index:current_index + 1]
        if len(window) <= 1:
            return 0.0
        span = max(0.0, window[-1].event_time - window[0].event_time)
        return 1.0 / (span + 1e-6)

    def slide_conflict_load(self, group: EventGroup, active_slides_before_new: Sequence[AtomicObject]) -> float:
        if not active_slides_before_new:
            return 0.0
        new_outer_count = len(group.outer_set)
        new_inner_count = len(group.inner_set)
        new_hold_count = len(group.new_holds)
        new_slide_count = len(group.new_slides)
        return float(new_outer_count + 0.5 * new_inner_count + 0.5 * new_hold_count + 1.0 * new_slide_count)

    def hand_span_pressure(
        self,
        group: EventGroup,
        active_holds: Sequence[AtomicObject],
        active_slides: Sequence[AtomicObject],
    ) -> float:
        current_outer = list(group.outer_set)
        current_span = 0.0
        if len(current_outer) >= 2:
            current_span = shortest_ring_distance(current_outer[0], current_outer[1]) / 4.0

        active_anchors: List[int] = []
        for obj in active_holds:
            active_anchors.extend(pos for pos in obj.outer_positions if pos)
        for obj in active_slides:
            if obj.slide_start is not None:
                active_anchors.append(obj.slide_start)
            if obj.slide_end is not None:
                active_anchors.append(obj.slide_end)

        if not current_outer or not active_anchors:
            return current_span

        max_nearest_dist = 0.0
        for current_pos in current_outer:
            nearest = min(shortest_ring_distance(current_pos, anchor) for anchor in active_anchors)
            max_nearest_dist = max(max_nearest_dist, float(nearest))
        active_span_bonus = max_nearest_dist / 4.0
        return current_span + active_span_bonus

    def local_density(self, event_groups: Sequence[EventGroup], current_time: float) -> int:
        start = current_time - 0.5
        return sum(1 for group in event_groups if start <= group.event_time <= current_time)

    def safe_log1p(self, value: float) -> float:
        return round(math.log1p(max(0.0, value)), 6)


def extract_section_lines(path: Path, difficulty: str) -> List[str]:
    marker_map = {"4": "&inote_4=", "5": "&inote_5=", "6": "&inote_6="}
    marker = marker_map[difficulty]
    lines = path.read_text(encoding="utf-8").splitlines()
    collecting = False
    result: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == marker:
            collecting = True
            continue
        if collecting and stripped == "E":
            break
        if collecting:
            result.append(stripped)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Simai maidata into MVP 22-field event records.")
    parser.add_argument("maidata", type=Path, help="Path to maidata.txt")
    parser.add_argument("--difficulty", default="6", choices=["4", "5", "6"], help="Chart difficulty section")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    section_lines = extract_section_lines(args.maidata, args.difficulty)
    parser_impl = SimaiMVPParser()
    records = parser_impl.parse_chart_section(section_lines)

    text = json.dumps(records, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
