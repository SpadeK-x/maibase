import math
from collections import Counter
from typing import Dict, List, Sequence


TECHNICAL_PROBE_METRICS = [
    "interval_cv",
    "interval_entropy",
    "rhythm_switch_ratio",
    "short_long_alternation_ratio",
    "low_density_rhythm_switch_ratio",
    "nonburst_rhythm_switch_ratio",
    "slide_density_mean",
    "slide_density_p90",
    "slide_outer_move_p90",
    "slide_compound_ratio",
    "slide_span_p90",
    "long_slide_ratio",
    "slide_span_jump_p90",
    "slide_conflict_active_ratio",
    "slide_tap_interrupt_ratio",
]


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def safe_ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def density_from_log(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(math.expm1(float(value)))
    return 0.0


def delta_from_log(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(math.expm1(float(value)))
    return 0.0


def shannon_entropy(values: Sequence[float], bucket_size: float = 0.02) -> float:
    if not values:
        return 0.0
    counter = Counter(int(round(value / bucket_size)) for value in values if value > 0.0)
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def is_rhythm_switch(prev_dt: float, curr_dt: float, ratio_threshold: float = 1.75) -> bool:
    if prev_dt <= 0.03 or curr_dt <= 0.03:
        return False
    larger = max(prev_dt, curr_dt)
    smaller = min(prev_dt, curr_dt)
    if smaller <= 1e-6:
        return False
    return (larger / smaller) >= ratio_threshold


def compute_technical_probe_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    raw_intervals = [delta_from_log(record.get("delta_time", 0.0)) for record in records]
    positive_intervals = [value for value in raw_intervals if value > 1e-6]
    mean_interval = mean(positive_intervals)
    interval_std = 0.0
    if positive_intervals:
        interval_std = math.sqrt(mean([(value - mean_interval) ** 2 for value in positive_intervals]))
    interval_cv = interval_std / mean_interval if mean_interval > 1e-6 else 0.0

    densities = [density_from_log(record.get("local_density_500ms", 0.0)) for record in records]
    rhythm_switches = 0
    low_density_switches = 0
    nonburst_switches = 0
    comparable_pairs = 0
    low_density_pairs = 0
    nonburst_pairs = 0
    alternations = 0
    alternating_windows = 0

    for i in range(1, len(raw_intervals)):
        prev_dt = raw_intervals[i - 1]
        curr_dt = raw_intervals[i]
        switched = is_rhythm_switch(prev_dt, curr_dt)
        if prev_dt > 0.03 and curr_dt > 0.03:
            comparable_pairs += 1
            if switched:
                rhythm_switches += 1

            pair_density = (densities[i - 1] + densities[i]) / 2.0
            if pair_density <= 4.5:
                low_density_pairs += 1
                if switched:
                    low_density_switches += 1

            if max(prev_dt, curr_dt) >= 0.08 and min(prev_dt, curr_dt) >= 0.04:
                nonburst_pairs += 1
                if switched:
                    nonburst_switches += 1

    for i in range(2, len(raw_intervals)):
        a = raw_intervals[i - 2]
        b = raw_intervals[i - 1]
        c = raw_intervals[i]
        if a <= 0.03 or b <= 0.03 or c <= 0.03:
            continue
        alternating_windows += 1
        ab = max(a, b) / max(min(a, b), 1e-6)
        bc = max(b, c) / max(min(b, c), 1e-6)
        if ab >= 1.5 and bc >= 1.5:
            if (a < b > c) or (a > b < c):
                alternations += 1

    slide_records = [record for record in records if int(record.get("slide_active", 0)) == 1]
    slide_density = [density_from_log(record.get("local_density_500ms", 0.0)) for record in slide_records]
    slide_outer_moves = [float(record.get("outer_move_dist", 0.0)) for record in slide_records]
    slide_spans = [float(record.get("slide_span", 0.0)) for record in slide_records]

    slide_span_jumps: List[float] = []
    prev_span = None
    for record in slide_records:
        span = float(record.get("slide_span", 0.0))
        if prev_span is not None:
            slide_span_jumps.append(abs(span - prev_span))
        prev_span = span

    metrics = {
        "interval_cv": interval_cv,
        "interval_entropy": shannon_entropy(positive_intervals),
        "rhythm_switch_ratio": safe_ratio(rhythm_switches, comparable_pairs),
        "short_long_alternation_ratio": safe_ratio(alternations, alternating_windows),
        "low_density_rhythm_switch_ratio": safe_ratio(low_density_switches, low_density_pairs),
        "nonburst_rhythm_switch_ratio": safe_ratio(nonburst_switches, nonburst_pairs),
        "slide_density_mean": mean(slide_density),
        "slide_density_p90": percentile(slide_density, 0.9),
        "slide_outer_move_p90": percentile(slide_outer_moves, 0.9),
        "slide_compound_ratio": safe_ratio(
            sum(1 for record in slide_records if str(record.get("event_type", "")) == "compound"),
            len(slide_records),
        ),
        "slide_span_p90": percentile(slide_spans, 0.9),
        "long_slide_ratio": safe_ratio(sum(1 for span in slide_spans if span >= 5.0), len(slide_spans)),
        "slide_span_jump_p90": percentile(slide_span_jumps, 0.9),
        "slide_conflict_active_ratio": safe_ratio(
            sum(int(record.get("slide_conflict_flag", 0)) for record in slide_records),
            len(slide_records),
        ),
        "slide_tap_interrupt_ratio": safe_ratio(
            sum(1 for record in slide_records if str(record.get("event_type", "")) == "tap"),
            len(slide_records),
        ),
    }
    return metrics
