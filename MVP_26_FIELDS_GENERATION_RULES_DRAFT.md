# MVP 26 Fields Generation Rules Draft

This draft extends:

- [MVP_21_FIELDS_SPEC_V2.md](E:/maibase/MVP_21_FIELDS_SPEC_V2.md)
- [MVP_21_FIELDS_GENERATION_RULES_V2.md](E:/maibase/MVP_21_FIELDS_GENERATION_RULES_V2.md)

It adds 5 new fields while preserving all existing V2 parsing and event-group logic.

## Scope

Unchanged from V2:

- atomic object parsing
- event-group merging
- outer two-slot preservation
- event-level trait union
- hold/slide active-state tracking
- local density calculation
- slide span definition

New in this draft:

- `rhythm_irregularity_local`
- `burst_compactness`
- `slide_conflict_load`
- `hand_span_pressure`
- `pattern_novelty_local`

## 1. `rhythm_irregularity_local`

### Intent

Measure how unusual the current inter-event spacing is relative to recent local timing.

### Inputs

- current raw inter-event interval `dt_t`
- recent non-zero raw inter-event intervals from the previous events

### Rule

For event `t`:

1. Let `dt_t` be:
   - `0` for the first event
   - otherwise `event_time[t] - event_time[t-1]`

2. Collect up to the previous 4 non-zero inter-event intervals:
   - `recent = [dt_(t-1), dt_(t-2), ...]`
   - skip zero values

3. If `recent` is empty:
   - `rhythm_irregularity_local = 0`

4. Else:
   - let `med = median(recent)`
   - let `eps = 1e-6`
   - compute:

   `rhythm_irregularity_local = abs(log(dt_t + eps) - log(med + eps))`

### Notes

- This uses raw time intervals, not `log1p(delta_time)`.
- This is a local rhythm irregularity proxy, not BPM-grid reconstruction.

## 2. `burst_compactness`

### Intent

Measure whether the current event belongs to a compact short burst.

### Approved version

Use the simple recent-event-span version.

### Rule

For event `t`:

1. Take the current event and up to the previous 4 events.
2. Let this local slice contain `m` events, where `1 <= m <= 5`.
3. Let:
   - `start_time = event_time` of the earliest event in the slice
   - `end_time = event_time` of the current event
   - `span = end_time - start_time`

4. If `m <= 1`:
   - `burst_compactness = 0`

5. Else:
   - let `eps = 1e-6`
   - compute:

   `burst_compactness = 1 / (span + eps)`

### Notes

- Larger value means more compact burst packing.
- This intentionally complements `local_density_500ms`.

## 3. `slide_conflict_load`

### Intent

Refine slide-time interference from binary presence into load intensity.

### Inputs

- whether there are active slides before the current event's new slides are added
- current event's newly introduced input objects

### Rule

Let `active_slides_before_new` be the active slides already occupying the current timestamp before adding any new slide starting now.

If `active_slides_before_new` is empty:

- `slide_conflict_load = 0`

Else:

1. Define:
   - `new_outer_count` = number of outer positions in the current event group, capped by the existing two-slot outer representation
   - `new_inner_count` = number of inner regions newly triggered in the current event group
   - `new_hold_count` = number of new hold objects in the current event group
   - `new_slide_count` = number of new slide objects in the current event group

2. Compute:

`slide_conflict_load = new_outer_count + 0.5 * new_inner_count + 0.5 * new_hold_count + 1.0 * new_slide_count`

### Notes

- This field is intended to coexist with `slide_conflict_flag`, not replace it immediately.
- It models conflict strength, not just conflict existence.

## 4. `hand_span_pressure`

### Intent

Approximate awkward stretch / spatial pressure without explicit hand assignment.

### Rule

For the current event:

1. Compute `current_span`:
   - if the current event has two outer positions:
     - let their shortest circular distance be `d`
     - `current_span = d / 4.0`
   - else:
     - `current_span = 0`

2. Compute `active_span_bonus`:
   - gather active occupancy anchors from currently active holds and slides
   - allowed anchors may include:
     - active hold outer positions
     - active slide start positions
     - active slide end positions when available
   - gather current event outer positions
   - if either side is empty:
     - `active_span_bonus = 0`
   - else:
     - compute the shortest circular distance from each current outer position to the nearest active anchor
     - take the maximum of these distances
     - normalize by `4.0`

3. Compute:

`hand_span_pressure = current_span + active_span_bonus`

### Notes

- This is a spatial awkwardness proxy.
- It is intentionally conservative and parser-friendly.
- It does not attempt full ergonomic simulation.

## 5. `pattern_novelty_local`

### Intent

Capture local configuration rarity using training-set frequency.

### Fitting rule

This field must be fit on training data only.

No eval/test sample may contribute to pattern frequency statistics used for its own encoding.

### Event token definition

For each event, construct a lightweight discrete token from:

- `event_type`
- `outer_count`
- `inner_count_bucket`
- `slide_active`
- `slide_shape_group`
- `slide_conflict_flag`

Suggested buckets:

- `outer_count` in `{0, 1, 2}`
- `inner_count_bucket` in `{0, 1, 2_plus}`

### Local pattern rule

1. Build event tokens in sequence.
2. For event `t`, form a local 3-event pattern using:
   - event `t-2`
   - event `t-1`
   - event `t`

3. If fewer than 3 events are available:
   - pad with a fixed `BOS` token on the left

4. Count pattern frequency over the training split:
   - `freq(pattern)`

5. Let:
   - `total_patterns` = total number of fitted 3-event patterns in the training split

6. Compute:

`pattern_novelty_local = -log((freq(pattern) + 1) / (total_patterns + 1))`

### Notes

- Higher value means the local pattern is rarer in the training distribution.
- This is a pattern rarity proxy, not a learned embedding by itself.

## Normalization recommendation

These 5 new fields should be treated as numeric continuous features.

Recommended handling:

- fit mean/std on the training split only
- standardize them together with the existing standardized numeric block

## Implementation staging recommendation

To keep the code change conservative:

1. First introduce the new fields in event JSON generation.
2. Then extend the encoder numeric block.
3. Then run an ablation:
   - baseline 21-field V2
   - 21 + `rhythm_irregularity_local`
   - 21 + 3 selected fields
   - full 26-field draft

Priority order for first implementation:

1. `rhythm_irregularity_local`
2. `slide_conflict_load`
3. `pattern_novelty_local`
4. `burst_compactness`
5. `hand_span_pressure`
