# MVP 23 Fields Generation Rules V1

This version extends:

- [MVP_21_FIELDS_SPEC_V2.md](E:/maibase/MVP_21_FIELDS_SPEC_V2.md)
- [MVP_21_FIELDS_GENERATION_RULES_V2.md](E:/maibase/MVP_21_FIELDS_GENERATION_RULES_V2.md)

It adds only:

- `slide_conflict_load`
- `hand_span_pressure`

All existing V2 parsing, grouping, occupancy, and slide-span rules remain unchanged.

## 1. `slide_conflict_load`

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

- This field is intended to coexist with `slide_conflict_flag`.
- It models conflict strength, not just conflict existence.

## 2. `hand_span_pressure`

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
- It does not attempt full left-hand / right-hand reconstruction.

## Normalization recommendation

These 2 new fields should be treated as numeric continuous features.

Recommended handling:

- fit mean/std on the training split only
- standardize them together with the existing standardized numeric block
