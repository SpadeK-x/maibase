# MVP 22 Fields Specification

This document fixes the official MVP field set for chart-to-action-state encoding.

The representation target is:

- `event group` as the basic unit
- player action state instead of raw chart object sequence
- support for tap / hold / slide / touch / compound

This field set is fixed as the basis for subsequent development.

## Core assumptions

1. One `event group` contains all objects at the same timestamp.
2. Outer ring may contain up to 2 simultaneous positions.
3. Inner screen may contain multiple simultaneous regions.
4. `inner_mask` describes newly triggered inner inputs in the current event only.
5. Hold and slide occupancy are tracked separately as persistent state.

## Official 22 fields

| Field | Meaning | Type | Representation |
|---|---|---|---|
| `delta_time` | Time difference from previous event | continuous | `log1p(float)` |
| `event_type` | Main action type of current event | categorical | embedding |
| `event_trait` | Union trait of current event | categorical | embedding |
| `outer_active` | Whether current event contains outer-ring input | binary | `0/1` |
| `outer_idx` | Up to 2 outer-ring positions in current event | structured categorical | `[idx1, idx2]`, each in `0..8` |
| `outer_pos_sin` | Sine encoding for outer positions | structured continuous | `[sin1, sin2]` |
| `outer_pos_cos` | Cosine encoding for outer positions | structured continuous | `[cos1, cos2]` |
| `inner_mask` | Inner-screen region set triggered in current event | set | 34-dim multi-hot |
| `inner_count` | Number of triggered inner regions in current event | integer/continuous | scalar |
| `cross_zone_flag` | Whether current event contains both outer and inner input | binary | `0/1` |
| `outer_move_dist` | Minimum matching movement cost between previous and current outer sets | continuous | normalized float |
| `inner_add_count` | Number of newly added inner regions vs previous event | integer/continuous | scalar |
| `inner_remove_count` | Number of removed inner regions vs previous event | integer/continuous | scalar |
| `hold_active` | Whether any hold is active at current event time | binary | `0/1` |
| `hold_remaining_time` | Maximum remaining time among active holds | continuous | `log1p(float)` |
| `slide_active` | Whether any slide is active at current event time | binary | `0/1` |
| `slide_remaining_time` | Maximum remaining time among active slides | continuous | `log1p(float)` |
| `slide_shape_group` | Shape group of dominant active slide | categorical | embedding |
| `slide_direction` | Direction label of dominant active slide | categorical | embedding |
| `slide_span` | Span of dominant active slide | continuous | normalized float |
| `slide_conflict_flag` | Whether current event introduces extra load during active slide | binary | `0/1` |
| `local_density_500ms` | Number of events in the last 500ms window | continuous | `log1p(float)` |

## Field details

### 1. `delta_time`

- Definition: `current_event_time - previous_event_time`
- First event uses `0`
- Stored as `log1p(value)`

### 2. `event_type`

Fixed enum:

- `tap`
- `hold`
- `slide`
- `touch`
- `compound`

Interpretation:

- `tap`: only tap-like outer/inner triggers
- `hold`: current event is mainly hold start, with no cross-type mixture
- `slide`: current event is mainly slide start, with no cross-type mixture
- `touch`: inner-only touch event
- `compound`: mixed action types in one event

### 3. `event_trait`

Fixed enum:

- `none`
- `b`
- `x`
- `bx`

Definition:

- Collect all traits appearing in all objects of the event
- Use their union as the event-level trait

Examples:

- `1/5` -> `none`
- `1b/5` -> `b`
- `2x/A5` -> `x`
- `1b/8x` -> `bx`

### 4. `outer_active`

- `1` if current event contains any outer-ring input
- `0` otherwise

### 5. `outer_idx`

Definition:

- `outer_idx = [idx1, idx2]`
- each slot takes values in `0..8`
- `0` means empty slot

Rules:

- If no outer input: `[0, 0]`
- If one outer position: `[k, 0]`
- If two outer positions: sort ascending, e.g. `[4, 5]`, `[1, 8]`

This field preserves up to 2 simultaneous outer positions without collapsing to a single main index.

### 6. `outer_pos_sin`, `outer_pos_cos`

Derived from `outer_idx`.

Rules:

- For each nonzero outer index `k`, compute circular encoding:
  - `sin(2pi * k / 8)`
  - `cos(2pi * k / 8)`
- Empty slot `0` maps to `0`

Stored as:

- `outer_pos_sin = [sin1, sin2]`
- `outer_pos_cos = [cos1, cos2]`

### 7. `inner_mask`

Fixed 34-dim order:

`[A1,A2,A3,A4,A5,A6,A7,A8,B1,B2,B3,B4,B5,B6,B7,B8,C1,C2,D1,D2,D3,D4,D5,D6,D7,D8,E1,E2,E3,E4,E5,E6,E7,E8]`

Meaning:

- describes newly triggered inner-screen inputs in the current event only
- does not store persistent hold/slide occupancy residue

### 8. `inner_count`

- Number of `1`s in `inner_mask`

### 9. `cross_zone_flag`

- `1` if the current event contains both outer input and inner input
- `0` otherwise

### 10. `outer_move_dist`

Definition:

- Compare current outer set and previous outer set
- Compute minimum total circular matching distance
- Normalize to `[0, 1]`

Distance basis:

- circular shortest distance on 8-key ring
- e.g. `1->2 = 1`, `1->8 = 1`, `1->5 = 4`

Matching rules:

- If both events have one outer key, use their direct distance
- If one event has one key and the other has two, match the single key to the nearer one
- If both have two keys, evaluate both pairings and choose the smaller total cost
- If either event has no outer input, use `0`

Recommended normalization:

- divide by `8`

### 11. `inner_add_count`

- Number of active bits in `current_inner_mask AND NOT previous_inner_mask`

### 12. `inner_remove_count`

- Number of active bits in `previous_inner_mask AND NOT current_inner_mask`

### 13. `hold_active`

- `1` if at current event time any hold is still active
- `0` otherwise

### 14. `hold_remaining_time`

- Maximum remaining duration among all currently active holds
- Use `0` if no hold is active
- Store as `log1p(value)`

### 15. `slide_active`

- `1` if at current event time any slide is still active
- `0` otherwise

### 16. `slide_remaining_time`

- Maximum remaining duration among all currently active slides
- Use `0` if no slide is active
- Store as `log1p(value)`

### 17. `slide_shape_group`

- Shape-group label of the dominant active slide
- Mapping table will be manually defined

### 18. `slide_direction`

- Direction label of the dominant active slide
- Mapping table will be manually defined

### 19. `slide_span`

Definition:

- `slide_span = positional_span + shape_span`

Where:

- `positional_span`: shortest circular distance between slide start and slide end
- `shape_span`: manually assigned extra complexity for the specific slide shape

Notes:

- `shape_span` mapping table will be manually defined
- final numeric value should be normalized consistently

### 20. `slide_conflict_flag`

- `1` when `slide_active = 1` and current event introduces extra new input burden
- otherwise `0`

Extra burden includes:

- new outer tap
- new inner tap
- new hold start
- new slide start

### 21. `local_density_500ms`

- Count events in `[event_time - 0.5, event_time]`
- include current event
- store as `log1p(value)`

## Event grouping rules

1. `,` advances time by one grid unit under the current `{n}` resolution.
2. `/` means simultaneous objects inside the same event.
3. All objects with the same absolute timestamp are merged into one `event group`.

## Occupancy rules

### Hold occupancy

- Hold start contributes to current event input
- Hold then remains active until its end time
- Persistent hold state is expressed only through:
  - `hold_active`
  - `hold_remaining_time`

### Slide occupancy

- Slide start contributes to current event input
- Slide is not expanded into multiple fake intermediate note tokens
- Persistent slide state is expressed only through:
  - `slide_active`
  - `slide_remaining_time`
  - `slide_shape_group`
  - `slide_direction`
  - `slide_span`

## Important modeling note

This specification is still an MVP design.

It is stronger than the original single-note five-tuple representation, but it is not yet a full hand-assignment model.

The main improvements fixed in this version are:

1. event-group representation instead of isolated note fragments
2. persistent hold/slide occupancy state
3. two-slot outer-ring preservation
4. explicit event-level trait field

