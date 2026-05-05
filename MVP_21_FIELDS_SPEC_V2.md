# MVP 21 Fields Specification V2

This is the revised MVP field set.

Revision from the previous 22-field version:

- remove `slide_direction`
- refine `slide_span` to include multi-segment path length

Reason:

- slide direction depends on both shape and start position
- direction is less important for classification than shape, timing, occupancy, and span
- removing it reduces ambiguity and keeps the representation cleaner
- complex multi-segment slides cannot be represented well by start/end span alone

## Official 21 fields

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
| `slide_span` | Span of dominant active slide | continuous | normalized float |
| `slide_conflict_flag` | Whether current event introduces extra load during active slide | binary | `0/1` |
| `local_density_500ms` | Number of events in the last 500ms window | continuous | `log1p(float)` |

## Fixed note

All generation logic remains the same as the previous MVP version except:

- `slide_direction` is removed
- any direction-related manual mapping table is no longer required
- `slide_span` now explicitly supports path-based span accumulation for multi-segment slides
