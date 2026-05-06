# MVP 23 Fields Specification V1

This version keeps the stable MVP 21-field V2 representation and adds only 2 structure-oriented fields.

Reason for this reduction:

- the larger 26-field experiment improved high-end discrimination
- but it damaged `13+` recall
- ablation results suggest only the two structure-oriented additions are consistently useful

The added fields are:

- `slide_conflict_load`
- `hand_span_pressure`

## Official 23 fields

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
| `slide_conflict_load` | Strength of extra load introduced while a slide is already active | continuous | normalized float |
| `hand_span_pressure` | Spatial stretch / awkwardness proxy from current layout and active occupancy | continuous | normalized float |

## Design note

This representation is intentionally conservative.

It does not include:

- local rhythm irregularity features
- burst compactness features
- train-fitted pattern novelty features

Those were tested experimentally, but are not part of the current mainline because they did not improve the `13+` boundary reliably.
