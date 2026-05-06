# MVP 26 Fields Specification Draft

This draft extends the current MVP 21-field V2 representation with 5 new fields.

The goal is not to replace the existing scheme, but to strengthen decision signals around:

- `13+ <-> 14` boundary
- irregular rhythm
- compact burst patterns
- slide-time conflict load
- awkward span pressure
- locally rare pattern structure

The base 21 fields remain unchanged.

## Added 5 fields

| Field | Meaning | Type | Representation |
|---|---|---|---|
| `rhythm_irregularity_local` | Local irregularity of current inter-event timing relative to recent context | continuous | normalized float |
| `burst_compactness` | Compactness of the most recent 5-event burst including current event | continuous | normalized float |
| `slide_conflict_load` | Extra input load introduced while a slide is already active | continuous | normalized float |
| `hand_span_pressure` | Spatial stretch / awkwardness proxy from current layout and active occupancy | continuous | normalized float |
| `pattern_novelty_local` | Local 3-event pattern rarity estimated from training-set frequency | continuous | normalized float |

## Full field list

The full representation becomes:

1. `delta_time`
2. `event_type`
3. `event_trait`
4. `outer_active`
5. `outer_idx`
6. `outer_pos_sin`
7. `outer_pos_cos`
8. `inner_mask`
9. `inner_count`
10. `cross_zone_flag`
11. `outer_move_dist`
12. `inner_add_count`
13. `inner_remove_count`
14. `hold_active`
15. `hold_remaining_time`
16. `slide_active`
17. `slide_remaining_time`
18. `slide_shape_group`
19. `slide_span`
20. `slide_conflict_flag`
21. `local_density_500ms`
22. `rhythm_irregularity_local`
23. `burst_compactness`
24. `slide_conflict_load`
25. `hand_span_pressure`
26. `pattern_novelty_local`

## Design intent of the new fields

### `rhythm_irregularity_local`

This field is intended to model:

- non-regular local rhythm
- unexpected spacing changes
- reading difficulty not explained by density alone

It is a local rhythm proxy, not a full music-theory beat alignment feature.

### `burst_compactness`

This field is intended to distinguish:

- evenly dense event streams
- short compact bursts packed into a narrower time span

The approved version for this draft is the simple recent-event-span version, not a fixed-width time-window count.

### `slide_conflict_load`

This field refines the current binary `slide_conflict_flag`.

It is intended to capture:

- how much extra input is introduced during active slide occupancy
- whether slide-time interference is light or heavy

### `hand_span_pressure`

This field is a proxy for:

- awkward layout stretch
- wide simultaneous positions
- active-occupancy plus new-input spatial pressure

It does not assume a full left-hand / right-hand reconstruction.

### `pattern_novelty_local`

This field is intended to capture:

- uncommon local chart configurations
- local pattern rarity beyond raw density and span

Because it depends on training-set frequency statistics, it must be built with strict train-only fitting.
