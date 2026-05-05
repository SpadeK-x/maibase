# MVP 21 Fields Generation Rules V2

This is the revised generation-rule document corresponding to:

- [MVP_21_FIELDS_SPEC_V2.md](E:/maibase/MVP_21_FIELDS_SPEC_V2.md)

Relative to the prior MVP rules:

- remove `slide_direction`
- keep all other parsing, grouping, occupancy, and field-generation rules unchanged

## Revision summary

The dominant active slide now contributes only:

- `slide_shape_group`
- `slide_span`

It no longer contributes:

- `slide_direction`

## Field-generation change

When generating slide-related fields:

- if no active slide:
  - `slide_shape_group = "none"`
  - `slide_span = 0`
- else:
  - use the dominant active slide
  - map `slide_shape_group` from the manual shape-group table
  - compute `slide_span` using the revised rule below

## Manual tables now required

Only these manual slide tables are required in V2:

- `SLIDE_SHAPE_GROUP_MAP`
- `SLIDE_SHAPE_SPAN_MAP`

`SLIDE_DIRECTION_MAP` is removed.

## Implementation note

All other rules from the previous generation-rule document remain valid:

- atomic object parsing
- event-group merging
- outer two-slot preservation
- event-level trait union
- hold/slide active-state tracking
- local density calculation

## Revised `slide_span` rule

`slide_span` is defined as:

- `slide_span = max(positional_span, path_span) + shape_span`

Where:

- `positional_span`:
  shortest circular distance between slide start and slide end

- `path_span`:
  path-based span derived from the number of sequential movement segments in the slide body

- `shape_span`:
  manually assigned extra complexity value from `SLIDE_SHAPE_SPAN_MAP`

### Path-span rule

For multi-segment slides such as:

- `4-6-2-8-2-6-4`

`path_span` is the sum of segment spans.

In the MVP implementation, for repeated same-shape chained numeric paths, this is computed as:

- the number of post-start numeric targets in the path

So:

- `4-6` -> `path_span = 1`
- `4-6-2` -> `path_span = 2`
- `4-6-2-8-2-6-4` -> `path_span = 6`

This prevents long return-path slides from collapsing to zero span when start and end positions are the same.
