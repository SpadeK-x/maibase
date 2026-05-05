# MVP 22 Fields Generation Rules

This document defines how raw Simai chart syntax should be mapped into the fixed 22-field MVP representation.

This file does not modify the field set itself.
It only defines the generation process.

Related reference:

- [MVP_22_FIELDS_SPEC.md](E:/maibase/MVP_22_FIELDS_SPEC.md)

## 1. Goal

The target is not to preserve raw Simai syntax as-is.

The target is:

- parse raw chart syntax
- convert it into time-ordered `event groups`
- maintain persistent hold/slide occupancy state
- generate one 22-field record per event

## 2. Full pipeline

The conversion pipeline is fixed as:

1. Simai line text
2. atomic object parsing
3. absolute time assignment
4. event-group merging
5. active hold / active slide state update
6. 22-field generation

## 3. Input scope

The source chart is read from Simai-style `maidata.txt` sections:

- `&inote_4=`
- `&inote_5=`
- `&inote_6=`

Each section ends at a line containing only:

- `E`

Each non-empty chart line is processed independently, but time is global across the whole chart.

## 4. Time rules

### 4.1 BPM

Current BPM is updated whenever a token begins with a BPM declaration:

- `(171)`
- `(88.8)`

The BPM update takes effect immediately from the point where it appears.

### 4.2 Grid resolution

Each chart line begins with a timing base:

- `{1}`
- `{4}`
- `{8}`
- `{16}`
- `{24}`
- `{32}`
- `{48}`
- `{64}`
- `{96}`
- `{128}`
- `{192}`

This timing base defines the time step of commas in that line.

### 4.3 Comma

`,` means:

- advance current time by one grid unit

Empty tokens between commas still advance time.

### 4.4 Slash

`/` means:

- simultaneous objects at the same timestamp

Objects linked by `/` belong to the same event group.

## 5. Atomic object parsing

Before generating event groups, each non-empty token must be parsed into one or more atomic objects.

Each atomic object must keep at least:

- `object_type`
- `start_time`
- `trait_set`
- `outer_positions`
- `inner_regions`
- `duration`
- `raw_shape`
- `slide_start`
- `slide_end`

## 6. Region parsing

### 6.1 Outer regions

Outer regions are:

- `1`
- `2`
- `3`
- `4`
- `5`
- `6`
- `7`
- `8`

### 6.2 Inner regions

Inner regions are:

- `A1-A8`
- `B1-B8`
- `C1-C2`
- `D1-D8`
- `E1-E8`

### 6.3 Region classification

Every parsed region must be classified as exactly one of:

- `outer`
- `inner`

## 7. Trait parsing

Traits currently used in the MVP:

- `b`
- `x`

Trait `f` may still appear in raw syntax, but it is not elevated to an official field in this MVP.

For every atomic object, collect:

- `has_b`
- `has_x`

Then define:

- `trait_set = {}`
- `trait_set = {b}`
- `trait_set = {x}`
- `trait_set = {b, x}`

## 8. Atomic object types

Atomic objects are parsed into one of:

- `tap`
- `hold`
- `slide`
- `touch`

### 8.1 Tap

An atomic object is `tap` when it is a direct trigger with no hold duration and no slide body.

Examples:

- `7`
- `2b`
- `8x`
- `1/8` -> two simultaneous tap atomic objects

### 8.2 Touch

An atomic object is `touch` when it targets only inner regions and has no hold duration and no slide body.

Examples:

- `A5`
- `C1f`
- `A1/D1/A8`

### 8.3 Hold

An atomic object is `hold` when it contains `h[...]`.

Examples:

- `7h[4:1]`
- `C1h[8:1]`
- `1xh[16:9]`

Each hold object must store:

- start time
- occupied region(s)
- duration
- end time
- trait set

### 8.4 Slide

An atomic object is `slide` when it contains a slide body, such as:

- `1-5[...]`
- `7<3<7[...]`
- `2>6>2[...]`
- `8w4[...]`
- `1V37[...]`
- `4b-1-3[...]`

For a slide object, parse:

- start time
- start position
- end position
- duration
- trait set
- raw slide shape symbol / path descriptor

Important:

- Slides are not expanded into fake intermediate `sm` / `se` tokens in this MVP.
- A slide contributes one start event and one persistent occupancy state.

## 9. Event-group construction

After all atomic objects are parsed with absolute times, merge them by timestamp.

Each unique timestamp becomes one `event group`.

Each event group must collect:

- all new outer positions triggered at this timestamp
- all new inner regions triggered at this timestamp
- all atomic object types starting at this timestamp
- all trait sets appearing at this timestamp
- all new holds starting at this timestamp
- all new slides starting at this timestamp

## 10. Current-event data containers

For each event group, construct these temporary containers:

- `event_outer_set`
- `event_inner_set`
- `event_object_types`
- `event_trait_union`
- `new_holds`
- `new_slides`

### 10.1 `event_outer_set`

Contains all outer positions triggered in the current event.

Rules:

- deduplicate positions
- keep at most 2 outer positions
- if more than 2 are parsed due to abnormal syntax, keep the first 2 after sorting ascending

### 10.2 `event_inner_set`

Contains all inner regions newly triggered in the current event.

Rules:

- deduplicate regions
- all valid inner regions are preserved

### 10.3 `event_object_types`

Contains the set of object types appearing in the current event:

- `tap`
- `hold`
- `slide`
- `touch`

### 10.4 `event_trait_union`

Union of all atomic object traits in the current event.

## 11. Persistent state containers

Maintain two active lists while iterating through event groups in time order:

- `active_holds`
- `active_slides`

Before generating fields for each event:

1. remove expired holds/slides
2. append `new_holds`
3. append `new_slides`
4. evaluate current active occupancy

## 12. Dominant active slide

Some slide fields need a single active slide:

- `slide_shape_group`
- `slide_direction`
- `slide_span`

When multiple slides are active, choose the dominant active slide as:

1. the most recently started active slide
2. if tied, the first parsed slide among them

## 13. Field generation rules

This section defines the actual generation rule for each of the 22 fields.

### 13.1 `delta_time`

- `current_event_time - previous_event_time`
- first event uses `0`
- apply `log1p`

### 13.2 `event_type`

Determine from `event_object_types`.

Rules:

- only `tap` present -> `tap`
- only `touch` present -> `touch`
- only `hold` present -> `hold`
- only `slide` present -> `slide`
- any mixture of two or more categories -> `compound`

Examples:

- `C1h/7` -> `compound`
- `1/8` -> `tap`
- `A5/A6/B4` -> `touch`
- `1h/8h` -> `hold`
- `1-5/7-3` -> `slide`

### 13.3 `event_trait`

Determine from `event_trait_union`.

Mapping:

- `{}` -> `none`
- `{b}` -> `b`
- `{x}` -> `x`
- `{b, x}` -> `bx`

### 13.4 `outer_active`

- `1` if `event_outer_set` is not empty
- `0` otherwise

### 13.5 `outer_idx`

Generate from `event_outer_set`.

Rules:

- no outer position -> `[0, 0]`
- one outer position `k` -> `[k, 0]`
- two outer positions `a, b` -> `[min(a,b), max(a,b)]`

### 13.6 `outer_pos_sin`

For each slot in `outer_idx`:

- `0 -> 0`
- `k -> sin(2pi * k / 8)`

Store as:

- `[sin1, sin2]`

### 13.7 `outer_pos_cos`

For each slot in `outer_idx`:

- `0 -> 0`
- `k -> cos(2pi * k / 8)`

Store as:

- `[cos1, cos2]`

### 13.8 `inner_mask`

Generate from `event_inner_set`.

Use fixed 34-dim index order:

`[A1,A2,A3,A4,A5,A6,A7,A8,B1,B2,B3,B4,B5,B6,B7,B8,C1,C2,D1,D2,D3,D4,D5,D6,D7,D8,E1,E2,E3,E4,E5,E6,E7,E8]`

Important:

- `inner_mask` contains only newly triggered inner inputs in this event
- persistent inner hold occupancy is not stored here

### 13.9 `inner_count`

- number of active bits in `inner_mask`

### 13.10 `cross_zone_flag`

- `1` if `event_outer_set` is not empty and `event_inner_set` is not empty
- `0` otherwise

### 13.11 `outer_move_dist`

Let:

- `prev_outer_idx = previous event outer_idx`
- `curr_outer_idx = current event outer_idx`

Interpret each as a set of one or two valid outer positions.

Rules:

- if either side has no valid outer position -> `0`
- if both sides have one position -> direct shortest circular distance
- if one side has one position and the other has two -> distance to nearer target
- if both sides have two positions -> evaluate both pairings and choose smaller total cost

Distance basis:

- shortest circular distance on the 8-position ring

Normalization:

- divide by `8`

### 13.12 `inner_add_count`

Compare current `inner_mask` against previous `inner_mask`.

- count bits that are `1` in current and `0` in previous

### 13.13 `inner_remove_count`

Compare current `inner_mask` against previous `inner_mask`.

- count bits that are `1` in previous and `0` in current

### 13.14 `hold_active`

- `1` if any hold in `active_holds` is still active at current event time
- `0` otherwise

### 13.15 `hold_remaining_time`

- if no hold is active -> `0`
- else take the maximum remaining time among active holds
- apply `log1p`

### 13.16 `slide_active`

- `1` if any slide in `active_slides` is still active at current event time
- `0` otherwise

### 13.17 `slide_remaining_time`

- if no slide is active -> `0`
- else take the maximum remaining time among active slides
- apply `log1p`

### 13.18 `slide_shape_group`

- if no active slide -> `none`
- else use the dominant active slide
- map its raw shape using the manual shape-group table

### 13.19 `slide_direction`

- if no active slide -> `none`
- else use the dominant active slide
- map its raw shape using the manual direction table

### 13.20 `slide_span`

- if no active slide -> `0`
- else use the dominant active slide

Definition:

- `slide_span = positional_span + shape_span`

Where:

- `positional_span = shortest circular distance from slide start to slide end`
- `shape_span = manually assigned complexity value for that raw shape`

Final value should be normalized consistently.

### 13.21 `slide_conflict_flag`

Set to `1` when:

- `slide_active = 1`
- and the current event introduces additional new input burden

Additional burden includes:

- any new outer tap
- any new inner touch
- any new hold start
- any new slide start

Otherwise:

- `0`

### 13.22 `local_density_500ms`

- count all events whose timestamps fall in `[current_time - 0.5, current_time]`
- include current event
- apply `log1p`

## 14. Duration parsing

Whenever a duration expression appears:

- `[4:1]`
- `[8:3]`
- `[16:9]`
- `[0.6757##4.2255]`

the parser must convert it into an absolute duration in chart time.

For standard ratio forms:

- use current BPM and current Simai duration semantics

For nonstandard duration syntax:

- preserve the exact parser logic needed to obtain a stable numeric duration
- all downstream fields consume only the final numeric duration

## 15. Inner hold handling

Inner holds such as:

- `C1h[8:1]`

must be treated as:

1. current-event inner trigger:
   - `C1` enters `inner_mask`
2. persistent hold occupancy:
   - contributes to `hold_active`
   - contributes to `hold_remaining_time`

After the starting event:

- future events should not keep `C1` in `inner_mask` unless `C1` is triggered again

## 16. Slide handling

Slides must never be expanded into note fragments for this MVP.

For a slide:

1. create one slide atomic object at start time
2. store its start/end/duration/shape/traits
3. insert it into `active_slides`
4. let later events observe its persistent occupancy through slide-state fields

## 17. Trait handling note

Traits affect scoring semantics and should be preserved at event level through `event_trait`.

This MVP does not add separate per-object trait fields.

Instead:

- all object-level `b/x` information is compressed into event-level trait union

## 18. Known MVP limitations

These are accepted limitations of the current MVP:

1. No explicit left-hand / right-hand assignment
2. Hold occupancy is collapsed to a binary active flag plus max remaining time
3. Slide occupancy is collapsed to a binary active flag plus dominant active slide
4. Inner action geometry is represented as a set, not as a detailed hand-shape model

## 19. Recommended implementation order

Implement in this order:

1. time parser
2. region parser
3. trait parser
4. atomic object parser
5. event-group merger
6. active hold / active slide tracker
7. field generator
8. normalization layer

## 20. Validation strategy

Before model training, validate the generator using:

1. pure outer tap sequences
2. inner-only touch sequences
3. hold + tap mixtures
4. slide + tap mixtures
5. mixed inner/outer events
6. BPM-changing sections
7. dense charts with simultaneous outer pairs

