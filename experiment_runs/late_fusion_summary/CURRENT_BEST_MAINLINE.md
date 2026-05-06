# Current Best Mainline

## Recommendation

Current recommended mainline is `late_fusion_v1_ce`.

- Input representation: formal `21 fields V2 / 84 dims`
- Event encoder: existing MLP event encoder + pooled chart embedding
- Fusion style: late fusion with chart-level structural probe features
- Probe preset: `v1`
- Loss: `ce`

## Recommended Command

```bash
python train_mvp_mlp_late_fusion.py \
  --encoded-dir ./encoded_all \
  --probe-events-dir ./events_all \
  --labels-csv ./labels.csv \
  --epochs 100 \
  --batch-size 256 \
  --eval-batch-size 32 \
  --lr 5e-4 \
  --pooling mean_max \
  --num-workers 2 \
  --probe-preset v1 \
  --loss-type ce \
  --save-model ./experiment_runs/late_fusion_probe/model.pth \
  --predictions-output ./experiment_runs/late_fusion_probe/predictions_test.csv \
  --misclassified-output ./experiment_runs/late_fusion_probe/misclassified_test.csv
```

## Probe Features

`v1` uses:

1. `busy_density_mean`
2. `busy_density_p90`
3. `outer_move_ge_0_25_ratio`
4. `span_jump_p90`
5. `slide_conflict_when_busy_ratio`
6. `busy_outer_move_p90`

## Main Metrics

Exact metrics are recomputed from `predictions_test.csv`.

| method | acc | 13 recall | 13+ recall | 14 recall | 14+ recall | 13+->13 | 13+->14 | 14->13+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_ce | 0.7131 | 0.8929 | 0.4524 | 0.7368 | 0.8000 | 12 | 11 | 1 |
| regression_searched_thresholds | 0.7541 | 0.8929 | 0.6905 | 0.5263 | 0.6000 | 11 | 2 | 8 |
| late_fusion_v1_ce | 0.7623 | 0.9464 | 0.5238 | 0.7895 | 0.6000 | 12 | 8 | 1 |
| late_fusion_v2_focal | 0.7541 | 0.9107 | 0.5952 | 0.6316 | 1.0000 | 10 | 7 | 2 |
| hybrid_v1_ce | 0.7377 | 0.9107 | 0.5238 | 0.7368 | 0.6000 | 10 | 9 | 1 |
| hybrid_v2_small_ce | 0.7213 | 0.9107 | 0.4524 | 0.7368 | 0.8000 | 14 | 8 | 1 |

## Why This Is The Mainline

`late_fusion_v1_ce` is not the best on every single metric, but it is the best balanced point:

- higher exact accuracy than baseline and other late-fusion variants
- clearly better than baseline on `13+ -> 14`
- does not collapse true `14` into `13+` like regression-threshold search
- best `14 recall` among all tested practical candidates

The regression route is useful as an analysis tool, but not stable enough as the mainline classifier because it trades away too much `14` stability.

## What Was Learned

1. Mainline improvement came from structural late-fusion features, not from switching the task objective alone.
2. Current formal 21-field event sequence already models intensity reasonably well.
3. The useful extra signal is mostly `busy-window structural pressure`, not more global density statistics.
4. The chart-level technical probe packages tried so far did not beat `v1`.

## V1 Ablation Result

`v1` works as a bundle. No single feature can be removed for free.

Most important signals from leave-one-out ablation:

- `slide_conflict_when_busy_ratio`
- `busy_density_mean`
- `busy_outer_move_p90`
- `span_jump_p90`

Relatively weakest item:

- `busy_density_p90`

But even that one is not free to remove if the goal is to keep overall `14` stability.

## Practical Guidance

- Use `late_fusion_v1_ce` as the current default training target for this project.
- Keep regression-threshold experiments as analysis tools, not as the formal replacement.
- Do not continue adding broad chart-level probe bundles blindly.
- If work continues, the next research step should move closer to local event-sequence modeling for rhythm / slide technicality instead of more global summary features.
