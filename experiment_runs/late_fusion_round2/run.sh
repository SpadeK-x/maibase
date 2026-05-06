#!/usr/bin/env bash
set -euo pipefail

cd .
echo '==> late_fusion_v1_ce'
python train_mvp_mlp_late_fusion.py --encoded-dir encoded_all --probe-events-dir events_all --labels-csv labels.csv --epochs 100 --batch-size 256 --eval-batch-size 32 --lr 0.0005 --pooling mean_max --seed 42 --num-workers 2 --probe-preset v1 --loss-type ce --save-model experiment_runs/late_fusion_round2/late_fusion_v1_ce/model.pth --predictions-output experiment_runs/late_fusion_round2/late_fusion_v1_ce/predictions_test.csv --misclassified-output experiment_runs/late_fusion_round2/late_fusion_v1_ce/misclassified_test.csv 2>&1 | tee experiment_runs/late_fusion_round2/late_fusion_v1_ce/train.log

cd .
echo '==> late_fusion_v2_ce'
python train_mvp_mlp_late_fusion.py --encoded-dir encoded_all --probe-events-dir events_all --labels-csv labels.csv --epochs 100 --batch-size 256 --eval-batch-size 32 --lr 0.0005 --pooling mean_max --seed 42 --num-workers 2 --probe-preset v2 --loss-type ce --save-model experiment_runs/late_fusion_round2/late_fusion_v2_ce/model.pth --predictions-output experiment_runs/late_fusion_round2/late_fusion_v2_ce/predictions_test.csv --misclassified-output experiment_runs/late_fusion_round2/late_fusion_v2_ce/misclassified_test.csv 2>&1 | tee experiment_runs/late_fusion_round2/late_fusion_v2_ce/train.log

cd .
echo '==> late_fusion_v1_focal'
python train_mvp_mlp_late_fusion.py --encoded-dir encoded_all --probe-events-dir events_all --labels-csv labels.csv --epochs 100 --batch-size 256 --eval-batch-size 32 --lr 0.0005 --pooling mean_max --seed 42 --num-workers 2 --probe-preset v1 --loss-type focal --save-model experiment_runs/late_fusion_round2/late_fusion_v1_focal/model.pth --predictions-output experiment_runs/late_fusion_round2/late_fusion_v1_focal/predictions_test.csv --misclassified-output experiment_runs/late_fusion_round2/late_fusion_v1_focal/misclassified_test.csv --focal-gamma 2.0 2>&1 | tee experiment_runs/late_fusion_round2/late_fusion_v1_focal/train.log

cd .
echo '==> late_fusion_v2_focal'
python train_mvp_mlp_late_fusion.py --encoded-dir encoded_all --probe-events-dir events_all --labels-csv labels.csv --epochs 100 --batch-size 256 --eval-batch-size 32 --lr 0.0005 --pooling mean_max --seed 42 --num-workers 2 --probe-preset v2 --loss-type focal --save-model experiment_runs/late_fusion_round2/late_fusion_v2_focal/model.pth --predictions-output experiment_runs/late_fusion_round2/late_fusion_v2_focal/predictions_test.csv --misclassified-output experiment_runs/late_fusion_round2/late_fusion_v2_focal/misclassified_test.csv --focal-gamma 2.0 2>&1 | tee experiment_runs/late_fusion_round2/late_fusion_v2_focal/train.log
