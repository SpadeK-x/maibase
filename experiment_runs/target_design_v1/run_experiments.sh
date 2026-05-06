#!/usr/bin/env bash
set -euo pipefail

cd .
echo '==> baseline_ce_train'
/root/miniconda3/bin/python train_mvp_mlp.py --encoded-dir encoded_all --labels-csv labels.csv --seed 42 --num-workers 2 --batch-size 256 --eval-batch-size 32 --epochs 100 --lr 0.0005 --pooling mean_max --save-model experiment_runs/target_design_v1/baseline_ce/model.pth --misclassified-output experiment_runs/target_design_v1/baseline_ce/misclassified_test.csv 2>&1 | tee experiment_runs/target_design_v1/baseline_ce/train.log

cd .
echo '==> baseline_ce_eval'
/root/miniconda3/bin/python evaluate_saved_mlp.py --encoded-dir encoded_all --labels-csv labels.csv --seed 42 --num-workers 2 --eval-batch-size 32 --pooling mean_max --model-path experiment_runs/target_design_v1/baseline_ce/model.pth --predictions-output experiment_runs/target_design_v1/baseline_ce/predictions_test.csv --misclassified-output experiment_runs/target_design_v1/baseline_ce/misclassified_test.csv --pair-output-dir experiment_runs/target_design_v1/baseline_ce/misclassified_pairs 2>&1 | tee experiment_runs/target_design_v1/baseline_ce/eval.log

cd .
echo '==> regression_train'
/root/miniconda3/bin/python train_mvp_regression.py --encoded-dir encoded_all --labels-csv labels.csv --seed 42 --num-workers 2 --batch-size 256 --eval-batch-size 32 --epochs 100 --lr 0.0005 --pooling mean_max --save-model experiment_runs/target_design_v1/regression/model.pth --predictions-output experiment_runs/target_design_v1/regression/regression_predictions_test.csv 2>&1 | tee experiment_runs/target_design_v1/regression/train.log

cd .
echo '==> regression_threshold_search'
/root/miniconda3/bin/python search_regression_thresholds.py --encoded-dir encoded_all --labels-csv labels.csv --seed 42 --num-workers 2 --eval-batch-size 32 --pooling mean_max --model-path experiment_runs/target_design_v1/regression/model.pth --threshold-step 0.02 --export-test-csv experiment_runs/target_design_v1/regression/regression_thresholded_test.csv 2>&1 | tee experiment_runs/target_design_v1/regression/threshold_search.log

cd .
echo '==> multitask_train'
/root/miniconda3/bin/python train_mvp_regression_multitask.py --encoded-dir encoded_all --labels-csv labels.csv --seed 42 --num-workers 2 --batch-size 256 --eval-batch-size 32 --epochs 100 --lr 0.0005 --pooling mean_max --cls-loss-weight 0.5 --save-model experiment_runs/target_design_v1/multitask/model.pth --predictions-output experiment_runs/target_design_v1/multitask/multitask_predictions_test.csv 2>&1 | tee experiment_runs/target_design_v1/multitask/train.log
