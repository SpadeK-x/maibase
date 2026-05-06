import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_command_plan(args) -> List[Dict[str, object]]:
    output_root = args.output_root
    baseline_dir = output_root / "baseline_ce"
    regression_dir = output_root / "regression"
    multitask_dir = output_root / "multitask"

    common = [
        "--encoded-dir", str(args.encoded_dir),
        "--labels-csv", str(args.labels_csv),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
    ]

    train_common = [
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--pooling", args.pooling,
    ]

    baseline_model = baseline_dir / "model.pth"
    regression_model = regression_dir / "model.pth"
    multitask_model = multitask_dir / "model.pth"

    return [
        {
            "name": "baseline_ce_train",
            "workdir": str(args.workdir),
            "command": [
                sys.executable,
                "train_mvp_mlp.py",
                *common,
                *train_common,
                "--save-model", str(baseline_model),
                "--misclassified-output", str(baseline_dir / "misclassified_test.csv"),
            ],
            "log_path": str(baseline_dir / "train.log"),
        },
        {
            "name": "baseline_ce_eval",
            "workdir": str(args.workdir),
            "command": [
                sys.executable,
                "evaluate_saved_mlp.py",
                *common,
                "--eval-batch-size", str(args.eval_batch_size),
                "--pooling", args.pooling,
                "--model-path", str(baseline_model),
                "--predictions-output", str(baseline_dir / "predictions_test.csv"),
                "--misclassified-output", str(baseline_dir / "misclassified_test.csv"),
                "--pair-output-dir", str(baseline_dir / "misclassified_pairs"),
            ],
            "log_path": str(baseline_dir / "eval.log"),
        },
        {
            "name": "regression_train",
            "workdir": str(args.workdir),
            "command": [
                sys.executable,
                "train_mvp_regression.py",
                *common,
                *train_common,
                "--save-model", str(regression_model),
                "--predictions-output", str(regression_dir / "regression_predictions_test.csv"),
            ],
            "log_path": str(regression_dir / "train.log"),
        },
        {
            "name": "regression_threshold_search",
            "workdir": str(args.workdir),
            "command": [
                sys.executable,
                "search_regression_thresholds.py",
                *common,
                "--eval-batch-size", str(args.eval_batch_size),
                "--pooling", args.pooling,
                "--model-path", str(regression_model),
                "--threshold-step", str(args.threshold_step),
                "--export-test-csv", str(regression_dir / "regression_thresholded_test.csv"),
            ],
            "log_path": str(regression_dir / "threshold_search.log"),
        },
        {
            "name": "multitask_train",
            "workdir": str(args.workdir),
            "command": [
                sys.executable,
                "train_mvp_regression_multitask.py",
                *common,
                *train_common,
                "--cls-loss-weight", str(args.cls_loss_weight),
                "--save-model", str(multitask_model),
                "--predictions-output", str(multitask_dir / "multitask_predictions_test.csv"),
            ],
            "log_path": str(multitask_dir / "train.log"),
        },
    ]


def write_plan(plan: List[Dict[str, object]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "experiment_plan.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return plan_path


def write_shell_script(plan: List[Dict[str, object]], output_root: Path) -> Path:
    script_path = output_root / "run_experiments.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in plan:
        log_path = Path(str(item["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        workdir = shlex.quote(str(item["workdir"]))
        command = " ".join(shlex.quote(str(part)) for part in item["command"])
        log = shlex.quote(str(log_path))
        lines.append(f"cd {workdir}")
        lines.append(f"echo '==> {item['name']}'")
        lines.append(f"{command} 2>&1 | tee {log}")
        lines.append("")
    script_path.write_text("\n".join(lines), encoding="utf-8")
    return script_path


def run_plan(plan: List[Dict[str, object]]) -> None:
    for item in plan:
        log_path = Path(str(item["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"==> {item['name']}")
        print(" ".join(shlex.quote(str(part)) for part in item["command"]))
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                item["command"],
                cwd=str(item["workdir"]),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(f"{item['name']} failed with exit code {process.returncode}. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the target-design comparison experiments.")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--encoded-dir", type=Path, default=Path("encoded_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiment_runs/target_design_v1"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pooling", type=str, default="mean_max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true", help="Run the plan immediately instead of only writing plan files.")
    args = parser.parse_args()

    plan = build_command_plan(args)
    plan_path = write_plan(plan, args.output_root)
    shell_path = write_shell_script(plan, args.output_root)
    print(f"plan_json={plan_path}")
    print(f"shell_script={shell_path}")

    if args.execute:
        run_plan(plan)
        print(f"completed_output_root={args.output_root}")


if __name__ == "__main__":
    main()
