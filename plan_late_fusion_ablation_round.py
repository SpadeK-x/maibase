import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, List

PROBE_PRESETS = {
    "v1": [
        "busy_density_mean",
        "busy_density_p90",
        "outer_move_ge_0_25_ratio",
        "span_jump_p90",
        "slide_conflict_when_busy_ratio",
        "busy_outer_move_p90",
    ],
    "technical_v2_small": [
        "rhythm_switch_ratio",
        "low_density_rhythm_switch_ratio",
        "slide_density_p90",
        "slide_outer_move_p90",
        "slide_span_p90",
    ],
    "hybrid_v2_small": [
        "busy_density_mean",
        "busy_density_p90",
        "outer_move_ge_0_25_ratio",
        "span_jump_p90",
        "slide_conflict_when_busy_ratio",
        "rhythm_switch_ratio",
        "low_density_rhythm_switch_ratio",
        "slide_density_p90",
        "slide_outer_move_p90",
        "slide_span_p90",
    ],
}


def shell_path(value: Path) -> str:
    return value.as_posix()


def build_jobs(args) -> List[Dict[str, object]]:
    base = [
        "--encoded-dir", shell_path(args.encoded_dir),
        "--probe-events-dir", shell_path(args.probe_events_dir),
        "--labels-csv", shell_path(args.labels_csv),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--lr", str(args.lr),
        "--pooling", args.pooling,
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--loss-type", "ce",
    ]

    preset_features = list(PROBE_PRESETS[args.base_preset])
    jobs: List[Dict[str, object]] = []

    full_out_dir = args.output_root / f"{args.base_preset}_full_ce"
    jobs.append(
        {
            "name": f"{args.base_preset}_full_ce",
            "workdir": shell_path(args.workdir),
            "command": [
                args.python_bin,
                "train_mvp_mlp_late_fusion.py",
                *base,
                "--probe-preset", args.base_preset,
                "--save-model", shell_path(full_out_dir / "model.pth"),
                "--predictions-output", shell_path(full_out_dir / "predictions_test.csv"),
                "--misclassified-output", shell_path(full_out_dir / "misclassified_test.csv"),
            ],
            "log_path": shell_path(full_out_dir / "train.log"),
        }
    )

    for feature in preset_features:
        ablated = [candidate for candidate in preset_features if candidate != feature]
        job_name = f"{args.base_preset}_drop_{feature}_ce"
        out_dir = args.output_root / job_name
        command = [
            args.python_bin,
            "train_mvp_mlp_late_fusion.py",
            *base,
            "--probe-features",
            *ablated,
            "--save-model", shell_path(out_dir / "model.pth"),
            "--predictions-output", shell_path(out_dir / "predictions_test.csv"),
            "--misclassified-output", shell_path(out_dir / "misclassified_test.csv"),
        ]
        jobs.append(
            {
                "name": job_name,
                "workdir": shell_path(args.workdir),
                "command": command,
                "log_path": shell_path(out_dir / "train.log"),
            }
        )
    return jobs


def write_plan(jobs: List[Dict[str, object]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "plan.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    return path


def write_shell(jobs: List[Dict[str, object]], output_root: Path) -> Path:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        lines.append(f"cd {shlex.quote(str(job['workdir']))}")
        lines.append(f"echo '==> {job['name']}'")
        command = " ".join(shlex.quote(str(part)) for part in job["command"])
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f"{command} 2>&1 | tee {shlex.quote(shell_path(log_path))}")
        lines.append("")
    script = output_root / "run.sh"
    script.write_text("\n".join(lines), encoding="utf-8")
    return script


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a leave-one-out ablation plan for late-fusion probes.")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--encoded-dir", type=Path, default=Path("encoded_all"))
    parser.add_argument("--probe-events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiment_runs/late_fusion_ablation_round"))
    parser.add_argument("--base-preset", type=str, default="v1", choices=sorted(PROBE_PRESETS.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pooling", type=str, default="mean_max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--python-bin", type=str, default="python")
    args = parser.parse_args()

    jobs = build_jobs(args)
    plan_path = write_plan(jobs, args.output_root)
    script_path = write_shell(jobs, args.output_root)
    print(f"plan_json={plan_path}")
    print(f"shell_script={script_path}")


if __name__ == "__main__":
    main()
