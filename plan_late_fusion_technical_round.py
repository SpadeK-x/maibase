import argparse
import json
import shlex
from pathlib import Path
from typing import Dict, List


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
    ]

    specs = [
        ("technical_v1_ce", "technical_v1", "ce"),
        ("technical_v1_focal", "technical_v1", "focal"),
        ("hybrid_v1_ce", "hybrid_v1", "ce"),
        ("hybrid_v1_focal", "hybrid_v1", "focal"),
    ]

    jobs: List[Dict[str, object]] = []
    for name, preset, loss_type in specs:
        out_dir = args.output_root / name
        command = [
            args.python_bin,
            "train_mvp_mlp_late_fusion.py",
            *base,
            "--probe-preset", preset,
            "--loss-type", loss_type,
            "--save-model", shell_path(out_dir / "model.pth"),
            "--predictions-output", shell_path(out_dir / "predictions_test.csv"),
            "--misclassified-output", shell_path(out_dir / "misclassified_test.csv"),
        ]
        if loss_type == "focal":
            command.extend(["--focal-gamma", str(args.focal_gamma)])
        jobs.append(
            {
                "name": name,
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
    parser = argparse.ArgumentParser(description="Generate the technical-probe late-fusion experiment plan.")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--encoded-dir", type=Path, default=Path("encoded_all"))
    parser.add_argument("--probe-events-dir", type=Path, default=Path("events_all"))
    parser.add_argument("--labels-csv", type=Path, default=Path("labels.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("experiment_runs/late_fusion_technical_round"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pooling", type=str, default="mean_max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--python-bin", type=str, default="python")
    args = parser.parse_args()

    jobs = build_jobs(args)
    plan_path = write_plan(jobs, args.output_root)
    script_path = write_shell(jobs, args.output_root)
    print(f"plan_json={plan_path}")
    print(f"shell_script={script_path}")


if __name__ == "__main__":
    main()
