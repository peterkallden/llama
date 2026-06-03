#!/usr/bin/env python3
"""Generate and optionally dry-run an end-to-end circuit-transfer workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


POC_DIR = Path(__file__).resolve().parent


def command_args_to_string(args: list[str]) -> str:
    return " ".join(str(arg) for arg in args)


def build_pipeline(args: argparse.Namespace) -> list[dict]:
    work_dir = args.work_dir
    plan = work_dir / "verification-plan.json"
    steering_results = work_dir / "steering-results.jsonl"
    distill_dataset = work_dir / "distill" / "train.jsonl"
    lora_dir = work_dir / "lora" / args.adapter_name
    lora_results = work_dir / "lora-results.jsonl"
    report = work_dir / "pipeline-report.json"

    return [
        {
            "name": "build-verification-plan",
            "kind": "light",
            "command": [
                sys.executable,
                str(POC_DIR / "verify_interventions.py"),
                "--candidates",
                str(args.candidates),
                "--output",
                str(plan),
            ],
        },
        {
            "name": "inspect-vector",
            "kind": "light",
            "command": [
                sys.executable,
                str(POC_DIR / "vector_io.py"),
                "inspect",
                "--vector-file",
                str(args.vector_file),
            ],
        },
        {
            "name": "steering-dry-run",
            "kind": "light",
            "command": [
                sys.executable,
                str(POC_DIR / "steering_injection.py"),
                "dry-run",
                "--plan",
                str(plan),
                "--vector-file",
                str(args.vector_file),
            ],
        },
        {
            "name": "steering-run",
            "kind": "heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "steering_injection.py"),
                "run",
                "--plan",
                str(plan),
                "--vector-file",
                str(args.vector_file),
                "--run-file",
                str(steering_results),
                "--model",
                args.model,
                "--top-k",
                str(args.top_k),
            ],
        },
        {
            "name": "evaluate-steering-success",
            "kind": "light-after-heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "circuit_transfer.py"),
                "evaluate-success",
                "--run-file",
                str(steering_results),
            ],
        },
        {
            "name": "prepare-distillation-dataset",
            "kind": "light-after-heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "distill_lora.py"),
                "prepare-dataset",
                "--run-file",
                str(steering_results),
                "--output",
                str(distill_dataset),
                "--min-delta",
                str(args.min_delta),
            ],
        },
        {
            "name": "train-lora",
            "kind": "heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "distill_lora.py"),
                "train-lora",
                "--dataset",
                str(distill_dataset),
                "--output-dir",
                str(lora_dir),
                "--model",
                args.model,
                "--rank",
                str(args.rank),
                "--top-k-kl-weight",
                str(args.top_k_kl_weight),
            ],
        },
        {
            "name": "evaluate-lora",
            "kind": "heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "evaluate_lora.py"),
                "run",
                "--plan",
                str(plan),
                "--adapter",
                str(lora_dir),
                "--run-file",
                str(lora_results),
                "--model",
                args.model,
                "--top-k",
                str(args.top_k),
            ],
        },
        {
            "name": "evaluate-lora-success",
            "kind": "light-after-heavy",
            "command": [
                sys.executable,
                str(POC_DIR / "circuit_transfer.py"),
                "evaluate-success",
                "--run-file",
                str(lora_results),
                "--no-require-ablation",
            ],
        },
        {
            "name": "write-report",
            "kind": "report",
            "path": str(report),
        },
    ]


def command_plan(args: argparse.Namespace) -> int:
    steps = build_pipeline(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "work_dir": str(args.work_dir),
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote pipeline plan to {args.output}")
    for step in steps:
        if "command" in step:
            print(f"- {step['name']} [{step['kind']}]: {command_args_to_string(step['command'])}")
    return 0


def command_dry_run(args: argparse.Namespace) -> int:
    steps = build_pipeline(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    executed = []
    for step in steps:
        if step["kind"] != "light":
            continue
        subprocess.run(step["command"], check=True)
        executed.append(step["name"])
    print(json.dumps({"executed": executed}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--candidates", type=Path, default=POC_DIR / "data" / "candidate_features.example.jsonl")
        subparser.add_argument("--vector-file", type=Path, default=POC_DIR / "data" / "steering_vector.example.json")
        subparser.add_argument("--work-dir", type=Path, default=Path("work") / "circuit-transfer")
        subparser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
        subparser.add_argument("--adapter-name", default="candidate-r4")
        subparser.add_argument("--rank", type=int, default=4)
        subparser.add_argument("--top-k", type=int, default=20)
        subparser.add_argument("--min-delta", type=float, default=2.0)
        subparser.add_argument("--top-k-kl-weight", type=float, default=0.2)

    plan = subparsers.add_parser("plan", help="write an end-to-end pipeline plan")
    add_common(plan)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(handler=command_plan)

    dry_run = subparsers.add_parser("dry-run", help="execute only light validation steps")
    add_common(dry_run)
    dry_run.set_defaults(handler=command_dry_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
