#!/usr/bin/env python3
"""Evaluate a trained LoRA adapter with the circuit-transfer result schema."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("circuit_transfer.py")
SPEC = importlib.util.spec_from_file_location("circuit_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
circuit_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = circuit_transfer
SPEC.loader.exec_module(circuit_transfer)

STEERING_MODULE_PATH = Path(__file__).with_name("steering_injection.py")
STEERING_SPEC = importlib.util.spec_from_file_location("steering_injection", STEERING_MODULE_PATH)
assert STEERING_SPEC and STEERING_SPEC.loader
steering_injection = importlib.util.module_from_spec(STEERING_SPEC)
sys.modules[STEERING_SPEC.name] = steering_injection
STEERING_SPEC.loader.exec_module(steering_injection)


def build_lora_result_row(step: dict, model_name: str, adapter_path: Path, baseline: float, adapter: float) -> dict:
    return circuit_transfer.make_result_row(
        case_id=step["case_id"],
        behavior=step["behavior"],
        model=f"{model_name}+lora:{adapter_path}",
        prompt=step["prompt"],
        target=step["target"],
        distractor=step["distractor"],
        template_id=step.get("template_id", "default"),
        intervention="lora-restore",
        layer=int(step["layer"]),
        feature_id=int(step["feature_id"]),
        baseline_logit_diff=baseline,
        intervention_logit_diff=adapter,
        graph_path=step.get("graph_path"),
        control_group=step.get("control_group", "target"),
        notes=f"adapter={adapter_path}",
    )


def run_plan(args: argparse.Namespace) -> list[dict]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    adapter_model = PeftModel.from_pretrained(base_model, args.adapter)
    adapter_model.eval()

    rows = []
    plan_steps = json.loads(args.plan.read_text(encoding="utf-8"))
    for step in plan_steps:
        if step["intervention"] != "inject-to-restore":
            continue
        target_id = steering_injection.token_id_for_text(tokenizer, step["target"])
        distractor_id = steering_injection.token_id_for_text(tokenizer, step["distractor"])
        with adapter_model.disable_adapter():
            base_logits = steering_injection.forward_logits(adapter_model, tokenizer, step["prompt"])
        adapter_logits = steering_injection.forward_logits(adapter_model, tokenizer, step["prompt"])
        baseline = float(base_logits[target_id].item() - base_logits[distractor_id].item())
        adapter = float(adapter_logits[target_id].item() - adapter_logits[distractor_id].item())
        row = build_lora_result_row(step, args.model, args.adapter, baseline, adapter)
        if args.top_k > 0:
            row["baseline_top_k"] = steering_injection.top_k_from_logits(base_logits, tokenizer, args.top_k)
            row["intervention_top_k"] = steering_injection.top_k_from_logits(adapter_logits, tokenizer, args.top_k)
            circuit_transfer.validate_result_row(row)
        circuit_transfer.append_result_row(args.run_file, row)
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))
    return rows


def command_dry_run(args: argparse.Namespace) -> int:
    plan_steps = json.loads(args.plan.read_text(encoding="utf-8"))
    restore_steps = [step for step in plan_steps if step["intervention"] == "inject-to-restore"]
    print(json.dumps({
        "adapter": str(args.adapter),
        "model": args.model,
        "restore_steps": len(restore_steps),
        "run_file": str(args.run_file),
        "top_k": args.top_k,
    }, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    run_plan(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--plan", type=Path, required=True)
        subparser.add_argument("--adapter", type=Path, required=True)
        subparser.add_argument("--run-file", type=Path, required=True)
        subparser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
        subparser.add_argument("--top-k", type=int, default=20)

    dry_run = subparsers.add_parser("dry-run", help="validate LoRA evaluation inputs without loading models")
    add_common(dry_run)
    dry_run.set_defaults(handler=command_dry_run)

    run = subparsers.add_parser("run", help="evaluate base + LoRA with Hugging Face PEFT")
    add_common(run)
    run.add_argument("--device-map", default="auto")
    run.add_argument("--torch-dtype", default="auto")
    run.set_defaults(handler=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
