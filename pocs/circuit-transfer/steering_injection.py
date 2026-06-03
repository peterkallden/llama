#!/usr/bin/env python3
"""Run residual-stream steering injections for verification-plan steps.

This runner uses Hugging Face Transformers when available. The vector format is
kept intentionally simple so the first experiment can start with a hand-exported
candidate vector and later switch to vectors emitted by circuit-tracer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("circuit_transfer.py")
SPEC = importlib.util.spec_from_file_location("circuit_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
circuit_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = circuit_transfer
SPEC.loader.exec_module(circuit_transfer)


@dataclass(frozen=True)
class SteeringVector:
    layer: int
    feature_id: int
    values: list[float]
    source: str = ""
    notes: str = ""

    @property
    def site(self) -> str:
        return f"blocks.{self.layer}.feature.{self.feature_id}"


def load_steering_vector(path: Path) -> SteeringVector:
    row = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted({"layer", "feature_id", "values"} - row.keys())
    if missing:
        raise ValueError(f"{path}: missing steering vector fields: {', '.join(missing)}")
    values = row["values"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: values must be a non-empty list")
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{path}: values must contain only numbers")
    return SteeringVector(
        layer=int(row["layer"]),
        feature_id=int(row["feature_id"]),
        values=[float(value) for value in values],
        source=row.get("source", ""),
        notes=row.get("notes", ""),
    )


def logit_diff_from_logits(logits: list[float], target_id: int, distractor_id: int) -> float:
    return float(logits[target_id] - logits[distractor_id])


def token_id_for_text(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r} encoded to {len(token_ids)} tokens; use a single-token target/distractor for this PoC"
        )
    return int(token_ids[0])


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers
    raise ValueError("Could not find Llama-style decoder layers at model.model.layers")


def patch_hidden_output(output, torch_vector, strength: float, mode: str, token_position: int | str):
    hidden = output[0] if isinstance(output, tuple) else output
    patched = hidden.clone()
    vector = torch_vector.to(device=hidden.device, dtype=hidden.dtype)

    if vector.numel() != hidden.shape[-1]:
        raise ValueError(f"vector has width {vector.numel()}, but hidden size is {hidden.shape[-1]}")

    position = slice(None) if token_position == "all" else int(token_position)
    selected = patched[:, position, :]
    if mode == "inject-to-restore":
        patched[:, position, :] = selected + strength * vector
    elif mode == "ablate-to-fail":
        denominator = vector.dot(vector).clamp_min(1e-12)
        projection_scale = (selected @ vector) / denominator
        patched[:, position, :] = selected - strength * projection_scale.unsqueeze(-1) * vector
    else:
        raise ValueError(f"unsupported steering mode: {mode}")

    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def forward_logit_diff(model, tokenizer, prompt: str, target: str, distractor: str) -> float:
    import torch

    target_id = token_id_for_text(tokenizer, target)
    distractor_id = token_id_for_text(tokenizer, distractor)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return float(logits[target_id].item() - logits[distractor_id].item())


def steered_logit_diff(
    model,
    tokenizer,
    prompt: str,
    target: str,
    distractor: str,
    vector: SteeringVector,
    mode: str,
    strength: float,
    token_position: int | str,
) -> float:
    import torch

    layers = get_decoder_layers(model)
    if vector.layer < 0 or vector.layer >= len(layers):
        raise ValueError(f"layer {vector.layer} is outside model layer range 0..{len(layers) - 1}")
    torch_vector = torch.tensor(vector.values)

    def hook(_module, _inputs, output):
        return patch_hidden_output(output, torch_vector, strength, mode, token_position)

    handle = layers[vector.layer].register_forward_hook(hook)
    try:
        return forward_logit_diff(model, tokenizer, prompt, target, distractor)
    finally:
        handle.remove()


def run_plan(args: argparse.Namespace) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vector = load_steering_vector(args.vector_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    model.eval()

    rows = []
    plan_steps = json.loads(args.plan.read_text(encoding="utf-8"))
    for step in plan_steps:
        if step["layer"] != vector.layer or step["feature_id"] != vector.feature_id:
            if not args.allow_vector_mismatch:
                raise ValueError(
                    f"plan step {step['site']} does not match vector {vector.site}; "
                    "pass --allow-vector-mismatch to override"
                )
        baseline = forward_logit_diff(model, tokenizer, step["prompt"], step["target"], step["distractor"])
        intervention = steered_logit_diff(
            model,
            tokenizer,
            step["prompt"],
            step["target"],
            step["distractor"],
            vector,
            step["intervention"],
            args.strength,
            args.token_position,
        )
        row = circuit_transfer.make_result_row(
            case_id=step["case_id"],
            behavior=step["behavior"],
            model=args.model,
            prompt=step["prompt"],
            target=step["target"],
            distractor=step["distractor"],
            template_id=step.get("template_id", "default"),
            intervention=step["intervention"],
            layer=vector.layer,
            feature_id=vector.feature_id,
            baseline_logit_diff=baseline,
            intervention_logit_diff=intervention,
            graph_path=step.get("graph_path"),
            control_group=step.get("control_group", "target"),
            notes=f"strength={args.strength}; token_position={args.token_position}; vector_source={vector.source}",
        )
        circuit_transfer.append_result_row(args.run_file, row)
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))
    return rows


def command_dry_run(args: argparse.Namespace) -> int:
    vector = load_steering_vector(args.vector_file)
    plan_steps = json.loads(args.plan.read_text(encoding="utf-8"))
    print(f"Loaded vector {vector.site} with width {len(vector.values)}")
    print(f"Would run {len(plan_steps)} plan steps with strength={args.strength}")
    for step in plan_steps:
        print(f"- {step['intervention']} {step['case_id']} at {step['site']}: {step['prompt']!r}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    run_plan(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--plan", type=Path, required=True)
        subparser.add_argument("--vector-file", type=Path, required=True)
        subparser.add_argument("--strength", type=float, default=1.0)

    dry_run = subparsers.add_parser("dry-run", help="validate inputs without loading a model")
    add_common(dry_run)
    dry_run.set_defaults(handler=command_dry_run)

    run = subparsers.add_parser("run", help="run the steering intervention with Hugging Face Transformers")
    add_common(run)
    run.add_argument("--run-file", type=Path, required=True)
    run.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    run.add_argument("--device-map", default="auto")
    run.add_argument("--torch-dtype", default="auto")
    run.add_argument("--token-position", default=-1)
    run.add_argument("--allow-vector-mismatch", action="store_true")
    run.set_defaults(handler=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

