#!/usr/bin/env python3
"""Affine residual-stream stitching between related Llama-family models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path


POC_DIR = Path(__file__).resolve().parent

STEERING_MODULE_PATH = POC_DIR / "steering_injection.py"
STEERING_SPEC = importlib.util.spec_from_file_location("steering_injection", STEERING_MODULE_PATH)
assert STEERING_SPEC and STEERING_SPEC.loader
steering_injection = importlib.util.module_from_spec(STEERING_SPEC)
sys.modules[STEERING_SPEC.name] = steering_injection
STEERING_SPEC.loader.exec_module(steering_injection)


@dataclass(frozen=True)
class PairedPrompt:
    prompt_id: str
    prompt: str
    split: str = "train"


def load_paired_prompts(path: Path) -> list[PairedPrompt]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = sorted({"prompt_id", "prompt"} - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_number}: missing fields: {', '.join(missing)}")
            prompts.append(PairedPrompt(row["prompt_id"], row["prompt"], row.get("split", "train")))
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def fit_affine_numpy(donor, recipient, ridge_lambda: float):
    import numpy as np

    donor = np.asarray(donor, dtype=float)
    recipient = np.asarray(recipient, dtype=float)
    if donor.ndim != 2 or recipient.ndim != 2:
        raise ValueError("donor and recipient activations must be rank-2 arrays")
    if donor.shape[0] != recipient.shape[0]:
        raise ValueError("donor and recipient must have the same number of samples")

    ones = np.ones((donor.shape[0], 1))
    x = np.concatenate([donor, ones], axis=1)
    regularizer = ridge_lambda * np.eye(x.shape[1])
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(x.T @ x + regularizer, x.T @ recipient)
    weight = solution[:-1, :]
    bias = solution[-1, :]
    prediction = donor @ weight + bias
    mse = float(np.mean((prediction - recipient) ** 2))
    return weight, bias, mse


def map_vector_values(values: list[float], weight: list[list[float]]) -> list[float]:
    if not weight or not weight[0]:
        raise ValueError("stitch weight must be a non-empty matrix")
    if len(values) != len(weight):
        raise ValueError(f"vector width {len(values)} does not match stitch input width {len(weight)}")
    mapped = []
    for output_index in range(len(weight[0])):
        mapped.append(float(sum(values[input_index] * weight[input_index][output_index] for input_index in range(len(values)))))
    return mapped


def collect_last_token_residuals(model, tokenizer, prompts: list[PairedPrompt], layers: list[int]):
    import torch

    decoder_layers = steering_injection.get_decoder_layers(model)
    for layer in layers:
        if layer < 0 or layer >= len(decoder_layers):
            raise ValueError(f"layer {layer} is outside model layer range 0..{len(decoder_layers) - 1}")

    activations = {layer: [] for layer in layers}
    hooks = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[layer].append(hidden[:, -1, :].detach().cpu())
            return output
        return hook

    for layer in layers:
        hooks.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt.prompt, return_tensors="pt").to(model.device)
                model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return {layer: torch.cat(values, dim=0) for layer, values in activations.items()}


def command_collect_activations(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_paired_prompts(args.prompts)
    layers = parse_layers(args.layers)
    tokenizer = AutoTokenizer.from_pretrained(args.donor_model)
    donor = AutoModelForCausalLM.from_pretrained(args.donor_model, device_map=args.device_map, torch_dtype=args.torch_dtype)
    recipient = AutoModelForCausalLM.from_pretrained(args.recipient_model, device_map=args.device_map, torch_dtype=args.torch_dtype)
    donor.eval()
    recipient.eval()

    donor_activations = collect_last_token_residuals(donor, tokenizer, prompts, layers)
    recipient_activations = collect_last_token_residuals(recipient, tokenizer, prompts, layers)
    payload = {
        "donor_model": args.donor_model,
        "recipient_model": args.recipient_model,
        "layers": layers,
        "prompts": [prompt.__dict__ for prompt in prompts],
        "donor": donor_activations,
        "recipient": recipient_activations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Wrote paired activations for {len(prompts)} prompts and {len(layers)} layers to {args.output}")
    return 0


def command_fit(args: argparse.Namespace) -> int:
    import torch

    payload = torch.load(args.activations, map_location="cpu")
    layer = int(args.layer)
    donor = payload["donor"][layer].numpy()
    recipient = payload["recipient"][layer].numpy()
    weight, bias, mse = fit_affine_numpy(donor, recipient, args.ridge_lambda)
    stitch = {
        "schema_version": "circuit-transfer-affine-stitch/v1",
        "donor_model": payload["donor_model"],
        "recipient_model": payload["recipient_model"],
        "layer": layer,
        "ridge_lambda": args.ridge_lambda,
        "mse": mse,
        "weight": weight.tolist(),
        "bias": bias.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stitch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"layer": layer, "mse": mse, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


def command_fit_json(args: argparse.Namespace) -> int:
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    weight, bias, mse = fit_affine_numpy(rows["donor"], rows["recipient"], args.ridge_lambda)
    stitch = {
        "schema_version": "circuit-transfer-affine-stitch/v1",
        "donor_model": rows.get("donor_model", "donor"),
        "recipient_model": rows.get("recipient_model", "recipient"),
        "layer": int(rows.get("layer", 0)),
        "ridge_lambda": args.ridge_lambda,
        "mse": mse,
        "weight": weight.tolist(),
        "bias": bias.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stitch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"layer": stitch["layer"], "mse": mse, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


def command_map_vector(args: argparse.Namespace) -> int:
    vector = steering_injection.load_steering_vector(args.vector_file)
    stitch = json.loads(args.stitch.read_text(encoding="utf-8-sig"))
    if vector.layer != int(stitch["layer"]) and not args.allow_layer_mismatch:
        raise ValueError(f"vector layer {vector.layer} does not match stitch layer {stitch['layer']}")
    mapped = {
        "layer": int(stitch["layer"]),
        "feature_id": vector.feature_id,
        "source": f"affine-stitch:{args.stitch}",
        "notes": f"mapped_from={vector.site}; donor={stitch['donor_model']}; recipient={stitch['recipient_model']}; mse={stitch['mse']}",
        "values": map_vector_values(vector.values, stitch["weight"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    steering_injection.load_steering_vector(args.output)
    print(f"Wrote mapped vector {mapped['layer']=} {mapped['feature_id']=} width={len(mapped['values'])} to {args.output}")
    return 0


def command_evaluate_plan(args: argparse.Namespace) -> int:
    plan = [
        {
            "name": "map-vector",
            "command": [
                sys.executable,
                str(POC_DIR / "affine_stitch.py"),
                "map-vector",
                "--stitch",
                str(args.stitch),
                "--vector-file",
                str(args.vector_file),
                "--output",
                str(args.mapped_vector),
            ],
        },
        {
            "name": "recipient-steering",
            "command": [
                sys.executable,
                str(POC_DIR / "steering_injection.py"),
                "run",
                "--plan",
                str(args.verification_plan),
                "--vector-file",
                str(args.mapped_vector),
                "--run-file",
                str(args.run_file),
                "--model",
                args.recipient_model,
                "--top-k",
                str(args.top_k),
            ],
        },
        {
            "name": "recipient-success",
            "command": [
                sys.executable,
                str(POC_DIR / "circuit_transfer.py"),
                "evaluate-success",
                "--run-file",
                str(args.run_file),
            ],
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote affine transfer evaluation plan to {args.output}")
    return 0


def command_dry_run(args: argparse.Namespace) -> int:
    prompts = load_paired_prompts(args.prompts)
    layers = parse_layers(args.layers)
    print(json.dumps({
        "donor_model": args.donor_model,
        "recipient_model": args.recipient_model,
        "prompts": len(prompts),
        "layers": layers,
        "activation_output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="validate paired prompts and activation collection settings")
    dry.add_argument("--donor-model", required=True)
    dry.add_argument("--recipient-model", required=True)
    dry.add_argument("--prompts", type=Path, required=True)
    dry.add_argument("--layers", required=True)
    dry.add_argument("--output", type=Path, required=True)
    dry.set_defaults(handler=command_dry_run)

    collect = subparsers.add_parser("collect-activations", help="collect paired donor/recipient residual activations")
    collect.add_argument("--donor-model", required=True)
    collect.add_argument("--recipient-model", required=True)
    collect.add_argument("--prompts", type=Path, required=True)
    collect.add_argument("--layers", required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--device-map", default="auto")
    collect.add_argument("--torch-dtype", default="auto")
    collect.set_defaults(handler=command_collect_activations)

    fit = subparsers.add_parser("fit", help="fit a ridge affine stitch from collected torch activations")
    fit.add_argument("--activations", type=Path, required=True)
    fit.add_argument("--layer", type=int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--ridge-lambda", type=float, default=0.01)
    fit.set_defaults(handler=command_fit)

    fit_json = subparsers.add_parser("fit-json", help="fit a stitch from lightweight JSON arrays for testing")
    fit_json.add_argument("--input", type=Path, required=True)
    fit_json.add_argument("--output", type=Path, required=True)
    fit_json.add_argument("--ridge-lambda", type=float, default=0.01)
    fit_json.set_defaults(handler=command_fit_json)

    map_vector = subparsers.add_parser("map-vector", help="map a donor steering vector into recipient space")
    map_vector.add_argument("--stitch", type=Path, required=True)
    map_vector.add_argument("--vector-file", type=Path, required=True)
    map_vector.add_argument("--output", type=Path, required=True)
    map_vector.add_argument("--allow-layer-mismatch", action="store_true")
    map_vector.set_defaults(handler=command_map_vector)

    evaluate = subparsers.add_parser("evaluate-plan", help="write commands to evaluate a mapped vector on recipient")
    evaluate.add_argument("--stitch", type=Path, required=True)
    evaluate.add_argument("--vector-file", type=Path, required=True)
    evaluate.add_argument("--mapped-vector", type=Path, required=True)
    evaluate.add_argument("--verification-plan", type=Path, required=True)
    evaluate.add_argument("--run-file", type=Path, required=True)
    evaluate.add_argument("--recipient-model", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.set_defaults(handler=command_evaluate_plan)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
