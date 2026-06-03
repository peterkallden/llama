#!/usr/bin/env python3
"""Distill successful steering interventions into a small LoRA adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
class DistillationExample:
    prompt: str
    target: str
    case_id: str
    template_id: str
    teacher_delta_logit_diff: float
    source_site: str


def normalize_target_text(text: str) -> str:
    return text if text.startswith(" ") else f" {text}"


def build_distillation_examples(
    rows: list[dict],
    min_delta: float,
    include_controls: bool = False,
) -> list[DistillationExample]:
    examples = []
    seen = set()
    for row in rows:
        circuit_transfer.validate_result_row(row)
        if row["intervention"] != "inject-to-restore":
            continue
        if not include_controls and row["control_group"] != "target":
            continue
        delta = float(row["delta_logit_diff"])
        if delta < min_delta:
            continue
        key = (row["prompt"], row["target"], row["case_id"], row["template_id"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            DistillationExample(
                prompt=row["prompt"],
                target=normalize_target_text(row["target"]),
                case_id=row["case_id"],
                template_id=row["template_id"],
                teacher_delta_logit_diff=delta,
                source_site=row["site"],
            )
        )
    return examples


def write_distillation_dataset(examples: list[DistillationExample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.__dict__, sort_keys=True) + "\n")


def load_distillation_dataset(path: Path) -> list[DistillationExample]:
    examples = []
    for row in circuit_transfer.load_jsonl(path):
        missing = sorted({"prompt", "target", "case_id", "template_id", "teacher_delta_logit_diff", "source_site"} - row.keys())
        if missing:
            raise ValueError(f"{path}: missing distillation fields: {', '.join(missing)}")
        examples.append(
            DistillationExample(
                prompt=row["prompt"],
                target=normalize_target_text(row["target"]),
                case_id=row["case_id"],
                template_id=row["template_id"],
                teacher_delta_logit_diff=float(row["teacher_delta_logit_diff"]),
                source_site=row["source_site"],
            )
        )
    return examples


def train_lora(args: argparse.Namespace) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = load_distillation_dataset(args.dataset)
    if not examples:
        raise ValueError(f"{args.dataset}: no distillation examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules.split(","),
    )
    model = get_peft_model(model, config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = next(model.parameters()).device

    for step in range(args.steps):
        example = examples[step % len(examples)]
        text = example.prompt + example.target
        encoded_full = tokenizer(text, return_tensors="pt").to(device)
        encoded_prompt = tokenizer(example.prompt, return_tensors="pt").to(device)

        labels = encoded_full["input_ids"].clone()
        prompt_length = encoded_prompt["input_ids"].shape[1]
        labels[:, :prompt_length] = -100

        outputs = model(**encoded_full, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if args.log_every and (step + 1) % args.log_every == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.item())}, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "base_model": args.model,
        "dataset": str(args.dataset),
        "examples": len(examples),
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules.split(","),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
    }
    (args.output_dir / "distillation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def command_prepare_dataset(args: argparse.Namespace) -> int:
    rows = circuit_transfer.load_jsonl(args.run_file)
    examples = build_distillation_examples(rows, args.min_delta, include_controls=args.include_controls)
    write_distillation_dataset(examples, args.output)
    print(f"Wrote {len(examples)} distillation examples to {args.output}")
    return 0


def command_dry_run(args: argparse.Namespace) -> int:
    examples = load_distillation_dataset(args.dataset)
    batches = math.ceil(max(len(examples), 1) / args.batch_size)
    summary = {
        "dataset": str(args.dataset),
        "examples": len(examples),
        "estimated_batches_per_epoch": batches,
        "rank": args.rank,
        "target_modules": args.target_modules.split(","),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_train_lora(args: argparse.Namespace) -> int:
    metadata = train_lora(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--batch-size", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-dataset", help="filter successful steering rows into LoRA training data")
    prepare.add_argument("--run-file", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--min-delta", type=float, default=2.0)
    prepare.add_argument("--include-controls", action="store_true")
    prepare.set_defaults(handler=command_prepare_dataset)

    dry_run = subparsers.add_parser("dry-run", help="summarize a distillation dataset and LoRA config")
    add_lora_args(dry_run)
    dry_run.set_defaults(handler=command_dry_run)

    train = subparsers.add_parser("train-lora", help="train a small PEFT LoRA adapter")
    add_lora_args(train)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--device-map", default="auto")
    train.add_argument("--torch-dtype", default="auto")
    train.add_argument("--log-every", type=int, default=10)
    train.set_defaults(handler=command_train_lora)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

