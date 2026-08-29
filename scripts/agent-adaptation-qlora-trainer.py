#!/usr/bin/env python3
"""External QLoRA/SFT worker for llama-agent adaptation bundles.

This program is deliberately not linked into llama-agent.  The C++ worker
claims an immutable queue directory and starts this program with paths inside
that directory.  It writes only a PEFT-derived GGUF adapter and a
``common_learning_training_result`` JSON document.  Evaluation and activation
remain separate host operations.

Required optional Python packages: torch, transformers, peft, bitsandbytes
and safetensors.  They are intentionally not dependencies of the inference
package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one llama-agent QLoRA SFT bundle")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="local Hugging Face training checkpoint; never a serving GGUF")
    parser.add_argument("--base-config", type=Path, required=True,
                        help="local Hugging Face base config/tokenizer directory for GGUF conversion")
    parser.add_argument("--converter", type=Path, required=True,
                        help="repository convert_lora_to_gguf.py")
    parser.add_argument("--adapter-id-prefix", default="agent-adaptation")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split", "train") != "train":
                continue
            prompt = value.get("prompt") or value.get("input")
            target = value.get("target")
            if not isinstance(prompt, str) or not isinstance(target, str) or not prompt or not target:
                fail(f"corpus row {line_number} requires non-empty prompt/input and target strings")
            pairs.append({"prompt": prompt, "target": target})
    if not pairs:
        fail("corpus contains no train-split SFT pairs")
    return pairs


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.job, args.corpus, args.corpus_manifest, args.converter):
        if not path.is_file():
            fail(f"required trainer file does not exist: {path}")
    for path in (args.model, args.base_config):
        if not path.is_dir():
            fail("QLoRA requires a local Hugging Face checkpoint directory, not a serving GGUF: " + str(path))
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1 or job.get("trainer_kind") != "qlora-sft":
        fail("QLoRA trainer accepts only schema-version 1 qlora-sft jobs")
    if not isinstance(job.get("id"), str) or not isinstance(job.get("corpus_bundle_hash"), str):
        fail("training job has incomplete identity")
    # Read the manifest before training, so a malformed bundle cannot consume a
    # GPU merely because its JSONL happens to parse.
    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jsonl_hash"), str):
        fail("corpus manifest is not a valid corpus-builder manifest")
    return job


def train(args: argparse.Namespace, job: dict[str, Any], pairs: list[dict[str, str]]) -> Path:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        fail("QLoRA dependencies are unavailable; install torch, transformers, peft, bitsandbytes and safetensors: " + str(exc))

    if not torch.cuda.is_available():
        fail("QLoRA trainer requires a CUDA-capable PyTorch runtime")
    if args.rank <= 0 or args.alpha <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        fail("QLoRA rank, alpha, batch size and gradient accumulation must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[value for value in args.target_modules.split(",") if value],
        bias="none",
    ))

    class SFTDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(pairs)

        def __getitem__(self, index: int) -> dict[str, Any]:
            pair = pairs[index]
            prompt_ids = tokenizer(pair["prompt"] + "\n", add_special_tokens=True).input_ids
            full = tokenizer(pair["prompt"] + "\n" + pair["target"], add_special_tokens=True,
                             truncation=True, max_length=args.max_seq_len).input_ids
            labels = list(full)
            for label_index in range(min(len(prompt_ids), len(labels))):
                labels[label_index] = -100
            return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(len(row["input_ids"]) for row in rows)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([row["input_ids"] + [pad] * (maximum - len(row["input_ids"])) for row in rows]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (maximum - len(row["attention_mask"])) for row in rows]),
            "labels": torch.tensor([row["labels"] + [-100] * (maximum - len(row["labels"])) for row in rows]),
        }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    adapter_id = args.adapter_id_prefix + "-" + hashlib.sha256(job["id"].encode()).hexdigest()[:16]
    hf_output = args.artifacts / "hf" / adapter_id
    training_args = TrainingArguments(
        output_dir=str(args.artifacts / "work"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=use_bf16,
        fp16=not use_bf16,
        disable_tqdm=True,
        seed=int(job.get("seed", 42)),
        remove_unused_columns=False,
    )
    Trainer(model=model, args=training_args, train_dataset=SFTDataset(), data_collator=collate).train()
    model.save_pretrained(hf_output, safe_serialization=True)
    tokenizer.save_pretrained(hf_output)
    artifact = args.artifacts / "adapters" / f"{adapter_id}.gguf"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(args.converter), "--outfile", str(artifact), "--outtype", "f16",
        "--base", str(args.base_config), str(hf_output),
    ], check=True)
    return artifact


def write_result(args: argparse.Namespace, job: dict[str, Any], artifact: Path) -> None:
    relative = artifact.relative_to(args.artifacts).as_posix()
    result = {
        "schema_version": 1,
        "job_id": job["id"],
        "adapter_id": artifact.stem,
        "artifact_path": relative,
        "artifact_sha256": sha256(artifact),
        "corpus_bundle_hash": job["corpus_bundle_hash"],
        "base_training_fingerprint": job["base_training_fingerprint"],
        "trainer_kind": job["trainer_kind"],
        "trainer_version": job["trainer_version"],
        "evaluation_revision": "not-run:" + job["id"],
        "evaluation_status": "not_run",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.result.parent, delete=False) as stream:
        json.dump(result, stream, separators=(",", ":"))
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, args.result)


def main() -> int:
    args = parse_args()
    try:
        job = validate_inputs(args)
        artifact = train(args, job, read_pairs(args.corpus))
        write_result(args, job, artifact)
        return 0
    except Exception as exc:  # The C++ worker records a bounded safe summary.
        print(f"agent adaptation QLoRA trainer failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
