#!/usr/bin/env python3
"""Optional QLoRA DPO trainer for immutable llama-agent preference bundles.

Uses the same queue job/result files and GGUF conversion path as the SFT
trainer. It trains only an adapter; evaluation and activation remain host
operations.
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
    parser = argparse.ArgumentParser(description="Train one llama-agent QLoRA DPO bundle")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--adapter-id-prefix", default="agent-adaptation-dpo")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
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


def corpus_identity(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "identity:fnv1a64:" + format(value, "016x")


def read_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("split", "train") != "train":
                continue
            prompt, chosen, rejected = value.get("prompt"), value.get("chosen"), value.get("rejected")
            if not all(isinstance(item, str) and item.strip()
                       for item in (prompt, chosen, rejected)):
                fail(f"corpus row {line_number} requires non-empty prompt, chosen and rejected strings")
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    if not pairs:
        fail("corpus contains no train-split DPO preference pairs")
    return pairs


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    for path in (args.job, args.corpus, args.corpus_manifest, args.converter):
        if not path.is_file():
            fail(f"required trainer file does not exist: {path}")
    for path in (args.model, args.base_config):
        if not path.is_dir():
            fail("DPO requires local Hugging Face checkpoint directories: " + str(path))
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1 or job.get("trainer_kind") != "qlora-dpo" or job.get("objective") != "dpo":
        fail("DPO trainer accepts only schema-version 1 qlora-dpo jobs with objective=dpo")
    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("objective") != "dpo":
        fail("corpus manifest objective does not match DPO trainer")
    if not isinstance(manifest.get("jsonl_hash"), str) or not manifest["jsonl_hash"]:
        fail("corpus manifest is missing its JSONL identity")
    if manifest["jsonl_hash"] != corpus_identity(args.corpus.read_bytes()):
        fail("corpus JSONL does not match the manifest hash")
    if args.rank <= 0 or args.alpha <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        fail("rank, alpha, batch size and gradient accumulation must be positive")
    if args.beta <= 0 or args.max_seq_len <= 1 or args.max_prompt_len <= 0:
        fail("beta and sequence limits must be positive")
    return job, read_pairs(args.corpus)


def train(args: argparse.Namespace, job: dict[str, Any], pairs: list[dict[str, str]]) -> Path:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail("DPO dependencies are unavailable; install the optional QLoRA trainer requirements: " + str(exc))

    if not torch.cuda.is_available():
        fail("DPO trainer requires a CUDA-capable PyTorch runtime")
    torch.manual_seed(int(job.get("seed", 42)))
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
    if not hasattr(model, "disable_adapter"):
        fail("installed PEFT version does not support adapter-disabled reference scoring")

    encoded = []
    for pair in pairs:
        prompt_ids = tokenizer(pair["prompt"], add_special_tokens=True,
            truncation=True, max_length=args.max_prompt_len).input_ids
        if tokenizer.eos_token_id is not None:
            prompt_ids.append(tokenizer.eos_token_id)
        chosen_ids = tokenizer(pair["chosen"], add_special_tokens=False,
            truncation=True, max_length=args.max_seq_len).input_ids
        rejected_ids = tokenizer(pair["rejected"], add_special_tokens=False,
            truncation=True, max_length=args.max_seq_len).input_ids
        if not chosen_ids or not rejected_ids:
            fail("tokenization produced an empty chosen or rejected response")
        prompt_ids = prompt_ids[:args.max_seq_len - 1]
        chosen_ids = chosen_ids[:args.max_seq_len - len(prompt_ids)]
        rejected_ids = rejected_ids[:args.max_seq_len - len(prompt_ids)]
        if not chosen_ids or not rejected_ids:
            fail("sequence limit removed the chosen or rejected response")
        encoded.append((prompt_ids, chosen_ids, rejected_ids))

    def batch_tensors(rows: list[tuple[list[int], list[int], list[int]]], side: int):
        sequences = [row[0] + row[side] for row in rows]
        labels = [[-100] * len(row[0]) + row[side] for row in rows]
        width = max(len(sequence) for sequence in sequences)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor([sequence + [pad] * (width - len(sequence)) for sequence in sequences],
            device=model.device)
        attention = torch.tensor([[1] * len(sequence) + [0] * (width - len(sequence)) for sequence in sequences],
            device=model.device)
        label_ids = torch.tensor([label + [-100] * (width - len(label)) for label in labels],
            device=model.device)
        return input_ids, attention, label_ids

    def sequence_logps(input_ids, attention, labels):
        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1, :]
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        token_logps = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return (token_logps * mask).sum(dim=-1)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()
    global_step = 0
    for _ in range(max(1, int(args.epochs + 0.999))):
        order = torch.randperm(len(encoded)).tolist()
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            rows = [encoded[index] for index in indices]
            chosen = batch_tensors(rows, 1)
            rejected = batch_tensors(rows, 2)
            policy_chosen = sequence_logps(*chosen)
            policy_rejected = sequence_logps(*rejected)
            with torch.no_grad(), model.disable_adapter():
                reference_chosen = sequence_logps(*chosen)
                reference_rejected = sequence_logps(*rejected)
            preference_margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
            loss = -torch.nn.functional.logsigmoid(args.beta * preference_margin).mean()
            (loss / args.gradient_accumulation).backward()
            global_step += 1
            if global_step % args.gradient_accumulation == 0 or offset + args.batch_size >= len(order):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    adapter_id = args.adapter_id_prefix + "-" + hashlib.sha256(job["id"].encode()).hexdigest()[:16]
    hf_output = args.artifacts / "hf" / adapter_id
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
        job, pairs = validate_inputs(args)
        artifact = train(args, job, pairs)
        write_result(args, job, artifact)
        return 0
    except Exception as exc:
        print(f"agent adaptation DPO trainer failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
