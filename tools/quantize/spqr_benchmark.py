#!/usr/bin/env python3
"""Reproducible benchmark harness for SPQR-guided quantization experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_VARIANTS = [
    {"name": "baseline_q4_k_m", "type": "Q4_K_M", "args": []},
    {"name": "baseline_q3_k_m", "type": "Q3_K_M", "args": []},
    {
        "name": "spqr_guided",
        "type": "Q3_K_M",
        "use_imatrix": True,
        "args": ["--mixed-policy", "spqr_guided", "--spqr-block-scoring"],
    },
    {
        "name": "spqr_layer_delta",
        "type": "Q3_K_M",
        "use_imatrix": True,
        "args": ["--mixed-policy", "spqr_layer_delta", "--spqr-block-scoring", "--adaptive-anchors"],
    },
    {
        "name": "spqr_layer_delta_rd",
        "type": "Q3_K_M",
        "use_imatrix": True,
        "args": [
            "--mixed-policy", "spqr_layer_delta",
            "--spqr-block-scoring", "--adaptive-anchors", "--rd-guided",
        ],
        "prefer_profile_imatrix": True,
    },
]

PPL_RE = re.compile(r"(?:Final estimate:\s+PPL\s*=|perplexity:)\s*([0-9.eE+-]+)")
QUANT_SIZE_RE = re.compile(r"quant size\s*=\s*([0-9.]+)\s*MiB\s*\(([0-9.]+)\s*BPW\)")
QUANT_TIME_RE = re.compile(r"quantize time\s*=\s*([0-9.]+)\s*ms")


def command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run_command(command: list[str], log_path: Path, dry_run: bool) -> dict:
    started = time.time()
    if dry_run:
        output = f"DRY RUN: {command_string(command)}\n"
        log_path.write_text(output, encoding="utf-8")
        return {"exit_code": 0, "elapsed_seconds": 0.0, "output": output}

    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = process.stdout
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return {
        "exit_code": process.returncode,
        "elapsed_seconds": time.time() - started,
        "output": output,
    }


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_variants(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_VARIANTS
    data = json.loads(path.read_text(encoding="utf-8"))
    variants = data.get("variants", data)
    if not isinstance(variants, list):
        raise ValueError("variant config must be a list or contain a 'variants' list")
    for variant in variants:
        if not all(key in variant for key in ("name", "type", "args")):
            raise ValueError("each variant requires name, type, and args")
    return variants


def parse_metrics(output: str) -> dict:
    metrics = {}
    if match := QUANT_SIZE_RE.search(output):
        metrics["reported_size_mib"] = float(match.group(1))
        metrics["reported_bpw"] = float(match.group(2))
    if match := QUANT_TIME_RE.search(output):
        metrics["reported_quantize_ms"] = float(match.group(1))
    if matches := PPL_RE.findall(output):
        metrics["perplexity"] = float(matches[-1])
    return metrics


def executable(build_bin: Path, name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return str(build_bin / f"{name}{suffix}")


def validate_paths(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    required = [args.input_model]
    optional = [args.imatrix, args.profile_imatrix, args.ppl_dataset]
    for path in required + [value for value in optional if value]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not args.dry_run:
        for name in ("llama-quantize", "llama-perplexity", "llama-cli"):
            if name == "llama-perplexity" and not args.ppl_dataset:
                continue
            if name == "llama-cli" and args.skip_generation:
                continue
            path = Path(executable(args.build_bin, name))
            if not path.is_file():
                raise FileNotFoundError(path)


def selected_imatrix(variant: dict, args: argparse.Namespace) -> Path | None:
    if not variant.get("use_imatrix", False):
        return None
    if variant.get("prefer_profile_imatrix") and args.profile_imatrix:
        return args.profile_imatrix
    return args.imatrix or args.profile_imatrix


def write_summary(path: Path, rows: list[dict]) -> None:
    fields = [
        "variant", "status", "quant_type", "model_size_bytes", "model_size_mib",
        "reported_bpw", "quantize_seconds", "perplexity", "generation_exit_code",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-bin", type=Path, required=True, help="directory containing llama.cpp executables")
    parser.add_argument("--input-model", type=Path, required=True, help="high-precision input GGUF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, help="standard or profile-enabled imatrix used by guided variants")
    parser.add_argument("--profile-imatrix", type=Path, help="preferred profile-enabled imatrix for the full RD variant")
    parser.add_argument("--ppl-dataset", type=Path, help="enable perplexity evaluation using this text file")
    parser.add_argument("--variants", type=Path, help="JSON variant configuration")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--ppl-context", type=int, default=512)
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--generation-prompt", default="Write a short explanation of quantization.")
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="continue after failed stages")
    parser.add_argument("--resume", action="store_true", help="reuse existing output models")
    parser.add_argument("--dry-run", action="store_true", help="print and record commands without executing them")
    args = parser.parse_args()

    validate_paths(args)
    variants = load_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    results_jsonl = args.output_dir / "results.jsonl"
    summary_csv = args.output_dir / "summary.csv"
    if not args.resume and results_jsonl.exists():
        results_jsonl.unlink()

    quantize_exe = executable(args.build_bin, "llama-quantize")
    perplexity_exe = executable(args.build_bin, "llama-perplexity")
    cli_exe = executable(args.build_bin, "llama-cli")
    summary = []

    for variant in variants:
        name = variant["name"]
        output_model = args.output_dir / f"{name}.gguf"
        row = {"variant": name, "quant_type": variant["type"], "status": "pending"}
        imatrix = selected_imatrix(variant, args)

        quant_command = [quantize_exe, *variant["args"]]
        if imatrix:
            quant_command += ["--imatrix", str(imatrix)]
        quant_command += [str(args.input_model), str(output_model), variant["type"]]
        if args.threads > 0:
            quant_command.append(str(args.threads))

        if args.resume and output_model.exists():
            quant_result = {"exit_code": 0, "elapsed_seconds": 0.0, "output": "resume: existing model\n"}
        else:
            quant_result = run_command(quant_command, logs_dir / f"{name}.quantize.log", args.dry_run)
        append_jsonl(results_jsonl, {
            "variant": name, "stage": "quantize", "command": quant_command,
            "exit_code": quant_result["exit_code"], "elapsed_seconds": quant_result["elapsed_seconds"],
            **parse_metrics(quant_result["output"]),
        })
        row.update(parse_metrics(quant_result["output"]))
        row["quantize_seconds"] = quant_result["elapsed_seconds"]

        if quant_result["exit_code"] != 0:
            row["status"] = "quantize_failed"
            summary.append(row)
            write_summary(summary_csv, summary)
            if args.keep_going:
                continue
            return quant_result["exit_code"]

        if output_model.exists():
            row["model_size_bytes"] = output_model.stat().st_size
            row["model_size_mib"] = output_model.stat().st_size / (1024 * 1024)

        if args.ppl_dataset:
            ppl_command = [
                perplexity_exe, "-m", str(output_model), "-f", str(args.ppl_dataset),
                "-c", str(args.ppl_context), "--chunks", str(args.ppl_chunks),
            ]
            if args.gpu_layers > 0:
                ppl_command += ["-ngl", str(args.gpu_layers)]
            ppl_result = run_command(ppl_command, logs_dir / f"{name}.perplexity.log", args.dry_run)
            ppl_metrics = parse_metrics(ppl_result["output"])
            row.update(ppl_metrics)
            append_jsonl(results_jsonl, {
                "variant": name, "stage": "perplexity", "command": ppl_command,
                "exit_code": ppl_result["exit_code"], "elapsed_seconds": ppl_result["elapsed_seconds"],
                **ppl_metrics,
            })
            if ppl_result["exit_code"] != 0 and not args.keep_going:
                row["status"] = "perplexity_failed"
                summary.append(row)
                write_summary(summary_csv, summary)
                return ppl_result["exit_code"]

        if not args.skip_generation:
            gen_command = [
                cli_exe, "-m", str(output_model), "-p", args.generation_prompt,
                "-n", str(args.generation_tokens), "-no-cnv",
            ]
            if args.gpu_layers > 0:
                gen_command += ["-ngl", str(args.gpu_layers)]
            gen_result = run_command(gen_command, logs_dir / f"{name}.generation.log", args.dry_run)
            row["generation_exit_code"] = gen_result["exit_code"]
            append_jsonl(results_jsonl, {
                "variant": name, "stage": "generation", "command": gen_command,
                "exit_code": gen_result["exit_code"], "elapsed_seconds": gen_result["elapsed_seconds"],
            })
            if gen_result["exit_code"] != 0 and not args.keep_going:
                row["status"] = "generation_failed"
                summary.append(row)
                write_summary(summary_csv, summary)
                return gen_result["exit_code"]

        row["status"] = "ok"
        summary.append(row)
        write_summary(summary_csv, summary)

    print(f"Wrote {results_jsonl}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
