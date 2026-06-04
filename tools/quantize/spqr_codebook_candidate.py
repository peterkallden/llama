#!/usr/bin/env python3
"""Offline AQLM-style codebook candidate analysis for GGUF tensors.

This tool is intentionally report-only. It does not write GGUF files, add runtime
formats, or implement AQLM. It samples high-precision tensor blocks and compares
simple scalar k-means reconstruction against a tiny additive codebook model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))

np = None
GGMLQuantizationType = None
GGUFReader = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GGUF tensor blocks for future additive/codebook quantization experiments.")
    parser.add_argument("model", type=Path, help="input F16/F32 GGUF model")
    parser.add_argument("--tensor", action="append", default=[], help="substring filter for tensor names; may be repeated")
    parser.add_argument("--max-tensors", type=int, default=16, help="maximum matching tensors to analyze")
    parser.add_argument("--max-blocks", type=int, default=64, help="maximum blocks per tensor")
    parser.add_argument("--block-size", type=int, default=256, help="number of weights per analyzed block")
    parser.add_argument("--codebooks", type=int, default=2, help="number of additive residual codebooks")
    parser.add_argument("--codebook-size", type=int, default=16, help="entries per codebook")
    parser.add_argument("--iters", type=int, default=8, help="k-means iterations per codebook")
    parser.add_argument("--seed", type=int, default=1, help="sampling seed")
    parser.add_argument("--output", type=Path, default=None, help="JSONL output path; defaults to stdout")
    return parser.parse_args()


def load_runtime_deps() -> None:
    global np
    global GGMLQuantizationType
    global GGUFReader

    import numpy as numpy
    from gguf.constants import GGMLQuantizationType as QuantizationType
    from gguf.gguf_reader import GGUFReader as Reader

    np = numpy
    GGMLQuantizationType = QuantizationType
    GGUFReader = Reader


def tensor_data_f32(tensor):
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(-1)
    if tensor.tensor_type == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(-1)
    return None


def kmeans_1d(values, k: int, iters: int):
    if values.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    k = max(1, min(k, values.size))
    if k == 1:
        centers = np.array([float(values.mean())], dtype=np.float32)
        return centers, np.zeros(values.shape, dtype=np.int32)

    centers = np.percentile(values, np.linspace(0.0, 100.0, k, dtype=np.float32)).astype(np.float32)
    codes = np.zeros(values.shape, dtype=np.int32)

    for _ in range(iters):
        distances = np.abs(values[:, None] - centers[None, :])
        codes = np.argmin(distances, axis=1).astype(np.int32)
        for i in range(k):
            mask = codes == i
            if np.any(mask):
                centers[i] = values[mask].mean()

    return centers, codes


def scalar_reconstruct(block, codebook_size: int, iters: int):
    centers, codes = kmeans_1d(block, codebook_size, iters)
    return centers[codes]


def additive_reconstruct(block, n_codebooks: int, codebook_size: int, iters: int):
    recon = np.zeros_like(block, dtype=np.float32)
    residual = block.astype(np.float32, copy=True)

    for _ in range(max(1, n_codebooks)):
        centers, codes = kmeans_1d(residual, codebook_size, iters)
        update = centers[codes]
        recon += update
        residual -= update

    return recon


def mse(a, b) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def analyze_tensor(tensor, data, args: argparse.Namespace, rng) -> list[dict]:
    if data.size < args.block_size:
        return []

    n_blocks = data.size // args.block_size
    block_ids = np.arange(n_blocks)
    if n_blocks > args.max_blocks:
        block_ids = np.sort(rng.choice(block_ids, size=args.max_blocks, replace=False))

    rows = []
    scalar_bpw = math.log2(args.codebook_size)
    additive_bpw = args.codebooks * math.log2(args.codebook_size)

    for block_id in block_ids:
        start = int(block_id) * args.block_size
        block = data[start:start + args.block_size].astype(np.float32, copy=False)

        scalar = scalar_reconstruct(block, args.codebook_size, args.iters)
        additive = additive_reconstruct(block, args.codebooks, args.codebook_size, args.iters)

        baseline_error = mse(block, scalar)
        codebook_error = mse(block, additive)

        rows.append({
            "tensor": tensor.name,
            "tensor_type": tensor.tensor_type.name,
            "block_id": int(block_id),
            "block_size": int(args.block_size),
            "scalar_kmeans_error": baseline_error,
            "additive_codebook_error": codebook_error,
            "error_ratio": codebook_error / baseline_error if baseline_error > 0 else 0.0,
            "scalar_index_bpw_est": scalar_bpw,
            "additive_index_bpw_est": additive_bpw,
            "codebooks": int(args.codebooks),
            "codebook_size": int(args.codebook_size),
        })

    return rows


def main() -> int:
    args = parse_args()
    if args.block_size <= 0 or args.codebook_size <= 0 or args.codebooks <= 0:
        raise SystemExit("block-size, codebook-size, and codebooks must be positive")

    try:
        load_runtime_deps()
    except ModuleNotFoundError as err:
        raise SystemExit(f"missing Python dependency: {err.name}. Install gguf-py requirements before analyzing a model.") from err

    reader = GGUFReader(args.model)
    rng = np.random.default_rng(args.seed)

    out = sys.stdout if args.output is None else args.output.open("w", encoding="utf-8")
    n_tensors = 0
    n_rows = 0

    try:
        for tensor in reader.tensors:
            if args.tensor and not any(pattern in tensor.name for pattern in args.tensor):
                continue

            data = tensor_data_f32(tensor)
            if data is None:
                continue

            rows = analyze_tensor(tensor, data, args, rng)
            if not rows:
                continue

            for row in rows:
                out.write(json.dumps(row, sort_keys=True) + "\n")
            n_tensors += 1
            n_rows += len(rows)

            if n_tensors >= args.max_tensors:
                break
    finally:
        if out is not sys.stdout:
            out.close()

    print(f"analyzed_tensors={n_tensors} analyzed_blocks={n_rows}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
