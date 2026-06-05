# SpQR-guided mixed quantization POC

This is an experimental quantization policy for `llama-quantize`. It borrows one idea from SpQR: use a sensitivity signal to decide which weights should receive more bits. It does not implement SpQR sparse outlier storage, sparse correction kernels, new runtime tensor formats, or AQLM/codebook quantization.

## Purpose

The policy starts from an existing high-precision GGUF model and writes a normal GGUF that uses existing `ggml_type` encodings. Sensitive tensors are promoted to higher precision, while less sensitive tensors stay at the requested base quantization.

The first implementation is tensor-level:

- high sensitivity: `Q5_K`, or `Q6_K` for token embeddings and output tensors
- medium sensitivity: `Q4_K`
- low sensitivity: the requested base quantization type
- token embeddings and output projection are not selected below `Q4_K` by this policy

## Sensitivity scoring

When an imatrix is supplied, the policy aggregates each tensor's importance values into a mean absolute score and assigns high, medium, and low buckets by percentile. Tensors without matching imatrix data fall back to name/category heuristics.

Without an imatrix, the policy uses conservative tensor-name defaults:

- high: `output.weight`, `token_embd.weight`, attention output projection, `ffn_down`
- medium: attention q/k/v, fused attention qkv/kv-b, `ffn_gate`, `ffn_up`
- low: other quantizable tensors

## Usage

Build `llama-quantize` as usual, then run:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided input-f16.gguf output-spqr-guided.gguf Q3_K_M
```

With imatrix data:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided --imatrix imatrix.gguf input-f16.gguf output-spqr-guided.gguf Q3_K_M
```

Use `--dry-run` to inspect tensor choices and estimated size without writing the output:

```bash
./build/bin/llama-quantize --dry-run --mixed-policy spqr_guided input-f16.gguf Q3_K_M
```

Enable report-only block sensitivity counts:

```bash
./build/bin/llama-quantize --dry-run --mixed-policy spqr_guided --spqr-block-report input-f16.gguf Q3_K_M
```

With imatrix data, `--spqr-block-report` buckets imatrix value blocks by percentile. Without imatrix data, it reports the tensor-level bucket as a one-block fallback. The flag does not change the output GGUF.

Use block distributions to drive tensor quantization choices:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided \
  --imatrix imatrix.gguf \
  --spqr-block-scoring \
  input-f16.gguf output-spqr-block-guided.gguf Q3_K_M
```

`--spqr-block-scoring` replaces the tensor-average imatrix bucket with a bucket derived from the tensor's block distribution. These v1 blocks are contiguous groups of imatrix input-dimension values, not independently encoded weight blocks. A tensor is protected when a meaningful fraction of its blocks are highly sensitive. The selected quant type still applies to the complete tensor because this POC does not introduce a mixed-type block encoding. If insufficient imatrix block data is available, the tool keeps tensor-level scoring.

Enable P-frame-style adjacent-layer analysis:

```bash
./build/bin/llama-quantize --dry-run \
  --mixed-policy spqr_layer_delta \
  --print-layer-delta-report \
  input-f16.gguf Q3_K_M
```

`spqr_layer_delta` compares matching transformer block tensors in layer `N` against layer `N-1` and uses relative delta norm plus cosine similarity as an extra compression signal. It does not store deltas, does not make layers depend on earlier layers at inference time, and does not add a new GGUF tensor type.

The console report prints each quantized tensor's sensitivity bucket, score source, selected type, and estimated size. The summary includes high/medium/low tensor counts, total output size, average bits per weight, and how many tensors were promoted above the base quantization.

## Validation sequence

1. Convert or obtain a small F16/BF16 GGUF model.
2. Quantize a baseline:

```bash
./build/bin/llama-quantize input-f16.gguf output-q4-k-m.gguf Q4_K_M
```

3. Quantize with the POC policy:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided input-f16.gguf output-spqr-guided.gguf Q3_K_M
```

4. Quantize with SpQR plus layer-delta guidance:

```bash
./build/bin/llama-quantize --mixed-policy spqr_layer_delta --print-layer-delta-report input-f16.gguf output-spqr-layer-delta.gguf Q3_K_M
```

5. Quantize using block-derived sensitivity:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided --imatrix imatrix.gguf --spqr-block-scoring input-f16.gguf output-spqr-block-guided.gguf Q3_K_M
```

6. If available, compare perplexity:

```bash
./build/bin/llama-perplexity -m output-q4-k-m.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-guided.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-layer-delta.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-block-guided.gguf -f wiki.test.raw
```

7. Run a few generation sanity checks:

```bash
./build/bin/llama-cli -m output-spqr-guided.gguf -p "Write a short explanation of quantization." -n 64
./build/bin/llama-cli -m output-spqr-layer-delta.gguf -p "Write a short explanation of quantization." -n 64
```

## Limitations and next steps

This POC is intentionally easy to remove or replace. The natural next step is to replace tensor-level scoring with block-level scoring:

```text
sensitivity_score(tensor_name, block_id) -> bucket
```

The quantization path is also structured so future experiments can introduce a backend such as:

```text
quantize_block_with_policy(block, policy, sensitivity_score)
```

That future backend could explore additive/codebook quantization, but no AQLM implementation is included here.

Layer-delta guidance is also report-only with respect to representation. A future format could store:

```text
anchor layer stored normally
following layer stored as quantized delta
```

That would require GGUF metadata changes and runtime reconstruction support. This POC avoids that and only uses adjacent-layer similarity as a quantization policy signal.

## Offline codebook candidate analysis

`tools/quantize/spqr_codebook_candidate.py` is a report-only helper for early AQLM-style experiments. It reads F16/F32 GGUF tensors, samples blocks, and compares simple scalar k-means reconstruction error against a tiny additive residual codebook model.

Example:

```bash
python tools/quantize/spqr_codebook_candidate.py input-f16.gguf \
  --tensor ffn_down \
  --block-size 256 \
  --codebooks 2 \
  --codebook-size 16 \
  --output codebook-candidates.jsonl
```

The output is JSONL with per-block reconstruction error and a rough index bits-per-weight estimate. It is not a production quantizer and does not write GGUF files.
