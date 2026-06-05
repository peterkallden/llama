# SpQR-guided mixed quantization POC

This POC explores whether more informed quantization decisions can improve the quality-to-size tradeoff of ordinary llama.cpp GGUF models without introducing a new runtime tensor format.

The central hypothesis is that uniform or mostly name-based quantization leaves useful quality on the table. Different tensors and transformer layers tolerate quantization differently, and calibration-time analysis can expose those differences before the model is written. The quantizer can then spend existing quantization types more deliberately: protect sensitive, unusual, or rare-event-heavy regions and compress redundant-looking regions more aggressively.

The work borrows selected ideas from several compression approaches:

- SpQR: use sensitivity and outlier-like signals to steer precision, without sparse outlier storage.
- MPEG-style compression: identify anchor/I-frame-like layers, detect scene changes, and use activity masking.
- Rate-distortion optimization: compare candidate quantization types by reconstruction error versus bits per weight.
- Two-pass encoding: an optional profile-enabled imatrix run acts as the first pass, collecting reusable calibration, weight-sample, RD, activity, and layer-delta analysis. `llama-quantize` acts as the second pass and uses that profile to choose existing tensor types.

The result is still a normal GGUF containing existing `ggml_type` tensor encodings. Inference does not reconstruct deltas, consult an imatrix, dynamically allocate bits, or require new CPU/GPU kernels.

## Purpose

The POC has three practical goals:

1. Determine which inexpensive analysis signals predict quantization tolerance.
2. Reuse those signals across repeated quantization experiments.
3. Evaluate mixed quantization policies while keeping storage, runtime, and removal cost low.

It intentionally separates measurement from policy. The optional imatrix quantization profile stores reusable raw statistics. `llama-quantize` interprets those statistics using adjustable policies and safety rules. This allows the same profile to support multiple base formats, RD lambdas, anchor thresholds, and future experiments.

This is a partial two-pass design rather than a hard requirement. With a profile-enabled imatrix, most expensive analysis is performed once during calibration and reused during quantization. With a standard imatrix or missing/incompatible profile entries, `llama-quantize` falls back to its existing local analysis, including sampling weight rows for RD candidate evaluation and directly comparing adjacent-layer weights. The fallback preserves compatibility but makes repeated quantization experiments slower and can produce slightly different choices from a precomputed profile.

## Current implementation

The branch currently implements:

- Opt-in `spqr_guided` mixed quantization using tensor-name heuristics and optional imatrix percentile sensitivity.
- Block-distribution sensitivity reports and block-derived tensor scoring.
- Adjacent-layer weight-delta analysis using relative delta norm and cosine similarity.
- P-frame-inspired `spqr_layer_delta` guidance that treats similar layers as more compression-friendly.
- Adaptive I-frame-style anchors for first, final, and robustly detected scene-change layers.
- Sampled rate-distortion candidate evaluation across the requested base type and compatible `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` types.
- An optional reusable quantization profile stored alongside normal imatrix data.
- Activity masking based on mean activity, variance, peak-to-mean ratio, and active-channel fraction.
- Robust scene-change detection combining layer delta, activity-profile changes, percentile filtering, and median absolute deviation.
- Conservative safety floors for token embeddings, output projection, and anchor layers.
- Reports for sensitivity, blocks, layer similarity, anchors, candidate RD costs, selected types, estimated size, and average bits per weight.

The reusable imatrix profile is optional. Standard imatrix files continue to work: sensitivity and block scoring use their normal activation data, while layer-delta and RD analysis fall back to being computed during quantization.

## Non-goals

This POC does not implement:

- SpQR sparse outlier storage or sparse correction kernels.
- A new GGUF runtime tensor encoding.
- Actual P-frame storage such as `layer N = layer N-1 + delta`.
- Dynamic precision selection during inference.
- New GPU kernels.
- Full AQLM or additive/codebook quantization.
- Global model-size-constrained bit allocation.

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

Enable adaptive I-frame-style anchors:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_layer_delta \
  --adaptive-anchors \
  --anchor-percentile 90 \
  --print-anchor-report \
  input-f16.gguf output-anchor-guided.gguf Q3_K_M
```

Adaptive anchors protect the first transformer layer, final transformer layer, and scene-change layers whose average adjacent-layer relative delta is at or above the configured percentile. High-sensitivity tensors in anchor layers receive at least `Q5_K`; other participating anchor tensors are not selected below `Q4_K`. Embeddings, output projection, norms, and tensors outside the transformer block layer-delta analysis are unaffected.

Enable sampled rate-distortion type selection:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_layer_delta \
  --adaptive-anchors \
  --rd-guided \
  --rd-lambda 0.002 \
  --rd-sample-rows 8 \
  input-f16.gguf output-rd-guided.gguf Q3_K_M
```

`--rd-guided` treats deterministically selected, evenly spaced tensor rows as analysis blocks while still choosing one existing GGUF type for the complete tensor. Block distortions are aggregated as `50% mean + 30% p90 + 20% worst`, which protects small difficult regions better than a single combined average. Q3 and Q5 establish a coarse curve; Q4 is evaluated when Q3 distortion or the Q3-to-Q5 improvement is meaningful, and Q6 is evaluated when Q5 distortion remains high. The requested base type is always evaluated by the local fallback. It selects the existing type with the lowest sampled cost:

```text
sensitivity_and_similarity_weight * normalized_reconstruction_error + rd_lambda * bits_per_weight
```

This remains a normal per-tensor GGUF mixture and requires no runtime changes. Token embeddings, output projection, and adaptive anchor tensors retain the existing safety policies instead of using the sampled selection. Use `--print-rd-report` to inspect every candidate. A dry run still reads tensor data and quantizes the sampled rows because estimated size alone cannot provide a distortion signal.

Optionally provide a soft global size target:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_layer_delta \
  --adaptive-anchors \
  --rd-guided \
  --rd-lambda 0.002 \
  --rd-target-bpw 3.8 \
  --print-rd-allocation-report \
  input-f16.gguf output-budget-guided.gguf Q3_K_M
```

`--rd-target-bpw` and `--rd-target-size-mib` are mutually exclusive and optional. Without either target, RD selection behaves as before. With a target, the allocator chooses one existing tensor type from each collected RD curve and searches for the lowest compression pressure that reaches the requested estimated tensor-payload size. `--rd-lambda` becomes the maximum permitted compression pressure: if the target would require more distortion, the output remains larger and the report marks it as quality-limited. GGUF metadata and alignment can also make the final file slightly larger than the MiB target. This candidate interface is the extension point through which future compression methods can participate in the same allocation.

Precompute reusable analysis while generating an imatrix:

```bash
./build/bin/llama-imatrix \
  -m input-f16.gguf \
  -f calibration-data.txt \
  --collect-quant-profile \
  --quant-profile-sample-rows 8 \
  --quant-profile-refine-top-k 16 \
  --quant-profile-refine-rows 32 \
  --quant-profile-block-size 256 \
  -o imatrix-profile.gguf
```

The optional profile stores namespaced GGUF tensors containing raw `Q3_K`/`Q4_K`/`Q5_K`/`Q6_K` candidate distortions, block importance summaries, sampled adjacent-layer delta metrics, and activity-mask statistics. Existing imatrix consumers ignore them and continue using the unchanged `.in_sum2` and `.counts` entries. `llama-quantize` automatically reuses compatible RD and layer-delta entries and falls back to its existing analysis when entries are absent, incompatible, or the requested base type is not represented.

Analysis profile version 5 adds selective high-fidelity RD refinement through the `rd_refinement` feature. The coarse pass uses `quant-profile-sample-rows` and adaptive candidate evaluation for every tensor. It ranks hotspots using tail distortion, Q3-to-Q5 curve gain, and activity risk, then reevaluates top-K tensors with all candidates over `quant-profile-refine-rows`. Refined curves replace their coarse curves before storage, while metadata records the refinement score and row counts. Version 2 through 4 profiles remain readable. Unsupported or malformed analysis entries are ignored without discarding the standard imatrix activation data.

Activity masking uses mean squared activity, cross-channel variance, peak-to-mean ratio, and active-channel fraction to create a bounded rare-event risk multiplier for RD distortion. This can protect tensors where average importance hides a small number of highly active channels. Adaptive anchors combine layer-delta with changes in activity statistics, then require scene-change candidates to pass both the configured percentile and a robust median-absolute-deviation threshold. The profile stores the raw signals; weighting and anchor selection remain adjustable quantization policy.

The profile does not currently survive combining imatrix files using `--in-file`, because candidate curves are model-bound rather than safely mergeable like activation sums. Generate the profile on the final calibration run.

The console report prints each quantized tensor's sensitivity bucket, score source, selected type, and estimated size. The summary includes high/medium/low tensor counts, total output size, average bits per weight, and how many tensors were promoted above the base quantization.

## Reproducible benchmark harness

`tools/quantize/spqr_benchmark.py` runs a fixed comparison matrix and records quantization time, output size, reported BPW, optional perplexity, and an optional generation sanity check. Its default baseline variants deliberately do not use an imatrix; guided variants use `--imatrix` when supplied, and the full RD variant prefers `--profile-imatrix`.

```bash
python tools/quantize/spqr_benchmark.py \
  --build-bin ./build/bin \
  --input-model input-f16.gguf \
  --output-dir benchmark-results \
  --imatrix imatrix.gguf \
  --profile-imatrix imatrix-profile.gguf \
  --rd-target-bpw 3.8 \
  --ppl-dataset wiki.test.raw
```

The harness writes per-stage logs, machine-readable `results.jsonl`, and a compact `summary.csv`. Use `--variants tools/quantize/spqr_benchmark.example.json` to customize the comparison matrix, `--resume` to reuse completed GGUF outputs, or `--dry-run` to inspect all generated commands without requiring model files or built executables.

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

6. Quantize using adaptive anchors:

```bash
./build/bin/llama-quantize --mixed-policy spqr_layer_delta --adaptive-anchors --print-anchor-report input-f16.gguf output-anchor-guided.gguf Q3_K_M
```

7. Quantize using sampled rate-distortion selection:

```bash
./build/bin/llama-quantize --mixed-policy spqr_layer_delta --adaptive-anchors --rd-guided --print-rd-report input-f16.gguf output-rd-guided.gguf Q3_K_M
```

8. If available, compare perplexity:

```bash
./build/bin/llama-perplexity -m output-q4-k-m.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-guided.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-layer-delta.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-spqr-block-guided.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-anchor-guided.gguf -f wiki.test.raw
./build/bin/llama-perplexity -m output-rd-guided.gguf -f wiki.test.raw
```

9. Run a few generation sanity checks:

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
