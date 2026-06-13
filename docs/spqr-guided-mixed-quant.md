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
- P-frame-inspired `--layer-delta-guidance` that measures adjacent-layer similarity and currently works primarily as a protection and anchoring signal.
- Adaptive I-frame-style anchors for first, final, and robustly detected scene-change layers.
- Sampled rate-distortion candidate evaluation across the requested base type and compatible `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K` types, with optional `IQ3_S` / `IQ3_M`-family probing for lower-bitrate experiments.
- An optional reusable quantization profile stored alongside normal imatrix data.
- Activity masking based on mean activity, variance, peak-to-mean ratio, and active-channel fraction.
- Robust scene-change detection combining layer delta, activity-profile changes, percentile filtering, and median absolute deviation.
- Conservative safety floors for token embeddings, output projection, and anchor layers.
- Reports for sensitivity, blocks, layer similarity, anchors, candidate RD costs, selected types, estimated size, and average bits per weight.

The reusable imatrix profile is optional. Standard imatrix files continue to work: sensitivity and block scoring use their normal activation data, while layer-delta and RD analysis fall back to being computed during quantization.

## Logit comparison side tool

The teacher-aware repair path inside `llama-quantize` is still a local proxy: it compares sampled FP rows against reconstructed quantized rows and uses that signal for mixed-precision decisions. A separate report-only tool, `llama-logit-compare`, can now be used when you want an actual teacher-vs-student output comparison without folding that work into imatrix generation.

This tool intentionally lives outside `llama-imatrix`. Imatrix remains a single-model analysis pass that produces reusable activation, RD, activity, and layer-delta profile data. `llama-logit-compare` is a two-model validation pass: it runs a teacher model and a student model over the same tokenized evaluation text and reports output drift metrics such as top-k KL, top-k overlap, rank drift, teacher margin, and argmax flip rate.

With `--layer-attribution`, the same pass also samples hidden-state tensors during decode and reports teacher-vs-student drift at three granularities:

- per tensor, for hotspot inspection such as `ffn_out-29`
- per family, for broader patterns such as `l_out` vs `ffn_out`
- per layer, so later quantizer decisions can attribute global logit damage back to a small set of suspect blocks

Example:

```bash
./build/bin/llama-logit-compare \
  --teacher teacher-f16.gguf \
  --student output-spqr-rd-repair.gguf \
  --layer-attribution \
  --layer-sample-tokens 8 \
  --logit-top-k 64 \
  --json-out logit-impact.json \
  -f eval.txt \
  -c 512 \
  --chunks 4
```

The first version is report-only. It does not yet change quantization choices directly, but it gives a concrete teacher/student signal that can later be fed into mixed-precision policy decisions.

For candidate-to-baseline comparisons, pass a baseline model as well:

```bash
./build/bin/llama-logit-compare \
  --teacher teacher-f16.gguf \
  --baseline standard-q4.gguf \
  --student output-spqr-rd-repair.gguf \
  --layer-attribution \
  --layer-sample-tokens 8 \
  --logit-top-k 64 \
  --json-out logit-delta.json \
  -f eval.txt \
  -c 512 \
  --chunks 4
```

In paired mode, `baseline_summary` and `candidate_summary` are both measured against the same teacher, and `delta` is reported as `candidate - baseline`. Lower KL, lower rank drift, fewer argmax flips, higher top-k overlap, and a less negative next-token logprob delta are better. `damage_score` is a first-pass aggregate where positive values mean the candidate looks worse than the baseline and negative values mean it looks better according to this lightweight logit gate. The score is intentionally conservative and report-only; use the component metrics when deciding whether a quantization path actually improved behavior.

When `--layer-attribution` is enabled, the JSON also includes `*_tensor_attribution`, `*_family_attribution`, and `*_layer_attribution`, plus paired `*_attribution_delta` arrays when a baseline is present. Those deltas are sorted by `delta_mean_mse`. `llama-quantize` currently consumes the paired tensor, family, and layer delta summaries as a local teacher-aware prior, while the full arrays remain useful for offline inspection and future allocator work.

`llama-quantize` can now consume that report as a first-step global gate:

```bash
./build/bin/llama-quantize \
  --imatrix imatrix-profile.gguf \
  --mixed-policy spqr_guided \
  --rd-guided \
  --quant-repair \
  --logit-report logit-delta.json \
  --logit-gate \
  input-f16.gguf output-q4.gguf Q4_K_M
```

If the report is paired and includes attribution deltas, `llama-quantize` now also uses it as a local prior inside the existing repair, demotion, budget-cap, and shrink passes. Tensor deltas are used first when a direct tensor-name or tensor-group match exists, and in the current implementation they are weighted more strongly than the broader layer and family fallback signals. The sign now matters more explicitly too: positive tensor deltas raise the local cost of demotion and can justify a one-step repair-time promotion in quality-first runs, while negative tensor deltas make the same tensor easier to compress in the later shrink and budget paths. Attribution coverage also matters: when the report includes comparison counts, those tensor/layer/family priors are scaled by a simple confidence estimate instead of being treated as equally certain. This is still not a full promotion auction, but it moves the current implementation from "global gate only" toward a lightweight teacher-aware allocator.

This first integration is still intentionally modest. The report is parsed once at startup and produces two effects:

- a model-level brake: if the paired or candidate-only metrics fail the configured thresholds, quant-repair, budget-first capping, budget shrink, and quality-validation demotion all become more conservative
- a tensor-local prior: when paired attribution deltas are available, tensor/layer/family drift are folded into the local repair, shrink, and budget scoring; positive tensor deltas act like a "protect or rescue" signal, while negative tensor deltas act like a "safe to compress" signal

Current knobs are `--logit-damage-threshold`, `--logit-kl-threshold`, and `--logit-flip-threshold`. The resulting pass/fail status is reported in the repair and RD summaries, and successful paired reports also show `tensor_deltas`, `layer_deltas`, `family_deltas`, and `local_allocator_prior=on` in the final quantization summary.

When the run is also budget-limited, `--logit-guided-search` enables a small second auction in the opposite direction: after the shrink pass reaches the target, the quantizer can spend back a capped amount of extra space on the highest-value teacher-attributed promotions. `--logit-search-top-k` limits how many ranked tensor opportunities are considered per pass, and `--logit-search-promote-budget-mib` controls the extra MiB ceiling above the original budget target. Setting that budget to `0` turns the pass into a budget-neutral search where every accepted promotion must be financed by additional safe demotions elsewhere.

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

Build `llama-imatrix` and `llama-quantize` as usual. The recommended path is a two-pass workflow: first collect calibration/profile data with `llama-imatrix`, then let `llama-quantize` run the full POC path: block scoring, adaptive anchors, RD-guided allocation, local RD refinement, and quant repair.

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

./build/bin/llama-quantize \
  --imatrix imatrix-profile.gguf \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --spqr-block-scoring \
  --adaptive-anchors \
  --rd-guided \
  --rd-local-refine-top-k 16 \
  --rd-local-refine-rows 32 \
  --quant-repair \
  --quant-repair-methods clipping,gain,scale \
  input-f16.gguf output-spqr-rd-repair.gguf Q4_K_M
```

In this flow, `--rd-guided` is the main allocation step: it chooses one existing tensor type from sampled candidate curves. `--rd-local-refine-top-k 16` enables the second-stage local RD refinement pass over the most uncertain tensors, and `--quant-repair` then acts as a repair-before-promote pass. By default it enables `clipping,gain,scale` and uses the teacher-aware repair depth when the needed proxy data can be computed. If `--quant-repair-methods` is supplied, the listed methods are used exactly, so omitting `scale` disables the scale sweep. Standard imatrix files are still accepted; missing profile data falls back to local quantizer-side sampling.

For lower-bitrate experiments, `--rd-include-iq3` opt-ins `IQ3_S` as an additional RD / repair candidate while still keeping the main K-ladder behavior unchanged by default. `IQ3_M` remains a recipe/output target rather than a single concrete tensor type, so this flag is mainly a way to probe whether the allocator starts preferring the `IQ3` family at all.

When `--rd-target-bpw` or `--rd-target-size-mib` is present, the repair flow now flips into a budget-first path: after bounded RD allocation, the quantizer first tries a one-step cheaper type cap for eligible tensors and only then runs `quant-repair` on the capped choice. This keeps the quality-first path unchanged when no explicit size target is set, while making target-driven runs behave more like "constrain then rescue" than "promote then trim".

With `--collect-quant-profile`, the imatrix profile also carries activity statistics. When those entries are present, `llama-quantize` uses them automatically during RD-guided scoring, anchor selection, and layer-delta guidance; there is no separate public `--activity-*` flag in this POC.

For a smaller sensitivity-only smoke test, run:

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
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --print-layer-delta-report \
  input-f16.gguf Q3_K_M
```

`--layer-delta-guidance` compares matching transformer block tensors in layer `N` against layer `N-1` and uses relative delta norm plus cosine similarity as an extra allocation signal on top of `spqr_guided`. It does not store deltas, does not make layers depend on earlier layers at inference time, and does not add a new GGUF tensor type. In the current POC, this signal has been more useful for deciding what to protect than for unlocking additional compression by itself, so it should be read as a conservative P-frame-inspired guidance pass rather than a delta-coding path. The old `--mixed-policy spqr_layer_delta` spelling is still accepted as a compatibility alias for `--mixed-policy spqr_guided --layer-delta-guidance`.

Enable adaptive I-frame-style anchors:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --adaptive-anchors \
  --anchor-percentile 90 \
  --print-anchor-report \
  input-f16.gguf output-anchor-guided.gguf Q3_K_M
```

Adaptive anchors protect the first transformer layer, final transformer layer, and scene-change layers whose average adjacent-layer relative delta is at or above the configured percentile. High-sensitivity tensors in anchor layers receive at least `Q5_K`; other participating anchor tensors are not selected below `Q4_K`. Embeddings, output projection, norms, and tensors outside the transformer block layer-delta analysis are unaffected.

Enable sampled rate-distortion type selection:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --adaptive-anchors \
  --rd-guided \
  --rd-lambda 0.002 \
  --rd-sample-rows 8 \
  input-f16.gguf output-rd-guided.gguf Q3_K_M
```

`--rd-guided` treats deterministically selected, evenly spaced tensor rows as analysis blocks while still choosing one existing GGUF type for the complete tensor. Block distortions are aggregated as `50% mean + 30% p90 + 20% worst`, which protects small difficult regions better than a single combined average. The primary K-quant candidate ladder is explicitly `Q3_K -> Q4_K -> Q5_K -> Q6_K`; Q3 and Q5 establish the coarse curve, Q4 is evaluated when Q3 distortion or the Q3-to-Q5 improvement is meaningful, and Q6 is evaluated when Q5 distortion remains high. The requested base type is also evaluated when it falls outside that ladder. With `--rd-include-iq3`, `IQ3_S` is also sampled as an extra low-bitrate candidate. It selects the existing type with the lowest sampled cost:

```text
sensitivity_and_similarity_weight * normalized_reconstruction_error + rd_lambda * bits_per_weight
```

This remains a normal per-tensor GGUF mixture and requires no runtime changes. Token embeddings, output projection, and adaptive anchor tensors retain the existing safety policies instead of using the sampled selection. Use `--print-rd-report` to inspect every candidate. A dry run still reads tensor data and quantizes the sampled rows because estimated size alone cannot provide a distortion signal.

When a requested K-quant is incompatible with a tensor shape, for example a 896-column row that is not divisible by 256, the SPQR/RD path now runs a small shape-aware scalar fallback check. It evaluates only safe scalar candidates with the same sampled, imatrix-weighted reconstruction metric and logs `scalar-fallback` lines in the RD report. The check is monotonic: it may keep or upgrade llama.cpp's existing fallback, but it never demotes `Q8_0` to `Q5_*` or `Q5_1` to `Q5_0`. Embeddings and output projection remain capped at `Q8_0` in this path. This is intentionally not a mixed-row or remainder format; it still writes one normal GGUF tensor type for the whole tensor.

After RD selection, enable quant repair as the repair-before-promote step:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --rd-guided \
  --quant-repair \
  --quant-repair-methods clipping,gain,scale \
  --quant-repair-accept-ratio 1.05 \
  --quant-repair-max-error 0.001 \
  input-f16.gguf output-spqr-rd-repair.gguf Q4_K_M
```

This is the main opt-in repair flag. It combines cheaper-candidate repair with exportable source-value repairs. The cheaper-candidate pass treats a selected tensor type as a candidate that can still be repaired downward if a cheaper compatible type measures as safe. It evaluates lower-rate candidates such as `Q6_K`, `Q5_K`, scalar `Q5_*`, `Q4_K`, scalar `Q4_*`, and `Q3_K` when their shape is compatible. With `--rd-include-iq3`, it also considers `IQ3_S` as an extra downward step. It reports weighted MSE, gain error, cosine/shape error, and outlier concentration. A cheaper candidate is accepted only if its composite error is close to the selected type or below the configured absolute weighted-error ceiling. Token embeddings and output projection remain protected.

The source-value repair side is used when the selected type is already low-bit, has high proxy error, or when a size target prevents simply promoting/upscaling tensors. In budget-limited runs using `--rd-target-bpw` or `--rd-target-size-mib`, repair becomes more permissive: a one-step budget-first type cap is attempted before the regular repair pass, cheaper-candidate repair uses looser acceptance gates, and the proxy-error threshold is lowered internally so the probe runs more often before the policy spends extra precision.

If the bounded RD pass and budget-first cap still leave the model above target, quant-repair now performs an extra shrink pass automatically. This pass now behaves as a small global demotion auction: every eligible tensor can contribute more than one cheaper candidate, those candidates are ranked globally by saved MiB versus estimated added quality cost, `proxy_safe` demotions are preferred first, and `repair_potential` demotions are admitted when the clipping/scale teacher-repair probe indicates the cheaper type can likely be rescued after demotion. When a teacher report is present, the bid score is also nudged by the local attribution prior: tensors with negative or low-confidence teacher pressure become easier to shrink, while tensors with stronger positive promotion pressure become more expensive to demote. This is the current "aggressive repair under hard budget" path.

The auction is budget-only. Quality-first runs without `--rd-target-bpw` or `--rd-target-size-mib` keep their earlier local validation behavior. In the console report, the shrink summary now includes `mode=auction`, the number of candidate bids considered, how many bids were selected, total MiB saved, total estimated quality cost, and `cost_per_mib` for the accepted demotions.

With `--logit-guided-search`, a follow-up `mode=buyback` summary is printed after shrink. This pass uses the same local teacher proxy and logit attribution priors to rank one-step promotions by estimated damage reduction per extra MiB, then tries to finance them with the best remaining demotion opportunities before consuming any optional extra MiB headroom. The summary now reports both the gross promotion spend and the financing demotions selected for it. The intent is to keep hard-budget runs from getting stuck in a one-way "only demote" loop once a few obviously harmful tensor choices have been identified.

For size-first exploration, add a compression opportunity report:

```bash
./build/bin/llama-quantize \
  --dry-run \
  --imatrix imatrix-profile.gguf \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --spqr-block-scoring \
  --adaptive-anchors \
  --rd-guided \
  --quant-repair \
  --print-compression-opportunity-report \
  input-f16.gguf Q4_K_M
```

This report is generated by `llama-quantize`, not by `llama-imatrix`, because it depends on the selected tensor type, shape fallback, budget mode, and repair results for the current run. It ranks compatible cheaper candidates by saved MiB per estimated extra composite error and labels them in two tiers:

- `proxy_safe`: the cheaper candidate already passes the same relaxed/safe gates used by repair.
- `repair_potential`: the cheaper candidate does not pass as-is, but the existing clipping/scale teacher-repair probe suggests it may become acceptable after repair.

It is intended as a dry-run planning aid for hard budget cases: instead of asking only which tensors need protection, it highlights where the model appears easiest to shrink next and which demotions may need extra repair work.

Repair methods can be controlled explicitly:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --rd-guided \
  --quant-repair \
  --quant-repair-methods clipping,scale \
  --quant-repair-min-error 0.002 \
  --quant-repair-min-improvement 0.05 \
  input-f16.gguf output-teacher-repaired.gguf Q3_K_M
```

`--quant-repair` defaults to `--quant-repair-depth teacher`. This means mixed-precision demotion/promotion gates use the teacher-aware proxy when it is available, then fall back to the local/basic gate for tensors or types where that proxy cannot be computed. Use `--quant-repair-depth basic` to force the older local-only behavior. The older `--quant-teacher-aware` flag remains as an alias for the teacher depth, while the `--quant-teacher-aware-*` flags are expert tuning knobs for the same path.

Repair depth and repair methods are separate:

| Option | Meaning |
| --- | --- |
| `--quant-repair-depth basic` | Uses the local tensor probe only. Candidate types are scored from sampled rows using weighted MSE, norm/gain drift, cosine/shape drift, and outlier concentration. This is enough for cheaper-candidate demotion and Q8-prevention decisions, but it does not add the teacher-aware block/rank proxy. |
| `--quant-repair-depth teacher` | Starts from the same basic probe, then blends in the teacher-aware proxy when it can be computed. The proxy compares sampled FP source rows against reconstructed candidate rows, optionally imatrix-weighted, and adds block-local cosine/norm drift, a small top-K rank/margin signal, and a lightweight feature gate. If the candidate type cannot be reconstructed with `to_float`, or if required imatrix data is missing for a type that needs it, the decision falls back to the basic score for that candidate. |

The method list controls which exportable repairs are attempted after a tensor/type has been selected:

| Method | Current behavior |
| --- | --- |
| `clipping` | Sweeps sampled absolute-value clipping thresholds at roughly p99.9, p99.5, p99.0, and p98.0, then requantizes the normal tensor type and accepts the result only if the proxy error improves enough. |
| `scale` | Sweeps small source multipliers before quantization. Standalone candidates are `0.970`, `0.985`, `0.995`, `1.005`, `1.015`, and `1.030`; when combined with clipping it tries the tighter set `0.985`, `0.995`, `1.005`, and `1.015`. For FFN tensors, the scale proxy also blends in a lightweight channel-importance profile from sampled activations and imatrix weights. |
| `gain` | Keeps the cheaper-candidate repair path and its gain diagnostics enabled. In that pass, gain drift is measured as relative row-norm drift and contributes to the composite candidate error. It is not a separate learned gain table or runtime-side correction; source-value multiplier sweeps are controlled by `scale`. |

This side is loosely OmniQuant-inspired because it tries cheap repairs before spending more bits, but it is not full OmniQuant. It does not learn clipping parameters, optimize equivalent transformations, or use runtime activation reconstruction. Instead, it treats the FP input tensor as a teacher, the selected quantized tensor as a student, and uses sampled imatrix-weighted reconstruction error as a diagonal layer-output proxy:

```text
E[||X(W - Wq)||^2] ~= sum_j imatrix[j] * (W[j] - Wq[j])^2
```

When the selected type has high proxy error, `clipping` sweeps a few clipping percentiles and `scale` sweeps small source multipliers before quantization. `gain` keeps the cheaper-candidate repair and gain-error diagnostics enabled. Accepted repairs are exportable because they are applied to the source values before writing the normal quantized tensor. No residual, codebook, learned transform, or runtime format is introduced. Without imatrix data, the pass falls back to an unweighted reconstruction proxy.

For FFN tensors, the `scale` path is slightly smarter than a plain global gain sweep. It now builds a lightweight channel-importance profile from sampled activations and imatrix weights, then blends that into the teacher proxy so high-activity FFN channels count more strongly. This gives the repair pass a cheap approximation of "preserve the useful FFN output channels first" without introducing a full block reconstruction pass.

The teacher repair depth pushes this one step further for mixed-precision decisions. It blends the local RD-style proxy with the output-aware teacher proxy when evaluating cheaper candidates, so demotion and bounded shrink decisions are judged less by raw weight error alone and more by an approximation of downstream behavior. `--quant-teacher-aware-mix` controls how strongly that teacher-side signal influences the gate and cost.

The teacher-aware gate also adds lightweight second-stage signals. A block-local gate measures cosine/norm drift between sampled teacher rows and reconstructed candidate rows, approximating hidden/output drift without a full forward pass. A local rank/margin proxy compares the top-K salient dimensions before and after quantization, giving a cheap stand-in for "did the important outputs keep their ordering and margin?" A small feature gate, inspired by Lillama's low-rank feature distillation objective, adds normalized L1, cosine, and norm drift over the same sampled teacher/candidate rows. This is only a PTQ decision signal: it does not introduce low-rank tensors, local gradient training, or a new GGUF/runtime representation. The block and rank terms are controlled by `--quant-teacher-aware-block-mix`, `--quant-teacher-aware-rank-mix`, and `--quant-teacher-aware-top-k`; the feature term is included in teacher depth with a fixed mild mix and is reported as `feature_mix`.

In quality-first runs without `--rd-target-bpw` or `--rd-target-size-mib`, teacher depth also runs a precision-validation pass. This is not trying to hit a smaller size target. Instead, expensive tensor choices are tested one step or more lower and kept demoted only when the teacher gate says the added damage is very small, either in absolute terms or per MiB saved. The report uses `precision-validation` lines with `teacher_delta`, `feature`, and `added_teacher_damage`. This borrows the useful part of the budget-forced path - extra precision must justify itself - without turning the run into a hard-budget search.

When a precomputed analysis profile is unavailable, the quantizer can selectively refine the most uncertain local curves before global allocation:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --rd-guided \
  --rd-sample-rows 8 \
  --rd-local-refine-top-k 16 \
  --rd-local-refine-rows 32 \
  --print-rd-refinement-report \
  input-f16.gguf output-locally-refined.gguf Q3_K_M
```

The coarse pass ranks locally sampled curves using Q3 tail distortion, Q3-to-Q5 improvement, and activity risk. The top-K tensors are then loaded again and evaluated over more rows with Q3, Q4, Q5, Q6, and the requested base type before any soft size target is allocated. Compatible curves loaded from an imatrix analysis profile are trusted and never reevaluated. This quantizer-side fallback keeps the feature usable without a profile, but profile-side refinement is more efficient because imatrix generation already retains the selected weight rows.

Optionally provide a soft global size target:

```bash
./build/bin/llama-quantize \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --adaptive-anchors \
  --rd-guided \
  --rd-lambda 0.002 \
  --rd-target-bpw 3.8 \
  --print-rd-allocation-report \
  input-f16.gguf output-budget-guided.gguf Q3_K_M
```

`--rd-target-bpw` and `--rd-target-size-mib` are mutually exclusive and optional. Without either target, RD selection behaves as before. With a target, the allocator uses a bounded budget pass: it chooses one existing tensor type from each collected RD curve and increases compression pressure until the requested estimated tensor-payload size is reached, as long as the available candidates and safety floors allow it. If the target is still unreachable after every allocatable tensor has moved to its smallest allowed candidate, the report marks the run as `bounded-limit`. The report also prints an estimated quality cost, measured as the additional weighted distortion versus the highest-quality RD allocation. GGUF metadata and alignment can make the final file slightly larger than the tensor-payload target. This candidate interface is the extension point through which future compression methods can participate in the same allocation.

Budget-limited runs also apply a mild bottom-first prior when scores are otherwise close: earlier transformer layers get a small compression bias and later layers get a small protection bias. This is inspired by feature-distillation compression work such as [Lillama](https://arxiv.org/pdf/2412.16719), where local feature matching makes layer-wise compression/recovery practical. In this POC it remains only a soft ranking multiplier (`0.94..1.08`) and is reported as `bottom_first_bias`; quality-first runs without a requested size target do not use it.

A practical budget-first teacher-aware run looks like this:

```bash
./build/bin/llama-quantize \
  --imatrix imatrix-profile.gguf \
  --mixed-policy spqr_guided \
  --layer-delta-guidance \
  --spqr-block-scoring \
  --adaptive-anchors \
  --rd-guided \
  --rd-local-refine-top-k 16 \
  --rd-local-refine-rows 32 \
  --rd-target-bpw 4.4 \
  --quant-repair \
  --logit-report logit-delta.json \
  --logit-gate \
  input-f16.gguf output-budget-q4.gguf Q4_K_M
```

In this mode the bounded RD allocator chooses the first budget-feasible profile, then `quant-repair` and the teacher-aware local prior spend the remaining quality budget through a global demotion auction rather than treating every later demotion equally. The budget phase is still one-way today: it is a budget rescue auction over cheaper candidates, not yet a full promote/demote market over the whole ladder, even though the quality-first repair pass can now use positive tensor deltas to justify a limited one-step promotion. In the report this now shows up as `safe_pressure`, `promotion_pressure`, and `budget_bias` on the ranked shrink opportunities, which makes it easier to see why one demotion bid beat another.

The POC is usually run from F16/BF16/F32 GGUF inputs, but it can also analyze and requantize an already-quantized source when the source ggml type exposes a `to_float` converter. In that case the quantizer logs the source type and automatically allows requantization for that run. If a quantized or non-floating source type cannot be converted back to float, the run fails early with an explicit error instead of silently producing an invalid profile or output.

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
./build/bin/llama-quantize --mixed-policy spqr_guided --layer-delta-guidance --print-layer-delta-report input-f16.gguf output-spqr-layer-delta.gguf Q3_K_M
```

5. Quantize using block-derived sensitivity:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided --imatrix imatrix.gguf --spqr-block-scoring input-f16.gguf output-spqr-block-guided.gguf Q3_K_M
```

6. Quantize using adaptive anchors:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided --adaptive-anchors --print-anchor-report input-f16.gguf output-anchor-guided.gguf Q3_K_M
```

7. Quantize using sampled rate-distortion selection:

```bash
./build/bin/llama-quantize --mixed-policy spqr_guided --layer-delta-guidance --adaptive-anchors --rd-guided --print-rd-report input-f16.gguf output-rd-guided.gguf Q3_K_M
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
