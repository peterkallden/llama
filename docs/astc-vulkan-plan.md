# ASTC Vulkan Experimental Implementation Plan

This plan implements the research direction in
[`astc-vulkan-research.md`](astc-vulkan-research.md) while keeping the change
small, optional, and compatible with an eventual upstream review.

The proof of concept starts in `pocs/astc-vulkan/`, parallel to the production
`ggml-vulkan` backend. It does not copy the production backend. Integration
into `ggml-vulkan` is a later decision gate after device-backed evidence.

## Engineering principles

- Do not change `ggml` core semantics or existing quantization types in the
  first implementation.
- Treat ASTC as an optional Vulkan representation selected only when the
  required image format and sampled-image features are present.
- Keep the existing buffer-backed Vulkan kernels as the mandatory fallback.
- Reuse existing ggml Vulkan conventions for naming, error handling, resource
  lifetime, shader generation, pipeline creation, and tests.
- Keep each commit narrowly scoped and independently reviewable.
- Use English for code, tests, diagnostics, and documentation.

## Phase 1: contracts and standalone benchmark

1. [x] Add host-neutral format contracts and focused CTest targets.
2. [x] Add a small standalone Vulkan test utility, outside the normal inference
   path.
3. [x] Query `VK_FORMAT_ASTC_4x4_UNORM_BLOCK`,
   `VK_FORMAT_ASTC_5x5_UNORM_BLOCK`, and `VK_FORMAT_ASTC_6x6_UNORM_BLOCK`
   for optimal-tiling sampled-image support.
4. [x] Verify that ASTC 4x4, 5x5, and 6x6 images can be allocated, uploaded, and
   transitioned for shader reads on a compatible device.
5. [x] Compile the minimal `texelFetch` validation shader when `glslc` is
   available.
6. [x] Upload a known-valid ASTC block and verify `texelFetch` results in a
   compute shader against a CPU reference when a compatible device is present.
7. [x] Implement sequential and deliberately non-local fetch patterns with
   Vulkan timestamp queries and collect an initial repeated sample.
8. [x] Record device name, driver version, supported formats, shader workgroup
   shape, elapsed GPU time, and represented bytes for the first host run.

Current status: capability, resource allocation/upload, shader compilation,
device execution/readback, and initial access-pattern instrumentation are
implemented and validated on the Intel UHD Graphics 620. CTest still skips
these tests on hosts that expose no physical ASTC-capable device.

Exit criterion: the selected target GPU accepts the selected formats as sampled images
and the benchmark produces deterministic, validated values.

Phase 1 result: met on Intel UHD Graphics 620 with Mesa 26.0.8. The benchmark
harness is deterministic for correctness and now supports scaled repeated
dispatches; its timing data is still exploratory and not a performance claim.

## Phase 2: offline ASTC-aware weight packer

1. [x] Define an experimental input tensor layout and metadata record; do not
   add it to the public GGUF specification.
2. [x] Add a host-only baseline encoder adapter that exercises 4x4, 5x5, and 6x6
   through the external astcenc library.
3. [x] Add a host-only tensor-to-RGBA staging and roundtrip smoke for 4x4, 5x5,
   and 6x6 using the same external encoder boundary.
4. Implement 4x4, 5x5, and 6x6 tensor packing paths.
5. Encode static weight blocks offline and preserve any scale, offset, outlier,
   or layout metadata needed by the shader in a companion buffer.
6. [x] Collect an initial elementwise and dot-product error measurement against
   the uncompressed host tensor.
7. Evaluate error distribution per block versus FP16/FP32 and an existing
   low-bit reference.
8. [x] Evaluate all 24 RGBA channel assignments against the initial objective.
9. Iterate on spatial block layout and tensor permutation only when it improves
   the measured objective.
10. [x] Add an activation-weighted neural objective so candidate layouts and
    channel assignments are scored on matrix-vector error as well as MSE.
11. [x] Evaluate `astcenc` quality presets as a first encoder-candidate search,
    using Basis Universal's search methodology while keeping the final
    representation standard ASTC blocks.
12. Evaluate deeper endpoint, partition, and rate-distortion candidates from
    Basis Universal as a research reference; do not add cross-block state to
    the GPU-resident representation.
13. [x] Report per-block reconstruction error so outlier blocks are visible
    independently of the global objective.
14. [x] Include a conservative block-tail penalty in candidate ranking while
    keeping its weight explicit for later calibration.
15. [x] Add a larger host fixture and CTest target so block-error percentiles
    are exercised beyond the tiny smoke matrix.
16. [x] Include smooth, pseudo-random, and sparse/outlier activation samples so
    candidate ranking is not tuned to one activation shape.
17. [x] Evaluate a block-local texel permutation and require it to win the
    measured objective per fixture before selecting it.
18. [x] Normalize activation loss by reference output energy so activation
    samples with different norms are comparable during candidate search.
19. [x] Evaluate per-block affine scale/offset reconstruction and report its
    metadata overhead separately from compressed ASTC bytes.
20. [x] Evaluate per-channel block affine calibration; retain it only as a
    measured negative candidate because of its quality/metadata tradeoff.
21. [x] Define a versioned offline pack artifact and companion metadata
    contract containing format, layout/permutation, mapping mode, encoder
    quality, and block calibration fields.
22. [x] Add an input path for real GGUF weight slices and representative
    activation traces; keep the fixture path for deterministic regression
    tests. The trace reader/writer is versioned and the capture tool obtains
    attention-layer inputs from a real SmolLM2 F16 inference pass. A prompt
    suite and other projection sites are still required before a layer-wide
    quality claim.
23. [x] Compare ASTC candidates against FP16 and the closest existing low-bit
    reference with explicit acceptance thresholds for relative matvec error.
    The first real-layer probe compares both ASTC block sizes and Q4_0; the
    threshold is intentionally a provisional Phase-2 gate until layer-wide
    traces are available.
24. [x] Evaluate a sparse residual/outlier sidecar and include its bytes in the
    pressure-aware storage decision. The first probe uses a fixed 1% top-error
    budget and reports index/value overhead explicitly.
25. [x] Evaluate a structured low-rank residual sidecar against sparse outliers,
    including its F16 storage cost and activation-weighted error at several
    ranks. Keep it as a research candidate until the multi-layer gate passes.
26. Implement the first reproducible offline packer artifact only after the
    preceding quality and storage gates pass.
27. [x] Validate packed channel-order metadata as an actual four-channel
    permutation before any artifact can be consumed.
28. [x] Select a matched small-model baseline for end-to-end checks. The
    `shibatch/tinybpe1m` F16 and Q4_0 artifacts provide the compact smoke
    fixture. Its files named TQ1_0/TQ2_0 were audited and are not true TQ
    tensor artifacts, so actual TQ tests use locally quantized 256-aligned
    SmolLM2 FFn-down matrices. Keep model binaries out of git; record their
    source and checksums in the WIP log.
29. [x] Evaluate channel semantics independently of block footprint: four
    independent scalar channels, two coarse-only pairs, two coarse-plus-
    residual pairs, and an RG16 high/low-byte control. Treat high/low bytes as
    a robustness control rather than a storage proposal; retain a residual
    only if its post-ASTC error improves the measured activation objective.
30. [x] Add a channel-layout search over the three canonical pairs and all 24
    ordered coarse/residual channel permutations. Keep the regression fixture
    small, and permit an explicit F16 GGUF tensor as the input for expensive
    real-layer sweeps.
31. [x] Add a one-weight-per-texel luminance-plus-alpha baseline which reports
    the *actual* dual-plane decisions made by the standard encoder. This keeps
    a learned/additive two-latent experiment distinct from the rejected
    two-independent-weights-per-texel residual layout.
32. Evaluate a calibrated two-latent decoder, `w = s_L L' + s_A A' + b`, with
    one scalar per texel. Include the three scalar decoder parameters in byte
    pressure and compare it to the scalar ASTC baseline on held-out
    activations.
33. [x] Add an ASTC block-mode audit to distinguish a L+A source image from blocks
    that actually use dual-plane, and report the second-plane component. Do
    not claim dual-plane benefits unless the encoded block stream demonstrates
    them.
34. Investigate a constrained offline encoder or encoder search only if the
    unconstrained audit shows that `astcenc` rarely selects the alpha plane.
    Vulkan sampling cannot request a block mode or BISE alphabet at runtime.
35. Prototype codec-aware optimization with a differentiable ASTC surrogate
    and exact `astcenc` projection/evaluation. Begin with alternating
    optimization of the latent fields and decoder parameters; assess
    PV-Tuning-style representation optimization only against held-out
    activation or model loss.
36. Evaluate AQLM-inspired learned additive codebooks only after the direct
    two-latent baseline is characterized. This is an analogy, not a claim that
    an ASTC texture automatically implements AQLM's vector codebooks.
37. [x] Add a structured block-residual control to test whether a low-frequency
    second latent is more ASTC-compatible than an independent per-value tail.
38. [x] Add a small codec-aware projection-level sweep. Keep the exact
    `astcenc` roundtrip as the acceptance oracle and report the result for each
    coarse alphabet.
39. Implement a small codec-aware optimizer/projection loop. Keep the exact
    `astcenc` roundtrip as the acceptance oracle and report held-out activation
    error after every projection step.
40. Add a model-loss or calibration-data entry point only after the optimizer
    reproduces the scalar and structured controls. Do not introduce a training
    dependency into the normal llama.cpp build for this research stage.
41. [x] Add optional activation-trace input to the latent smoke so projection
    candidates can be scored on held-out layer inputs.
42. [x] Add an AQLM-inspired row/column additive control, `W ~= row + column`,
    to separate spatially structured additive semantics from a residual tail.
43. [x] Split projection scoring into calibration and holdout activation traces;
    report the selected coarse alphabet and its held-out error.
44. [x] Decide the encoder boundary: retain standard `astcenc` for baselines,
    and treat a constrained `astcenc` fork as an offline research component,
    never as a Vulkan runtime decoder.
45. [x] Define a constrained encoder experiment that can prefer or restrict legal
    ASTC block modes (L+A endpoint modes, dual-plane alpha, weight grids, and
    BISE alphabets), while reusing the standard bit packing and reference
    decoder. The first implementation probes the public astcenc search knobs;
    a source fork is still required for exact mode restrictions. Validate every
    emitted block with `astcenc_get_block_info` and CPU decode.
46. Add a fallback-preserving encoder adapter so a missing or experimental
    encoder cannot affect normal llama.cpp builds or silently emit a
    non-standard block stream.
47. [x] Add a minimal block-policy prepacker as a deliberately bounded Track-3
    control. It reduces each ASTC footprint to a constant signal but delegates
    final legal bit packing to standard `astcenc`; do not present it as a custom
    ASTC bitstream implementation.
48. [x] Add Track 4, an opt-in late candidate-ranking metric in a side
    `astcenc` fork. Preserve the encoder's normal candidate generation, legal
    ASTC packing, and standard decoder; re-rank final L+A candidates by
    reconstructed weight MSE rather than RGBA MSE.
49. Add activation-aware Track-4 scoring using captured calibration traces.
    Start with a diagonal activation-energy approximation, then validate exact
    incremental `||X(W - W_hat)^T||_F^2` scoring, including cross terms, on a
    held-out trace. Do not move the metric into a Vulkan shader or normal
    llama.cpp build.
50. Treat activation-aware ASTC selection as **error shaping**: minimize
    `tr((W - W_hat) H (W - W_hat)^T)`, with `H = E[x x^T]`, rather than raw
    ASTC or weight MSE. Log each retained candidate's ASTC MSE, reconstructed
    weight MSE, activation loss, endpoint mode, partition count, weight grid,
    and dual-plane component.
51. Re-run all L+A/latent experiments under the Track-4 metric and the common
    calibration/holdout traces. Preserve prior standard-ASTC scalar results as
    their own baseline cohort; they are unaffected because the experimental
    metric is opt-in. Re-evaluate true TQ1_0 and TQ2_0 separately on the same
    256-aligned real layers and traces, across more than one projection, before
    using either as an ASTC-Q quality or throughput control.
52. [x] Make every encoder-ranked latent representation specify its runtime
    affine decoder before encoding. Do not fit `s_L`, `s_A`, or `b` after
    decode when using those parameters in an ASTC candidate score. Add an
    isolated llama PoC build that can link the side-fork encoder only when
    explicitly supplied with its static library and headers.
53. [x] Run the scalar Track-4 weight-aware control under the same fixed
    decoder, model tensor, and holdout trace. Treat matching standard output
    as the expected result for replicated RGBA scalar storage; a scalar gain
    requires the subsequent activation-aware global selector.
54. [x] Implement a side-fork synthetic coordinate-descent contract over a
    block-local shortlist of legal ASTC candidates. Maintain the exact cached
    layer residual, run forward and reverse sweeps, and verify that concatenated
    selected 128-bit blocks decode identically to the chosen mixed matrix.
55. [x] Extend coordinate selection to a real-layer offline adapter. Start with a
    bounded pool from complete legal encodes (quality presets and mapping
    metrics), then expose a richer internal astcenc shortlist only if that pool
    demonstrates a calibration/holdout improvement.
56. [x] Expose a bounded per-block shortlist from the experimental encoder while
    preserving ordinary ASTC candidate generation, legal bit packing, and CPU
    decode. A candidate record must include decoded weight error, transformed
    calibration-output delta, ASTC mode diagnostics, and exact 16-byte payload.
57. [x] Select shortlists for **both** low local activation loss and diversity of
    activation-weighted error direction. Start from the best candidate and add
    candidates by farthest-point or clustering distance in `E_c L`, where
    `H_I = L L^T`; retain the ordinary lowest-loss candidate as a control.
58. [x] Re-run scalar ASTC 4x4, 5x5, and 6x6 with that candidate pool, exact
    cached-residual coordinate selection, nested calibration/validation splits,
    and an untouched holdout. Add an explicit penalty for replacing the
    uniform baseline so a larger pool cannot silently overfit calibration.
59. [~] Prototype Hessian-guided **block error feedback** as an offline encoder
    alternative. After locking a legal ASTC block, project its error into a
    low-rank input-sensitivity basis and modify only targets of later blocks.
    First compare it with exact coordinate descent on small fixtures; it must
    beat a matched greedy baseline on holdout before a real-layer run.
59a. On one fixed legal ASTC candidate pool, compare exactly three selectors:
     local neural ranking, cached-residual coordinate descent, and Hessian
     feedback. Coordinate descent is the offline oracle-ish reference; the
     first feedback test may only choose from the fixed pool, so candidate
     generation cannot confound selector quality.
59b. Report recovered coordinate-descent gain when the denominator is positive:
     `G_HF = (L_local - L_HF) / (L_local - L_CD)`. Also report all three
     absolute losses, since the ratio is undefined when local and coordinate
     descent tie and can be negative or exceed one on a finite holdout.
59c. Log an error-cancellation budget for every accepted block decision:
     residual norm before/after and `cos(R, DeltaY)`. Since `R_next = R -
     DeltaY`, positive alignment identifies a direct residual-reducing choice.
     Keep forward and reverse coordinate sweeps separate in the log.
59d. Measure candidate diversity in the activation-weighted error space, not
     RGBA space. Use `D_c = E_c X^T` directly, or a thin QR/SVD basis of the
     calibration trace when `H_I = X^T X` is rank-deficient. Retain local-best,
     low-correlation, negative-correlation, and principal-direction candidates
     only when each has an explicit activation-loss bound.
59e. Only after the fixed-pool comparison is interpretable, allow Hessian
     feedback to alter the source target for later ASTC blocks and regenerate
     their legal candidates. Report this stronger encoder jointly with the
     fixed-pool result; it is not a like-for-like selector comparison.
59f. Add a reproducible `--selector-compare` smoke that runs all three fixed-pool
     selectors on the same small fixture. Treat the first parallel Jacobi
     feedback implementation as a diagnostic baseline: it must not be called a
     Hessian win unless it improves holdout loss and reports positive recovered
     coordinate-descent gain.
60. Add a two-sided objective only after input-only shaping is stable. Capture
    output sensitivity separately and evaluate `tr(H_O E H_I E^T)` first with
    a diagonal `H_O`, then a bounded Kronecker-factored approximation. A plain
    layer-output trace has `H_O = I`; it is not a full-model Hessian.
61. Test a projected-residual beam/trellis selector only if a diverse shortlist
    and Hessian feedback both show holdout gains. Its state must be a bounded
    4--16-dimensional sensitivity projection, and its emitted result remains
    independently decodable standard ASTC blocks.
62. Treat layout as part of error shaping: test zero-runtime-cost row/column
    permutations and sign flips before rotations. The WHT64 result is a
    negative control, so dense rotations require a separate quality gain large
    enough to pay for fusion and metadata.
63. Revisit L+A only after scalar error shaping has a holdout improvement.
    Test the invariant redundant latent family `L <- L + alpha R`,
    `A <- A - (s_L/s_A) alpha R`, which preserves the pre-codec reconstructed
    weight but may give ASTC a more favorable signal.
64. Consider ASTC-noise-aware adapter tuning or codec-aware fine-tuning only
    after the post-training tracks above are measured. Inject exact projected
    ASTC decode error rather than generic Gaussian noise, preserve the normal
    llama.cpp build, and gate it on model-level validation.
65. Add a scaling/generalization gate before interpreting any full-layer
    selector result. A 32x64 probe has 2,048 weights, whereas the 576x576
    attention matrix has 331,776 weights and thousands of selectable ASTC
    blocks. Report the number of blocks, candidates per block, calibration
    samples, validation samples, and held-out samples for every result.
66. Scale calibration evidence with selector freedom. Split prompts or token
    positions into selection, validation, and untouched holdout cohorts; do
    not use a larger matrix as a reason to reuse the holdout or tune directly
    on it. Repeat on attention and FFN projections, because their activation
    covariance and useful ASTC layout may differ materially.
67. Report the offline search budget separately from runtime. Candidate
    generation, covariance sketches, coordinate sweeps, and feedback are
    allowed to be expensive at pack time, but memory must remain bounded by
    streaming block candidates and a compact sensitivity basis. Runtime remains
    a standard ASTC image fetch plus the fixed reconstruction only.
68. Require a full-layer result to beat its uniform scalar or neural ASTC
    baseline on a predeclared holdout before attributing any gain to block
    cooperation. More blocks create more opportunities for cancellation only
    when their legal candidate errors span useful sensitivity directions; they
    also create more degrees of freedom for overfit.

The packer must optimize a numerical objective. A generic image compressor is
useful as an initial baseline but is not assumed to be optimal for neural
weights.

The first implementation should invoke an explicitly versioned external
ASTC encoder (for example `astcenc`) behind a host-side adapter. If the encoder
is unavailable, the adapter must skip with a clear diagnostic; it must not
silently produce a non-standard or incomplete ASTC bitstream.

Exit criterion: every selected format can represent a small known matrix and report
reconstruction and dot-product error reproducibly.

## Research track: ASTC-native few-level weights

This is a parallel research track, not a replacement for the Q4 comparison.
Its objective is not the smallest possible model file. It asks whether standard
ASTC sampled-image decoding can deliver a useful few-level weight
representation to the shader ALU with lower runtime pressure than a packed
low-bit buffer plus shader-side unpack/dequantization.

The first design uses **one scalar weight per ASTC texel**. Therefore a 6x6
block provides `128 / 36 = 3.56` physical bits per weight before any external
metadata. The existing four-scalars-per-RGBA-texel packing must not be used to
claim this rate: it would provide only 0.89 physical bits per scalar and cannot
losslessly represent independently chosen ternary or five-level weights. RGBA
packing remains a separate structured-compression candidate.

The representation family is called **ASTC-Q** provisionally. Candidate weight
alphabets are:

| Candidate | Target alphabet | Initial ASTC layouts | Purpose |
|---|---|---|---|
| ASTC-QT | `{-1, 0, +1}` | 4x4, 5x5, 6x6 | ternary semantic baseline |
| ASTC-Q5 | five symmetric or learned affine levels | 5x5, 6x6 | few-level/BitNet-adjacent candidate |
| ASTC-Q8 | eight levels | 4x4, 5x5, 6x6 | approximately Q3-like comparison |
| ASTC-Q16+ | sixteen or more levels | 4x4, 5x5, 6x6 | rate-distortion ladder |

An ASTC block may use asymmetric local endpoints, partitions, and a separate
per-block scale only when their resident bytes are included in the comparison.
Endpoint interpolation is not assumed to produce exact desired levels; this is
an empirical representability question for legal standard ASTC blocks and the
target driver. Larger ASTC footprints (8x8 through 12x12) are deferred until
their sampled-image support is queried per Vulkan device.

### Three staged experiments

1. **Representation microbenchmark.** Encode synthetic ternary and few-level
   block patterns with standard `astcenc`; measure exact/near-level recovery,
   level drift, block error, byte pressure, and sampled shader values. Include
   smooth, clustered, random, and outlier patterns. Compare one scalar/texel
   with the existing RGBA structured layout without conflating their rates.
2. **Post-training weight experiment.** [initial smoke complete] Apply
   ASTC-QT/Q5/Q8/Q16 to a small existing ternary or few-level model matrix.
   Compare quality and bytes with a conventional packed ternary/few-level
   representation and with Q4. The runtime comparison must include shader
   unpack/dequant instructions, not merely compressed storage size. The first
   SmolLM2 projection fails the quality gate, so it is evidence for training,
   not a candidate packed artifact.
3. **ASTC-aware training experiment.** Train or fine-tune through a
   differentiable surrogate for the constrained endpoint/interpolation grid,
   then project and validate against legal ASTC blocks. Compare model loss and
   held-out quality with the post-training experiment. The discrete `astcenc`
   search remains an offline projection/validation stage, not an operation in
   the backward pass.

The acceptance gate for entering a Vulkan ASTC-Q matvec scaffold is a
reproducible, held-out quality/byte frontier that is competitive with the
packed baseline on the same small model. A capacity win alone is insufficient;
the experiment must also measure whether texture decoding reduces total shader
work or improves latency/energy on a real device.

## Research track: ASTC-Latent additive weights

**ASTC-Latent** is a second, deliberately separate hypothesis. It asks whether
one logical weight per ASTC texel can be represented as two learned numerical
latents and reconstructed after standard texture sampling:

\[
\hat w = s_L L' + s_A A' + b.
\]

The initial source layout is `R=G=B=L, A=A`. It is chosen to resemble a
luminance-plus-alpha signal, but it must not be confused with a promise that
every encoded block uses ASTC dual-plane. In dual-plane mode, ASTC assigns a
second interpolation weight grid to **one selected component**, while the
other components share the first grid; it does not provide two arbitrary,
independent endpoint planes. The offline encoder selects this mode per block,
and the Vulkan sampler exposes only the decoded values. The PoC therefore
queries the public `astcenc_get_block_info` API and reports both the total
dual-plane count and the count whose second component is alpha.

This layout still has one scalar weight per texel, hence the physical rate is
8.00, 5.12, and 3.56 bits per weight for 4x4, 5x5, and 6x6 respectively before
image-edge padding and the three decoder scalars. It is not the earlier
two-weights-per-texel experiment, whose apparent rate was lower only by asking
the same 128-bit ASTC block to represent more independent values.

The research inspiration is additive quantization and codec-aware training,
not a claim of equivalence. AQLM learns additive vector codebooks and optimizes
them with model-aware objectives; ASTC-Latent initially only supplies two
sampled latent fields. The progression is therefore:

1. **Post-training control.** Compare scalar replicated RGBA with a fixed
   coarse-plus-residual L+A construction, then fit `s_L`, `s_A`, and `b` after
   the real ASTC roundtrip. This validates the contract and likely rejects a
   naive high-frequency residual.
2. **Codec-aware representation.** Optimize L and A through a differentiable
   surrogate, repeatedly project them through the real encoder, and choose
   candidates by held-out `||XW-X\hat W||^2` rather than weight MSE alone.
3. **Model-aware fine-tuning.** If the small-model gate is promising, use
   calibration/model loss and compare limited representation optimization with
   PV-Tuning-style alternatives. AQLM-like codebooks are then a candidate
   parameterization, not an assumption.

Standard ASTC footprints 8x6 (2.67 bits/texel) and 8x8 (2.00 bits/texel) are
valid future rungs. They remain outside the initial contract until the Vulkan
capability probe verifies them on the target device and the 4x4/5x5/6x6
one-weight-per-texel result establishes a useful quality trend.

## Phase 3: compute matvec kernel

### Current status after the first device-backed controls

The isolated Phase-3 scaffold now has separate, reproducible shaders for
FP32-buffer, ASTC 4x4/5x5/6x6, Q4_0, and TQ2_0. All use the same 64-invocation
row reduction and the same deterministic activation fixture. Exact F16-derived
weights are exported from the GGUF reader; ASTC payloads and packed Q4/TQ2
fixtures are generated from that same source tensor. The Intel UHD 620 probe
validated all paths on ANV/Mesa without modifying `ggml-vulkan`.

The first 576x512 steady-state timing sweep showed ASTC4 at approximately
158.1 us per dispatch, Q4_0 at 173.4 us, ASTC5 at 172.3 us, TQ2 at 187.9 us,
and ASTC6 at 196.5 us. These are mechanism measurements, not quality claims:
the Q4/TQ2 shaders are correctness-first controls and should be compared with
the production kernel only after a fair profiling pass. The result supports a
bandwidth/capacity hypothesis but does not establish an end-to-end speedup.

The next required gates are:

1. [~] Measure FP32, Q4, TQ2, and ASTC on exactly the same shape and report
   elementwise plus activation-relative error from one calibration/holdout
   trace. The L+A ASTC, scalar ASTC, Q4, and TQ2 values are now available; a
   single combined timing harness remains.
2. [~] Separate raw buffer loads, sampled FP32 loads, ASTC decode, and packed
   dequantization with minimal microbench shaders. The first sampled-FP32
   control is now present; its scalar result is provisional until the common
   warm-up harness is added.
3. Repeat hot-cache and streaming/capacity regimes, then test batch sizes
   1, 2, 4, 8, and 16.
4. Add true TQ1 only after the TQ2 control is stable; its base-3 unpacking is a
   separate stress case, not a drop-in replacement for TQ2.

No Phase-4 integration is authorized by this plan until these gates produce a
quality/storage/latency Pareto result on more than one representative workload.

Target-device policy: the current Intel integrated-GPU measurements validate
the Vulkan ASTC mechanism, not portable performance. Repeat the fixed harness
on at least one native ASTC mobile family (Mali, Adreno, or Apple) before
using timing results to select a representation. Keep scalar ASTC as the
runtime baseline and L+A as a retained activation-aware/codec-aware research
track. The related BCn/CUDA experiment is intentionally outside this plan.

1. Add dedicated experimental compute shaders following the existing Vulkan
   shader source/build conventions.
2. Bind ASTC images through sampled-image descriptors and bind companion
   metadata/activations/output through the ordinary buffer descriptor path.
3. Use `texelFetch` at a fixed mip level; do not enable filtering or sRGB.
4. Map adjacent invocations to adjacent tensor values and make the reduction
   dimension cache-friendly in image coordinates.
5. Implement 4x4, 5x5, and 6x6 variants, or compile-time specializations, using the
   same observable numerical contract.
6. Compare against the closest existing Vulkan Q4-style matvec kernel with
   identical tensor shapes and activations.

Primary measurements:

- GPU time and token-path latency;
- represented weight bytes and resident GPU memory;
- achieved throughput and estimated effective bandwidth;
- dot-product/logit error;
- sensitivity to workgroup shape and access locality.

Exit criterion: the ASTC kernel is correct, has a clear comparison baseline,
and demonstrates either a credible bandwidth reduction, competitive latency,
or a useful memory-capacity improvement.

## Phase 4: isolated ggml Vulkan integration

1. Add an internal experimental tensor/resource wrapper for ASTC images.
2. Add runtime capability detection and an explicit opt-in experimental switch.
3. Route only supported static weight tensors to the ASTC kernel.
4. Route all unsupported devices, tensors, and operations to the existing
   Vulkan buffer path without changing observable behavior.
5. Ensure resource destruction, synchronization, descriptor allocation, and
   device-loss handling follow the existing backend patterns.

This phase must not make ASTC a requirement for building or using Vulkan.

Exit criterion: a small real model or selected real layers run correctly with
ASTC enabled and automatically fall back when disabled or unsupported.

## Phase 5: model-level validation

1. Compare output logits against the F16 baseline for fixed prompts. The
   matched tinybpe1m family is the first smoke fixture; an ASTC runtime must
   consume the same F16 tensor source before comparing against Q4 or a true,
   block-compatible TQ variant.
2. Run perplexity or another model-appropriate quality evaluation on the same
   corpus and tokenizer.
3. Measure prompt processing and token generation separately.
4. Test at least two distinct GPU/driver families where practical.
5. Profile with vendor tools when available; do not infer texture-decoder
   parallelism from API-visible properties.

Decision points:

- Prefer 4x4 where accuracy dominates.
- Evaluate 5x5 as the intermediate rate/quality point before assuming that
  4x4 or 6x6 is optimal.
- Prefer 6x6 where real measured bandwidth or capacity gains dominate and
  quality remains acceptable.
- Consider mixed per-tensor/per-layer selection only after each individual
  format is understood.
- Stop or redesign the approach if texture-pipeline cost or model-quality loss
  outweighs the reduced representation size.

## Upstream-readiness checklist

Before proposing any upstreamable portion:

- split research tooling from production backend changes;
- make experimental behavior opt-in and fully fallback-safe;
- avoid unnecessary public API, file-format, and core-tensor changes;
- retain existing coding conventions and generated-shader workflow;
- include focused correctness tests and reproducible benchmark instructions;
- document device requirements and known limitations without performance
  claims that cannot be reproduced.

The first potentially upstreamable contribution is likely capability plumbing
or a general sampled-image helper only if it benefits more than this
experiment. The ASTC weight format itself should remain experimental until it
has model-level evidence across devices.
