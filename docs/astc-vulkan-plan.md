# ASTC Vulkan Experimental Implementation Plan

This plan implements the research direction in
[`astc-vulkan-research.md`](astc-vulkan-research.md) while keeping the change
small, optional, and compatible with an eventual upstream review.

The proof of concept starts in `pocs/astc-vulkan/`, parallel to the production
`ggml-vulkan` backend. It does not copy the production backend. Integration
into `ggml-vulkan` is a later decision gate after device-backed evidence.

## Current low-rate quality track: neural standard-ASTC search

The runtime contract remains deliberately conventional: an ordinary legal ASTC
image is sampled through Vulkan and decoded by fixed-function texture hardware.
This track does **not** introduce a custom ASTC runtime decoder. It investigates
whether the offline encoder can retain and select better legal standard-ASTC
payloads for neural weights, particularly at `8x6` and `8x8` footprints.

1. [x] Add `--encoder-search standard|neural` to the isolated latent harness.
   `standard` is the unmodified astcenc reference path. `neural` always keeps
   every stock result as a mandatory candidate, including the exact scalar
   fallback, and performs a second wider astcenc search only to supply extra
   legal payloads.
2. [x] Exact-decode the retained payloads and cap the neural bank per ASTC
   block using mandatory stock candidates, the local activation-loss winner,
   and calibration-direction diversity. The active build's v1 input is a
   two-budget stock-astcenc search; callback candidates will be added only
   when the isolated astcenc side fork is linked. The existing conflict-aware
   selector and validation-prefix rule are intentionally unchanged.
3. [x] Add a focused CTest smoke for the neural candidate-recall path.
4. [x] Compare `standard` and `neural` on Pythia `blk.0.ffn_down` and
   `blk.1.ffn_down` at `8x6` and `8x8`, with the same source, traces,
   validation-selected prefix, and candidate cap. Layer-0 holdout improves by
   4.14% (8x6) and 16.65% (8x8); layer-1 is worse at 8x6 and ties at 8x8.
   Neural recall is therefore a tensor-aware experimental profile, not an
   unconditional replacement for standard search.
5. [x] Record ASTC block-mode histograms for scalar-anchor and selected payloads
   (endpoint class and level, partition count, weight-grid geometry and
   precision, dual-plane state). The harness emits this audit for scalar
   anchors and selected payloads. Use the evidence to add mode-family quotas
   ahead of image-oriented early pruning in the astcenc side fork.
6. [x] Test zero-sum weight-grid gauge bases `L = q + G`, `A = q - G` before
   adding a semantic correction. X-ramp, Y-ramp, and bilinear-saddle bases
   improve validation-stopped holdout over constant gauge on both Pythia
   tensors and both `8x6`/`8x8` crops. These remain in the semantic decoder
   null space before ASTC encoding.
7. [x] Evaluate the first global 3- and 5-level source fields plus gauge. They
   lose clearly on Pythia layer-0 `8x8`, so a simple few-level source is not a
   main path. A future learned/activation-aware source projection remains a
   separate research item. Next, add `10x6` (about 2.13 b/w) before broader
   `10x8`/`10x10` low-rate sweeps.
8. [ ] Only after a fixed candidate space is understood, evaluate richer
   objectives (two-sided sensitivity/Block-LDLQ) and later mixed-footprint
   macrotiles. These are not part of the neural-search-v1 gate.

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
59g. Add a stability-gated fixed-pool selector using two equal calibration
     shards. Normalize each shard gain by its residual energy and accept a
     proposal only when the lower shard gain remains positive. Compare it with
     local, coordinate, feedback, and conflict-aware results before collecting
     larger traces.
59h. Add input-Hessian diagnostics (sample/column count, rank, eigenvalue range,
     and damping reference) to the selector harness. Do not interpret
     Block-LDLQ target regeneration until the trace rank and conditioning are
     adequate for the selected column panel size.
59i. Add a standalone Block-LDLQ conditional-target contract with a known
     correlated Hessian. Require the regenerated legal candidate to reduce the
     quadratic objective before wiring target updates into ASTC encoding.
59g. Add a stability-gated fixed-pool selector using two equal calibration
     shards. Normalize each shard gain by its residual energy and accept a
     proposal only when the lower shard gain remains positive. Compare it with
     local, coordinate, feedback, and conflict-aware results before collecting
     larger traces.
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

### Current execution gate: candidate capacity before an internal pool

The next implementation step is an opt-in bounded candidate-capacity sweep.
Use the existing `astcenc` search with limits 1, 2, 4, and 8, preserve legal
standard ASTC output, and compare those streams in the existing selector
matrix. Record:

- candidate pool size and unique per-block choices;
- calibration and holdout activation-relative error;
- recovered coordinate-descent gain;
- encode time and compressed bytes (the latter must remain unchanged);
- whether extra candidates improve holdout rather than only calibration.

Interpretation rule: this whole-image pool is a proxy for candidate diversity,
not evidence that an internal neural-aware ASTC encoder exists. If it shows a
stable benefit, expose a side-fork-only per-block top-K diagnostic interface
and feed those legal blocks into Block-LDLQ/GPTVQ target regeneration. If it
does not, keep the existing search and move effort to richer calibration
traces and the target-regeneration engine.

The side fork now provides the opt-in per-block callback and the PoC has
validated a bounded top-K collector. The next algorithmic gate is therefore
not a larger pool: implement target regeneration over these legal block
alternatives. Start with the two-block Block-LDLQ contract, extend it to a
small row/column block, and compare local, coordinate, conflict-aware, and
regenerated-target selectors on identical candidate snapshots. Keep the
callback and collector out of normal Vulkan execution.

The first target-regeneration prototype is now measured. It is a useful 6x6
signal but regresses 4x4/5x5, so it remains an experimental selector. Before
any GPU integration, replace the diagonal approximation with a bounded
block-LDL update, add a candidate-coverage diagnostic, and repeat the exact
three-way comparison on larger calibration traces and at least one additional
real tensor. The current CTest smoke is a contract/regression gate, not a
quality threshold.

The compact block-LDL update is now the preferred 6x6 research path. Before
GPU work, run the full 4x4/5x5/6x6 matrix on the same larger traces, expose
the damping and block-order parameters for ablation, and add a coverage
report showing how many retained candidates are actually reachable per ASTC
block. Any mixed-format decision must wait for those regressions and for a
second real tensor.

The second-tensor control did not reproduce the `attn_q` Block-LDLQ win;
stability gating was marginally better on `attn_k`. The plan therefore treats
Block-LDLQ as tensor-sensitive and requires per-tensor selector choice only as
an offline research result. No automatic mixed-format/runtime policy is
allowed until the trace size, tensor coverage, and GPU timing gates are met.

The bounded proxy has now been run. It is retained as a regression/evaluation
gate, while the next implementation target is a side-fork-only per-block
top-K snapshot. That snapshot must be opt-in, thread-safe or explicitly
single-threaded for diagnostics, and must return only legal 16-byte ASTC
blocks. No Vulkan runtime or standard llama backend path may depend on it.

Per-block candidate ownership and coverage diagnostics are now enforced. The
next selector experiments can therefore be interpreted as genuine block-local
comparisons. Keep the baseline streams available as escape candidates and
reject any block with insufficient legal coverage from quality claims.

### Variable coding policy

The design remains open to variable coding. We are not required to encode an
entire layer with one identical representation: each ASTC block may choose a
different legal mode, partition, dual-plane state, candidate, or selector
result. The initial runtime contract stays simpler by using one footprint per
tensor and a uniform sampled-image layout. Mixed footprints within one tensor
are deferred because they require additional metadata and address/decode
handling. This is a deferred optimization, not an architectural restriction.

Calibration-size ablation is now a required reporting dimension. Use
`--max-calibration-samples` while keeping the holdout trace fixed, and report
both sample count and shard composition. Do not tune damping, block order, or
format choice against a reduced holdout; reduced traces are mechanism probes
only.

Block order is now a measured algorithmic parameter. Forward order is the
reference because target regeneration is directional; reverse order is kept
as a negative control. The next implementation should derive a bounded
Hessian-pivoted order offline and compare it against forward order without
changing the runtime texture layout.

The first bounded pivot probe is complete. Sorting row-local ASTC blocks by
absolute Hessian connectivity improved calibration in one case but regressed
fixed-holdout error on both `attn_q` and `attn_k`. Treat this heuristic as a
negative control, not as the default order. The next gate is a three-way
comparison on identical candidate snapshots (local ranking, forward
Block-LDLQ, and improved Hessian feedback/target regeneration) across 4x4,
5x5, and 6x6, with fixed holdout traces and at least two real tensors.

The complete bounded matrix is now available. Forward Block-LDLQ wins only
for `attn_q` at 6x6; local ranking wins the other five tensor/footprint cases,
while coordinate descent loses to local in all six. Treat the 6x6 `attn_q`
result as a promising but tensor-sensitive research signal. Before expanding
the search or integrating GPU execution, improve candidate-direction
coverage and test activation-weighted/Hessian-whitened scoring with local as
the mandatory fallback.

An opt-in angular shortlist experiment is complete. Selecting extra legal
ASTC candidates by cosine separation in activation-error space leaves the
6x6 `attn_q` Block-LDLQ result essentially unchanged (`0.34425` versus
`0.34428`) and does not change the winner in the six-case matrix. It gives a
larger secondary improvement on `attn_k` 5x5 but remains above local. Keep it
as a candidate-direction diagnostic; the next required gate is larger,
sharded calibration data and fixed holdout evaluation, with local fallback.

The angular calibration-size ablation confirms the same pattern: four
calibration samples overfit, while eight and ten samples make 6x6 Block-LDLQ
slightly better than local on the `attn_q` crop. Treat this as evidence for a
minimum-trace/data-diversity gate, not as a quality threshold. Add explicit
shard-stability reporting and conservative cross-shard acceptance before
testing larger traces or GPU performance.

Five- and ten-shard ablations show that strict gating collapses to local when
the calibration set is too small. Keep four shards as an opt-in probe for the
current ten-sample fixture, scale shard count with trace size, and defer a
separate worst-shard penalty until larger calibration data exists.

The stability selector now supports a configurable shard count. Four-shard
gating improved the 6x6 holdout on both real tensor controls and beat local on
`attn_k`, making it the current robust fallback. Keep the default at two for
compatibility, but prioritize four-shard stability versus Block-LDLQ on larger
traces before GPU timing or runtime policy work.

Four-shard diagnostics are now emitted for selector comparisons. The current
6x6 `attn_q` Block-LDLQ result has lower mean calibration error but higher
dispersion and a worse worst shard than local, so the next implementation is
an offline conservative gate: penalize shard variance and reject any candidate
that regresses the worst shard. Compare it directly with the existing
stability selector before making further quality or GPU claims.

The damping ablation did not change the current 6x6 decisions across
`1e-5`--`1e-3`. Keep the parameter exposed for future ill-conditioned traces,
but prioritize calibration diversity, block ordering, and candidate direction
retention over further damping tuning.

Regression coverage now includes a dedicated angular-shortlist/four-shard
stability smoke; the focused ASTC suite is 4/4 green. The current real-model
calibration capture has only ten samples, which is insufficient for a genuine
larger-trace claim. Acquire or generate a disjoint, larger calibration trace
before changing selector defaults or starting GPU performance interpretation.

The larger 1,872-position capture is now available, with a 256-sample bounded
matrix evaluated against the fixed holdout. Block-LDLQ is competitive or
better than local in five of six tensor/footprint cases, while four-shard
stability or conflict-aware often wins by a larger margin. The current policy
is therefore per-tensor offline selector choice with local fallback, not a
global Block-LDLQ default. The next gates are end-to-end layer/model scoring
and then device-backed Vulkan sampled-image validation.

Layer-1 attention controls now favor four-shard stability on both `attn_q` and
`attn_k`, reinforcing per-tensor offline selection with local fallback. A wide
1,536-column trace serialization contract is also covered by the input smoke.
The remaining FFN limitation is data capture, not ASTC or selector width:
`ffn_down` needs post-FFN intermediate activations. Add that PoC-only capture
hook before making FFN quality claims.

### Current execution gate: wide FFN traces

The PoC-only capture hook is now implemented. It taps the activation before
`build_lora_mm(down, cur)` and exposes the per-layer `n_ff(layer)` width to the
trace tool. This keeps the production Vulkan path unchanged while allowing
real FFN-down inputs to be evaluated. The first SmolLM2 validation produced an
11 x 1,536 trace and a bounded 6x6 ASTC run over all 1,536 columns.

Required follow-up before GPU/runtime work:

1. Capture a disjoint FFN calibration/holdout pair with enough samples.
2. Evaluate 4x4, 5x5, and 6x6 using the same candidate snapshot and selector
   matrix (local, Block-LDLQ, stability, conflict-aware).
3. Increase the row tile while retaining the full 1,536-column width; report
   edge-padding separately from asymptotic block bandwidth.
4. Keep the capture API opt-in and staging-only until end-to-end quality and
   device-backed Vulkan sampling have been measured.

The first disjoint FFN control (32 rows, all 1,536 columns) is complete. Local
ranking remained best on 4x4 and 5x5 holdout error; two-shard stability tied
local on 6x6. This is consistent with the conservative fallback policy and
does not justify a global selector or footprint default.

The tile-height follow-up confirmed the bandwidth accounting: at full 1,536
columns, a 32-row 6x6 crop measured 4.000 b/w because of edge blocks, while a
128-row crop measured 3.6667 b/w and approached the 3.5556 b/w asymptote. Use
tall/full-height tiles for rate studies and keep short crops only for functional
contracts.

The local Vulkan inventory now identifies Intel UHD 620 as the only enumerated
device advertising ASTC LDR. This can serve as a first device-backed sampler
check, but it must be reported separately from the intended ARM/Mali target.

The isolated Vulkan build is green. Intel
UHD 620 passes ASTC 4x4/5x5/6x6 capability, image, and shader-device tests;
NVIDIA 920MX and llvmpipe are recorded as unsupported for sampled ASTC. A
32x1,536 scalar ASTC payload also passes the GPU matvec contract on Intel. The
next gate is a larger repeated workload plus a matched buffer-backed control,
not a production-backend integration.

### Remaining path to an ASTC Vulkan driver

1. Keep the sidecar manifest and format geometry as the stable boundary. Add
   versioned atlas/page metadata and deterministic payload validation before
   model-loader work.
2. Extract the existing PoC Vulkan image, view, sampler, staging upload, and
   descriptor setup into a reusable sidecar resource module. Preserve the
   current shader smoke as a consumer so behavior remains regression-tested.
3. Add an ASTC matvec resource/session API that accepts a manifest record,
   uploads its blocks, binds the sampled image, and exposes scale/bias and
   reconstruction mode as explicit parameters. Keep one-tensor-per-image as
   the first implementation; atlas packing is an optimization after the
   resource contract is proven.
4. Define the first llama-facing adapter at the PoC boundary (FFN-down
   weights, 1,536-column support, 4x4/5x5/6x6 selection). Do not hook the
   production scheduler until correctness against the FP16 reference and
   fallback behavior are tested end to end.
5. Add device capability selection and a hard fallback for devices without
   sampled ASTC. Record that Intel/Mesa timings are only one implementation;
   ARM/Mali, Adreno, and Apple measurements remain required target-specific
   work.
6. Add larger-tile and batched dispatch benchmarks, then compare ASTC against
   FP16, Q4_0, TQ2_0, and sampled-FP32 using the same matrix and launch shape.
7. After the sidecar path is clean, perform a refactor pass and decide whether
   any narrow, upstreamable interface belongs in ggml-vulkan. Until then all
   code remains under `pocs/astc-vulkan`.

The repeated GPU controls, sidecar resource/session, FFN adapter, and real
activation-buffer dispatch are now implemented and device-smoke validated.
The remaining quality gate is full 576-row projection followed by model-level
logits/loss comparison. Production scheduler integration remains deferred
until that evidence and fallback behavior are available.

The full 576-row FFN-down projection and 4x4/5x5/6x6 GPU comparison are now
complete. The next implementation is model-level output capture/injection so
we can measure logits or loss while keeping the normal FP16 path as the oracle.

### Model-level validation checkpoint

The first half of this gate is complete: an opt-in llama graph/output capture
now exposes the real FFN-down result, and the Vulkan FFN smoke can compare its
ASTC result against that FP16 reference trace. The remaining work is to run a
matched prompt on the same model, capture input and output for a disjoint
calibration/holdout split, and report layer-output and then logits/loss error.
Only after that evidence should a narrow `ggml-vulkan` integration boundary be
considered; devices without sampled ASTC must continue through the existing
buffer-backed path.

The real layer-output comparison is now complete for all three initial
footprints. Full-transformer logits/loss injection remains the final quality
gate before any production scheduler integration. The current implementation
deliberately stops at an opt-in capture/replay boundary so the standard llama
and `ggml-vulkan` execution paths remain unchanged.

The CPU replay gate is now operational and has produced full-model logits
comparisons for all three initial footprints. The remaining model-quality work
is a larger prompt/holdout suite with loss or perplexity, followed by a design
review of whether a GPU-side replacement can reuse the same graph-input
contract. No production `ggml-vulkan` routing should be added before that
evidence is available.

The replay utility is `astc-vulkan-model-replay-smoke`. It reconstructs the
selected ASTC footprint from the same decoded RGBA fixture and captured FFN
input, then runs a normal reference context and an override context against
the same prompt. It is a validation harness, not a runtime driver.

### Latest quality checkpoint

The replay utility now compares logits at every prompt position and reports a
small next-token cross-entropy metric in addition to MSE, relative MSE, and
top-1 agreement. This closes the first model-level quality gate for the
single-prompt fixture. The observed 4x4/5x5/6x6 results preserve the expected
quality ordering, but the short prompt is not enough to select a production
default. Before any scheduler work, repeat the measurement on disjoint
calibration and holdout prompts and add the existing FP16/Q4/TQ baselines.

A second matched prompt/activation trace has now been captured and replayed.
It confirms the footprint ordering in logit error, but also demonstrates that
cross-entropy and greedy agreement can diverge on a short prompt. Treat this
as a holdout diagnostic only; require a multi-prompt aggregate before selecting
4x4, 5x5, or 6x6, and compare against FP16, Q4_0, TQ1_0, and TQ2_0 using the
same targets.

Two further matched prompt/activation pairs have now been added, giving a
four-prompt aggregate. Mean relative logits MSE is `3.15e-4`/`1.93e-3`/
`7.82e-3` for 4x4/5x5/6x6, with mean top-1 agreement of 95.2%/88.0%/82.9%.
Mean loss delta is +0.0043/+0.0460/+0.2351. This prioritizes 4x4 for a
conservative path and retains 5x5 as a tunable compromise, but does not select
a production default. The next gate is the same suite against FP16, Q4_0,
TQ1_0, and TQ2_0, followed by a larger model.

The first TQ audit is complete. On the real FFN-down tensor, TQ1_0 and TQ2_0
have different packed rates but identical reconstructed values and quality;
some other tensors remain Q4_0 in the mixed GGUFs. Keep these files as useful
bandwidth controls, but do not treat them as definitive pure-TQ baselines
until a tensor-type manifest or clean re-quantization is available.

The baseline smoke is now implemented and the four-prompt Q4/TQ comparison is
recorded in the WIP log. Q4_0 is materially closer to FP16 than the current
single-layer ASTC replay. TQ1_0 and TQ2_0 are both valid GGUF types but produce
identical very poor metrics in the local fixtures, so the next TQ step is a
fixture/dequantization audit before drawing algorithmic conclusions.

The clean re-quantization check is now complete: regenerated TQ1_0/TQ2_0 files
match the existing fixtures byte-for-byte, and the quantizer reports 181/272
tensor fallback conversions. Keep TQ in the comparison matrix, but label it as
a mixed-format control. The larger-model transition can proceed with FP16 and
Q4 first; TQ should be included once a tensor-type manifest is available.

Qwen2.5-Coder-1.5B-Instruct Q4_K_M is now available under `/home/prbm/models`
and passes the trace-capture architecture gate. Its 8,960-wide FFN is the
selected larger-model target. The official GGUF set has no F16 file, so plan a
safetensors-to-F16-GGUF conversion before using Qwen as the definitive ASTC
quality oracle; Q4/Q8 remain useful runtime controls.

The implementation remains intentionally sidecar-only. The override is an
opt-in graph input selected only for an exact token-batch shape; ordinary
reserve shapes and all default contexts continue to use the normal FFN-down
matmul. This keeps the production Vulkan backend untouched while giving us a
model-level oracle for the future GPU replacement design.

### Next model-validation sweep: Pythia before Qwen FP16 conversion

Use Pythia-1.4B as the first larger FP16 codec reference. Preserve the official
EleutherAI safetensors as provenance, validate the locally runnable F16 GGUF,
and compare a same-source Q4_K_M baseline before any ASTC ranking claims.

Required measurements:

1. Capture matched layer-0 FFN-down traces with disjoint calibration and
   holdout prompts.
2. Run the ASTC quality harness for 4x4 and 6x6, recording model dimensions,
   bytes, layer-output error, activation-relative error, logits, loss, and
   top-1 agreement.
3. Compare ASTC against FP16 and same-source Q4_K_M; retain 5x5 only as a
   compromise diagnostic if the full sweep is too expensive.
4. Repeat the selected measurements on Qwen2.5-Coder after obtaining an
   F16 oracle, because Pythia does not cover the 1,536-column target.
5. Keep all model-specific outputs labelled by architecture and source type;
   do not combine Pythia and Qwen numbers into one aggregate.

The sidecar boundary stays unchanged. No production `ggml-vulkan` routing,
shader ABI, or standard Vulkan path is modified by this model transition.

The quality harness now supports `--max-rows` for explicitly labelled large
model screening and `--skip-residual-analysis` to avoid the small-fixture
low-rank/sparse pass on matrices with tens of millions of values. These are
measurement controls only; the ASTC encoder and sidecar ABI are unchanged.
The Pythia 256-row screen is a parameter-ranking checkpoint. Before using it
to select a format, run the full 2,048-row layer and then the model-level
capture/replay on disjoint prompts.

The full selected Pythia run is now complete. The 4x4 candidate passes the
activation gate at `0.0902300` with 7,633,640 total bytes, while 6x6 measures
`0.227049` with 4,149,320 total bytes and fails the gate. Treat 4x4 as the
current quality-oriented ASTC candidate and 6x6 as an explicitly bandwidth-
oriented mode. The next required step is exporting the chosen ASTC payload and
running the existing model replay harness on multiple holdout prompts; these
layer metrics alone do not justify scheduler integration.

The selected-candidate run is reproducible with `--screen-only`; the default
quality smoke still evaluates the broader transform/basis matrix for small
fixtures.

Pythia TQ2 has now been added as an explicitly labelled low-bit control. Its
full-model result is logits-relative-MSE `1.3086276`, top-1 agreement `0%`, and
loss delta `+8.060108`; do not compare this directly to the single-layer ASTC
metric. Keep TQ2 in the bandwidth matrix, but reserve quality conclusions for
the ASTC layer-output and model-replay gates.

The Pythia L+A screening is complete on a labelled 256-row submatrix. At 4x4,
luminance+alpha additive reaches activation-relative-MSE `0.0159568`; at 6x6
it degrades to `0.690284`, while the block-residual form reaches `0.0281505`.
Keep L+A as a viable representation family, but prioritize block-local
residual/error shaping and activation-aware candidate ranking before attempting
a full-row L+A default.

### Current execution gate: end-to-end ASTC path

The next three steps are executed in order on the isolated sidecar:

1. Export a legal ASTC payload, decoded RGBA32F companion, source FP32 matrix,
   and self-describing decoder metadata. Use the real Pythia FFN-down shape
   (2,048 rows x 8,192 columns) and keep 4x4 and 6x6 as separate artifacts.
2. Replay the exported L+A representation through the model override and
   record logits MSE, relative logits MSE, loss delta, and top-1 agreement
   against the same FP16 model and prompt trace.
3. Submit the identical ASTC bytes to the Vulkan sampled-image path. Compare
   GPU output with the CPU reconstruction and with the source FP16 matvec.

The metadata decoder contract is:

\[
\hat w = s_L (R+G+B)/3 + s_A A + b.
\]

This is deliberately a PoC ABI, not production `ggml-vulkan` integration.
The GPU result is hardware- and-driver-specific (ASTC format support,
sampler precision, cache and dispatch geometry), so CPU replay remains the
portable quality reference and Vulkan remains the execution-path reference.
Large model exports are measurement jobs rather than CTests; CTests cover
contracts, input validation, and shader compilation.

The 4x4 L+A artifact has completed all three gates on Pythia: export,
model-level replay, and Vulkan sampled-image execution. Keep its metrics as a
baseline for later neural-aware/error-shaped encoders; do not promote it to a
runtime default yet because the model loss delta remains material. Run the
same gates for 6x6 before comparing bandwidth and quality.

The first structured-residual screen confirms that block-constant Alpha is a
stronger baseline than row, column, or plane-like Alpha on the current Pythia
6x6 fixture. Do not increase spatial degrees of freedom blindly. Add a
confidence-gated correction experiment next: estimate residual smoothness,
activation gain, and shard stability, then attenuate or disable Alpha for
blocks whose correction is not reliable.

The 6x6 gates are now complete. The payload is 7,474,752 bytes for the
2,048x8,192 layer and the Vulkan path is numerically correct, but model replay
has relative logits MSE `1.1037144` and loss delta `+5.4399957`. Treat this as
the lower-bound bandwidth control for the current L+A encoder. The next
quality work should target block-local residual/error shaping or neural-aware
candidate selection before any 6x6 runtime experiment.

### Follow-up benchmark gate

After correctness is established, compare a matched FFN dispatch suite for
plain Vulkan buffer weights (FP16/Q4 control) versus ASTC 4x4 and 6x6 sampled
weights. Report dispatch latency, effective tokens/s, bytes transferred,
GPU-vs-CPU synchronization, and shader occupancy where the driver exposes it.
The existing E2E smoke is a correctness harness, not an inference-speed
claim; a full token-loop benchmark must use identical prompts, batch shapes,
workgroup geometry, and warm-up policy for all three paths.

The first dispatch benchmark is complete on the full Pythia-shaped matrix:
buffer FP32 `14.632 ms`, ASTC 4x4 `9.047 ms`, ASTC 6x6 `7.816 ms`, and sampled
R32F `2.208 ms` per dispatch. Thus ASTC is `1.62x`/`1.87x` faster than the
current storage-buffer control, but slower than ordinary sampled F32 on this
device. Treat these as steady-state GPU timestamps only; upload amortization,
Q4 comparison, and end-to-end token throughput remain separate gates.

### Alpha correction follow-up: decoded candidate scoring

The first activation-fitted constant-Alpha sweep did not beat block mean on
the Pythia 256-row 6x6 screen: `0.0281505` for block mean versus `0.0284773`,
`0.0284287`, and `0.0288424` for activation-optimal, shard-gated, and full
confidence-gated variants. Keep the result and implementation as a regression
experiment. Do not add row, column, or plane capacity: those were decisively
worse.

The next error-correction step is instead codec-aware candidate scoring:

1. Generate the existing legal ASTC candidates for a block unchanged.
2. Decode each candidate (software reference during offline encoding).
3. Score its *decoded* `L + A` correction with activation loss, optionally on
   calibration shards, so the score includes ASTC's own perturbation.
4. Use a separate captured trace for final holdout evaluation; never use the
   shards used for fitting as evidence of generalization.

This preserves the key hypothesis—Alpha is a low-dimensional neural
error-correction side channel—while aligning the optimization objective with
the actual fixed-function decoder.

The existing experimental `MAP_NEURAL_LA` ASTC-fork mode is not that objective.
On the Pythia 6x6 block-residual control it measured `0.1104824` relative
activation MSE, versus `0.0281505` for standard ASTC. Keep it only as a
negative regression control. The next candidate scorer must be supplied with
activation traces and score decoded candidate blocks after the complete
`L + A` reconstruction; passing decoder scales alone is insufficient.

The first disjoint trace check confirms the need for that gate. On 30 held-out
Pythia positions, scalar 6x6 is `0.0292211` and block-mean Alpha is
`0.0293151`; activation-optimal and gated forms are worse still. Keep the
representations and tests, but require a material held-out improvement over
the scalar control before running a full-layer L+A export or changing runtime
planning.

### Decode-in-the-loop block-Alpha gate

Before any further LDLQ or full-layer work, implement a small offline search
over legal ASTC blocks. Use a single shared L+A affine decoder per tensor and
include neutral Alpha as a first-class candidate. For every block, encode a
bounded set of constant-Alpha source perturbations, deduplicate their decoded
16-byte blocks, and rank their marginal activation loss against neutral Alpha
on calibration shards. Evaluate the chosen per-block result on the disjoint
trace. Promote this track only when it beats the neutral/scalar control on
holdout; otherwise keep Alpha as a documented negative control and move to a
different representation family.

The small exact-block fixture has passed candidate legality and decoder
contract validation. It also showed positive local gain but worse assembled
calibration/holdout error, so independent per-block commit is explicitly
rejected. The next substep is **decode-in-the-loop conflict-aware commit**:
propose the best decoded candidate per block in parallel, then accept only a
non-conflicting proposal that reduces the current activation residual. Edge
padding and full Pythia dimensions remain out of scope until this small gate
beats neutral Alpha on holdout.

Log pairwise cosine/correlation and a compact SVD summary of the decoded
candidate output deltas. This tests whether many locally useful Alpha blocks
occupy only a few effective correction directions. The conflict-aware selector
must re-evaluate gain after every accepted proposal; an initial static sort is
not sufficient because each commit changes the residual.

The initial exact-block gate passes: conflict-aware selection improved held-out
relative MSE from `0.0603581` (neutral) to `0.0574917`, while independent
local selection regressed to `0.0686602`. Proceed to diagnostics and scaling,
not LDLQ yet: log candidate-direction diversity, add deterministic edge-block
handling, and repeat on a larger Pythia crop with the same disjoint traces.

Deterministic clamp-padded edges pass the 25x25 gate and preserve the held-out
conflict-aware gain. Scale next to successively larger Pythia crops, logging
local-positive, accepted, and total-block ratios alongside rank, cosine and
cumulative residual reduction.

The 48x48 and 96x96 runs pass the internal neutral-L+A gate, but the 96x96
result is still marginally above scalar 6x6. Add cumulative residual-gain
logging and test a larger crop or second tensor; do not claim a format-level
win until conflict-aware Alpha beats scalar on holdout.

Validation-prefix selection has been added and a second-tensor repeat is now
complete. Both confirm the correction-space mechanism but neither produces a
reliable scalar-beating result. The next experiment should alter only the
candidate family (for example a wider, bounded constant-Alpha dictionary) or
calibration coverage, then repeat the same three-trace gate. Keep LDLQ behind
that gate.

The scalar-anchored gauge-only gate passes both layer-0 and layer-1 96x96
crops, and now passes a layer-0 192x192 scale gate: scalar/neutral holdout is
`0.0426420`, full conflict-aware holdout is `0.0239173`, and an independently
validation-selected prefix reaches `0.0237592`. It makes scalar an exact
candidate-family baseline and removes the L+A base-representation penalty.

Next, scale gauge-only further before adding the bounded correction-plus-gauge
family. Preserve the calibration → validation budget → untouched holdout
protocol, record the complete commit path compactly, and diagnose how selected
gauge candidates change ASTC block modes and payloads relative to scalar.
Only after this anchored family remains beneficial at larger scale should the
project add semantic correction `c`, Block-LDLQ, or more expensive
full-model objectives.

The offline selector must keep candidate output deltas sparse: an ASTC block
only affects its own output rows, so dense full-matrix deltas are forbidden at
large crop sizes. Validation must be updated through the same decoded sparse
deltas rather than by repeatedly replaying the whole matrix. The optional
`--decode-loop-log` CSV records commit index, calibration and validation
relative MSE, marginal residual gain, and cumulative gain. This log is for
choosing a validation-only stopping prefix; holdout remains read only after a
prefix is fixed. Gauge diagnostics compare every accepted candidate payload
against its scalar baseline using the public `astcenc_get_block_info` API.

The 192x192 diagnosis shows that all 710 accepted gauge blocks changed payload,
while none changed dual-plane state or first endpoint mode; 177 changed an
endpoint/weight quantization-level count, 15 changed partition count, and five
changed the weight grid. Treat null-space steering as a way to traverse the
whole discrete ASTC feasible set, particularly its quantization boundaries,
not as an attempt to force a specific Alpha or dual-plane mode.

For full-tensor scaling, parallelize only independent per-block source
candidate generation and exact ASTC encode/decode. Preserve deterministic
block indexing and collect candidates before the existing serial,
conflict-aware residual commit. This separation keeps the experiment
reproducible: parallel work expands the legal dictionary, while the global
selection definition stays unchanged.

This parallel candidate stage is implemented as `--candidate-threads N` and
has passed one-versus-four-thread equality checks on 48x48 and 192x192 crops.
The next gate is a larger crop and a second tensor with the same equality
check. Full-tensor execution should then use chunked block results so memory
does not grow with the entire candidate dictionary at once; only after that
engineering gate should the project run the full `2048x8192` gauge-only study.

The 384x384 layer-0 gate now passes with scalar holdout `0.0549210` versus
validation-stopped gauge holdout `0.0272700`. The effective rank is `1,263`
over 4,096 blocks, so the full-tensor path must not assume one global low-rank
correction basis. Keep diagnostics local/chunkable and preserve the exact
conflict-aware commit contract. The next quality gate is the same protocol on
layer 1 before full-tensor engineering.

The layer-1 192x192 gate also remains positive: scalar holdout `0.0567970`
versus validation-stopped gauge holdout `0.0533107` (about 6.1% lower). The
large layer-0 gain is therefore not universal, but the mechanism survives on a
second tensor. Full-tensor work must retain per-tensor calibration, candidate
selection, and validation stopping; a single global gauge policy is not
supported by the evidence.

Row-strip selection is now byte-identical to the whole-crop selector at 48x48
and 192x192, including the full validation trajectory. Use it as the mandatory
regression baseline for chunking: a chunked implementation must first produce
the same per-strip sequences and the same merged global prefix before it is
allowed to run a full tensor.

That streaming gate is now complete. `--row-strip-chunked` generates, decodes,
selects, and releases one six-output-row strip at a time, retaining only a
compact sequence of selected blocks for the exact global merge. Its
`--decode-loop-payloads` artifact contains the final conflict-aware ASTC
payload stream in row-major block order. On layer-0 48x48 and 192x192 fixtures,
chunked and resident row-strip selection have byte-identical final payloads,
CSV commit trajectories, validation prefixes, and ordinary summary fields.

The 192x192 fixture has 32 strips and peaks at 192 resident alternatives,
with a measured candidate workset of 179,712 bytes. This counter includes
candidate metadata, decoded values, and calibration deltas; it intentionally
does not claim whole-process RSS or include the persistent full-matrix buffers.
It is therefore a reproducible measure of the object the streaming design is
meant to bound. The next full-tensor sweep may scale candidate residency only
with one `6 x 8192` row strip, while retaining the compact selected sequence
needed to perform the global validation-prefix merge exactly.

Before a full `2048x8192` run, keep the current candidate family and selection
semantics frozen. First audit selector cost at strip width: exhaustive
candidate-pair diagnostics and repeated greedy rescans are acceptable at
192x192 but not automatically at 1,366 blocks per strip. Any speed-up must
have a small-fixture byte-identical regression against the present chunked
selector; streaming reduces memory, not the definition of conflict-aware
selection. Only then run the full-tensor gauge-only gate, record per-strip
gain/acceptance statistics, and proceed to bounded `c + delta` or LDLQ work.

The cost audit now separates this cleanly. `--row-strip-light-diagnostics`
suppresses only the quadratic effective-rank/cosine diagnostic, not candidate
generation, decoded deltas, selection, global merge, or validation stopping.
`--row-strip-log` records per-strip candidate count, accepted steps, local
residual gain, candidate-generation time, and exact selection time. A layer-0
`6x1536` strip used 24.43 seconds for generation and 0.033 seconds for
selection with four candidate threads. Eight threads reduced generation to
18.35 seconds and retained byte-identical payload and commit CSV. Thus the
near-term full-tensor bottleneck is offline ASTC encoding; do not redesign the
exact selector before a full-width strip profile proves it necessary.

The full-width gate has passed. One layer-0 `6x8192` strip has 1,366 ASTC
blocks, 8,150 unique non-neutral candidates, and a 7,628,400-byte candidate
workset. With eight candidate threads it takes 95.95 seconds for legal ASTC
generation/decode and 1.35 seconds for exact conflict-aware selection. A
conservative 342-strip estimate is therefore about 9–10 hours, dominated by
offline ASTC encoding. This is now an authorized full-tensor gauge-only run:
use chunked selection, light diagnostics, a global commit log, a per-strip log,
and a final payload artifact. Do not change the candidate family or inspect
holdout until validation has selected its prefix.

### Post-baseline candidate-generation optimization order

The active full-tensor result is a frozen thorough-preset baseline. Do not
change its encoder path while it is running. An inspection of the current PoC
establishes three facts for the next sweep:

1. The side-fork is already correct for this host: it links
   `libastcenc-native-static.a`, built with `-march=native`; this i5-8250U
   exposes AVX2, F16C, and SSE4.1. SIMD is therefore not a missing variable.
2. The hot gauge path currently calls `astc_roundtrip()` per candidate, which
   allocates and frees an `astcenc_context` each time. Replace this with one
   persistent context per candidate-generation worker, preserving the exact
   configuration, source blocks, payloads, and deterministic ordering. Gate it
   with byte-identical payload and commit-CSV regressions before using it for a
   quality comparison.
3. Payload deduplication already happens immediately after encode and before
   decode/reconstruction/delta construction. It does not avoid the encode work,
   so it remains useful but cannot solve the dominant cost alone.

After context reuse, run a controlled `thorough` / `medium` / `fast`
ablation on the same frozen tensor and calibration/validation/holdout split.
Measure elapsed generation time, unique payloads, accepted candidates, and
validation-stopped holdout. This is a new encoder comparison, not a rerun of
the frozen baseline. Only if a cheaper preset preserves useful correction space
should the project add a two-stage preset policy, adaptive gauge grid, cache
audit, or a neural-gauge ASTC search preset. Reuse of block analysis across
gauge values is a later fork-level optimization: first profile preparation,
context allocation, ASTC search, decode, reconstruction, and delta creation
separately, then optimize only proven repeated work.

The exact-preserving context gate has now passed. The opt-in
`--persistent-worker-contexts` path allocates one parent context for immutable
tables and one child context per outer candidate worker; each worker resets and
reuses its own context. Its source staging buffer and short payload-dedup set
are likewise worker/block local. On the full-width `6x8192` fixture it produces
the identical payload SHA-256 and identical commit CSV as the allocation-per-
candidate baseline, while generation falls from 95.95 seconds to 8.19 seconds
per strip. The projected full layer-0 run is now about 55–65 minutes, including
selection and final assembly. This is the new baseline command. Source-block
caching remains an audit first, and a lower-level reusable block-analysis API
remains a separate, more invasive fork experiment after the new full result.

### Frozen performance reference configuration

Use the following as the reproducible encoder-performance baseline on this
host: eight outer candidate workers; one persistent parent context; one child
context and source scratch buffer per worker; shared immutable ASTC tables;
fixed block-local payload deduplication; streamed six-row strips; thorough
preset; fixed scalar-anchored gauge family; light diagnostics only for
full-width/full-height runs. The binary links the native astcenc build, so the
record must also name the host ISA (`AVX2`, `F16C`, `SSE4.1`) and compiler
`-march=native` configuration.

Every exact-preserving optimization must match this reference on the same
fixture by final payload SHA-256, global commit CSV, validation prefix, and
holdout. Candidate-space-changing work (preset, grid, cache policy beyond
exact source identity, or internal search pruning) is instead a new encoder
experiment and must report candidate count and quality separately.

### Profile axes and offline-algorithm ownership

An ASTC profile must name five independent axes. This prevents a nominal
bitrate from hiding a change in source, representation, or selection
objective:

| Axis | Current values | Contract |
| --- | --- | --- |
| Offline source `q` | FP16/BF16 reference; Q4, Q3, TQ research projections; future ASTC-Q8/ASTC-Q16 and few-level sources | `q` is an encoder-only latent. It is not resident alongside ASTC payloads and does not add runtime bits/weight. |
| Representation | scalar; scalar-anchored gauge L+A; PV-lite gauge; future `c + gauge` | Scalar is mandatory as the `delta = 0` fallback within the *same* source and footprint family. |
| ASTC footprint | 4x4, 5x5, 6x6; experimental 8x6, 10x6, 8x8, 10x8 | The footprint alone fixes physical ASTC storage rate: `128 / (width * height)` bits/weight for one logical value per texel. |
| Selector | candidate generation, exact CPU decode, conflict-aware commit, validation prefix | Selection is scoped per tensor, source, representation, footprint, corpus, and artifact version. Calibration never directly chooses an exported final prefix. |
| Encoder profile | `standard-thorough`; experimental neural-recall | Both emit ordinary legal 16-byte ASTC blocks. `standard-thorough` is the reference; neural recall is opt-in until it wins reproducibly on held-out tensors. |

The current product boundary is deliberately narrower than the research matrix:
FP16-derived scalar/gauge profiles at 4x4, 5x5, and 6x6 are the only standard
scheduler candidates, each with an independent selector and hard Q4_K_M/Q3_K_M
fallback. Q4/Q3/TQ sources, PV-lite, neural recall, and all footprints from
8x6 onward remain experimental encoder inputs or explicit replay profiles;
their source quantization is never a hidden runtime sidecar.

`astc-vulkan-hash.{h,cpp}` owns the stable FNV-1a cache/metadata fingerprint.
It is intentionally non-cryptographic; provenance remains SHA-256.
`astc-vulkan-gauge.{h,cpp}` owns the deterministic scalar-anchored gauge and
PV-lite factor families. Its candidate ordering is artifact-visible because it
participates in deterministic tie breaking.
`astc-vulkan-block-ldlq.{h,cpp}` owns the documented order/configuration
surface for the offline Block-LDLQ experiment. The matrix implementation
remains local to the latent smoke until its private capture/candidate data
model is extracted as a coherent selection core; this avoids exposing a
misleading half-runtime API.

Relevant sources are Fowler/Noll's [FNV reference](http://www.isthe.com/chongo/tech/comp/fnv/),
[PV-Tuning](https://arxiv.org/abs/2405.14852),
[QuIP# / LDLQ](https://arxiv.org/abs/2402.04396), and
[GPTVQ](https://arxiv.org/abs/2402.15319). These are offline encoder methods;
Vulkan continues to sample standard ASTC through fixed-function decoding.

At a later mixed-rate stage, source/representation changes should begin per
tensor or page. Mixed footprints may then be selected per macro-tile only
within compatible row-height classes (H4: 4x4; H5: 5x5; H6: 6x6/8x6/10x6;
H8: 8x8/10x8). Per-ASTC-block mixed footprints remain out of scope because the
metadata, address translation, and shader branching can erase the bandwidth
gain.

After the active full-tensor run, first profile its 8.19-second strip
generation time into source construction, ASTC search, payload packing/dedup,
decode, reconstruction, and activation-delta construction. Then audit only the
hit rate of an exact source-block cache. Preserve source data bitwise in such a
cache, including footprint, deterministic edge padding, gauge family, and
encoder configuration. Do not build a cache policy until the hit-rate audit
justifies it.

The current full run does not retain source gauge-factor identities after its
per-strip candidate dictionary is released. Therefore it cannot report the
selected-δ distribution retroactively. Add factor metadata to candidate,
local-positive, and committed counters for the next fulltensor/cross-tensor
run; use those measurements, rather than a guessed grid, before introducing a
coarse-to-fine adaptive gauge search.

### Current checkpoint after the first five gates

The runtime and research gates through the side-by-side driver PoC are now
complete. The Intel UHD 620 is the local ASTC device target; the NVIDIA
GeForce 920MX is a valid Vulkan device but does not expose sampled ASTC
formats on this host and must take the capability-gated fallback path.

The next work is intentionally model-level rather than more sampler plumbing:

1. Build a logits/loss replay harness from the existing FP16 reference model.
2. Compare FP16, Q3, Q4, TQ1/TQ2, scalar ASTC, and gauge ASTC on the same
   calibration and evaluation inputs.
3. Run `c + delta` only with scalar anchoring and an exact scalar fallback.
4. Revisit Block-LDLQ/GPTVQ as an offline selector per tensor, not a global
   default.
5. Keep full transformer scheduler integration and upstreaming deferred until
  the model-level quality gate demonstrates a repeatable benefit.

The next gate has now been partially exercised. The model replay tool has an
explicit `--cpu-only` control and reports logits/loss for the full FFN-down
override. Full-model Vulkan replay on this host currently reaches a device-loss
boundary, so CPU replay is the quality oracle while isolated ASTC Vulkan FFN
decode remains the GPU correctness oracle. A first bounded `c + delta` sweep
also exists, but gauge-only remains better on its initial fixture; do not make
semantic correction the default until it wins on disjoint validation and
holdout traces.

The first footprint-level logits replay is now available as a diagnostic
control. It confirms the expected 4x4 quality lead and provides a reproducible
6x6 model-level loss signal, but the two-token prompt is insufficient for a
format-quality claim. The next quality run must use a larger held-out corpus
and compare against native Q3/Q4/TQ baselines under the same CPU-only replay
contract.

The first same-prompt native logits controls are now available for Q4_0,
Q3_K_M, TQ1_0, and TQ2_0. They are retained as diagnostics alongside ASTC
4x4/5x5/6x6 layer replay. The next quality gate is a larger, disjoint token
corpus; short-prompt layer replay must not be promoted to a model-wide
quality claim.

### Execution checkpoint after the AVX2 and replay sweeps

The host-side encoder has an exact-preserving AVX2 performance variant. It
matches the native candidate pool, payload decisions, commit sequence, and
loss on the bounded gauge fixture while reducing wall time by about 12x. Keep
this as a build/performance axis; it must not change the frozen quality
baseline, and ARM builds need their own ISA selection.

The next quality gate is blocked by the model-replay harness rather than the
ASTC codec: `--cpu-only` still initializes the existing Vulkan backend on this
host and longer full-layer prompts do not reach a stable logits result. The
plan therefore requires a genuinely CPU-only replay (or a known-good Vulkan
device) before a larger perplexity/logits comparison is considered valid.
The side-by-side ASTC Vulkan driver and isolated FFN GPU decode remain valid
and continue to be tested independently.

The immediate five-sweep order is:

1. Make CPU-only logits replay deterministic for a multi-prompt held-out
   corpus.
2. Compare FP16, Q4, Q3, TQ1/TQ2, scalar ASTC, and gauge ASTC on that corpus.
3. Re-test scalar-anchored `c + delta` per tensor with scalar as exact
   fallback.
4. Run the three-way local/coordinate/Hessian selector comparison on one
   fixed legal candidate pool per tensor.
5. Profile native/AVX2 encoder generation and measure the side-by-side Vulkan
   path against the existing buffer path, without changing production
   `ggml-vulkan`.

### Model replay gate update

The `--cpu-only` replay path now supplies an explicit CPU device list, and a
Vulkan-free PoC build provides a stable quality oracle. Full-layer SmolLM2
replay with a seven-token prompt is working. On the independent holdout trace,
6x6 scalar ASTC had loss delta `+8.2563765`, while the scalar-anchored gauge
stream reduced it to `+4.8854536`. This is meaningful mechanism evidence but
still a one-layer/short-corpus diagnostic; it is not yet a model-wide quality
claim.

The same full-layer run recorded scalar 4x4/5x5 deltas of `+7.9402708` and
`+7.9460386`, and native controls for Q4, Q3, TQ1, and TQ2. The short-prompt
loss/MSE disagreement keeps the larger corpus gate mandatory. The next step
is to add a deterministic multi-prompt corpus contract, then rerun all
formats and per-tensor c+delta/selector comparisons before any scheduler or
upstream decision.

### GPU and selector follow-up

The ASTC Vulkan path and the F32 buffer control both pass on the Intel UHD
620. A small fixture showed a modest ASTC timestamp advantage, but the full
576x1536 layer was about 4% slower with ASTC sampling. The stable measured
benefit is therefore the roughly 8.9x lower weight payload (384 KiB versus
3.38 MiB), not a portable compute-speed claim. Keep target-GPU benchmarking
as a later driver gate and do not change scheduling based on this host result.

The c+delta, neural-rank, and selector-compare CTests pass together (3/3).
These remain offline/opt-in controls. The next blocking item is still a
genuinely CPU-only, multi-prompt logits harness; only after it is stable may
the format and per-tensor selector results be promoted to a model-level
comparison.

The full-layer scalar-anchored c+delta gate is now measured. It reduces the
6x6 scalar replay loss delta from `+8.2563765` to `+7.0543591`, but gauge-only
remains better at `+4.8854536`; validation also selects an early c+delta
prefix. Keep c+delta as an opt-in per-tensor candidate family with scalar
fallback, not as the default representation. The next required experiment is
the larger multi-prompt corpus followed by the fixed-pool three-way
local/coordinate/Hessian comparison.

### Pre-driver quality gate completion

The pre-driver quality gate has now been extended to a second tensor. Full
`blk.1.attn_output` gauge-only selection on `576x576` reached holdout
activation-relative MSE `0.057085516` from a scalar control of `0.064070374`,
with validation-prefix holdout `0.045835827`. A bounded layer-1 c+delta run
reached `0.041410601` after validation stopping, but gauge-only remained
better at `0.036094848` on that crop. These measurements confirm that the
representation is useful but tensor-sensitive; no global default is implied.

The fixed-pool layer-1 selector gate also completed. Conflict-aware selection
was best on holdout (`0.28072521`), while coordinate descent (`0.29552618`),
Block-LDLQ (`0.29555012`), and Hessian feedback (`0.30294451`) did not
generalize better on this split. Keep conflict-aware selection as the current
research default and retain the other selectors as opt-in baselines.

The larger multi-prompt/model-wide corpus is still an explicit future gate,
because the present replay traces are activation-space controls rather than a
perplexity benchmark. It is not a blocker for beginning the isolated driver
expansion: the driver must consume standard ASTC payloads and preserve the
quality oracle and fallback contracts. The next implementation phase is:

1. [ ] Add the ASTC driver-side manifest/atlas upload path beside the existing
   PoC, without changing production `ggml-vulkan`.
2. [ ] Add capability-gated sampled ASTC format selection and a deterministic
   F32/storage-buffer fallback.
3. [ ] Add one shader matvec/FFN dispatch over the manifest, with explicit
   synchronization and readback contracts.
4. [ ] Run the Intel ASTC target, NVIDIA fallback, and CPU/reference paths in
   one CTest matrix; record bandwidth/dispatch timings separately from model
   quality.
5. [ ] Only after the isolated driver matrix is green, evaluate scheduler
   integration and an upstream-facing API proposal.

### Driver expansion checkpoint

The first two driver-expansion steps are complete. Format arithmetic is now a
standalone module, manifest version 2 records representation and affine
reconstruction metadata, and payloads can be checked with an optional FNV-1a
integrity hash. The upload/session path rejects size or hash mismatches before
allocation and the FFN adapter propagates manifest reconstruction parameters.

The capability gate has been exercised on the local devices: Intel UHD 620
supports sampled ASTC 4x4/5x5/6x6, NVIDIA GeForce 920MX and llvmpipe do not.
The resource smoke passes on Intel, while the existing fallback behavior is
preserved for unsupported devices. The production `ggml-vulkan` backend is
unchanged.

Before writing the dispatch abstraction, move the plain reconstruction data
structure next to the manifest contract and define its stable shader layout.
Then implement one FFN dispatch with explicit `texelFetch`/nearest semantics,
storage-buffer barriers and a CPU-oracle tolerance. Keep atlas, bindless
descriptors, multiple pages, and scheduler integration deferred until this
single-tensor dispatch is byte/metric reproducible.

### First dispatch checkpoint

The reconstruction data and matvec push-constant layout are now shared by the
manifest/resource contract and `astc-ffn-matvec.comp`; the C++ layout is
compile-time checked at 24 bytes. A full layer-0 GPU dispatch on Intel UHD 620
passes the CPU-oracle comparison for both ASTC 4x4 and 6x6 (GPU-vs-CPU MSE
below `3e-13`). This validates the isolated sampled-image path, but does not
yet justify scheduler integration or a portable performance claim.

Next, add explicit fallback/error-path tests to the sidecar matrix, then wrap
the upload plus dispatch lifetime in a small RAII session. Keep the existing
FFN smoke as an oracle-facing executable during this transition. Atlas/pages,
descriptor indexing and integration with production `ggml-vulkan` remain
later gates.

The final pre-RAII review added the remaining correctness guards: manifest v1
is fallback-only, tensor width and height are both checked, duplicate names
and payload-blob bounds are rejected, and staging supports non-coherent
host-visible memory with an explicit flush. The FFN output buffer now stores
one scalar F32 per result rather than an unused `vec4`. Device preflight checks
exact format features, image extent limits and valid queue/device handles.

All focused host tests pass (`6/6`), Intel ASTC resource smoke passes, and the
metadata-backed full-layer 6x6 dispatch continues to match the CPU oracle
within `2.5e-13` MSE. The contract-discovery phase is complete. Proceed with
the contained RAII dispatch/session extraction; do not add scheduler or
upstream integration until the extracted session reproduces the existing E2E
smoke and fallback matrix.

The fallback/error-path gate is now green. Host-only tests reject malformed
payloads before Vulkan allocation, preserve deterministic unsupported-device
fallback, and cover manifest v1 compatibility plus v2 reconstruction/hash
metadata. The E2E smoke passes the manifest-derived metadata and payload hash
through the adapter; Intel ASTC dispatch still matches the CPU oracle below
`3e-13` MSE.

The next implementation item is the RAII dispatch/session extraction. It must
retain one sampled image, one descriptor set, one pipeline, and explicit
activation/output buffers with the current synchronization behavior. Keep the
existing smoke as a reference implementation until the extracted object has
the same output and fallback behavior. No scheduler integration or upstream
API change is part of this gate.

The final pre-RAII checkpoint is green: a metadata/hash-backed ASTC 4x4 E2E
dispatch matches the CPU oracle at `2.6461919e-13` MSE, and the six focused
driver/contract tests pass. The sidecar now has a stable single-tensor driver
boundary for both 4x4 and 6x6. Proceed next with the contained RAII
dispatch/session extraction; keep production `ggml-vulkan`, atlas policy and
scheduler integration out of that change.

### Driver expansion implementation checkpoint

55. [x] Extract the sidecar compute lifetime into a reusable RAII
    `astc_vulkan_matvec_session`. Keep resource ownership, descriptor binding,
    push constants, command recording, synchronization, and host readback in
    one implementation; do not modify production `ggml-vulkan` code.
56. [x] Route the FFN E2E smoke through the shared session and preserve the
    scalar GPU-output contract. Repack activation traces with a wider source
    stride into the session's contiguous input buffer.
57. [x] Run the session on ASTC 4x4 and 6x6 with the Intel render device and
    compare GPU output against the CPU oracle. Treat no-ASTC devices as a
    capability-gated skip and keep fixture quality metrics separate from the
    dispatch correctness gate.
58. [x] Add host-only session error-path tests for invalid configuration and
    safe reset. Add device-backed tests for dimensions, missing shader words,
    tensor/image shape mismatch, and repeat-run behavior once the harness can
    obtain a sampled ASTC device without privileged setup.
59. [x] Add a thin sidecar driver facade that owns instance/device/queue
    selection and composes manifest loading, tensor upload and matvec sessions.
    Keep it opt-in and parallel to llama's existing Vulkan backend.
60. Add a scheduler-facing adapter experiment behind an explicit build option;
    compare scalar ASTC, gauge-only ASTC, Q4/TQ controls and FP16 using the
    same activation traces and model outputs before any upstream proposal.

The isolated sidecar facade is now also exercised by the FFN E2E executable;
its 4x4 and 6x6 GPU outputs match the CPU oracle below `2e-14` MSE on the
render-enabled Intel device. Begin item 60 with an offline/single-tensor
comparison harness. It must not require scheduler changes and must report
dispatch correctness, storage bytes and activation/model quality as separate
metrics.

The first real-tensor comparison baseline is recorded in the WIP log: on a
32-row Pythia F16 FFN-down crop, Q4_0 is `0.0034055566` activation-relative
MSE, while the generic block-affine/Hadamard ASTC controls are `0.19758811`
(4x4) and `0.4618935` (6x6). Keep this control distinct from gauge-only ASTC;
item 60 must compare each representation from the same tensor and traces.

The first full-tensor Q4/TQ controls are now available on Pythia
`blk.0.ffn_down.weight` (2048x8192): Q4_0 `0.0015319726`, TQ1_0 and TQ2_0
`0.24722138` activation-relative-MSE, with FP16 as the zero-error reference.
Proceed with ASTC payload export/replay for this same shape before introducing
any scheduler-facing build option.

The same-tensor ASTC gate is now also complete for the bounded 32x8192 crop:
scalar ASTC holdout `0.0219951`, conflict-aware gauge holdout `0.012050252`,
and validation-stopped gauge holdout `0.013314081`. The selected gauge stream
replays through the Intel sidecar with GPU-vs-CPU MSE `2.4764415e-15` and the
same activation error. Implement item 60 as an explicit opt-in comparison
executable; keep scheduler integration and upstream changes deferred.

The opt-in `astc-vulkan-sidecar-compare-smoke` is now implemented and has
replayed scalar and gauge payloads on the same Pythia crop. With 30 holdout
samples it reports scalar `0.026625491` versus gauge `0.016833613` activation
relative MSE, while both GPU results match the CPU oracle below `5e-14` MSE.
Extend this tool next with a manifest-driven artifact set and model-output
comparison; do not wire it into the production scheduler yet.

The sidecar comparison path now reuses its initialized dispatch session for
identical shader/sample-count runs. This removes repeated descriptor/pipeline
setup without changing payloads or numerical contracts. The next item is to
make the comparison artifact set manifest-driven and add a model-output
reference metric before any scheduler-facing adapter is attempted.

The manifest-backed replay gate is now green on the Pythia 32x8192 crop. With
30 holdout samples, scalar and gauge report model-output relative MSE
`0.026655061` and `0.016833603`, while GPU-vs-CPU dispatch MSE remains below
`3e-15` for both. Treat this as a layer-output gate; full-model token/logit
comparison is still a separate later step using the existing model-replay
harness.

`astc-vulkan-artifact-pack` now provides the repeatable packaging step from
exported ASTC + metadata files to a v2 manifest and payload blob, including
shape, representation, affine decode and checksum. The packaged scalar/gauge
Pythia crop replays successfully through the sidecar with model-output
relative MSE `0.026625414` and `0.016833603`, respectively. Proceed next with
larger/full-tensor artifacts and keep full-model logits as a separate gate.

The complete ASTC-labelled regression suite is green after this work (33/33;
device tests are capability-gated skips outside the privileged render-device
run). The next remaining driver-plan item is the explicit scheduler-facing
adapter experiment, but it must first consume larger/full-tensor manifest
artifacts and compare full-model outputs; the production `ggml-vulkan` path
remains untouched.

The comparison executable now supports v2 manifest + payload-blob inputs for
scalar and gauge artifacts, with normal range/checksum validation, and an
optional F32 reference-output metric. It retains the old explicit-file mode
for regression compatibility. The sidecar validates activation divisibility
before session reuse. Next, create/replay real manifest-backed scalar and
gauge artifacts, then use the resulting metrics as the gate for any
scheduler-facing adapter experiment.

The full layer-0 gauge artifact is now packaged and replayed: `2048x8192`,
7,474,752 bytes, payload SHA-256
`eb8cf938307703012900453dd4767bcb09433e3ab9f2a083c4732c1190ac0d54`.
Full-model CPU replay reports relative logits MSE `0.75192497`, top-1
agreement `10%`, and loss delta `+4.4685255`; the existing full additive 6x6
control reports `0.90135289`, `0%`, and `+4.5750505`. Gauge improves the
control but remains well behind Q4 quality.

The isolated scheduler-adapter experiment is implemented behind
`GGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER`; its host contract and full
tensor Intel dispatch smoke are green. The adapter is deliberately parallel
to the production backend. The remaining gate before any upstream proposal
is a same-prompt/full-model comparison matrix including FP16, Q4, TQ1/TQ2,
scalar ASTC, and gauge ASTC, followed by performance measurements.

### Format expansion gate: Q3/Q4/TQ controls and ASTC 8x6/8x8

The next evaluation round keeps the current scalar-anchored ASTC 6x6 path
and adds a controlled density ladder. `Q4_K_M` remains the practical main
baseline; `Q3_K_M` is the closest conventional low-bit comparison for ASTC
6x6/8x6; `TQ2_0` and `TQ1_0` are few-level/ternary controls. `FP16` remains
the source and quality oracle rather than a target quantizer.

Add ASTC `8x6` (2.6667 bits/value) and `8x8` (2.0 bits/value) as separate
footprint experiments. Do not mix them into the 6x6 quality baseline until
each has its own scalar and gauge-only artifact. The intended comparisons are:

* ASTC 6x6 (~3.56 b/v) against Q3_K_M and Q4_K_M.
* ASTC 8x6 (~2.67 b/v) against Q3_K_M and TQ2_0.
* ASTC 8x8 (2.0 b/v) against TQ2_0 and TQ1_0 as an aggressive research rung.

For every representation, keep the same FP16 source tensor, calibration /
validation / holdout traces, prompt corpus, tensor shape and random seed. Report
storage rate, tensor activation error, model-output/logit error and GPU dispatch
time as separate metrics. ASTC must be reported both without gauge selection
(scalar codec control) and with gauge selection; the latter is an optimizer
comparison as well as a format comparison.

Implementation order:

1. Add format-aware footprint arithmetic and edge-mask tests for 8x6 and 8x8.
2. Export/replay scalar ASTC 8x6 and 8x8 on the existing Pythia tensor.
3. Add gauge candidate generation and conflict-aware selection for both
   footprints, preserving scalar fallback and deterministic padding.
4. Extend the isolated Vulkan shader/session to accept the two new ASTC block
   sizes, with capability-gated tests (`VK_FORMAT_ASTC_8x6_UNORM_BLOCK` and
   `VK_FORMAT_ASTC_8x8_UNORM_BLOCK`).
5. Add same-tensor Q3_K_M, Q4_K_M, TQ2_0 and TQ1_0 replay controls, then run
   the fixed prompt/model-output matrix.
6. Only after the quality gate is reproducible, benchmark cold upload,
   hot-cache dispatch and batched inference on the target device.

The driver must remain format-generic at the manifest/resource boundary. No
production `ggml-vulkan` changes, direct Q-format reinterpretation as ASTC,
or scheduler integration belongs to this phase. If a device lacks an ASTC
footprint, the adapter must fall back deterministically to the existing path.

### Experimental-footprint implementation update

The first implementation slice for the density-ladder expansion is complete
in the sidecar. `astc_vulkan_footprint` now covers `8x6` and `8x8`, including
block arithmetic, artifact-pack/decode parsing, tensor contracts, Vulkan
format mapping, and atlas placement. The new footprints are explicitly
classified as experimental by `astc_vulkan_footprint_is_experimental()`.

The device policy is deliberately asymmetric:

* 4x4, 5x5, and 6x6 remain the standard compatibility gate.
* 8x6 and 8x8 are probed independently and exercised only when the selected
  device advertises the required sampled-image and transfer features.
* The sidecar requires an explicit `allow_experimental` opt-in for these
  footprints; a caller that does not opt in receives a deterministic error and
  can use the normal buffer-backed fallback.

This keeps experimental support in the parallel `pocs/astc-vulkan` boundary;
no production `ggml-vulkan` code or GGUF semantics are changed. The chunked
row-strip selector also no longer assumes a 36-texel block, so 8x6 (48 texels)
and 8x8 (64 texels) can use the same streamed candidate path without a buffer
overrun. Edge padding remains deterministic and excluded from neural scoring.

The capability/resource/shader smoke results on this host are:

| Device | 8x6 | 8x8 |
| --- | --- | --- |
| Intel UHD Graphics 620 | sampled/upload/shader pass | sampled/upload/shader pass |
| NVIDIA GeForce 920MX | unsupported; fallback | unsupported; fallback |
| llvmpipe | unsupported; fallback | unsupported; fallback |

The next quality gate is unchanged: export scalar and scalar-anchored gauge
artifacts for 8x6 and 8x8, then compare them with the same FP16 source and
traces against Q3_K_M, Q4_K_M, TQ2_0, and TQ1_0. Device smoke proves only the
standard Vulkan resource contract; it is not a portable performance claim for
Mali, Adreno, or Apple GPUs.

### Cross-tensor result and revised gate

Repeating the 8x6/8x8 matrix on `blk.1.ffn_down.weight` confirms that the
steering family is tensor-dependent. Conflict-aware selection improves every
source in the bounded screen, but the magnitude ranges from small FP16 gains
to large TQ gains, and full conflict selection can overfit calibration. The
quality gate therefore requires the validation-stopped prefix and an untouched
holdout for every tensor; full conflict selection is retained only as a
diagnostic oracle.

Before scaling to a full tensor or adding Block-LDLQ, do the following:

1. Export scalar and validation-stopped streams with manifests for both
   footprints and both Pythia tensors.
2. Replay/decode those artifacts and compare decoded values and activation
   metrics against the direct latent-smoke path.
3. Run the fixed-prompt output matrix against FP16, Q3_K_M, Q4_K_M, TQ2_0,
   TQ1_0, scalar ASTC and gauge ASTC. Record model-level metrics separately
   from tensor activation MSE.
4. Only if the artifact and model-level results reproduce, measure Intel Vulkan
   cold upload, hot-cache dispatch and batched execution. Keep 8x6/8x8
   experimental and opt-in; no production backend or GPU encoder changes are
implied.

### Artifact/Vulkan replay status

The scalar and gauge payload paths for 8x6 and 8x8 now pass a serialization
gate: selector payload, packed manifest payload and CPU-decode output agree
byte-for-byte/float-for-float where applicable. Intel UHD Graphics 620 also
passes the isolated FFN e2e contract for both footprints with GPU-vs-CPU MSE
below `6e-15` over four samples. This validates the current shader/resource
plumbing only; it does not establish mobile-GPU performance or model quality.

The next gate is model-facing rather than more driver plumbing: run the fixed
prompt/output comparison for FP16, Q3_K_M, Q4_K_M, TQ2_0, TQ1_0, scalar ASTC
and validation-stopped gauge ASTC. Keep full conflict paths as diagnostics,
not as ship candidates, because calibration overfit is visible in the current
screens. Only after that comparison should we benchmark cold upload,
hot-cache and batched inference.

### Latest quality gate: density-ladder controls on Pythia

The first comparable screen now covers FP16, Q3_K_M, Q4_K_M, TQ2_0 and TQ1_0
as source/control files for ASTC 8x6 and 8x8. It uses one tensor and one fixed
trace family, with scalar ASTC as the exact fallback and scalar-anchored gauge
selection as the experimental variant. The crop is intentionally bounded
(`8x2048`) so that all five source files can be compared without turning this
gate into an overnight full-tensor run.

The screen shows meaningful conflict-aware holdout reductions for every source
at both footprints, but also strong calibration/holdout separation. Therefore
the plan now treats validation stopping and an untouched holdout as mandatory,
and reports full conflict selection only as a diagnostic upper-bound path.
The TQ1/TQ2 equality on this tensor is recorded as an observation of the
current loader/crop, not as a claim that the formats are interchangeable.

The next implementation/evaluation order is:

1. Package the scalar and validation-stopped gauge streams as manifest-backed
   8x6/8x8 artifacts and replay them through the CPU oracle.
2. Run the same artifact/replay protocol on a second Pythia tensor (and retain
   the existing 6x6/Q3/Q4/TQ controls) to separate tensor effects from format
   effects.
3. Add the fixed-prompt/model-output matrix: FP16 reference, Q3_K_M, Q4_K_M,
   TQ2_0, TQ1_0, scalar ASTC and gauge ASTC.
4. Only after the quality artifacts are reproducible, run Intel Vulkan upload,
   hot-cache and batched dispatch measurements. Keep 8x6/8x8 opt-in and do not
   change production `ggml-vulkan` or reinterpret GGUF quant bytes as ASTC.

### Low-bitrate artifact and rate--distortion contract

Gauge steering is now treated as a codec-navigation mechanism layered above a
source weight field `q`, not as an FP16-specific residual codec:

```
q -> (L = q + delta, A = q - delta) -> legal ASTC payload -> runtime decode
```

`q` may be FP16-derived scalar, Q4, Q3, or a few-level/TQ source. A source
format screen alone is not a model-quality claim, however: comparisons against
Q3/Q4/TQ must also be made against a shared FP16 reference and fixed prompt
corpus before a final rate/quality conclusion is drawn.

The exported **validation-selected payload is the oracle**. Full conflict
commit remains a diagnostic path only and must never silently become the
payload used by artifact, Vulkan, or model-facing replay. The selector export
therefore needs a materialized snapshot at the validation-selected commit
prefix, with all later commits excluded.

Every manifest-backed artifact must contain, or be bound to by deterministic
sidecar metadata:

* tensor dimensions and tensor name;
* ASTC footprint and physical bits per weight;
* source quantization family and source-model fingerprint;
* affine/runtime decoder constants and representation kind;
* calibration, validation, and selection-contract hashes;
* validation-selected commit count and deterministic commit-order hash;
* edge-padding/masking contract version; and
* payload byte size and SHA-256.

CPU replay must decode the exported bytes directly and reproduce the selector
result. Vulkan replay must consume that same payload, never a regenerated
encode. This makes padding geometry, selection prefix, serialization, and
runtime reconstruction independently auditable.

The low-bitrate decision gate is a common rate--distortion matrix, using a
shared FP16 tensor/reference and fixed calibration, validation, and untouched
holdout traces:

| Footprint | Physical rate | Required controls |
| --- | ---: | --- |
| 4x4 | 8.00 b/w | scalar and gauge + validation |
| 5x5 | 5.12 b/w | scalar and gauge + validation |
| 6x6 | 3.56 b/w | scalar and gauge + validation |
| 8x6 | 2.67 b/w | scalar and gauge + validation, experimental opt-in |
| 10x6 | 2.13 b/w | scalar and weight-grid gauge + validation, experimental opt-in |
| 8x8 | 2.00 b/w | scalar and gauge + validation, experimental opt-in |

Report scalar and validation-selected gauge holdout MSE side by side for every
row, then add model-output metrics separately. The concrete low-rate success
criterion is not merely an improvement within one footprint: for example, an
8x6 gauge artifact approaching scalar 6x6 quality at lower resident rate would
move the practical rate--distortion frontier. Only artifacts that pass this
gate proceed to target-device upload and dispatch measurements.

The validation-prefix exporter is now implemented in the isolated latent
smoke. It materializes a payload and decoded-reference snapshot at the best
validation commit, while retaining the full conflict stream only for
diagnostics. This closes the largest artifact-contract gap; the next check is
to package and replay these prefix snapshots with manifests, then run the
common rate--distortion and fixed-prompt model-output matrix.

Validation-prefix artifact replay now passes the serialization and Vulkan
contract on a bounded `8x256` Pythia tensor: packed-byte CPU decode is exact,
and the Intel FFN e2e path matches CPU with MSE `1.2027e-16`. A mismatched
`8x2048` weight file is rejected before device execution, confirming shape
validation. The next gate is to generate these prefix artifacts for the full
rate--distortion ladder and compare model outputs against a shared FP16
reference.

The first block-aligned FP16 rate--distortion ladder is now measured on
`blk.0.ffn_down.weight`: 4x4, 5x5, 6x6, 8x6 and 8x8 use footprint-compatible
crop heights, so reported rates are 8.0, 5.125, 3.5625, 2.6667 and 2.0 b/w.
Validation-selected gauge improves holdout at every rung in this bounded
screen. This is the baseline for the next Q3_K_M/Q4_K_M/TQ1_0/TQ2_0 matrix;
those controls must use the same aligned geometry and a common FP16 reference.

The aligned Q3/Q4/TQ control sweep is complete for the planned low-rate rungs:
6x6 against Q3/Q4/TQ, 8x6 against Q3/Q4/TQ, and 8x8 against TQ1/TQ2. The
results are retained as activation-level controls only. The next required
comparison is model-output/logit quality against one shared FP16 reference,
using the materialized validation-prefix artifacts; no footprint is promoted
from experimental status based on activation MSE alone.

### Gap review: gates before model-level claims

Before starting full-model comparisons, the following contracts are explicit
requirements for every exported ASTC artifact:

1. **Provenance is separate from the runtime manifest.** Keep the Vulkan
   manifest lean, but ship a hash-bound provenance sidecar containing the
   source model/tensor fingerprint, source family (`FP16`, `Q3_K_M`,
   `Q4_K_M`, `TQ1_0`, or `TQ2_0`), footprint, representation, decoder
   constants, calibration/validation/holdout trace hashes, selector
   configuration, validation-selected commit count, commit-order hash,
   padding/mask version, payload byte count, and SHA-256. Existing FNV-1a is a
   fast stale-payload check, not a cryptographic identity.

2. **Baselines are separated by purpose.** Every model-facing table must show
   (a) the shared FP16 reference, (b) an FP16 model with the tested layer
   replaced by a dequantized Q/TQ source, (c) scalar ASTC from that same source,
   and (d) validation-selected gauge ASTC from that same source. This separates
   FP16-to-source-quantization error from source-to-ASTC error.

3. **Crops are activation screens, not model claims.** A cropped tensor may be
   used for fast selector/rate screening. Logit or perplexity claims require
   the complete layer shape and a complete artifact; the runtime must reject
   shape-incompatible weights rather than truncate or reshape them.

4. **Validation is a fixed, disjoint protocol.** Calibration, validation, and
   untouched holdout traces must be disjoint prompt/token shards with fixed
   seeds. Validation stopping may select the exported prefix, but holdout is
   read-only for tuning. Where feasible, aggregate several validation shards
   and report shard variance.

5. **Rate is measured on the real tensor.** Report payload bits divided by the
   number of real (non-padding) weights, plus separately reported metadata and
   edge-padding overhead. Footprint-aligned crops define the screening axis;
   only full-tensor byte counts define the resident-rate claim.

6. **Hardware results are scoped.** Intel UHD 620 Vulkan replay currently
   establishes correctness only. Device feature gates and scalar fallback are
   mandatory. Performance claims require a target mobile GPU, matched
   shader/workgroup settings, cold/hot and batched measurements, and comparison
   with native Q/TQ buffer kernels. Successful texture decode does not imply a
   GPU ASTC encoder.

7. **TQ1/TQ2 equality is a diagnostic.** If a bounded crop produces identical
   TQ1/TQ2 decoded values, retain both format labels but verify their full-tensor
   dequantized bytes and GGUF fingerprints before attributing a quality or rate
   advantage to either format.

The immediate next gate is a reproducible full-layer artifact matrix with these
controls, followed by model-output replay. Vulkan replay must consume the exact
selected payload bytes and sidecar identity; it must not regenerate or reselect
candidates at runtime.

### Current execution order after the gap review

The next sweeps are deliberately incremental:

1. Keep the corrected CPU-only model baseline harness and run a fixed
   multi-prompt control matrix for FP16, Q3, Q4, TQ1, and TQ2.
2. Materialize full-layer scalar and validation-selected ASTC artifacts with
   the provenance sidecar fields above; replay the exact bytes through the CPU
   oracle and model-output harness.
3. Add the same source-family controls to the model-output table. Treat the
   existing bounded crop results as screening evidence only.
4. Only when replay and provenance are reproducible, measure Vulkan cold upload,
   hot-cache dispatch, and batching on the available device. Keep 8x6/8x8
   experimental and retain deterministic fallback.
5. Defer `c+delta`, Block-LDLQ, search-preset pruning, and GPU encoding until
   the full-layer matrix establishes a trustworthy baseline.

### Artifact replay checkpoint

The existing manifest-backed `32x8192` Pythia artifact was replayed from its
packed blobs (not regenerated ASTC bytes) against the 30-sample holdout trace.
Scalar and gauge both matched the CPU oracle on Intel Vulkan (`GPU-vs-CPU MSE`
`1.79e-15` and `2.82e-15`); activation/model-output relative MSE was `0.026655061`
for scalar and `0.016833603` for gauge. This is a successful serialization and
device-correctness gate for the bounded tensor, not a full-model quality claim.
The next artifact must add the provenance sidecar and full-layer shape before
the same comparison is promoted to logits/perplexity evidence.

### Full-layer replay reproducibility note

The existing raw layer-0 gauge payload can be independently decoded and
replayed: the `2048x8192` payload is 7,474,752 bytes with SHA-256
`eb8cf938307703012900453dd4767bcb09433e3ab9f2a083c4732c1190ac0d54`, and its
decoded RGBA-F32 fixture is 256 MiB. A replay with the current 11-token prompt
and holdout trace produced relative logits MSE `0.80219534` and loss delta
`+6.6148289`. The earlier `0.75192497` / `+4.4685255` result used a different
prompt/trace pair and must remain a separate historical diagnostic. This
non-equivalence is precisely why the next full artifact must bind corpus,
trace hashes, decoder metadata, and payload identity in its provenance record.

### Native low-bit export rate note

The isolated quant-export tool now supports `f32`, `q4_0`, `tq1_0`, and
`tq2_0`. On the aligned `8x2048` Pythia control, the packed sizes were
`3,456` bytes for TQ1 (`1.6875` bits/weight) and `4,224` bytes for TQ2
(`2.0625` bits/weight). These are physical packed rates including block
overhead, not ideal ternary entropy. Future rate tables must use measured
native rates rather than theoretical format labels.

### Provenance sidecar implementation checkpoint

`astc-vulkan-artifact-pack` now accepts `--provenance` and emits a deterministic
offline sidecar containing source model/tensor identity, source quantization
family, ASTC footprint and representation, affine decoder constants,
calibration/validation/holdout identifiers, selector configuration,
validation-selected prefix, commit-order and padding contracts, payload size,
and SHA-256 hashes for the payload and manifest. The compact runtime manifest
and Vulkan shader contract are unchanged. The sidecar writer is isolated in
`astc-vulkan-provenance` and has its own ASTC-labelled contract test; the full
suite is green at 30/30 after the change.

The sidecar is now a prerequisite for full-layer quality artifacts. Real trace
and commit-order hashes must be supplied for production measurements; synthetic
values are allowed only in the serialization test. This keeps future 8x6/8x8,
Q3/Q4, and TQ comparisons replayable without allowing provenance machinery to
alter candidate selection or runtime execution.

### Real provenance replay checkpoint

The bounded `32x8192` Pythia gauge stream was repackaged with real calibration,
validation, holdout, source-weight, and commit-log hashes and validation prefix
`283`. The payload remained byte-identical, and packed manifest/blob replay on
Intel Vulkan matched the CPU oracle at `2.8224076e-15` MSE over 30 samples.
This is an artifact-identity/runtime gate only; the same gauge payload occupied
both comparison slots in this check, so no scalar-versus-gauge quality claim is
derived from it. The next required artifact remains a full-shape scalar and
validation-prefix gauge pair bound to the same corpus.

### Separated provenance quality replay

The bounded `32x8192` Pythia scalar and validation-selected gauge artifacts now
have separate provenance-bound manifests/blobs with identical source and trace
identity. Intel Vulkan replay over 30 holdout samples reproduced scalar
model-output relative MSE `0.026625414` and gauge `0.016833603`; GPU-vs-CPU
MSE remained at `4.49e-14` and `2.82e-15`. This closes the bounded provenance
quality gate without changing runtime code. It does not promote a full-model
claim: full-shape validation-prefix payloads are still required.

### Full-layer validation-prefix and model gate checkpoint

The full Pythia layer-0 gauge selector now materializes validation prefix
`323071` for the complete `2048x8192` tensor. Provenance-packed scalar and
gauge artifacts replayed on Intel Vulkan with CPU agreement below `1e-12` MSE;
activation-relative MSE was `0.41405234` for scalar and `0.012900798` for
validation-prefix gauge.

The same fixed 13-token prompt and FP16 model were then used for model replay.
Scalar ASTC had logits-relative MSE `0.76162122` and loss delta `+6.5926616`;
gauge had `0.76110259` and `+6.576394`. Native controls on the same prompt were
Q3 `0.13334801`, Q4 `0.036831488`, TQ1 `1.3231683`, and TQ2 `1.4566528`.
Thus gauge is a real activation-level improvement over scalar, but not yet a
model-quality replacement for Q3/Q4. The next plan gate is multi-prompt/full-
model validation and only then device timing or experimental 8x6/8x8 ranking.

### Full-layer activation replay checkpoint

The complete Pythia `blk.0.ffn_down.weight` scalar 6x6 stream is now exported
and provenance-packed (`2048x8192`, `7,474,752` bytes). Replay of the separate
scalar artifact and the existing full gauge conflict-diagnostic artifact over
30 holdout samples matched the CPU path (`9.90e-13` and `8.28e-15` GPU-vs-CPU
MSE). Activation-relative MSE was `0.41405234` for scalar and `0.012899721`
for gauge. The gauge result is not yet a validation-prefix/model claim: its
payload contains the full conflict stream while the commit log identifies
validation-best prefix `323020`. The next gate is to rerun the full gauge
selector with `--validation-payload`, package that exact prefix, and replay it
against the full scalar control.

### Post-gate regression checkpoint

After full-layer validation-prefix export, provenance packaging, CPU/Vulkan
replay, and same-prompt model controls, the complete ASTC-labelled CTest suite
was rerun with **30/30 passing** in about 131 seconds. The isolated branch is
clean and no production `ggml-vulkan` routing was modified. The next work is
therefore a multi-prompt/full-model quality matrix and target-device timing,
not another correctness fix.

### Multi-prompt and first timing checkpoint

Three fixed technical prompts were replayed through the full Pythia model using
the complete scalar and validation-prefix gauge artifacts. Mean logits-relative
MSE was `0.89747548` for scalar and `0.89152133` for gauge; mean loss deltas were
`+6.81348787` and `+6.82224413`, respectively. Gauge's activation advantage
therefore remains a small, metric-dependent model-level effect.

A cold end-to-end Intel UHD 620 Vulkan replay of both full artifacts over 30
samples took `7.14 s` wall time with peak RSS `933864 KiB`, while GPU-vs-CPU MSE
remained below `1e-12`. Treat this as POC correctness/timing only. Before any
driver promotion, add hot-cache/batched dispatch timing and a native Q/TQ
buffer-kernel comparison on the target mobile GPU.

### Hot-cache dispatch checkpoint

The sidecar smoke now supports an opt-in `--repeat N` timing mode. It warms the
uploaded resources once, then measures repeated Vulkan dispatches without
recreating the sidecar or re-uploading the artifact. On the full `2048x8192`
Pythia layer-0 pair, three repetitions on Intel UHD 620 measured `212.31454 ms`
per scalar dispatch and `211.84634 ms` per validation-prefix gauge dispatch.
Correctness and activation errors were unchanged (GPU-vs-CPU MSE below
`1e-12`). This is a device-specific warm-session reference only; it must not be
extrapolated to Mali/Adreno/Apple GPUs.

The timing option is deliberately isolated from selection and artifact format:
the same provenance-bound payload bytes are uploaded and decoded, and the
default remains one measured repetition. Focused regression tests passed 4/4.
The next gates are (1) native Q3_K_M/Q4_K_M/TQ1_0/TQ2_0 buffer-kernel timing and
quality comparison, (2) replay on a representative mobile Vulkan device, and
(3) only after those measurements, deciding whether a neural-gauge-specific
ASTC encoder or GPU encoder is justified.

### Native shader-control checkpoint

On the Intel UHD 620, a 100-dispatch shader timestamp using the same `32x1536`
fixture measured `8.462 us` for ASTC 6x6 sampled matvec, `6.247 us` for the
FP32 storage-buffer control, `15.515 us` for Q4_0, and `13.643 us` for TQ2_0.
All four paths passed their respective CPU/oracle checks. These values are
useful for validating the separated shader contracts and for selecting the
next fixture size, but they are not portable performance claims and do not
replace native ggml Q3_K_M/Q4_K_M/TQ1_0/TQ2_0 graph measurements.

The next implementation/evaluation order is now fixed: enlarge the tiled
fixture and add the remaining Q3/TQ1 controls where the shader contract exists;
run quality and hot-dispatch measurements from the same artifact provenance;
then replay on a representative mobile Vulkan adapter. Keep GPU ASTC encoding
deferred until those measurements show that host candidate generation, rather
than shader dispatch, is the limiting cost.

### Larger-fixture timing checkpoint

The unified 100-dispatch comparison was repeated on the common SmolLM2 fixture
using the correct orientation (`512` columns × `576` rows). Per-dispatch GPU
timestamps were ASTC 4x4 `210.015 us`, ASTC 5x5 `151.299 us`, ASTC 6x6
`123.699 us`, FP32 buffer `108.608 us`, Q4_0 `183.719 us`, and TQ2_0
`243.029 us`; every path passed its CPU/oracle validation. The corrected
orientation matters because the payload layout is row-major and TQ2's 256-value
blocks expose the mistake immediately. The result reinforces that footprint
and format rankings are workload-dependent; the next quality/performance gate
must use the same artifact dimensions and separate cold upload from hot
dispatch timing.

### Q3 baseline and TQ1 experimental-control checkpoint

The isolated shader harness now has a regular Q3_K packed-matvec baseline and an
experimental TQ1_0 control. Both use ggml-compatible byte layouts, a shared
64-thread reduction, and CPU-oracle validation; they are not production
`ggml-vulkan` graph paths.
On a Pythia `512x576` fixture, Q3_K measured `289.658 us` and TQ1_0
`189.408 us` per 100-dispatch timestamped run, with packed sizes `126,720` and
`62,208` bytes and activation-relative MSE `0.022221782` and `0.60165063`.
The implementation exposed and fixed TQ1's bytewise modulo-256 trit extraction,
which is now documented as part of the contract. Five focused contract/shader
tests pass.

Q3_K therefore participates in the normal rate/quality matrix; TQ1_0 remains
explicitly experimental because of its much larger activation error. These
controls complete the mechanism-side format ladder needed before driver
promotion. The next gate remains a same-artifact quality/rate matrix (including
Q3_K_M and TQ1/TQ2 model controls), followed by target-mobile replay. No
scheduler integration, GGUF reinterpretation, or GPU ASTC encoder work should
start from these isolated shader timings alone.

### 10x6 low-rate bridge checkpoint

`10x6` is now implemented as a standard-ASTC, experimental sidecar footprint.
It covers 60 logical values per 16-byte block (`2.1333... b/w`) while retaining
the existing six-output-row selector strips. The format is included in the
artifact, resource, capability, device, and shader contracts; all standard
runtime paths remain capability-gated and production `ggml-vulkan` is not
modified. Private manifest and artifact format IDs remain backward compatible:
existing `8x8 = 4` is unchanged and `10x6 = 5` is appended.

The first matched Pythia layer-0 weight-grid-gauge screen places validation
stopped 10x6 at `0.069978585` with standard search and `0.068986023` with
neural recall, between aligned 8x6 (`0.052656622`, 2.67 b/w) and 8x8
(`0.082529435`, 2.00 b/w). A second-tensor screen improves 10x6 neutral from
`0.22739616` to `0.21359412`. This makes 10x6 a normal *evaluation* rung, but
not a normal runtime profile: it remains explicit experimental opt-in until
artifact replay and model-facing evaluation establish a practical use case.

Next steps are deliberately narrow:

1. [x] Materialize neutral and validation-prefix 10x6 artifacts with full
   provenance, then verify CPU and Vulkan replay from the payload bytes alone.
   The bounded layer-0 pair is byte-identical after packing, has GPU/CPU MSE
   below `4e-16`, and improves holdout activation MSE from `0.14171192` to
   `0.10563760` at the selected prefix.
2. [x] Add 10x6 and 8x8 to the shared source/rate/quality table next to Q3,
   Q4, TQ2, and TQ1 controls; keep model-facing results separate from
   activation MSE. The first matrix uses footprint-aligned `12x2040` (10x6)
   and `16x2040` (8x8) crops of `blk.0.ffn_down.weight`; TQ1/TQ2 are
   byte-identical on the tested crops and remain a shared source control
   pending a genuinely different tensor/crop.
3. Only if the artifact gate holds, evaluate `10x8` and `10x10` on small,
   aligned crops. Do not expand footprint support on model tensors merely to
   collect a lower nominal rate.
4. Keep neural recall validation-selected per tensor. Do not make it the
   default until a cross-tensor selection rule is demonstrated.

### PV-Tuning and model-preserving rounding track

The present gauge encoder is **not** PV-Tuning. It has an exact discrete
candidate step (legal ASTC encode/decode, conflict-aware commit, and
validation-prefix stopping), but it does not yet alternate that step with a
continuous model-loss optimization. This distinction becomes important for
10x6 and 8x8, and likely essential below roughly 2 b/w.

PV-Tuning is a suitable next offline-only layer because it is representation
agnostic: it alternates a continuous parameter update with a discrete update
over the deployed representation, rather than relying on a straight-through
estimator. In this project the deployed representation remains a set of legal
standard 128-bit ASTC blocks; no custom runtime decoder or shader-side training
state is introduced. Reference: Malinovskii et al., *PV-Tuning: Beyond
Straight-Through Estimation for Extreme LLM Compression*, arXiv:2405.14852.

The first reusable algorithm boundary is now available in
`astc-vulkan-pv.{h,cpp}`. It implements a bounded coordinate P-step followed
by caller-supplied exact V-projection and objective callbacks. It remains an
offline research primitive and is not enabled by any standard scheduler
profile. Its next integration gate is to connect the projector to the exact
ASTC artifact oracle and compare it against the frozen PV-lite pool on the
same calibration/validation/holdout corpus.

Artifact metadata now carries additive `objective` and `optimizer` fields.
Existing streams default to `activation-relative-mse`/`none`; future PV and
YAQA exports must identify their objective and optimizer explicitly. This
keeps replay comparisons attributable without changing version-1 payload
layout or requiring any runtime PV/YAQA state.

The first ASTC adaptation must be deliberately small:

1. Freeze a provenance-bound 10x6/8x8 candidate pool and scalar-anchored
   fallback.
2. **P step:** optimize only continuous codec-safe variables on a calibration
   subset: layer/group affine reconstruction constants and a small coefficient
   vector for the existing zero-sum gauge bases. Do not optimize arbitrary
   per-weight residuals.
3. Re-encode, exact-decode, and deduplicate every updated source block.
4. **V step:** choose only among legal payload candidates using the current
   conflict-aware selector, then select the exported prefix on validation.
5. Accept an iteration only if a disjoint holdout, and then model-facing
   metric, improve. The scalar/gauge-neutral payload must remain a mandatory
   candidate at every iteration.

This is naturally block-/strip-granular rather than scalar-granular: a
10x6/8x8 ASTC block's payload, mode family, and gauge coefficient are the
discrete variables. It reuses the exact artifact oracle already built, and
keeps all optimization offline.

For the 1.6--0.9 b/w ladder, add a later **YAQA-style** objective rather than
blindly increasing PV capacity. YAQA/model-preserving adaptive rounding argues
that immediate layer activation error can be a poor proxy for model output and
uses a two-sided Kronecker-factored sensitivity approximation. Reference:
Tseng, Sun, and De Sa, *Model-Preserving Adaptive Rounding*,
arXiv:2505.22988. The ASTC form is a candidate score

The reusable two-sided score is now implemented in
`astc-vulkan-yaqa.{h,cpp}` as `tr(H_O E H_I E^T)`, with explicit dimension
validation. It is ready to rank a frozen legal candidate pool, but sensitivity
factor construction and model-facing replay remain open gates. The score must
not replace activation-MSE globally until it improves a disjoint held-out
model metric.

\[
  \mathcal L_{\mathrm{two\text{-}sided}}(E)
  = \operatorname{tr}(H_O E H_I E^T),
\]

where `H_I` comes from input traces and `H_O` from downstream/model-output
sensitivity. It should rank the *same* legal ASTC candidate bank, not define a
new ASTC syntax. Implement it only after a fixed-pool ablation establishes that
the richer objective improves validation-selected, model-facing results over
activation-MSE; otherwise candidate-space and objective changes would be
confounded.

Priority is therefore: shared 10x6 source/rate matrix -> PV-lite fixed-pool
ablation on 10x6 and 8x8 -> model-facing gate -> YAQA-style two-sided scoring
for 10x8/10x10 and lower. This protects the central contract: GPU runtime sees
only standard ASTC sampling plus the existing cheap semantic reconstruction.

### Current sweep gates

- [x] Matched 10x6/8x8 activation matrix on Pythia layer 0, with Q3/Q4/TQ
  controls and a separate layer-1 TQ cross-check.
- [x] Verify that the local TQ1/TQ2 crops are independent before treating them
  as separate format evidence. They are not: both tested crops dequantize to
  byte-identical FP32 matrices despite different GGUF type metadata.
- [x] Compare neural candidate recall against standard search on identical
  crops. No holdout gain was found; keep it experimental and do not replace
  the standard candidate path.
- [x] Run the bounded PV-lite coefficient-grid gate with a frozen legal payload
  pool and scalar/gauge-neutral fallback. It uses exact ASTC decode and
  validation-prefix stopping; it is intentionally not described as a complete
  continuous PV-Tuning P-step.
- [x] Repeat the PV-lite gate on a second tensor using F16/Q3 sources (the
  current TQ1/TQ2 crops are byte-identical) and materialize/replay the selected
  artifacts. Full PV-lite did not beat the standard weight-grid control there.
- [x] Add an opt-in full-PV v1 candidate path in latent-smoke. `--pv-alternate`
  performs bounded two-coordinate P-steps (gauge and block correction),
  exact ASTC V-projection, and injects the resulting legal payload into the
  existing conflict-aware selector. It is a candidate-family experiment, not
  a production profile or a claim of the complete PV-Tuning optimizer.
- [ ] Perform the model-facing replay gate with a full-shape PV artifact; crop
  artifacts are intentionally rejected by the model-replay shape contract.

PV-lite-grid is now implemented as an opt-in coefficient-grid experiment. It
must remain separate from the standard/thorough reference path until artifact
replay and a second-tensor holdout gate pass. Its current cost is roughly 4.6x
standard on a small F16/10x6 crop, so future work should seek adaptive
coarse-to-fine coefficients or candidate reuse before considering default use.

Preset policy after the current ablation: keep `thorough` as the reference
configuration, allow `medium` for exploratory sweeps after cross-tensor
confirmation, and use `fast` only for candidate-space screening. Do not mix a
preset change with the first PV-lite quality claim.

Policy decision: PV-style optimization is mandatory for the experimental
`low-rate-neural` encoder profile covering 10x6, 8x8, and lower-rate footprints
(10x8, 10x10, and beyond). It is not mandatory for the standard reference
profile or for runtime Vulkan code. The low-rate profile must retain the
scalar/gauge-neutral payload as an exact fallback and must fail the artifact
gate rather than silently exporting a standard-only stream. 8x6 remains
standard-gauge by default with PV-lite available for research because its
bounded PV result was inconclusive.

The bounded layer-1 confirmation is complete: medium was faster than thorough
(`0.36 s` vs `0.45 s`) and selected a better holdout prefix in that run, but
the result remains exploratory because the preset also changes the neutral
candidate. Thorough remains mandatory for reference artifacts.

The 8x6 check did not show a repeatable PV-lite gain (F16 was slightly worse;
Q3 was effectively unchanged), so 8x6 remains standard-gauge reference plus
PV-lite experimental opt-in. The first PV-lite 10x6 artifact replay is now
byte-identical on CPU; Vulkan replay and a second-tensor PV-lite gate remain
open.

The PV-lite 10x6 payload has now also passed isolated Intel Vulkan sampling
from the exported artifact bytes. The remaining PV-lite gate is a second
tensor with genuinely different source values plus a model-facing metric;
driver integration and broader footprint promotion remain blocked on that
evidence.

### PV-lite implementation review and next sweep

The implementation is now correctly scoped as a **PV-inspired coefficient-grid
search**, not as full PV-Tuning. Its P-side is a fixed, zero-sum source family:
the mandatory neutral member plus X-ramp, Y-ramp, and saddle gauges at
`+/-0.25`, `+/-0.50`, and `+/-0.75`. Its V-side is stronger and exact: every
source member is encoded as a legal ASTC block, decoded by the reference
decoder, deduplicated by payload, coordinated by the conflict-aware selector,
and exported only at the validation-selected prefix. This is the right
conservative profile for the first low-rate experiments.

Before another quality claim, the following implementation gates apply:

1. [x] Make validation artifacts self-describing: record `encoder_profile`,
   candidate-family/grid version, and `validation_prefix`; the runtime artifact
   must be the validation payload, never the all-commit diagnostic stream.
2. [x] Add an unaligned 10x6 edge regression. Clamp padding remains
   deterministic but is excluded from gauge-headroom and semantic loss, so it
   cannot shrink the valid tensor's gauge range.
3. [x] Preserve the selected PV factor through streamed row-strip selection
   and record factor index, basis, gauge, and correction in the commit log.
   This is the evidence needed for a later adaptive/coarse-to-fine grid.
4. [x] Profile PV-lite with persistent worker contexts. The persistent path
   reproduces payloads, decoded references, and commit CSV byte-for-byte; on a
   12x240 10x6 crop it reduced wall time from `6.34 s` to `0.59 s`.
   Generation/decode sub-phase profiling remains a later performance task.
5. [x] Add a fixed coarse-grid control (`0, +/-0.50` for constant/X/Y/saddle)
   and record factor usage. On the second tensor it matched the standard
   weight-grid candidate family and was more robust than the full PV-lite grid.
   A genuinely adaptive grid is still gated on independent calibration data;
   no factor is pruned using the final holdout.
6. [ ] Only after those gates, implement a true alternating PV step over a
   small set of group affine/gauge coefficients. Each update must reproject
   through exact ASTC encode/decode and retain the scalar-neutral fallback.
7. [ ] Reserve YAQA-style two-sided/model-preserving scoring for `10x8`,
   `10x10`, and lower rates after a fixed-pool ablation; it must change the
   objective without simultaneously changing the candidate family.

The cross-tensor result is an important negative gate: on the second
`blk.1.ffn_down.weight` tensor, full PV-lite did not beat the standard
weight-grid gauge on validation-stopped holdout (F16 `0.13591` vs `0.11339`,
Q3 `0.12655` vs `0.10871`). The fixed coarse grid matched the standard control.
PV-lite therefore remains a useful diagnostic/candidate-space experiment, but
is not promoted to the low-rate default until a model-facing and cross-tensor
holdout gate succeeds.

The combined `PV-lite grid + encoder-search neural` control was also checked
on matched F16 layer-0 low-rate crops. At 10x6, validation-stopped holdout was
`0.07738508` for PV-lite alone and `0.07738508` for the combination; at 8x8 it
was `0.11633482` in both cases. Candidate counts increased (741 to 761 at
10x6, 982 to 1014 at 8x8) without a quality gain. Therefore the broader neural
candidate-recall path remains an optional diagnostic and is not automatically
stacked on the low-rate PV profile.

The existing scalar-anchored `c+delta` family was also checked as a bounded
PV-adjacent control. It did not close the gap: validation-stopped holdout was
`0.11721536` at 10x6 and `0.14530883` at 8x8 on the F16 layer-0 crop, both
worse than the corresponding PV-lite results. This is evidence against adding
more unconstrained correction dimensions before a model-facing objective is in
place.

### Driver expansion checklist (side-fork only)

The driver remains an isolated PoC under `pocs/astc-vulkan`; production
`ggml-vulkan` is not modified. The runtime contract is deliberately narrow:
offline tools emit ordinary legal ASTC blocks, while Vulkan performs only
capability-gated image upload, fixed-function texture decode, sampling, and
the existing cheap semantic reconstruction. PV/YAQA state and the neural
encoder are never required on the device.

The remaining driver gates are:

1. [x] Keep footprint arithmetic, serialized IDs, artifact parsing, resource
   mapping, capability probing, device upload smoke, and shader sampling in one
   side-fork format matrix. `10x8` is appended as ID 6 and remains experimental;
   existing IDs are unchanged.
2. [x] Add contract coverage for `10x8`: 80 texels per 128-bit block, nominal
   1.60 bits/texel, deterministic block rounding, artifact acceptance, and
   atlas placement.
3. [x] Capability-gate every experimental footprint. A device without the
   sampled/transfer feature returns the existing skip/fallback result before a
   Vulkan image is created; no format is silently substituted.
4. [x] Exercise the same `10x8` payload path through CPU latent smoke and the
   isolated Vulkan shader/device smokes. The bounded smoke is not a portable
   performance claim.
5. [x] Materialize at least one full-shape validation-prefix artifact for each
   promoted encoder profile and replay it from artifact bytes alone (CPU
   oracle, then Vulkan sampled replay). Crop artifacts remain diagnostic only.
6. [x] Run the model-facing replay gate on a known-good Vulkan device or a
   genuinely CPU-only backend. The current host's unrelated Vulkan model path
   can lose the device; that is a test-environment limitation, not permission
   to weaken the ASTC contract.
7. [ ] Add the final format comparison matrix (Q4_K_M, Q3_K_M, TQ2_0, TQ1_0,
   FP16 oracle, ASTC 6x6/8x6/10x6/8x8/10x8) with storage bytes, model-facing
   quality, cold upload, hot-cache dispatch, and batched timing. Keep target
   device and driver identity beside every timing.
8. [ ] Define the production-facing adapter only after the full-shape gate:
   manifest/version/hash validation, capability selection, descriptor/image
   lifetime, dispatch synchronization, and deterministic fallback to the
   existing quantized path. This adapter remains opt-in until mobile coverage
   is measured.
9. [ ] Validate on a target mobile/embedded GPU (Mali, Adreno, or Apple) before
   making any throughput or energy claim. Desktop Intel Vulkan is a reference
   implementation check, not a proxy for mobile texture-unit scheduling.

The current 10x8 screening result is consistent with keeping it experimental:
it supplies 1.60 bits/texel versus 2.00 for 8x8 and was viable in the bounded
latent path, but there is not yet a full-shape model result or cross-device
performance evidence. Mixed footprints are intentionally deferred: one
Vulkan image has one fixed ASTC `VkFormat`, so a mixed 8x8/10x8 design needs
separate images or atlases plus macro-tile metadata and shader address/dispatch
logic. An offline rate/quality composition tool is safe to add later, but it
is not part of the first driver adapter.

The exact same-prompt native control matrix is now recorded for the current
Pythia reference and CPU-only backend. It establishes Q4_K_M and Q3_K_M as
the model-facing quality references, with TQ1_0/TQ2_0 retained as aggressive
few-level controls; the existing full-shape 6x6 scalar/gauge replay is the
ASTC driver reference. The final matrix gate remains open until equivalent
full-shape ASTC artifacts exist for the low-rate footprints and timings are
measured on the target device.

The sidecar adapter now carries an explicit fallback policy in every binding:
Q4_K_M is the preferred quality fallback and Q3_K_M is the low-memory fallback.
The sidecar does not execute either path itself; the normal llama scheduler
remains the owner of fallback dispatch. This makes the future production
adapter contract deterministic without coupling the PoC to scheduler internals.

### Current execution order (production gate and low-rate evaluation)

The remaining work is intentionally ordered so that production integration is
not influenced by experiments that have not survived a full model replay:

1. **Production sidecar/adapter:** complete the opt-in scheduler boundary for
   standard 4x4/5x5/6x6 scalar and scalar-anchored gauge; 6x6 remains the
   default profile. Any malformed artifact, shape or
   representation mismatch, missing sampled-ASTC capability, or unavailable
   Vulkan device must leave the binding in `kFallback` and expose the normal
   `Q4_K_M` (quality) and `Q3_K_M` (low-memory) choices. No experimental
   footprint is automatically selected by this boundary.
   Each standard footprint has its own offline selector and artifact profile:
   4x4, 5x5, and 6x6 are not one shared tuning stream. This keeps candidate
   pools, validation prefixes, and tie-breaking decisions traceable to the
   exact block geometry while allowing a complete per-tensor/page profile.
   The runtime adapter still exposes one fixed `VkFormat` per bound image.
2. **Target-device matrix:** use the exact artifact bytes and fixed model /
   prompt contract to measure FP16, Q4_K_M, Q3_K_M, TQ2_0, TQ1_0, ASTC 6x6
   scalar/gauge, and each low-rate ASTC profile. Record model-facing quality,
   storage bytes, cold upload, hot-cache dispatch, and batched timings with
   device/driver identity. Intel Vulkan is a reproducibility reference, not a
   mobile-performance proxy.
3. **Full-shape low-rate artifacts:** materialize provenance-bound neutral and
   validation-prefix streams for 8x6, 10x6, 8x8, and 10x8. Replay the exact
   bytes through the CPU oracle and Vulkan sampler, then decide from the full
   model replay whether each low-rate point is retained. Crop results remain
   screening evidence only. The first three artifacts are v1-corpus records;
   10x8 has also been completed on a regenerated, explicitly identified v2
   CPU-capture corpus after the original temporary traces expired. A final
   matrix must keep those corpus identities separate or regenerate all four
   under one retained corpus before ranking formats.
4. **Low-rate research (after the model gate):** apply full-shaped PV
   alternation and YAQA-style two-sided/model-preserving scoring to 10x8,
   10x10, and lower-rate candidates. Keep the candidate pool, objective, and
   source family independently versioned so an improvement is attributable.
5. **Mixed footprint (last):** investigate composition only if the uniform
   matrix shows a stable hard-tile population. The first mixed form is
   per-tensor or per-page selection, not per ASTC block: a page is assigned one
   selector/profile and retains simple address calculation. A later macro-tile
   study may use footprint-height classes (H6: 6x6/8x6/10x6; H8: 8x8/10x8)
   with separate images/atlases and explicit tile metadata. Per-block mixing
   is out of scope until metadata and shader branching show a net benefit.
   Mixed rates must not leak into the first production adapter.

The required artifact set for step 3 is four full-shape pairs (`neutral` and
`validation-prefix`) plus manifest/provenance and decoded CPU references. The
low-rate matrix must report nominal storage density from the 128-bit block:
8x6 = 2.67, 10x6 = 2.13, 8x8 = 2.00, and 10x8 = 1.60 bits/texel. These are
block rates, not independently exact bits/weight; edge padding is deterministic
and excluded from semantic loss. Batching and hot-cache measurements happen
after correctness/replay so timing cannot mask an artifact or decoder mismatch.
### Latest algorithm-integration status

The offline YAQA module now includes a low-rank trace-backed score for a
candidate error matrix, equivalent to `tr(H_out E H_in E^T)` when
`H_in = X^T X` and `H_out = Y^T Y`, evaluated as `||Y E X^T||_F^2`. A replay
smoke consumes explicit artifact metadata and matching input/output traces;
it does not regenerate ASTC payloads. This is an output-covariance proxy for
the next gate, not a substitute for a model-loss Hessian.

The new trace pair and replay numbers are recorded in `astc-vulkan-wip.md`.
They are intentionally not promoted to the quality matrix because the
exploratory prompt differs from the retained v2 model-facing corpus and the
8x8/10x8 model replays did not pass the production-quality gate. PV and YAQA
remain offline-only and cannot add runtime state or alter the standard Vulkan
decoder contract.

Before driver promotion, the remaining algorithm gates are:

1. Use the bounded/lightweight PV candidate family as the default offline PV
   experiment. Full-shaped PV alternation remains optional research only: it
   is not required for the driver gate and may be run later on dedicated
   hardware/time budgets.
2. Compare PV artifacts using the exact CPU oracle and model replay before
   changing any scheduler/profile eligibility.
3. Use the trace-backed YAQA score on `10x8`/`10x10` as a sensitivity
   comparison, then replace the proxy with a genuine downstream/model-loss
   factor when the capture path is available.
4. Keep the final production matrix separate from exploratory prompt/corpus
   results and retain all historical tests as named regression contracts.

The full-PV run is explicitly not a required dependency. It was stopped after
proving that the full-shape search is too expensive for the current CPU budget.
The bounded PV path remains the reproducible, low-cost candidate generator;
future full-PV work must be opt-in and artifact-versioned.

### Experimental D2 paired-density track

The low-rate plan now has a separate **D2 paired-density** research profile.
D1 stores one logical weight per ASTC texel. D2 stores two weights from
adjacent output rows at the same reduction/input column in one texel, so the
nominal logical density is:

`128 / (footprint_width * footprint_height * 2)` bits per weight.

The first iso-rate gate is D1 `10x8` versus D2 `8x5`, both 1.60 nominal
bits/weight. D2's initial `RG/B` mapping stores `q0` redundantly in R/G and
`q1` in B; `R/GB` is the mandatory orientation control. Alpha is encoder-only
codec steering and has no semantic runtime role. This deliberately differs
from D1 L+A gauge, where both latent values enter reconstruction.

Implementation gates, in order:

1. The independent `astc-vulkan-paired` contract module, 8x5 format support,
   deterministic adjacent-row pairing, and deterministic odd-row padding.
2. CPU exact ASTC encode/decode artifacts with fixed-Alpha negative controls,
   then constant/ramp/saddle steering; the scalar/D1 artifact remains the
   fallback.
3. Aligned Pythia crops on two tensors: D1 10x8, D2 8x5 RG/B fixed Alpha,
   D2 RG/B steering, and D2 R/GB steering, all with frozen
   calibration/validation/untouched-holdout splits.
4. Only if D2 passes that iso-rate gate: a D2 footprint ladder and paired
   Vulkan artifact replay. Full PV, few-level source families, and RGB
   common-mode/subtractive layouts remain later experiments.

The next D2 implementation boundary is an offline selection core that consumes
only exact-decoded legal payloads and their activation-space deltas relative to
the neutral paired-D2 baseline. It greedily commits compatible positive calibration gains,
then materializes the best independent validation prefix. It deliberately owns
neither ASTC candidate generation nor Vulkan upload: this keeps the standard
codec oracle, candidate-family experiments, artifact writer, weighted scoring,
and YAQA reranking independently testable. Candidate zero is mandatory for
every block and represents the byte-level neutral D2 baseline. A D1/Q fallback
is a tensor-level scheduler decision, because it cannot share D2's physical
block geometry.

The first bounded Pythia crop replay does not yet pass the iso-rate gate:
D1 `10x8` is lower activation-loss than the unselected D2 `8x5` cases on
calibration, validation, and holdout. D2 therefore remains experimental while
the decode-in-the-loop candidate pool and conflict-aware selection are added.

The paired shader must consume one texel once and accumulate both mapped
output rows; otherwise the intended fetch/bandwidth benefit is lost. D2's
logical row-strip height is twice its ASTC texture-block height, so row-strip
selection, padding masks, CPU oracle, and shader indexing need their own
explicit geometry contract. D2 is never auto-scheduled into the production
adapter until its artifact and model gates pass.

For D2 and other profiles at or below roughly 1.6 nominal b/w, selection uses
an objective hierarchy rather than raw weight-MSE: retain weight-error/tail
limits as safety diagnostics, shortlist legal payloads by activation-relative
error, optionally rerank that shortlist with diagonal output-weighted
activation, then use trace-backed YAQA as the stronger two-sided control.
Validation chooses the commit prefix; holdout remains read-only. The bounded
D2 steering codebook replaces full PV iteration as the default candidate
family: neutral, signed X/Y/saddle, and signed diagonal steering bases.

The bounded paired model smoke now exposes this hierarchy directly: its default
is `--objective activation`, while `--objective yaqa --output-trace path`
reranks the same exact-decoded D1/D2 payload candidates with a matching
FFN-down output trace. This is an offline whole-crop gate only. The subsequent
blockwise producer should use YAQA as a bounded rerank of the already legal
activation shortlist, then pass the selected payload alternatives to the
conflict-aware selector and validation-prefix artifact stage.

### D2_8x5 exact candidate/selection gate

Completed the first bounded producer for `D2_8x5`: each physical block emits
the deterministic Alpha codebook under both semantic layouts, exact ASTC
payload/layout pairs are deduplicated, each candidate is decoded through
astcenc, and the resulting logical-weight deltas are scored in activation
space. The conflict-aware selector then receives only those exact deltas,
commits calibration-positive candidates serially, and materializes the best
independent validation prefix.

The first real Pythia run is intentionally small (`20x128`, nine traces) and
is a plumbing gate, not an R-D claim. It showed the expected calibration and
validation reduction but no holdout improvement. A second 20x256 split used
64 physical blocks and 1,408 unique legal payload/layout candidates: its
validation prefix reduced D2-neutral holdout activation MSE from `8.114e-4`
to `4.565e-4`. The next quality gate is a larger, independently split crop and
bounded YAQA reranking of the same legal candidate pool. Only a passing
iso-rate D1/Q comparison should proceed to full artifact materialization,
CPU/Vulkan replay, and then `D2_10x5`.

The D2 smoke now has an optional neural ASTC backend linked against the
isolated neural-rank fork. The fork's D2-specific semantic metric must be used
for this comparison; the original neural L+A flag is invalid for paired D2
because D2's RGB lanes are not one replicated latent. Standard astcenc remains
the immutable reference backend. The backend comparison is valid only when
source, codebook, exact decode, selector, validation prefix, and holdout split
are identical.

#### D2 offline performance contract

D2 candidate generation is an offline pack-time operation. Each producer
worker owns persistent `RG/B` and `R/GB` astcenc contexts plus worker-local
source, decode, payload-deduplication, and activation-delta scratch. Context
creation, image-object construction, and heap allocation must not occur in the
per-candidate inner loop. Where supported by the isolated neural fork, sibling
contexts may share immutable ASTC lookup tables; failure to share is a
performance fallback, not a change to payload semantics.

Every optimization at this boundary must reproduce the reference candidate
artifact: same legal payload bytes, candidate order/tie-break behavior,
selector commit sequence, validation prefix, and replayed CPU loss. The
producer records a timing breakdown for source construction, ASTC encode,
exact decode, activation-delta construction, selector, and objective replay.
This keeps later changes—source caching, encoder-preset ablation, adaptive
steering grids, internal block-analysis reuse, or batched GPU ranking—from
being mistaken for representation improvements. GPU work can score/offline
rank an already-generated candidate pool later, but it neither replaces the
standard ASTC CPU encoder nor adds a runtime decoder.

The first optional structure-bank implementation is now available in the
neural-rank fork. It observes legal block structures from a neutral pass and
filters later numerical fits to that bank. This is an experiment-only
optimization: the initial D2 result was about `2.95x` faster on a 20x256 crop
but regressed untouched holdout, so the full neural search remains the quality
reference. Any future bank widening must prove candidate recall and holdout
equivalence or improvement before it can be selected by the producer.

### D2_10x5 implementation gate

`D2_10x5` is now implemented through the bounded PoC chain. It has an explicit
compile-time profile for the 10-column/5-row physical block, dedicated
standard CPU model and selector smokes, and a neural side-fork selector
variant. The nominal density is `128 / (10 * 5 * 2) = 1.28` bits/weight; the
five-row block still reconstructs ten local logical rows, so the shader and
selector retain the same row-strip factorization as D2_8x5.

The device smoke checks the standard `VK_FORMAT_ASTC_10x5_UNORM_BLOCK` path
when the selected device advertises it, including both `RG/B` and `R/GB`, the
16-lane reduction, and proposal-gain arithmetic. Devices that expose only
8x5 keep the 8x5 gate and report 10x5 as unsupported. This is a capability
condition, not a substitute format.

The profile remains experimental: no full-size D2_10x5 artifact, metadata-
aware production upload, or model-level quality claim has been promoted. The
next gate is the production execution order below, starting with the standard
6x6 scalar/gauge adapter and its Q4_K_M/Q3_K_M fallback before any D2 profile
can enter the scheduler.

### Offline ranking backend selection

The packer has an explicit stage plan rather than a hard-coded CPU/GPU choice.
`candidate_encode` is always CPU astcenc; `delta_score`, `objective`,
`proposal_gain`, and `conflict_commit` may be CPU or GPU. The initial default
is the all-CPU plan. A GPU plan keeps decoded deltas and residuals device-side;
it is invalid to request GPU proposal/commit while producing deltas on CPU,
because that would disguise per-iteration host transfers as acceleration.

GPU ranking uses a temporary sampled ASTC candidate atlas: every legal CPU
payload occupies one physical raster block, while compact buffers describe its
source block, neutral candidate, and paired-D2 layout. The final artifact still
contains only the selected original payload and existing D2 layout metadata.
Before the GPU plan can become selectable, it must prove exact FP32 GPU delta
equivalence to the CPU oracle on bounded payloads, then reproduce the CPU
selector's commit sequence and validation prefix. Device policy and a default
choice come only after those gates and measured transfer/batch costs.

Implementation status: the candidate-atlas transport, a Vulkan delta session,
and a hardware-smoke executable now exist. An initial Intel UHD 620/Mesa
failure was traced by the validation layer to host-side handle-order bugs:
descriptor sets were allocated before their descriptor pool, the compute
pipeline was created before its pipeline layout, and the instance advertised
Vulkan 1.0 while using SPIR-V 1.3. These lifecycle/API-order issues are now
fixed. The Intel smoke executes both paired D2 layouts and passes five
repeated init/dispatch/readback/reset runs with validation enabled. GPU
ranking remains an opt-in experiment guarded by a per-device preflight; CPU
is still the production default. The next GPU gate is to reproduce CPU commit
order and validation prefix before moving YAQA or conflict-aware proposals to
the device.

YAQA is the exception to the ASTC hardware constraint: it is pure buffer math
and can be preflighted on a discrete NVIDIA device. A CPU-equivalent
one-invocation trace-score smoke passes there. The scalable implementation is
two GPU stages—parallel trace-pair partials per candidate, then a per-candidate
reduction—with no per-candidate host readback. Integrate that session and prove
batch scores against the CPU trace oracle before allowing GPU YAQA in the
ranking plan. This does not make NVIDIA an ASTC-decode target.

The batch GPU YAQA smoke now dispatches the two-stage path for seven candidate
error matrices and eight trace samples and matches the CPU oracle for every
candidate. The next implementation gate is a reusable session with persistent
device buffers and configurable candidate/sample counts; then proposal gains
can consume the score vector without a host round-trip. ASTC candidate encode
and ASTC decode remain separate CPU/reference stages on this NVIDIA path.

GPU resource contract: candidate-ranking command buffers must be allocated
after their command pool, descriptor sets after their descriptor pool, compute
pipelines after their pipeline layout, host-visible writes must be flushed and
explicitly made visible to compute, GPU outputs must be made visible and
invalidated before host reads, and reset must wait before destroying dependent
descriptors, buffers, and sampled images. The current implementation satisfies
this basic contract, and the Intel validation/repetition gate is green. Add
selector-equivalence and validation-prefix replay before making the ASTC GPU
path selectable.

The next batch form is now explicit. CPU astcenc workers keep generating legal
payloads, while a persistent GPU session consumes a rectangular candidate atlas
in batches. A 16-lane D2 delta workgroup reduces the eight or ten reduction
columns for one `(candidate, calibration sample, logical row)`. A second GPU
pass consumes the device-resident delta and residual and emits one proposal
gain, `2<R,d>-||d||^2`, per candidate. This reduces normal ranking readback
from a candidate-by-sample-by-row delta field to one float per candidate. CPU
continues to own exact conflict-aware commit order initially. A ring of atlas
upload slots and overlapped CPU-producer/GPU-consumer scheduling are later
throughput work; neither may alter payloads, tie-breaking, or validation stop.

The worker boundary is source-block safe. A batch planner groups complete
candidate pools up to a configured resident-candidate limit; it never splits a
pool, so candidate zero remains the local baseline and all baseline indices
remain valid. A range atlas builder preserves global source-block IDs while
using local record indices for each upload. This is the first step toward a
bounded producer/consumer queue; persistent Vulkan atlas slots and overlapped
submissions remain the next throughput implementation.

The first persistent-slot primitive is now available: an initialized ranking
session can update a same-footprint, same-dimension atlas and rebuild its
records without recreating the sampled image/view/sampler, descriptor set,
pipelines, command pool, or GPU buffer capacity. The current update uses a
short-lived staging copy command for safety; a later ring implementation may
reuse staging buffers and fences to overlap slots. This keeps the first slot
contract deterministic while avoiding per-batch pipeline setup.

### D2 per-block layout metadata (v3 groundwork)

YAQA may select either `RG/B` or `R/GB` for each physical D2 ASTC block. The
semantic choice is not encoded by standard ASTC itself, so manifest v3 adds a
separate packed layout blob for `paired-d2` artifacts. It contains little-endian
`uint32` words in physical ASTC raster-block order: bit zero is block zero,
zero means `RG/B`, and one means `R/GB`. The first concrete profile is
`D2_8x5` at 1.60 b/w. The same CPU selector/model smoke and GPU transport
contract now cover `D2_10x5` at 1.28 b/w; both have a five-row physical
texture stripe and therefore a ten-row logical D2 stripe. They remain
distinct Vulkan image formats, not an interchangeable generic “H10” format.
For D2 8x5 the layout bit costs one bit per 80 logical weights (`0.0125 b/w`,
with only final-word rounding overhead); for D2 10x5 it costs one bit per 100
logical weights (`0.01 b/w`). Production still keeps paired-D2 behind the
metadata-aware adapter gate until full artifact/model replay is complete.

The paired shader accepts a storage-buffer layout map and retains a uniform
layout fallback when no words are supplied. Existing v1/v2 scalar/gauge
artifacts remain readable; the production scheduler continues to reject
`paired-d2` until the metadata-aware resource upload, exact artifact replay,
and model gates are complete. D1 deliberately receives no new per-block
metadata in this step. Mixed D1 footprints remain a later macro-page/atlas
experiment, not an incidental consequence of the D2 bitplane.

### D1 GPU scoring foundation for scalar and PV families

D1 GPU scoring is a separate transport from paired D2. It accepts any defined
D1 footprint (`4x4`, `5x5`, `6x6`, `8x6`, `10x6`, `8x8`, `10x8`) and records
only a semantic decoder profile (`scalar` or scalar-anchored `gauge_la`), not
per-block side metadata. CPU astcenc remains the only legal payload producer;
the D1 atlas is an offline sampled-ASTC upload used solely to score many exact
candidate payloads in parallel.

PV now has a batch-objective seam. The bounded coordinate optimizer retains
CPU projection, accepted-step order, tie-breaking, validation prefix, and
artifact ownership. Its two `+/-` projected candidates for one coordinate can
be scored together by a CPU or Vulkan/YAQA backend. This keeps both the
fixed-grid PV-lite and optional full alternating PV behaviorally reproducible:
GPU changes numerical throughput only, never ASTC encoding or the deployed
bytes. The next implementation gate is a D1 Vulkan delta/proposal session
whose `4x4` and `6x6` output matches the CPU oracle before adding `10x6` and
`8x8` low-rate batches.

### D1 two-stage rate screening

The original F16/BF16 tensor is first screened before any ASTC encoding. For
each candidate footprint and source-level family (currently 16-level/Q4-like
or 8-level/Q3-like), the screen uses a local affine scalar quantizer per
physical footprint block and scores the resulting error with diagonal input
activation energy. This screen is separable over blocks, so it can be run as a
broad GPU compute batch even on a device that cannot sample ASTC. It selects a
bounded, rate-aware shortlist using `weighted_error + lambda * b/w`.

The screen must never export a payload or claim ASTC quality. In particular,
L+A gauge and semantic reconstruction remain absent from this first pass: their
usefulness depends on ASTC's real discrete encode/decode choices. The exact
second pass runs CPU astcenc only for the shortlist, then evaluates exact
scalar/gauge candidates with activation or YAQA scoring, conflict-aware
selection, validation stopping, and holdout. This prevents a broad CPU ASTC
search while preserving the existing artifact oracle.

The D2 PV preparation also has a dedicated contract test for both paired
semantic mappings (`RG/B` and `R/GB`). It exercises the batched `+/-` PV
objective seam and verifies deterministic equivalence of the layout choices.
This is a CPU contract only; a D2 Vulkan PV session remains gated on a real
paired-D2 candidate atlas and exact GPU/CPU score comparison.

The first Vulkan kernel (`d1-prescreen.comp`) mirrors this CPU oracle and
builds successfully for Vulkan 1.1. It evaluates one footprint/source-level
candidate per invocation and writes one score per candidate. A device session
and buffer-lifecycle smoke now bind the same buffers and compare every GPU
score against the CPU oracle. The smoke passed on the available Vulkan
compute device. This remains a pre-screen gate: its result may reduce the CPU
astcenc search, but cannot replace exact ASTC artifact replay.

`astc-vulkan-latent-smoke` now accepts `--d1-prescreen-gpu <shader.spv>`.
It uses the one-shot Vulkan backend to score the same tensor/trace inputs,
compares every GPU value against the CPU oracle, and only then uses the GPU
score vector for the normal rate-aware shortlist. Any initialization or score
contract failure prints its reason and falls back to CPU. A small `4x8` latent
run on the available compute device passed this path and encoded only the four
screened footprints in the subsequent exact ASTC pass.

### D2 PV comparison gate

The D2 implementation now exposes `--pv-alternate 1` in the paired selection
smoke. This is a bounded, two-coordinate PV-style candidate generator (x/y
steering) layered on the deterministic D2 codebook. Each trial is projected by
the exact ASTC encoder and CPU reference decoder; the existing selector and
validation-prefix rules remain authoritative. The flag is experimental and
must not become the D2 default until it wins on disjoint holdout data across
both the standard and isolated neural astcenc backends.

The current gate sequence is:

1. Keep the D2 codebook as the production-safe candidate family and scalar
   fallback.
2. Compare codebook versus bounded PV on matched 8x5 and 10x5 crops, then on
   at least two tensors and a larger artifact.
3. Record candidate count, unique payloads, roundtrips, encode/decode time,
   calibration, validation and untouched holdout loss.
4. Only after a stable quality gain, evaluate GPU batched scoring for the exact
   projected payloads. GPU scoring must not be confused with GPU ASTC encoding;
   candidate generation remains CPU-side in this phase.

The first crop showed a small standard-backend 8x5 holdout improvement, while
the neural backend and a smaller crop did not. Therefore PV remains an
opt-in research profile and is not yet suitable for scheduler promotion.

### Model-adjacent cache contract

Approved artifacts are stored as a sidecar directory next to the base GGUF:

```text
model.gguf.astc-vulkan/
  manifest.astcv
  payload.astcpack
  layout-map.bin              # only if any record is paired-D2
  provenance.txt              # optional research provenance
  source.gguf.sha256
  manifest.sha256
  payload.sha256
  layout-map.sha256           # only for paired-D2
```

The cache is an overlay, not a replacement model: the base GGUF still owns
architecture, tokenizer and all non-ASTC tensors. Loader integration accepts
`auto` (the adjacent directory), an explicit cache directory, or the manifest
path. It must verify the source GGUF and every cache file before binding any
tensor, then fail closed to normal Q4/Q3 routing on a cache miss or mismatch.

JIT cache creation is deliberately split by representation. Scalar artifacts
may eventually be created under an explicit user opt-in. Gauge/YAQA artifacts
must be prebuilt with versioned calibration/validation data; the first user
prompt is not valid calibration input. The cache creator uses a staging
directory plus atomic rename and refuses to overwrite a pre-existing cache.

Paired-D2 layout metadata is supported by the cache now, but D2 remains an
experimental transport profile and is rejected by the automatic D1 scheduler
until it completes artifact-backed and full-model quality gates.

### Memory-resident cache admission

Cache I/O and device residency are separate concerns. The cache verifier and
loader must stream manifest checksums and read only the tensor payload range
being bound; an aggregate `payload.astcpack` must never become a host-RAM
requirement. Device admission uses a conservative default of 80% of currently
available host RAM and 80% of the selected device-local Vulkan heap (or the
current `VK_EXT_memory_budget` value where exposed). Integrated GPUs use the
smaller of the host and device limits.

The admission calculation includes actual Vulkan image allocation requirements
and the temporary upload staging allocation, rather than estimating from ASTC
rate alone. A rejected reservation is a normal fallback to Q4/Q3. The next
scheduler implementation step is a cumulative resident-set planner: preload
every approved ASTC tensor when their summed allocations fit, otherwise retain
a bounded layer/tensor working set and stream the rest. This applies the same
budget API with nonzero resident bytes and avoids either a whole-model RAM copy
or an uncontrolled GPU-memory overcommit.

The cumulative residency planner is now implemented as an isolated scheduler
primitive. It preserves caller-provided layer/tensor order, preloads all
approved records when their measured allocations fit, and otherwise returns a
bounded prefix for streaming. It does not reorder or evict resources; those
policies remain scheduler responsibilities. The next production step is to
feed it per-record Vulkan requirements and retain/release the resulting
resident set around layer execution.

### Full-shape evidence gate (current)

The next production-quality evidence set is fixed to the same Pythia tensor,
the same captured activation corpus, and disjoint calibration/validation/
holdout samples for all three selected shapes:

* D1 `10x8` (1.60 b/w), scalar-anchored gauge;
* D2 `8x5` (1.60 logical b/w), paired `RG/B` and `R/GB` layouts;
* D2 `10x5` (1.28 logical b/w), paired `RG/B` and `R/GB` layouts.

Each artifact must be replayed from exported bytes (never re-encoded), and
must report physical payload size, layout-map size, SHA-256,
calibration/validation-prefix, untouched activation holdout, and full
model/logit replay. Vulkan sampled-ASTC replay is a separate correctness and
timing gate; CPU exact decode remains the quality oracle.

The D2 full-shape selector uses bounded row-strip streaming. A D2 physical
block covers ten logical output rows, so candidate generation and conflict
selection are performed per ten-row strip and released before the next strip.
This prevents O(full-tensor × candidates × trace) resident allocations. The
implementation reports `selection_scope=row-strip-independent`; the
small-crop global selector remains the regression reference. D2 stays
experimental until all three full-shape artifacts and model replay pass.

Required sequence:

1. Finish/export D1 `10x8` full-shape artifact and replay it.
2. Generate/export D2 `8x5` and `10x5` full-shape artifacts with layout maps.
3. Run the matched model/logit replay matrix and compare against D1 `6x6`,
   D1 `10x8`, Q3/Q4 and TQ controls.
4. Run artifact-backed Vulkan replay/timing on the available target GPU;
   record that this validates the hardware/runtime path, not cross-device
   quality.

### Evidence gate status (2026-09-04)

Items 1–3 are complete for the v3 Pythia tensor. D1 `10x8`, D2 `8x5`, and D2
`10x5` artifacts were exported, SHA-256 verified, and replayed from bytes in
the CPU model harness. D1 `10x8` is the strongest 1.60 b/logical-weight point
in this matched run; D2 `8x5` is a weaker same-rate experimental alternative,
and D2 `10x5` is a lower-rate but materially lower-quality fallback. The
remaining item is artifact-backed Vulkan replay/timing, followed by a final
cross-check against the Q3/Q4/TQ controls before any scheduler promotion.
