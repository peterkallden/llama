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
