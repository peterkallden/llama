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

## Phase 3: compute matvec kernel

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
