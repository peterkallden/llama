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

1. Add host-neutral format contracts and focused CTest targets.
2. Add a small standalone Vulkan test utility, outside the normal inference
   path.
3. Query `VK_FORMAT_ASTC_4x4_UNORM_BLOCK` and
   `VK_FORMAT_ASTC_6x6_UNORM_BLOCK` for optimal-tiling sampled-image support.
4. Upload known pre-encoded ASTC images and verify `texelFetch` results in a
   compute shader against a CPU reference.
5. Measure sequential and deliberately non-local fetch patterns with Vulkan
   timestamp queries.
6. Record device name, driver version, supported formats, shader workgroup
   shape, elapsed GPU time, and bytes represented.

Exit criterion: the selected target GPU accepts both formats as sampled images
and the benchmark produces deterministic, validated values.

## Phase 2: offline ASTC-aware weight packer

1. Define an experimental input tensor format and metadata record; do not add
   it to the public GGUF specification.
2. Implement 4x4 and 6x6 packing paths.
3. Encode static weight blocks offline and preserve any scale, offset, outlier,
   or layout metadata needed by the shader in a companion buffer.
4. Evaluate elementwise error, dot-product error, and error distribution per
   block versus FP16/FP32 and an existing low-bit reference.
5. Iterate on channel assignment and tensor permutation only when it improves
   the measured objective.

The packer must optimize a numerical objective. A generic image compressor is
useful as an initial baseline but is not assumed to be optimal for neural
weights.

Exit criterion: both formats can represent a small known matrix and report
reconstruction and dot-product error reproducibly.

## Phase 3: compute matvec kernel

1. Add dedicated experimental compute shaders following the existing Vulkan
   shader source/build conventions.
2. Bind ASTC images through sampled-image descriptors and bind companion
   metadata/activations/output through the ordinary buffer descriptor path.
3. Use `texelFetch` at a fixed mip level; do not enable filtering or sRGB.
4. Map adjacent invocations to adjacent tensor values and make the reduction
   dimension cache-friendly in image coordinates.
5. Implement 4x4 and 6x6 variants, or compile-time specializations, using the
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

1. Compare output logits against the baseline for fixed prompts.
2. Run perplexity or another model-appropriate quality evaluation.
3. Measure prompt processing and token generation separately.
4. Test at least two distinct GPU/driver families where practical.
5. Profile with vendor tools when available; do not infer texture-decoder
   parallelism from API-visible properties.

Decision points:

- Prefer 4x4 where accuracy dominates.
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
