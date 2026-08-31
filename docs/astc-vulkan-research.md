# ASTC-Backed Weight Storage for Vulkan

## Status

This document records an experimental research direction. It does not propose a
new core `ggml` quantization type or a replacement for the existing Vulkan
backend. The normal buffer-backed Vulkan path remains the required fallback.

## Objective

Evaluate whether a Vulkan GPU's fixed-function ASTC texture decoder can serve
as a hardware-assisted representation and dequantization path for *static* LLM
weights. The intended data flow is:

```text
quantized tensor
    -> ASTC-aware offline packing
    -> ASTC sampled image in GPU memory
    -> hardware ASTC decode
    -> compute shader reconstruction and dot product
```

ASTC is not programmable general-purpose compute. A shader asks for a texel;
the implementation decodes the associated compressed block in the texture
pipeline. The experiment is therefore about changing the representation and
access pattern, not about executing custom instructions on an ASTC unit.

## Motivation

Autoregressive LLM inference is frequently constrained by reading model
weights rather than by arithmetic throughput. Existing low-bit Vulkan kernels
typically read packed values from `VkBuffer` storage and explicitly unpack and
dequantize them in the compute shader.

ASTC uses a 128-bit compressed block for every two-dimensional block extent.
The same 128 bits encode 16 texels with ASTC 4x4 and 36 texels with ASTC 6x6:

| Format | Compressed block | Texels per block | Nominal bits per texel |
|---|---:|---:|---:|
| ASTC 4x4 | 128 bits | 16 | 8.00 |
| ASTC 6x6 | 128 bits | 36 | 3.56 |

If every RGBA texel is interpreted as four reconstructed scalar weights, the
storage density is 2.00 and 0.89 nominal bits per reconstructed channel,
respectively. These values are *not* the number of independent, exact bits per
weight: ASTC shares coding parameters across a block. They express the memory
traffic opportunity that must be balanced against reconstruction error.

## Mathematical correspondence

A conventional affine quantizer reconstructs a scalar as:

\[
x_i = s(q_i-z),
\]

where `q_i` is a quantized code, `s` is a scale, and `z` is a zero point.

In simplified form, ASTC reconstruction can be viewed as an endpoint
interpolation:

\[
\hat{x}_i = E_0 + w_i(E_1-E_0),
\]

where `E_0` and `E_1` are block endpoints and `w_i` is a decoded weight.
ASTC has additional modes, partitions, and endpoint representations; the
equation is a useful model, not a complete format specification.

The research question is whether an ASTC-aware packer can select endpoints,
weights, channel assignments, and a tensor layout that make `\hat{x}_i`
sufficiently close to `x_i` for a matrix-vector product:

\[
\hat{y}_r = \sum_j \hat{W}_{rj} a_j.
\]

The relevant error is consequently not only elementwise reconstruction error,
but also dot-product error, logit error, and ultimately model quality.

## 4x4 and 6x6 hypotheses

ASTC 4x4 has more compressed bits available per texel and is expected to be a
higher-fidelity baseline. ASTC 6x6 has materially lower weight-storage traffic
and is the stronger bandwidth hypothesis, but shares endpoint and weight
information across more values. LLM weights have no guarantee of the local
spatial correlation that image-oriented ASTC encoding assumes.

Both formats must be evaluated. A viable implementation may choose the format
per tensor or per layer rather than globally.

## Parallelism and layout

Vulkan does not expose an ASTC decoder count, ASTC latency, or requests per
cycle. The implementation must therefore maximize ordinary GPU parallelism:

- dispatch sufficient workgroups to maintain occupancy;
- arrange adjacent invocations to fetch adjacent texels;
- map the contiguous reduction dimension to nearby image coordinates;
- use `texelFetch` and a fixed mip level initially, avoiding filtering as an
  additional source of approximation;
- measure actual GPU time and cache behavior instead of assuming decoder
  throughput.

A texture fetch returns the requested texel, normally a `vec4`, rather than a
whole ASTC block. Nearby fetches may benefit from texture/block-cache reuse,
but the implementation is device-specific and must not be assumed by the
correctness path.

## Scope and non-goals

The first experiment covers static model weights, Vulkan compute, LDR UNORM
ASTC 4x4 and 6x6 sampled images, and a matrix-vector kernel. It excludes
runtime ASTC encoding, KV-cache storage, filtering-based computation,
vendor-specific assembly, and changes to the public GGUF format.

KV cache is a possible later experiment. Unlike static weights, it is updated
for every token and therefore needs an efficient, valid compressed-block write
path before it can be considered.

## Basis Universal as an encoder-method reference

The supplied Basis Universal material is useful for improving the *offline
encoder methodology*, but it is not a drop-in replacement for the Vulkan
runtime path. Its UASTC representation uses ASTC-like endpoints, weights,
partitions, quantization tables, and 128-bit blocks, yet UASTC is a distinct
bitstream and must not be uploaded as a standard Vulkan ASTC image. The
repository also describes a canonical compact intermediate representation that
can be transcoded to a device format at load time. These ideas support a
two-stage design: keep a canonical tensor/latent representation, then produce
ordinary ASTC 4x4 or 6x6 blocks for the selected GPU.

The encoder techniques worth borrowing are candidate search and objective
design: search endpoint modes, partitions, weight grids, dual-plane choices,
component assignments, and tensor permutations; then score each candidate
after ASTC quantization and decode. Endpoint prediction, DPCM, supercompression,
and cross-block rate-distortion coding are useful for disk or network size, but
must not be copied into the GPU-resident format when they compromise independent
random access to a block.

For neural weights, image PSNR or generic MSE is not sufficient. The packer
should rank candidates with a weighted objective such as:

\[
L = \alpha\,\mathrm{MSE}(W,\hat W)
  + \beta\,\mathbb{E}_a\left[(Wa-\hat W a)^2\right]
  + \gamma\,L_{\mathrm{outlier/tail}},
\]

where `a` is drawn from representative or held-out activation vectors. A
later model-level pass can add logit or perplexity loss. This directly tests
whether a candidate's endpoint/weight decisions preserve the computation that
llama actually performs. The current smoke only has one fixed activation
vector, so it is a baseline rather than a production-quality criterion.

The current evidence reinforces this separation. Generic `astcenc` row-major
packing produced poor weight reconstruction; a 24-way channel search improved
one 4x4 dot-product sample but did not make the representation acceptable, and
a naive magnitude-grouped spatial order was worse than identity. Basis-style
candidate search is therefore a promising next step, while blindly adopting
UASTC or image-oriented heuristics is not.

## References

- [Khronos Vulkan format definitions](https://docs.vulkan.org/spec/latest/chapters/formats.html)
- [Khronos compressed image formats](https://docs.vulkan.org/spec/latest/appendices/compressedtex.html)
- [Arm: ASTC Does It](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/astc-does-it)
- [Arm ASTC encoder format overview](https://github.com/ARM-software/astc-encoder/blob/main/Docs/FormatOverview.md)
- [Real-Time Neural Materials using Block-Compressed Features](https://arxiv.org/abs/2311.16121)
- [Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors](https://arxiv.org/abs/2506.06040)
- [Khronos Vulkan ML tutorial: sampler-assisted resizing](https://github.khronos.org/Vulkan-Site/tutorial/latest/ML_Inference/Desktop_Applications/05_real_time_camera.html)
- [Basis Universal: UASTC texture specification](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-Texture-Specification)
- [Basis Universal: Textures as Universal Latents](https://github.com/BinomialLLC/basis_universal/wiki/Basis-Universal%3A-Textures-as-Universal-Latents)
- [Basis Universal repository](https://github.com/BinomialLLC/basis_universal)
