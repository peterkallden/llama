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

## Additive latents and codec-aware optimization

ASTC-Latent is a separate hypothesis from storing a conventional Q4 tensor in
an ASTC image. The proposed runtime contract is a sampled, cheap additive
reconstruction:

\[
\hat w = s_L L' + s_A A' + b.
\]

The initial `RGB=L, A=A` layout should be understood as a luminance-plus-alpha
*source signal*, not as an instruction to the decoder. ASTC dual-plane blocks
have two interpolation weight grids, with the second grid applied to one
selected component. They do not expose two arbitrary independent endpoint
planes, and Vulkan exposes no API to request a dual-plane block at sampling
time. The offline packer must inspect its final bitstream, and a future
constrained encoder must prove that any intended dual-plane selection is legal
and effective.

This direction is inspired by two complementary research strands. AQLM shows
that learned additive, input-adaptive representations can be useful for LLMs
at very low rates, but its learned vector codebooks are not automatically
provided by ASTC. Neural-material work demonstrates the relevant systems
pattern: train through a block-codec approximation, export ordinary compressed
textures, and use fixed-function texture decode at runtime. PV-Tuning is a
candidate method for optimizing a discrete representation without relying only
on a straight-through estimator. The immediate experiment should remain much
simpler: establish a post-training two-latent control, optimize by held-out
activation error, and only then move to codec-aware latent or codebook tuning.

## Neural error shaping for independently decodable blocks

The central research principle is broader than ASTC: a hardware-decodable block
codec does not need to minimize the numerical error of every block. It should
select legal blocks whose *aggregate* error lies in directions that a layer or
model is least sensitive to. For a linear layer with weight error
`E = W - W_hat` and calibration inputs `X`, the immediate layer-output error is

\[
\|X E^T\|_F^2 = \operatorname{tr}(E H_I E^T),
\qquad H_I = X^T X.
\]

This is the input-side, GPTQ-like objective already used by the offline
adapter. It differs from texture RGBA MSE and decoded weight MSE: it can accept
a larger numerical error if the error is in an input direction the layer does
not use.

The existing ASTC coordinate-selection contract gives a small but important
proof of mechanism. It replaces only complete, legal 128-bit ASTC blocks and
updates the exact cached output residual. On synthetic fixtures it beat either
uniform stream, so cross-block error cancellation is real. On the first
real-weight run it improved ordinary image ranking but did not beat the uniform
neural-ranked stream on held-out activations. A third whole-image stream made
the holdout result worse. This is evidence of insufficient candidate diversity
and calibration overfit, not evidence that global selection is intrinsically
beneficial. Every later error-shaping experiment must therefore use nested
calibration/validation splits and retain the uniform stream as a control.

### Candidate diversity before global optimization

For a legal candidate `c` for one ASTC block, let `E_c` be its decoded weight
error and let `L` satisfy `H_I = L L^T`. Its sensitivity-weighted error is

\[
Z_c = E_c L.
\]

The candidate list should not be merely the `K` smallest values of
`||Z_c||_F`. Those candidates can be nearly identical and leave coordinate
selection no useful tradeoff. The first candidate should be the local optimum;
the remaining bounded candidates should cover different directions of `Z_c`,
for example via a loss threshold followed by farthest-point selection or
clustering. Exact cached-residual selection can then exploit genuinely opposed
or complementary output deltas. This is the immediate next experiment because
it improves the information available to the current legal-block selector
without changing the runtime, ASTC stream, or Vulkan shader.

### From coordinate selection to Hessian-guided error feedback

The next offline algorithm is a block-codec analogue of adaptive rounding.
Once a block has been committed, its error can be projected into a bounded
input-sensitivity basis and fed forward as a target correction for blocks that
are not yet encoded. This is *error diffusion in sensitivity space*, not image
space. It must first be checked against exact coordinate descent on small
fixtures, because coordinate descent already accounts for cross terms exactly
on its calibration trace. Error feedback is useful only if it reaches a better
holdout tradeoff at materially lower search cost or with a larger candidate
space.

The first comparison must be deliberately narrower. Given the *same fixed
pool of legal ASTC candidates*, compare local neural ranking, exact
cached-residual coordinate descent, and Hessian feedback. Coordinate descent
is an oracle-ish offline reference because it sees the actual current layer
residual. Fixed-pool feedback tests whether a compact curvature approximation
recovers that information; it must not be mixed with the separate benefit of
regenerating later candidates from a changed source target. Conditional target
regeneration is the stronger second-stage encoder experiment.

For comparable layers, report both absolute losses and the recovered
coordinate-descent gain

\[
G_{HF} = \frac{L_{local} - L_{HF}}{L_{local} - L_{CD}}.
\]

This ratio is reported only when `L_local > L_CD`; finite holdout noise can
otherwise make it undefined, negative, or greater than one. During selection,
record the residual before and after every accepted block and the alignment
between the proposed output delta and that residual. With
`R_next = R - DeltaY`, positive `cos(R, DeltaY)` is direct evidence of error
cancellation rather than merely a different ASTC block choice.

Candidate diversity must also be measured in neural space. The practical form
is `D_c = E_c X^T`, the candidate's output-error delta over calibration
examples. This avoids constructing a full Hessian. If a whitened basis is
needed, use a thin QR or SVD of `X`, because typical trace counts make
`H_I = X^T X` rank-deficient. Retain a bounded set containing a local optimum,
low-correlation and negative-correlation alternatives, and candidates covering
the next principal error direction, each subject to an activation-loss bound.

The first implementation uses simultaneous residual feedback (a block-Jacobi
approximation) to make the decisions parallel. Initial smoke measurements are
deliberately treated as a negative-control baseline: on the small synthetic
fixture, local neural ranking reached `7.02e-5`, coordinate descent `4.56e-5`,
and feedback `1.17e-4` on holdout. On the bounded SmolLM2 layer probe, feedback
also trailed local/coordinate selection for 4x4, 5x5, and 6x6. This is useful
evidence that reading a shared residual is not by itself a sufficient Hessian
approximation; later work needs damping, a compact curvature model, or a
conflict-aware schedule. The comparison harness remains valuable because all
three selectors use the same legal candidate pool.

The follow-up acceptance pass keeps proposal generation parallel but rechecks
each proposal against the updated residual. It improved the synthetic 4x4
holdout beyond coordinate descent, while on the SmolLM2 probe it reduced
calibration error without beating local holdout error. A two-shard gate that
requires positive normalized gain in both calibration halves restored holdout
behavior close to local ranking. This separates conflict overshoot from
calibration-specific directions and motivates collecting larger traces before
using more shards.

### Two-sided sensitivity is a later, stronger objective

ASTC blocks span both output and input dimensions. If a defensible
output-sensitivity factor becomes available, candidate ranking can use

\[
\mathcal L_{2D} = \operatorname{tr}(H_O E H_I E^T).
\]

The normal activation reconstruction loss is the special case `H_O = I`.
It must not be described as a full-model Hessian. A nontrivial `H_O` requires
separate downstream/model-loss sensitivity capture or a Kronecker-factored
approximation. This added information is promising for 5x5 and 6x6 blocks but
should follow, not replace, the one-sided holdout gate.

### Deliberately exotic follow-ons

A projected-residual beam or trellis can choose a sequence of ASTC candidates
using a compact 4--16-dimensional state rather than a full residual matrix.
This borrows the *search principle* of trellis-coded quantization while keeping
the final representation as ordinary, randomly addressable ASTC blocks. It is
not justified until a diverse candidate shortlist shows a holdout gain.

The L+A representation has a second, more unusual opportunity. With
`W = s_L L + s_A A + b`, the change

\[
L' = L + \alpha R, \qquad
A' = A - \frac{s_L}{s_A}\alpha R
\]

leaves the uncompressed reconstructed weight unchanged. It creates a family of
redundant latent signals that can be searched for ASTC-friendly correlation,
dual-plane use, and sensitivity-shaped post-decode error. This is a later
offline representation search; it does not imply that Vulkan exposes new ASTC
operations or that the two latent fields survive ASTC independently.

Finally, codec-aware fine-tuning can inject *measured projected ASTC decode
error* or the real encode/decode operation in the forward path. This is more
meaningful than generic Gaussian noise, but it belongs after the post-training
experiments because it introduces training cost and can hide a weak codec
representation behind model adaptation.

### Scaling interpretation

The bounded 32x64 selection probe must not be extrapolated directly to a full
model layer. A 576x576 projection contains 162 times as many weights and,
depending on ASTC footprint, thousands rather than tens of independently
selectable blocks. This is favorable only conditionally: more blocks provide
more opportunities for cancellation, but they also multiply the selector's
degrees of freedom and its ability to overfit a small calibration trace.

The relevant quantity is therefore not matrix size alone. A larger layer helps
only if the calibration covariance is representative and the legal ASTC
candidate errors span directions that the layer can usefully cancel. Every
full-layer claim must record the candidate count, block count, calibration,
validation, and holdout cohort sizes; it must beat the corresponding uniform
ASTC stream on an untouched holdout. Attention and FFN matrices must be
reported separately, since their activation geometry and ASTC-compatible
structure need not agree.

Search complexity is deliberately an offline concern. It may grow with model
size, but the implementation must stream candidate blocks and retain only a
compact sensitivity sketch rather than a full candidate-by-trace tensor. None
of this changes the runtime contract: the deployed resource remains an ordinary
ASTC image, and the shader performs the same sampled reconstruction regardless
of how its blocks were selected.

## Encoder boundary decision

The research does not need a custom Vulkan decoder. A legal ASTC block stream
is interpreted by the standard Vulkan driver and its fixed-function texture
unit; the shader still sees an ordinary sampled `vec4`. The missing control is
offline: the public `astcenc` API exposes quality/tuning knobs and block
diagnostics, but it does not make the complete block-mode, endpoint-mode,
partition, dual-plane, and BISE search space a stable application contract.

The preferred implementation is consequently a **constrained astcenc fork or
upstreamable encoder extension**, not a clean-room ASTC rewrite:

1. Keep the current external `astcenc` adapter as the scalar/Q4 and generic
   baseline.
2. Reuse its legal ASTC bit packing, endpoint quantization, BISE packing, and
   reference decode. Add a neural objective and candidate restrictions in the
   offline search layer.
3. Initially expose candidate preferences for L+A endpoint modes, alpha
   dual-plane, weight-grid size, partitions, and few-level endpoint/weight
   alphabets. Do not force a mode until block-info statistics and quality show
   that forcing it helps.
4. Validate each output three ways: `astcenc_get_block_info`, CPU
   encode/decode roundtrip, and the existing Vulkan `texelFetch` smoke on a
   capable target device.

Only if the fork cannot express the needed search should we consider writing a
small constrained packer from the ASTC specification. That packer would still
emit standard 128-bit blocks. A non-standard block format would require a
shader decoder and storage-buffer path, removing the fixed-function bandwidth
hypothesis and therefore belongs to a separate experiment.

This boundary also keeps upstream risk manageable: the encoder is an optional
offline tool, while llama.cpp retains the ordinary Vulkan buffer path and a
clear fallback when ASTC support or the experimental encoder is unavailable.

## References

- [Khronos Vulkan format definitions](https://docs.vulkan.org/spec/latest/chapters/formats.html)
- [Khronos compressed image formats](https://docs.vulkan.org/spec/latest/appendices/compressedtex.html)
- [Arm: ASTC Does It](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/astc-does-it)
- [Arm ASTC encoder format overview](https://github.com/ARM-software/astc-encoder/blob/main/Docs/FormatOverview.md)
- [Khronos ASTC data-format specification](https://github.com/KhronosGroup/DataFormat/blob/main/astc.txt)
- [AQLM: Extreme Compression of LLMs via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [QuIP: 2-Bit Quantization of Large Language Models with Guarantees](https://arxiv.org/abs/2307.13304)
- [QTIP: Quantization with Trellises and Incoherence Processing](https://arxiv.org/abs/2406.11235)
- [YAQA: Model-Preserving Adaptive Rounding](https://arxiv.org/abs/2505.22988)
- [BaKron: Efficient Quantization with Kronecker-Factored Hessians](https://arxiv.org/abs/2608.06291)
- [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/abs/2405.14852)
- [Real-Time Neural Materials using Block-Compressed Features](https://arxiv.org/abs/2311.16121)
- [Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors](https://arxiv.org/abs/2506.06040)
- [Khronos Vulkan ML tutorial: sampler-assisted resizing](https://github.khronos.org/Vulkan-Site/tutorial/latest/ML_Inference/Desktop_Applications/05_real_time_camera.html)
- [Basis Universal: UASTC texture specification](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-Texture-Specification)
- [Basis Universal: Textures as Universal Latents](https://github.com/BinomialLLC/basis_universal/wiki/Basis-Universal%3A-Textures-as-Universal-Latents)
- [Basis Universal repository](https://github.com/BinomialLLC/basis_universal)
