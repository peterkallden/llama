# ASTC Vulkan Weight Storage: Work-in-Progress Log

This is the engineering log for the experimental ASTC-backed weight-storage
work. It records decisions, observations, and changes against
[`astc-vulkan-research.md`](astc-vulkan-research.md) and
[`astc-vulkan-plan.md`](astc-vulkan-plan.md).

## Scope and status

The work is on the isolated branch `kallden/vulkan-astc-int4-decoder`.
The normal llama.cpp/ggml Vulkan backend remains unchanged and is the required
fallback. The experiment currently has no production ASTC resource path. It
now has a device-backed validation executable that can upload a known-valid
constant ASTC block, execute the compiled shader, and read the reconstructed
texels when a compatible physical device is available.

Current phase: **Phase 1, contract and host-smoke foundation**.

## Decisions made

### Repository structure

The proof of concept lives in `pocs/astc-vulkan/`, parallel to the production
backend rather than copying it. This keeps the experiment independently
buildable while avoiding a second copy of device, pipeline, descriptor, and
memory-management code. It is enabled only with `LLAMA_ASTC_VULKAN_POC=ON`.

### Representation

- Evaluate both ASTC 4x4 and ASTC 6x6.
- Treat the two formats as alternatives, not as baseline and optional bonus.
- Use RGBA conceptually as four reconstructed scalar channels per texel.
- Interpret 128-bit block size and texel footprint as storage-density contracts,
  not as a claim of independently exact bits per weight.
- Begin with static model weights. KV-cache updates are a later research track.

### GPU and Vulkan model

- ASTC is used through Vulkan sampled images and `texelFetch` in compute
  shaders; the decoder itself is fixed-function and is not directly
  programmable.
- The shader is responsible for tensor-coordinate mapping, companion metadata,
  accumulation, and any remaining numerical correction.
- We will not assume a portable number of ASTC decoder units, requests per cycle,
  or latency. These must be measured on each device.
- The first shader must use nearest/fixed-level fetches. Filtering is deferred
  because it would add a second approximation mechanism.

### Repository and upstream strategy

- Keep this work isolated from the production path until measurements justify an
  integration.
- Follow existing llama.cpp/ggml C++, CMake, shader-generation, descriptor,
  resource-lifetime, and test conventions.
- Avoid public GGUF and `ggml` core changes in the first implementation.
- Preserve the buffer-backed Vulkan path and make ASTC opt-in and fallback-safe.
- Keep each future change small enough for independent upstream review.

## Mathematical contract

The reference affine quantizer is:

\[
x_i = s(q_i-z).
\]

The simplified ASTC interpolation model is:

\[
\hat{x}_i = E_0 + w_i(E_1-E_0).
\]

For a matrix-vector operation the relevant output is:

\[
\hat{y}_r = \sum_j \hat{W}_{rj}a_j.
\]

Therefore, tests must eventually cover element error, dot-product error, and
model-level logit/quality error. Elementwise error alone is insufficient.

The nominal storage densities are:

| Format | Texels/block | Nominal bits/texel | Nominal bits/channel (RGBA) |
|---|---:|---:|---:|
| ASTC 4x4 | 16 | 8.00 | 2.00 |
| ASTC 6x6 | 36 | 3.56 | 0.89 |

ASTC endpoint, partition, and weight sharing is the source of compression and
the principal numerical risk. A normal image compressor is only a baseline;
the eventual packer must optimize neural-weight and dot-product objectives.

## Implemented in the current sweep

- Added `pocs/astc-vulkan/` with an opt-in CMake target, host-neutral format
  constants, contract test, and no-GPU host smoke.
- Added an optional standard-Vulkan capability probe for sampled ASTC 4x4 and
  6x6 formats.
- Added a minimal `texelFetch` compute shader source; compilation is conditional
  on `glslc` availability.
- Kept production `ggml-vulkan` sources and shaders untouched.
- Added an optional device shader smoke that uses a known-valid ASTC void-extent
  block, sampled-image descriptors, a compute pipeline, and host readback.

The first focused build was configured with `LLAMA_BUILD_TESTS=ON`,
`LLAMA_BUILD_EXAMPLES=ON`, `LLAMA_ASTC_VULKAN_POC=ON`, and `GGML_VULKAN=OFF`.
Both ASTC tests passed under CTest (`2/2`). This confirms the contract and
host-smoke layer without making unsupported claims about a GPU.

The next sweep adds an optional Vulkan capability probe and a minimal
`texelFetch` compute shader. The probe returns CTest skip code 77 when no Vulkan
device supports both sampled ASTC formats. The shader compile smoke is only
registered when `glslc` is available.

In the second focused run, the probe built and CTest passed (`3/3`). The local
Vulkan installation enumerated an Intel UHD Graphics 620 device with both
formats sampleable; the NVIDIA GeForce 920MX and llvmpipe devices reported no
sampled ASTC support. `glslc` was not installed, so the shader compile smoke was
not registered. This confirms that capability must be selected per device and
not inferred from the presence of a Vulkan loader alone.

In the third focused run, the device resource smoke built and CTest passed
(`4/4` including the previous tests). The local Intel UHD Graphics 620 device
accepted both formats for sampled and transfer-destination usage, and both image
uploads and layout transitions completed. The NVIDIA GeForce 920MX and llvmpipe
remain excluded by the capability probe. `glslc` is still unavailable locally,
so shader compilation and texel readback remain the next device-backed step.

In the fourth focused run, the validation shader compiled successfully with the
Android NDK `glslc` (1.4.341 toolchain) and CTest passed (`5/5`). The compile
test confirms GLSL/SPIR-V syntax and descriptor declarations only; it does not
yet execute the shader or validate decoded texel values.

The final review for this etapp passed `git diff --check` and a separate
`-Wall -Wextra -Werror` compilation of all new C++ sources. The device smoke was
refactored to take its staging size from the shared ASTC contract instead of
duplicating the 128-bit constant. A final rebuild and ASTC CTest run passed
(`5/5`).

In the fifth focused run, the device shader smoke was wired into CMake and CTest
for both 4x4 and 6x6. The shader writes one reconstructed `vec4` per texel into
a storage buffer, and the host validates every channel against the 0.5 constant
encoded in the void-extent block. The shader compiled and all host tests passed;
the four device-dependent tests were skipped in this sandbox because no
physical ASTC-capable Vulkan device was exposed at run time. Earlier probing had
seen an Intel UHD Graphics 620 with both formats, so the executable remains
ready for a host/device run outside the restricted environment.

The sixth sweep made the bandwidth arithmetic an explicit host contract. The
shared header now computes block counts and padded image storage in 64-bit
bytes, and the contract test locks the 4096x4096 examples to 64 MiB for 4x4
and 7,463,824 bytes (about 7.12 MiB) for 6x6. These figures describe the
compressed image allocation, including edge-block padding; they are not a claim
about effective model weight bits after metadata or alignment.

The seventh sweep added `astc-vulkan-tensor-contract.h` and a focused CTest for
the first tensor mapping proposal. A logical row-major matrix packs four scalar
columns into each RGBA texel, rounds the final texel channels, and then applies
ASTC block rounding to the texel image. The metadata record is private to the
PoC and carries only logical dimensions plus an affine scale/offset; no GGUF or
ggml public type changed. This makes the future packer/shader boundary explicit
without pretending that an encoder already exists.

The eighth sweep added optional Vulkan timestamp queries around the compute
dispatch in the shader-device smoke. When the selected queue reports
`timestampValidBits`, the executable prints elapsed nanoseconds using the
physical device's `timestampPeriod`; otherwise correctness still runs without a
timing claim. This is instrumentation only: sequential/non-local access
patterns and a bandwidth comparison are intentionally not inferred yet.

In the ninth sweep, the host render nodes were exposed to the test process and
the full ASTC CTest label passed (`8/8`). The Intel UHD Graphics 620 accepted
both ASTC formats; image-resource smoke and shader readback passed for 4x4 and
6x6. One run reported approximately 18.5 microseconds for the 4x4 dispatch and
20.17 microseconds for 6x6 with the current 64-invocation workgroup. These are
smoke-test timestamps, not a bandwidth benchmark or a decoder-throughput claim.

The tenth sweep added an explicit access-pattern push constant to the validation
shader. The default path maps adjacent invocations to adjacent texels; the
`nonlocal` path applies a bijective `(index * 7 + 3) mod texel_count` permutation
for both smoke image sizes. CTest now runs both patterns for 4x4 and 6x6, so
future timing data will compare identical decoded values under different
locality assumptions.

The eleventh sweep collected ten timestamp samples for each pattern on Intel
UHD Graphics 620 using Mesa 26.0.8-1ubuntu0.3. The current shader uses a
64-invocation workgroup and each smoke image occupies one 128-bit ASTC block.

| Format | Pattern | Min (ns) | Median (ns) | Mean (ns) | Max (ns) |
|---|---|---:|---:|---:|---:|
| 4x4 | sequential | 17,333 | 22,917 | 23,133 | 36,417 |
| 4x4 | nonlocal | 17,167 | 22,333 | 22,917 | 36,917 |
| 6x6 | sequential | 18,250 | 20,583 | 22,083 | 31,167 |
| 6x6 | nonlocal | 18,500 | 25,167 | 25,242 | 37,250 |

The samples are intentionally treated as a harness check, not a performance
claim. The images are too small and the variance too large to establish a
locality effect. The next benchmark must use larger ASTC images and repeated
dispatches while keeping upload and pipeline setup outside the timed region.

The twelfth sweep added an explicit `--benchmark` mode. It fills a 192x192
image (36,864 bytes for 4x4 and 16,384 bytes for 6x6), dispatches 20 times, and
places timestamps around only the post-upload compute section. Five samples per
configuration on the same Intel/Mesa setup were:

| Format | Pattern | Min (ns) | Median (ns) | Mean (ns) | Max (ns) |
|---|---|---:|---:|---:|---:|
| 4x4 | sequential | 341,083 | 414,833 | 423,233 | 490,833 |
| 4x4 | nonlocal | 343,417 | 374,500 | 381,867 | 432,667 |
| 6x6 | sequential | 351,000 | 433,917 | 438,717 | 527,333 |
| 6x6 | nonlocal | 363,667 | 394,833 | 397,967 | 436,333 |

The nonlocal pattern happened to be faster in this short sample, so the data
does not establish a locality benefit. Driver scheduling, cache state, and the
small benchmark size still dominate. The result is useful as a repeatable
harness baseline and reinforces that a larger, statistically controlled sweep
is needed before comparing 4x4 and 6x6.

Earlier, the host did not provide the optional `astcenc` executable. Phase 2
therefore treats encoding as an external offline tool boundary and does not add
a local ASTC bitstream implementation until its format and numerical
requirements are justified. Vulkan runtime tests remain independent of that
tool.

The thirteenth sweep used the newly installed `libastcenc-dev` CMake package to
run a host-only encoder/decode smoke. It compresses a deterministic RGBA float
fixture at 4x4 and 6x6, decompresses both outputs, and reports MSE/max error.
The fixture produced MSE 0.00060992 (4x4) and 0.00067205 (6x6), with maximum
channel errors of about 0.0557 and 0.0771 respectively. The smoke therefore
uses MSE <= 0.005 as its baseline contract and reports max error for visibility;
these values are not acceptable neural-weight quality limits yet. The Vulkan
path remains independent: this is an offline baseline adapter, not yet a
neural-weight-aware packer or a GGUF format change.

The fourteenth sweep added a tensor roundtrip smoke. A deterministic 12x48
weight matrix is packed as four normalized scalar channels per texel, encoded
with ASTC 4x4 and 6x6, decoded on the CPU, mapped back through the affine
weight scale, and evaluated with a fixed activation vector. The test reports
element MSE/max error and dot-product error for both formats. This is the first
weight-oriented numerical contract; it still uses a generic image encoder and
does not claim production quantization quality.

The first weight-oriented run is intentionally a negative result. The generic
image encoder produced the following metrics after mapping the decoded UNORM
values back to the original [-1, 1] weight range:

| Format | Compressed bytes | MSE | Max error | Dot-product error |
|---|---:|---:|---:|---:|
| 4x4 | 144 | 0.03747012 | 0.696776 | 0.133961 |
| 6x6 | 64 | 0.09105601 | 0.970683 | 0.377323 |

The roundtrip is technically valid, but these errors are far beyond an
inference-quality target. The likely cause is that image-oriented ASTC shares
endpoints and interpolation weights across spatial neighborhoods, while the
current row-major matrix mapping has no corresponding spatial correlation. This
rules out treating a generic image encode as a usable weight packer. The
adapter is retained as a baseline and regression fixture only.

The fifteenth sweep searched all 24 RGBA channel assignments for the same
12x48 matrix and fixed activation vector. The best MSE order was `0231` for
both formats. It reduced the 4x4 dot-product error from 0.133961 to 0.041689,
but left MSE at 0.03734365; for 6x6 it left MSE at 0.09071333 and increased
dot-product error to 0.513958. Channel assignment is therefore worth exposing
as a packer degree of freedom, but it cannot compensate for the larger spatial
correlation mismatch. Spatial block layout and objective-driven packing remain
the next research target.

The sixteenth sweep compared the identity texel order with a deterministic
magnitude-grouped order that sorted rows and four-column groups by aggregate
absolute weight. The grouped order was a deliberately simple locality and
correlation hypothesis, not a final packer. It was worse on this fixture:

| Format | Layout | Best order | MSE | Max error | Dot-product error |
|---|---|---|---:|---:|---:|
| 4x4 | identity | `0231` | 0.03734365 | 0.696776 | 0.041689 |
| 4x4 | grouped | `0123` | 0.06857636 | 0.908702 | 2.727908 |
| 6x6 | identity | `0231` | 0.09071333 | 0.842460 | 0.513958 |
| 6x6 | grouped | `1203` | 0.13164938 | 1.004278 | 3.026635 |

The result is useful because it rejects a plausible but unsupported spatial
heuristic. ASTC block locality must be designed around a neural objective and
validated on more than one fixture; it should not be inferred from magnitude
sorting alone.

The seventeenth sweep reviewed the supplied Basis Universal implementation and
documentation. The useful transfer is methodological: candidate search over
endpoint modes, partitions, weight grids, dual-plane/component choices, and
permutations, followed by scoring after exact ASTC quantization and decode.
UASTC itself is not the runtime format for this experiment; it is a distinct
ASTC-like intermediate/supercompressed representation. Cross-block predictors,
DPCM, and rate-distortion choices may improve file size, but the GPU-resident
path must retain independently addressable standard ASTC 4x4/6x6 blocks.

The next quality objective will combine elementwise error with activation-
weighted matrix-vector error (and later logit error), for example
`alpha * MSE + beta * E[(W*a - W_hat*a)^2] + gamma * tail_error`. This gives
Basis-style encoder search a llama-relevant score instead of relying on image
PSNR. No change is made to the Vulkan runtime or ggml tensor path yet.

The eighteenth sweep implemented that first neural objective in the host
weight smoke. Four deterministic activation vectors are evaluated across all
rows, and each candidate reports activation MSE in addition to element MSE and
the legacy first-vector dot error. Candidate ranking now minimizes
`elementwise_MSE + activation_MSE`; this is deliberately simple and keeps the
weights between the two terms explicit for later calibration.

With the updated objective, the identity layout remained selected for both
formats. The best channel orders changed because the objective now sees four
activation patterns:

| Format | Layout | Best order | MSE | Activation MSE | Objective |
|---|---|---|---:|---:|---:|
| 4x4 | identity | `2301` | 0.03734940 | 0.05564304 | 0.09299244 |
| 4x4 | grouped | `0132` | 0.07066771 | 0.19674188 | 0.26740959 |
| 6x6 | identity | `0312` | 0.09106646 | 0.33504080 | 0.42610726 |
| 6x6 | grouped | `1230` | 0.13563544 | 0.37468951 | 0.51032495 |

This is still a host-side search fixture, not a claim about model quality. The
next improvement is to feed representative activation samples from a real
llama layer and then search ASTC encoder candidates under this objective.

The nineteenth sweep added an encoder-quality search over `astcenc`'s fast,
medium, and thorough presets. Each preset is evaluated across the existing
channel and spatial candidates, and the best result is selected by the neural
objective. This is the first concrete candidate-search method inspired by the
Basis Universal approach; the output remains ordinary ASTC blocks and no
runtime Vulkan code changes.

On the deterministic fixture, `medium` remained best for 4x4 with objective
0.09299244. For 6x6, `thorough` reduced the objective to 0.28531916 compared
with 0.42610726 for medium, despite the same 64-byte image footprint. This is
promising evidence that encoder search can improve quality without changing
the runtime representation, but it is not yet a model-quality result and may
cost substantially more offline packing time.

The twentieth sweep added per-block error accounting to the weight smoke. Each
reconstructed scalar is attributed to the encoded ASTC block (after any texel
permutation), and the result reports P95 and maximum block MSE alongside the
global metrics. The selected candidates measured block P95/max of 0.06200861
for 4x4 medium and 0.09616162 for 6x6 thorough. Because this fixture contains
only 9 and 4 blocks respectively, P95 equals the worst block; larger real
layers are required before interpreting the percentile statistically.

The twenty-second sweep added an optional 64x256 host fixture (`--large`) and
registered it as a separate CTest. The larger image represents 16x16 ASTC 4x4
blocks or 11x11 ASTC 6x6 blocks, making the block percentile meaningful while
keeping the original small fixture as a fast contract test.

The larger fixture selected 4x4 identity/thorough with objective 0.36033993
and 6x6 identity/medium with objective 0.84021780. The ordering differs from
the small fixture, which confirms that encoder quality and channel choices must
be evaluated per tensor shape and activation distribution rather than frozen
from one toy matrix. The absolute errors remain far too high for inference;
this is a search-harness result only.

The twenty-third sweep broadened both fixtures to six activation distributions:
three smooth patterns, one bounded pseudo-random pattern, and one sparse
outlier pattern in addition to the existing baseline. The selected formats did
not change, but channel orders and activation losses did. For the larger
fixture, 4x4 identity/thorough scored 0.46121455 while 6x6 identity/medium
scored 1.11635062; the sparse/random samples exposed much larger 6x6
activation error than the smooth-only run. This reinforces that representative
activation data is a prerequisite for trusting an offline encoder choice.

The twenty-fourth sweep tested a block-local texel reversal that preserves
independent ASTC blocks while changing the order within each block. On the
small fixture it improved the selected objective to 0.11280460 for 4x4 and
0.31777791 for 6x6, compared with 0.13701558 and 0.33248337 for identity.
On the larger fixture it was worse than identity for both formats (0.53930804
versus 0.46121455 for 4x4, and 1.21237273 versus 1.11635062 for 6x6). The
candidate remains in the search harness, but selection is now explicitly
fixture-dependent; no universal permutation is assumed.

The twenty-fifth sweep added reference-energy normalization to the activation
loss. The smoke now reports both raw activation MSE and relative activation MSE;
candidate ranking uses the latter together with weight MSE and block-tail loss.
On the small fixture, block-reversed candidates reached objectives 0.09723277
(4x4) and 0.26363581 (6x6). On the large fixture, identity remained better at
0.23307591 (4x4) and 0.55860837 (6x6), while block reversal scored 0.26691154
and 0.59842254. The normalization reduces sensitivity to activation amplitude,
but it does not remove the need for representative layer data.

The twenty-sixth sweep added a per-block affine mapping candidate. Each ASTC
block can use its own weight `scale` and `offset`, with 8 bytes of host-side
metadata per block; the compressed image remains a standard ASTC resource. On
the small fixture, block-reversed/block-affine reached objectives 0.11169542
for 4x4 and 0.23868437 for 6x6, but required 72 and 32 metadata bytes
respectively. On the large fixture, identity/block-affine 6x6 reached
0.52582064 versus 0.55860837 for global mapping, at the cost of 968 metadata
bytes; 4x4 global mapping remained best at 0.23307591 versus 0.23746265 for
block-affine. This makes affine calibration a plausible 6x6 option, but its
metadata must be included in any bandwidth or capacity comparison.

The twenty-seventh sweep tested separate affine `scale/offset` pairs for each
of the four logical channels inside a block. It did not win the neural
objective on either fixture. The small fixture's best per-channel candidate
reached 0.11736877 (4x4) and 0.24216012 (6x6), versus 0.09723277 and 0.23868437
for the best global/shared-affine candidates, while adding 32 bytes per block
instead of 8. The large fixture likewise selected shared/global mappings. The
per-channel mode remains useful as a documented negative control, but is not a
recommended representation.

The twenty-ninth sweep added a private versioned pack-metadata contract to
`astc-vulkan-tensor-contract.h`. It records the ASTC format, logical tensor
shape, texel width, packed channel order, layout ID, mapping ID, encoder
quality, compressed bytes, and calibration bytes. The contract validates
version, shape, supported IDs, and non-zero compressed storage; it intentionally
does not define GGUF serialization or host endianness yet. The tensor-contract
CTest now covers both a valid record and an inconsistent-shape rejection.

The thirtieth sweep strengthened that metadata contract by validating the
packed RGBA channel order as exactly one occurrence of channels 0 through 3 and
rejecting non-zero bits outside the packed byte. A duplicate-channel record is
now a compile-time negative test. This keeps malformed offline artifacts from
reaching a future shader path.

The twenty-eighth sweep wired the affine mapping modes into the same candidate
search as channel order, block order, and encoder quality. Shared block-affine
stores one scale/offset pair per ASTC block; per-channel affine stores four.
The latter never won the normalized objective and multiplies metadata by four,
so future pressure-aware selection should consider only global and shared
block-affine mappings. The host smoke now prints compressed bytes and metadata
bytes for every candidate, making this tradeoff observable before any Vulkan
resource format is designed.

The twenty-first sweep added a conservative block-tail term to the candidate
objective:
`MSE + activation_MSE + 0.25 * max_block_MSE`. The coefficient is intentionally
explicit and small; it prevents a single poor block from being hidden by the
global average without allowing the tiny fixture to dominate selection. The
selected candidates were unchanged (4x4 identity/medium and 6x6
identity/thorough), while their reported objectives became 0.10849460 and
0.30935956 respectively. This is a ranking guardrail, not a calibrated model
quality threshold.

## Test sweep policy

After each implementation sweep:

1. run `git diff --check`;
2. configure/build with `LLAMA_ASTC_VULKAN_POC=ON`;
3. run the focused `astc` CTest label;
4. run broader relevant backend tests when the build permits;
5. revisit `astc-vulkan-plan.md` and update this log with completed phases,
   failed hypotheses, and new constraints;
6. do not claim GPU support or speedup until a real device benchmark exists.

## Next sweep (superseded by real-model input sweep)

1. Add a reader for real GGUF weight slices and representative activation
   traces, with explicit support limited to documented source tensor types.
2. Add FP16 and the closest existing low-bit reference to the same objective and
   report total resident bytes, including ASTC metadata.
3. Evaluate sparse residual/outlier sidecars at fixed budgets and include their
   index/value overhead in pressure-aware selection.
4. Calibrate objective weights on real layer distributions and define a
   reproducible quality gate for entering Phase 3.
5. Evaluate deeper endpoint/partition candidates only if the quality gate and
   measured storage tradeoff justify their offline search cost.
6. Keep encoder availability separate from Vulkan runtime tests and do not alter
   any ggml tensor path until weight errors are materially reduced.

## Thirty-first sweep: real GGUF input and quality comparison

The input path now reads rank-2 F16/F32 tensor slices directly from a GGUF file
using ggml's public gguf reader APIs. It validates tensor byte sizes and
dimensions, converts F16 to F32 for the existing host-side matrix contract, and
keeps the deterministic generated fixture as a CTest. A small listing mode was
added so a real model can be inspected without changing the runtime path.

The first real input is `SmolLM2-135M-Instruct-F16.gguf` (135M parameters,
F16, 271 MB as published) and `blk.0.attn_q.weight` is a 576x576 matrix. The
download was verified against the published SHA-256 before use. No model file
is checked into the repository.

Reproduction uses the published Hugging Face file and verifies SHA-256
`5157ca60744d21631818364854ac8e4452e1b8022d2ab4c8a2f9cda2344afb30` before
running the probe.

## Thirty-second sweep: ASTC/Q4 reference and residual budget

`astc-vulkan-quality-smoke` now compares the loaded F16 matrix with a Q4_0
reference and standard ASTC 4x4/6x6 roundtrips. It reports compressed bytes,
global versus shared block-affine metadata, elementwise MSE, activation-weighted
relative matvec MSE, and total bytes after a 1% sparse residual sidecar. The
probe uses deterministic activation vectors when no trace is supplied and can
consume the versioned binary activation-trace format from the input reader.

On `blk.0.attn_q.weight`, Q4_0 measured 0.00990 relative activation MSE. ASTC
4x4 measured 0.23267 (global) and 0.18155 (block-affine), while ASTC 6x6
measured 0.51503 and 0.46071 respectively. Correcting the largest 1% errors
reduced these values only to 0.20513/0.15572 and 0.42483/0.38467. The
provisional quality gate of 0.10 therefore fails for all ASTC variants. This is
useful evidence against entering Phase 3 with the generic baseline; it does
not yet rule out a neural-weight-aware encoder or a larger residual budget.

The complete ASTC CTest label was rebuilt and passed 14/14 tests, including the
new GGUF/input contract, host encoder/weight fixtures, Vulkan capability and
device checks, and 4x4/6x6 shader fetch variants. The large host fixture remains
deliberately slow (about 105 seconds) because it exhaustively searches the
current candidate set.

The storage comparison is equally important: the 576x576 F16 matrix is
663,552 bytes and Q4_0 is 186,624 bytes. ASTC 4x4 uses 82,944 compressed bytes
before calibration metadata, while 6x6 uses 36,864. Shared block-affine
metadata raises totals to 124,416 and 55,296 bytes respectively, before the
26,544-byte 1% residual sidecar. Metadata and residuals must therefore be
counted as resident bandwidth, not treated as free compression overhead.

## Thirty-third sweep: residual frontier and layer spread

The quality probe now accepts `--residual-percent`, allowing the sparse
sidecar budget to be swept without recompiling. On `blk.0.attn_q.weight`, the
best shared block-affine 4x4 result moved from 0.15572 relative activation MSE
at 1% residuals to 0.14208 at 2% and 0.11766 at 5%; it still missed the 0.10
gate. The corresponding 6x6 results were 0.38467, 0.35303, and 0.27153. The
sidecar grows linearly from 26,544 bytes at 1% to 132,712 bytes at 5% for this
576x576 matrix, so the quality gain is not free bandwidth.

The same 1% comparison was run on `blk.0.attn_k.weight`,
`blk.0.attn_v.weight`, and `blk.0.ffn_down.weight`. Q4_0 activation-relative
MSE was 0.01123, 0.01072, and 0.00844 respectively. Shared block-affine ASTC
4x4 after residual correction measured 0.20587, 0.21486, and 0.21572; ASTC
6x6 measured 0.43791, 0.45376, and 0.45442. This spread is sufficiently
consistent to keep the Phase-3 gate closed for the generic ASTC baseline.

The residual parameter remains an offline research control. It is not yet a
runtime format field, and no ASTC artifact is emitted until a multi-layer
quality/storage frontier passes review.

## Thirty-fourth sweep: nonlinear transform negative control

The quality probe now includes a reversible signed-square-root transform as a
candidate before ASTC encoding. It was intended to allocate more code points
near zero while keeping the standard LDR ASTC resource unchanged. On the real
attention-Q layer it made the result worse: 4x4 block-affine corrected
activation-relative MSE rose from 0.15572 (linear) to 0.22766 (signed-sqrt),
and 6x6 rose from 0.38467 to 0.53908. The global mappings showed the same
direction. The transform is therefore retained only as a reproducible negative
control and is not part of the proposed representation.

## Thirty-fifth sweep: low-rank residual candidate

The probe now estimates a low-rank correction to the shared block-affine ASTC
matrix using deterministic power iterations. It reports F16-sized `U` and `V`
sidecar bytes for ranks 4, 8, 16, and 32, together with corrected elementwise
and activation-weighted errors. On `blk.0.attn_q.weight`, rank 32 improved
4x4 corrected activation-relative MSE from 0.15572 to 0.12836, but increased
the total representation to 198,144 bytes versus 186,624 bytes for Q4_0. The
6x6 rank-32 point remained at 0.25487 and was not competitive.

On the larger `blk.0.ffn_down.weight`, rank 32 4x4 reached 0.20821 at 466,944
bytes, while Q4_0 was 0.00844 at 497,664 bytes. This shows that low-rank
residuals can improve the ASTC frontier, especially when the base format has a
larger storage budget, but the current generic ASTC base still misses the
quality gate by a wide margin. The algorithm is therefore evidence for a
possible hybrid design, not a runtime format decision yet.

## Thirty-sixth sweep: Geldreich/XUASTC and adjacent quantization research

Rich Geldreich's XUASTC work is directly relevant as an encoder-design
reference. XUASTC treats ASTC endpoints, partitions, and interpolation weights
as a structured latent space, then performs analysis-by-synthesis: it evaluates
candidate ASTC configurations after transform/quantization and selects the
lowest-distortion reconstruction. Its optional DCT/DPCM representation is a
CPU-side supercompression layer which reconstructs legal ASTC block weights
before ordinary ASTC decoding. It is therefore not a direct GPU-resident format
for this PoC, but its candidate-search discipline is applicable.

The key design decision is to replace image MSE inside the offline ASTC search
with an activation-aware loss. Given captured input activations `X`, score a
candidate reconstruction using `|| (W - W_hat) X^T ||_F^2`, optionally with a
tail penalty. This makes ASTC mode, partition, endpoint, channel-layout, and
calibration selection answer the inference question rather than the image
question. Block candidates can be evaluated incrementally by applying their
output delta to cached calibration outputs; a simple independent-block MSE is
not sufficient when activation columns are correlated.

The small H4 rotation tested earlier was not a faithful test of the stronger
rotation methods used by QuaRot, QuIP#, and SpinQuant. Their useful direction is
larger randomized signed Hadamard rotations and/or learned orthogonal rotations
over hidden dimensions. The next test should use fixed 64-wide or 128-wide
Walsh-Hadamard groups with signed permutations, chosen offline and applied to
both the weight columns and activation groups. A 64-wide group divides this
model's 576-wide hidden dimension and aligns naturally with sixteen RGBA texels
or a 4x4 ASTC block. Full learned rotations are a later option because they add
metadata and model-wide fusion constraints.

The low-rank work also changes the residual experiment. EoRA and the newer
preserve-then-quantize/SRR formulation make the residual activation-aware, and
reserve some rank to preserve dominant directions *before* quantizing the
remaining matrix. The next low-rank experiment should use activation covariance
from captured traces, compare rank splits `k`/`r-k`, and quantize the F16
sidecar itself before accepting its byte count.

References:

- Rich Geldreich, [XUASTC LDR Weight Grid DCT](https://github.com/BinomialLLC/basis_universal/wiki/XUASTC-LDR-Weight-Grid-DCT)
  and [JPEG for ASTC](https://github.com/BinomialLLC/basis_universal/wiki/JPEG-for-ASTC).
- Ashkboos et al., [QuaRot](https://arxiv.org/abs/2404.00456);
  Tseng et al., [QuIP#](https://arxiv.org/abs/2402.04396); Liu et al.,
  [SpinQuant](https://arxiv.org/abs/2405.16406).
- Liu et al., [EoRA](https://arxiv.org/abs/2410.21271); Cho et al.,
  [Preserve-Then-Quantize / SRR](https://arxiv.org/abs/2602.02001).

## Thirty-seventh sweep: WHT64 signed rotation

The quality probe now includes a deterministic signed WHT64 basis. It applies a
64-wide orthogonal transform to each hidden-dimension group, with the inverse
transform used for reconstruction; the 576-wide SmolLM2 matrices divide into
nine complete groups. This is closer to the larger randomized rotations used by
QuaRot and QuIP# than the earlier four-value H4 test.

The first fixed sign mask did not improve ASTC. On `blk.0.attn_q.weight`,
4x4 shared block-affine corrected activation-relative MSE was 0.22080 for
WHT64 versus 0.15572 for the unrotated baseline; 6x6 was 0.45355 versus
0.38467. This is a preliminary mask result, not a theorem about all rotations.
The likely explanation is architectural: WHT64 decorrelates hidden channels,
while ASTC's generic image encoder benefits from local spatial/channel
correlation. QuIP# gains from a rotation because it follows with a dedicated
vector/lattice quantizer, which ASTC does not provide.

The next rotation test, if retained, must search several signed masks and score
them with the same activation-aware objective. It should not be promoted based
on a single fixed mask.

## Updated next sweep

1. [x] Capture representative attention-layer input activations from the llama
   evaluation path and feed them to the quality probe. Extend this from the
   first prompt/layer to a calibrated prompt suite and the remaining projection
   sites before making a layer-wide quality claim.
2. [x] Add a direct dequantized Q4_0 matvec comparison and report per-layer
   gate results rather than one aggregate tensor. The current probe covers
   four real SmolLM2 layers using deterministic activation vectors; captured
   model traces remain the next fidelity step.
3. [x] Sweep residual budgets (0.25%, 1%, 2%, 5%) and make the
   quality/storage frontier explicit for 4x4 and 6x6.
4. Implement an activation-aware ASTC analysis-by-synthesis score, then sweep
   ASTC modes/layouts under that score before any shader integration.
5. Evaluate signed-permuted WHT64 rotation candidates; retain only choices that
   improve the quality/storage frontier on multiple real layers.
6. Replace residual-only low-rank fitting with activation-aware
   preserve-then-quantize rank allocation.
7. Keep Phase 3 Vulkan matvec work limited to a correctness scaffold until a
   candidate passes the real-layer quality gate.

## Open questions

- Which ASTC image formats are actually sampled in compute on each target GPU?
- How should rows, columns, channels, and reduction tiles map to 2D image
  coordinates for cache locality?
- Is a generic ASTC encoder numerically adequate, or is a neural-weight-aware
  encoder required from the start?
- Does 6x6 reduce actual DRAM traffic enough to offset texture-pipeline latency?
- Can a mixed 4x4/6x6 policy improve quality/performance over either global
  choice?

## Thirty-eighth sweep: real layer-input trace and an activation-aware gate

The new `astc-vulkan-trace-capture` PoC tool loads a normal GGUF through
llama.cpp, enables its existing internal layer-input extraction hook, decodes a
prompt, and writes a versioned F32 trace through the same reader/writer
contract used by the quality probe. It has no production backend changes and
does not create a public API: the layer-input hook remains an explicitly
documented internal dependency of the isolated research tool.

For the prompt `The quick brown fox uses a compact neural network.`, layer 0 of
SmolLM2 produced ten 576-wide activation vectors. Evaluated against
`blk.0.attn_q.weight`, Q4_0 measured 0.00721 relative activation MSE. The best
current generic ASTC result, 4x4 shared block-affine with a 1% elementwise
sparse sidecar, measured 0.13778; 6x6 measured 0.29117. This confirms the
earlier synthetic result's qualitative conclusion on genuine model inputs, and
keeps the Phase-3 kernel gate closed.

The trace also changes the immediate research priority. Sparse residuals are
currently selected by largest *elementwise* errors, even though the relevant
loss is output error over the captured activations. The next test will compare
that baseline against an activation-energy-weighted residual selector, then
move to full activation-aware analysis-by-synthesis only if it yields a real
quality/byte improvement. This is the smallest practical reuse of the
preserve-then-quantize/EoRA lesson without prematurely writing a bespoke ASTC
encoder.

## Thirty-ninth sweep: activation-aware sparse residual selection

The quality probe now has a second sparse-sidecar selector. Instead of taking
the largest elementwise reconstruction errors, it greedily takes the weight
whose correction gives the largest exact reduction in the calibration matvec
loss. The selector maintains the current output-error vector for each matrix
row, so each selection scores cross terms with already selected coefficients;
it is more faithful than ranking `error^2 * activation_energy` independently.
The artifact size is deliberately unchanged: index plus F32 residual value for
each selected coefficient.

With the ten-token calibration trace, the 1% sidecar changed 4x4 shared
block-affine from 0.13778 to 0.04934 relative activation MSE, and 6x6 from
0.29117 to 0.09359. This shows that a neural objective can materially change
the *usefulness* of an ASTC base without adding bytes. It does not yet make the
format viable because selection on the same trace is optimistic.

The new `--selection-trace` separates calibration from evaluation. On a second,
semantically different 25-token prompt, selected from the first ten-token
trace, 4x4 improved from 0.14688 (elementwise) to 0.11600 and 6x6 from 0.33208
to 0.25247. The gain generalizes partly but misses both the 0.10 Phase-2 gate
and Q4_0's 0.00770. Accordingly this is a retained calibration tool, not a
runtime design decision. The next objective must calibrate on a larger, more
representative prompt set and use a strict holdout before a candidate is
allowed into a packed artifact.

## Fortieth sweep: ASTC-native few-level weight direction

The next research track distinguishes an ASTC-native quantizer from the
current "encode a normal FP16 weight matrix as an image" baseline. The
provisional family name is **ASTC-Q**: a weight is assigned a small semantic
alphabet (ternary, five-level, eight-level, or sixteen-level), while ASTC
endpoints, interpolation weights, and possibly partitions provide the physical
representation and hardware reconstruction.

The byte arithmetic is deliberately explicit. For the proposed initial layout,
one scalar weight occupies one ASTC texel, so 6x6 is `128 / 36 = 3.56` physical
bits per scalar before companion data. This is a meaningful Q3/Q4-adjacent
format. It is not valid to quote 3.56 bits per scalar while storing four
independent weights in RGBA channels: that layout is nominally 0.89 bits per
scalar and can only work if the weights have substantial ASTC-exploitable
structure.

Three experiments are now recorded in the implementation plan: a legal-ASTC
representability microbenchmark, a post-training comparison against packed
ternary/few-level weights, and ASTC-aware training through a differentiable
surrogate followed by legal-ASTC projection. This is intentionally separate
from the existing Q4 path. The common runtime hypothesis is not minimum
bit-density, but whether fixed-function texture reconstruction decreases the
total cost of delivering usable weights to the shader ALU.

## Forty-first sweep: ASTC-Q few-level representability smoke

`astc-vulkan-level-smoke` now exercises legal standard ASTC encoding and CPU
decode for ternary, five-level, eight-level, and sixteen-level scalar alphabets
on both 4x4 and 6x6 blocks. Each scalar is represented by one texel (replicated
across RGBA only to satisfy the sampled image contract), so the reported rate
does not incorrectly count four independent weights per texel. The patterns
cover smooth gradients, clustered regions, deterministic random levels, and
rare outliers.

The 6x6 results are a useful boundary rather than a blanket success. Ternary
and five-level values stayed at 100% near-level recovery with MSE below
`3e-6` across the tested patterns. Random eight-level values fell to 35.07%
near-level recovery (MSE `0.00262`), and random sixteen-level values reached
61.63% (MSE `0.00233`); smooth and clustered patterns remained near-exact.
The corresponding 4x4 random cases were much stronger (99.22% and 97.66%
near-level recovery). This is direct evidence that 6x6's 3.56 physical
bits/texel are useful for structured few-level weights, but insufficient for
arbitrary high-entropy Q3/Q4-like symbols.

The smoke is intentionally a representability test, not a model-quality gate.
The next step is to feed real ternary/few-level weight matrices through the
same path, then compare against a packed ternary reference and include the
shader-side unpack/dequantization cost. ASTC-aware training remains justified
only if real weights can be made more like the successful smooth/clustered
cases without unacceptable model loss.

The focused ASTC regression run after this addition passed 14/14 tests (the
intentionally long large encoder fixture was excluded from that quick pass;
it had already completed successfully in the preceding full run). This covers
the new level smoke together with the existing contracts, encoder, input, and
Vulkan shader/device checks.

## Forty-second sweep: post-training ASTC-Q on real weights

`astc-vulkan-native-quant-smoke` applies global ternary, five-, eight-, and
sixteen-level quantization to the real 576x576 `blk.0.attn_q.weight` matrix,
then roundtrips the quantized values through legal ASTC 4x4 and 6x6 blocks.
It reports an ideal packed few-level byte count, actual ASTC resident bytes,
element MSE, and activation-relative matvec MSE using the captured holdout
trace.

The result separates two effects cleanly. ASTC adds almost no error beyond the
already quantized values for ternary, five-level, and sixteen-level cases; for
the eight-level random-like matrix it adds a modest additional distortion in
6x6. The dominant problem is the post-training few-level projection itself:

| Levels | Ideal packed bytes | 6x6 ASTC bytes | 6x6 activation-relative MSE |
|---:|---:|---:|---:|
| 3 | 65,732 | 147,456 | 0.9552 |
| 5 | 96,296 | 147,456 | 1.3333 |
| 8 | 124,416 | 147,456 | 2.0843 |
| 16 | 165,888 | 147,456 | 0.2781 |

For reference, the same trace gave Q4_0 `0.00770`. The 6x6 ASTC rate is
3.5556 bits/weight for this one-scalar-per-texel layout, but it is still 2.24x
the ideal ternary byte count. This rules out claiming that ASTC storage alone
turns an ordinary pretrained matrix into a useful BitNet model. It strengthens
the ASTC-aware-training hypothesis: the model must learn both the few-level
alphabet and the spatial/block structure that 6x6 can reconstruct.

## Forty-third sweep: matched tiny-model baseline and artifact audit

The `shibatch/tinybpe1m` F16 and Q4_0 artifacts are a compact, reproducible
model-level smoke fixture. The files are kept outside the repository under
`/tmp/llama-astc-models/tinybpe1m`; the repository only commits the short
evaluation corpus and records the source repository, not a model binary.

The downloaded artifact checksums are:

| File | SHA-256 |
|---|---|
| `tinybpe1m.F16.gguf` | `17d21f397845a9ce92f18a55c7f0698a4016ef475041b996fd5a39b3f30d8d89` |
| `tinybpe1m.TQ1_0.gguf` | `944bfa1694ae5683b5e5e39f496e71f3ea4c0240d5d1a7cef53bf4fda699f2aa` |
| `tinybpe1m.TQ2_0.gguf` | `18c653e44063a5589d748658319b3e8b03a5c8e35dc1ff9f2c6091b09ded2d44` |
| `tinybpe1m.Q4_0.gguf` | `8f37ac9de3a907abf512ce20ac4545de5091a2509e059289fb1eefde14342d79` |

The source is [shibatch/tinybpe1m](https://huggingface.co/shibatch/tinybpe1m),
which publishes the matched quantization family and documents the model as a
small llama.cpp validation model.

With a fixed prompt (`Tom and Jerry are`), seed 42, temperature 0, and 16
predicted tokens, the F16 and Q4 files produced:

| Variant | Deterministic continuation |
|---|---|
| F16 | `ly. Every day, Jerry would go` |
| Q4_0 | `ed. Every day, Jerry would go` |

On the committed 256-token-context evaluation corpus, one perplexity chunk
gave F16 `44.1675` and Q4_0 `53.3438`. These numbers have a large uncertainty
because the tiny model and one chunk are only a smoke baseline; they are not a
quality claim.

The two downloaded artifacts whose filenames claim `TQ1_0` and `TQ2_0` must
not be used as ternary controls. A GGUF tensor audit found 28 `q4_0` tensors,
one `q5_0` tensor, and one `q8_0` tensor in each file. Their matching outputs
and PPL are therefore expected but do **not** measure TQ behavior. The files
are retained only as provenance evidence for this validation check.

Running `astc-vulkan-quality-smoke` on the same model's F16
`blk.0.attn_q.weight` (128x128) measured Q4_0 activation-relative MSE
`0.00709`. The existing ASTC 4x4 and 6x6 block-affine paths with a 1% sparse
sidecar reached `0.25143` and `0.45726` respectively before activation-aware
selection (`0.13744` and `0.26716` after selection). The native ASTC-Q smoke
also reported 6x6 resident rate `3.78125` bits/weight after edge padding, with
level-16 element MSE `0.000464` versus level-3 `0.005926`.

The immediate conclusion is that F16 is required as the same-source teacher
for a fair ASTC comparison, while Q4 is the initial model-level control. A
complete model-level ASTC result still requires a loader/runtime path that can
fetch ASTC-resident weights during inference; the current measurements are the
weight and matvec gates that must pass before changing the GGUF format.

## Forty-fourth sweep: true TQ1 versus TQ2 behavior and alignment limit

The PoC GGUF reader now uses ggml's registered host dequantizer for any
rank-two tensor type that provides one. Its contract smoke creates and
roundtrips actual TQ1_0 and TQ2_0 tensors, so the reader can no longer mistake
a filename or metadata field for a tensor's physical type.

Running `llama-quantize --pure` on tinybpe1m proved that this model is not a
valid TQ fixture: its 128- and 352-wide rows are not divisible by TQ's 256
element block, so llama.cpp correctly falls back to Q4_0. SmolLM2 has a valid
case: each `ffn_down` matrix is 576x1536, and 1536 is divisible by 256. Local
TQ1_0 and TQ2_0 files generated from the same SmolLM2 F16 source contain true
TQ tensors for that projection.

On `blk.0.ffn_down.weight`, TQ1_0 and TQ2_0 decoded to the same values under
the reference post-training quantizer, producing element MSE `0.02509` and
relative matvec MSE `0.70405` on the deterministic activation fixture. This
is a storage-packing comparison, not a trained ternary-model result.

The storage and implementation tradeoff is:

| Format | Representation | Nominal rate | Current Vulkan source status |
|---|---|---:|---|
| TQ1_0 | base-3 packed trits (five trits/byte plus a small high part) | 1.6875 bpw | no TQ1 shader/pipeline in this checkout |
| TQ2_0 | two bits/weight, ternary codes | 2.0625 bpw | dedicated Vulkan dequant/matvec shaders present |

TQ1 therefore remains the density leader, but its base-3 unpacking is a poor
first target for a texture experiment and would require a new Vulkan shader
path. TQ2 is the more practical GPU control because the backend already knows
how to execute it and its two-bit code layout is closer to a shader-friendly
ASTC comparison. Neither result changes the ASTC quality gate: ASTC must still
beat or approach the TQ/Q4 matvec frontier on the same F16 source before it is
considered a runtime replacement.

## Forty-fifth sweep: ASTC rate ladder and retained-format tests

The test structure now preserves every investigated format as a named,
reproducible probe. Fast contract and encoder tests run under the `astc` CTest
label. The model-backed `format-smoke` and `native-quant-smoke` tools are
explicit opt-in because they require a local model path and are intentionally
not silently folded into routine CTest. `native-quant-smoke` accepts explicit
`--levels` and `--blocks` lists, so each rate/semantic hypothesis is recorded
by its invocation rather than replaced by the next experiment.

The scalar-per-texel ASTC ladder now includes 4x4, 5x5, and 6x6. The standard
ASTC storage rates before edge padding and companion metadata are 8.00, 5.12,
and 3.56 bits per scalar respectively. Intel UHD Graphics 620 with Mesa
reported sampled-image support for all three formats. The Vulkan capability
probe, resource smoke, shader-device smoke, and nonlocal shader-device smoke
all retain separate 4x4/5x5/6x6 names.

On the same 576x1536 SmolLM2 FFn-down matrix, with 1024 source levels to make
the post-quantization error negligible, the current offline encoder frontier
was:

| Format | Effective bpw | Relative matvec MSE |
|---|---:|---:|
| Q3_K | 3.4375 | 0.02432 |
| Q4_0 | 4.5000 | 0.00844 |
| ASTC 6x6 | 3.5556 | 0.06502 |
| ASTC 5x5 | 5.1690 | 0.01639 |
| ASTC 4x4 | 8.0000 | 0.00163 |

These measurements use the deterministic activation fixture because the
existing trace capture records layer input, whereas FFn-down consumes an
intermediate activation. They establish a rate-distortion frontier, not model
quality or GPU throughput. The important result is nevertheless clear: 6x6
has a real capacity/bandwidth experiment at Q3-adjacent density but needs a
better ASTC-aware representation; 5x5 is the intermediate research point;
and 4x4 is a deliberately less dense, high-fidelity texture-resident control.

## Forty-sixth sweep: channel semantics, RG16, and residual pairs

`astc-vulkan-channel-semantic-smoke` retains four explicitly named channel
meanings on a deterministic 32x256 weight matrix: four independent weights per
RGBA texel, two coarse-only weights per texel, two coarse-plus-residual weights
per texel, and a two-value RG16 high/low-byte control. It runs every semantic
against 4x4, 5x5, and 6x6 and reports physical bytes, effective bits per
logical weight, element MSE, and activation-relative matvec MSE.

The first result rejects an attractive but unsafe interpretation of the
channels. ASTC does not preserve the high and low byte of a synthetic 16-bit
word reliably enough for the bytes to be recombined after decode. At 4x4,
RG16 pairs measured `0.04217` element MSE and `0.02921` relative activation
MSE. This is much worse than using the channels as approximate numerical
values, so RG16 remains a negative control rather than a candidate format.

The naive residual construction also lost to coarse-only storage:

| Format | Pair rate (bpw) | Coarse-only relative MSE | Coarse + residual relative MSE |
|---|---:|---:|---:|
| 4x4 | 4.0000 | 0.00379 | 0.01908 |
| 5x5 | 2.8438 | 0.00834 | 0.03350 |
| 6x6 | 2.0625 | 0.00688 | 0.03207 |

The coarse quantization is spatially structured enough for ASTC to approximate
it, while its ordinary per-value residual is high-frequency and ASTC destroys
it. This does not invalidate a two-component representation; it narrows the
next hypothesis. A residual must be structured before it enters ASTC (for
example block-local, low-rank, sparse sidecar, or ASTC-aware-trained), rather
than treating decoded RG bytes as a lossless 16-bit container.

The same tool now retains a channel-layout optimizer. The default CTest sweep
uses the three semantically distinct pairings `RG+BA`, `RB+GA`, and `RA+GB`.
`--all-pair-permutations` expands this to all 24 ordered assignments of
`coarse(value0), residual(value0), coarse(value1), residual(value1)`. The
tool also accepts `--model path --tensor name`, loading a real F16 GGUF matrix
through the same PoC reader used by the other quality tools.

The exhaustive small-fixture search found that pairing affects the objective,
but does not change the residual conclusion:

| Format | Best coarse-only pair | Relative MSE | Best coarse+residual pair | Relative MSE |
|---|---|---:|---|---:|
| 4x4 | `BR+GA` | 0.00373 | `RB+GA` | 0.01807 |
| 5x5 | `RG+BA` | 0.00834 | `RB+GA` | 0.03310 |
| 6x6 | `AR+GB` | 0.00643 | `RA+GB` | 0.03101 |

On the real SmolLM2 F16 `blk.0.attn_q.weight` matrix, the 6x6 pair layout is
only 1.7778 bpw because it stores two logical weights per texel. The best of
the three canonical coarse-only pairings reached `0.83658` relative matvec
MSE; the best naive residual pairing reached `4.36999`. This is deliberately
retained as a capacity stress test, not promoted as an ASTC-Q candidate.

## Forty-seventh sweep: additive luminance-plus-alpha latent contract

The next research direction is **ASTC-Latent**: one logical weight per texel
is reconstructed from two decoded numerical components,

\[
\hat w = s_L L' + s_A A' + b.
\]

The first source layout is `R=G=B=L, A=A`. This is intentionally unlike the
previous residual-pair test: it does not try to place two unrelated weights in
one texel. It retains 8.00, 5.12, and 3.56 physical bits per logical weight at
4x4, 5x5, and 6x6 before padding and three decoder scalars.

The first executable, `astc-vulkan-latent-smoke`, contains two controls. The
`scalar-rgba` control replicates a normalized scalar weight in all four
channels. The `luminance-alpha-additive` control stores a 16-level coarse
value as RGB and the bounded residual as alpha. Both make a real `astcenc`
roundtrip and fit the three affine reconstruction constants from the decoded
values. This is a post-training contract test, not yet codec-aware training.

The smoke also inspects every final 128-bit block through
`astcenc_get_block_info`. It reports `dual-plane=N/M` and the subset whose
second interpolation component is alpha. This is material: a L+A source image
does not guarantee dual-plane use. ASTC dual-plane gives one chosen component
a second interpolation weight grid; it is not two independent arbitrary
channels, and Vulkan sampling gives the application no runtime block-mode
control.

This sweep makes three design decisions before collecting quality numbers:

- retain L+A as the primary codec-aware-training hypothesis;
- keep the initial implementation encoder-observed rather than claiming a
  forced dual-plane mode; and
- call the relation to AQLM *inspiration*, not equivalence. AQLM's learned
  additive codebooks are a stronger later parameterization than the two raw
  per-texel fields tested here.

The next measurement must compare the scalar and L+A controls on the
deterministic fixture and a real F16 layer, record actual dual-plane selection,
then decide whether a constrained encoder search is justified. The loss gate
will be held-out activation error before model-loss fine-tuning. The 8x6 and
8x8 ASTC rungs remain deferred until device capability is verified.

The first real-layer measurement used SmolLM2 F16
`blk.0.attn_q.weight` (576 x 576) and the deterministic activation suite. It
also corrected an initial test-harness issue: the scalar RGBA control has
collinear RGB and alpha features, so it must fit a two-parameter
`s_L L' + b` decoder rather than a singular three-parameter regression.

| Format | Mode | Effective bpw | Relative matvec MSE | Dual-plane blocks | Alpha-plane blocks |
|---|---|---:|---:|---:|---:|
| 4x4 | scalar RGBA | 8.0000 | 0.00206 | 0 / 20736 | 0 |
| 4x4 | coarse L + residual A | 8.0000 | 0.14506 | 3250 / 20736 | 3250 |
| 5x5 | scalar RGBA | 5.1914 | 0.01992 | 0 / 13456 | 0 |
| 5x5 | coarse L + residual A | 5.1914 | 0.69131 | 103 / 13456 | 103 |
| 6x6 | scalar RGBA | 3.5556 | 0.08391 | 0 / 9216 | 0 |
| 6x6 | coarse L + residual A | 3.5556 | 0.91305 | 0 / 9216 | 0 |

The precise finding is not that two latents cannot work. It is that splitting
an ordinary post-training weight into a 16-level coarse field and an
independent high-frequency residual field makes the second component poorly
compressible. At 4x4 the encoder does recognize the deliberately
luminance-plus-alpha-like signal and selects alpha dual-plane in 15.7% of
blocks, but this is still far below the scalar control. At 6x6 it selects no
dual-plane blocks and loses almost all useful activation fidelity. This is
evidence for the proposed next stage — jointly structured/codec-aware latents
and model-aware loss — rather than support for adding a residual to an
unmodified model.

## Forty-eighth sweep: structured second latent

The latent smoke now includes a third control: `luminance-alpha-block-residual`.
It quantizes a weight to 16 coarse levels, replaces the residual inside each
ASTC footprint with that footprint's mean residual, and stores the result as
`RGB=L, A=A`. This intentionally discards high-frequency residual information
to test whether a spatially smooth second latent is easier for ASTC to retain.

On the deterministic 32 x 256 fixture the activation-relative errors were:

| Format | Scalar RGBA | Free residual L+A | Block-mean residual L+A |
|---|---:|---:|---:|
| 4x4 | 0.0000615 | 0.0015541 | 0.0006792 |
| 5x5 | 0.0003547 | 0.0076515 | 0.0009650 |
| 6x6 | 0.0012688 | 0.0097293 | 0.0014733 |

The block residual is a meaningful improvement over the free residual, and it
also causes the encoder to avoid dual-plane on this smooth signal. It is still
not competitive with the scalar control. On the real SmolLM2
`blk.0.attn_q.weight` layer the corresponding errors were `0.23891`, `0.25656`,
and `0.26793` for 4x4, 5x5, and 6x6. Thus a hand-constructed smooth residual is
not a candidate representation either.

This narrows the next experiment rather than ending the latent track. The
second field must be learned or jointly optimized so that it is both useful to
the model and spatially compatible with ASTC. The next implementation will
therefore prototype an outer codec-aware projection loop, with exact ASTC
roundtrip and held-out activation loss as the oracle. A differentiable
surrogate or PV-Tuning-style update belongs around that loop, not inside the
Vulkan runtime.

## Forty-ninth sweep: coarse-alphabet projection search

The latent smoke now accepts `--search-levels` and evaluates 3, 5, 8, 16, and
32 coarse levels. Each candidate is projected through the real `astcenc`
roundtrip and calibrated with the affine `s_L L' + s_A A' + b` decoder. This is
the first small codec-aware search; it is deliberately not called training,
because it searches one hand-designed latent family and uses the deterministic
activation suite.

On the deterministic fixture, the best projection level varied by footprint:

| Format | Best searched levels | Activation-relative MSE |
|---|---:|---:|
| 4x4 | 5 | 0.0005892 |
| 5x5 | 5 | 0.0050519 |
| 6x6 | 16 | 0.0097293 |

The search changes the residual trade-off but does not beat the scalar control
at the same footprint. This is still useful: it proves that the offline
projection layer can expose real ASTC mode/quality feedback, and it gives the
future optimizer a concrete oracle. The next step is to replace the fixed
coarse-plus-residual construction with learned or calibration-optimized latent
fields, evaluated on held-out activation vectors after every exact projection.

The latent smoke now accepts `--trace path`, using the same versioned trace
reader as the other PoC quality tools. On the captured SmolLM2 layer-0 holdout
trace, the scalar control measured relative matvec errors of `0.0014712`,
`0.0143491`, and `0.0652828` for 4x4, 5x5, and 6x6. The free L+A residual was
`0.113342`, `0.629460`, and `0.914666`; the block residual was `0.218061`,
`0.222477`, and `0.229024`. The ordering is consistent with the synthetic
calibration suite, so the negative result is not caused by evaluating only the
vectors used to construct the candidates. The trace path is now the required
oracle for future projection/latent optimization.

## Fiftieth sweep: row/column additive control

To test the spatial-structure side of the AQLM analogy, the smoke now includes
`row-column-additive`. It stores a normalized row mean in RGB and a normalized
column mean in alpha, then fits the same affine decoder after ASTC sampling.
This is a deliberately simple additive factorization, not a learned codebook.

On the deterministic fixture its activation-relative errors were `0.90524`,
`0.90932`, and `0.90084` for 4x4, 5x5, and 6x6. The real SmolLM2 attention
layer was similarly poor (about `0.90` on all three footprints). ASTC selects
alpha dual-plane frequently for this layout, but the factorization has thrown
away too much weight information. It is retained as a negative control: an
AQLM-like additive representation must learn vector/group codebooks jointly
with model loss; simple row/column statistics are not sufficient.

## Fifty-first sweep: calibration versus holdout projection selection

The projection search now accepts separate `--calibration-trace` and `--trace`
files. It selects the coarse alphabet using the calibration trace while
printing the holdout activation error for the same exact ASTC projection. This
prevents the search from being rewarded for overfitting the vectors used to
choose a candidate.

On the SmolLM2 `blk.0.attn_q.weight` layer, using
`smollm2-layer0.trace` for calibration and `smollm2-layer0-holdout.trace` for
evaluation, the best searched alphabet was 3 levels for all three footprints:

| Format | Selected levels | Calibration relative MSE | Holdout relative MSE |
|---|---:|---:|---:|
| 4x4 | 3 | 0.003241 | 0.003588 |
| 5x5 | 3 | 0.018776 | 0.021264 |
| 6x6 | 3 | 0.072968 | 0.083220 |

The holdout ordering remains below the scalar controls (`0.001471`, `0.014349`,
and `0.065283`). The selected alphabet therefore generalizes, but the current
hand-designed L+A family still does not win. The calibration/holdout split is
now the acceptance oracle for the next learned-latent projection loop.


## Fifty-second sweep: encoder boundary decision

The experiments now make the encoder requirement clearer. Standard `astcenc`
is sufficient to prove the Vulkan resource, shader, bitrate, and scalar
quality baselines, but its public API does not provide a stable way to request
the exact endpoint mode, weight grid, dual-plane component, partition, or BISE
alphabet that an ASTC-Latent representation may need.

The selected architecture is therefore an **offline constrained astcenc
fork/extension**, not a new runtime decoder:

- keep standard `astcenc` as the baseline and fallback;
- reuse its legal ASTC bit packing, endpoint quantization, BISE handling, and
  reference decode;
- add neural-objective candidate restrictions/preferences in the offline
  search; and
- validate emitted blocks with block-info inspection, CPU roundtrip, and the
  Vulkan sampled-image smoke.

If this cannot express the required latent forms, a small constrained packer
may be written from the ASTC specification, but it must still emit standard
128-bit blocks. A non-standard format would require a shader decoder and would
no longer test the fixed-function texture-bandwidth hypothesis. No vendor
assembler is needed for either path.

## Fifty-third sweep: three encoder-track controls

`astc-vulkan-encoder-track-smoke` now makes the three encoder tracks
executable on the same L+A signal:

1. **Standard** uses the unmodified `astcenc` search.
2. **Constrained-search** uses only public `astcenc_config` knobs: independent
   channel error weights, a one-partition limit, and a relaxed dual-plane
   early-out threshold.
3. **Minimal-block-policy** replaces each source footprint by its per-block
   mean before invoking standard `astcenc`. This is a bounded prepacker control,
   not a custom ASTC bitstream.

All three produce ordinary 128-bit ASTC blocks. On a two-by-two-block fixture:

| Format | Standard MSE | Constrained MSE | Minimal policy MSE | Standard dual-plane | Constrained dual-plane |
|---|---:|---:|---:|---:|---:|
| 4x4 | 0.00000976 | 0.00001077 | 0.003403 | 4/4 | 4/4 |
| 5x5 | 0.00001791 | 0.00002268 | 0.004821 | 4/4 | 4/4 |
| 6x6 | 0.00002871 | 0.00004484 | 0.005861 | 4/4 | 3/4 |

The public search knobs therefore do not yet justify a source fork: they
slightly worsen the generic signal while leaving dual-plane selection mostly
unchanged. Track 3 also confirms that a simple block-constant policy discards
too much latent information. The next fork experiment must operate at the
candidate search/objective layer — preserving ASTC endpoint and weight detail —
not by collapsing the input before encoding.

## Fifty-fourth sweep: neural-aware late candidate ranking (Track 4)

Track 4 is now implemented and tested in the adjacent, isolated
`astc-encoder-neural-rank` fork at commit `c4b0acf`. It changes neither Vulkan
nor llama.cpp production code. The fork preserves `astcenc` candidate
generation, endpoint/weight search, BISE packing, and emitted 128-bit ASTC
blocks. It only changes the final symbolic candidate error calculation when
the opt-in `ASTCENC_FLG_MAP_NEURAL_LA` flag is selected.

For this experimental mapping, replicated RGB represents latent `L` and alpha
represents latent `A`. With explicit scales, the final ranking error is:

```
W       = s_L * mean(R, G, B) + s_A * A
error   = (W_source - W_decoded)^2
```

The flag is mutually exclusive with ASTC normal-map and RGBM metrics. It is
off by default and is built only with an opt-in research CMake target. The
fast one-plane path falls back to the generic one-plane scorer under this
metric, so the result remains correct while preserving the ordinary optimized
path for all non-experimental encodes.

The CTest `neural-rank-smoke` evaluates eight deterministic 6x6 L+A fixtures.
Each fixture contains a varying neural signal plus an anti-correlated latent
carrier; therefore channel error and reconstructed-weight error need not agree.
All eight fixtures chose different legal ASTC block streams under the neural
metric. Weight MSE fell by 48--66% (about 56% on average), while RGBA MSE rose,
as expected, because the new metric permits L/A errors to cancel after the
cheap `L+A` reconstruction. The neural selection used dual-plane blocks for
7/9, 7/9, 6/9, 5/9, 5/9, 7/9, 7/9, and 7/9 blocks across the fixtures, versus
7/9, 6/9, 5/9, 4/9, 4/9, 3/9, 2/9, and 4/9 for standard RGBA ranking.

This is the intended proof of mechanism: standard-compatible ASTC candidate
blocks can be worse under image reconstruction yet materially better after a
neural latent decoder. It is not an LLM-quality or throughput result. This
first stage only replaces the *late* score after astcenc's ordinary shortlist
has been formed; its earlier pruning remains image-oriented. The next stage is
therefore to widen or preserve that shortlist and score candidates against
captured activations. A diagonal activation-energy score is a useful first
approximation, but final selection must use exact incremental activation loss
because activation columns can be correlated.

## Fifty-fifth sweep: ASTC error-shaping objective and evaluation cohorts

Track 4 is now interpreted as the first **ASTC error-shaping** experiment. For
a layer `Y = W X^T`, ASTC reconstruction `W_hat = W + E`, and calibration
activation covariance `H = E[X^T X]`, the relevant objective is:

```
Delta Y = (W - W_hat) X^T
L_act   = ||Delta Y||_F^2 = tr((W - W_hat) H (W - W_hat)^T)
```

ASTC is therefore allowed to make large local texel errors when they either
cancel in the latent decoder or lie in directions that the layer inputs rarely
use. This is deliberately different from maximizing image fidelity. A
candidate report must retain ASTC/RGBA MSE, decoded-weight MSE, calibration and
holdout activation loss, endpoint mode, partition count, weight grid, and
dual-plane component, so the source of any gain remains inspectable.

Exact activation loss is globally coupled across blocks in a matrix row: the
cross terms in `H` mean that independently minimizing one block's weighted MSE
does not necessarily minimize the whole-layer loss. The initial diagonal
activation-energy approximation is useful for cheap pruning only. Final
selection must maintain cached layer-output error and evaluate each candidate
incrementally against a calibration trace, then report the untouched holdout
trace.

The new opt-in astcenc metric does **not** invalidate previous standard-ASTC,
scalar, Q4_0, or native TQ decode measurements; those calls retain ordinary
astcenc ranking. It does require a new cohort for every L+A/latent result,
because those results previously used image-oriented candidate selection. The
old and new cohorts must not be compared as though they used the same encoder
objective.

TQ coverage also needs a deliberate follow-up. The input contract proves that
the local SmolLM2 FFn-down fixture contains genuine 256-aligned TQ1_0 and
TQ2_0 data, but their current equal decoded values and `0.70405` relative
matvec MSE come from one post-training quantization and a deterministic input
fixture. That is not a sufficient TQ baseline. The future comparison will run
true TQ1_0, TQ2_0, Q4_0, scalar ASTC, ASTC-Q, and Track-4 L+A candidates over
the same calibration/holdout traces on multiple 256-aligned projections. It
will report quality, resident bytes, and shader availability separately:
TQ2 has an existing Vulkan matvec route in this checkout, whereas TQ1 does
not, so their runtime comparison cannot be assumed symmetric.

## Fifty-sixth sweep: exact synthetic activation-loss contract

The side-fork CTest now additionally treats each 18x18 L+A fixture as a linear
weight matrix and evaluates the exact output loss `||(W - W_hat) X^T||_F^2` on
five deterministic, correlated activation samples. This is intentionally not
a diagonal weighted-MSE proxy. The Track-4 encoder still chooses blocks using
decoded-weight MSE in this test; the activation loss is measured afterward to
test whether semantic error cancellation transfers to the layer output.

The neural-ranked stream reduced exact activation MSE in seven of eight
fixtures, while retaining the earlier eight-of-eight decoded-weight MSE gains.
The remaining fixture is an explicit reminder that weight-MSE selection is not
the final objective. The CTest requires a majority activation-loss improvement,
not universal improvement. The next integration must select candidate blocks
against calibration-trace output error and confirm the selected result on a
holdout trace.

## Fifty-seventh sweep: fixed decoder and real Track-4 holdout result

The latent PoC previously fit `s_L`, `s_A`, and `b` after ASTC decode. That is
an acceptable optimistic analysis baseline, but it is invalid for encoder
ranking: the encoder cannot score against a decoder whose coefficients are
learned only after its candidate has been chosen. Each latent constructor now
provides a fixed affine decoder before encoding. For the additive construction:

```
L = (coarse - minimum) / range
A = 0.5 + residual / (2 * residual_radius)
W = range * L + 2 * residual_radius * A + minimum - residual_radius
```

The same `range` and `2 * residual_radius` coefficients are passed to the
Track-4 astcenc metric. The side fork is linked only by an explicit, separate
CMake build with its static archive and include path; the ordinary llama build
continues to link the packaged astcenc and cannot enable `--neural-rank`.
`astc-vulkan-latent-smoke` remains the standard CTest, while the isolated build
adds opt-in `astc-vulkan-latent-neural-rank-smoke`; both passed. The latter is
kept separate because it deliberately exercises an external research fork and
takes about 24 seconds even on the small synthetic fixture.

On SmolLM2 F16 `blk.0.attn_q.weight` (576x576), the isolated build encoded
both ordinary and neural-ranked L+A latents and evaluated the existing holdout
trace. This run did **not** select on the supplied calibration trace yet; it
is a real-weight/holdout measurement of the fixed weight-aware candidate score.

| Footprint | Standard L+A holdout relative MSE | Track-4 L+A holdout relative MSE | Scalar ASTC holdout relative MSE |
|---|---:|---:|---:|
| 4x4 | 0.11310736 | 0.00877819 | 0.00149140 |
| 5x5 | 0.62712064 | 0.08579853 | 0.01438145 |
| 6x6 | 0.94989797 | 0.35078510 | 0.06550879 |

Track 4 lowers L+A holdout error by approximately 92%, 86%, and 63% for 4x4,
5x5, and 6x6 respectively. It also changes dual-plane selection substantially:
4x4 rises from 3,250 to 12,206 blocks, 5x5 from 103 to 10,457, and 6x6 from
zero to 8,040. This is strong evidence that ordinary image ranking was choosing
the wrong ASTC block family for this latent decoder.

However, scalar ASTC remains better at every footprint. The result proves the
candidate-ranking mechanism on real weights, not that L+A is ready to replace
the scalar baseline. The next required step is exact calibration-trace ranking
with cached output-error cross terms and an untouched holdout. Only then should
we revisit neural texture layouts, permutations, or TQ controls.

## Fifty-eighth sweep: scalar Track-4 weight-aware control

The isolated run was repeated with scalar RGBA storage also passed through the
Track-4 hook. Scalar uses the fixed runtime-valid mapping
`W = range * mean(R, G, B) + minimum`, with `s_A = 0`. Because the source
replicates the same scalar in all four channels, ordinary RGBA MSE is already
proportional to the relevant scalar reconstruction error. The expected control
outcome is therefore no material change.

That is exactly what the SmolLM2 `blk.0.attn_q.weight` holdout run produced:

| Footprint | Standard scalar relative MSE | Track-4 scalar relative MSE |
|---|---:|---:|
| 4x4 | 0.0014914023 | 0.0014914023 |
| 5x5 | 0.014381453 | 0.014382067 |
| 6x6 | 0.065508788 | 0.065508788 |

No scalar block chose a dual plane in either encoding. This validates the
interpretation of the preceding L+A improvement: it is not a generic effect of
turning on a new encoder flag, but comes from scoring the tradeoff between two
latent channels according to their fixed neural reconstruction. It also sets a
clear boundary: weight-aware late ranking alone cannot improve scalar storage.
The next scalar experiment must score candidate assignments against exact
calibration layer output, using an offline shortlist and cached residual so the
block cross terms are retained.

## Fifty-ninth sweep: exact coordinate descent over legal ASTC blocks

The side fork now contains a small exact coordinate-descent contract. For each
6x6 block it retains two independently encoded, legal candidates: the normal
RGBA-ranked block and the fixed L+A neural-ranked block. It starts from the
normal stream, computes `R = Y - Y_hat`, and for each candidate replacement
uses:

```
Delta Y = (W_hat_candidate - W_hat_current) X_block^T
score   = ||R - Delta Y||_F^2
```

After accepting a replacement it updates `R` immediately. It runs a forward
block sweep followed by a reverse sweep, so the score includes cross terms with
previously selected blocks rather than approximating them with independent
weight errors. The test also concatenates the selected 16-byte blocks into a
new ASTC stream, CPU-decodes it, and requires an exact match to the mixed-block
matrix used for scoring.

On all eight deterministic L+A fixtures, the selected mixed stream has lower
activation MSE than the initial normal stream. It also beats the best uniform
whole-image choice on every fixture in this small candidate pool. In particular,
fixture 2 had a worse uniform neural-ranked stream than standard
(`0.00240484` versus `0.00224996`), yet coordinate selection reached
`0.00090233`. Reverse sweeps further improved fixtures 0, 4, and 7. This is a
proof that candidate blocks can cooperate through output-error cancellation;
it is not yet a real-model result or an exhaustive astcenc internal shortlist.

The next adapter will apply the same algorithm to real calibration and holdout
traces. Its first candidate pool should remain bounded and non-invasive: legal
full-matrix encodes from several astcenc quality/mapping settings, split into
their independently decodable blocks. Only a positive real-layer result
justifies modifying astcenc further to retain deeper internal candidates.

## Sixtieth sweep: real-trace coordinate-selection adapter

`astc-vulkan-latent-smoke` now implements that adapter without changing the
production Vulkan backend or ASTC bitstream. With `--neural-rank
--coordinate-select`, it creates two complete standard-compatible L+A streams:
ordinary astcenc ranking and the Track-4 fixed-decoder ranking. It selects only
whole 16-byte ASTC blocks, using the calibration trace and the exact update

```
||R - Delta Y||^2 = ||R||^2 - 2 R . Delta Y + ||Delta Y||^2.
```

The held-out trace is passed only after selection and is printed separately.
A forward and reverse sweep are used. The selected bytes are decoded again and
must reproduce the exact matrix used by the selector; this guards against
accidentally treating ASTC candidates as an abstract, non-serializable format.

The focused synthetic contract passes. For the 576x576 SmolLM2 projection, the
interactive runner terminates an end-to-end encode before it can return a
result, even with a small activation subset. This is an execution-window limit,
not a negative model result. The tool therefore also exposes `--coordinate-only`,
`--footprint`, `--preset`, and `--max-samples` so the real experiment can be
run in a batch environment without encoding unrelated controls. `thorough`
remains the default; lower presets are only adapter/debug probes and must be
reported separately. The next run must use a batchable runner, report both
trace sample counts, and repeat 4x4, 5x5, and 6x6 before interpreting quality.

## Sixty-first sweep: bounded real-layer selection result

To exercise the full adapter within the available execution window, the first
real run used the leading 32x64 submatrix of SmolLM2 F16
`blk.0.attn_q.weight`. This is a **bounded adapter probe**, not a layer result:
it preserves actual model weights and matching trace columns, but must not be
compared directly with full 576x576 measurements. Both traces were loaded
before cropping, then cropped identically to the first 64 input columns.

The supplied files contain 10 calibration and 25 independent holdout samples.
All encodes used the default `thorough` preset and the fixed L+A decoder.

| Footprint | Standard L+A holdout | Uniform neural-rank holdout | Coordinate-selected holdout |
|---|---:|---:|---:|
| 4x4 | 0.03396980 | 0.02782733 | 0.02974956 |
| 5x5 | 0.20582444 | 0.15407998 | 0.16202227 |
| 6x6 | 0.52479088 | 0.33293869 | 0.34494348 |

Coordinate selection improves every footprint over ordinary image-ranked L+A:
approximately 12%, 21%, and 34% for 4x4, 5x5, and 6x6. It does **not** beat
the uniform neural-ranked stream on this independent holdout. That is a useful
negative result: with a two-stream pool, block-level cancellation can improve
the calibration objective but can still overfit the limited calibration trace.
The required next experiment is therefore a broader legal candidate pool
(different quality presets and/or fixed latent mappings), selected solely on a
larger calibration set and evaluated on an untouched holdout. It is not yet
justified to change ASTCENC's internal candidate generation or claim a model
quality win over uniform neural ranking.

## Sixty-second sweep: legal three-stream pool control

The coordinate selector was generalized from two hard-coded streams to a pool
of complete legal ASTC streams. The first diversity control adds a third L+A
candidate encoded with the neural metric at `fast` quality, alongside ordinary
ranking and neural `thorough` ranking. This is deliberately not a new format,
decoder, or ASTCENC candidate generator.

On the same bounded SmolLM2 submatrix and all available 10/25 calibration/
holdout samples, the larger pool lowers calibration error but worsens holdout:

| Footprint | Two-stream holdout | Three-stream holdout | Uniform neural holdout |
|---|---:|---:|---:|
| 4x4 | 0.02974956 | 0.03142205 | 0.02782733 |
| 5x5 | 0.16202227 | 0.17937036 | 0.15407998 |
| 6x6 | 0.34494348 | 0.39039515 | 0.33293869 |

This is a useful control rather than a setback. The extra stream is useful
enough to reduce the in-sample objective, so the selector is functioning, but
the small calibration trace cannot safely support this number of per-block
degrees of freedom. Freeze the current pool and prioritize a larger,
representative calibration corpus plus a regularized selection objective
(for example, a penalty for departing from the uniform neural stream). Only
after that should we revisit richer ASTC latent layouts or internal encoder
shortlists.

## Sixty-third sweep: FP16-derived ASTC payload through Vulkan

The GPU path is now validated as a separate mechanism contract. The shader
device smoke accepts an externally packed standard ASTC payload and a CPU
decoded RGBA reference. `astc-vulkan-latent-smoke` can export these two files
from an F16 GGUF matrix without changing the production backend.

Using the leading 32x64 region of SmolLM2 F16 `blk.0.attn_q.weight`, a standard
ASTC 4x4 L+A payload (2,048 bytes) was encoded offline, uploaded to the exposed
Intel UHD Graphics 620 Vulkan device, fetched with `texelFetch` in a compute
shader, and compared against the CPU ASTC decoder. The dispatch passed; the
reported shader timestamp was 4,333 ns for this one 2,048-texel validation
dispatch.

This establishes byte-for-byte portable resource use and CPU/GPU decode
agreement for real model-derived data on this driver. It does **not** establish
cross-vendor numerical behavior, end-to-end LLM quality, or a performance win:
the dispatch is deliberately a decode/readback validation and includes neither
the affine reconstruction nor a matrix-vector kernel. Those remain separate
gates. The F16 GGUF is the correct source-of-truth for future GPU experiments;
all ASTC, metadata, and reference artifacts must be derived from the same
tensor and recorded footprint.

## Sixty-fourth sweep: correctness-first GPU ASTC matvec

The validation path now has a distinct compute shader which, for each output
row, samples all ASTC texels with `texelFetch`, applies the fixed runtime L+A
decoder, and accumulates a dot product with a deterministic activation vector.
The host computes the same dot product from the CPU-decoded RGBA payload; no
FP16 source weights are used as the numerical reference after encoding.

The exported 32x64 SmolLM2 FP16-derived 4x4 payload passed this CPU/GPU matvec
comparison on the Intel UHD 620. The timestamp was approximately 52.2 us for
the 32 row results. This is a **correctness-first** number only: the shader has
one invocation per output row and serially reduces 64 columns, so it does not
represent a viable inference kernel or a comparison with ggml Vulkan. The next
GPU change is a workgroup-parallel row reduction, retaining this serial kernel
as the oracle contract. Only then should we compare texture-path timing against
an ordinary buffer-backed FP16/Q4 reference on the same device.

## Sixty-sixth sweep: workgroup-parallel ASTC matvec across footprints

The ASTC matvec shader now uses one 64-invocation workgroup per output row.
Each invocation samples strided columns, performs the fixed L+A reconstruction,
and contributes to a shared-memory tree reduction. CPU results are still
computed from the ASTC-decoded RGBA reference, not the original F16 weights.

For the same FP16-derived 32x64 SmolLM2 probe and 32 output rows, all formats
passed CPU/GPU comparison on UHD 620:

| Footprint | ASTC payload bytes | GPU timestamp |
|---|---:|---:|
| 4x4 | 2,048 | ~9.33 us |
| 5x5 | 1,456 | ~10.00 us |
| 6x6 | 1,056 | ~10.33 us |

The earlier serial 4x4 oracle measured ~52.2 us, so the workgroup reduction
is materially faster while retaining the numerical contract. These timings are
still too small and too isolated to compare formats or claim a bandwidth gain:
they contain no activation upload, no realistic batching, and no ordinary
buffer/Q4 kernel. The immediate next runtime control is a buffer-backed
FP16-equivalent matvec with the same output shape and activation formula;
Q4 follows only when its unpack/dequant contract is equally explicit.

## Sixty-seventh sweep: buffer control prepared

A buffer-backed FP32 matvec shader is now compiled beside the ASTC shader. It
uses the same 64-thread row workgroup, activation formula, and shared-memory
reduction, but reads one float weight per element from an SSBO. This establishes
the intended apples-to-apples control. The remaining work is host binding and
readback for this shader; no ASTC-versus-buffer timing is reported until both
paths run against the same exported FP16-derived matrix and CPU oracle.

## Sixty-fifth sweep: native ASTC evidence for the Intel validation device

The exposed validation adapter reports `Intel(R) UHD Graphics 620 (KBL GT2)`,
PCI device `0x5917`, using ANV/Mesa 26.0.8. No ASTC- or Mesa-forcing
environment variable was present during the Vulkan tests. This is Kaby Lake
Gen9 hardware, not llvmpipe or the separately enumerated NVIDIA device.

The evidence for native sampler decode on this device is stronger than format
advertisement alone: Mesa's Gen9 history includes a dedicated workaround for a
hardware ASTC 5x5 sampler defect. The runtime accepts sampled 4x4, 5x5, and
6x6 images and the shader tests exercise all three. This supports interpreting
the UHD 620 results as native ASTC texture-hardware behavior for this driver.

It remains a device-specific conclusion, not a portability assertion. Mesa can
implement ASTC emulation on hardware that lacks it, and a future benchmark must
record adapter PCI ID, driver, environment, and format features. The generic
runtime contract remains: unsupported or emulated configurations may validate
correctness but cannot substantiate the compressed-bandwidth hypothesis.

## Sixty-eighth sweep: FP32 buffer control and first apples-to-apples timing

The host-side shader smoke now has an explicitly isolated `--buffer-matvec`
control. It binds the same output SSBO, uses the same 64-invocation workgroup,
activation function, shared-memory reduction, push constants, and dispatch
geometry as `astc-matvec.comp`. The only intentional difference is the weight
source: the control reads one FP32 value per matrix element from a storage
buffer, while the ASTC path performs `texelFetch` followed by the fixed L+A
reconstruction in the shader. No `ggml-vulkan` source or production Vulkan
pipeline is changed.

The control accepts an optional `--weights weights-f32.bin` file, so a future
run can use the exact FP16-derived source matrix. Until that artifact is
available, the smoke can derive FP32 control weights from the same decoded RGBA
reference and affine constants used by the ASTC run. That current mode is a
transport/decode control, not an original-FP16 quality comparison; the source
matrix provenance is therefore recorded explicitly rather than inferred.

On the Intel UHD Graphics 620 (Kaby Lake, ANV/Mesa 26.0.8), the bounded
SmolLM2 F16-derived 32x64 probe was run with 20 in-command-buffer dispatches
and the reported timestamp divided by 20:

| Path | Footprint argument | Per-dispatch GPU timestamp |
|---|---:|---:|
| FP32 storage-buffer control | 4x4 | ~2.88 us |
| ASTC sampled image | 4x4 | ~2.36 us |
| ASTC sampled image | 5x5 | ~2.46 us |
| ASTC sampled image | 6x6 | ~2.40 us |

The small workload and shared shader reduction make these numbers sensitive to
driver scheduling and timestamp granularity. They are a mechanism signal only:
ASTC was not slower than the buffer control in this run, but this is not yet a
bandwidth claim, a model-quality result, or an end-to-end inference speedup.
The next measurement must use a larger representative matrix (or a tiled test
fixture), preserve the exact FP16 source for both paths, include upload/cache
effects separately from steady-state sampling, and repeat across adapters with
native versus emulated ASTC identified.

The existing 4x4/5x5/6x6 ASTC validation CTests remain unchanged and pass. The
buffer path is deliberately a manual/experimental control at this stage because
its model-derived binary fixtures are not checked into the repository.

## Current execution plan after the buffer control

1. [x] Keep ASTC matvec and FP32 buffer control in separate experimental
   shaders and verify both with the same CPU reconstruction oracle.
2. [x] Add repeated GPU timestamping and report per-dispatch values for 4x4,
   5x5, and 6x6.
3. [x] Export one exact FP32/F16-derived source matrix and matching ASTC
   payload/reference fixtures from the same tensor; run both paths over a
   larger matrix and a larger batch of rows.
4. [x] Add an explicit Q4/TQ2 buffer control with named shaders and preserved
   CTests, then compare quality, bytes/weight, and GPU time against ASTC.
5. [ ] Measure quality on the exact common-shape artifacts, add a sampled-FP32
   microbenchmark, and test hot-cache versus streaming/capacity regimes.
6. [ ] Only after the controls are stable, prototype activation-aware
   candidate ranking and re-run the earlier 4x4/5x5/6x6, TQ1/TQ2, scalar,
   and ASTC-Q evaluations from the same source artifacts.

## Sixty-ninth sweep: full-size FP16-derived buffer comparison

The latent exporter now writes the cropped source matrix as an explicit F32
binary fixture with `--export-weights`. This is the F16 GGUF tensor converted
to F32 by the shared ggml reader; it is not a re-read of an ASTC payload. The
same 576x576 `blk.0.attn_q.weight` tensor was exported alongside independent
4x4, 5x5, and 6x6 ASTC payload/reference files. The Vulkan buffer control was
updated to accept that exact source through `--weights` and to validate its
own CPU dot-product oracle.

For timing, each path executed 1,000 matvec dispatches in one command buffer;
the host reports the timestamp interval divided by the repeat count. The
workgroup geometry and activation function are identical between paths. On
the Intel UHD Graphics 620 (Kaby Lake, ANV/Mesa 26.0.8) the result was:

| Path | Resident weight payload | Per-dispatch GPU timestamp |
|---|---:|---:|
| FP32 storage-buffer control | 1.27 MiB | ~144.6 us |
| ASTC 4x4 sampled image | 324 KiB | ~153.9 us |
| ASTC 5x5 sampled image | 211 KiB | ~157.8 us |
| ASTC 6x6 sampled image | 144 KiB | ~160.7 us |

The ASTC sampled path is therefore about 6%, 9%, and 11% slower than this
buffer control in this first full-size steady-state probe, despite reducing
resident weight bytes by roughly 4x, 6x, and 9x respectively. This is a
useful correction to the earlier tiny-fixture hint: compression density does
not automatically translate into faster shader execution. Texture decode and
sampling latency can dominate, and the current kernel still performs the same
floating-point reduction in both cases.

This is not yet a whole-model inference conclusion. It excludes activation
traffic, cache warm-up policy, batching, other matrix shapes, and Q4/TQ2
controls. It also measures one native Intel implementation; other adapters may
have different ASTC sampler throughput. The result does, however, establish a
credible baseline for the next experiments: any ASTC variant must beat this
same-source buffer timing while preserving its quality gate, or demonstrate a
separate memory/energy benefit that justifies the decode cost.

## Seventieth sweep: isolated Q4_0 and TQ2_0 Vulkan controls

The PoC now includes named Q4_0 and TQ2_0 compute shaders. They read the
standard ggml packed layouts directly from an SSBO and perform dequantization
inside the same 64-thread row-reduction structure as the FP32 and ASTC paths.
The host has a small `astc-vulkan-quant-export` tool that quantizes the exact
F16-derived GGUF matrix with ggml's reference quantizers, preserving the
packed bytes as the fixture under test. A CPU oracle decodes the same packed
bytes, so a pass cannot be caused by comparing a quantized result with the
unquantized source.

TQ2 requires a 256-element block. For a common shape, the leading 576x512
region of `blk.0.attn_q.weight` was exported in all four forms:

| Path | Packed weight payload | Nominal rate |
|---|---:|---:|
| FP32 buffer source | 1,179,648 bytes | 32.00 bpw |
| Q4_0 | 165,888 bytes | 4.50 bpw (including 16-bit scale/block) |
| TQ2_0 | 76,032 bytes | 2.0625 bpw |
| ASTC 4x4 / 5x5 / 6x6 | 294,912 / 191,168 / 132,096 bytes | 8.00 / 5.18576 / 3.58333 bpw (edge-padded) |

All five shaders passed the Vulkan/CPU correctness gate. With 1,000
dispatches in one command buffer on the Intel UHD Graphics 620, the first
steady-state timestamps were:

| Path | Per-dispatch GPU timestamp |
|---|---:|
| Q4_0 SSBO dequant | ~173.4 us |
| TQ2_0 SSBO dequant | ~187.9 us |
| ASTC 4x4 sampled image | ~158.1 us |
| ASTC 5x5 sampled image | ~172.3 us |
| ASTC 6x6 sampled image | ~196.5 us |

This is a mechanism comparison, not a quality ranking. On this adapter and
shape, ASTC 4x4 is faster than the standalone Q4_0 shader despite its larger
resident payload; ASTC 5x5 is close to Q4_0, while 6x6 is slower than both Q4
and TQ2. The result reinforces that block density and sampler/dequant latency
must be measured together. It also gives TQ2 a real GPU control rather than
the earlier offline-only result.

The next gate is to calculate elementwise and activation-relative error for
these exact 576x512 artifacts, using the captured layer trace, then repeat the
timing with a batched/multi-row workload. TQ1 remains a separate follow-up:
its base-3 unpacking and 1.6875 bpw layout should not be conflated with TQ2's
two-bit shader.

## Seventy-third sweep: common-shape quality gate

The exact 576x512 F16-derived source was evaluated with the 25-sample layer-0
holdout trace. The current L+A ASTC payloads, Q4_0, and TQ2_0 were decoded and
scored with the same activation-relative objective:

| Representation | Nominal/effective rate | Element MSE | Activation-relative MSE |
|---|---:|---:|---:|
| Q4_0 | 4.50 bpw | 0.00081560 | 0.0076115 |
| ASTC L+A 4x4 | 8.00 bpw | 0.010458 | 0.13270 |
| ASTC L+A 5x5 | 5.18576 bpw | 0.053897 | 0.68481 |
| TQ2_0 | 2.0625 bpw | 0.053860 | 0.53004 |
| ASTC L+A 6x6 | 3.58333 bpw | 0.078718 | 0.99737 |

This changes the immediate interpretation of the GPU timings. ASTC 4x4 was
faster than the standalone Q4 shader, but its present L+A representation is
far less faithful on real activation traces. The result is not an ASTC-format
failure: earlier scalar-RGBA ASTC runs on the same family of weights had much
lower error, especially at 4x4 and 5x5. It does show that the L+A additive
mapping needs activation-aware candidate ranking or training before it can be
compared as a serious Q4 replacement.

The next experiment therefore has two coupled parts: export and time the
scalar-RGBA ASTC payload with the same Vulkan shader, then compare scalar and
L+A at equal footprint. Only after that should we spend effort on a richer
latent representation. The Q4/TQ2 controls remain useful because they expose
the quality/latency frontier that ASTC must approach.

## Seventy-fourth sweep: scalar ASTC versus L+A

The scalar-RGBA exporter path now emits the same common 576x512 F16-derived
matrix with `R=G=B` and no residual alpha stream. The existing ASTC matvec
shader can consume it by using the scalar affine constants
`scale_l=10.03125`, `scale_a=0`, and `offset=-4.78125`.

On the 25-sample holdout trace, scalar ASTC activation-relative MSE was
approximately 0.00194 (4x4), 0.01799 (5x5), and 0.07693 (6x6). For reference,
the current L+A payloads measured 0.13270, 0.68481, and 0.99737 respectively;
Q4_0 was 0.00761 and TQ2_0 was 0.53004. The scalar mapping is therefore much
more faithful than L+A on this layer, especially at 4x4 and 5x5, although Q4
still has the best error in this particular comparison.

The corresponding 1,000-dispatch scalar-ASTC timestamps were ~162.5 us,
~148.8 us, and ~166.7 us for 4x4, 5x5, and 6x6. They differ materially from
the L+A timings (~158.1 us, ~172.3 us, and ~196.5 us), showing that the latent
mapping changes both shader arithmetic and the encoder's block statistics. This
is a useful result: ASTC footprint alone does not determine runtime, and the
representation layout is now an explicit performance variable.

The scalar and L+A figures come from separate process runs and remain
directional until a single harness controls warm-up and ordering. The next
quality/runtime comparison should use scalar ASTC, L+A ASTC, Q4, and TQ2 in
one run, then move to a streaming working set. L+A-specific training should be
deprioritized until it can close this scalar quality gap.

## Seventy-fifth sweep: execution boundary and target-device interpretation

The timed ASTC matvec performs `texelFetch` from a Vulkan ASTC image. At that
point the shader requests a sampled texel; ASTC block decoding occurs in the
GPU's texture/sampler implementation, not in the CPU and not in shader source
code. CPU work is deliberately outside that path: `astcenc` creates payloads
offline, and the CPU decoder supplies correctness references. The Vulkan image
upload occurs before the timestamp interval. Therefore the reported interval
measures resident GPU execution, not model loading or host-to-device transfer.

The Intel UHD 620 is an integrated GPU, so its CPU and GPU share system memory
rather than crossing a discrete-GPU PCIe link during resident texture fetches.
On a discrete target, an initial ASTC upload may cross PCIe, but repeated
inference still samples the resident GPU image and does not perform a per-fetch
CPU or PCIe round trip. Model-load transfer and steady-state inference must be
reported as separate measurements.

UHD 620 establishes a Vulkan resource/correctness contract and gives a useful
mechanism signal; it is not a performance proxy for Mali, Adreno, Apple, or
other mobile ASTC targets. Relative texture throughput, ASTC decoder design,
cache hierarchy, unified-memory behavior, driver scheduling, and shader-ALU
balance can all change the result. A target-device matrix is therefore a
required later gate, with each run recording adapter, driver, ASTC footprint,
and whether native decode evidence is available.

L+A remains a retained research representation. Its current post-training
mapping loses to scalar ASTC on this layer, but it still exposes two useful
ideas unavailable to scalar replication: dual-plane-compatible latent fields
and cancellation-aware reconstruction after decode. It should be revisited
through activation-aware candidate ranking and codec-aware optimization, not
discarded. Scalar ASTC is the immediate practical baseline; L+A is the
neural-dequantization hypothesis to test once the runtime controls are stable.

The parallel BCn/CUDA work is related only at the architectural level
(fixed-function texture decompression as neural dequantization). This branch
remains ASTC/Vulkan-focused so that its format, hardware, quality, and runtime
claims stay independently interpretable.

## Seventy-sixth sweep: neural error-shaping research direction

The research direction is now named **neural error shaping for block codecs**:
the encoder should coordinate independently legal ASTC blocks so the aggregate
error falls in directions to which the layer or model is insensitive. For a
weight error `E` and calibration input covariance `H_I`, the current objective
is `tr(E H_I E^T)`, equivalently the exact squared layer-output error on the
captured input trace. This is an offline encoder/selector concern; Vulkan still
samples an ordinary standard ASTC image and has no knowledge of calibration,
Hessians, candidates, or selection state.

Some of this mechanism has already been evaluated. The legal-block
coordinate-descent contract (sweep 59) demonstrated cross-block cancellation
on every deterministic fixture. The real-trace adapter (sweeps 61--62) then
showed the necessary caution: two and three whole-image streams improve the
calibration objective, but neither beats uniform neural ranking on the small
independent holdout. The present evidence is therefore **not** a quality win;
it shows that a narrow, highly correlated candidate pool can overfit.

The next practical experiment is consequently not a custom ASTC encoder and
not a new shader. It is a side-fork encoder interface that retains a bounded
per-block set of legal candidates. The shortlist must include the local
activation-loss optimum and several candidates whose transformed errors
`E_c L`, for `H_I = L L^T`, differ in direction. Coordinate selection can then
be rerun on scalar ASTC 4x4, 5x5, and 6x6 with nested calibration/validation
splits, a replacement penalty, and a strictly untouched holdout.

Hessian-guided block error feedback is retained as the first more exotic
algorithm: committed block error is projected into a compact
input-sensitivity basis and used to alter future offline encoding targets.
It must first be measured against exact coordinate selection on controlled
fixtures. Two-sided Kronecker scoring, `tr(H_O E H_I E^T)`, is a later
experiment: the present layer-output trace has only `H_O = I`; a nontrivial
output factor requires separately captured downstream/model-loss sensitivity.
Projected-residual beam/trellis selection, latent gauge redundancy for L+A,
and ASTC-noise-aware fine-tuning are explicitly deferred until these scalar
post-training gates show holdout value.

This positioning is supported by GPTQ and QuIP's sensitivity-aware adaptive
rounding, QTIP's high-dimensional trellis search, YAQA's model-aware
Kronecker-factored objective, and BaKron's efficient two-sided solver. They
are algorithmic precedents, not claims that those quantizers or their runtime
kernels can be substituted for ASTC. The final payload of every proposed ASTC
experiment remains standard, independently decodable 128-bit blocks.

## Seventy-seventh sweep: diverse scalar candidate-pool control

The latent selector now exposes `--coordinate-diverse`. It creates a bounded
pool of four complete, legal L+A ASTC streams: ordinary ranking, neural
ranking at `thorough`, `medium`, and `fast` encoder presets. For every ASTC
block, the selector retains the ordinary stream as a stable baseline, the
lowest local activation-loss candidate, and up to two additional candidates
chosen by farthest-point distance between their calibration output-error
vectors. The resulting stream is still assembled from complete 16-byte blocks;
no Vulkan shader or production backend is involved.

The contract and the existing neural-rank smoke both pass. On the bounded
SmolLM2 F16 `blk.0.attn_q.weight` probe (32x64 region, 10 calibration and 25
holdout samples), the new pool reduced calibration error but did not beat the
uniform neural-ranked stream on holdout:

| Footprint | Uniform neural holdout | Diverse-pool calibration | Diverse-pool holdout | Shortlist average |
|---|---:|---:|---:|---:|
| 4x4 | 0.02782733 | 0.02415803 | 0.03129320 | 3.852 / 4 |
| 5x5 | 0.15407998 | 0.10975492 | 0.16197053 | 3.934 / 4 |
| 6x6 | 0.33293869 | 0.28715630 | 0.38492473 | 4.000 / 4 |

This is a useful negative control. Directional diversity alone does not
prevent overfitting when the calibration set is still small and the candidate
streams are correlated by construction. It does, however, demonstrate that
the per-block shortlist is populated and that the selector can lower the
in-sample objective. The next required change is nested calibration/validation
selection plus a replacement penalty, followed by an untouched holdout. The
Hessian-feedback and two-sided objectives remain deferred until that gate is
passed.

## Seventy-eighth sweep: regularized calibration/validation selection

The selector now supports `--coordinate-regularized`. It splits the supplied
calibration trace into equal selection and validation halves, chooses a
replacement penalty from a fixed offline grid, then reruns the selected
penalty on all calibration samples before measuring the untouched holdout.
The penalty is normalized to the initial output-error energy and is charged
when a block departs from the uniform neural-ranked candidate. This is a
selection guard only; it adds no runtime metadata or shader work.

The new contract passed. On the same bounded SmolLM2 32x64 probe, the
validation-selected penalty was zero for all three footprints. The
regularized selector nevertheless changed the holdout relative to the prior
diverse run:

| Footprint | Uniform neural holdout | Prior diverse holdout | Regularized holdout | Validation relative MSE | Selected penalty |
|---|---:|---:|---:|---:|---:|
| 4x4 | 0.02782733 | 0.03129320 | 0.03045620 | 0.03188133 | 0 |
| 5x5 | 0.15407998 | 0.16197053 | 0.16128243 | 0.17747256 | 0 |
| 6x6 | 0.33293869 | 0.38492473 | 0.38919650 | 0.47478864 | 0 |

This is a modest improvement for 4x4 and 5x5 over the unregularized
diverse-pool selector, but it still does not beat the uniform neural stream;
6x6 worsens. The zero penalty is itself informative: on this small split,
the validation data did not justify preferring the uniform stream strongly
enough to pay for the added restriction. The method is now structurally ready,
but it needs a larger calibration corpus and more layers before any penalty
or selector policy can be considered reliable. Hessian-guided feedback remains
the next algorithmic sweep, beginning with a synthetic matched baseline.

## Seventy-ninth sweep: Hessian-guided feedback mathematical contract

The first error-feedback contract is now isolated in
`test-astc-vulkan-error-shaping`. It uses two scalar block decisions with
identical `{0, 1}` candidate alphabets and a positive correlated input Hessian

```
H = [[1.0, 0.9],
     [0.9, 1.0]]
```

For the reference `(0.49, 0.45)`, independent local rounding selects `(0, 0)`
and has quadratic error `0.8395`. After committing the first block, the
conditional target for the second is shifted to

```
target_2 = w_2 + H_21 / H_22 * (w_1 - q_1) = 0.891
```

so feedback selects `(0, 1)` and reduces the same quadratic error to `0.0575`.
This is a deliberately minimal proof that a locally worse second choice can be
globally better under correlated sensitivity. It is not an ASTC quality result:
the fixture has no ASTC candidates, endpoint decisions, or bitstream. It is a
guard against implementing the feedback sign or Hessian indexing incorrectly
before adding the real legal-block adapter.

The contract passes alongside the existing ASTC tests. The next implementation
must replace the scalar choices with a bounded ASTC block shortlist, use a
low-rank calibration-sensitivity basis for larger matrices, and compare against
the exact cached-residual coordinate selector on matched synthetic ASTC
fixtures. Only then is a real-layer feedback run justified.

## Eightieth sweep: scaling and generalization implications

The 32x64 selection probe contains 2,048 weights. The full 576x576 attention
projection contains 331,776 weights, or 162 times as many; it has 20,736,
13,456, and 9,216 ASTC blocks at 4x4, 5x5, and 6x6 respectively. This gives a
global selector more possible compensating choices, so the small-probe negative
results do not reject block-level error shaping. It also gives the selector far
more opportunities to fit accidental details of a trace.

The project policy is therefore updated: full-layer size is not treated as a
quality argument by itself. Every larger run must increase and partition its
trace evidence, report selection/validation/holdout cohort sizes and candidate
freedom, and beat its uniform baseline on untouched data. Offline search is
allowed to be expensive but must use streaming candidate storage and a compact
sensitivity basis; deployed Vulkan work remains unchanged. Attention and FFN
projections are separate cohorts rather than interchangeable measurements.

## Eighty-first sweep: selector-comparison criteria

The next Hessian-feedback result will use a fixed pool of legal ASTC blocks and
compare three selectors only: local neural ranking, exact cached-residual
coordinate descent, and Hessian feedback. This keeps candidate generation out
of the first comparison. Coordinate descent is treated as an oracle-ish offline
upper reference, while feedback is judged by how much of its improvement it
recovers at lower search cost.

Alongside absolute activation losses, the result will report recovered gain
`(L_local - L_HF) / (L_local - L_CD)` only when the denominator is positive.
The selector will also log residual norm and `cos(R, DeltaY)` for accepted
blocks. Candidate diversity will be computed from calibration output deltas
`E_c X^T`, rather than visual RGBA difference; this is essential because the
trace Hessian is low rank when there are far fewer calibration samples than
weight columns. A feedback method that later regenerates ASTC blocks from
modified targets is a separate, stronger encoder experiment and must not be
presented as the same fixed-pool comparison.

## Eighty-second sweep: first three-way selector comparison

The PoC now exposes `--selector-compare`. It constructs one fixed pool of four
legal ASTC streams (standard, neural-rank, fast, and medium), derives the same
bounded activation-space shortlist for every block, and evaluates:

1. local neural ranking, independently per block;
2. cached-residual coordinate descent with forward and reverse sweeps; and
3. simultaneous residual-feedback rounds, a parallel Jacobi approximation to
   the Hessian-guided selector.

The initial results are a useful negative control. On the 8x32 synthetic 4x4
fixture, holdout relative MSE was `7.0236e-5` local, `4.5552e-5` coordinate,
and `1.1711e-4` feedback (`G_HF=-1.899`). On the 32x64 SmolLM2
`blk.0.attn_q.weight` probe, results were:

| Footprint | Local | Coordinate | Feedback |
| --- | ---: | ---: | ---: |
| 4x4 | 0.03032 | 0.03062 | 0.03239 |
| 5x5 | 0.14044 | 0.16937 | 0.16929 |
| 6x6 | 0.35711 | 0.39286 | 0.35967 |

These numbers do not establish a quality win. They do establish a reproducible
three-way harness and show that a shared residual with simultaneous updates is
not yet a reliable curvature model. The next implementation should add
damping or conflict-aware acceptance and retain this negative-control result.

## Eighty-third sweep: conflict-aware acceptance

The PoC now also performs a parallel-proposal/sequential-accept pass. Each
block proposes its best candidate from the frozen residual, proposals are
ordered by predicted gain, and each proposal is re-evaluated against the
residual after earlier commits. This preserves parallel candidate scoring while
rejecting simultaneous overshoot.

On the synthetic 8x32 4x4 fixture, conflict-aware selection reached
`4.3416e-5` holdout relative MSE versus `7.0236e-5` local and `4.5552e-5`
coordinate descent, recovering `1.087` of the coordinate gain. On the bounded
SmolLM2 probe it improved calibration consistently, but did not beat local on
holdout:

| Footprint | Local calibration | Conflict calibration | Local holdout | Conflict holdout |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 0.03015 | 0.02470 | 0.03032 | 0.03100 |
| 5x5 | 0.15952 | 0.11950 | 0.14044 | 0.16254 |
| 6x6 | 0.38280 | 0.30295 | 0.35711 | 0.35839 |

The result isolates the remaining issue: conflict handling fixes calibration
overshoot, but the selected directions can still be calibration-specific. The
next sweep therefore adds two-shard stability gating with no new candidate
pool.

## Eighty-fourth sweep: two-shard stability gating

The fixed-pool harness now computes two equal calibration shards and accepts a
proposal only when its normalized residual-energy gain is positive in both
shards. This is deliberately conservative with the current ten-sample trace;
larger traces are still needed before using more shards.

On the synthetic 8x32 4x4 fixture, stability gating matched the conflict-aware
result at `4.3416e-5` holdout. On the bounded SmolLM2 probe:

| Footprint | Local calibration | Stability calibration | Local holdout | Stability holdout |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 0.03015 | 0.02710 | 0.03032 | 0.03029 |
| 5x5 | 0.15952 | 0.13413 | 0.14044 | 0.14322 |
| 6x6 | 0.38280 | 0.33915 | 0.35711 | 0.35840 |

Stability gating removes much of the calibration-only gain from
conflict-aware selection and restores holdout behavior close to local ranking.
It has not yet produced a robust quality win, but it supports calibration
stability as the correct regularization axis and should remain in the comparison
matrix while larger activation traces are collected.

## Eighty-fifth sweep: input-Hessian diagnostics

The selector comparison now prints a small-probe diagnostic for
`H_I = X^T X`: calibration sample count, column count, estimated positive
rank, eigenvalue range, and a reference damped condition number. The current
SmolLM2 probe reports 10 samples, 64 columns, and rank 10; the synthetic 32-
column fixture reports 4 samples and rank 4. This confirms that the present
trace is intentionally a mechanism probe, not enough evidence for a full
Block-LDLQ claim. Larger traces must precede interpretation of target
regeneration, and the factorization contract will explicitly cover damping and
rank-deficient inputs.

## Eighty-sixth sweep: Block-LDLQ contract

The first Block-LDLQ contract is now checked independently of ASTC encoding.
For a committed block error `e_C = q_C - W_C`, the continuous conditional
target for a future block is shifted by
`W_F - H_FF^-1 H_FC e_C`. The contract uses a two-block correlated Hessian
where independent local rounding chooses `{0, 0}`, while the regenerated target
chooses `{0, 1}` and reduces quadratic error from `0.8395` to `0.0575`.

This is the target-update mechanism we will insert into ASTC block selection.
The test is built CPU-only in an isolated directory because the existing
Vulkan build directory lacks the optional SPIRV-Headers package; no production
Vulkan target is changed.

## Eighty-seventh sweep: bounded candidate-capacity pool

The latent harness now exposes an opt-in `--candidate-sweep` mode. It encodes
the same L+A source with ASTC's existing per-mode candidate limits `1, 2, 4,
8`, then adds those standard-compatible streams to the fixed-pool selector
comparison. This is intentionally not a new ASTC bitstream and does not alter
the normal Vulkan path. It measures whether additional candidates already
available inside `astcenc` provide useful alternative block choices before we
add a true per-block callback or a custom neural-aware encoder.

The pool is still a whole-image bounded proxy: each stream is independently
encoded and the selector chooses blocks across streams. It therefore tests
candidate diversity and selection behavior, but it is not yet an internal
ASTC candidate pool. The next gate is to compare pool size, unique block
choices, calibration/holdout loss, and encoding cost on the synthetic fixture
and the SmolLM2 layer trace. A real internal top-K API remains opt-in and
isolated to the side fork if this proxy shows a measurable benefit.

On the synthetic 8x32 6x6 fixture, the eight-stream pool produced five unique
candidate choices in the local, conflict-aware, and stability selections. The
conflict-aware and stability selectors reached `1.2993e-4` activation MSE,
slightly better than the coordinate result `1.3453e-4`; simple feedback was
worse at `4.4502e-4`.

On the bounded SmolLM2 `blk.0.attn_q.weight` probe (32x64, ten calibration
samples), the richer pool changed the holdout picture as follows:

| Footprint | Local | Conflict-aware | Stability |
| --- | ---: | ---: | ---: |
| 4x4 | 0.02875 | 0.03114 | 0.02990 |
| 5x5 | 0.14465 | 0.16566 | 0.14348 |
| 6x6 | 0.34705 | 0.34072 | 0.35044 |

These values are activation-relative holdout MSE. The pool is promising for
6x6 conflict-aware selection and keeps stability close to local ranking, but
it is not a universal win: 5x5 still overfits and 4x4 remains mixed. This is
exactly the intended gate. We should not claim that a larger candidate pool
solves the problem; the next implementation should expose true per-block
top-K candidates so Block-LDLQ/GPTVQ target regeneration can rank alternatives
against a common, not whole-image, candidate set.

## Seventy-first sweep: common-shape FP32 baseline

The missing FP32 point for the Q4/TQ2 comparison is now measured on the exact
same 576x512 source shape. The buffer control consumed the exported F16-derived
F32 matrix with 1,000 dispatches and reported approximately 167.0 us per
dispatch on the UHD 620. Combining this with the immediately preceding
common-shape run gives the provisional mechanism table:

| Path | Per-dispatch GPU timestamp |
|---|---:|
| FP32 storage-buffer control | ~167.0 us |
| ASTC 4x4 sampled image | ~158.1 us |
| Q4_0 SSBO dequant | ~173.4 us |
| ASTC 5x5 sampled image | ~172.3 us |
| TQ2_0 SSBO dequant | ~187.9 us |
| ASTC 6x6 sampled image | ~196.5 us |

This puts ASTC 4x4 about 5% below the FP32 control and about 9% below the
standalone Q4 shader for this shape, while ASTC 5x5 is essentially tied with
Q4. These measurements were collected in separate process runs, so they are a
directional result until the forthcoming harness executes all paths under one
controlled warm-up/timing protocol. The quality axis is still intentionally
open: the next report must join this timing table with elementwise and
activation-relative errors from the same source and trace.

## Seventy-second sweep: sampled-FP32 texture isolation

To separate generic texture sampling from ASTC block decoding, the PoC now has
an `R32_SFLOAT` sampled-image shader. It uses the same 64-thread reduction and
the same affine reconstruction arithmetic as the ASTC shader; each scalar texel
is replicated across the synthetic RGBA lanes only to keep the ALU shape
comparable. The image is populated from the exact F16-derived 576x512 FP32
source, and the CPU oracle validates the original source weights.

On the UHD 620, 1,000 dispatches measured approximately 56.4 us per dispatch.
The corresponding SSBO FP32 control was ~167.0 us and ASTC 4x4 was ~158.1 us on
the same common shape. This strongly suggests that the current scalar SSBO
control is limited by its buffer access path and that ASTC's extra cost is not
just “being a texture”: ASTC block reconstruction remains materially more
expensive than a plain sampled FP32 texel on this adapter.

This result must not be overinterpreted as a production-kernel comparison. The
SSBO and sampled controls are intentionally transparent scalar shaders, and
the ASTC path performs real four-channel decode plus affine reconstruction. The
useful conclusion is methodological: future profiling must report at least
three mechanisms (buffer load, sampled-FP32 load, ASTC load) before attributing
a timing difference to bandwidth or fixed-function parallelism.

The immediate next quality gate remains unchanged: evaluate FP32, Q4_0, TQ2_0,
and ASTC 4x4/5x5/6x6 on the same source and activation trace. Only after that
should a streaming working-set test decide whether ASTC's resident-size saving
can compensate for its sampler/decode latency.

## Eighty-eighth sweep: side-fork per-block top-K callback

The isolated `astc-encoder-neural-rank` fork now exposes an opt-in
`astcenc_candidate_callback`. It receives the physical 16-byte block, source
block position, local post-realignment ASTC error, partition count, and plane
metadata. The callback is disabled by default and is not part of the llama
Vulkan runtime. The PoC invokes it with a single compressor thread, making the
collector deterministic and avoiding an implicit thread-safety promise.

The latent harness uses this callback to retain up to four unique candidates
per block, splice each candidate into the neural baseline stream, decode it
through astcenc, and feed the resulting per-block alternatives to the existing
selectors. This is the first real per-block candidate pool rather than a
whole-image capacity proxy.

On the bounded SmolLM2 layer (32x64, ten calibration samples), the collector
observed 1386/1013/1005 callbacks and retained 512/364/264 block candidates
for 4x4/5x5/6x6 respectively. Holdout activation-relative MSE was:

| Footprint | Local | Coordinate | Conflict-aware | Stability |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 0.03034 | 0.03127 | 0.03184 | 0.03030 |
| 5x5 | 0.14707 | 0.19443 | 0.16330 | 0.15644 |
| 6x6 | 0.36614 | 0.41279 | 0.37485 | 0.37095 |

The richer per-block pool is not a quality win by itself. It can lower
calibration loss while worsening holdout, especially for 5x5. This confirms
that candidate generation and candidate selection must be separated: the
callback proves that legal alternatives are available, while the next engine
must regenerate future targets using Block-LDLQ/GPTVQ-style Hessian feedback
and evaluate candidates in a block-local, calibration-stable way.

## Eighty-ninth sweep: first Block-LDLQ target regeneration

The selector matrix now includes a small Block-LDLQ-style path. It forms the
input Gram/Hessian `H = X^T X`, processes ASTC blocks in scan order, selects a
legal candidate against a regenerated continuous target, and shifts future
column targets with a damped cross-Hessian update. This is intentionally a
diagonal/damped PoC, not yet a full block-LDL factorization; the independent
two-block contract remains the mathematical reference.

On the bounded SmolLM2 layer with the per-block top-K pool, holdout
activation-relative MSE was:

| Footprint | Local | Block-LDLQ | Conflict-aware | Stability |
| --- | ---: | ---: | ---: | ---: |
| 4x4 | 0.03034 | 0.03201 | 0.03184 | 0.03030 |
| 5x5 | 0.14707 | 0.18849 | 0.16330 | 0.15644 |
| 6x6 | 0.36614 | 0.35580 | 0.37485 | 0.37095 |

The 6x6 result is the first real-model holdout improvement from target
regeneration: it beats local and the other selectors in this bounded probe,
although it does not yet beat the earlier whole-image conflict-aware proxy.
The 4x4 and 5x5 results are negative, so no general quality claim is allowed.
The next refinement is a true block-LDL update (not diagonal damping), with
explicit candidate validity/coverage checks and larger calibration traces.

## Ninetieth sweep: compact block-LDL update

The target-regeneration prototype now solves a damped `H_FF` system for each
future ASTC-width column block and applies `H_FF^-1 H_FC` to the target. This
replaces the earlier scalar/diagonal update while retaining the same legal
candidate pool and scan-order control. It is still a bounded research
implementation, not a production quantizer or a full sparse factorization.

On the bounded SmolLM2 6x6 probe, the block solve reduced calibration
activation-relative MSE to `0.29496` and holdout to `0.34428`, compared with
`0.32886`/`0.41279` for coordinate selection and `0.36614` for local holdout.
The previous diagonal Block-LDLQ prototype measured `0.35580` holdout, so the
compact future-block solve recovers additional quality. The result is a
promising 6x6 signal, not yet a cross-footprint claim; 4x4 and 5x5 remain
required regression points in the next full sweep.

## Ninety-first sweep: cross-tensor 6x6 control

The same per-block pool and compact block-LDL update were run on
`blk.0.attn_k.weight` from the same SmolLM2 FP16 model, using the same
calibration and holdout traces and the same 32x64 crop. Holdout
activation-relative MSE was:

| Selector | Holdout |
| --- | ---: |
| Local | 0.24283 |
| Coordinate | 0.29983 |
| Block-LDLQ | 0.25469 |
| Conflict-aware | 0.27234 |
| Stability | **0.25335** |

The target-regeneration result is close but does not beat local ranking on
this tensor; stability gating is best by a small margin. This is an important
boundary: the 6x6 Block-LDLQ gain on `attn_q` is not yet universal. Keep the
method as a candidate, but require multiple tensors and larger traces before
claiming a model-level advantage.

## Ninety-second sweep: per-block candidate ownership

Each captured top-K candidate now carries its source block, and every selector
filters candidates so a block only sees the baseline streams plus alternatives
captured for that same block. The comparison also reports minimum, maximum,
and average candidate coverage. On the synthetic 6x6 fixture coverage was
6--8 candidates per block (7.5 average), with no change to the previously
observed selector ordering. This removes a misleading global-pool degree of
freedom and makes subsequent Block-LDLQ/stability results easier to interpret.

### Design decision: variable block coding remains allowed

The experiment does not assume one identical coding decision for every block
or even every tensor. ASTC already permits per-block modes, partitions,
quantization levels, and dual-plane choices, and the offline selector may
choose among them. We will initially keep one footprint per tensor to retain a
simple runtime layout; mixed 4x4/5x5/6x6 footprints inside one tensor remain a
later optimization requiring explicit metadata and address handling.

## Ninety-third sweep: calibration-size ablation with fixed holdout

The harness now accepts `--max-calibration-samples` separately from
`--max-samples`, preventing calibration-size experiments from silently
changing the holdout set. On `attn_q` with 6x6 and the per-block pool, the
Block-LDLQ/local holdout pairs were:

| Calibration samples | Local | Block-LDLQ |
| ---: | ---: | ---: |
| 4 | 0.35908 | 0.39023 |
| 8 | 0.35378 | 0.35297 |
| 10 | 0.36614 | 0.34428 |

The result supports the larger-trace requirement but is not monotonic: a
four-sample Hessian overfits badly, eight samples are near parity, and the
full ten-sample trace gives the clearest Block-LDLQ gain. Calibration diversity
therefore matters in addition to sample count; future runs must report both
trace size and shard composition.

## Ninety-fourth sweep: Block-LDLQ damping ablation

The Block-LDLQ damping factor is now exposed as `--ldlq-damping`. On the
SmolLM2 `attn_q` 6x6 per-block pool, factors `1e-5`, `1e-4`, and `1e-3`
produced identical selector results (`0.34428` holdout for Block-LDLQ). The
current candidate set is therefore not sensitive to this damping range; the
dominant variables remain calibration geometry, block order, and which legal
candidate directions are retained. Damping stays configurable for future
larger-trace and ill-conditioned cases, but is not the next optimization
target.

## Ninety-fifth sweep: Block-LDLQ order ablation

Block-LDLQ now exposes `--ldlq-order forward|reverse`. On the SmolLM2
`attn_q` 6x6 pool, forward order retained the earlier `0.34428` holdout
result, while reverse order produced `0.36614`, equal to local ranking. This
confirms that target regeneration is directional: it shifts information to
future blocks and cannot be treated as an order-independent score. A future
Hessian-pivoted or head-aware order is therefore more promising than further
damping tuning.

## Ninety-sixth sweep: Hessian pivot-order probe

The order control now also accepts `--ldlq-order pivot`. This probe sorts the
ASTC-width blocks within each row by a simple Hessian connectivity score (the
sum of absolute Gram-matrix entries for the block's input columns). It keeps
the candidate pool, footprint, calibration trace, and reconstruction code
unchanged; only the directional target-regeneration order changes.

On the SmolLM2 32x64 crop with the per-block 6x6 pool, the results were:

| Tensor | Order | Calibration | Holdout |
| --- | --- | ---: | ---: |
| `attn_q` | forward | 0.29496 | **0.34428** |
| `attn_q` | reverse | 0.32263 | 0.34564 |
| `attn_q` | pivot | 0.30718 | 0.36120 |
| `attn_k` | forward | 0.25548 | **0.25469** |
| `attn_k` | pivot | 0.25236 | 0.26147 |

The pivot heuristic improves calibration over reverse on `attn_q`, but loses
the forward holdout result on both tensors. This is a useful negative control:
Hessian magnitude alone is not a sufficient pivot criterion for ASTC target
regeneration. The spatial/block direction and the calibration distribution
still dominate. Keep forward order as the reference, retain pivot as a
regression control, and do not claim a general order improvement.

The next experiment is therefore a clean three-way selector comparison on the
same legal candidate snapshots: local neural ranking, forward Block-LDLQ, and
an improved Hessian-feedback/target-regeneration variant. It must be run over
the 4x4, 5x5, and 6x6 footprints, both calibration and fixed holdout traces,
and at least the `attn_q`/`attn_k` tensor pair before any GPU performance
interpretation.
