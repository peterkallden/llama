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

## Forty-third sweep: matched tiny-model baseline

To make the next model-level comparison reproducible, the validation fixture is
the `shibatch/tinybpe1m` family. It provides the same tokenizer, architecture,
and training source as F16, Q4_0, TQ1_0, and TQ2_0 GGUF files. The files are
kept outside the repository under `/tmp/llama-astc-models/tinybpe1m`; the
repository only commits the short evaluation corpus and records the source
repository, not a model binary.

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
predicted tokens, the variants produced:

| Variant | Deterministic continuation |
|---|---|
| F16 | `ly. Every day, Jerry would go` |
| TQ1_0 | `low. Every day, Jerry would g` |
| TQ2_0 | `low. Every day, Jerry would g` |
| Q4_0 | `ed. Every day, Jerry would go` |

On the committed 256-token-context evaluation corpus, one perplexity chunk
gave F16 `44.1675`, TQ1_0 `52.1643`, TQ2_0 `52.1643`, and Q4_0 `53.3438`.
These numbers have a large uncertainty because the tiny model and one chunk
are only a smoke baseline; they are not a quality claim.

Running `astc-vulkan-quality-smoke` on the same model's F16
`blk.0.attn_q.weight` (128x128) measured Q4_0 activation-relative MSE
`0.00709`. The existing ASTC 4x4 and 6x6 block-affine paths with a 1% sparse
sidecar reached `0.25143` and `0.45726` respectively before activation-aware
selection (`0.13744` and `0.26716` after selection). The native ASTC-Q smoke
also reported 6x6 resident rate `3.78125` bits/weight after edge padding, with
level-16 element MSE `0.000464` versus level-3 `0.005926`.

The immediate conclusion is that F16 is required as the same-source teacher
for a fair ASTC comparison, while TQ/Q4 are useful controls. A complete
model-level ASTC result still requires a loader/runtime path that can fetch
ASTC-resident weights during inference; the current measurements are the
weight and matvec gates that must pass before changing the GGUF format.

## Forty-fourth sweep: TQ1 versus TQ2 behavior

The TQ controls were run with four additional fixed prompts, temperature 0,
seed 42, and 12 generated tokens. TQ1_0 and TQ2_0 produced the same
continuation for every prompt in this smoke set, and both produced PPL
`52.1643` on the one-chunk corpus. This is expected behavior rather than
evidence that the formats are interchangeable in general: both formats encode
the same ternary alphabet and can differ only when quantization or conversion
pipelines choose different values.

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
