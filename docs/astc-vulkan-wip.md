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

## Test sweep policy

After each implementation sweep:

1. run `git diff --check`;
2. configure/build with `LLAMA_ASTC_VULKAN_POC=ON`;
3. run the focused `astc` CTest label;
4. run broader relevant backend tests when the build permits;
5. revisit `astc-vulkan-plan.md` and update this log with completed phases,
   failed hypotheses, and new constraints;
6. do not claim GPU support or speedup until a real device benchmark exists.

## Next sweep

1. Test block-local tensor permutations and channel assignments against the
   elementwise and dot-product objective.
2. Expand the benchmark matrix and repeat count enough to report stable
   distributions across image sizes and workgroup shapes.
3. Keep encoder availability separate from Vulkan runtime tests and do not alter
   any ggml tensor path until weight errors are materially reduced.

## Open questions

- Which ASTC image formats are actually sampled in compute on each target GPU?
- How should rows, columns, channels, and reduction tiles map to 2D image
  coordinates for cache locality?
- Is a generic ASTC encoder numerically adequate, or is a neural-weight-aware
  encoder required from the start?
- Does 6x6 reduce actual DRAM traffic enough to offset texture-pipeline latency?
- Can a mixed 4x4/6x6 policy improve quality/performance over either global
  choice?
