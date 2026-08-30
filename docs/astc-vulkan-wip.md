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

1. Run the new shader-device smoke on the Intel/target GPU and capture decoded
   values for both block sizes.
2. Add timestamp-query measurements for sequential versus non-local fetches.
3. Record device name, driver version, shader workgroup shape, elapsed GPU time,
   and represented bytes.
4. Revisit the plan after numerical and bandwidth evidence, before designing a
   weight packer or changing any ggml tensor path.

## Open questions

- Which ASTC image formats are actually sampled in compute on each target GPU?
- How should rows, columns, channels, and reduction tiles map to 2D image
  coordinates for cache locality?
- Is a generic ASTC encoder numerically adequate, or is a neural-weight-aware
  encoder required from the start?
- Does 6x6 reduce actual DRAM traffic enough to offset texture-pipeline latency?
- Can a mixed 4x4/6x6 policy improve quality/performance over either global
  choice?
