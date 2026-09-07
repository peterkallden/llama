# ASTC Vulkan neural-weight PoC

This directory contains an isolated research implementation for using standard
ASTC texture blocks as compressed neural-weight storage in Vulkan.  It is kept
outside `ggml` core so that the experiment can evolve without changing the
normal llama.cpp Vulkan backend or its supported GGUF quantization formats.

The runtime principle is deliberately narrow:

```text
offline encoder -> ordinary 16-byte ASTC blocks -> Vulkan texture fetch
             -> GPU fixed-function ASTC decode -> small shader reconstruction
```

The shader does **not** implement ASTC decompression.  A Vulkan driver that
supports the selected ASTC LDR format performs that part in texture hardware.
The work here decides which legal, standard ASTC blocks to write and how to
interpret the decoded channels as weights.

Cache artifacts are approved when they are built and recorded in the
manifest/evidence. At runtime, ordinary selection admits only artifacts that
passed their model and Vulkan gates; an explicit experimental/research launch
may additionally admit artifacts whose quality evidence is incomplete. A
cache miss always falls back to the native GGUF tensor path.

The current llama FFN bridge can execute both D1 and paired-D2 artifacts through
the same opt-in provider interface.  Admission, artifact evidence and the
semantic decoder still remain representation-specific.

Example using an existing cache (no cache generation):

```text
llama-cli -m model-Q4_K_M.gguf -ngl 99 \
  --astc-cache model-Q4_K_M.gguf.astc-vulkan.d1-6x6-50m \
  --astc-profile balanced
```

An explicitly experimental artifact requires the opt-in flag:

```text
llama-cli -m model-Q4_K_M.gguf -ngl 99 \
  --astc-cache model-Q4_K_M.gguf.astc-vulkan.d1-6x6-50m \
  --astc-profile balanced --astc-research
```

The runtime integration is deliberately split into three layers:

```text
immutable cache + evidence -> runtime overlay -> FFN provider -> backend dispatch
```

`astc-vulkan-runtime-overlay.*` owns cache validation, artifact policy and
page residency. The new llama FFN-provider bridge owns the graph substitution:
it preserves the ordinary GGUF matmul unless a layer reports an already-ready,
resident artifact. Finally, a backend-specific provider performs D1/D2
dispatch. This separation prevents a cache miss, a memory-budget denial or an
unsupported ASTC format from changing model execution: each becomes a normal
native-GGUF fallback.

The initial provider bridge uses a generic CPU custom-op boundary so it can be
verified independently. It is not the final performance path. The production
step is to implement the same provider contract inside ggml-vulkan with a
shared atlas/device owner; the current tensor-local sidecar must not be
instantiated once per model layer.

The integration seam is a generic, opt-in Vulkan external-operation interface,
versioned as `ggml.vulkan.external_op.v1`. `ggml-vulkan` knows only how to
lend the current device, command buffer and tensor-buffer views to a registered
dispatcher; it has no ASTC-specific types or lifecycle. The ASTC provider owns
its node registry, ASTC images, descriptors, pages and cache resources, and
installs its dispatcher only when the opt-in provider is enabled. Until those
resources are created on the same ggml-owned device, the provider remains on
the verified CPU custom-op path. A failed or unavailable native capability
must therefore leave the normal GGUF fallback untouched.

The provider installs the dispatcher only while its opt-in lifetime is active.
Borrowed `VkDevice`, `VkQueue` and `VkPhysicalDevice` handles are never
destroyed by ASTC; ASTC owns only its images, descriptors, pages and related
resources. The current provider still uses the verified CPU custom-op bridge
until native resources can be created on the same ggml-owned device.

The native path is now available for both D1 and D2: their dispatch sessions
bind ASTC's sampled image plus ASTC-owned layout metadata, borrow the
activation/output `VkBuffer` views supplied by ggml, and record compute work
into an already-open command buffer. These methods never submit, wait or map
memory. Graph binding is still opt-in through the ASTC provider. A generic
dispatch-time device preflight declines the binding when the actual ggml
command-buffer device does not match the borrowed device, preserving the
normal GGUF/custom-op fallback.

The device smoke now verifies this native-recording contract for both D1 and
D2 with real external activation/output buffers. It submits the caller-owned
command buffer, reads the result back, and compares it with the ordinary
session path. This is a command-recording/device-lifecycle gate only; it does
not yet enable llama graph-node binding. For paired D2, the smoke deliberately
uses the physical storage height (the logical height is twice the ASTC image
height for the five-row layout), keeping that geometry explicit at the test
boundary.

The model-level cache planner sits above the D1/D2 encoders. It first filters
already validated tensor artifacts by model/Vulkan evidence, then may apply
workload usage and measured-benefit metrics, followed by the existing memory
residency planner. It does not encode at runtime. Runtime GGUF families are
not hardcoded to Q4: an F16/BF16-derived cache can be admitted for Q4, Q3,
TQ2, TQ1 or another structurally matching family, but each runtime base still
requires its own model/Vulkan replay gate.

Storage-page planning is separate from artifact selection. Pages are grouped
by ASTC footprint, semantic decoder, normalization contract (including paired
layout and row-scale presence); encoder profile is provenance only. The current runtime
keeps tensor-local streamable bands. Cross-tensor atlas packing is a later
storage optimization and does not change the D1/D2 shader ABI.

## Offline GPU-encoder experiment

`gpu-encoder/` is a separate offline backend. It does not alter the cache
format, scheduler, runtime shader ABI, or ordinary inference path. The cache
generator can use it explicitly (or use the default `hybrid` mode) for physical
full-image encodes:

```text
D1 scalar source blocks
  -> GPU physical ASTC proposer
  -> CPU astcenc exact finisher / legality oracle
  -> ordinary 16-byte ASTC payload
  -> CPU + Vulkan fixed-function decode verification
```

The directory boundary is intentional:

| Directory | Owns | Does not own |
|---|---|---|
| `gpu-encoder/shared/` | physical source texels, request/batch contracts, symbolic proposal descriptors, selector-delta hand-off and future Vulkan session | D1 gauge or any D2 semantic decision |
| `gpu-encoder/d1/` | one-weight-per-texel scalar source construction; later D1 gauge families | ASTC BISE packing or paired-row semantics |
| `gpu-encoder/d2/` | reserved for paired layouts, D2-LA, scales, prediction and layout metadata | shared codec logic or D1 special cases |

The current code supplies reusable Vulkan compute sessions and minimal D1
`4x4`, `5x5`, `6x6`, and `8x6` proposers, alongside the deterministic CPU
proposal reference. A proposer emits a bounded scalar-plus-gauge candidate
bank; the CPU finisher creates legal payloads, validates them with
`astcenc_get_block_info`, and performs exact CPU decode. The next local ranking
stage scores those *decoded* blocks in activation space, retaining scalar as a
mandatory fallback and preserving additional alternatives for later
conflict-aware selection.
`astcenc` remains both the legal payload reference and the fallback throughout
v1; no custom ASTC runtime decoder is planned.

The latent generator accepts `--backend hybrid|cpu` (the default is
`hybrid`) and `--gpu-proposer-shader path`. Hybrid means GPU proposal
generation followed by the exact CPU `astcenc` finisher; `--backend cpu`
forces the reference path. A shader and an explicit matching `--footprint` are
required before hybrid is enabled. Missing Vulkan support, an unsupported
shape, or a proposer failure falls back to CPU with an informational message.
The D2 frontend now has its own representation-specific hybrid seam:
`direct-neutral`, `direct-steered`, and D2-LA candidates are finished by the
same CPU oracle and ranked only after exact paired decode. This remains an
offline candidate-bank path; cache generation and scheduling still require
their separate artifact, model, and Vulkan gates.

D1 now exposes the matching frontend seam for scalar and gauge-L+A candidates.
Both frontends hand exact decoded payloads to the same global selector delta
format; only their decoded-weight reconstruction differs. This lets the GPU
proposer remain a backend choice while CPU and hybrid runs share commit order,
validation-prefix semantics, and artifact payload handling.
The representation-neutral selector adapter in
`gpu-encoder/shared/astc-gpu-selector-adapter.*` owns the common trace
validation, mandatory zero-baseline construction, delta-shape checks, and
neutral-candidate ordering. D1 and D2 retain their own decoded semantic
reconstruction and physical geometry; no paired-row assumptions are moved into
the shared layer.

D2 has a separate initial `8x5` frontend. It owns paired logical-row source
construction and a per-physical-block layout map; the shared GPU proposer sees
only ordinary RGBA texels, and the shared CPU finisher only sees physical ASTC
blocks. The D2 hybrid bridge now carries its exact decoded candidates into the
existing global conflict-aware selector as activation-space deltas. This is a
transport/verification seam, not a D2 quality or scheduler promotion: D2's
artifact/model gates remain separate.

Build and run the contract smoke from a configured build directory:

```bash
cmake --build build-astc-neural-rank --target test-astc-vulkan-gpu-encoder -j4
ctest --test-dir build-astc-neural-rank -R test-astc-vulkan-gpu-encoder --output-on-failure
```

## Representations

Every ASTC block is exactly 128 bits (16 bytes).  The physical ASTC footprint
therefore determines the storage rate.  `b/w` below means physical bits per
logical neural weight, before the tiny D2 layout map and cache manifest.

| Profile | Representation | ASTC footprint | Rate | Status | Intended use |
|---|---|---:|---:|---|---|
| `d1-4x4` | D1 scalar | 4x4 | 8.000 b/w | standard | High-fidelity comparison point |
| `d1-5x5` | D1 scalar | 5x5 | 5.120 b/w | standard | Balanced D1 point |
| `d1-6x6` | D1 scalar or gauge-L+A | 6x6 | 3.556 b/w | standard | Main D1 operating point |
| `d1-8x6` | D1 gauge-L+A | 8x6 | 2.667 b/w | experimental | Low-rate D1, six-row family |
| `d1-10x6` | D1 gauge-L+A | 10x6 | 2.133 b/w | experimental | Lower-rate six-row family |
| `d1-8x8` | D1 gauge-L+A | 8x8 | 2.000 b/w | experimental | Extreme-rate D1 |
| `d1-10x8` | D1 gauge-L+A | 10x8 | 1.600 b/w | experimental | Extreme-rate D1 |
| `d2-6x5` | D2 paired | 6x5 | 2.133 b/w | experimental | Higher-quality five-row paired D2 |
| `d2-8x5` | D2 paired | 8x5 | 1.600 b/w | experimental | Two logical output rows per texel |
| `d2-10x5` | D2 paired | 10x5 | 1.280 b/w | experimental | Lower-rate paired D2 |

`6x5`, `8x5`, `8x6`, `10x5`, `10x6`, `8x8`, and `10x8` are standard ASTC LDR
formats.  They remain opt-in here because this PoC still needs wider
device-quality and model-replay evidence.  Runtime also checks the selected
Vulkan format capability rather than assuming that every device accepts it.

### Choosing a starting point

These profiles are an ASTC storage/decoder choice, not a replacement label for
the input GGUF quantization.  In particular, `Q4_K_M` describes the existing
GGUF's native weight representation; `d1-6x6` describes the ASTC sidecar's
physical storage rate and semantic decoder.  Re-encoding an already-quantized
GGUF compounds its source quantization error with ASTC error.  When quality is
important, use an FP16/BF16 tensor from the same model as the offline source
and compare the exported ASTC artifact against Q4/Q3/TQ controls.

| Goal / starting model | Recommended first choice | Why | Important caveat |
|---|---|---|---|
| Establish an ASTC quality ceiling from FP16/BF16 | `d1-4x4` scalar, then gauge-L+A | 8 b/w gives ASTC the most local freedom and is the cleanest decoder/device correctness control | It saves little versus ordinary low-bit GGUF formats; use it for quality evidence, not maximum compression |
| FP16/BF16 with a balanced ASTC trial | `d1-5x5` scalar/gauge | 5.12 b/w is the intermediate rate-quality point | Still needs the same artifact replay and model validation as every other profile |
| FP16/BF16 with the main practical ASTC experiment | `d1-6x6` scalar-anchored gauge, with scalar fallback | 3.556 b/w is the established D1 operating point and sits between Q3 and Q4 density | This is the recommended first ASTC cache to build, but it is not yet a drop-in quality replacement for Q4_K_M |
| Existing `Q4_K_M` model | Keep native Q4_K_M as the normal inference baseline; evaluate `d1-6x6` from the matching FP16 source | Q4_K_M is the current practical baseline and hard fallback; 6x6 is the fair main ASTC comparison | Avoid presenting a cache encoded only from Q4_K_M as FP16-equivalent; it is useful for source-robustness experiments, not the primary quality oracle |
| Existing `Q3_K_M` model | Compare `d1-6x6`; optionally screen `d1-8x6` | Q3_K_M is the closest conventional low-bit control for the 3.56 and 2.67 b/w D1 rungs | `d1-8x6` is experimental and must pass validation-stopped artifact/model replay first |
| Ternary/few-level source (`TQ1_0`, `TQ2_0`, or a true few-level projection) | `d1-8x6` first, then `d1-8x8` only as a research rung | These rates are meaningful low-bit controls and gauge can search legal ASTC reconstructions above the source field | TQ1/TQ2 behavior must be verified per real tensor; a bounded crop is not a model-quality claim |
| About 1.6 b/w, willing to use paired rows | `d2-8x5` | D2 stores two logical rows per texel at the same physical ASTC payload cost, with paired dispatch and a layout map | Experimental; require explicit experimental-profile selection in the cache, capability checks, and a D2 model-quality result |
| About 2.13 b/w, willing to use paired rows | `d2-6x5` | Keeps D2's ten-logical-row stripe while giving the ASTC block more bits per paired weight | Experimental; intended as the next D2 quality rung before `d2-8x5` |
| About 1.28 b/w | `d2-10x5` | Lowest currently implemented D2 rate | Research only. It is a rate-distortion experiment, not an automatic choice for a normal model |
| Below 2 b/w without D2 | `d1-8x8` or `d1-10x8` only for controlled research | Provides a standard-ASTC low-rate ladder | These profiles are intentionally opt-in until full-tensor/model evidence demonstrates an acceptable frontier |

In short: for a normal **Q4_K_M** model, keep Q4_K_M as the reliable path and
start an ASTC investigation with an FP16-derived `d1-6x6` artifact.  Use
`d1-4x4` if the immediate question is decoder fidelity, `d1-5x5` when a
middle point is useful, and lower-rate D1/D2 profiles only after the same
tensor has a validation-selected artifact and model-level comparison.

### D1: one logical weight per texel

D1 maps one neural weight to one decoded ASTC texel.  Two semantic decoders
are available:

- **Scalar** stores one normalised weight field; decoded channels reconstruct
  the same scalar weight.
- **Gauge L+A** uses a luminance/alpha-style semantic decoder.  Its offline
  source can use the null-space family

  ```text
  L = q + delta
  A = q - delta
  (L + A) / 2 = q
  ```

  before encoding.  `delta` does not alter the intended source weight.  It
  steers the discrete ASTC encoder toward a different legal payload, and the
  actual decoded result is ranked after exact ASTC round-trip.

The practical D1 pipeline is scalar-anchored: scalar remains a mandatory
fallback candidate, while gauge candidates may be selected only when they
improve the relevant offline objective.

### D2: two logical weights per texel

D2 trades independent per-texel channels for density.  A physical `W x H`
ASTC image represents two logical output rows per texel, hence:

```text
logical tensor coverage = W reduction columns x (2 * H) output rows
```

For the direct layouts, one logical weight is duplicated across two RGB lanes
and the other uses the remaining lane:

```text
RG/B:  R,G = q0; B = q1; A = steering
R/GB:  R = q0; G,B = q1; A = steering
```

The one-bit-per-physical-block layout map selects `RG/B` or `R/GB`.  D2's
shader reconstructs both logical rows together (`astc-paired-matvec.comp`) so
they share the same sampled texel and activation traversal.  The layout bit
is necessary metadata, but is tiny: it covers 60/80/100 logical weights for
`6x5`/`8x5`/`10x5`, respectively (about 0.0167/0.0125/0.0100 b/w before
container overhead).

The current D2 main path combines:

- semantic-balanced ASTC channel weights, so a duplicated lane does not
  accidentally receive twice the image-error budget;
- source-derived alpha steering, where alpha changes encoder decisions but is
  not a runtime semantic weight;
- the D2 layout map;
- an activation/sensitivity-aware selection objective.  YAQA-style diagonal
  output sensitivity can weight the two paired rows differently.

Common/difference transforms, explicit semantic dual-plane search, mixed
footprints, and full PV tuning remain research tracks rather than cache/runtime
defaults.  An optional offline row-pairing sweep is available for D2 8x5
chunked selection (`--row-pairing optimized`).  It chooses one pairing for
each ten-logical-row stripe from calibration/validation projections, keeps the
adjacent pairing as the control, and does not export a cache until the pair-map
  metadata contract is versioned.  The optional `--row-transform givens` bank
  adds a bounded 2x2 rotation sweep on top of pairing; it is research-only and
  remains disabled by default.  With calibration-only pairing selection on the
  current Qwen crop, adjacent, pairing-only, and pairing+Givens reached
  0.00094672, 0.00077082, and 0.00067761 holdout MSE respectively.  This is a
  promising result, but it still needs model replay and a versioned pair/
  transform-map artifact before it can affect exported caches.

### D2-LA and artifact variants

`D2-LA` is a separate paired semantic, not an encoder-profile alias. It maps
the two logical lanes to ASTC luminance and alpha:

```text
layout 0: RGB = q0, A = q1
layout 1: RGB = q1, A = q0
```

The paired shader reconstructs luminance from RGB and takes the other logical
lane from Alpha. It remains a standard ASTC texture fetch; no shader ASTC
decoder is introduced. On the current Pythia gates D2-LA is promising but
still experimental. In particular, whether `neutral` or
`validation-selected` is better is tensor-specific, and absmax-normalized
artifacts are eligible only after their own model/Vulkan replay.

Manifest v4 allows a cache to retain these alternatives simultaneously. Each
artifact has an immutable payload/layout range plus its semantic,
normalization, variant, provenance, and evidence. `row-scales.bin` is optional
and independently hash-verified; the current research implementation stores
one F32 scale per output row and applies it after the reduction. Scheduler
selection therefore chooses a prevalidated artifact per tensor – it never
re-runs an encoder or guesses a codec configuration during inference.

### Lightweight scheduler profiles

The artifact policy accepts `quality`, `balanced`, `compact`, `speed`, and
`auto`. The older spelling `size` remains a backward-compatible alias. These names only rank
already validated cache artifacts:

| Profile | Ranking rule in this PoC |
|---|---|
| `quality` | Lowest model loss delta, then logits error |
| `balanced` | Evidence-first quality ordering |
| `compact` | Lowest bits/weight, then model loss |
| `speed` | Deterministic quality ordering until device timings are recorded |
| `auto` | Deterministic quality ordering until a combined policy exists |

This is intentionally a small foundation. It does not encode at runtime,
inspect weights, or assume that a lower bitrate is faster. Device timing,
memory limits, and tensor-specific model evidence remain separate eligibility
gates.

## Offline selection algorithms

The offline encoder is more than a conventional image compressor.  It creates
legal standard-ASTC candidates, decodes them exactly, scores their neural
effect, and writes the winning 16-byte payloads.  The following components are
used or available in this directory.

| Component | Role | Default/use |
|---|---|---|
| Stock `astcenc` candidate search | Standard image-oriented reference encoder | Required reference and legal payload oracle |
| Neural-rank `astcenc` fork | Retains/ranks a broader legal candidate set using neural information | Offline experimental encoder, most relevant to low-rate D1/D2 studies |
| Exact ASTC round-trip | Decode each candidate before measuring it | Required; source-space MSE is not the oracle |
| Conflict-aware selection | Accepts candidate changes against the current activation residual | Main selector; avoids summing locally-good but correlated corrections |
| Validation-prefix stopping | Chooses the export prefix using validation traces, leaving holdout untouched | Required artifact-selection contract |
| D1 GPU prescreen | Filters/provisions promising D1 blocks before expensive CPU ASTC search | Optional offline acceleration |
| D2 GPU prescreen | Shortlists paired-D2 representation profiles with calibration-only activation energy before CPU ASTC search | Optional offline acceleration; preserves semantic/scale/Alpha family coverage |
| YAQA score/replay | Provides a lightweight two-sided/sensitivity-aware candidate score | Optional offline ranking input; D2 can use it by configuration |
| PV-lite codebook | Small, structured steering family | Optional; preferred over expensive full PV as the normal experiment |
| Full PV alternation | More exhaustive codec-aware optimization | Optional research-only path; deliberately not a default |
| Block-LDLQ / target regeneration | Hessian-style error propagation over candidate choices | Research path; not required to replay an artifact |

The selector contract is always:

```text
calibration traces -> proposal/commit sequence -> validation selected prefix
                                                   -> untouched holdout report
```

Only the payload at the selected prefix may be packed into an artifact.  A
full calibration commit sequence is diagnostic data, not automatically the
deployed result.

## Build prerequisites and CMake configuration

Required baseline dependencies:

- a C++17 compiler and CMake;
- Vulkan headers and loader; `glslc` is needed for shader targets;
- a Vulkan device/driver supporting the desired ASTC LDR texture format for
  device replay;
- an `astcenc` package exposing `astcenc::astcenc-static` for ASTC encoder,
  decode, and artifact tools.

Minimal isolated PoC configuration:

```bash
cmake -S . -B build-astc \
  -DGGML_VULKAN=ON \
  -DLLAMA_ASTC_VULKAN_POC=ON \
  -DGGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER=ON

cmake --build build-astc --parallel 2
```

If CMake cannot discover the system `astcenc` package automatically, point it
at the directory containing `astcencConfig.cmake`:

```bash
cmake -S . -B build-astc \
  -DGGML_VULKAN=ON \
  -DLLAMA_ASTC_VULKAN_POC=ON \
  -DGGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER=ON \
  -Dastcenc_DIR=/absolute/path/to/lib/cmake/astcenc
```

The optional neural-rank build uses an **isolated** `astcenc` fork.  This keeps
research-only candidate-ranking changes separate from the normal package and
from the Vulkan runtime:

```bash
cmake -S . -B build-astc-neural \
  -DGGML_VULKAN=ON \
  -DLLAMA_ASTC_VULKAN_POC=ON \
  -DGGML_VK_ASTC_EXPERIMENTAL_SCHEDULER_ADAPTER=ON \
  -DGGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK=ON \
  -DGGML_VK_ASTC_NEURAL_RANK_LIBRARY=/absolute/path/to/libastcenc.a \
  -DGGML_VK_ASTC_NEURAL_RANK_INCLUDE_DIR=/absolute/path/to/astcenc/include
```

To register the optional GPU candidate-ranking device smoke, also set:

```bash
-DGGML_VK_ASTC_EXPERIMENTAL_GPU_RANKING_DEVICE_SMOKE=ON
```

The normal llama.cpp Vulkan backend remains controlled by `GGML_VULKAN`; none
of the ASTC flags changes its regular quantized tensor path.

## Building and publishing a cache

### Enklaste användarflödet

Tänk på en ASTC-cache som en färdig, offline-byggd sidofil. Du behöver normalt
bara känna till tre saker:

1. `build` skapar cachen från en modell och ett activation-trace.
2. `verify` kontrollerar att cachen är hel och hör till modellen.
3. `inspect` visar representation, footprint, bitrate och evidensstatus.

Runtime startar inte encoding automatiskt. Om cachen saknas används den vanliga
GGUF-vägen.

För en cache som byggs direkt från den modell som ska köras:

```bash
build-astc-neural/bin/astc-vulkan-cache build \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/activations.trace \
  --footprint 6x6 --representation scalar --backend hybrid \
  --cache auto

build-astc-neural/bin/astc-vulkan-cache verify \
  --model /absolute/path/model.gguf --cache auto

build-astc-neural/bin/astc-vulkan-cache inspect \
  --model /absolute/path/model.gguf --cache auto
```

`hybrid` är standard: GPU-proposer där den finns och CPU-astcenc som exakt
finisher. Använd `--backend cpu` när du vill ha en ren CPU-reference.

### F16-källa med Q4/Q3 som runtime-bas

Det här är ett separat, avancerat flöde. Cachen byggs en gång från F16/BF16
och binds sedan strukturellt till en Q4- eller Q3-GGUF av samma logiska modell:

```bash
build-astc-neural/bin/astc-vulkan-cache bind \
  --source-model /absolute/path/model-f16.gguf \
  --runtime-model /absolute/path/model-q4_k_m.gguf \
  --family q4_k_m \
  --cache /absolute/path/model-f16.gguf.astc-vulkan
```

`bind` kontrollerar modellens arkitektur, tensornamn och former. Det är inte ett
kvalitetsgodkännande; Q4/Q3-basens model-replay måste fortfarande passera innan
schedulern får välja artefakten automatiskt. Det äldre kommandot `admit-base`
finns kvar som alias.

An ASTC cache is deliberately an **offline** sidecar.  Model loading validates
and consumes an already-built cache; it never starts a long ASTC encode job
just-in-time.

The offline flow is:

```text
GGUF/FP16 tensor + representative activation traces
  -> D1 or D2 candidate generator and selector
  -> selected ASTC payload + manifest (+ D2 layout map)
  -> artifact validation/replay
  -> atomic cache publish beside the GGUF or at an explicit cache root
```

The selection programs produce raw payload and report files.  The artifact
pack/decode utilities turn the selected stream into the cache contract.  A
representative D2 selection command is:

```bash
build-astc-neural/bin/astc-vulkan-paired-selection-smoke-neural \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/input.trace \
  --rows 2048 --columns 8192 \
  --calibration-samples 8 --validation-samples 7 \
  --channel-weights balanced-a025 --source-derived-alpha 1 \
  --row-strip-chunked 1 \
  --report /tmp/d2-8x5/report.txt \
  --export-payload /tmp/d2-8x5/payload.raw \
  --export-layout /tmp/d2-8x5/layout.raw
```

`--export-neutral 1` is available only for controlled profile studies. It
exports the scalar-anchored neutral candidate through the same artifact path
instead of the validation-selected stream, and labels the report accordingly.
It is useful for a fair neutral/selected replay matrix; normal cache artifacts
must continue to export the validation-selected stream.

`D2_6x5` uses the same arguments and artifact layout as `D2_8x5`; select its
dedicated target when preparing its matrix:

```bash
build-astc-neural/bin/astc-vulkan-paired-selection-smoke-neural-6x5 \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/input.trace \
  --rows 2048 --columns 8192 \
  --calibration-samples 8 --validation-samples 7 \
  --channel-weights balanced-a025 --source-derived-alpha 1 \
  --row-strip-chunked 1 \
  --report /tmp/d2-6x5/report.txt \
  --export-payload /tmp/d2-6x5/payload.raw \
  --export-layout /tmp/d2-6x5/layout.raw
```

Run the same cache pack, byte replay, CPU/Vulkan oracle, and prompt-matched
model replay gates as `D2_8x5`; the new target alone is not evidence that the
profile is scheduler-eligible.

Use `astc-vulkan-latent-smoke --help` for the corresponding D1 generator
options.  It supports scalar-anchored and gauge-L+A candidate generation.
For a small physical hybrid smoke (using the generated 4x4 proposer shader):

```bash
build-astc/bin/astc-vulkan-latent-smoke \
  --backend hybrid --footprint 4x4 \
  --gpu-proposer-shader build-astc/pocs/astc-vulkan/astc-gpu-d1-proposer.comp.spv \
  --max-rows 8 --max-columns 16
```

Use `--backend cpu` for a byte-for-byte reference run. The hybrid path is an
offline producer; runtime still samples ordinary standard ASTC blocks.
After packing a conventional artifact directory, publish it with the cache
tool:

```bash
# Lists stable profile names, physical format, rate, and experimental state.
build-astc/bin/astc-vulkan-cache profiles

# artifact-dir must contain manifest.astcv and payload.astcpack; D2 also has
# layout-map.bin. provenance.txt is optional but recommended.
# Publish an already generated artifact directory atomically.
build-astc/bin/astc-vulkan-cache publish \
  --model /absolute/path/model.gguf \
  --artifact-dir /absolute/path/artifact-dir \
  --storage-profile d2-8x5 \
  --cache auto

# Verify source hashes, manifest/payload checksums, and every tensor record.
build-astc/bin/astc-vulkan-cache verify \
  --model /absolute/path/model.gguf --cache auto

# Use `inspect` when the individual tensor/artifact evidence should also be printed.
build-astc/bin/astc-vulkan-cache inspect \
  --model /absolute/path/model.gguf --cache auto
```

### Registering a quantized runtime base

An ASTC artifact is normally generated from the highest-quality available
source (usually F16/BF16). A Q4 or Q3 GGUF of the same model revision can be
registered as a *structurally compatible runtime base* without encoding the
ASTC payload again:

```bash
build-astc/bin/astc-vulkan-cache admit-base \
  --source-model /absolute/path/model-f16.gguf \
  --model /absolute/path/model-q4_k_m.gguf \
  --family q4_k_m \
  --cache /absolute/path/model-f16.gguf.astc-vulkan
```

The command requires the exact original source GGUF, validates the cache
against it, and then checks that the runtime GGUF has the same architecture,
tensor names, and tensor shapes. It writes a small provenance record below
`compatible-bases/`; it does **not** copy or re-encode payload data.

Admission is intentionally only a structural gate. It records no model or
Vulkan quality evidence and does not make the runtime cache auto-selectable.
That keeps a F16-derived cache from silently being used on a Q3/Q4 base whose
activation distribution has not yet passed its own replay gate.

`--cache auto` creates/uses a sibling directory derived from the GGUF name.
An explicit path is supported when the cache belongs on another filesystem:

```bash
--cache /absolute/path/to/model.gguf.astc-vulkan
```

For portable model-level replay, use the normal `llama-astc-replay` binary
(the older `astc-vulkan-model-replay-smoke` name remains a regression alias):

```bash
build-astc-neural/bin/llama-astc-replay \
  --model /absolute/path/model-f16.gguf \
  --cache /absolute/path/model-f16.gguf.astc-vulkan \
  --tensor blk.0.ffn_down.weight \
  --activations /absolute/path/activations.trace \
  --layer 0 --width 8192 --height 2048 \
  --prompt-file /absolute/path/replay-prompts.txt \
  --gpu-shader build-astc-neural/pocs/astc-vulkan/astc-paired-matvec.comp.spv \
  --streamed --research
```

`--prompt-file` supplies the same deterministic text corpus on every machine;
it is a plain UTF-8 text file and may be produced from a HuggingFace dataset
by an external preprocessing step. The activation trace must correspond to
the tokenized text. `--prompt` remains useful for a one-prompt smoke. The
replay tool emits model logits/loss/top-1 metrics and keeps the exact cache
payload, model hashes, and runtime backend explicit in the invocation.
Add `--source-model` when the cache was generated from F16/BF16 and the
runtime model is Q4/Q3; the tool verifies the previously created
`admit-base` record before replay. `--evidence evidence.json` writes a small
machine-readable result containing model/cache hashes and logits/loss/top-1
metrics.

For a small, reproducible D1 cache build, the cache tool now provides a
bounded orchestration command. It runs the existing latent exporter, packs a
v4 scalar artifact, and publishes it through the same atomic/hash-validated
path as `publish`. The command does not perform model replay and therefore
stores the artifact with `model_gate=false` and `vulkan_gate=false`; run the
replay gates and update/publish an evidence-bearing artifact before enabling
it in the production scheduler.

```bash
build-astc-neural/bin/astc-vulkan-cache build \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/calibration.trace \
  --footprint 6x6 \
  --backend hybrid \
  --gpu-proposer-shader build-astc-neural/pocs/astc-vulkan/astc-gpu-d1-proposer.comp.spv \
  --preset medium \
  --cache auto
```

`--backend cpu` forces the CPU `astcenc` reference path. `--artifact-dir`
keeps the generated manifest/payload/provenance for inspection instead of
using an automatically removed temporary staging directory. `--max-rows`
and `--max-columns` are available for bounded smoke builds. This first
version intentionally handles one D1 scalar tensor; D1 gauge and D2 will use
the same staging/publish contract after their model-level evidence is wired
into the builder.

The builder also has an explicit paired-D2 path for the supported `6x5`,
`8x5`, and `10x5` footprints. D2 requires logical dimensions because one
physical texel represents two output-row weights, and it exports a layout map
alongside the ASTC payload:

```bash
build-astc-neural/bin/astc-vulkan-cache build \
  --representation paired-d2 \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/calibration.trace \
  --footprint 8x5 --rows 2048 --columns 8192 \
  --calibration-samples 8 --validation-samples 7 \
  --paired-semantic la --channel-weights balanced-a025 \
  --source-derived-alpha 1 --row-scale none --workers 8 \
  --cache auto
```

`--row-scale absmax` additionally carries the per-output-row scale blob.
`--workers N` controls the number of CPU ASTC contexts used for parallel D2
block generation (default: 4); selection and commit ordering remain
deterministic after the worker phase. Use a value near the available physical
cores and leave headroom for the rest of the system.

Before spending CPU time on several exact D2 builds, the cache tool can screen
a small *profile bank* using only the calibration prefix of the trace. It is a
bounded, non-oracle planning phase: it preserves direct/L+A, unscaled/scaled,
and steering alternatives, then reports the profiles that should be sent to
the existing exact CPU finisher. It never changes a cache artifact by itself.

```bash
build-astc-neural/bin/astc-vulkan-cache d2-prescreen \
  --model /absolute/path/model.gguf \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/calibration.trace \
  --footprint 8x5 --rows 2048 --columns 8192 \
  --d2-prescreen gpu --d2-prescreen-top-k 3
```

`--d2-prescreen cpu` is the deterministic reference. `gpu` mirrors its cheap
proxy on a generic Vulkan compute device, checks the result against the CPU
calculation, and falls back to CPU if that check cannot pass. Exact `astcenc`
encode/decode, validation-prefix selection, and model replay remain mandatory
after the screen; this command merely avoids launching every expensive profile
blindly.

This D2 producer currently records model/Vulkan gates as false; its cache is
therefore suitable for replay and inspection but remains ineligible for
automatic production scheduling until those gates are established.

`create` is the lower-level equivalent when the artifact pieces are held in
separate paths:

```bash
build-astc/bin/astc-vulkan-cache create \
  --model /absolute/path/model.gguf \
  --manifest /absolute/path/manifest.astcv \
  --payload /absolute/path/payload.astcpack \
  --layout /absolute/path/layout-map.bin \
  --provenance /absolute/path/provenance.txt \
  --storage-profile d2-8x5 --cache auto
```

For D1, omit `--layout`.  Use `d1-6x6`, `d1-8x6`, and so forth as appropriate.
`publish`/`create` validate that the selected storage profile agrees with all
manifest records, then publish atomically. They do not generate ASTC blocks;
`install` and `--profile` remain compatibility aliases. The bounded D1
`build` command above is the first explicit producer path; it reuses the
existing generator and packer rather than duplicating ASTC encoding logic.

### Cache contents

For a GGUF named `model.gguf`, the automatic cache root is conceptually:

```text
model.gguf.astc-vulkan/
  manifest.astcv       binary KASTCVM1 v3 tensor contract
  payload.astcpack     concatenated standard 16-byte ASTC blocks
  layout-map.bin       D2 only: direct RG/B versus R/GB map
  provenance.txt       optional encoder/objective/trace provenance
  source.gguf.sha256   source identity
  manifest.sha256      integrity checksum
  payload.sha256       integrity checksum
  layout-map.sha256    D2 only, when a map exists
  compatible-bases/    optional structural Q3/Q4 runtime-base records
```

The manifest records tensor name, dimensions, ASTC footprint, representation,
payload byte ranges, decoder constants, validation-selected commit count, and
the deterministic edge/padding contract.  Artifact replay must consume these
bytes directly; it must not regenerate ASTC payloads.

## Runtime use today

The isolated scheduler adapter can resolve a matching cache and bind its
payload to the D1 or D2 sidecar dispatch.  The corresponding device smoke is
the current runtime entry point:

```bash
# D1: ordinary single-weight reconstruction.
build-astc/bin/astc-vulkan-scheduler-adapter-smoke \
  --model /absolute/path/model.gguf --cache auto \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/input.trace \
  --footprint 6x6 \
  --shader build-astc/pocs/astc-vulkan/astc-ffn-matvec.comp.spv

# D2 6x5 L+A: paired rows; the smoke deliberately enables experimental
# admission. A future llama-facing caller must make the same explicit opt-in.
build-astc/bin/astc-vulkan-scheduler-adapter-smoke \
  --model /absolute/path/model.gguf --cache auto \
  --tensor blk.0.ffn_down.weight \
  --trace /absolute/path/input.trace \
  --footprint 6x5 \
  --shader build-astc/pocs/astc-vulkan/astc-paired-matvec.comp.spv
```

The smoke passes `allow_experimental=true` to exercise the isolated D2 path.
The adapter itself admits D2 only when all of the following hold: the cache is
v4 and integrity-valid, the artifact is `paired-d2` with `6x5` or `8x5` L+A
semantic reconstruction, both model and Vulkan gates passed, and the embedding
caller explicitly enables experimental admission. D1 continues to use its
existing standard path. Any other paired artifact (`10x5`, direct-D2, legacy
cache, missing evidence, or disabled opt-in) falls back cleanly to the normal
Q4/Q3 route. The adapter never attempts runtime ASTC encoding.

## Usage-aware cache planning

Cache admission and atlas/page planning are offline operations above the D1/D2
artifact format. A small text metrics file keeps workload/device observations
out of the immutable manifest:

```text
# astc-usage-v1
# tensor invocations tokens native_bytes astc_bytes native_gpu_ns astc_gpu_ns path_probability execution_order
blk.0.ffn_down.weight 128 128 1048576 327680 4200 3500 1.0 0
```

The cache tool can then rank already validated artifacts, apply a device/host
budget, and split compatible storage pages without encoding or inspecting
weights at runtime:

```bash
build-astc/bin/astc-vulkan-cache plan \
  --model /absolute/path/model.gguf \
  --cache /absolute/path/model.gguf.astc-vulkan \
  --usage /absolute/path/usage.txt \
  --policy balanced --device-budget 4294967296 --page-bytes 67108864
```

`--require-usage 1` makes missing tensor observations fall back to native Q;
`--require-benefit 1` does the same when measured timing/traffic shows no
positive ASTC benefit. Pages never mix footprint, D1/D2 semantic decoder,
normalization, paired layout, or row-scale requirements. The command reports
the chosen ordering and page sizes; a later Vulkan owner remains responsible
for image allocation, streaming and atlas addressing. Experimental artifacts
remain excluded unless `--allow-experimental 1` is explicitly supplied;
`--allow-unverified 1` is reserved for offline/research replay.

The runtime page owner consumes the resulting page list without changing the
cache format. It validates D1/D2 storage classes, assigns deterministic
runtime-local slots, and exposes resident/fallback resolution for a tensor.
Its first mode is static residency: the planner-selected prefix is loaded for
the model run. Lifecycle states already include upload and eviction so a
future asynchronous Vulkan owner can be added without changing scheduler or
manifest contracts. Actual image allocation and upload continue to use the
existing `astc_vulkan_tensor_session` and stream-loader paths.
For a resident entry, the owner can also load one validated material range
(payload plus optional D2 layout and row-scales) into plain buffers. A
non-resident entry returns an explicit native fallback, so the scheduler never
uploads an unadmitted page.
The page-owner contract test covers both scalar D1 and paired-D2 material;
the device smoke forwards resident D1 material through the existing sidecar
upload path and reports whether a compatible Vulkan device was available.

Before building artifacts, use the discovery phase to avoid encoding tensors
that are never used or do not justify their storage cost:

```bash
build-astc/bin/astc-vulkan-cache discover \
  --source-model /absolute/path/source-f16.gguf \
  --usage /absolute/path/usage.txt \
  --output /absolute/path/discovery.tsv \
  --footprint 8x5 --representation paired-d2 \
  --max-cache-bytes 1073741824
```

This produces a ranked tensor shortlist before any ASTC payload is generated.
The shortlist is a build hint, not quality evidence: selected entries still
need artifact generation and model/Vulkan replay. The existing `plan` command
remains the post-build residency selector; it does not replace discovery.
For a dense-model first pass, `--usage auto` synthesizes a conservative
tensor-order baseline without requiring a pre-existing cache. A real usage
file is still preferred when routing, batching or tensor heat differs from
that static assumption.

For a simpler user-facing invocation, select a high-level profile instead of
choosing D1/D2 explicitly:

```bash
build-astc/bin/astc-vulkan-cache discover \
  --source-model /absolute/path/source-f16.gguf \
  --usage auto --output /absolute/path/discovery.tsv \
  --profile balanced
```

These profiles are conservative first-pass discovery defaults, not approval
claims: `quality` starts at D1 4x4 scalar, `balanced` at D1 6x6 scalar,
`compact` at D2 8x5 paired, and `speed`/`auto` at D1 6x6 scalar. Explicit
`--footprint` and `--representation` override those defaults for research
and reproducible comparisons. The resulting shortlist still requires the
normal artifact, model-replay, Vulkan, and memory gates. For post-build
residency planning, the same user intent can be passed as `--profile` to
`plan`; `--policy` remains the explicit expert spelling and wins if both are
provided.

## Offline GPU encoder seam

The experimental GPU encoder is an offline proposer, not a runtime ASTC
decoder. A persistent generic Vulkan session accepts D1 or D2 physical source
blocks in deterministic contiguous batches and returns symbolic proposals. A
source-ID storage buffer preserves the caller's global IDs across dispatches;
the device smoke covers this with non-contiguous IDs and one-block batches.
D1/D2 candidate banks may add a common opaque identity envelope, while all
paired-row/layout semantics remain in their respective frontends. The D2
frontend additionally binds proposal retention to CPU legal finish and exact
activation ranking over reconstructed paired rows; the retained bank is input
to the established global selector, not a replacement for it.

The CPU finisher has `reference` and `guided` modes. Both currently use the
same exact astcenc legal-payload search and CPU decode oracle; guided mode only
means that an upstream proposal bank bounded the blocks to finish. Endpoint or
weight-grid hint seeding is deliberately not claimed until a separate
quality-preserving experiment exists.

## Source map

The files are deliberately grouped by responsibility rather than by one large
driver implementation.

| Area | Important files | Responsibility |
|---|---|---|
| Format and tensor contract | `astc-vulkan-format.*`, `astc-vulkan-tensor-contract.h`, `astc-vulkan-contract.h` | Footprints, byte geometry, representation enum, serializable invariants |
| Artifact/container | `astc-vulkan-manifest.*`, `astc-vulkan-provenance.*`, `astc-vulkan-artifact-pack.cpp`, `astc-vulkan-artifact-decode.cpp` | Pack, decode, validate and document byte-identical artifacts |
| Cache | `astc-vulkan-cache.*`, `astc-vulkan-cache-tool.cpp` | Hash validation, atomic cache publish, profile admission, inspection |
| Offline algorithms | `astc-vulkan-gauge.*`, `astc-vulkan-paired.*`, `astc-vulkan-paired-layout.*`, `astc-vulkan-paired-selector.*`, `astc-vulkan-objective.*`, `astc-vulkan-yaqa.*`, `astc-vulkan-pv.*`, `astc-vulkan-block-ldlq.*` | D1/D2 representations and ranking experiments |
| Offline GPU support | `astc-vulkan-d1-prescreen*`, `astc-vulkan-d2-prescreen*`, `astc-vulkan-gpu-ranking*`, `astc-vulkan-yaqa-*-device-smoke.cpp` | Optional GPU prescreening, proposal scoring, YAQA batching; never ASTC hardware encoding |
| Vulkan resource/dispatch | `astc-vulkan-resource.*`, `astc-vulkan-dispatch.*`, `astc-vulkan-paired-dispatch.*`, `astc-vulkan-sidecar.*` | Image/buffer lifetime, upload, D1 dispatch and D2 paired dispatch |
| Scheduler boundary | `astc-vulkan-scheduler-adapter.*`, `astc-vulkan-ffn-adapter.*` | Cache admission, format/representation routing, fallback contract |
| Shaders | `shaders/astc-ffn-matvec.comp`, `shaders/astc-paired-matvec.comp` | D1 and D2 semantic reconstruction after fixed-function texture fetch |
| Tests/smokes | `tests/`, `*-smoke.cpp`, `*-replay-smoke.cpp` | Contract tests, artifact replay, model/crop quality, and device smoke coverage |

`CMakeLists.txt` is intentionally the only build-registration point for these
targets and their optional feature gates.

## Why `astcenc` is third-party rather than driver code

`astcenc` is the reference-quality ASTC encoder and CPU decoder from the ASTC
ecosystem.  It is used here because it can emit and validate ordinary
standard-compliant 128-bit ASTC blocks.  Reimplementing the codec in the
driver would create a second, hard-to-validate ASTC implementation and would
not improve runtime, because the runtime decoder is the GPU texture unit.

The normal dependency is supplied as an external CMake package target:

```text
astcenc::astcenc-static
```

The neural-ranking fork is a separately supplied static library and include
directory.  It exists only to research broader candidate retention/ranking;
it is not linked into a production Vulkan shader and does not alter the
standard ASTC payload format.  This boundary gives us three useful properties:

1. an ordinary system `astcenc` remains the regression/reference oracle;
2. neural encoder experiments stay isolated and can be upstreamed or removed
   without changing ggml core;
3. every cache payload remains consumable by a standard ASTC-capable Vulkan
   implementation.

## Practical guardrails

- Never treat an offline source value as proof of runtime quality: use exact
  ASTC decode, artifact replay, and then device replay.
- Keep calibration, validation, and holdout traces disjoint.  Validation picks
  the exported commit prefix; holdout only reports it.
- Preserve scalar D1 and normal Q4/Q3 fallback candidates.
- Do not mix ASTC footprints inside one texture image.  Mixed footprints, if
  adopted later, should use coarse macro-tiles/atlases plus explicit metadata.
- Keep the D2 layout map with its payload.  A D2 payload without its matching
  map is not a valid runtime artifact.
- Do not add cache files to source control.  They are model-specific derived
  artifacts and can be large.

## References

- Arm, *ASTC: The Future of Texture Compression* and the `astcenc` project.
- Khronos, Vulkan ASTC LDR sampled-image format capability rules.
- QuIP / QuIP# for LDLQ-style error feedback and incoherence ideas.
- GPTVQ for block/vector target regeneration.
- PV-Tuning (arXiv:2405.14852) for codec-aware low-bit optimization.
- YAQA for two-sided, sensitivity-aware quantization objectives.
