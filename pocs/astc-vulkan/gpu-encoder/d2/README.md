# D2 GPU-encoder frontend

D2 owns paired-row source construction, direct RG/B and R/GB layouts, D2-LA,
row normalization, prediction, layout metadata, and D2-specific scoring. None
of those semantic decisions belong in the shared physical proposal layer or in
the D1 frontend.

The initial implemented seam is deliberately narrow:

```text
normalized logical row pairs + per-block layout map
  -> D2 6x5, 8x5, or 10x5 RGBA physical source blocks
  -> generic Vulkan physical proposer
  -> generic CPU astcenc legal-payload finisher
  -> exact decoded D2 activation ranking
  -> CPU/Vulkan fixed-function decode oracle
```

`astc-gpu-d2-source.*` owns the paired geometry and deterministic padding.
`astc-gpu-d2-candidates.*` owns the mandatory direct-neutral baseline and
records each alternative's layout and semantic decoder. The shared finisher
does not know whether the physical texels came from D1 or D2.

`astc-gpu-d2-hybrid.*` binds those layers together. It accepts proposals from
either the generic Vulkan proposer or the CPU reference proposer, retains a
bounded physical candidate bank, finishes legal payloads with CPU `astcenc`,
and ranks the *decoded paired weights* against activation traces. Thus
`direct-neutral`, `direct-steered`, and `luminance-alpha` remain offline
alternatives without leaking paired semantics into the shared encoder.
Direct-neutral is mandatory. The output is a local bank for the existing
conflict-aware selector, not a published cache artifact or model-level choice.

`astc_gpu_d2_make_selector_candidates()` is the explicit hand-off to that
selector. It converts every exact decoded candidate into calibration/validation
output deltas relative to the neutral decoded block, preserving the selected
16-byte payload. This keeps validation-prefix export and global conflict
ordering identical between CPU and hybrid proposal backends.

This is **not** a D2 quality claim or a cache/runtime change. The implemented
five-row profiles are direct D2 6x5 (2.13 bpw), 8x5 (1.60 bpw), and 10x5
(1.28 bpw). Alpha-steered direct-D2 and D2-LA now use this exact local ranking
contract, but still require the global selection, artifact, model, and Vulkan
quality gates before admission. D2 8x8/10x10 remain a separate eight/ten-row
geometry milestone.

The device smoke exercises all three implemented five-row footprints and the
full hybrid hand-off. It is a correctness gate only; full model replay and
cache publication remain later gates.

The first full-shape GPU model replay has now passed for the existing evidence-
approved D2-LA 8x5 artifact (`blk.1.ffn_down.weight`, 2048x8192). Resident and
streamed sidecar loading produced byte-identical metrics: logits MSE
`2.956062`, relative logits MSE `0.30820498`, top-1 agreement `0.70`, and loss
delta `+1.5306693` on the 10-token Pythia replay trace. This validates the
artifact, paired dispatch, GPU matvec, and streamed lifecycle path, but the
quality result remains experimental and is not a production scheduler gate.
