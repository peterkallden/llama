# FlyDelta verification snapshot: 2026-10-28

This is a historical verification record for commit
`bcf47549e02ed3e2a18813279cb0cd74d52277fa`. It is not a test result for a
later checkout. In particular, `c88a8ebda` subsequently added layer-specific
concept localization and its tests; that change is outside this snapshot.

## Build and contracts

The full Vulkan/Cozo build completed. A selected run of 230 agent CTests
reported 215 passed, two failed and 13 skipped. The failures reproduced when
rerun serially: `test-agent-loop` failed the reflection overflow assertion and
`test-agent-tool-rounds` failed a closed-plan step-count assertion. Their root
causes were not established by that verification run. The skipped cases
included model tests without a model in the ordinary CTest environment and
platform or environment-dependent sandbox and debugger cases.

Selected passing contracts included coefficient search, dose summary,
outcome guidance, guidance qualification, Whirlpool, Deep search,
experiment orchestration, collection, worker, evaluator and server binding.
Passing these tests establishes their exercised contracts, not the success of
all production workflows.

## Model-backed observations

The serial traced model profile used Qwen2.5 1.5B Instruct and automatic GPU
fitting with an explicitly verified NVIDIA device. Model adaptation, FlyDelta
A/B, cvec batch and runtime checks passed their process, trace and GPU checks.
In A/B, the candidate overlay was applied and both baseline and candidate were
host-verified `PASS`; the paired outcome was `neutral` with quality delta zero.
The cvec batch arms also both returned `PASS` with equal teacher-forced margins.
These observations demonstrate the exercised execution path, not an
improvement in steering.

Concept synthesis failed because teaching-relation lineage lacked a nonzero
`byte_length`. The repair smoke failed its generation context budget
(`prompt=1472`, `reserved_output=1`, `context=1024`). The causal-patch test
skipped because its rejected baseline tool call was not produced, so no patch
comparison ran. Dataset-question repair was stopped during processing without
final arm results. A separate dataset-question profile passed its process and
trace checks, but only five of twelve scenarios selected the expected tool.
These are recorded outcomes, not confirmed diagnoses or claims about current HEAD.

No raw-versus-normalized-versus-direction-dose comparison completed. Model
quality, dose effects, restart/retry behavior and changes after this snapshot
need separate evidence. An HTTP vertical smoke also failed its terminal and
research event-stream contracts; that result is separate from the FlyDelta
model comparisons.
