# FlyDelta responsibilities and result interpretation

## Current verification boundary

The ownership and reporting refactor dated 2026-10-27 is implemented but has not
been compiled or tested. The interrupted model experiment is retained separately
and supplies no validation of this refactor. Verification awaits explicit user
authorization. Search algorithms, admission thresholds and lifecycle policy are
not intentionally changed by this work.

## Follow one bounded slice

```text
host-authorized material and job
  -> existing queue and worker
  -> daemon workflow for the current phase
  -> existing resident model batch host
  -> original arm observations
  -> existing search/Oracle decisions
  -> durable state and typed next action
  -> commit acknowledgement
```

The next action schedules another bounded slice through the same queue. It does
not create a second planner or runtime. Learning credit, review, promotion and
activation retain their own existing authority boundaries.

## Code ownership

| Location | Responsibility |
| --- | --- |
| `tools/agent/daemon/agent-daemon-flydelta.cpp` | Public configuration/registration facade and runtime setup |
| `tools/agent/daemon/flydelta/bootstrap.cpp` | BootstrapZoom, AdaptiveAlpha and Orthogonal host workflows |
| `tools/agent/daemon/flydelta/search.cpp` | Initial WHERE pipeline and post-Bootstrap Shallow/Deep workflows |
| `tools/agent/daemon/flydelta/evaluation.cpp` | Evaluation/causal/counterfactual workflows and evaluator callback composition |
| `tools/agent/daemon/flydelta/augmentation.cpp` | Donor capture and representation augmentation workflows |
| `tools/agent/daemon/flydelta/state-codec.cpp` | Pure typed orchestration/observation codecs and identity helpers |
| `tools/agent/daemon/flydelta/state.cpp` | Scope-bound lifecycle publication/resolution callbacks and caches |
| Existing daemon `provider`, `binding`, `concept-*` modules | Provider creation, execution binding and concept workflows |
| `tools/agent/runtime/agent-flydelta-server-host.cpp` | Actual arm preparation, execution, capture, verification and execution telemetry |
| `tools/agent/adaptation/flydelta/` | Authorized resource resolution, materialization and dose calibration |
| `common/agent/adaptation/flydelta/experiment/` | Typed domain algorithms and routing policy |
| `common/agent/adaptation/flydelta/flydelta-trace.cpp` | Worker-result presentation and trace serialization; no search decisions |
| Existing queue, worker-commit and lifecycle stores | Queue ownership, durable effects, continuation recovery and acknowledgement |

The private `state.h` exposes the codec/binding seam independently of the large
provider header. No new persistent schema, URI scheme or evidence database is
introduced. Persisted JSON names, ordering, defaults and identity calculations
remain at the existing compatibility boundary.

## Distinguish the questions a result answers

| Dimension | Meaning |
| --- | --- |
| Execution | Whether model work ran and an overlay was physically applied |
| Diagnostics | Margin and representation geometry; useful for exploration |
| Individual verification | Whether one actual response satisfied its verifier |
| Paired outcome | HELPED/HARMED/NEUTRAL/UNKNOWN against the original baseline |
| Semantic progress | Comparable reached/remaining goals and regression coverage |
| Search decision | Whether the existing policy retains/refines/stops the experiment |
| Lifecycle decision | Whether evidence earns learning/review/promotion/activation |

Neither a positive margin nor a complete process establishes semantic success.
Conversely, UNKNOWN must not hide a recorded improvement in a particular goal.
Individual candidate success is copied from its original verifier observation,
not reconstructed from HELPED: a neutral pair can contain two successful arms.
When the original observation is absent, `candidate_evaluation_available` is false.
The trace does not invent an individual verdict or a matched baseline.

The optional original observations carried by intermediate region/coefficient
trials are reporting data. Search selection, scoring and promotion keep consuming
their existing fields and rules. They are not a second semantic evaluator.

## Dose interpretation

```text
direction and coefficients
  -> requested dose and immutable calibration
  -> existing normalization and DoseController
  -> actually applied coefficients/intervention
  -> model behavior and host verification
```

Coefficient magnitude, intervention norm, residual-relative dose and the legacy
behavior-delta-relative displacement are different quantities. Trace carries the
original requested/applied measurements and calibration reference when available;
it does not recalculate them. Availability markers distinguish missing values
from actual zero. DoseController decisions still bound the experiment. Unsupported
TFO/dose combinations retain their existing rejection behavior.

## Durable state has several owners

- BootstrapZoom state preserves the bounded trial profiles and refinement phase.
- A Deep search anchor preserves a reconstructible basis/intervention and objective.
- Orchestration state identifies the current phase, budget and continuation intent.
- Queue state determines job ownership and terminal status.
- Commit intent/acknowledgement tracks durable effects and follow-up scheduling.

Caches accelerate resolution but do not replace the scoped resource store or
journal. A completed inference, a published checkpoint and an acknowledged worker
commit are different events. Refactoring must preserve their ordering, cancellation
behavior and idempotent recovery; it must not migrate experiments to Deep just to
reuse its persistence path.

## Experiment entry points

Both dose scripts delegate to `scripts/agent-flydelta-dose-smoke-common.sh`:

- `test-agent-flydelta-dose-paired-model-smoke.sh`: raw and direction-dose.
- `test-agent-flydelta-dose-comparison-model-smoke.sh`: raw, normalized, direction-dose.

These select the dedicated fixed-rank `--dose-comparison` native path. The existing
low-rank rank-sweep script remains a different experiment. Model/build/device
settings are external; the default GPU layer request is auto-fit `-1`.

Before model execution the runner records commands, profile, model/binary/suite
fingerprints, Git revision/diff identity and required coverage in a fresh run
manifest. Complete stdout/stderr is retained per condition. Interrupting the runner
stops its child and prevents later conditions. These experiments have no process
timeout and do not start daemon FlyDelta workers.

The report joins actual full-generation baseline and candidate traces by arm ID.
It checks prompt equality and physical overlay confirmation. Missing/ambiguous
responses remain visible alongside any usable partial data. Raw model text and
prompts live only in explicit experiment artifacts; bounded production lifecycle
records do not acquire them.

`dose-comparison.json` separates process status, coverage, cross-condition
comparability and behavior observations. `dose-comparison.md` provides the raw
before/with-arm table, detailed dose/progress information and trace line references.
A complete trace with an UNKNOWN verifier is possible; it proves coverage, not
improved steering. Daemon crash recovery requires a separate verification scenario.

## Verification prepared, not run

The planned checks include the existing orchestration, coefficient, evaluator,
worker, collection, dose and server-binding CTests, plus the new pure state-codec
contract and offline runner/summary contracts. These cover legacy decoding,
neutral/successful pairs, absent observations, actual zero dose, response pairing,
wrong experiment flags and interruption. All new C++ checks use explicit CHECK
branches that remain active with NDEBUG.

After authorization: build in the existing Vulkan/Cozo tree with ccache, execute
focused checks serially, then model-free recovery and finally the explicitly
selected traced model experiment. Record those results before claiming behavioral
preservation or model improvement.
