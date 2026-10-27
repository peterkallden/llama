# FlyDelta responsibilities and result interpretation

## Current verification boundary

The ownership and reporting refactor dated 2026-10-27 is implemented but has not
been compiled or tested. The interrupted model experiment is retained separately
and supplies no validation of this refactor. Verification awaits explicit user
authorization. The ownership refactor preserves search and lifecycle policy.
The subsequent partial-progress guidance change below intentionally extends
search prediction and qualification utility, while preserving promotion and
activation authority. That change is also uncompiled and untested.

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
provider header. The ownership refactor introduces no separate persistent store,
URI scheme or evidence database. The subsequent progress addition extends the
existing journal payload with versioned fields and explicitly versions predictor
and qualification identities, while retaining the legacy observation reader.

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

The original observations carried by intermediate region/coefficient trials
preserve host verification. Reporting consumes them directly; progress guidance
uses only validated paired objective assessments projected from those observations.
They are not a second semantic evaluator or promotion authority.

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

## Partial-progress outcome guidance

The same outcome adviser now has two distinct statistical targets:

- The existing HELPED/NEUTRAL/HARMED/UNKNOWN paired outcome.
- A verifier-owned goal transition: improved, unchanged, regressed, solved, or
  unavailable because identity, comparability or control coverage is missing.

UNKNOWN full outcomes may have known partial progress. Missing legacy progress
is not a negative training example. A failed parser or absent assessment cannot
establish regression-free improvement. A detected lost goal or declared control
regression remains negative evidence even when other controls are incomplete.
"Regression-free" is restricted to the declared host assessment; it does not
claim coverage of unrelated model capabilities.

`common_flydelta_progress_evidence` stores the original baseline and candidate
assessments, their arm/evidence references and optional parent assessment. The
host derives the transition using the existing objective identities and strict
goal-preservation rules. Serialized labels are not accepted as truth. Coefficient
lifecycle rows add `outcome_guidance.schema_version=2`; the shared reader accepts
version 1 as outcome-only history. Captured prompts and model outputs remain in
the existing trace/resource paths.

The domain-neutral conjugate posterior implementation remains in
`common/agent/prediction/outcome-guidance.cpp`. FlyDelta builds a separate progress
posterior through that implementation. It retains progress coverage uncertainty
independently of full-outcome uncertainty. History with only unknown progress
can train the coverage probability but cannot contribute partial selection utility
until known progress assessments exist locally or in the qualifying backoff.
The partial reward is
`progress_reward * P(known progress) * P(improved | known progress)` and is added
to the existing HELPED-minus-HARMED search utility. Solved observations and HELPED
arms receive no additional actual partial-success reward. The configurable
`runtime.adaptation.flydelta.outcome_guidance.progress_reward` defaults to 0.25,
accepts 0..1, and can be set to zero to disable that utility term. The existing
`off`, `shadow` and qualified `experimental` modes retain their authority limits.
This numerical weight is a policy choice, not a measured optimal value.

Progress partitions preserve objective schema/verifier identity and dose policy /
calibration identity. Exact partitions also distinguish immutable basis and signed
coefficient direction. A bounded compatible backoff supplies only prior weight.
Each fixture group contributes at most one normalized observation unit; repeated
arms are not independent proof of generalization. Posterior interval-product
bounds are diagnostics, not an asserted joint 95-percent credible interval.

### Local continuation and reproducibility

The history snapshot and cached projection retain two evidence views:

1. Other compatible fixture groups provide the cross-fixture forecast.
2. Earlier completed waves of the current fixture provide local progress evidence,
   restricted to the same task, search lineage, objective, direction and dose bucket.

These views share the journal and predictor. Local rows never increase the number
of independent fixture groups used for qualification. When resuming an anchor,
local candidates are compared to its stored goal assessment as well as the
original baseline. Losing a parent goal cannot count as cumulative improvement.
This is discrete, conservative local evidence reuse, not a continuous surrogate
model or a new optimizer.

A wave identity binds job, phase, parent-anchor reference and committed logical
budget usage. A new wave freezes a fresh history snapshot. Retrying that wave
reuses its snapshot, stable observation identities and decision cursor; later
journal writes do not alter an interrupted decision. The next committed wave may
consume newly registered progress. Conflicting observations with one stable ID
are excluded rather than resolved by append order.

The full-generation baseline is reused as the progress query. No additional
model invocation, GPU allocation or inference branch is introduced. Projection
aggregation is cached for the wave and local counts are reused per candidate
partition within a decision. Benefits in total compute require empirical evidence
of fewer model evaluations to an equal verified objective; fixed-budget searches
may instead spend the same compute and find better results.

### Qualification, diagnostics and pending verification

The existing matched policy union records paired progress evidence alongside
actual arm outcomes. Qualification reports partial improvements, progress
regressions and unavailable progress separately from HELPED/HARMED/UNKNOWN.
It applies the configured partial reward, does not double-count HELPED as partial
success, and rejects qualification when the guided policy increases the observed
progress-regression rate. Missing control coverage cannot earn partial utility.
No automatic promotion or activation follows from a qualified search policy.

The policy fingerprint versions the progress contract, projection builder and
reward. Previous qualifications cannot authorize this changed selection policy.
Independent fixture requirements and existing full-outcome harm checks remain.
Progress Brier diagnostics use chronological holdouts and do not add holdout
observations to training history.

Added source contracts cover paired evidence round trips, parent-goal loss,
missing control coverage, legacy history, separate progress/full outcomes, signed
directions, duplicate/conflicting observations, isolated local history and
partial-progress qualification. They have not been executed. Required next work,
after explicit authorization, is compilation, the focused outcome/Deep/lifecycle/
qualification CTests, restart/retry coverage through daemon paths, and traced
model comparisons at equal inference budgets. No quality or compute improvement
is claimed by the source implementation alone.
