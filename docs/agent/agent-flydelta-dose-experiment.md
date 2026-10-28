# Residual-normalized coefficient experiments

Status: host configuration defaults to `direction_dose`; `raw_coefficients` remains
available as an explicit alternative. Model experiments must be evaluated separately
from deployment and learning decisions. This document does not assert that the
latest working-tree changes have been compiled or tested.

## Quantities and provenance

`requested_strength` and `executed_strength` remain coefficient L2 norms. Legacy
`relative_dose` remains displacement relative to the reference behavior delta:
`sqrt(progress^2 + leakage^2)`. The existing diagnostic cosine compares the observed
displacement with that delta. Neither is an angle or dose relative to the baseline residual.

The experiment adds requested/applied intervention norms and residual fractions.
Extra norm diagnostics are opt-in (`record_intervention_norms`) for the raw control;
ordinary production searches do not run the extra vector compositions. A shared
internal helper updates the measurements after each observed execution, including
staged full generation and TFO diagnostics. Missing diagnostics are explicit rather
than being represented as a measured zero.
For direction `d = Bc`, requested intervention is `q * median_norm * d / ||d||`.
Norms use the composed vector, including basis overlap. Requested norm/fraction
precede coefficient clipping; applied values describe the final coefficients,
including dose-controller backoff. These describe an additive vector at the
injection site, not a guarantee about downstream displacement or answer quality.

Calibration is versioned and carries model/tokenizer/template identity, layer,
dimension, capture layout, token selection, sample norms, median and reference.
The capture helper rejects incompatible sites and invalid/zero samples. The host
adds tokenizer/template identities; proposal generation validates them against
the fixture. Prefill/decode populations must be calibrated separately. The smoke
currently supports generation-boundary prompt captures only; it does not claim
decode calibration or equivalence of different token-coverage schedules.
Calibration records are preserved in the smoke trace and comparison JSON; no
production calibration store or automatic reuse is introduced.

## Existing search integration

`raw_coefficients` retains the coordinate/TFO defaults. `residual_fraction` uses
the coordinate directions at one normalized dose. `direction_dose` adds signed
pairwise directions in both polarities, a bounded dose grid and two local refinement proposals
(0.75 and 1.25 times the best executed diagnostic direction/dose). Two proposal
slots are reserved for refinement when the total budget permits. Missing margins
skip refinement. The no-op counts toward `max_candidates`; full-generation
frontier limits and separately configured retries retain existing meanings.

Both calibrated modes require valid calibration and coordinate search. TFO plus
calibration fails explicitly. Numerically cancelling directions are skipped;
all-degenerate candidate sets fail. Duplicate composed interventions are removed
before inference. Calibrated dose-control state is isolated per arm and its retry;
it is not reused across different directions, layers or searches. Legacy state
sharing is unchanged. Rejected calibrated doses are excluded from refinement and the full-generation
frontier. Geometry does not grant HELPED or activation authority.

## Prepared verification

Contract tests cover median calibration, invalid provenance, actual basis norms,
zero/non-finite inputs, clipping, duplicate/cancelling directions, deterministic
proposals, budget bounds, shared batch execution, refinement and rejection before
model execution. Checks remain active with NDEBUG. Use the recorded test results
for the revision under evaluation; the presence of a test is not proof it passed.

For model tests, use
`scripts/test-agent-flydelta-dose-paired-model-smoke.sh` for raw versus
direction-dose, or `scripts/test-agent-flydelta-dose-comparison-model-smoke.sh`
for raw, normalized and direction-dose. Both use the same runner and the dedicated
`--dose-comparison --production-bindings` native path; neither launches a rank
sweep or unrelated cvec benchmark. The paired profile uses two aggregate scenarios;
the three-condition profile defaults to eight scenarios.

Set the existing model/build paths and Vulkan device externally. GPU layers default
to `-1` for automatic fitting. Each attempt gets a fresh directory and a manifest
before inference, including exact commands, content fingerprints and requirements.
All conditions are serial, trace is mandatory, and these profiles have no wall-clock
timeout. Interrupting the runner terminates its active child, records interruption,
and prevents subsequent conditions from starting. No daemon worker is started.

Each condition uses the same replay suite, fixed rank-one dose comparison, 16-proposal ceiling,
full-generation limits and zero dose retries. Normalization excludes the current
evaluation scenario from its calibration sample set. This is leave-one-scenario-out
norm calibration, not held-out direction learning: existing direction acquisition
still uses the original replay data. A missing calibration set fails explicitly.
Different conditions can exhaust their distinct candidate sets before the ceiling;
actual arms and time are reported, rather than claiming equal consumed compute.

`dose-comparison.json` preserves calibration provenance, arm diagnostics, process
exit codes, host outcomes and search wall time, including failed or missing
conditions. It also pairs full-generation responses by baseline/candidate arm IDs
and preserves complete prompt messages and model content. `dose-comparison.md`
shows a before/with-arm table and detailed dose/progress/verification information.
Canonical repair echoes and diagnostic captures cannot substitute for full arms.
Missing response identities, mismatched prompts, unconfirmed overlays or missing
host evaluation make coverage incomplete. Execution, coverage, comparability and
behavior are reported separately; UNKNOWN does not erase recorded partial progress. Each attempt uses a fresh directory, so interrupted runs cannot reuse
stale logs. A failed condition does not suppress the other conditions' diagnostics;
an interrupt stops further inference. The summary is still written and returns a
nonzero exit code when the comparison is incomplete.

Cross-condition matching uses evaluation scenario, rank, layer and basis identity.
Scope-local resource IDs are preserved for provenance but may legitimately differ
between processes. A complete trace does not imply a known host verdict: verified
full arms are counted separately, and UNKNOWN remains an unknown quality result. Raw model exchanges, failures and timeout diagnostics remain in the logs.
A successful fixture or diagnostic score is not evidence of improved steering.
Independent evaluation fixtures, repeated runs and agreed quantitative acceptance
thresholds are still required before any production-adoption decision. No thresholds
or production adoption are chosen by this implementation.

## Static review follow-up

The review isolated measurement overhead from the production defaults, centralized
metric updates, checked proposal/metadata alignment before baseline execution, and
excluded unexecuted/unapplied arms from refinement. Signed mixed directions now
include both polarities. Offline Python summary contracts are registered with CTest
for partial failures, damaged/missing records, calibration overlap and comparison
identity. These contracts and the revised C++ contracts remain unrun.

This remains an experimental implementation of normalized dosing and bounded
joint exploration, not evidence that either strategy improves steering. In
particular, a small candidate budget can truncate the configured direction/dose
grid, and coefficient clipping can make several requested doses physically
identical. Deduplication avoids paying for those duplicates; the actual trace and
requested/applied measurements, rather than the configured grid alone, define
what was evaluated. Independent direction-learning/evaluation separation and
production-adoption criteria remain prerequisites for a later production decision.

### 2026-10-27 verification boundary

The shared runner, strict response pairing and new contract cases have not been
compiled or executed. The interrupted experiment predates this refactor and is not
validation of these changes. Do not launch a build, CTest or smoke until the user
explicitly authorizes verification.

## Execution evidence and comparison identity

Dose traces distinguish the requested residual-relative intervention norm,
`proposed_intervention_norm` after the coefficient bound, and the applied norm
following DoseController decisions. `proposal_norm_limit_known` indicates whether
the pre-bound request was retained; it is unknown for legacy raw coefficient
proposals and anchor-centered proposals whose original request was not retained.
`proposal_norm_limited` does not imply `dose_safety_limited`. Coefficient norms, residual fractions and the legacy
behavior-relative displacement retain their separate units.

New run manifests use `flydelta-dose-run.v2`. Each measured group emits an effective
execution context: model identity (including quantization via its content hash),
tokenizer/template identities, capture layout/site, actual context and parallel
sequence counts, cache types, greedy sampling/seed, response and search budgets,
and verifier revision. Explicit generation tracing also emits JSON records with
logical messages, the final rendered prompt and its content fingerprint. These
records use the existing trace opt-in; they are not enabled for ordinary requests.

The smoke's tokenizer and template labels are fixture labels, not independent
hashes of the actual tokenizer or template. The model-file fingerprint binds the
embedded tokenizer, while the rendered-prompt fingerprint identifies the concrete
expanded template output for each compared request.

The comparison reader preserves legacy records and their full outputs. It reports
an unmatched comparison when effective contexts, rendered prompts or calibration
measurements are missing or differ. Scope-local fixture resource names alone do
not invalidate matching. Requested doses/candidate choices may differ by policy.
The smoke repeats unsteered generations three times by default in dose mode
(`--baseline-repeats`), while the production generation path stays greedy. The
logged seed is therefore provenance, not a source of random sampling. Repeats can
measure execution variability but do not estimate stochastic decoding quality.

Use `agent-flydelta-dose-summary.py TRACE.log --trace` to summarize observed
phase execution and retain all recorded model exchanges and adviser decisions.
A completed phase callback or explicit executed arm can establish phase coverage;
the persisted plan may describe a later slice. Evidence-depth labels, requested
phases and next-action proposals do not by themselves establish execution. An absent phase remains
unobserved in that trace. A second baseline wave is not an anchor resume.

For opt-in model CTest validation, `scripts/agent-model-test-profile.py` discovers
the existing registered model tests, requires an explicit model, runs serially,
uses automatic GPU fitting and retains individual logs plus a JSON status file.
A missing executable or exit code 77 is a configuration error in this profile;
ordinary model-free/synthetic tests retain their existing skip semantics.
