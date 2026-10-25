# Residual-normalized coefficient experiments

Status: implemented for opt-in experiments; not compiled or executed during this change.
Production defaults, host verdicts, promotion rules and legacy persistence field meanings are unchanged.

## Quantities and provenance

`requested_strength` and `executed_strength` remain coefficient L2 norms. Legacy
`relative_dose` remains displacement relative to the reference behavior delta:
`sqrt(progress^2 + leakage^2)`. The existing diagnostic cosine compares the observed
displacement with that delta. Neither is an angle or dose relative to the baseline residual.

The experiment adds requested/applied intervention norms and residual fractions.
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
pairwise directions, a bounded dose grid and two local refinement proposals
(0.75 and 1.25 times the best executed diagnostic direction/dose). Two proposal
slots are reserved for refinement when the total budget permits. Missing margins
skip refinement. The no-op counts toward `max_candidates`; full-generation
frontier limits and separately configured retries retain existing meanings.

Both calibrated modes require valid calibration and coordinate search. TFO plus
calibration fails explicitly. Numerically cancelling directions are skipped;
all-degenerate candidate sets fail. Duplicate composed interventions are removed
before inference. Calibrated dose-control state is isolated per arm and its retry;
it is not reused across different directions, layers or searches. Legacy state
sharing is unchanged. Geometry does not grant HELPED or activation authority.

## Prepared verification

Contract tests cover median calibration, invalid provenance, actual basis norms,
zero/non-finite inputs, clipping, duplicate/cancelling directions, deterministic
proposals, budget bounds, shared batch execution, refinement and rejection before
model execution. Checks remain active with NDEBUG. They have not been run.

Run `scripts/test-agent-flydelta-dose-comparison-model-smoke.sh` after the existing
Vulkan/Cozo build is available. The wrapper uses the normal model-smoke helpers,
hashes the model content, forces resident tracing and runs raw, normalized and
direction-dose conditions serially. Configure the usual model/build paths and GPU
device externally. No additional worker is started. The comparison intentionally
skips unrelated cvec representation benchmarks.

Each condition uses the same replay suite, rank sweep, 16-proposal ceiling,
full-generation limits and zero dose retries. Normalization excludes the current
evaluation scenario from its calibration sample set. This is leave-one-scenario-out
norm calibration, not held-out direction learning: existing direction acquisition
still uses the original replay data. A missing calibration set fails explicitly.
Different conditions can exhaust their distinct candidate sets before the ceiling;
actual arms and time are reported, rather than claiming equal consumed compute.

`dose-comparison.json` preserves calibration provenance, arm diagnostics, host
outcomes and search wall time. Fixture/rank mismatches or missing arms fail the
summary. Raw model exchanges, failures and timeout diagnostics remain in the logs.
A successful fixture or diagnostic score is not evidence of improved steering.
Independent evaluation fixtures, repeated runs and agreed quantitative acceptance
thresholds are still required before any production-adoption decision. No thresholds
or production adoption are chosen by this implementation.
