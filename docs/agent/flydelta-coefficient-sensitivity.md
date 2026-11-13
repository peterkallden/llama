# Optional coefficient sensitivity proposals

The existing TFO-lite search can consume one local sensitivity proposal. This
is an experimental WHAT refinement at the selected WHERE, not layer discovery,
a replacement optimizer, or evidence of behavioral improvement.

## Host configuration

Set `common_flydelta_coefficient_search_config::sensitivity.enabled` explicitly.
The default is false. This is a C++ host configuration seam; no daemon JSON or
model-controlled setting is introduced. The host must select TFO-lite, provide
at least `2 * rank` population slots, two iterations, and a candidate budget
larger than `2 * rank`. `ridge`, `target_margin_delta`, and `trust_radius` are
positive finite values; the trust radius cannot exceed the coefficient bound.

## Mathematics and execution

At the initial zero center, the opt-in initial population contains coordinate
pairs before mixed proposals. Existing runners and DoseController execute them.
For each coordinate, a matched applied pair estimates
`s[j] = (margin(+a e[j]) - margin(-a e[j])) / (2a)`.
Margins are normalized teacher-forced decision margins from the same fixture.
Missing, asymmetric, safety-limited, known-harmed or unavailable observations
cannot contribute. Every basis coordinate needs a matched pair.

The helper solves `(s s^T + ridge (B^T B + I)) dc = target * s` and scales the
result into the coefficient trust ball. This is a regularized local proposal,
not an exact constrained optimizer. Flat or invalid measurements yield no
proposal. The helper owns no runner, store, scheduler or outcome authority.

One new candidate may occupy a next-generation population slot. It does not
change the TFO anchor. The ordinary runner applies dose control again, and all
existing generation, Oracle, retention and activation gates remain in force.
Executed trials carry `mutation_kind = sensitivity_proposal`. There are no extra
unbudgeted probe calls, and no Bayesian posterior changes.

## Limits and verification

This first version estimates only at the initial zero center and within the
existing basis. Calibrated unequal +/- doses may prevent a proposal; they are
never silently treated as symmetric. No protected-behavior sensitivity matrix
is estimated, and diagnostic margin gains do not establish safety or HELPED.
The default disabled path retains the existing proposal sequence.

Contract cases are added to `test-agent-flydelta-coefficient-search`: deterministic
proposal, bounded coefficients, asymmetric probe rejection, flat signal and
disabled behavior. Compilation, execution and model efficacy require separate
verification. Evaluate verified outcome per intervention budget against the
same TFO configuration with this option disabled.

## Matched model smoke and tracing

Run `scripts/test-agent-flydelta-sensitivity-model-smoke.sh --model /path/model.gguf`
after building `llama-agent-flydelta-dataset-question-repair-model-smoke`.
Select the build with `LLAMA_AGENT_BUILD_DIR`. The script runs control and
sensitivity serially, with resident tracing forced on, separate logs and
ledgers, input hashes, and a `comparison.json` report. It never builds implicitly.

Both runs request the same initial coordinate stencil, seed, rank-dependent
population, iterations, candidate budget and full-generation policy. Only the
treatment enables the additional proposal. This isolates proposal value from
initial-stencil changes. Other dose-comparison modes are rejected for this test.
The host-owned sensitivity trace callback reports measurement count, rejection
reason, computed proposal and candidate admission. Actual execution is reported
by the existing per-arm trace with `mutation = sensitivity_proposal`.

Missing eligible fixtures, mismatched bases, or no executed sensitivity proposal
are inconclusive (exit 77), not evidence that the feature works. Reports compare
verified HELPED counts, margins, harmed outcomes and executed arm counts. Because
this staged smoke does not stop at the first HELPED result, it does not establish
fewer interventions to success; that needs a subsequent stopping-policy trial.
