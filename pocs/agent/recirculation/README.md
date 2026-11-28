# Tokenwise recirculation PoC

## Experimental alternative direction

`--experiment-mode alternative-direction` is an offline comparison. It fits one
unit direction from `object_location` development cases only. For each case it
replaces the new location in the `UPDATE` sentence with the known old location,
requires equal token counts, and captures the destination layer input at the
second-to-last prompt token in both versions. It averages normalized
updated-minus-counterfactual differences. The expected/incorrect answer labels
are used only to construct these development counterfactuals and to score the
separate evaluation cases. Training IDs, exclusions, capture site and semantic
role are recorded in the trace. This is an experimental direction, not a
FlyDelta concept candidate or verified repair direction; it bypasses no
FlyDelta evidence or promotion gate.
The evaluator also reports post-hoc alignment with each held-out
updated-minus-counterfactual capture when token counts match. Held-out labels
are used for this diagnostic only, after the development direction is fixed;
alignment never selects an arm or changes its patch.

The evaluator replays the same final six prompt tokens and intervenes at the
second-to-last position in every arm. It reports baseline, ordinary
recirculation with one and two rounds, alternative direction with one round,
alternative direction followed by a recirculation update, a fixed alternative
patch replayed for two rounds, and a cyclically permuted direction control.
It also replays a two-round fixed-direction control whose first patch is
identical to the alternative arm and whose second patch has the same norm as
the alternative-plus-recirculation arm. This isolates direction change from
second-round dose change within a declared numerical tolerance.
Each initial injected displacement has norm `alpha * ||destination||`; the
two-round arms share the same replay budget. The fixed-patch arm tests whether
source feedback adds anything beyond repeating the same intervention. The
permutation is a geometry-preserving control, not a second semantic concept.
All arms use teacher-forced expected and incorrect continuations and record
both token log-probability paths, patch norms, replay wall time and entropy decisions. The
reported margin is `expected total - incorrect total`; no arm is selected
online by the answer label.

```sh
<build>/bin/llama-recirculation-poc \
  --model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --source 7 --destination 2 --alpha 0.03 \
  --experiment-mode alternative-direction \
  --dev-cases-file pocs/agent/recirculation/data/development-v2.jsonl \
  --eval-cases-file pocs/agent/recirculation/data/evaluation-v2.jsonl \
  --case-id v2-evaluation-object-0 --trace alternative-direction.jsonl
```

Predeclare case IDs and all parameters before a held-out run. The current v2
data use repeated templates, so separation by ID is weaker than an independent
task distribution. The counterfactual can also encode lexical differences
between locations. A result on these cases would establish neither a general
alternative-reasoning ability nor a valid online commit rule. The evaluator
does not change resident inference, the frozen model, or FlyDelta activation.

In a 2026-11-28 Qwen3 0.6B Q8_0 CPU run with layers `7 -> 2` and alpha `0.03`,
all eight development object contrasts had equal token counts. On eight
separate v2 evaluation object cases, mean margin changes against baseline
were +0.0327 for ordinary one-round recirculation, -0.0098 for ordinary
two-round, -0.0641 for alternative one-round, and -0.0457 for alternative
followed by recirculation. The latter improved four cases and worsened four.
Replaying the same alternative patch for a second round matched its one-round
margin, as expected for a deterministic fixed patch. The permuted-direction
control had mean -0.0050. Held-out contrast alignment was unavailable for
three cases with unequal counterfactual token counts; the five defined
cosines were 0.1503, 0.2000, 0.0645, -0.1206 and 0.0111. These observations
do not support this particular direction as a reliable alternative-state
intervention. They do not rule out better grounded concept directions.
The final matched-second-round control had mean margin change -0.0391; the
alternative-plus-recirculation arm was -0.0066 below that control on average,
with mixed per-case signs. The development contrast alignment ranged from
0.270 to 0.478, higher than the defined held-out alignments. This supports a
weak-transfer interpretation for this particular extraction rule, not a
conclusion about FlyDelta concept directions generally.

The Qwen2.5 1.5B Q4_K_M exploratory run used layers `14 -> 4` and the same
alpha. Its trace contains all eight development fits and all eight evaluation
case results. Alternative-plus-recirculation changed the margin by -0.4517 on
average, with one improvement and seven regressions. The ordinary two-round
arm changed it by -0.4676. The run printed its `PASS` case count, but its
shell process was interrupted after that output and returned 130; treat these
as complete case observations without a clean process-exit proof. It preceded
the matched-second-round control, so no Qwen2 result is reported for that arm.

## Historical UPDATE-case regression

`--experiment-mode historical-window` replays a six-token prompt window around
the known `UPDATE` position from the original evaluation fixtures. It compares
the original targeted single-token arm with a forced window replay and the
window's predeclared entropy-based commit policy. The known position is an
offline diagnostic oracle. The expected and incorrect answer strings are used
only to score teacher-forced continuations, never to select the window or
commit a trial. The first continuation token uses Pass A logits in both window
arms; subsequent tokens use the respective exported KV state. Baseline parity
with the original scorer is reported for every case.

### Feedback geometry experiment

The historical-window trace now also reports the patch displacement norm for
each round, cosine between successive displacement vectors relative to the
original destination input, and the successive norm ratio. A zero displacement
has undefined cosine or ratio and is written as JSON `null`. The trace gives
token log probabilities for the expected and incorrect teacher-forced
continuations in the baseline, refined, and matched-dose arms, plus per-token
refined-minus-baseline deltas. The branches remain separate even when their
tokens or lengths differ.

The matched-dose arm reruns one round with alpha chosen so that its injected
patch displacement norm matches the final patch of the requested multi-round
run. When the final and first-round norms already agree within the declared
1e-4 times max(1, first-round norm), it keeps the original alpha to avoid a
numerically different run for a negligible dose change. The trace reports the
achieved relative norm error. This matches the injected layer-input
displacement, not CPU work, later KV changes, or behavioral effect.
It is a control for dose magnitude, not an alternate online policy. The
existing entropy commit decision is recorded but does not select the reported
forced or matched outcomes.

Predeclare the model, pair, alpha, window and case IDs before running. Use one
round as the replay parity control, then two rounds and its norm-matched
one-round control. Keep the three historical cases as mechanism probes. For a
quality comparison, select any analysis rule using development cases and run
the fixed rule once on separate held-out cases. The existing v2 fixtures
remain templated and need broader independent tasks for a general claim.
Report every case, including regressions, no-trigger results, undefined
geometry, both token paths, total answer margin, and replay wall time. A
distinct direction at comparable norm and a favorable held-out margin would support further investigation; it
would not by itself establish better reasoning or justify changing the commit
rule. No new layer choice, dose policy or FlyDelta promotion is part of this
experiment.

An initial Qwen3 0.6B Q8_0 mechanism run on the three historical cases used
two rounds with layers `7 -> 2` and alpha `0.03`. The successive patch
direction cosines were 0.999911, 0.999921 and 0.999952; norm ratios were
0.999911, 0.999921 and 0.999953. The matched one-round arm retained alpha
`0.03`, with relative patch-norm errors below 0.00009. Its margin gains were
+0.0456, +0.2222 and +0.0751, compared with two-round gains of -0.1694,
+0.2905 and -0.0053. Thus a nearly unchanged patch direction and magnitude
did not imply a nearly unchanged answer margin in these cases. This is a
three-case diagnostic, not evidence that iteration generally helps. The trace
contains both continuation branches and their per-token effects.

```sh
<build>/bin/llama-recirculation-poc \
  --model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --source 7 --destination 2 --alpha 0.03 \
  --experiment-mode historical-window \
  --eval-cases-file pocs/agent/recirculation/data/evaluation.jsonl \
  --case-id eval-object-1 --case-id eval-object-2 \
  --case-id eval-relation-1 --window-rounds 2 \
  --trace historical-window.jsonl
```

This mode is a regression probe for previously positive targeted-margin cases,
not a new held-out evaluation. Do not tune its pair, dose, window or commit
rule on these cases and then claim an independent improvement. A positive
margin change means the expected continuation became relatively more likely;
it does not imply that the model preferred the expected answer.

On 2026-11-28, the current Qwen3 0.6B Q8_0 build produced these margin changes
relative to its own baseline at two rounds (`alpha=0.03`, layers `7 -> 2`):

| Case | Original targeted | Forced window | Entropy policy | Committed |
| --- | ---: | ---: | ---: | :---: |
| `eval-object-1` | +0.0456 | -0.1694 | -0.1694 | yes |
| `eval-object-2` | +0.2222 | +0.2905 | 0 | no |
| `eval-relation-1` | +0.0751 | -0.0053 | 0 | no |

The baseline parity delta was zero for all three cases. None changed the
preferred answer. The entropy rule accepted the harmful window in the first
case and rejected the beneficial forced window in the second, so this test does
not support entropy reduction as an answer-quality commit rule. The earlier
archived targeted gains were +0.2170, +0.2536 and +0.2260 respectively; a
current-build recomputation retains their positive signs but differs in size.
The archived run used an earlier, partly uncommitted PoC snapshot. Consequently
the old and current numbers must not be treated as exact numerical replicas.
As a mechanism control, a one-round forced window gave exactly the same gains
as the current targeted arm for all three cases (+0.0456, +0.2222, +0.0751),
while the entropy rule rejected all three. Thus the two-round differences are
caused by the additional refinement rather than the six-token replay or the
continuation scorer. The eight selected recirculation and FlyDelta CTests
passed, including the new historical-window bounds contract.

## Experimental window refinement (in progress)

`window.h` tests a separate, bounded mechanism: save a full prefix context,
decode six fixed tokens, restore the prefix into a fresh trial context, and
replay those tokens with a replacement at one layer-input position. Round one
uses the same norm-matched patch as targeted recirculation. Later rounds use
the source-layer input captured during the previous patched replay:

`z_next = (1 - 0.5) * z_prev + 0.5 * (d_A + alpha * normmatch(s_prev, d_A))`.

Each round starts from the identical prefix. Only the last trial state is
eligible for continuation. The trial is selected when normalized next-token
entropy drops by at least 0.001 against the unpatched window. This is an
internal confidence proxy, not evidence that the answer is correct. The
mechanism rejects zero-norm or non-finite captures, overlarge patches, a
target without two earlier and one later window token, and more than three
rounds. Uncertainty rise and normalized activation-difference curvature can
select the first eligible target; a no-trigger result keeps baseline state.

Run the current functional mechanism check with a CPU Qwen model:

```sh
<build>/bin/llama-recirculation-poc --model MODEL.gguf \
  --source 7 --destination 2 --alpha 0.1 \
  --experiment-mode window-smoke --trace window-smoke.jsonl
```

The smoke checks identity replay, one-round equivalence with targeted replay,
one- through three-round source feedback, no-trigger handling, context-state
restore, and one continuation token. This path is a teacher-forced mechanism
check. The separate resident smoke below tests the active agent transport.
The experimental FlyDelta recall below is available through a resident opt-in;
do not interpret this teacher-forced smoke as evidence for active-agent benefit.

The smoke also exports the refined trial sequence through
`llama_state_seq_get_data`, imports it into a larger fresh context with
`llama_state_seq_set_data`, and compares the next decoded logits with a direct
refined continuation. It repeats the comparison after one more token. This
probes the state-transfer seam needed by a resident slot;
it does not yet prove transfer into a live multi-slot server context.

An opt-in agent generation request now carries `common_agent_recirculation_request`
through the resident server task. It is accepted only for an isolated CPU
completion with one slot, no speculative decoder, no streaming, no prompt
cache, and a Qwen2/Qwen3/Phi3 model. After six generated tokens have been
decoded, the server replays the fixed window in separate contexts, compares
the unpatched trial logits against the live logits, and imports the chosen
sequence KV only when the predeclared entropy rule accepts the trial. The
immediately sampled token still uses the original logits; imported KV can
affect later tokens. The final response carries attempt, target, rounds,
entropies, commit, and failure reason. This is experimental transport data,
not a FlyDelta `HELPED` result or an automatic activation.

The resident smoke exercises that request/result path:

```sh
<build>/bin/llama-recirculation-resident-smoke MODEL.gguf
```

The smoke checks execution and transport, not answer quality. A separate
locked development/holdout comparison is still needed before using the
internal entropy rule as an effective policy.

Resident CPU checks on 2026-11-28 used Qwen3 0.6B Q8_0 and Qwen2.5 1.5B
Q4_K_M. Both completed 20 generated tokens with two replay rounds and returned
the typed observation. The default number-list prompt did not pass the
0.001 entropy-gain commit rule on either model. For Qwen3 with alpha=0.2 and
the prompt `What might be inside an unopened box?`, entropy fell from
0.02198 to 0.02019 and the refined sequence was imported into the live KV.
The 20-token completion matched the baseline text. This verifies execution
and state selection, not an answer-quality benefit. During integration, the
Qwen2 Q4 trial baseline initially differed from live generation when the
trial prefixed one token at a time. Matching the resident batch-prefill and
context settings resolved the mismatch; the live/trial logit comparison
remains a required guard before import.

## Experimental FlyDelta recall

The resident host can wrap its normal inference adapter with a scope-bound
recirculation memory. The opt-in host configuration requires an existing
FlyDelta lifecycle journal and an exact model-profile fingerprint. A request
must also set `recirculation_experiment_mode`, `scope`, and a host-selected
`recirculation_context_key`. After a committed replay, the host appends an
`observed` `flydelta_experiment` record containing the layer pair, dose,
window size, target mode, rounds and entropy observation. This is not a
candidate, an active adapter, or a `HELPED` outcome.
The same host opt-in fields are available through normal
`common_agent_inference_options`. They participate in the resident model
reuse key, so switching journal or fingerprint rebuilds the host instead of
reusing stale inference state. No daemon or CLI flag enables this by default.

On a later request with the same namespace, project, model profile and context
key, the host retrieves the newest matching profile and injects
the typed recirculation request before generation. A different identity or
ordinary generation cannot recall it. The resident smoke exercises the
complete remember/recall path when the first run commits. A context key is
supplied by the experimental host; the model does not infer one from its own
answer. This mechanism is for scoped experimentation. It does not establish
that a lower-entropy replay improves task outcomes.
The default scope is one session; an explicit project memory scope permits
reuse across sessions in that project. Turn and global scope are rejected.
The model-free CTest covers in-memory isolation and reopening the existing
JSONL lifecycle journal before recall.
If a recalled profile completes its rounds but no longer meets the same
entropy-gain rule, an append-only suppression record prevents further recall
for that context. An explicitly tested different configuration can supersede
the suppression. Technical replay failures do not suppress a profile.

The 2026-11-28 resident Qwen3 smoke with the unopened-box prompt confirmed
`committed=1` on the explicit run and `recalled=1` on the next generation in
the same session/context. The default Qwen2 smoke reported `committed=0` and
`recalled=0`. Both are transport and policy observations; neither measures a
verified improvement in answer quality.

On 2026-11-28, the model-free recirculation CTests passed 5/5. CPU smokes passed
for Qwen2.5 1.5B Q4_K_M and Qwen3 0.6B Q8_0. Both showed zero maximum logit
error for identity replay and one-round targeted equivalence. Later-round
patch changes were nonzero. On the single Qwen3 smoke window, normalized
next-token entropy changed from 0.09727 to 0.09880, 0.09564 and 0.09950 for
one, two and three rounds respectively. For Qwen2.5 it changed from 0.25056
to 0.23616, 0.23767 and 0.24439. These are confidence-proxy observations
on one fixed token window per model, not answer-quality evidence.

## Targeted token trigger experiment

`--experiment-mode target-evaluate` compares four one-intervention policies on the
same held-out cases and the configuration in an existing locked selection file:
the known update position (diagnostic oracle), the final token of a host window,
the first online activation-novelty threshold crossing, and a deterministic
random position in the same window. The window is the four-token suffix ending
at the update position. It is derived from the input's `update_text` span and
does not inspect either answer. The host-window arm can coincide with the oracle
on these fixtures; their equal results are an expected control, not independent
evidence of position discovery. The random position uses a stable hash of the
case ID. Novelty uses cosine distance between consecutive destination layer
inputs captured in Pass A. Its first window token has no previous window capture
and cannot trigger. Zero-norm vectors do not trigger. A threshold crossing
commits at most one A/rewind/B replay. A no-trigger sequence keeps Pass A's KV.

For example, after a locked development selection exists:

```sh
<build>/bin/llama-recirculation-poc --model MODEL.gguf \
  --experiment-mode target-evaluate \
  --eval-cases-file pocs/agent/recirculation/data/evaluation-v2.jsonl \
  --selection-file LOCKED_SELECTION.json --novelty-threshold 0.15 \
  --trace target-evaluation.jsonl
```

The default threshold of 0.15 is exploratory, not calibrated. Fix a threshold
using development material before treating held-out results as a comparison.
`target_evaluation` rows retain no-trigger cases, selected position, margin
change, per-arm time, and log-probability deltas from the selected token onward
for both answer paths. Lag 1 uses Pass A logits and should be unchanged when
the earlier KV prefix is identical.
This tests token targeting only. It does not learn a trigger or alter model
weights, layer-pair selection, dose selection, or production inference.

Recirculation returns a normalized deep-layer representation to a shallower layer, allowing information from one token to affect subsequent tokens through the KV cache. This opt-in experiment uses frozen weights, CPU inference, one sequence and at most one extra pass per token. It can replay every token or only a selected position. The production agent does not automatically invoke this PoC.

## Dependency contract

Let `H(t)` be the committed KV prefix containing only positions below `t`.
The operation is serial, including prefill:

| Phase at input position `t` | Reads | Produces | Retained output |
| --- | --- | --- | --- |
| A | token `t`, confirmed history from B positions `<t` | provisional KV at `t`, `d(t)`, `s(t)`, logits A(t) | copied captures and A(t) |
| Rewind | synchronized provisional KV at `t` | history from B positions `<t` | provisional current KV removed |
| B | same token and position, same prefix, replacement at `d` | patched KV at `t`, logits B(t) | patched KV only |
| Next input `t+1` | token `t+1`, history including B's KV at `t` | next A logits and captures | A logits read out once |

Replacement is `beta*d + alpha*(norm(d)/norm(s))*s`. Defaults: alpha=0.1,
beta=1, one extra pass. Beta=0.9 is an explicit alternative experiment.
Source and destination are zero-based layer-input boundaries; boundary N>0
corresponds to the output of block N-1 and boundary 0 is the embedding input.
Capture is pre-patch. The Qwen2, Qwen3 and Phi3 patches are before attention and FFN.

[Recirculation v3, section 2 and Figure 3](https://arxiv.org/html/2608.17981v3#S2)
defines `z[t+1,t,d] = alpha*f(z[t,t,s]|d,t) + beta*z[t,t,d]`, with the source
rescaled to the destination L2 norm and `beta=1` by default. Its recurrence
crosses both depth and token step: the second stack for token `t` supplies the
history used by token `t+1`, and readout is from the first stack. The A / rewind
/ B sequence above has those dependencies for three successive tokens:
`A0 -> B0 KV -> A1 -> B1 KV -> A2 -> B2 KV`, while readout is `A0, A1, A2`.
This matches the paper's one-extra-iteration serial dependency schedule. The
implementation does not reproduce the paper's model setup, data, layer sweep or
adaptive policy, and does not claim its reported results.

The available [minimal PyTorch gist](https://gist.github.com/huskydoge/1ff29693e2172226ec26081f208b19d6)
describes itself as an unaffiliated serial reference, not the authors' serving
implementation. Its one-position rewind, same-token second pass, retained B
cache and first-pass readout match the dependency sequence above. It corroborates
the schedule but is not authoritative. No author-maintained executable reference
was identified. The gist uses block outputs; this PoC captures zero-based block
inputs, so the boundary immediately before block `d` corresponds to output of
block `d-1`.

The `targeted_same_token_replay` arm uses ordinary inference before and after one
identified token and runs A / rewind / B only at that position. This isolates a
critical-token deep-to-shallow transfer described as related work in the paper;
it is not the paper's every-token recurrence. A norm-matched donor control takes
a source vector from another preselected case and rescales it to the target
case's source norm without reading the donor's expected answer.

The FlyDelta two-pass helper runs independent generations and is not used.
No FlyDelta learning, activation, or promotion is connected to this experiment.

## Interfaces and failure handling

`runner.h` orchestrates a contiguous sequence through a small backend interface.
`llama-backend.h` exclusively owns a fresh CPU context with one sequence, one
thread, ordinary causal attention, and no offload or context shifting. Config
is copied and immutable. It rejects models other than Qwen2, Qwen3 and Phi3 or unsupported memory
before the first decode. The runner does not access concrete cache cells.

The experimental functions in `llama-ext.h` query replay support, validate the
current last position, and remove that position. Memory implementations reject
support by default. Standard KV accepts only a single exclusive sequence,
without SWA, cell sharing, queued copies or pending shifts. `pos + 1` overflow
and wrong/missing last positions are rejected. These operations require
exclusive context ownership and synchronized decode. They are not a general
transaction, do not restore sampling/output state, and do not undo earlier
context shifting. This runner never shifts or imports a prompt cache.

Capture and logits are copied before B. `applied` continues to mean row match
while preparing graph input; success additionally requires completed B and
committed state. Hooks are cleared on every exit. Decode failure, exceptions,
invalid captures or failed rewind discard the context; further steps fail.
A discarded context cannot be reused. Disabled execution runs one ordinary
pass without capture or patch and returns its logits.

## Build and contracts

Configure with `LLAMA_AGENT_BUILD_POCS=ON`, `LLAMA_RECIRCULATION_POC=ON` and
`BUILD_TESTING=ON`. Recirculation is off by default. Build:

```sh
cmake --build <build> --target llama-recirculation-poc test-agent-recirculation-contract test-agent-recirculation-runner test-agent-recirculation-memory test-agent-recirculation-window test-agent-recirculation-historical-window -j8
ctest --test-dir <build> -R '^llama-agent-recirculation-(contract|runner|memory|window|historical-window|analysis)-ctest$' --output-on-failure
```

Assertions remain enabled in release builds. Contracts exercise normalization,
invalid inputs/overflow, identity mixing, ordering, disabled flow, owned logits,
128 successive token operations, sequence isolation, capture/rewind/B failures,
unmatched patch, exceptions and rejection after discard. A filtered-layer KV
fixture uses production cache metadata operations without loading model weights;
it checks last-position removal, prefix preservation, re-insertion, wrong
sequence/position, shared source and mirror, multiple sequences, and SWA.
It does not prove tensor-level KV correctness. Model smokes supply that evidence.

## Prepared smokes (run separately)

Full tracing is always enabled. Keep artifacts outside the source directory:

```sh
# Model-free contracts and explicit model skip:
bash pocs/agent/recirculation/smoke.sh <build> <trace-dir>
# Model-backed correctness, 32 prompt tokens then greedy continuation to 128:
bash pocs/agent/recirculation/smoke.sh <build> <trace-dir> <model.gguf>
# Teacher-forced loss on held-out text (at least 129 tokens):
bash pocs/agent/recirculation/smoke.sh <build> <trace-dir> <model.gguf> <held-out.txt>
```

The wrapper archives revision, tracked diff and PoC sources, CTest output,
model stdout/stderr and JSONL. The binary requires `--trace` for model execution
and returns skip code 77 without `--model`. `--source`, `--destination`,
`--alpha`, `--beta`, and `--steps` (128-512) select explicit experiments.
Every token emits phase results, position, input token, norm diagnostics,
rewind/patch/commit status and timings. Comparison records include the entire
patch and measured logit differences. Failures emit a terminal failure record.
Model identity includes canonical file path, size and modification time;
build revision is accompanied by an explicit worktree-snapshot requirement.

The correctness smoke compares ordinary decode with disabled and identity
mixtures, and compares B to an independent single-pass patch-only replay
context at every token. At positions 0, 31 and the final position, a new context
replays the complete patched history and independently checks A's logits.
A new sequence is checked for isolation. A real CPU abort in B must discard
the context and reject reuse; wrong sequence/position patches must not match.
Numerical tolerance is per logit:
`abs(a-b) <= 1e-4 + 1e-4*max(abs(a),abs(b))` on CPU. Any mismatch fails.
The reference and identity arms share only model weights, not KV or hooks.

Optional `--cases-file <jsonl>` evaluates state tracking using separate fresh
baseline/recirculation contexts, greedy decoding and at most 64 generated tokens:

```json
{"prompt":"A key starts in a box. Mira moves it to a bag. Reply with only its current location.","expected":"bag"}
```

Results record full answers, exact match after trimming whitespace, time to
first token, total time, and generated count. Use externally prepared held-out
cases; keep parameter tuning material separate. The interface imposes a 64-case
budget and 512-token prompt budget. No built-in cases imply benchmark accuracy.

Teacher-forced baseline and recirculation use identical supplied tokens and
report mean loss. Loss on the built-in prompt is only diagnostic. Token timing
records support cost comparison between arms; end-to-end correctness time
includes reference overhead. The summary reports model/context/compute buffer
bytes for the active context, not peak process RSS. Multi-context reference
smokes consume substantially more memory than a single-arm inference run.

## Targeted state-transfer evaluation

The authored development and held-out case files are under `data/`. They cover
object movement, changed spatial relations and contextual meaning changes.
Each case includes its task type, exact critical token text, expected and
contrastive continuations, and a predetermined donor of a different task type
for the norm-matched control. The loader renders the
model's default system/user chat template, verifies exact token round-trip and
locates the last token touched by the update text. Ambiguous anchors, a change
to prompt tokenization at the continuation boundary, missing donors or invalid
layer settings fail explicitly.

Run Qwen2.5 1.5B in two separate invocations. Keep selection files and traces
outside the source tree. Evaluation reads the locked selection and never tunes
on the held-out cases.

```sh
mkdir -p <trace-dir>
<build>/bin/llama-recirculation-poc --model ~/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --experiment-mode tune --dev-cases-file pocs/agent/recirculation/data/development.jsonl \
  --selection-file <trace-dir>/selection.json --trace <trace-dir>/development.jsonl
<build>/bin/llama-recirculation-poc --model ~/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --experiment-mode evaluate --eval-cases-file pocs/agent/recirculation/data/evaluation.jsonl \
  --selection-file <trace-dir>/selection.json --trace <trace-dir>/evaluation.jsonl
```

Tuning sweeps three source/destination input-layer pairs derived from model depth
and `alpha` values `0.03`, `0.1`, `0.2`, always with `beta=1`. It selects the
highest mean development change in total answer margin over baseline. Ties use
the listed pair and dose order. Evaluation reports baseline, every-token replay,
targeted replay and norm-matched control. Each continuation score is the sum of
token log probabilities; the margin is expected total minus incorrect total.
Each choice is scored in a fresh context. Token traces include arm, position,
patch outcome, norms and timings. Summaries include expected-answer preference,
changes against baseline/control, elapsed scoring time and active context
memory. These small authored sets are a mechanism probe, not a general quality
benchmark.

The model-free path runs the contract tests and records a model skip:

```sh
bash pocs/agent/recirculation/smoke.sh <build> <trace-dir>
```

Model smokes run separately and require an explicit model.

## Findings and verification (2026-11-21)

Contract, runner and memory tests passed (3/3), and 128-token correctness smokes completed for Qwen2.5 1.5B Q4_K_M and Qwen3 0.6B Q8_0. Model-specific conclusions are limited to these models, their quantizations and CPU configuration. Phi3 has graph hooks but was not model-verified in these runs.

On six held-out authored cases per model, Qwen3's development-selected pair 7 -> 2 at alpha 0.03 improved mean answer margin from 0.8304 to 0.9211; four cases improved and two regressed. The geometry-ranked pair 21 -> 7 at alpha 0.2 decreased it to 0.6024. Qwen2.5's geometry-ranked pair 14 -> 4 at alpha 0.03 changed mean margin from 4.3672 to 4.3499, with three cases improving and three regressing. Expected-answer preference remained 3/6 for Qwen3 and 6/6 for Qwen2.5. Different model sizes and quantizations prevent an isolated architecture comparison.

In the repair proxy, Qwen3 had three baseline failures. The selected pair improved the answer margin by 0.2170, 0.2536 and 0.2260 respectively, without changing the preferred answer. That is a measurable continuous improvement for those comparisons; it is not evidence of a solved task or a saved reflection round. Qwen2.5 had no baseline failures, so its repair branches were not exercised. Every-token replay preserved the baseline preference count on both models.

These are exploratory results on small, authored fixed-answer fixtures. Margin is the sum of expected continuation log probabilities minus the sum for one incorrect continuation; answer length and the chosen alternative affect it. The geometry heuristic is unvalidated, and no concept-reference direction, real code-refactoring verifier, generalized benefit or resonance has been established. The full experiment traces and input hashes are needed to reproduce the reported measurements.

### Single-position propagation measurement

Evaluation reports now include `expected_path_lag_effects` for each case. All arms teacher-force the same prompt and expected continuation from fresh contexts. Only the targeted arm and its matched control patch the selected update position. Lag 1 is the probability of the token immediately after that position, read from Pass A's logits; it should have zero treatment effect because Pass B's logits are not exposed. Later lags can reflect the changed KV history. The reported curve includes the prompt suffix as well as the expected continuation, with absolute token positions and cumulative log probability gain. The incorrect continuation remains part of the existing answer-margin comparison.

`logprob_gain_vs_baseline` is treatment minus baseline log probability; positive means lower token loss. `logprob_gain_vs_control` compares with the predetermined donor control. These are paired measurements on a fixed token path, not free-generation accuracy. `relative_patch_norm` is the actual replacement displacement divided by destination norm; unavailable zero-norm ratios are null. No concept reference direction is assumed or fabricated.

Inspect the full lag curve and regressions before claiming useful propagation. Select layer pairs and dose on development data only and freeze them for evaluation. The curve does not establish resonance or general model improvement.

### Original bounded experiment

The following describes the original experiment and its historical results. The current wrapper runs the v2 comparison described below.

Run `bash pocs/agent/recirculation/experiment.sh BUILD_DIR MODEL.gguf OUTPUT_DIR` after building `llama-recirculation-poc` and the three contract targets. This serial CPU experiment archives input hashes and experiment sources, runs development selection, then evaluates locked configurations and extracts `report.json`. It can take substantial time because each answer branch uses a fresh context.

Three layer pairs are derived from model depth (source at one, two and three quarters of depth; destination at one third of source). Development captures rank pairs with `max(0, cosine) * (1 - cosine)`, averaged across cases. This heuristic favors some alignment and some novelty; it is not a validated concept or safety metric. All pairs are tested with alpha 0.03, 0.1 and 0.2, beta 1. Dose is selected separately for each pair using development answer-margin gain. Held-out reports compare the geometry-ranked pair, the first distinct fixed pair, and the overall development margin winner. Each report includes baseline, every-token, targeted and donor-control arms and the future-token gain curve. The comparison is exploratory: it does not establish statistical significance or paper equivalence. No concept reference directions are available in these fixtures, so concept-transfer cosine is intentionally unavailable.

### Conditional repair and persistent-context scenario

Use `--experiment-mode repair --eval-cases-file FILE --selection-file LOCKED_SELECTION` with `--model` and `--trace`. The locked development selection is reused; overlapping development/evaluation IDs are rejected. Each case runs baseline and an every-token persistent connection. Baseline failures (answer margin <= 0) additionally run a plain deterministic retry, a single-position recirculation retry and a reflection prompt containing the predetermined failed candidate and generic failure feedback. The expected answer is never included in feedback. All retries use fresh contexts and replay the same prefix. The report records rescues, persistent regressions and scoring costs. Plain deterministic retry is an identity control, not an exploration baseline.

This is a forced-choice proxy on the existing state-tracking fixtures, not a generated-code refactoring benchmark. A positive margin is the experimental verifier. Reflection is a prompted reconsideration, not the agent reflection runtime. `reflection_avoided_proxy` describes a margin sign change only. `retry_chain_us` includes baseline, recirculation retry and reflection when needed; it excludes external verification. The report is emitted as a `repair_scenario` JSONL event. Do not promote a persistent connection from these small fixtures.

### Low-dose response comparison (v2)

`experiment.sh` now uses the separate authored `development-v2.jsonl` and `evaluation-v2.jsonl` sets (24 cases each, eight per task type). Original fixtures remain historical references. These sets share task templates but use distinct scenarios; this is still a small mechanism probe with limited independence, not a general reasoning benchmark.

Tuning includes alpha 0.01, 0.03, 0.1 and 0.2 over the same three candidate pairs. `analysis.py lock` creates six frozen comparisons: cosine, development margin winner and `low_dose_response`, each at common alpha 0.01 and its development-selected dose. Low-dose response ranks observed margin gain at 0.01. Dividing by 0.01 yields the same ranking; this is not a general derivative or Jacobian. Higher-dose rank agreement is reported from development measurements only. Evaluation reuses identical configurations without repeated inference and rejects overlapping development/evaluation IDs. Older selection files retain their previous evaluation behavior.

`analysis.py report` reports both continuation branches, token and cumulative effects, continuation boundaries, first numerical effect lag, median and mean paired gains, negative outcomes per task type, and paired exploratory percentile bootstrap intervals (10,000 resamples, seed 1729, no multiple-comparison correction). Lag 1 equality applies only to targeted interventions with identical prior KV, not every-token replay. Curves on diverging answer paths remain separate. Retention ratios with nonpositive selected-dose gain are null; negative cases remain in the report.

The locked holdout is evaluated once per model after data and choices are frozen. Do not retune from its results. A necessary implementation correction requires an explicitly documented replacement run. Use CPU inference for comparability with historical results. No production activation, concept reference, adaptive dose or resonance model is added.

#### Preliminary Qwen3 v2 results

The locked v2 evaluation for Qwen3 0.6B Q8_0 completed on 24 held-out authored cases. The two primary comparisons below report expected-versus-incorrect answer margin changes against the no-intervention baseline. They are paired, fixed-path measurements, not free-generation accuracy.

| Method | Comparison | Source -> destination | Alpha | Mean gain | Median gain | Improved / regressed | Exploratory 95% paired bootstrap interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cosine | Common dose | 21 -> 7 | 0.01 | +0.0536 | +0.0357 | 14 / 10 | [-0.0241, +0.1300] |
| Low-dose response | Common dose | 21 -> 7 | 0.01 | +0.0536 | +0.0357 | 14 / 10 | [-0.0241, +0.1300] |
| Development margin | Common dose | 7 -> 2 | 0.01 | +0.0383 | +0.0957 | 15 / 9 | [-0.0220, +0.0938] |
| Development margin | Selected dose | 7 -> 2 | 0.10 | +0.0463 | +0.0353 | 14 / 10 | [-0.0378, +0.1487] |

Cosine and low-dose response selected the same pair and selected dose, so this run cannot distinguish their outcomes. The difference between the cosine and margin-selected pairs at the common dose was +0.0154 in favor of cosine. The intervals above include zero and are exploratory without multiple-comparison correction. The small, templated fixtures do not establish a general improvement, resonance, or saved reflection round. Qwen2.5's v2 experiment is still running; no cross-model conclusion is available yet. The original six-case findings above are a separate historical experiment.
