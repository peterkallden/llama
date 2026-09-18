# FlyDelta sideband learning — design hypothesis

## Status and purpose

**Status: V0 host seam, bounded CPU capture, two-pass experiment, explicit CLI
activation and request-scoped server-context activation are implemented;
the host evaluator, split validation, search queue, verified transition
adapters, low-rank coefficient search and experimental artifact materializer
are now implemented, while automatic runtime learning and activation are not.**
FlyDelta is not enabled by default and is not a replacement for
the current model-adaptation path. It must not be activated until it has
passed explicit evaluation and promotion gates.

FlyDelta is a proposed, small, host-controlled associative sideband for a
frozen language model. It records only host-certified experience and can
produce a bounded, reversible activation-steering overlay. The base GGUF is
never edited. It therefore complements the existing learning ladder:

```text
turn evidence
  ├─ project-/user-specific fact        -> memory
  ├─ repeatable host workflow           -> procedure / blueprint
  ├─ repeated model behavior deficiency -> corpus / LoRA candidate
  └─ repeated recognizable behavior     -> FlyDelta sideband candidate
```

The host stays authoritative. The model may generate candidates, but it
cannot certify evidence, write an active sideband, or activate one.

```text
generation != evidence
evidence   != learning update
learning   != promotion
promotion  != activation
```

The existing transaction, corpus, trainer, evaluator and LoRA-registry route
is described in [Agent model adaptation](agent-model-adaptation.md). FlyDelta
reuses its evidence and promotion principles but is not a LoRA adapter or a
training-corpus format.

The current implementation contains the model-free sparse encoder,
recognition memory, bounded delta memory, versioned JSON artifact and
compatibility checks, plus an immutable host-owned `.flyd` artifact store,
host-certified candidate, capture-manifest and sideband-evaluation contracts.
The artifact codec now has a complete schema-v2 form: the artifact carries the
bounded steering basis needed to compose an activation underlay. The explicit
activation loader accepts only an already verified, compatible v2 artifact and
never resolves files, selects a sideband or bypasses the host gate. No daemon,
profile or server inference path performs artifact discovery or automatic
activation. The CLI generation path accepts a host-prepared activation
snapshot on the per-generation request and can apply its cvec to a fresh
context. The CLI agent driver carries the same immutable snapshot through
planner, draft, reflection, continuation and tool-follow-up requests within a
turn; a later top-level turn still starts without activation unless the host
supplies a new snapshot. The server-context path maps the same snapshot to an
internal immutable task cvec, keeps it on the server slot, batches only equal
cvec identities and clears prompt/KV state when the identity changes. An
absent cvec explicitly restores the no-op state. It can also request a bounded prompt layer-input
capture on the allowlisted graph architectures that expose the required
internal staging tensor; unsupported or unknown architectures fail closed. The
generic two-pass helper discards the capture context before running the second
fresh generation. The profile/catalog contract can declare sideband identities,
and the host-only registry resolves an explicitly named, active sideband after
exact identity/layout checks. It still does not load artifact files or select
a sideband on behalf of the model.

The runtime assembly has an opt-in capture-candidate handoff. It reuses the
existing `enable_adaptation_capture` and
`adaptation_config.collection_allowed` switches and adds:

```text
enable_flydelta_capture_candidates
flydelta_model_profile_fingerprint
flydelta_capture_layout_revision
flydelta_max_capture_candidates
flydelta_capture_job_enqueue
```

Only a host-discovered `candidate_ready` source (currently a tool failure plus
a successful recovery with evidence references) is queued. The handoff
contains transaction/evidence IDs and fingerprints, not prompts, tool output
or hidden-state tensors. Reflection, research and user-correction matches
remain non-ready until a host supplies an explicit relation and verifier. The
collector is bounded, idempotent and best-effort; it cannot make the active
turn fail. A host may provide `flydelta_capture_job_enqueue` to place a
reference-only `donor_capture` job on the existing FlyDelta experiment queue.
The common worker then invokes a host callback for fresh capture/inference and
returns only redacted, identity-checked capture manifests. Queue pressure or
temporary worker unavailability remains pending host work rather than a turn
failure; the daemon dispatcher now owns an optional dedicated FlyDelta lane.
Its configured worker count is reserved from `limits.worker_count`, and status
reports distinguish a configured lane from a running lane. The daemon bootstrap
provides the queue/budget seam but does not invent an evaluator callback: a host
integration must attach the callback that resolves references and performs the
bounded work before jobs are consumed.

The adaptation boundary now also has one shared, reference-only evidence
contract (`common_adaptation_evidence`). It is a view over the existing
learning transaction ledger, not a second evidence store. A small turn router
can identify several sources in the same result: tool repair, reflection,
research, user correction and dataset/resource use. It deliberately does not
infer workflow/code or explicit plan-revision evidence from broad flags. A
host must complete a source relation with task, baseline, candidate and
verifier references before it becomes comparable evidence. Source discovery is
therefore not promotion and cannot activate FlyDelta.

The evidence contract is schema version 2. The version bump is intentional:
every host-certified relation must carry an explicit, bounded `behavior_key`
in addition to its source, scope, task fingerprint, baseline/candidate
references, verifier reference and transaction IDs. Schema-v1 evidence and
serialized evidence without that key are rejected; there is no implicit
tool-repair default or compatibility migration. This keeps a planning,
research, procedure or dataset observation from being accidentally compared
with a different behavior. Capture manifests use the same explicit
source/behavior identity when they become activation material.

## Natural dataset-question smoke

`docs/examples/agent-flydelta-dataset-question-suite.json` is the stable,
host-owned English regression suite for exercising model-facing dataset tool
selection with a compact model such as Qwen. The questions deliberately ask
for different kinds of work so the expected tool is not always
`dataset.inspect`:

| Scenario | User intent | Expected tool |
| --- | --- | --- |
| content overview | What the dataset contains at a high level | `dataset.inspect` |
| schema description | Column names, types and nullability | `dataset.schema` |
| representative rows | A bounded view of actual records | `dataset.sample` |
| numeric summary | Count, min, max, mean and standard deviation | `statistics.describe` |
| region distribution | Frequency of values in one column | `statistics.value_counts` |
| regional sales summary | Grouped sum and average | `data.aggregate` |
| filtered north orders | Declarative region filter | `data.filter` |
| largest orders | Projection, sort and bounded limit | `data.query` |
| report column shape | Rename and drop host-approved columns | `data.transform` |
| required fields check | Not-null and uniqueness validation | `dataset.validate` |
| amount outliers | Bounded IQR outlier detection | `statistics.outliers` |
| regional numeric profile | Numeric statistics grouped by region | `statistics.describe` |

The stable suite contains twelve scenarios: the first six cover the basic
dataset inspection path and the next six exercise data operations. New cases
are kept in `docs/examples/agent-flydelta-dataset-question-suite-incremental.json`
until they have been reviewed and host-verified; it currently contains four
additional projection, grouped-count and integrity scenarios. This keeps an
incremental model run from repeating the full suite while preserving the same
model-facing seam and prompt format. Each scenario includes a host-readable plan using the canonical
`dataset://local/sales` reference. The contract smoke validates the JSON,
catalog membership, plan shape and dataset bindings. The optional model smoke
uses the same file and the real compact tool descriptions, captures a bounded
layer-input window, and prints the selected tool and model output for every
question. It is observational by design: a mismatch is a repair observation,
not a HELPED result and not training evidence. A separate host execution and
verification step is required before a failed selection, repaired selection or
counterfactual can enter FlyDelta.

That host-execution bridge is now covered by a Cozo-backed companion smoke.
It creates the same bounded `dataset://local/sales` fixture, registers the
ordinary native `analysis` data adapters, and executes the canonical plan
through the real tool registry. The optional model companion then performs:

```text
natural model selection
  -> host parses, validates and executes it
  -> host-certified malformed, unknown or invalid tool call
  -> model retry with the host diagnostic and compact contract
  -> host executes the repaired call
  -> learning transactions + FlyDelta capture delta
```

An executable but different read-only tool remains `UNKNOWN` in this first
fixture: a query may sometimes be an acceptable alternative to an aggregate,
and this suite has no semantic-equivalence oracle. Such a result creates no
failure transaction and no positive FlyDelta evidence. Only a host-rejected
original call followed by a host-executed canonical repair is recorded as
`tool_use/dataset/structured_call_repair`. Six compatible natural pairs are
the existing threshold for aggregate direction candidates; the bridge itself
does not activate a `.flyd` artifact.

The model bridge now evaluates a declared fixture verification mode rather
than treating an executable expected tool as universally correct. Its default
dataset-repair mode is `normalized_call`: the registry's host normalization and
schema path is reused, and the normalized model arguments must match the
host-authored canonical repair. Fixtures may explicitly choose `tool_only` for
routing-only behavior or `result_oracle` when the canonical host result is the
actual semantic target. The margin/search objective remains separate from this
decision: it must compare a fixture-bound positive/negative pair and can guide
search, but it cannot itself create `HELPED` or learning credit.

The model-facing margin seam is explicit. A bounded inference backend may
implement `score_teacher_forced_choice()` for one fixture-bound choice slot;
the CLI/Qwen backend scores the positive and negative alternatives in fresh
contexts and returns total plus length-normalized log probabilities. The
alternatives are behavior-specific: when the tool name changes they may be
tool-slot continuations, while an argument-only repair uses complete canonical
call tails after the same prefix. The dataset repair smoke extracts the actual
failed JSON call (including fenced JSON) and compares it with the host's
canonical repaired call, so a same-tool argument repair still has a usable
margin. An unavailable backend or an unparseable/absent negative continuation
leaves the margin unavailable; it does not turn the observation into a failure
and does not block host verification.

The bridge uses the explicit `generation_boundary` capture position. It reads
the final prompt row in each turn but identifies it as “immediately before the
model generates the tool call”, rather than as an absolute prompt-row number.
Repair prompts naturally contain more context, so their absolute lengths may
differ. Ordinary `prompt_row` captures remain strict and must still have the
same absolute token index; a generation-boundary request must use
`token_index: -1`.

Run the always-safe contract check with:

```text
llama-agent-flydelta-dataset-question-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite.json
```

Run the matching host-only repair contract with:

```text
llama-agent-flydelta-dataset-question-repair-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite.json
```

Run the optional model repair bridge with:

```text
LLAMA_AGENT_MODEL=/path/to/model.gguf \
LLAMA_AGENT_THREADS=3 \
LLAMA_AGENT_DATASET_QUESTION_SUITE=docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json \
scripts/test-agent-flydelta-dataset-question-repair-model-smoke.sh
```

Validate the incremental cases independently with:

```text
llama-agent-flydelta-dataset-question-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite-incremental.json
```

The second working batch is in
`docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json` and
contains eight additional discovery, inspection, query, aggregation and
statistics cases. Run it independently while collecting candidates:

```text
LLAMA_AGENT_MODEL=/path/to/model.gguf \
LLAMA_AGENT_DATASET_QUESTION_SUITE=docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json \
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-model-smoke.sh
```

The repository wrapper uses the same model default, timeout handling and
three-thread limit as the other model smokes:

```text
LLAMA_AGENT_BUILD_DIR=build-agent-cozo \
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-model-smoke.sh
```

Use `LLAMA_AGENT_MODEL=/path/to/model.gguf` to select another model. The
underlying model-backed CTest is registered separately and has
`SKIP_RETURN_CODE 77`: it is intentionally optional when no model is present.
The contract CTest always runs and does not load a model.

The binary can still be invoked directly when a custom suite is needed:

```text
LLAMA_AGENT_THREADS=3 \
llama-agent-flydelta-dataset-question-model-smoke \
  --model /path/to/model.gguf \
  --suite docs/examples/agent-flydelta-dataset-question-suite.json
```

The third incremental batch is in
`docs/examples/agent-flydelta-dataset-question-suite-incremental-v3.json`.
It adds twelve new model-facing questions without reusing the earlier
mechanical choose-wrong/choose-right prompt. The cases vary projections,
multiple predicates, membership filtering, two-dimensional grouping,
multi-measure aggregation, grouped numeric descriptions, grouped outlier
analysis, ordered distinct results, alternate transform order and bounded
schema inspection. All twelve canonical plans execute against the shared
Cozo sales fixture, so this batch is suitable for collecting additional
host-certified repair transitions.

Validate and run the batch independently with:

```text
llama-agent-flydelta-dataset-question-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite-incremental-v3.json

llama-agent-flydelta-dataset-question-repair-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite-incremental-v3.json
```

The model smoke still evaluates each scenario as its own model turn. The
repair model smoke keeps those turns separate while aggregating compatible
layer-2 captures within the one worker invocation; use that form when the
purpose is to test the Bootstrap/Shallow/Deep evidence gate. A separate
process per scenario is useful for isolated logs but cannot by itself reach
the aggregate deep threshold.

The repair model smoke uses the same host configuration and learning-domain
policy as `llama-agent`. Pass the existing `--config` path (or set
`LLAMA_AGENT_CONFIG`) to apply `runtime.adaptation.collection_allowed` and
`runtime.adaptation.domains.families`; no FlyDelta-specific family define is
needed. The policy is applied to the canonical tool family (`data` for
`data.aggregate`), while the suite's full `behavior_key` remains the
aggregation identity. This makes it possible to focus an expensive model run
on one enabled correction family, for example:

```text
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_DATASET_QUESTION_SUITE=docs/examples/agent-flydelta-dataset-question-suite-incremental-v3.json \
LLAMA_AGENT_CONFIG=examples/agent-host-config-adaptation.json \
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-repair-model-smoke.sh
```

Without `--config`, the model smoke keeps its previous all-family behavior;
this preserves its role as an isolated experimental harness. With a config,
the host policy is authoritative: collection disabled means no selected
families, `tool_use: true` enables all tool families, and an entry under
`families` overrides the default for that family.

For a Bootstrap group, the same smoke now also exercises the worker seam:
one retained raw direction is placed in a bounded counterfactual job, then
the worker runs a fresh baseline and a fresh overlay generation and sends
both through the host verifier. This is an experimental arm, not activation
of an artifact. `HELPED` requires a host-known baseline failure and a
host-known candidate pass; `UNKNOWN` and `NEUTRAL` remain searchable
diagnostics only. Shallow and Deep continue to select larger budgets, while
the current dataset smoke does not execute Deep search yet.

### Bounded production search order

The production worker executes one bounded slice at a time. The evaluator
does not recursively run the next phase; the FlyDelta orchestrator returns a
typed `next_action`, persists the resumable state, and the host scheduler
decides when (or whether) to enqueue the next slice.

The normal rank-one path is:

```text
Whirlpool (WHERE)
  -> Bootstrap rank-1
  -> UtilityGate
  -> BootstrapZoom (local profile/scale refinement)
  -> UtilityGate
  -> AdaptiveAlphaSearch when rank-1 utility remains promising
  -> UtilityGate / AlphaResponseGate
```

`AdaptiveAlphaSearch` is not a new worker lane and does not increase evidence
capacity. It is a bounded continuation of `refine_bootstrap`, transported by
the existing Bootstrap state reference. Its expansion status is explicit:
`helped`, `saturated`, `safety_limited`, `budget_limited`, or
`upper_bound_reached`; a budget-limited search with a still-improving last
point is marked `range_not_exhausted` and must not be treated as plateau
evidence. A verified `HELPED` arm runs the bounded minimum-effective bracket
before it can enter the normal lifecycle.

If natural evidence has `effective_rank >= 2`, the next permitted capacity is
Shallow controls; it does not need to spend AdaptiveAlpha first. Otherwise a
rank-one surface may be retained/refined, or use the explicitly experimental
orthogonal/augmentation escape. Search utility may select the next region,
but only host verification creates learning credit:

```text
Evidence depth -> search capacity
Observed utility -> bounded search expenditure
Host outcome -> learning/promotion credit
```

The worker trace records the refinement kind and AdaptiveAlpha status so a
resume can be audited without replaying prior model arms.

The repair smoke partitions its captures by the scenario's explicit
`behavior_key` before assessing depth. The key describes the behavior being
learned, not merely the fact that a tool was used. Samples from
`data.query`, `data.filter`, `data.aggregate`, statistics and schema
operations therefore cannot accidentally satisfy one another's deep gate.
The current v3 supplement uses keys such as
`tool_use/dataset/data_query` and
`tool_use/dataset/statistics_describe`.

For each group the smoke reports observation count, compatible and rejected
samples, effective/stable rank, pairwise alignment, condition number,
geometry stability, the selected search depth and its bounded search budget.
Depth is incremental: Bootstrap starts with one compatible sample, Shallow
requires a small geometrically independent set, and Deep additionally
requires enough stable evidence for aggregate direction builders and
TFO-lite. Sample count alone is never sufficient; a near-collinear group
remains shallow.

This report is an aggregation/depth test, not a claim that deep model search
has run. The dataset repair smoke currently ends after producing retained
direction candidates and prints `deep_search_executed=no`. A later worker
stage can consume a Deep-ready group's retained samples and run margin,
coefficient and full-generation searches without changing the partition or
promotion rules. Only a host-verified improvement may leave the experimental
state.

`--strict` makes a non-matching model selection return failure; without it,
the smoke completes and reports mismatches so they can be inspected and fed
into the normal host-controlled repair path. The suite contains no Swedish
prompts and no synthetic “choose the wrong tool, then choose the right tool”
instruction. That keeps natural model mistakes separate from the real
host-certified corpus.

For an incremental model run, point the existing wrapper at the supplement
instead of the stable suite:

```text
LLAMA_AGENT_MODEL=/path/to/model.gguf \
LLAMA_AGENT_DATASET_QUESTION_SUITE=docs/examples/agent-flydelta-dataset-question-suite-incremental.json \
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-model-smoke.sh
```

When the supplement is mature, append its scenarios to the stable suite in a
reviewed change and replace it with the next increment. Do not treat a
model-only mismatch as learning evidence; the host-certified repair/e2e smoke
remains the step that creates FlyDelta data.

## Intended mechanism

The full research hypothesis is a sparse contextual association:

```text
selected hidden state h
       |
       v
normalize -> sparse deterministic projection -> top-k winners (phi)
                                                   |
                                                   v
                                         recognition + delta memory
                                                   |
                                                   v
                                      steering coefficients alpha
                                                   |
                                                   v
                                  per-layer steering basis B_l * alpha
                                                   |
                                                   v
                                    residual addition: h'_l = h_l + delta_h
```

Conceptually, `h'_l = h_l + B_l W TopK(R Norm(h_l))`. `R` is a fixed sparse
projection, `W` a delta-rule associative memory, and `B_l` a small per-layer
steering basis. A normalized local update can be:

```text
prediction = W * phi
error      = target - prediction
W          = decay * W + learning_rate * confidence * error * phi^T
                                      / (epsilon + ||phi||^2)
```

This is a hypothesis to test, not a claim that it avoids forgetting. It must
demonstrate held-out retention and a low false-intervention rate.

### Recognition and correction are different memories

| Part | Question | Permitted result |
| --- | --- | --- |
| Recognition | “Is this context sufficiently familiar?” | familiarity, novelty, confidence |
| Correction | “Which bounded steering coefficients are relevant?” | clipped coefficients only |

Recognition is a gate, not evidence of truth. Low familiarity, high novelty,
an artifact compatibility mismatch, failed validation, or disabled policy must
all produce a no-op.

## Safety and ownership boundaries

- FlyDelta must not relax policy, confirmation, authentication, capability
  checks, required tool execution, schema validation, resource ownership, or
  other deterministic host contracts.
- It must not treat outages, malformed provider metadata, unavailable
  resources, timeouts, policy refusals, or host-contract faults as model
  behavior defects.
- Raw user content, tool secrets, credentials and unredacted hidden states
  must not silently become sideband artifacts. Existing scope, redaction,
  provenance and revocation rules are upstream requirements.
- Updates are nearline/offline. A serving session never mutates its active
  sideband. A worker produces an immutable candidate; evaluation and explicit
  host promotion precede activation.
- Activation is profile-specific, reversible, observable and kill-switchable.
  Candidate/canary/active/retired/rejected states should mirror the current
  adaptation lifecycle without sharing its LoRA type.
- V0 compatibility is exact-model only: base model, tokenizer, template,
  architecture/layout and artifact fingerprints must match. A sideband is not
  presumed portable between Q2/Q4/Q8/FP16 variants merely by family name.

## Integration seams in the current code

### Certified evidence: reuse the learning-observation boundary

`common_learning_observation` and `common_learning_transaction` already carry
host-owned scope, signal, verification, provenance, plan/tool context and
cause classification. They are the correct upstream seam. Initial qualification
should require:

```text
cause        = model_behavior
verification = host_verified or user_confirmed
scope/policy = collection permitted
evidence     = bounded, redaction-attested, reproducible
```

The sideband references observation/transaction IDs; it does not copy
arbitrary tool output into another durable store. Do not put raw activation
buffers into the generic observation contract: they are model-specific,
potentially sensitive, and need bounded artifacts and retention policy.

### Evidence sources and alternative candidates

The source router is intentionally additive: one turn may produce more than
one source match. The current derived matches are:

| Source | Discovered from | Candidate-ready by default |
| --- | --- | --- |
| `tool_repair` | tool failure plus successful recovery | yes, when both have evidence refs |
| `reflection_alternative` | reflection or reflection learning hint | no; host verification is required |
| `research_alternative` | research result, checkpoint or verification | no |
| `user_correction` | explicit user correction signal | no |
| `dataset_resource` | plan observation containing resource/dataset refs | no |

`workflow_code` and `planning_revision` are valid shared source kinds but must
be created by a host integration that knows the exact baseline, alternative and
verifier. The router must not treat `result.revised` alone as proof of a plan
revision, because that flag can also describe a response revision.

The shared ledger also names explicit `research_verification`,
`procedure_verification` and `blueprint_verification` signals. These are useful
for routing one corpus into different source views, but they do not make those
sources FlyDelta-eligible. FlyDelta remains a narrower adapter with explicit
capture and counterfactual requirements.

The completion helper enforces the seam in this order:

```text
turn result
  -> source discovery
  -> host supplies immutable relation refs
  -> evidence contract validation
  -> counterfactual evaluation
  -> basis/gate/promotion
```

The learning observer can forward the discovered source matches through its
optional host collector callback. This callback is deliberately best-effort
and runs after the transaction append; a collector or downstream evidence
store must therefore be idempotent and must never be able to fail the active
turn. It is only a notification/hand-off seam, not an activation path.

Reflection is consequently a candidate generator, not an authority. A model
reflection may suggest several alternatives, but only the host verifier can
mark one as a useful comparison arm. An unverified alternative remains an
observation and cannot update a basis or active sideband.

Contrastive steering needs a separate immutable capture manifest with the
observation ID, profile/template fingerprints, positive/negative execution
references, verifier evidence, capture/layout revision and redaction
attestation.

### Experiment seed, job and queue — implemented contract slice

The host turns a qualified, host-verified relation into a
`common_flydelta_experiment_seed`. The seed carries the behavior key, full
scope (`namespace_id`, `project_id`, `session_id`, optional `turn_id`), model
and inference-layout fingerprints, baseline/candidate/verifier references,
the source evidence reference and transaction IDs. It carries no prompt,
tool output, credential or activation tensor. The seed is the stable
provenance boundary between the learning ledger and an experiment.

The seed is wrapped in a typed `common_flydelta_experiment_job`. V1 has four
explicit job kinds:

```text
        basis            behavior-delta references -> candidate steering basis
counterfactual   capture references + alpha grid -> host-evaluated trials
search_pipeline  capture + behavior-delta references -> composed direction/layer/scale search
delta_memory     train split examples -> DeltaMemory update
```

`delta_memory` jobs are train-only. Validation and holdout examples are
inputs to evaluation, never silently consumed by the learner. Alpha selection
only accepts an executed, host-known `HELPED` trial and applies a magnitude
penalty; `UNKNOWN` and `HARMED` trials cannot be promoted as a useful
intervention. This makes the experiment contract explicit without treating a
lower internal loss or a model self-description as evidence.

The host can run the alpha grid through
`common_flydelta_run_alpha_search()`. It runs one no-overlay baseline and one
fresh host callback per configured alpha on the same fixture. The helper only
classifies the callback's host-verifier results and delegates selection to the
conservative selector. Once a trial is selected, the host may call
`common_flydelta_training_example_from_alpha_selection()` to create the typed
train input. That function requires the selected trial to be executed,
host-known and `HELPED`; it validates the supplied sparse context, basis
revision, coefficients, split and confidence before returning an example.
This is the explicit seam:

```text
baseline + alpha trials
        -> verified selection
        -> selected training example
        -> train-only DeltaMemory batch
```

It is not a hidden training loop: the callback still owns inference and
verification, the host supplies the basis coefficients, and a queue/evaluator
must still decide whether the resulting candidate can be persisted or
promoted.

`common_flydelta_training_corpus_validate()` is the explicit corpus boundary
for train, validation and holdout data. It requires a non-empty train split,
one basis revision, the expected split tag in each bucket, unique example IDs
and unique evidence references under a bounded total. The evaluator accepts
only the train references for a `delta_memory` job; validation and holdout
remain evaluation inputs and cannot be consumed accidentally by the learner.

The typed job can be placed in the dedicated bounded FlyDelta filesystem
queue. Its lifecycle is:

```text
host-verified seed
        -> typed experiment job
        -> pending/<job-key>/job.json
        -> atomic claim to running/<job-key>
        -> evaluator/learner-owned result references
        -> succeeded | failed | cancelled
```

The queue is deliberately parallel to, rather than inside, the existing
QLoRA adaptation queue. It uses atomic staging/rename, a hash-derived safe
queue key, duplicate suppression across all states, bounded job/status
metadata and a status file containing only a safe summary. It stores the job
envelope, not captures, hidden states or raw evidence. The shared lifecycle
journal can record `flydelta_experiment` and `flydelta_result` events, but the
queue itself does not evaluate, build a basis, train DeltaMemory, promote an
artifact or activate a sideband.

This separation is intentional: the existing
`llama-agent-adaptation-worker` understands corpus-backed SFT/QLoRA jobs and
must not infer FlyDelta semantics from a different JSON shape.
`common_flydelta_evaluate_job()` now provides the one-job evaluator seam: it
resolves only typed host callbacks for counterfactual reports, behavior deltas or
train examples, validates the returned contracts, and returns typed results.
It never reads paths, sees raw prompt/tool data, performs model-side
verification or activates an artifact. The queue worker remains the lifecycle
runner; wiring a production callback and durable result store is still a
separate integration step.

The contract tests cover scope-preserving job JSON round-trips and the queue
`enqueue -> claim -> complete` flow, including duplicate and byte-bound
rejection. They do not claim that a queue entry has produced a useful model
intervention.

#### Host collection, worker execution and promotion seam

The collection bridge is intentionally a host operation, not a direct hook
from the runtime observer. The observer can report a learning signal and
transaction IDs, but it cannot invent the immutable baseline, repaired
candidate, verifier, capture or delta references required by an experiment.
The host therefore fills a
`common_flydelta_experiment_collection_request` only after verification has
produced those references. With `enabled: false` the bridge is a no-op. With
`enabled: true` it validates the evidence, scope, fingerprints, split and
reference kind, builds the typed job and enqueues it. A repeated
evidence/kind request returns `already_present`; it does not create another
job. This is the same conservative idempotency rule as the learning stores.
The request's `behavior_key` must equal the evidence relation's key; a
source match alone is never enough to cross-wire two behaviors.

The bridge accepts only reference IDs and bounded configuration. It must never
place prompts, raw responses, tool payloads, credentials or activation
tensors in the queue. A job may be `basis`, `counterfactual`, `search_pipeline` or
`delta_memory`; the reference fields must match that kind and a job may not
mix incompatible scopes or model/layout identities.

`common_flydelta_experiment_worker_run_once()` owns one queue transition:

```text
pending -> running -> succeeded | failed | cancelled
```

It claims at most one job. The callback is host/evaluator-owned: it resolves
the already-authorized references, performs bounded inference/evaluation or
learning work, and returns typed reports. The worker validates the result and
requires every counterfactual report to carry the claimed job ID. Callback
diagnostics are reduced to a safe summary in queue state. The current worker
does not resolve arbitrary paths, run a trainer by inference, or activate a
sideband; a separate operator-controlled evaluator can be added later.

For counterfactual reports, promotion classification is deliberately four
valued: `helped`, `neutral`, `harmed` and `unknown` (described as HELPED,
NEUTRAL, HARMED and UNKNOWN). A passing overlay without
a comparable passing/failing baseline is not `HELPED`. The promotion helper
can produce an `eligible` summary only when the bounded policy thresholds and
host evaluation report pass. Even then, explicit host approval is required;
approval admits and stages the manifest as `canary`, while activation remains
a separate reversible registry operation. No queue completion or successful
worker callback mutates the live model or makes an overlay active.

The complete current contract is therefore:

```text
host verifier
  -> collection request (references only)
  -> idempotent FlyDelta queue
  -> one-job worker/evaluator callback
  -> typed outcome reports
  -> eligible summary + passed evaluation
  -> explicit host approval
  -> canary
  -> separate activation decision
```

`common_flydelta_sideband_controller` makes the last lifecycle transitions
explicit: promotion can only reach `canary`, activation requires a second
explicit host approval, and retire/revoke are separate operations. It is a
thin orchestration layer over the existing promotion policy and metadata-only
registry; it does not discover a sideband or change the live model by itself.

The model-free pipeline, evaluator, corpus and lifecycle tests cover this
sequence, including a mixed outcome set, split-leakage rejection and the
final canary/active transitions. They are contract tests, not evidence that
Qwen has learned a useful correction. The composed
`search_pipeline` job now runs the existing direction → layer → scale
helpers through a typed evaluator callback. Its result retains every arm,
and `common_flydelta_collect_search_pipeline_refinement_job()` can queue the
next bounded search for an aligned UNKNOWN/NEUTRAL arm without changing the
job kind or losing lineage. Only a host-verified HELPED arm can be selected.

The queue/evaluator boundary is also connected by
`common_flydelta_experiment_worker_run_evaluator_once()`. It claims at most
one reference-only job, invokes the common evaluator, and copies its typed
counterfactual, direction, basis, search-pipeline or DeltaMemory result into
the worker result. Direction results additionally transport the bounded
aggregation snapshot, evidence-depth assessment and the corresponding search
budget. The bridge does not resolve paths, create model contexts or make a
promotion decision; those remain owned by the evaluator callbacks and the
explicit lifecycle controller.

Each worker result also carries a bounded `common_flydelta_trace` and its
machine-readable `trace_json` form. The trace is append-only search history,
not learning state: it records the job and phase, behavior and baseline
references, evidence/search rank, budgets, `next_action`, every derived arm's
layer/scale/coefficients, decision-margin values and baseline-relative
deltas, cosine/progress/leakage/shift-norm geometry, host evaluation status
and outcome. Search-pipeline slices additionally retain the existing
Whirlpool round traces (probes, recentering, radius and model-evaluation
count). The worker bounds this material and never includes prompts, model
outputs, credentials or activation tensors.

For resumable rank-one BootstrapZoom/AdaptiveAlpha work, the queue transports
only `bootstrap_zoom_state_ref`. For later rank-one plateau, orthogonal-search or
augmentation slices, it transports the separate opaque `search_state_ref`.
The evaluator's state-aware callback resolves
that opaque reference before the next bounded slice and persists a new,
immutable reference after it. `common_flydelta_configure_bootstrap_zoom_lifecycle_callbacks()` binds those two callbacks to the existing host lifecycle
store; it stores bounded phase/progress metadata, AdaptiveAlpha response
status and range state, never captures, prompts or model state. The state also
retains every host-classified BootstrapZoom arm
and its selected best safe arm (layer profile, scale, decision-margin delta
and optional geometry). Thus an aligned `UNKNOWN` is retained as an
experimental candidate for later refinement rather than lost after a worker
slice. This remains search state only: `UNKNOWN`/`NEUTRAL` grant neither
DeltaMemory learning credit nor promotion. Collection preserves the reference when it enqueues a
`search_pipeline` refinement. The host still owns the runner callback and
the scheduling decision to enqueue that follow-up job—there is deliberately
no hidden worker-local state store or autonomous daemon loop.

An orthogonal escape creates a new experimental search surface, not a new
evidence set. Persisted state records `surface_revision`, its parent
revision/reference, `search_rank`, `evidence_rank`, the surface origin and
bounded rank-two control trials. This makes the surface resumable and
comparable across worker slices without allowing a search-derived residual to
pretend that host-certified evidence became rank two. A useful surface may be
re-centered locally by Whirlpool, but it remains experimental until the normal
host-verifier and lifecycle gates classify it.

For `run_orthogonal_search`, the common layer first converts the persisted
BootstrapZoom state into a typed, reference-free orthogonal-search input. It
contains the local layer profile, bounded arm diagnostics with explicit safety
eligibility, and the selected rank-one intervention, but never prompts,
activations or a model context. The host/model adapter then performs one
bounded fresh-context orthogonal probe slice and returns its typed result
through the existing state-aware pipeline callback. The worker transports and
persists that result; it does not resolve model state or execute the
orthogonal probes itself.

#### Incremental aggregation and search depth

Search sophistication follows both evidence depth and observed subspace
utility. These are separate gates, not one combined score:

```text
evidence depth  -> which representation/search level is allowed
observed utility -> whether additional search budget is earned
```

Evidence depth is established from compatible samples, effective rank,
alignment, condition number and stability. Utility is observed from the
region/arm diagnostics: decision-margin movement, safe geometry
(`cosine`/`progress`/`leakage`/`shift_norm`) and, eventually, host-verified
outcomes. `UNKNOWN` and `NEUTRAL` may earn search priority when these signals
are useful, but they never earn learning or promotion authority.

The current implementation is intentionally incremental. Whirlpool already
uses `safe_to_continue`, `promising` and `search_score` to select a WHERE
continuation, including a useful UNKNOWN arm. The common orchestration seam
now also provides a host-neutral UtilityGate with asymmetric qualifying and
non-qualifying streaks. It decides `stop`, `retain`, `refine_bootstrap`,
`escalate_shallow`, `escalate_deep`, or `allow_tfo_lite` from normalized
decision-margin movement and bounded geometry. Evidence depth is a capacity
ceiling, not permission to skip stages: every retained region begins with
Bootstrap; a qualifying Bootstrap result may enter Shallow, qualifying
Shallow controls may enter Deep, and only qualifying Deep controls may enable
TFO-lite. A Deep evidence plan therefore records only that TFO is *permitted
by evidence* and still requires the UtilityGate after rank-two controls.

The worker does not invent a model context or invoke this decision recursively;
the evaluator executes one bounded slice and the host schedules the next job.
For resumable model-facing work, the evaluator now has an optional state-aware
search callback. It receives the previous
`bootstrap_zoom_state_ref`, advances only the remaining BootstrapZoom or
AdaptiveAlpha budget selected by the typed phase, and returns a new
reference-safe state. A host-owned resolver/persister stores
that state in the artifact/state registry; the queue carries only the opaque
reference. This lets a later sample resume the local search without repeating
completed arms while keeping activations, prompts and verifier payloads out of
the worker envelope.
The typed BootstrapZoom callback remains deliberately separate from the
generic post-Bootstrap state reference: the latter is an ownership/transport
seam for the host's plateau/orthogonal/augmentation state, not a claim that
those model-facing phases run automatically in the legacy callback. The
evaluator selects `run_search_pipeline_with_search_state` only when the job
already carries that opaque reference. An initial job without it must use the
typed BootstrapZoom callback or the legacy runner; a configured post-Bootstrap
callback cannot bypass BootstrapZoom. The selected callback consumes the
opaque reference for one bounded post-Bootstrap slice and returns the next
reference through the evaluator/worker report. If it is not configured, the
existing BootstrapZoom-aware or legacy runner remains unchanged. When the host
supplies the paired orchestration-state resolver/persister, the evaluator
converts the completed slice's region-arm margin/geometry into the common
UtilityGate input, invokes the pure FlyDelta orchestrator, persists the updated
plan/history, and returns `next_action` to the worker report. The worker still
does not execute the next phase itself. The intended invariant is:

```text
allowed_depth = evidence gate
next_depth    = utility gate + evidence gate
```

Thus six nearly collinear samples do not automatically justify Deep, and a
small independent set does not justify expensive search unless its shallow
controls show useful margin/geometry. This distinction is a design invariant.
The worker advances at most one bounded plan transition per invocation: the
host scheduler may delay or refuse the returned action for resource reasons,
but it does not reimplement the FlyDelta phase policy or skip the rank-two
control stencil.

The daemon dispatcher also accepts a host-owned
`common_flydelta_model_adapter`. This is a registration bundle: it advertises
model-facing capabilities and supplies the already-bounded worker callback.
When present, the dispatcher uses it to populate the ordinary worker callback;
worker budgeting, queue claims and result validation remain unchanged. The
production adapter must be constructed by the runtime host, where model
residency, capture/fixture resolution, fresh contexts and host verification
are available. The common worker never creates that adapter, and an advertised
capability is not evidence that a correction works.

The ordered transition at a rank-one plateau is:

```text
Bootstrap / BootstrapZoom
  -> plateau UtilityGate
  -> orthogonal search (experimental search_rank = 2)
  -> rank-two controls [1,0], [0,1], [1,1]/sqrt(2), [1,-1]/sqrt(2)
  -> UtilityGate
  -> optional local Whirlpool recenter on the new surface
  -> Deep/TFO only when evidence capacity and utility both allow it
```

The recenter is a WHERE operation on the new surface; it does not replace the
rank-two controls and does not itself grant learning credit. If no useful
surface utility is observed, the state is retained for later compatible
samples instead of spending a deeper search budget.

The worker now carries an explicit reference-only continuation between the
adaptive `WHERE` phase and evidence-driven `WHAT` work. Whirlpool may retain
the safest/highest-scoring `UNKNOWN` region arm; that is a search decision,
not a learning verdict. The host then resolves compatible samples for the
same behavior identity and selected layer, assesses evidence depth, and uses
the resulting plan:

```text
Whirlpool / region result
  -> selected WHERE continuation (HELPED preferred; otherwise safe promising arm)
  -> compatible WHAT samples for behavior + model/context + capture layout + layer
  -> evidence depth (maximum allowed depth)
  -> Bootstrap rank-1
  -> UtilityGate -> Shallow rank-2 controls, when capacity and utility allow
  -> Deep capacity + positive Shallow utility
  -> robust aggregation / Deep basis
  -> Deep controls
  -> UtilityGate -> coefficient search / TFO-lite, when utility allows
  -> margin/geometry ranking -> bounded full generation -> host verifier
```

`Shallow` and `Deep` always run the small rank-2 controls before TFO-lite.
Whirlpool therefore never supplies a coefficient basis itself: it selects
*where* compatible directions are evaluated. `UNKNOWN` may guide this hand-off
but cannot update DeltaMemory, activate a sideband, or promote an artifact.

The direction worker is incremental at the evidence boundary, not a second
durable corpus. A new host-certified relation appends its reference to the
normal learning/evidence source. A subsequent direction job may contain that
new reference together with the previously retained references. The evaluator
ingests them one at a time through `common_flydelta_incremental_aggregation`
and returns a bounded snapshot containing counts, retained sample IDs, a
running mean and variance. The snapshot is derived worker output; the
append-only learning ledger and corpus remain the source of truth. Rebuilding
the snapshot is therefore safe and idempotent: the aggregator tracks stable
sample/delta IDs and ignores duplicates during both incremental ingest and
snapshot rebuild. The worker does not need to hold the full corpus or all
activation tensors in memory.

Samples are first checked against two deliberately separate contracts:

```text
upstream admission invariants
  scope
  tokenizer fingerprint
  template fingerprint
  generation/capture semantics

aggregation identity
  behavior_key
  model/profile and execution context
  capture layout
  layer
```

Admission invariants must be validated by the host and/or asserted when the
aggregator receives a batch. Aggregation identity answers which admitted
deltas belong to one evidence geometry; it is not a semantic classifier.
Scope, tokenizer, template and generation-semantics fingerprints are checked
when supplied, but are not silently reinterpreted as behavior groups.
Incompatible or `HARMED` samples are rejected from the aggregate.
`UNKNOWN` and `NEUTRAL` samples remain valid experimental material, but they
cannot become positive learning or promotion evidence merely by being
aggregated. Retention is bounded by `aggregation_max_retained_samples`; the
configured bound must still allow the deep threshold to be reached.

Every model-facing arm also separates evaluation from decisiveness:
`host_evaluated` means the host attempted the evaluation, `verifier_known`
means that an applicable verifier produced a decisive classification, and
`host_outcome` is `HELPED`, `NEUTRAL`, `HARMED` or `UNKNOWN`. Search trace and
lineage may retain evaluated `UNKNOWN`/`NEUTRAL` arms. Only a known
host-verified `HELPED` result can create positive intervention credit. Margin
data is stored both absolutely and as a delta against the immutable
`fixture_baseline_ref`; a separate `surface_parent_best_ref` records the
comparison against the prior experimental surface.

Search depth is assessed from compatible sample count and geometry before
model-side search is expanded. Sample count alone is insufficient: effective
rank, alignment stability and condition bounds prevent several near-duplicate
samples from pretending to be a multidimensional basis.

| Depth | Gate | Model/search budget | Diagnostic role |
| --- | --- | --- | --- |
| Bootstrap | one compatible sample or effective rank about one | up to 4 region arms plus bounded rank-1 BootstrapZoom/AdaptiveAlpha arms; no coefficient search | run the smallest model experiment and collect cosine, progress, leakage, shift norm and decision margin; a useful signal may refine alpha/profile locally |
| Shallow | at least 2 compatible samples and effective rank at least 2 | up to 8 region arms, up to 4 coefficient proposals, top 1 full arm | compare a small rank-2 basis and cheap margin/geometry controls |
| Deep | at least 6 compatible samples, effective rank at least 2, stable geometry and valid condition bound | up to 32 region arms, up to 16 coefficient proposals, top 3 full arms; TFO-lite allowed | build robust aggregate/Deep basis and run Deep controls; coefficient search/TFO-lite only after positive Deep UtilityGate |

The depth result chooses a budget; it does not itself run a model or promote a
candidate. Bootstrap therefore does perform diagnostics when its small model
arms run, but those diagnostics are only search signals. The same rule holds
at every depth: cosine, progress, leakage, shift norm and decision margin may
rank or refine the next experiment, while only a host-verified baseline-fail /
candidate-pass outcome can create `HELPED` evidence.

#### Architecture invariants for phase escalation

The following rules are part of the FlyDelta architecture contract, not merely
defaults of the current worker implementation:

```text
Evidence depth -> search capacity
Utility        -> search expenditure
Host outcome   -> learning credit
```

More precisely:

* `retain` never causes automatic escalation. It preserves the candidate and
  its diagnostics for a later bounded continuation.
* `Shallow` requires genuine compatible evidence with `effective_rank >= 2`.
  Whirlpool geometry, a positive margin, or an experimental residual axis is
  not evidence rank.
* `Deep` requires both Deep evidence capacity and a positive `UtilityGate`
  decision after Shallow controls. The Deep basis is built before Deep
  controls are evaluated. Evidence alone grants capacity; it does not spend
  the model-side search budget.
* `TFO-lite` is the final escalation step. It requires Deep controls and a
  positive Deep utility decision.
* Orthogonal and augmentation searches may create an experimental
  `search_rank = 2`, but they never increase `evidence_rank` and cannot create
  learning or promotion credit by themselves.
* When natural evidence opens Shallow, pending orthogonal/plateau intent is
  cleared. An old experimental escape must not survive as a second phase flag
  alongside the natural Shallow plan.
* `BootstrapZoom` remains a rank-one local refinement. It is the normal final
  refinement while the evidence is rank-one, but it may be skipped when real
  rank-two evidence is already available and the worker can enter Shallow
  directly.

The resulting normal path is therefore:

```text
Bootstrap -> BootstrapZoom -> AdaptiveAlpha (when rank-1 utility remains promising)
  -> real evidence_rank >= 2 -> Shallow controls
  -> positive Shallow utility + Deep capacity -> Deep controls
  -> positive Deep utility -> TFO-lite
```

When rank-one evidence remains rank-one, plateau and orthogonal search are
optional escape paths rather than mandatory phases. They are useful for
experimental exploration, but the normal worker may retain/refine Bootstrap
and wait for another compatible sample instead.

This gives the worker a single evolving pipeline rather than three separate
algorithms:

```text
new evidence reference
  -> bounded re-aggregation
  -> evidence capacity: Bootstrap | Shallow | Deep
  -> Bootstrap model/geometry diagnostics
  -> optional Shallow -> Deep -> TFO transitions when both capacity and utility support them
  -> host verification and experimental retention
```

The worker still processes one queue job per invocation. Spreading jobs over
time is intentional: each turn pays only for evidence collection, while the
worker performs the bounded re-aggregation and any model experiments later.
No active sideband is mutated while this happens.

#### Rank-one plateau escape

BootstrapZoom is allowed to continue locally while the evidence remains
rank-one. If its safe arms remain in one WHERE region, improve the
decision-margin but show diminishing returns over successive rounds, the
`Rank1PlateauGate` may request an experimental orthogonal search. This is a
search escape, not an evidence-depth transition:

```text
Bootstrap/BootstrapZoom
  -> Rank1PlateauGate
  -> OrthogonalResidualSearch
  -> +/- residual probes
  -> experimental rank-2 controls
  -> UtilityGate
  -> Deep/TFO-lite only if separately earned
```

The residual fit consumes the actual bounded intervention descriptors for the
arms. Alpha-only arms cannot manufacture a second axis. Safe `UNKNOWN` and
`NEUTRAL` arms may contribute search response, while `HARMED` arms are
excluded; none of these observations increases `evidence_rank`. Decision
margin is preferred when enough arms provide it. If margin is unavailable,
the fit may fall back to a structured geometric response derived from the
normalized parallel/perpendicular shift components (`sqrt(progress^2 +
leakage^2)`). That fallback may open one experimental orthogonal probe, but
positive decision utility is still required before spending more rank-two or
Deep/TFO budget. A successful fit is marked `search_derived` and remains in
the experimental lifecycle.

If no stable residual signal can be found, the later fallback is
`RepresentationAugmentation`. Its donor context must first qualify at the
text level, and only then may its latent delta be probed. Both paths use the
same geometry bounds, fresh-context rule, host verifier and lifecycle gates.

#### Representation augmentation — implemented escape seam

Representation augmentation is the bounded escape used when a useful search
surface has plateaued but the current WHAT surface has no sufficiently useful
unexplored axis. It is not a second learning pipeline and it does not raise
natural evidence depth. The worker keeps the existing `search_state_ref` and
stores a typed augmentation state behind it:

```text
current search surface
  -> host-owned donor candidates
  -> fresh target / target+donor qualification
  -> latent delta: h(target+donor) - h(target)
  -> residualize against the permitted V0 evidence surface
  -> augmentation controls
  -> local Whirlpool recenter
  -> optional Deep/TFO-lite only after positive control utility
```

The donor contract is reference-only. A donor is never promotable by itself,
and a donor that merely supplies the exact answer is contextual assistance,
not generalizable learning evidence. A positive margin may qualify a donor for
experimental search; only a separately host-verified `HELPED` result can
provide learning credit.

The V0 control set is fixed and small: existing surface, donor residual,
positive combination and negative combination. For a rank-one parent, the V0
surface is explicitly `[natural d, donor residual]`; a failed orthogonal axis
is retained as diagnostic history but is not included in that surface. The
residual is computed as `(I - P_Q)c`, where `Q` is an orthonormalized view of
the permitted natural evidence surface. A usable donor therefore increments
`search_rank` by at most one and increments the surface revision, while
`evidence_rank` remains unchanged. Combining orthogonal and donor axes into a
rank-three surface is a later experiment and must be enabled explicitly by a
future surface policy; it is not part of V0.

Augmentation phases are persisted as `DISCOVER_DONORS`, `QUALIFY_DONOR`,
`CAPTURE_DONOR`, `BUILD_LATENT_DELTA`, `RUN_CONTROLS`, `LOCALIZE_SURFACE`,
`FULL_GENERATION`, `VERIFY` and `DONE`. `RUN_CONTROLS` precedes recentering and
TFO-lite. The state machine is resumable and uses the same append-only
experimental lifecycle store as the other FlyDelta search states.

### Verified repair materialization — implemented adapters

`common_flydelta_capture_candidate_from_transition()` is the narrow bridge
from a host-certified baseline/repaired relation to a capture candidate. It
checks source, behavior key, scope, transaction IDs and execution/verifier
references before producing a reference-only candidate. The existing manifest
builder then adds template/execution fingerprints, evidence hash, byte bound
and redaction attestation. No hidden state is captured by this adapter.

`common_flydelta_behavior_deltas_from_verified_transition()` is the matching
delta bridge. It requires the same transition, evidence, manifest and
baseline/repaired captures to agree, then delegates to the existing aligned
per-layer subtraction. This gives the host an explicit path:

```text
host repair + verification
  -> transition
  -> capture candidate / manifest
  -> two fresh captures
  -> per-layer behavior deltas
  -> direction candidates
```

### Decision margin and low-rank search — implemented CPU seam

`flydelta-coefficient-search` is a source-neutral, forward-only search helper.
It stores total positive/negative logprob and token counts, and reports both
the total and length-normalized margin. It also builds a small orthonormal
per-layer basis with bounded Gram–Schmidt and proposes a no-op plus a bounded
positive/negative coordinate stencil. The callback may use the margin to pick
which proposals deserve full generation. The helper itself still derives
`HELPED` only from the host-verified baseline/candidate result; margin,
geometry and coefficient size cannot create evidence or promotion.

The current order is deliberately cheap:

```text
geometry -> decision margin -> bounded coefficient proposals
         -> full host generation for selected proposals -> outcome
```

No gradient, SFT, LoRA or RFM path is hidden behind this component. A future
multi-layer or model-specific adapter can compose the same coefficient vector
with the existing `B_l * c` overlay contract.

### TFO-lite coefficient search — implemented experimental seam

The coefficient seam now has two explicit strategies:

```text
coordinate (default)
    no-op plus bounded +/- coordinate arms

tfo_lite (opt-in)
    small seeded population, bounded local perturbations and a fixed
    evaluation budget for mixed coefficient vectors
```

TFO-lite is an alternative to enumerating a coefficient grid. It does not
replace direction or layer search and it never searches all three dimensions
at once. The host first selects a compatible low-rank basis and layer; the
population then searches only the coefficient vector inside that basis.

Every arm is identified by its coefficients, iteration, parent trial and
mutation kind. The baseline is always a separate no-op arm. The population is
deduplicated, seeded for reproducibility and clipped to the configured L2
trust region. The default bounds are intentionally small: this is an
experiment helper, not a background optimizer.

The margin/quality score is only a search signal:

```text
diagnostic fitness
    = decision-margin delta
    - coefficient-norm penalty
    - orthogonal-leakage penalty (when geometry is available)
    - harm penalty
```

If no decision margin is available, the host-provided quality delta is used.
Coefficient arms reuse the same `cosine`, `progress`, `leakage` and
`shift_norm` object produced by representation diagnostics; the coefficient
search does not define a second leakage metric. These values are copied into
the experimental lifecycle record, while leakage affects only diagnostic
fitness and follow-up ordering.
The host may use this score to retain or refine `UNKNOWN` and `NEUTRAL` arms,
but it cannot create `HELPED`. Only a host-verified baseline-fail /
candidate-pass result can request repeatability review or later promotion.

TFO-lite arms are written through the existing experimental lifecycle store.
They remain `experimental` and keep their lineage; they do not update
DeltaMemory, change the active sideband or bypass the champion/holdout gates.
The same seam is source-neutral and can later be used for planning, research,
procedure and other verified behaviors by supplying a different host fitness
and verifier callback.

The intended evaluation sequence is:

```text
direction candidates
        -> layer plan
        -> compatible low-rank basis
        -> coordinate baseline / TFO-lite challenger arms
        -> decision-margin ranking
        -> bounded full generation for top candidates
        -> host verification
        -> experimental lifecycle / HELPED review
```

For small rank and a narrow grid, coordinate search remains the preferred
cheap baseline. TFO-lite is useful when mixed coefficients would otherwise
make the grid grow rapidly. Comparisons must use the same seed, bounds and
evaluation budget; the implementation makes no assumption that TFO-lite is
better than the baseline.

### Experimental artifact materialization — implemented seam

`common_flydelta_build_experimental_artifact()` creates a schema-v2 artifact
from validated encoder/memory/compatibility data and a bounded steering basis.
`common_flydelta_persist_experimental_artifact()` computes the canonical hash,
writes the immutable `.flyd` through the existing artifact store and admits a
matching metadata entry as `experimental`. A retry is idempotent at the file
store; registry admission remains explicit. Experimental entries are not
profile-resolvable or active. They must still pass review, canary evaluation
and separate host approval before activation.

This also applies when a model-backed search is `NEUTRAL` or `UNKNOWN`: the
candidate and its diagnostics may be retained as an experimental challenger
for later inputs/refinement, but they cannot update the active sideband,
DeltaMemory or promotion state without a host-verified `HELPED` result.

The model-backed smoke now exercises the same experimental seam with a real
model: after the existing direction, layer and geometric scale diagnostics it
runs a bounded TFO-lite coefficient search on the selected layer/basis. The
smoke reports every coefficient arm, mutation lineage and elapsed time. This
is still evidence collection and regression measurement; a model-only
mismatch, `UNKNOWN` or `NEUTRAL` result is never promoted automatically.
The remaining work is production wiring for persisted capture/delta references
and a host callback that can supply a real decision-margin signal before
ranking expensive generation arms.

### Model-facing tool repair and dataflow contract

The small-model test surface must exercise what the model actually receives,
not only internal learning records. The compact preflight renders family
descriptions first (`dataset`, `data`, `openapi`, and others); exact tool names
and schemas are exposed only after the host accepts a family selection. The
model-facing projection removes host controls such as scan limits,
materialization, backend and timeout fields while retaining typed dataset,
resource and continuation references.

The canonical compact dataset chain is:

```text
dataset.list() as candidates
  -> dataset.inspect(dataset=$candidates.datasets[0])
  -> host resolves $from_step + $json_pointer
```

An unknown family, unknown alias, self-reference or malformed indexed
reference is a repairable host contract failure; it must not be accepted as
learning evidence. The model-facing contract test covers family selection,
OpenAPI and dataset projections, this list/inspect chain and these malformed
repair cases. Native, MCP and OpenAPI providers share the same host repair and
verification boundary; only provider provenance differs.

### Host-generated synthetic tool cases

The host can generate a bounded, deterministic fixture set directly from a
resolved `common_tool_definition`, without invoking a model. The first
generator supports missing required fields, wrong types, invalid enum values,
unexpected properties and malformed JSON. It uses the full host input schema
to create the valid baseline, the invalid mutation and the expected repair,
then re-runs the existing host schema validator to certify the result.

Each case also carries the model-facing input projection and the compact tool
description. This is intentional: a later model-backed run must give the
model exactly the contract it would receive during normal inference, rather
than exposing the full host schema by accident. The case can therefore be
used as a fixture for both sides of the seam:

```text
host tool definition
  -> full-schema mutation
  -> host validator: invalid mutation + valid repair
  -> model projection + compact contract
  -> optional model repair attempt
```

The exported synthetic JSONL is test/corpus material, not model evidence. A
host-generated invalid call does not demonstrate that a model made a mistake,
does not create a FlyDelta capture pair and must not be promoted to a learning
candidate by itself. Only a later model attempt, followed by host
verification and (for FlyDelta) a valid counterfactual experiment, can cross
that boundary. Synthetic cases are consequently useful for contract
coverage, controlled model experiments and negative examples, while keeping
`Generation != Evidence != Learning != Promotion` intact.

When a mandatory tool step fails while later steps are still pending, the
runtime gives the failed step precedence: it enters reflection instead of
deferring the whole plan as an incomplete continuation. The CLI reflection
contract exposes `reset`, `retry` and `replace_steps`, and removes `accept`
from the decision enum while a mandatory failure remains. Reflection also
receives a bounded copy of the host failure and its `repair_context`; ordinary
successful observations continue to use the smaller completed-result view.
For built-in materializing data tools, the host may remove a leaked
step-level `mode:"tool"` from the argument object before strict validation.
This is a narrow structural canonicalization, not permission to ignore
arbitrary unknown tool arguments.

### Promotion: parallel sideband registry

`common_learning_adapter_registry` is LoRA-specific. FlyDelta should use a
parallel sideband registry, or a future generic overlay registry with an
explicit `kind`; it must not masquerade as a LoRA manifest.

A FlyDelta manifest needs: format/kind, ID and revision, artifact SHA-256,
base architecture and exact model fingerprint, tokenizer/template fingerprints,
inference-layout revision, layer dimensions, encoder configuration fingerprint,
strict byte/dimension bounds, source/evaluation references and lifecycle state.
The artifact must contain neither credentials nor unbounded raw conversation
content. Import must reuse the established containment, symlink, byte-bound,
hash and atomic-write discipline.

The implemented `common_flydelta_sideband_registry` is metadata-only. Admission
validates the artifact hash, model/layout identity and dimensions. A manifest
must move through `candidate -> canary -> active`; activation requires a passed
evaluation. Profile resolution is explicit and rejects missing profile entries,
inactive sidebands, mismatched model/tokenizer/template/architecture/layout
identity, or mismatched dimensions. The caller then loads and verifies the
referenced artifact and passes its bounded basis payload to
`common_flydelta_prepare_activation()`. There is no implicit “first sideband”
selection.

The separate `common_flydelta_artifact_store` is the byte/artifact seam. It
stores the existing canonical JSON codec under an absolute host-owned root and
accepts only normalized relative `.flyd` paths. Reads and writes are bounded,
reject symlinks and traversal, verify the artifact schema and canonical
`sha256:<64 hex>` content hash,
and install new files through a temporary file followed by an atomic rename.
An identical retry is idempotent; a different artifact cannot replace an
existing path. The store does not decide lifecycle status, evaluate a
candidate, or activate a sideband. TTL, scope ownership, revocation and a
persistent registry/journal still belong in the next lifecycle integration
sweep.

The sideband manifest now carries an explicit `namespace_id`, `project_id`,
optional `expires_at_epoch_ms` and revocation reason. Its in-memory registry
enforces the lifecycle transitions `candidate -> canary -> active -> retired`
and supports explicit `revoke(reason)`. Expired entries cannot enter canary or
be resolved; revoked entries cannot be resolved. Manifest JSON includes these
fields so a later persistent registry can reuse the same contract. The current
registry is still process-local: persistence of registry transitions, durable
scope ownership and a recovery-safe revocation journal are intentionally not
implemented yet.

### Model profile and residency: resolve immutable metadata

`common_agent_model_profile` currently describes a base model plus LoRA
overlays, and the residency manager caches handles by profile. A FlyDelta
selection belongs beside—not inside—the LoRA list. For example, a future
profile may conceptually contain:

```json
{
  "id": "qwen-agent-canary",
  "base_model_id": "qwen-agent",
  "adapters": [{ "adapter_id": "agent-lora-v2", "scale": 0.8 }],
  "sidebands": [{
    "sideband_id": "flydelta://sideband/tool-repair-v1",
    "scale": 0.12
  }]
}
```

This is the profile/catalog contract for identity and scale; it is not an
artifact-loading configuration by itself. Per-turn coefficients and the
mutable inference context never belong in the resident model handle. The
residency cache key includes declared sideband identity and scale; the registry
additionally checks the resolved artifact hash and compatibility metadata.

### Current llama.cpp control-vector seam: static V0 is viable

llama.cpp exposes `llama_set_adapter_cvec(ctx, ...)`, which adds configured
control vectors at layer residual outputs. Agent CLI generation creates a
fresh `llama_context` per turn, applies overlays, then decodes. This gives a
safe experimental V0:

```text
host-side context features or embedding
       -> recognition + coefficient prediction
       -> compose cvec: delta_l = B_l * alpha
       -> create fresh llama context
       -> set cvec before first decode
       -> normal prompt and generation
```

V0 is **context-selected static steering**, not the full hidden-state loop.
It does not read an intermediate activation and alter the same decode pass. It
is still sufficient to validate artifact compatibility, gating, composition,
rollback, latency and interference.

The cvec must be set before prompt evaluation on a fresh context. Changing it
after prompt tokens are in the KV cache would mix unsteered and steered state;
that is not valid V0 behavior. Contexts must not be shared across cvec values.

### CLI V1 bounded capture and two-pass seam — implemented

The CLI can opt in to a bounded prompt capture request. The request carries
explicit layer indices, a prompt token index, model/layout identities and a
byte bound. The host validates the request before enabling the internal
llama.cpp staging tensors, and validates the captured dimensions and finite
values before exposing the snapshot. The architecture guard is deliberately
fail-closed and is currently an explicit list of graph implementations known
to publish the staging tensor; it must be kept in sync with llama.cpp graph
changes.

`common_flydelta_run_two_pass()` provides the first end-to-end host seam. It
runs a capture-only first pass, builds an immutable host-owned activation
snapshot, discards the first inference context, and runs the normal second
pass on a fresh context with the optional static overlay. The helper does not
persist state, train a model or give the model authority over the capture.
This is an experiment and bounded data handoff, not automatic FlyDelta
learning.

### Dynamic same-pass hidden-state capture/injection is a later seam

The public API has output embeddings and an evaluation callback; internal
graphs label layer outputs such as `l_out`. The scheduler callback is an
observation/debug facility, may transfer backend tensors to host memory, and
is not a safe public dynamic-intervention API.

The full loop needs an explicitly staged route:

1. **V1 two-pass experiment:** bounded capture/prefill, derive a code,
   discard that context, compose cvec and run a second fresh generation. This
   avoids mixed KV state but doubles prompt work.
2. **V2 optional llama.cpp hook:** a versioned, device-aware activation
   observer/intervention contract at documented residual slots. Begin CPU-only;
   unsupported backends must no-op. GPU support needs kernels and scheduling,
   not per-layer host copies.

The CLI V1 capture is CPU/internal-staging scoped and is not a stable public
llama.cpp API. Neither same-pass route belongs in generic inference until
static V0 and the two-pass experiment show useful, bounded results.

## Recommended implementation sweeps

### 1. Model-free associative core — implemented

Add `common/agent/adaptation/flydelta/` with a deterministic sparse encoder,
bounded delta/recognition memory, novelty scorer, artifact codec and strict
compatibility validator. Use synthetic vectors only. Tests cover determinism,
tie breaking, update/prediction, clipping, novelty no-op, tampering and atomic
candidate snapshots.

### 2. Host-certified candidate and evaluation contracts — implemented

Build sideband candidates from qualified existing transactions and separate
capture manifests. Prove with fixtures that host-contract, policy, resource and
transport failures do not qualify. Measure verified success, held-out
retention, regression/interference, false intervention, no-op rate, runtime
intervention rate, latency/token and overlay bytes per improvement.

### 3. CPU-only static cvec V0 — explicit seam implemented

The CLI generation path now has an agent-owned overlay seam between context
creation and the first `llama_decode`. A host-prepared activation snapshot is
carried by the generation request through conversation, continuation and tool
follow-up calls. The CLI validates the snapshot against model
embedding/layer dimensions, scales a bounded cvec, applies it to the fresh
context and fails closed on invalid data. An absent or disabled snapshot is a
no-op, and a snapshot is never stored on the inference object; this prevents
one turn's overlay from leaking into the next turn. The current API is an
explicit seam, not a profile/configuration feature; the caller that eventually
wires it must gate it to the CPU V0 experiment. Server-context, residency
reuse, CUDA/Vulkan, Android and dynamic capture remain disabled until
independently tested.

### 4A. Host-owned counterfactual contract — implemented

`flydelta-experiment.*` defines an immutable experiment fixture, two trial
records and a host callback seam. The seam runs the same fixture once with the
baseline profile and once with the candidate overlay. It does not perform
inference itself, persist learning or accept a model self-assessment as
verification. Each known result must carry host evidence.

The report classifies the pair as:

```text
baseline fail + candidate pass -> HELPED
baseline pass + candidate fail -> HARMED
baseline pass + candidate pass  -> NEUTRAL
unknown verification or both fail -> UNKNOWN
```

This is intentionally an individual causal experiment, not yet an aggregate
promotion decision. The fixture carries task, model, tokenizer, template,
tool-catalog, resource-snapshot and verifier fingerprints so a later runner
cannot silently compare different environments. The quality delta is recorded
but never overrides the host outcome classification.

### 4B. Repair transitions, contrast sets and intervention credit — implemented

`flydelta-evidence.*` connects a failed host-verified attempt to its repaired
attempt using existing learning transaction IDs and one immutable task
fingerprint. It builds bounded positive/negative pairs only from aligned
transitions in the same namespace, project and session. Main-model descriptions
can remain metadata for later clustering, but host evidence decides correctness.

Intervention credit is derived from the 4A counterfactual report and preserves
positive and negative evidence (`HELPED`, `HARMED`, `NEUTRAL`). `UNKNOWN` is
never eligible for a learning update. A successful repair by itself is not
treated as proof that a future overlay caused the improvement.

### 4C. Capture and bounded basis — implemented

`flydelta-basis.*` accepts a bounded, host-captured and redacted behavior
delta. A tool repair is only one possible source of that delta.
The builder normalizes and clusters similar directions. `HELPED` may create or
update a direction; `HARMED` and `NEUTRAL` can only add evidence to an existing
similar direction; `UNKNOWN` is a no-op. This makes `WHAT` (basis) separate from
`WHEN/HOW MUCH` (recognition and delta memory).

Each delta carries its model-profile fingerprint, execution-context
fingerprint, capture-layout revision and layer index. A basis builder is
configured for one model/layout/context tuple and never clusters vectors from
different layers. This is a compatibility guard, not a substitute for the
later runtime artifact validation. The capture manifest is schema version 3;
the context fingerprint is required so dynamic MCP/OpenAPI catalogs and other
host execution contexts cannot be mixed accidentally.

The CLI capture hook and two-pass host helper now provide a bounded way to
produce the activation inputs for a manifest on supported graph
architectures. The basis component itself still does not run inference or
choose what is correct: the host must produce a valid capture manifest and
counterfactual report first. The objective remains minimum intervention,
repeatable verified lift and minimum collateral change.

### Experimental artifact revisions

An offline FlyDelta search may retain an immutable `.flyd` revision in the
normal artifact store and sideband registry before it has promotion evidence.
Its payload can contain a candidate steering basis and an experimental
DeltaMemory. The append-only search lifecycle records lineage, geometry,
decision-margin diagnostics and host outcomes by reference.

Such a manifest has status `experimental`. It is searchable by the host but is
never resolvable by a model profile, cannot enter canary directly and cannot
become active. A host must explicitly call `promote_experimental` after an
experimental evaluation; the revision then becomes `candidate` and must still
pass the ordinary canary and activation gates. This keeps deliberately
overfit two-pair experiments useful without giving them serving authority.

### 4D. CPU direction search — implemented

`flydelta-direction-search.*` is the inexpensive `WHAT` search between
host-certified capture materialization and model-side counterfactual
evaluation. It accepts a bounded set of aligned behavior deltas for one
behavior, model profile, capture layout and layer. It has two explicit modes:

```text
learning
    only HELPED + learning-eligible samples

experimental
    HELPED, NEUTRAL and UNKNOWN samples may propose bounded challengers;
    HARMED samples remain rejected
```

Learning mode is the default and keeps the strict promotion boundary. The
experimental mode exists for search, so a valid delta does not need to have
already earned `HELPED` intervention credit. Every candidate emitted in that
mode is marked `experimental_only`; it may be evaluated, retained and refined
by the existing lifecycle, but it cannot update DeltaMemory or become
promotion evidence. The host still supplies the outcome/credit and this
builder never infers correctness.

The component emits up to three comparable candidates:

```text
raw_repair
    first normalized delta; control candidate

normalized_trimmed_mean
    normalized deltas, low-centrality samples removed, then mean-normalized

diagonal_whitened_mean
    the retained mean weighted by inverse per-dimension variance plus ridge
```

Centrality is measured by each sample's median pairwise cosine. A configurable
alignment threshold and trim fraction make the aggregate robust to one or a
few anomalous repairs. The diagonal-whitened candidate is deliberately not a
full covariance LDA implementation: it is deterministic, bounded and cheap
enough for the host, while avoiding an unstable dense matrix for small sample
sets.

The host uses six natural host-certified repair pairs as the Deep aggregate
threshold. It is not a blanket ban on model-facing experiments below six
pairs. Bootstrap may run a bounded rank-1 arm from one compatible sample, and
Shallow may run a bounded rank-2 experiment when two to five samples are
actually independent enough. The six-pair threshold applies when the host
wants the larger aggregate WHAT builders and Deep coefficient search; it is a
cost/evidence gate, not a claim that six examples are sufficient for
promotion.

This is a candidate builder, not a verifier or learner. It does not run model
inference, inspect tool choice, create `HELPED`, update `DeltaMemory`, write a
sideband or activate an overlay. The next host step must evaluate the emitted
directions with the normal bounded scale/layer search. Only a host-verified
baseline-fail/candidate-pass result can produce intervention credit.

The same separation applies when the input is `UNKNOWN` or `NEUTRAL`: those
observations can seed an experimental direction search and its lineage, but
they remain search material until a later host counterfactual establishes a
causal outcome. A strict learning job continues to reject them. This lets the
cheap search use information already collected without confusing “worth
trying” with “proved helpful”.

The raw candidate remains important as a control. A direction that looks good
under CPU alignment but does not improve the host-verified tool-choice result
must remain an experiment result, not learning evidence. The same contract can
later be used for `reflection_alternative`, `planning_revision` or
`research_alternative`; the behavior key and evidence source keep those
domains separated.

The Qwen model smoke is wired to this seam for its L2 scale experiment. Its
current fixture contains one natural host-certified repair pair, so the smoke
reports and evaluates only `raw_repair` and marks `deep_ready=no`; it does not
manufacture additional contrast samples. This is the Bootstrap path, not an
exception to the model-facing contract. Once two sufficiently independent
pairs are available, the same smoke seam may evaluate the Shallow rank-2
path; once six stable natural pairs are available, it may evaluate the Deep
aggregate candidates under the normal bounded scale budget. The CPU contract
test exercises the multi-sample trimmed and whitened paths independently of
model inference.

The direction search is also available through the existing experiment queue
as job kind `direction`. The job contains only bounded behavior-delta
references; the evaluator resolves those references, builds the same WHAT
candidates and returns them to the worker. A direction worker result must
contain at least one validated candidate. This keeps queue execution,
reference resolution and result validation on the same authority boundary as
the existing basis, counterfactual and DeltaMemory jobs.

Two generic candidate builders share the same candidate format:

```text
token_margin_direction
    normalized positive-output minus negative-output row

execution_boundary_prototype
    normalized mean(positive captures) minus mean(negative captures)
```

The caller supplies tokenization/unembedding rows or host/model boundary
captures. Consequently these builders are not OpenAPI- or tool-specific and
can be used for tool choice, planning, research, structured output and
procedure behavior. They are still only direction material: the existing
layer search, scale search and host verifier decide whether a direction is
useful. No margin or geometric signal can create `HELPED` on its own.

The model-facing decision boundary uses the same generic seam. A bounded
`decision_pair` contains two host-selected alternatives plus tokenizer and
template fingerprints; it is not limited to tool names. The host adapter finds
the first divergent token and may resolve the corresponding output-head rows
through a callback. The existing `token_margin_direction` then computes:

```text
normalize(U[positive_token] - U[negative_token])
```

This is an output-space decision signal, not automatically an intermediate
layer injection. It can rank or guide the existing layer/coefficient/scale
search. The existing `execution_boundary_prototype` remains the generic
positive-minus-negative capture direction, and coefficient search can combine
it with another direction without introducing a tool-specific candidate type.
A missing common prefix, ambiguous divergence or incompatible fingerprint
fails closed.

Search objective, verification objective and learning authority are separate
contracts. A model-facing margin is search evidence only; it must be bound to
the same `behavior_key`, fixture/surface revision and host-selected decision
pair as the arm being evaluated. Implementations should retain the pair
identity, positive/negative references and a typed scope such as
`tool_choice`, `normalized_call`, `selected_arguments` or
`full_continuation`. A margin improvement may guide Whirlpool, BootstrapZoom,
UtilityGate, augmentation or coefficient search, but it cannot create
`HELPED`.

The host verification chain is separate:

```text
generated arm
  -> parse and schema validation
  -> host normalization and execution
  -> routing check
  -> semantic verifier (when the fixture supplies one)
  -> host_outcome
```

For tool-oriented fixtures the supported verification modes are
`tool_only`, `normalized_call` and `result_oracle`. The first accepts the
expected executable tool; the second compares host-normalized arguments with
the host-authored canonical repair; the third compares the bounded execution
result with the fixture oracle. A different executable tool remains
`UNKNOWN` for a tool-selection behavior unless the fixture explicitly declares
an equivalence policy. Schema-valid or executable is therefore not synonymous
with semantically correct. Only the host outcome, under the declared
verification mode, can provide learning credit.

Host normalization must be semantic for the fixture family: equivalent
predicate forms, measure ordering or irrelevant defaults may normalize to the
same call, while a different filter predicate or aggregate measure must not.
This reuses the runtime registry/schema path rather than creating a second
validator in FlyDelta.

### 4H. Verified candidate-to-delta materialization — implemented

`flydelta-capture.*` now has a reference-only factory from a qualified,
host-verified capture candidate to a capture manifest. The factory requires
the candidate and evidence to share the same source and transaction, requires
explicit redaction attestation, and records bounded template/evidence
fingerprints and the positive/negative execution references. It does not
capture tensors, copy prompts or tool results, or make the candidate eligible
for activation. The byte count is supplied by the host for the already
bounded capture payload and is checked again by the manifest validator.

`common_flydelta_behavior_deltas_from_captures()` is the next pure materializer:
it accepts the manifest's baseline and candidate hidden-state captures, verifies
the exact model/layout/token/layer/dimension alignment and pair byte count,
then emits one `candidate - baseline` delta per layer. The existing behavior-delta
validator rejects non-finite and zero deltas. The helper does not run
inference, infer which arm was correct, persist an artifact or update a basis;
the host must first provide the verified relation and counterfactual credit,
then explicitly pass the resulting delta to the basis builder.

The current host-level repair smoke covers this complete bounded seam:

```text
verified repair relation
  -> capture manifest
  -> baseline/candidate captures
  -> per-layer behavior delta
  -> HELPED credit
  -> basis direction and gated overlay
```

This is still preparation, not automatic learning. Capture-manifest source,
scope,
TTL, artifact persistence and revocation remain lifecycle work; the current
manifest uses observation/transaction and execution references but does not
claim to implement those policies by itself.

### 4G. Verified source handoff — implemented

The capture candidate collector has an explicit
`observe_verified_relation()` path for reflection alternatives, research,
user corrections and other sources that broad runtime discovery must not
promote automatically. It requires matching source kinds, valid reference-only
evidence and host verification. An unverified relation is a successful
no-op; it does not enter the queue. A verified relation becomes a bounded
candidate containing transaction/evidence references and model/layout
fingerprints, never raw prompts, outputs or hidden tensors. Candidate IDs
include the source kind so multiple qualified sources from one transaction do
not overwrite each other. This handoff is still only candidate preparation;
it does not capture, train, approve or activate an overlay.

### 4D. Explicit gating and promotion — implemented

`flydelta-gate.*` provides the host-owned no-op gate for explicit opt-in,
approved candidate status, familiarity/novelty thresholds, basis availability
and bounded scale. A low-confidence or unknown context returns a stable no-op
reason rather than an activation fallback. Only after this decision should the
existing static cvec seam receive data. `flydelta-promotion.*` aggregates
individual counterfactual reports and marks a summary `eligible` only after
bounded trial, help, harm and unknown-ratio thresholds pass. It never marks a
summary `approved`; explicit host/curator approval is still required. A
non-qualifying summary remains `observed` so negative evidence is retained.

The gate and promotion code do not load artifacts, capture activations, persist
learning or wire the server. Those remain separate runtime work.

### 4E. Basis-to-overlay composition — implemented

`flydelta-overlay.*` is the next small seam after the gate. It composes the
host-selected, model/layout-compatible basis directions and bounded
coefficients into the existing static cvec overlay contract. The composition
is deliberately a pure operation: it does not resolve an artifact, load a
model, capture activations, persist state or start inference.

The layout is explicit and matches `llama_set_adapter_cvec`:

```text
layer 0                 -> no cvec slot
layer 1, [0 .. n_embd)  -> first slot
layer 2                 -> second slot
...
```

The helper creates a full zero-filled buffer for layers `1..n_layers-1`, adds
each direction multiplied by its coefficient and the gate scale, and runs the
existing overlay validator before returning. A refused gate produces a clean
empty no-op even if there is no artifact. Any dimension, layer, coefficient,
finite-value or byte-bound mismatch fails closed. The resulting overlay uses
scale `1.0` because the gate scale has already been applied during composition;
this avoids applying the same scale twice in the CLI seam.

This is still not automatic FlyDelta activation. A future caller must provide
an approved artifact, an explicit gate decision and a fresh compatible model
context. Server/resident-model wiring, dynamic hidden-state capture, CUDA/
Vulkan and Android remain outside this sweep.

### 4F. Host activation preparation — implemented

`flydelta-activation.*` is the single host-owned preparation seam for a
future caller. It performs the order explicitly: evaluate the gate, require
candidate/model/layout metadata only after an approved decision, then call
the bounded basis-to-overlay composer. A rejected gate returns a successful
empty result; invalid active metadata or composition bounds fail closed.

This keeps CLI and future server integrations from independently repeating
approval, no-op and compatibility conditions. The result is still an
ephemeral request value. It does not resolve a registry artifact, persist a
decision, alter a resident model or make FlyDelta active by itself.

### 4F.1. Schema-v2 artifact to activation underlay — implemented

`common_flydelta_prepare_activation_from_artifact()` is the explicit bridge
from a complete v2 artifact to the existing activation request. It requires:

```text
schema_version == 2
content_hash == common_flydelta_artifact_hash(artifact)
artifact compatibility == expected model/profile compatibility
sparse code dimension == artifact encoder expansion dimension
artifact bounds and basis dimensions == valid
```

The helper reconstructs the bounded delta memory from the serialized weights,
predicts coefficients for the host-supplied sparse code, converts the
serialized per-layer basis to the cvec composer and then invokes the existing
host-owned gate. A refused gate is a successful empty no-op. Any hash,
compatibility, dimension, finite-value or byte-bound mismatch fails closed.

The v2 payload is intentionally compact and explicit:

```json
{
  "schema_version": 2,
  "encoder": { "seed": 42, "input_dim": 2, "expansion_dim": 2,
               "fan_in": 2, "winners": 1 },
  "memory": { "expansion_dim": 2, "target_dim": 1,
              "max_abs_weight": 1.0 },
  "model_n_embd": 4096,
  "model_n_layers": 32,
  "il_start": 1,
  "il_end": 31,
  "steering_basis": [
    { "layer_index": 12, "values": [/* n_embd floats */] }
  ]
}
```

The basis count must equal `memory.target_dim`, each direction must use the
model embedding dimension, and every direction must fall inside the declared
layer interval. The artifact hash covers this payload. Schema v1 remains a
readable/serializable memory-only artifact for the first experimental slices,
but it cannot be activated through this loader because it has no complete
steering basis. There is no automatic v1-to-v2 migration.

### 4G. Per-turn propagation and backend boundary — implemented

The resolved activation is held as an immutable shared snapshot on
`common_agent_request` and propagated to each generation request. The CLI
runtime driver also preserves the snapshot while it rebuilds requests between
planner, draft, reflection, continuation and tool-follow-up slices. This
avoids copying full cvec buffers during continuation and makes the lifetime
explicit: one top-level turn may reuse its snapshot, but a later turn starts
without one unless the host supplies a new decision. The CLI consumes the
snapshot on its fresh context. The server-context adapter maps it to the
generic `server_task_cvec` field. Server admission validates model dimensions
and byte bounds; slot scheduling prevents different cvecs from sharing a
context-wide cvec, and a cvec change invalidates the slot's prompt/KV state.
This is static, turn-scoped activation only: it does not make hidden-state
capture request-scoped, inject a direction mid-decode, resolve `.flyd` files in
the server, or change host scope and promotion policy. If speculative decoding
is active, the same cvec is validated against and applied to the draft context
when one exists; a dimension/layout mismatch is rejected at task admission.

This boundary is intentional. The current server-context adapter is
request-scoped: the public session-host turn request may carry one immutable
activation snapshot, which is copied into the runtime request and mapped to
the server task cvec. It must not mutate startup `common_params` or resident
KV state. A cvec change still requires a fresh prefill/context according to
the server admission rules above.

The contract coverage is split deliberately: `test-agent-flydelta-activation`
checks v2 hash/basis loading, active composition and explicit-opt-in no-op;
`test-agent-prepared-generation` checks server-task cvec mapping, dimension,
byte-bound and identity contracts; `llama-agent-inference-smoke` checks that a
host-provided activation survives the CLI driver request builder. The model-backed Qwen+Nomic smoke remains a
baseline runtime check; it does not claim a useful learned FlyDelta effect
until a host-verified artifact and a real HELPED counterfactual exist.

### Current smoke coverage

The host-level repair path is covered by
`llama-agent-flydelta-tool-repair-smoke` and the CTest
`llama-agent-flydelta-tool-repair-ctest`. Its deterministic fixture exercises
the complete control flow:

```text
wrong data.describe
  -> host-verified repair to data.inspect
  -> aligned contrast set
  -> baseline/candidate counterfactual = HELPED
  -> eligible promotion summary
  -> explicit host approval
  -> recognition-gated delta memory
  -> familiar context applies overlay
  -> novel context is a no-op
```

This is deliberately model-free. It proves the host contracts and bounded
state transitions, but it does not train a model or demonstrate that a cvec
changes a model's tool choice. The optional
`llama-agent-flydelta-model-ab-smoke` also exercises a real Qwen2 prompt
capture before running fresh baseline and candidate arms. On a model where
both arms pass, it reports `outcome=neutral`; it must not claim causal lift
without a baseline failure and a candidate improvement. The existing
`scripts/test-qwen-nomic-agent.sh` can be used separately for the small-model
inference/embedding smoke, for example:

```bash
LLAMA_AGENT_BUILD_DIR=build-agent-sqlite \
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_EMBEDDING_MODEL=/path/to/nomic-embed-text-v1.5.Q4_K_M.gguf \
LLAMA_AGENT_THREADS=3 \
LLAMA_AGENT_N_PREDICT=32 \
scripts/test-qwen-nomic-agent.sh
```

That smoke validates ordinary Qwen generation together with Nomic query
embedding. It remains separate from FlyDelta. The model A/B smoke accepts
`--model` or `LLAMA_AGENT_MODEL` and limits the documented local example to
three threads:

```bash
LD_LIBRARY_PATH=build-agent-sqlite/bin \
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_THREADS=3 \
build-agent-sqlite/bin/llama-agent-flydelta-model-ab-smoke --threads 3
```

The smoke verifies capture and both arms with a host-owned response check and
reports `outcome=neutral` when both pass. That is intentional: without a
baseline failure and a counterfactual lift, it must not claim that the overlay
helped. The A/B smoke proves model loading, registry resolution, cvec
preparation, bounded Qwen2 capture and fresh-context application; it does not
train a model or make the overlay persistent. Capture remains disabled unless
the smoke/host explicitly requests it.

The same executable accepts `--multi-arm` for a bounded CPU technical smoke.
It runs a no-op baseline plus four small scales of the same deliberately tiny,
non-learned direction through separate fresh contexts, measures each arm, and
uses the existing alpha-search selector to report the smallest verified HELPED
scale. A neutral result validates isolation and cost only; it is not evidence
against FlyDelta because the direction is not a host-certified repair basis.
The smoke intentionally does not share a baseline KV cache: cvec affects
prefill, so sharing it would invalidate the counterfactual.
A full `llama-agent` rebuild is required when shared agent/runtime libraries
have changed; otherwise an incremental executable may be out of sync with
those libraries.

The optional `llama-agent-flydelta-runtime-smoke` is the first end-to-end
runtime wiring check. It uses the public session-host turn contract and runs
exactly three fresh arms against the same prompt: a no-op baseline, a small
activation, and a larger activation. The model profile is resident between
arms, while each arm receives a fresh runtime context so the cvec affects
prefill consistently. The backend is selectable with `--backend
server-context|cli` (or `LLAMA_AGENT_BACKEND`), and the local example keeps
the thread limit at three:

```bash
LD_LIBRARY_PATH=build-agent-cozo/bin \
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_THREADS=3 \
build-agent-cozo/bin/llama-agent-flydelta-runtime-smoke \
  --backend server-context --threads 3 --n-predict 16
```

This smoke proves activation propagation through the host, model residency,
backend execution, and host verification. Its result is intentionally named
`execution_verified_neutral`: a tiny synthetic direction is only a wiring
probe, not a learned repair basis and not evidence of a HELPED outcome. A
real HELPED result still requires a host-certified baseline failure, a
verified repair, and counterfactual evaluation showing that the overlay
caused the improvement. The CTest entry is registered with skip code 77 when
no model is supplied; the model-backed command above is the explicit local
verification path.

The optional `llama-agent-flydelta-repair-model-smoke` goes one step further
and exercises the complete model-backed bridge:

```text
host-controlled failed selection (data.describe)
  -> host-controlled repaired selection (data.inspect)
  -> aligned hidden-state capture pair
  -> candidate-minus-baseline behavior delta
  -> host-approved basis direction
  -> one train-split DeltaMemory example
  -> three fresh inference arms
       baseline (no-op), scale 0.01, scale 0.02
  -> host verification of every arm
```

It uses the regular CLI generation adapter and creates a fresh context for
every arm. This is a real model/capture/training/inference test, but it is not
automatic turn-time learning. The successful two-pass repair is not treated as
causal evidence for the later overlay; only the three-arm result can produce
`HELPED`. If the overlay arms fail alongside the baseline, the result remains
`UNKNOWN` and the smoke reports no selected overlay. This prevents a
host-verified repair from becoming a false causal training label.

Example:

```bash
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_THREADS=3 \
build-agent-cozo/bin/llama-agent-flydelta-repair-model-smoke \
  --n-predict 96 --threads 3
```

For the model-facing augmentation seam, the same smoke can additionally be
run with `--force-plateau-escape --force-representation-augmentation`.
This is an evaluation mode only: it creates a second host-certified donor
context, residualizes its capture against the selected rank-one surface and
runs the four augmentation controls. It may report `recenter_augmented_surface`
from the UtilityGate, but it cannot change `evidence_rank`, create learning
credit or promote a sideband. The run uses fresh model contexts and is
intentionally separate from the normal evidence-depth policy.

The model-backed test skips without a model and is capped at three CPU
threads. It complements, rather than replaces, the model-free repair smoke
and the public session-host runtime smoke. When an overlay arm remains
`UNKNOWN` because it fails like the baseline, the smoke additionally reports
the geometry between that arm's hidden-state shift and the host-certified
behavior delta. The diagnostic fields are:

```text
cosine    direction of the arm shift relative to the behavior delta
progress  projection onto the behavior delta, normalized by delta length²
leakage   orthogonal residual relative to delta length
shift_norm length of the arm's hidden-state shift
```

These values are calculated per matching captured layer by
`flydelta-representation-diagnostics.*`. They are emitted only after the
counterfactual classifier has identified an `UNKNOWN` arm. A positive cosine
or progress is an `aligned` diagnostic signal, not a successful outcome: it
may justify extended tuning or a later experiment, but it cannot update
DeltaMemory or promote an artifact by itself. The diagnostics are deliberately
outside the counterfactual outcome and promotion contracts; they never create
`HELPED`, select an alpha, or serve as learning evidence.

#### DoseController — shared intervention safety seam

The search policies propose an intervention strength; they do not silently
change it at execution time. `DoseController` is the shared low-level safety
seam used by Whirlpool, the bounded scale primitive, AdaptiveAlphaSearch and
rank-2 coefficient/TFO-lite arms.
It is deliberately not an orchestrator: it does not choose a layer, score a
behavior, decide `HELPED`, change evidence depth or grant learning credit.

Its roles are separate:

```text
search policy proposes requested strength
  -> DoseController observes geometry
       absolute dose: shift_norm       (hard safety envelope)
       relative dose: sqrt(progress² + leakage²)
  -> accept, explicit retry_lower, safety_boundary or reject
  -> search policy records the observation and chooses its next proposal
```

`shift_norm` is the absolute safety bound. `progress` and `leakage` are the
orthogonal components relative to the same behavior delta, so their Euclidean
combination is the relative intervention dose when that geometry is
comparable. `separation_calibrated` remains a prior/reference scale; the
DoseController is the posterior empirical safety response from an actual
model observation. They are complementary and must not be conflated.

Every dose-aware arm records both `requested_scale` and `executed_scale`.
When a request is outside the envelope, an explicit bounded retry may be
issued at `proposed_safe_strength`; there is no hidden clamp. The trace also
records `dose_action`, `relative_dose`, `dose_safety_limited` and a bounded
reason. A retry remains the same search arm/coordinate, while its executed
strength is the safe observation that was actually run.

The controller state is local to a compatible search surface. Reuse therefore
requires explicit compatibility of model fingerprint, tokenizer/template and
capture semantics, context identity, surface revision, basis/direction,
layer/profile and intervention semantics. It is not a universal alpha cache.
The controller regulates how strongly a proposed intervention may be
exercised; the search policy still decides whether the resulting response is
useful.

### 4I. Coarse-to-fine layer search — implemented

When an overlay arm remains `UNKNOWN`, the host may use its per-layer
representation diagnostics to decide where a bounded follow-up experiment is
worth trying. This is a search over the existing cvec/activation seam; it does
not require a change to llama.cpp's model graph or a second inference backend.

The planner applies the following sequence:

```text
per-layer diagnostics
    -> cosine gate and progress/(1 + leakage) ranking
    -> separated numeric local maxima
    -> singleton layer trials
    -> adjacent pair trials only if no singleton is host-verified HELPED
```

The adjacency is numeric (`L-1`/`L` or `L`/`L+1`), not adjacency in a sparse
capture list. A missing captured layer is never invented. At most two separated
regions are selected by default, with bounded singleton and neighborhood
counts. Directions must be compatible with the captured/basis layer set.

Layer candidates share one total intervention budget. A singleton receives the
full budget; a two-layer candidate receives `total_scale / sqrt(2)` per layer.
This makes singleton and pair results comparable instead of giving pairs an
automatic advantage from twice the injected energy. The implementation is
`flydelta-layer-search.*`, and `common_flydelta_run_layer_search()` owns the
baseline-once, singleton-first ordering while the caller owns fresh contexts,
overlay composition and host verification.

Only a host-certified `HELPED` trial may be selected. `UNKNOWN`, `NEUTRAL` and
`HARMED` remain diagnostics or retention evidence and cannot promote a layer
mask, update `DeltaMemory` or create a training example. If the baseline already
passes, neighborhood expansion is skipped. The model-backed repair smoke
captures a bounded dense layer profile and exercises this path after its existing
three-arm alpha search; it reports the selected mask and trial count without
claiming that the model learned a repair.

The model smoke uses `layer-input` captures. Consequently, a candidate injected
at layer `L` must be diagnosed at a later captured layer; measuring the same
layer would only observe the state before its own injection and can produce a
misleading zero shift. The current L2 scale experiment therefore injects at L2
and measures the propagated effect at L3.

After layer search has identified a narrow candidate, the host can run a
separate trust-region scale search. The current model smoke uses L2 and the
geometric sequence `0.02, 0.04, 0.08, 0.16, 0.32, 0.64`, bounded by cosine,
leakage and shift-norm checks. If an arm becomes host-verified `HELPED`, one midpoint is
tested between that arm and the last smaller attempted scale; this is a small
bracket refinement intended to find a sufficient minimum without turning the
smoke into an unbounded strength search. Geometry can stop escalation, but it
cannot create `HELPED`.

The model-backed region mode also exercises the current orchestration
boundary after Whirlpool has finished. It selects the safest/promising region
continuation, resolves the matching behavior delta, assesses evidence depth,
builds the Bootstrap/Shallow/Deep plan and evaluates the host-neutral
UtilityGate. A single natural repair pair therefore reports `Bootstrap` and
may be retained when its margin/geometry is useful, but cannot enter Shallow,
Deep or TFO-lite. The smoke does not manufacture a second WHAT direction just
to force a rank-2 result; the two-sample dataset-question smoke is the
model-facing path for that continuation.

The observed order is:

```text
Whirlpool region result
  -> continuation selection (UNKNOWN may continue)
  -> compatible behavior evidence
  -> evidence-depth plan
  -> UtilityGate (margin + geometry)
  -> Bootstrap, or later Shallow/Deep when evidence permits
```

A positive margin movement can make an UNKNOWN continuation worth retaining.
For Bootstrap it may additionally authorize the bounded `refine_bootstrap`
branch, but it cannot authorize Shallow/Deep or create `HELPED`. Deep/TFO
evaluation still requires the separately aggregated compatible sample set and
its rank/depth gate.

`refine_bootstrap` is a rank-1 local optimizer, not a hidden Shallow path. It
first probes the selected layer at `0.5x`, `1.0x` and `1.5x` of the selected
scale, keeping only a safe normalized margin improvement above a small epsilon.
It then probes local singleton/pair/triplet profiles and an optional
opposite-sign singleton control. A profile always uses that layer's compatible
direction:

\[
\Delta h_L = \alpha w_L d_L, \qquad \sum_L w_L^2 = 1.
\]

Thus `L24+L25` has the same total intervention-energy budget as `L24`; it uses
`d_{24}` at layer 24 and `d_{25}` at layer 25, rather than transporting one
representation into another layer. The deterministic smoke stencil is capped
at eight extra model trials. It preserves fresh-context isolation, logs margin
and geometry for each arm, and never grants learning credit without a
host-verified baseline-fail / candidate-pass result.

### 4I.1. Layer×scale intervention-region scan — implemented

The earlier layer experiment used one very small scale while ranking layers.
That is useful as a cheap probe, but it must not exclude a layer whose signal
only becomes visible at a larger safe scale. The model smoke therefore has an
opt-in `--region-scan` path that treats layer and scale as a small interacting
search region:

```text
all dense-captured basis-compatible layers
        -> cheap separation/Fisher/slope profile
        -> signal-driven anchors (bounded to 4--6)
        ×
relative scales 0.02, 0.04, 0.08, 0.16
        -> bounded singleton trials
        -> retain active local regions
        -> adjacent pair trials around those regions
```

The host-scheduled dense discovery pass is capture-only: it computes per-layer separation,
projected variance, Fisher-like score and local slope from matched
model-facing captures. It selects a bounded set of signal-driven anchors.
Those anchors constrain the first singleton region arms; the complete dense
layer set remains available so adjacent numeric neighbors can still be
validated. If no anchors are supplied, the helper retains its legacy first-N
singleton behavior. This keeps the discovery result as a search hint rather
than learning evidence.

The region helper runs one no-overlay baseline, evaluates singleton layers
first, and uses the existing geometry safety checks (`cosine`, `progress`,
`leakage` and `shift_norm`) to decide whether a layer should continue up the
scale ladder. A layer with no useful signal can be stopped after the configured
number of stalled scales; unsafe geometry stops it immediately. If no
singleton is host-verified `HELPED`, only adjacent numeric layer pairs around
active singleton regions are considered. The helper has a hard trial budget;
the model smoke's configured maximum is 32 overlay trials plus the baseline,
although early stopping normally makes the run smaller.

When separation calibration is enabled in the scale-search configuration,
the scale ladder is interpreted as a relative fraction of a measured
positive/negative activation separation. The runner receives the resolved
bounded scale, while the trial keeps the requested relative scale and reports
whether the resolved value was clamped. Refinement brackets the requested
relative values, so calibration does not distort the search interval. The
existing geometry bounds can stop escalation, and a clamped arm is never
treated as safe to escalate.

The smoke captures through the layer after the candidate window so an
injection at layer `L` is measured downstream rather than at its own
pre-injection input. Pair candidates divide one total intervention budget by
`sqrt(2)` per layer, preserving comparability with singleton candidates.
The bounded region search is now the default strategy of
`common_flydelta_run_search_pipeline()`. The legacy layer-plan composition is
still available only when `use_intervention_region_search=false`; it is kept
as a migration/fallback seam, not as a second normal algorithm. The model
smoke's `--region-scan` flag now exercises this same default pipeline, rather
than a parallel region implementation. Neither path updates `DeltaMemory` or
promotes `UNKNOWN` or `NEUTRAL`. The typed pipeline runner transports the
optional teacher-forced decision margin alongside geometry, so a model adapter
can rank arms without changing the host-verification rule.

#### 4I.2. Adaptive Whirlpool WHERE search — implemented

Whirlpool is now the normal adaptive `WHERE` strategy in the composed pipeline.
The fixed region scan remains available as an explicit comparison mode and
compatibility fallback. Whirlpool is a cheap, derivative-free adaptive search
provider for `WHERE`; it is not another direction builder, learner or
promotion path.

Its first implementation is layer-only. It consumes the dense discovery
profile and/or retained layer anchors as an initial prior, evaluates a small
set of discrete probes around a centre, moves the centre toward the best
diagnostic result and shrinks the neighbourhood for the next round:

```text
dense all-layer capture
  -> separation/Fisher/geometry prior
  -> initial layer centre
  -> local layer probes
  -> margin + geometry objective
  -> recentre and shrink
  -> bounded host-verified arms
```

When a teacher-forced decision margin is available, its change is the primary
search signal. Geometry contributes a bounded secondary signal and leakage is
a penalty. Without margin, the provider falls back to quality and the existing
geometry signals. `cosine`, `progress`, `leakage` and `shift_norm` remain
diagnostics only. `UNKNOWN` and `NEUTRAL` may guide later probes, but only a
host-known baseline-fail / candidate-pass trial can be selected as `HELPED`.

Whirlpool does not mix activation spaces. Samples and directions remain
partitioned by layer/region, model profile, execution context and capture
layout. Its output is a proposed layer region; the existing pipeline then
continues to scale search, low-rank coefficient search and lifecycle recording
through the same runner. A future `layer x capture-position` variant must
extend the capture contract first; the current last-prompt-token capture does
not contain a position search dimension.

The evidence-depth budgets keep the search cheap:

```text
Bootstrap -> one small probe round, layer-only, fixed WHAT and scale
Shallow   -> one or two local rounds, then small rank-2/margin controls
Deep      -> bounded multi-round refinement, then coefficient/TFO search
```

The generic pipeline enables Whirlpool by default. Callers and tests may set
`use_whirlpool_search=false` when they need the fixed region strategy for an
explicit comparison or fallback. A worker can still tune its round/probe
budget from evidence depth while retaining the same fresh-context,
experimental-lifecycle and promotion boundaries.

Whirlpool evaluation is not judged by immediate `HELPED`. Its trace reports
`centre_before`, `probed_layers`, `best_probe_layer`, `centre_after`, the
radius history, `best_search_score` and `model_evaluations`. These are the
measurements for comparing it with the fixed region scan: same fixture,
direction, scale and arm budget, then compare the best observed search score
and region reached per model call. The separate region `selection` field keeps
its original meaning and is populated only by a host-known `HELPED` result.
This prevents a search accelerator from being incorrectly evaluated as a
promotion mechanism.

The model smoke prints the trace as machine-readable lines after a
`--region-scan` run:

```text
whirlpool_search=completed model_evaluations=... best_trial_index=...
whirlpool_round round=0 centre_before=... radius_before=...
    probed_layers=... best_probe_layer=... centre_after=... radius_after=...
```

The `region_trial` lines that follow contain the corresponding layer mask,
scale, host outcome and geometry (`cosine`, `progress`, `leakage` and
`shift_norm`). This makes the selected search region reusable by a later
shallow/deep smoke without repeating the discovery pass. Pass an explicit
comma-separated layer list with `--region-layers 22,24,25` (or
`LLAMA_AGENT_REGION_LAYERS=22,24,25`) to constrain the next region search to
those layers. The override is a search input only; it does not create
learning evidence or bypass host verification. The current smoke still uses
the same fixed WHAT direction and scale for both modes, so a later deep run
can compare Whirlpool with coefficient/TFO search from this saved layer
region. The smoke still performs its normal model-load and capture preparation
before entering the constrained region phase; skipping that preparation
requires a future persisted search-state input.

The deterministic contract test `test-agent-flydelta-whirlpool-ab` compares
the fixed region scan and Whirlpool on the same host-verifier landscape and
with the same maximum arm budget. It verifies the measurable search property:
a diagnostic prior can make Whirlpool reach the useful region earlier while
both strategies still select only a host-known `HELPED` trial. This is not a
claim that Whirlpool always wins; the real-model A/B comparison remains an
explicit smoke experiment.

In the latest local Qwen run all six L2 scale arms (`0.02, 0.04, 0.08, 0.16,
0.32, 0.64`) remained `UNKNOWN`. The scale phase took `38.2 s` for seven model
calls including its baseline; the complete smoke took `87.6 s` and 16 model
calls. Progress increased approximately linearly from `0.0032` to `0.1017`,
leakage from `0.0071` to `0.2281`, and the measured downstream shift norm from
`0.0102` to `0.3274`. All arms stayed inside the trust region, so the search
ended at the configured geometric-trial bound (the next doubled scale would
be `1.28`, above the absolute scale limit of `1.0`), not because of saturation
or an unsafe geometry. No midpoint or selection was produced. It still cannot
cross the model's tool-choice boundary without a host-verified `HELPED` result.

The corrected pre-discovery local Qwen region run then exercised the composed pipeline with
28 overlay trials plus a baseline (29 model calls, about 337 seconds with
three threads). All singleton scales on layers 1--4 remained `UNKNOWN` and
the model continued to emit `data.describe`; no candidate was selected. The
downstream measurements were no longer zero: singleton alignment stayed
roughly between `0.407` and `0.499`, while progress increased with scale. The
adjacent layer-pair arms reduced alignment and were not promising at the
configured geometry bounds. This is an evaluation result, not training
evidence: the run confirms that the region pipeline and downstream capture
work, but it does not yet demonstrate a behavioral flip.

### 4J. Composed direction/layer/scale search — implemented

`common_flydelta_run_search_pipeline()` is the normal composition point for
the existing searches. For each already-built direction candidate it now runs
the bounded layer×scale intervention-region scan: singleton layer candidates
are tried over the configured scale ladder, and adjacent neighborhoods are
expanded only when no singleton produces a host-verified `HELPED` result.
The older diagnostic layer planner remains an explicit fallback for callers
that set `use_intervention_region_search=false`.

```text
direction candidate
        -> layer diagnostics and local-maxima plan
        -> singleton layer arms
        -> scale arms per layer candidate
        -> optional adjacent layer pairs
        -> host-classified outcome
```

The pipeline runner is the only model-facing seam. It receives the direction,
layer mask and total scale budget and owns fresh contexts, overlay
composition, activation capture and host verification. A two-layer mask must
apply the existing `total_scale / sqrt(2)` per-layer budget rule. The runner
may interpret one direction candidate across a selected mask; the host owns
that mapping because the direction representation is deliberately independent
of llama.cpp graph details.

The result keeps the complete trace for either strategy. Region mode stores
each region arm and its optional decision margin directly; fallback mode keeps
the legacy nested layer/scale trace. Only a selected `HELPED` arm fills the
pipeline selection; geometry, margin, `UNKNOWN` and `NEUTRAL` remain available
for lifecycle refinement but cannot create evidence. This preserves the
decision order:

```text
geometry -> decision margin -> full generation -> host outcome
```

The margin is a ranking signal only. It may choose which arm receives full
generation, but it can never manufacture `HELPED`.

### 4K. Rank-2 WHAT and MIX search — low-level primitive

After a region scan has identified a useful layer and the orchestrator has
allowed Shallow/Deep work, the host may pass the
compatible WHAT candidates for that same layer to
`common_flydelta_run_deep_search()`. The helper ranks candidates by the
model-facing margin when available, otherwise by their existing alignment,
keeps a bounded rank (two by default), and builds the existing orthogonal
low-rank basis. It does not mix directions from unrelated layers.

This helper is deliberately not the normal phase-transition policy. It is a
bounded evaluator primitive called by a worker slice after the orchestrator
has established the applicable evidence capacity and UtilityGate result.
The normal order is:

```text
Shallow evidence basis
    -> Shallow controls
    -> UtilityGate
    -> Deep capacity + positive utility
    -> robust aggregation / Deep basis
    -> Deep controls
    -> UtilityGate
    -> coefficient search / TFO-lite
```

The low-level helper itself then evaluates:

```text
region arms
    -> margin/geometric ranking
    -> compatible directions on one layer
    -> rank-2 basis B_L
    -> coordinate or TFO-lite coefficients c
    -> fresh full-generation arms
    -> host verification
```

The runtime overlay contract is unchanged:

```text
delta_h[L] = B_L * c
```

`common_flydelta_run_low_rank_coefficient_search()` owns the bounded
coordinate/TFO-lite proposals. Its fitness may use decision margin, leakage
and intervention norm, but only the host-classified counterfactual outcome
can select a coefficient arm. All executed coefficient arms can be recorded
through the existing experimental lifecycle helper, including UNKNOWN and
NEUTRAL results with lineage. No arm is admitted to active state merely
because its margin or geometry improved.

The two-phase deep-search helper uses separate callbacks for these phases:

```text
diagnostic callback
    -> bounded coefficient arms
    -> margin/geometric ranking
    -> top-K selection

full-generation callback
    -> fresh baseline
    -> fresh generation of top-K arms
    -> host counterfactual classification
```

`common_flydelta_append_deep_search_lifecycle()` records both phases through
the same experimental lifecycle store. Full-generation records point back to
their diagnostic parent through `parent_trial_index` and use the mutation kind
`full_generation_top_arm`. `HELPED` is therefore possible only after the
second callback and host comparison; margin, cosine, progress and leakage
remain search diagnostics. The deep helper intentionally stays an explicit
continuation after region search because it requires multiple compatible WHAT
directions for one layer; it is not silently invoked when only one direction
exists.

### 4K. Source-neutral behavior transitions — implemented

The shared evidence contract and FlyDelta job envelope are source-neutral.
They can carry host-certified relations from tool repair, reflection
alternatives, planning revisions, research alternatives, dataset/resource
handling, workflow/code validation, procedure/blueprint learning and explicit
user corrections. `common_flydelta_behavior_transition_from_evidence()` is the
generic adapter; the longer-named tool-repair helper is retained only as a
strict convenience adapter for the existing failure/recovery signals.

This does not mean every source is automatically captured or activated. The
runtime collector only queues sources explicitly marked as candidate-ready.
Other sources must arrive through an explicit host-verified relation. The
source is copied into the capture manifest and remains part of the evidence,
job and artifact provenance. Callers should partition their basis and
behavior key by the learned behavior so unrelated corrections cannot be
clustered into one direction. Explicit host relations also carry the
`behavior_key` into capture-candidate identity; broad runtime discovery may
leave it empty until the host completes that relation. Thus two explicit
behaviors cannot collide merely because they share a source and transaction.

The model-facing roles remain unchanged: the model may produce alternatives
and representations, while the host verifier decides whether an outcome is
valid. Facts, provider failures and unverified model self-descriptions are not
FlyDelta evidence.

### 4L. Candidate search lifecycle — contract slice implemented

FlyDelta keeps three independent concepts separate:

```text
outcome       = what the host established
disposition   = what the bounded search should do next
champion      = best verified candidate in this experiment population
active        = sideband currently approved for runtime use
```

Every alpha, layer and coefficient trial is written as an experimental search
record. The record carries its bounded `search_kind`, optional associated
`experimental_artifact_id` and the literal `artifact_status=experimental`.
Several trials therefore share one immutable experimental `.flyd` revision;
the search does not create or activate a new sideband for every arm.

`HELPED` produces `validate_repeatability` and the separate
`review_candidate` artifact action; it does not activate or silently promote a
sideband. The host may explicitly move the referenced registry entry from
`experimental` to `candidate` after the repeatability/holdout review.
`HARMED` produces `reject`. `UNKNOWN` and `NEUTRAL` produce `refine` only
when bounded geometry or sequence-margin diagnostics provide a useful search
signal and budget remains. Otherwise they remain retained experimental
history, never positive learning evidence.

Candidate lineage records the parent, mutation kind, generation, direction,
layer mask, scale and intervention budget. The composed pipeline can append
every executed scale arm to the same lifecycle store with stable
`direction/layer/scale` idempotency suffixes. This makes alpha, layer and
direction refinements traceable without rewriting the original observation or
creating a parallel experiment journal.

The experiment champion is distinct from the active sideband. A challenger
must use the same model profile, fixture-set revision and verifier revision;
it must have host-verified lift, pass holdout/no-regression gates and strictly
beat the current experiment champion. The active sideband registry and its
explicit `candidate -> canary -> active` transition remain unchanged.

The current C++ contract covers the disposition, artifact action and champion
decisions. A `refine` disposition can now create the next bounded
counterfactual queue job
through the existing collection seam; the candidate lineage generation is
included in the job variant so successive refinements do not collide. The
decision and lineage can also be appended to the existing
`common_learning_lifecycle_store`. No model self-claim, diagnostic score or
experiment champion may bypass host verification or mutate active runtime
state.

### 4M. Runtime candidate journal bridge

The normal runtime path now has an optional journal bridge next to the capture
collector. When adaptation capture, capture-candidate collection and
`enable_flydelta_candidate_lifecycle` are enabled, the host assembly creates a
separate lifecycle store and routes the existing source observer through
`common_flydelta_runtime_candidate_observer`.

For a completed host-visible tool repair the bridge performs this sequence:

```text
learning transaction accepted
        ↓
source routing sees failure + successful recovery
        ↓
bounded capture candidate queued
        ↓
candidate lifecycle record: observed
```

The lifecycle record is reference-only. It contains candidate and transaction
identities, source/behavior metadata, model/capture fingerprints and evidence
references; it does not contain prompts, tool output, hidden states or
credentials. The candidate is still only `observed`: no outcome is inferred,
no `refine` job is scheduled, no training target is made and no sideband is
activated. A later host-owned capture/evaluation step must supply aligned
evidence and diagnostics before the candidate search lifecycle can choose
`refine` or `validate_repeatability`.

The bridge uses the same idempotent candidate identity as the bounded capture
queue. Repeated observer callbacks therefore do not duplicate either the
candidate or its journal record. If the lifecycle store is disabled, the
capture queue continues to work exactly as before. If the optional lifecycle
store cannot be opened, the assembly keeps the runtime usable and exposes the
failure through its existing adaptation error surface; ordinary user turns
are not failed by this auxiliary path.

The runtime assembly options are host-facing rather than model-facing:

```text
enable_adaptation_capture = true
enable_flydelta_capture_candidates = true
enable_flydelta_candidate_lifecycle = true
flydelta_lifecycle_backend = auto | in_memory | cozo | sqlite | jsonl
flydelta_lifecycle_path = <required for persistent backends>
```

`auto` follows the existing adaptation-store backend order: in-memory when no
path is supplied, otherwise Cozo, then SQLite when compiled. JSONL is explicit
and portable. The lifecycle journal is separate from the learning transaction
ledger; a JSONL lifecycle path must not be the same file as the transaction
JSONL path.

### Adaptive rank-one alpha response search

`flydelta-alpha-response-search` is the bounded HOW-MUCH primitive used for a
selected rank-one region after BootstrapZoom has earned further refinement.
It does not replace Whirlpool or create a new worker lane; the existing typed
Bootstrap state marks the next slice as `adaptive_alpha` and the orchestrator
continues to own the phase transition:

```text
Whirlpool / BootstrapZoom seed
        -> geometric alpha expansion
        -> bounded golden-section zoom inside observed safe bounds
        -> optional minimum-effective HELPED bracket
        -> AlphaResponseGate / next bounded action
```

The search prefers the fixture-bound decision-margin delta and subtracts a
leakage penalty. If the model-facing margin is unavailable it falls back to
the existing geometry utility; `cosine`, `progress`, `leakage` and
`shift_norm` remain safety/diagnostic signals. It never creates `HELPED`,
learning credit or a promoted artifact. A budget-limited search whose last
safe utility is still rising is marked `range_not_exhausted`; it remains a
bounded `refine_bootstrap` continuation and cannot be interpreted as plateau
evidence. A verified `HELPED` result performs the bounded minimum-effective
bracket before the ordinary lifecycle can consume it.

### 5. Optional dynamic hook

Only propose an upstream-quality llama.cpp hook if the evidence warrants it.
It needs backend behavior, tensor lifetime, threading, architecture coverage,
abort/error semantics and a zero-overhead disabled path.

## First evaluation

Start with one narrow, host-verifiable behavior: a redacted structured-tool or
code-repair fixture, not free-form factual knowledge or policy behavior.

```text
A  base model
B  existing memory/procedure assistance
C  static contrastive steering
D  recognition-gated FlyDelta sideband
```

Use fixed fixtures, baseline replay and held-out evaluation. Promotion requires
an improvement without material regression; a successful worker run or lower
internal loss is not enough.

## Open decisions

1. V0 context signal: host features, an embedding model, or two-pass
   model-derived features. The first two are cheaper and safer.
2. Initial domain: coding repair or structured tool behavior. Tool contracts
   remain host-enforced even if a sideband improves model choices.
3. Registry shape: a sideband-specific registry now, or a deliberately planned
   generic overlay registry later.
4. Exact redaction, scope, TTL and revocation semantics for capture manifests.

Until these are decided, documentation, model-free experiments and strict
non-activation are the correct next steps—not an automatic learning loop.
