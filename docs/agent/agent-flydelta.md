# FlyDelta sideband learning — design hypothesis

## Status and purpose

**Status: V0 host seam, bounded CPU capture, two-pass experiment, explicit CLI
activation and request-scoped server-context activation are implemented;
the host evaluator, split validation and explicit sideband lifecycle are now
implemented, while automatic runtime learning and activation are not.** FlyDelta is not enabled by default and is not a replacement for
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
```

Only a host-discovered `candidate_ready` source (currently a tool failure plus
a successful recovery with evidence references) is queued. The handoff
contains transaction/evidence IDs and fingerprints, not prompts, tool output
or hidden-state tensors. Reflection, research and user-correction matches
remain non-ready until a host supplies an explicit relation and verifier. The
collector is bounded, idempotent and best-effort; it cannot make the active
turn fail.

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

`docs/examples/agent-flydelta-dataset-question-suite.json` is a small,
host-owned English scenario suite for exercising model-facing dataset tool
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

The suite intentionally contains sixteen scenarios: the first six cover the
basic dataset inspection path, while the next six exercise data operations in
slightly different ways and the final four add distinct projections, grouped
counts and integrity checks. That gives the optional model smoke more natural
selection attempts without changing the model-facing seam or adding a second
prompt format. Each scenario includes a host-readable plan using the canonical
`dataset://local/sales` reference. The contract smoke validates the JSON,
catalog membership, plan shape and dataset bindings. The optional model smoke
uses the same file and the real compact tool descriptions, captures a bounded
layer-input window, and prints the selected tool and model output for every
question. It is observational by design: a mismatch is a repair observation,
not a HELPED result and not training evidence. A separate host execution and
verification step is required before a failed selection, repaired selection or
counterfactual can enter FlyDelta.

Run the always-safe contract check with:

```text
llama-agent-flydelta-dataset-question-contract-smoke \
  docs/examples/agent-flydelta-dataset-question-suite.json
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

`--strict` makes a non-matching model selection return failure; without it,
the smoke completes and reports mismatches so they can be inspected and fed
into the normal host-controlled repair path. The suite contains no Swedish
prompts and no synthetic “choose the wrong tool, then choose the right tool”
instruction. That keeps natural model mistakes separate from the real
host-certified corpus.

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

The seed is wrapped in a typed `common_flydelta_experiment_job`. V1 has three
explicit job kinds:

```text
        basis            behavior-delta references -> candidate steering basis
counterfactual   capture references + alpha grid -> host-evaluated trials
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
tensors in the queue. A job may be `basis`, `counterfactual` or
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
Qwen has learned a useful correction. Automatic observer-to-queue wiring,
real capture/delta materialization from a model repair, and an actual
baseline-fails/candidate-helps Qwen case remain follow-up work.

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

Each delta carries its model-profile fingerprint, capture-layout revision and
layer index. A basis builder is configured for one model/layout pair and never
clusters vectors from different layers. This is a compatibility guard, not a
substitute for the later runtime artifact validation.

The CLI capture hook and two-pass host helper now provide a bounded way to
produce the activation inputs for a manifest on supported graph
architectures. The basis component itself still does not run inference or
choose what is correct: the host must produce a valid capture manifest and
counterfactual report first. The objective remains minimum intervention,
repeatable verified lift and minimum collateral change.

### 4D. CPU direction search — implemented

`flydelta-direction-search.*` is the inexpensive `WHAT` search between
host-certified capture materialization and model-side counterfactual
evaluation. It accepts a bounded set of aligned behavior deltas for one
behavior, model profile, capture layout and layer. Every sample must carry
`HELPED` intervention credit and be learning-eligible; `UNKNOWN` or
uncertified material is rejected rather than silently becoming training data.

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

The host uses six natural host-certified repair pairs as the first
model-facing deep-search threshold. Below six, the builder may still be used
for cheap CPU diagnostics or an explicitly configured CPU study, but the
runtime must not spend model inference on aggregate direction evaluation. At
six pairs, the host may compare the aggregate candidates on a holdout and
then use the normal layer search and four-arm scale search. The threshold is a
cost/evidence gate, not a claim that six examples are sufficient for
promotion.

This is a candidate builder, not a verifier or learner. It does not run model
inference, inspect tool choice, create `HELPED`, update `DeltaMemory`, write a
sideband or activate an overlay. The next host step must evaluate the emitted
directions with the normal bounded scale/layer search. Only a host-verified
baseline-fail/candidate-pass result can produce intervention credit.

The raw candidate remains important as a control. A direction that looks good
under CPU alignment but does not improve the host-verified tool-choice result
must remain an experiment result, not learning evidence. The same contract can
later be used for `reflection_alternative`, `planning_revision` or
`research_alternative`; the behavior key and evidence source keep those
domains separated.

The Qwen model smoke is wired to this seam for its L2 scale experiment. Its
current fixture contains one natural host-certified repair pair, so the smoke
reports and evaluates only `raw_repair` and marks `deep_ready=no`; it does not
manufacture additional contrast samples. Once six natural pairs are
available, the same smoke seam can evaluate the aggregate candidates under the
normal four-arm runtime scale bound. The CPU contract test exercises the
multi-sample trimmed and whitened paths independently of model inference.

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
captures a small layer window and exercises this path after its existing
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

### 4J. Source-neutral behavior transitions — implemented

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

### 4K. Candidate search lifecycle — contract slice implemented

FlyDelta keeps three independent concepts separate:

```text
outcome       = what the host established
disposition   = what the bounded search should do next
champion      = best verified candidate in this experiment population
active        = sideband currently approved for runtime use
```

`HELPED` produces `validate_repeatability`; it does not activate a sideband.
`HARMED` produces `reject`. `UNKNOWN` and `NEUTRAL` produce `refine` only
when bounded geometry or sequence-margin diagnostics provide a useful search
signal and budget remains. Otherwise they are retained as unproven history,
never as positive learning evidence.

Candidate lineage records the parent, mutation kind, generation, direction,
layer mask, scale and intervention budget. This makes alpha, layer and
direction refinements traceable without rewriting the original observation.

The experiment champion is distinct from the active sideband. A challenger
must use the same model profile, fixture-set revision and verifier revision;
it must have host-verified lift, pass holdout/no-regression gates and strictly
beat the current experiment champion. The active sideband registry and its
explicit `candidate -> canary -> active` transition remain unchanged.

The current C++ contract covers the disposition and champion decisions. A
`refine` disposition can now create the next bounded counterfactual queue job
through the existing collection seam; the candidate lineage generation is
included in the job variant so successive refinements do not collide. The
decision and lineage can also be appended to the existing
`common_learning_lifecycle_store`. No model self-claim, diagnostic score or
experiment champion may bypass host verification or mutate active runtime
state.

### 4L. Runtime candidate journal bridge

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
