# FlyDelta sideband learning — design hypothesis

## Status and purpose

### Internal causal residual-patch seam

`layer_input_residual` patching is a bounded causal-experiment seam, not a
FlyDelta activation or lifecycle mechanism. A request supplies one prepared
F32 vector and one absolute prompt position. `replace` sets that layer-input
row to the vector; `add` adds it. The graph applies the change before the
remaining layer work and its downstream KV/logit computation.

The seam is request-scoped, default-off and serialized in V0: patched slots
are not co-batched, prompt/KV reuse is disabled, and an unmatchable
layer/position/width fails instead of broadcasting or falling back. Capture
and injection share the `layer-input:generation-boundary:v1` coordinate system: capture is
pre-patch, while the model consumes the patched state.

It creates no learning credit, artifact, promotion or activation authority.
Behavioral claims still require full generation and the host semantic oracle.

### Causal-patch smoke evidence

The Qwen instruct causal-patch smoke is an experiment-only model-behavior
check. Its default `schema-selection` fixture probes a natural
`dataset.inspect` -> `dataset.schema` repair case; the historical
`overview-selection` (`statistics.describe` -> `dataset.inspect`) case remains
selectable with `--fixture`. Its bounded default probes layer 21;
`--all-layers` expands the probe to 20--22 and `--full-matrix` expands the direction/alpha set. It captures a
baseline and a host-correction context at their respective final generation
positions, then patches the baseline position with the repair state or a
derived direction. Because the two prompts may have different token counts,
the repair vector is a transferred state at the baseline decision site, not a
claim that both captures are the same prompt coordinate.

The smoke runs teacher-forced scoring before any full generation. Its positive
and negative continuations are complete model-facing tool-call continuations
from the selected fixture; the tool name must be part of the continuation
because the scorer prefers it over the legacy choice fields.
Only margin-gated frontier arms reach full generation. The smoke uses a fixed
model-facing semantic predicate for this diagnostic and is not a replacement
for the production host Oracle/counterfactual evaluator. It does not run TFO
and it does not create learning credit or promotion evidence.

Teacher-forced fixtures are fail-closed at the shared scoring seam: empty or
identical effective positive/negative continuations are rejected before model
scoring. This protects both the resident server path and the CLI fallback from
silently turning a malformed contrast into a zero-margin diagnostic. The
always-runnable contract test is
`llama-agent-flydelta-teacher-forced-contrast-contract-ctest`.

The causal-patch smoke is also fail-closed at the fixture boundary. It first
requires the naturally failed tool and subsequent host-repaired tool declared
by the selected fixture. By default that is `dataset.inspect` followed by
`dataset.schema`; the historical overview fixture declares the reverse pair.
If either side is already correct,
missing, or otherwise differs, it reports `fixture_not_counterfactual`, returns
CTest's skip code, and does not execute the patch matrix. The repair request is
constructed from the rejected model-facing call and the canonical host repair,
matching the production repair-smoke shape rather than replacing the task with
an opposite instruction. The smoke additionally runs a scaled exact-delta
alpha=1 arm and reports the norm of
`base + (repair - base) - repair`; this must be within tolerance before the
alpha=1 path is considered equivalent to exact replacement. These checks are
experiment-fixture guards, not production evidence or Oracle outcomes.

An earlier run reported zero baseline/repair direction norms because capture
requests were allowed to reuse prompt/KV state. Capture now forces a fresh full
prompt, and the integrity smoke reports distinct finite states with non-zero
deltas at the tested layers. A second smoke bug compared identical positive
and negative continuations and therefore produced a false zero margin; this is
now corrected.

An earlier bounded Qwen run did produce finite, distinct captures
(`delta_norm=9.50928` at layer 21), and patched frontier requests reported
`applied=yes`. It is **not causal evidence**: its baseline had already selected
`dataset.inspect`, so the run only tested retention of a correct decision. The
fixture guard above now rejects that condition before the patch matrix begins.
No `HELPED`, learning credit or promotion evidence was created. A future run is
interpretable only when the declared baseline failure and host repair both pass
the counterfactual fixture gate.

The causal smoke uses the existing runtime-host/driver seam over the resident
server-context inference session. It does not construct a planner directly
through a smoke-local CLI entry point. The host-owned capture request is
propagated through the generic runtime request builder; the residual patch
itself remains an experiment-only decorator and is applied only to planner
generations, so an absolute layer/position from a planner capture cannot leak
into family selection, tool execution, reflection, or later runtime phases.

The Qwen Instruct validation on this path is currently fixture-limited rather
than seam-limited: schema selection, the dataset overview, OpenAlex argument
repair, and OpenAlex known-id all produced a valid first planner candidate, so
the smoke correctly reported `fixture_not_counterfactual` and did not run the
patch matrix. The OpenAlex candidate included the required search, per-page,
and select arguments on its first attempt. This is runtime-wiring evidence,
not a claim that a causal overlay helped.

Capture is also fail-closed for model architectures whose graph does not
publish the optional `t_layer_inp` seam. Such a request now returns an empty
capture with a controlled failure and the model smoke skips; it must not reach
an internal graph assertion. This is an execution-safety guard, not evidence
about the model's tool choice.

**Status: V0 host seam, bounded capture, explicit CLI activation and
request-scoped server-context activation are implemented. The daemon's
production model-host binding runs the bounded FlyDelta phase chain through
the common batch arm host: Whirlpool, Bootstrap/BootstrapZoom, AdaptiveAlpha,
orthogonal escape, Shallow/Deep controls, TFO-lite, representation
augmentation and the host-material concept capture/synthesis path. The
separate teacher-score batch, post-Bootstrap state resolvers, next-action
scheduler and lifecycle persistence are part of the same production seam.
Automatic learning credit, automatic promotion and automatic sideband
activation remain disabled; policy-driven first canary admission is a bounded
progress-only host operation through the existing review journal. Explicit
host review and lifecycle operations remain separate, durable gates. The
daemon installs the reuse-first observer from its shared candidate-index and
teaching-material host seams, and invokes admission only after the existing
worker has durably persisted a counterfactual report. Generic runtime assembly
still never guesses a challenger, artifact or deployment binding.**
FlyDelta is not enabled by default and is not a replacement for
the current model-adaptation path. It must not be activated until it has
passed explicit evaluation and promotion gates.

New sessions should follow the [FlyDelta session bootstrap](agent-flydelta-session-bootstrap.md).
This document is the authoritative architecture/status source for the current
repository; status claims below require production caller, persistence and
consumer evidence.

Development is performed on the currently selected work branch. Related
verified sweeps may be checkpointed with a local commit after the
documentation gate. Synchronization with an integration branch is a separate
owner-directed operation and is not implied by a local FlyDelta commit.

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

The first concrete blueprint-to-search adapter is
dataset-inspect-summarize. It compiles a host-resolved dataset task into a
bounded state graph, delegates path search to the existing A* proposer, and
returns a proof-carrying workflow proposal. The proposal contains the
blueprint/graph revisions, ordered state and transition refs, resolved
dataset binding and a deterministic path fingerprint. It can choose
select -> inspect -> operation when schema is unknown, or
select -> operation when the host already has a trusted schema. The
materialized steps are still only a proposal: Tool Contract, Workflow and
Procedure/Blueprint Oracles must validate them before any proposal can be used
as contrast material. The existing blueprint-selection seam now accepts an
optional host-owned materializer: when the host supplies the task state and a
successful Oracle gate, the verified proposal can be converted into the
ordinary `common_plan_state` before `plan_store.create`. Without that callback,
or when it returns `not_applicable`, the existing blueprint instance and
planner path are unchanged. No ordinary runtime turn invokes this adapter
implicitly, and it does not write lifecycle state.

Workflow and procedure direction: the host Workflow Oracle validates the
model's canonical implementation of a selected workflow; it does not replace
the procedure or blueprint that supplied the intended pattern. The built-in
bootstrap package now includes bounded dataset-inspection and OpenAPI-paged
retrieval blueprints, so the tool-aware library has a first deterministic
baseline. The existing bounded A* proposer and its Workflow/Blueprint proposal
adapter now provide the next support layer: they may propose a bounded
workflow/contrast path when a host supplies the transition graph and
canonicalizer. The adapter is not an implicit runtime route; the existing
host-owned contrast provider must register those callbacks for a concrete
blueprint case. The Workflow/Procedure Oracle must still validate the
resulting endpoints and semantic truth. A* never creates evidence or
lifecycle state by itself.

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
The standard daemon now persists the candidate as a session-scoped resource
before enqueueing. Broad source discovery remains pending until the host-owned
baseline, candidate and verifier references are present; the explicit verified
relation path supplies those fields and the candidate scope. This prevents a
syntactically valid queue item from reaching donor capture without a resolvable
fixture or from reading another session's resource.
The common worker then invokes a host callback for fresh capture/inference and
returns only redacted, identity-checked capture manifests. Queue pressure or
temporary worker unavailability remains pending host work rather than a turn
failure; the daemon dispatcher now owns an optional dedicated FlyDelta lane.
Its configured worker count is reserved from `limits.worker_count`, and status
reports distinguish a configured lane from a running lane. The daemon bootstrap
provides the queue/budget seam but does not invent an evaluator callback: a host
integration must attach the callback that resolves references and performs the
bounded work before jobs are consumed.

The runtime assembly exposes host-owned teaching providers. The narrow
`procedure_teaching_request_provider` and
`user_correction_teaching_request_provider` receive the completed turn and
durable learning transaction and may return a typed procedure/blueprint
or correction teaching request only when the host already has immutable
baseline, conditioned, verifier, contrast and provenance references. The
assembly runs the request through the existing teaching-relation admission
helper, maps it to
the shared `common_adaptation_evidence` contract and sends it through the
same FlyDelta runtime observer, lifecycle store and capture queue used by
other sources. A missing request, a `no_contrast` result or a missing provider
is a normal unresolved outcome; no references are synthesized and no learning
credit is created. The callback is invoked after the learning transaction is
durable, and its failure cannot fail the user turn.

For a production server-context runtime, the startup composition is explicit:

```text
start resident server context
  -> host binding factory supplies prepare/finalize/verifier callbacks
  -> register common_flydelta_model_host
  -> create common_flydelta_model_adapter
  -> start the reserved FlyDelta worker lane
  -> schedule bounded concept-capture jobs only when the orchestrator requests them
```

The daemon owns this composition seam, but it does not invent semantic
verification callbacks. A host integration must provide the binding factory;
without it, FlyDelta may be configured but remains intentionally idle. The
teaching-material runtime uses the existing Cozo adaptation database when the
adaptation backend is `auto` with a persistent path or explicitly `cozo`, and
stores its own relation alongside the transaction ledger. An empty path keeps
the in-memory behavior used by tests and ephemeral hosts. Concept capture is
therefore demand-driven, not an unconditional model operation at process
startup.

When FlyDelta is enabled, daemon startup also opens the configured FlyDelta
lifecycle backend, constructs the existing sideband review store and registry,
and replays the append-only review decisions before the resident model binding
and worker lane are started. The review store is not a second persistence
system: it uses the same lifecycle journal as the daemon's FlyDelta resource
provider. Replay restores only explicit host/operator decisions; it does not
turn `NEUTRAL` or `UNKNOWN` into learning credit and it never lets model output
activate a sideband. Daemon status exposes whether the review store was bound
and whether replay completed.

These are separate startup facts:

```text
review journal/registry replayed -> lifecycle decisions are restored
model-host binding registered    -> model-facing FlyDelta work is available
```

Review replay does not manufacture an evaluator callback, and a model binding
does not approve or activate a sideband.

The complete host configuration is illustrated by
`docs/examples/agent-host-config-flydelta-full.json`. It is JSON configuration,
not JSONL: JSONL is used by selected portable ledgers and daemon request
protocols, while FlyDelta queue entries are written as reference-only files
under the configured `runtime.adaptation.flydelta.queue_path`. The example
keeps the transaction Cozo database, FlyDelta lifecycle Cozo database, queue
directory, model catalog, and ordinary agent stores separate so they can be
inspected and recovered independently. It still requires the host integration
to provide the semantic server-context binding factory; no config file can
invent the prepare/finalize/verifier callbacks.

`user_taught_concept_relation_provider` is intentionally different: it may
return zero or more *already grounded* relations for an explicit user
principle. Each relation represents one host-verified minimal contrast and
must carry a contrast reference. The provider cannot manufacture pairs from
the statement alone. The existing concept builder later requires at least two
compatible trajectories, so a single accepted relation remains experimental
material rather than concept learning credit.

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

## Production daemon phase boundary

The production daemon is a bounded-slice scheduler around the common batch
model host. The production path is:

```text
daemon job
  -> prepare/finalize arm callbacks
  -> batch capture/overlay/generation
  -> separate batch teacher scoring
  -> Whirlpool
  -> Bootstrap rank-1
  -> BootstrapZoom state persistence/resume
  -> AdaptiveAlpha when rank-1 utility remains promising
  -> UtilityGate
  -> orthogonal escape or Shallow controls
  -> Deep basis and Deep controls when capacity and utility allow
  -> optional TFO-lite after positive Deep utility
  -> representation augmentation on the rank-one escape path
  -> concept capture/synthesis when grounded teaching material is ready
  -> grafted direction and a new bounded FlyDelta search slice
```

These phases are production-bound through the daemon's typed post-Bootstrap
state resolver, the same batch arm host, lifecycle persistence and the
host-scheduler `next_action` seam. The evaluator still executes exactly one
bounded phase slice; it never recursively runs the following phase.

### Experiment routes and orchestration facade

The experiment control plane now has an explicit, host-neutral route layer
under `common/agent/adaptation/flydelta/experiment/`. The existing
`common_flydelta_orchestrate_search_slice(...)` entry point remains the public
facade for current callers and delegates to the route selected by the current
experiment phase:

```text
common_flydelta_orchestrate_search_slice
  -> route dispatcher
     -> Bootstrap route
     -> Shallow-controls route
     -> Deep-controls route
        -> existing UtilityGate
        -> existing plan-transition helper
        -> existing typed next_action/result
```

The routes only assemble the completed-slice control flow. They validate the
bounded input, call the existing utility and transition contracts, translate
the result to the existing `next_action`, and return without recursively
starting another slice. They do not execute a model, construct prompts,
write lifecycle/review state, create promotion evidence, grant learning
credit, or change Oracle truth. Model execution remains in the resident
server-context host and lifecycle persistence remains at the existing host
seam. The first route extraction is intentionally behavior-preserving; phase
specific policy changes are outside its scope.

The host-neutral experiment implementations are being separated behind the
same orchestration facade in small ownership-preserving steps. Plateau
detection is owned by `experiment/flydelta-rank1-plateau.cpp`, UtilityGate by
`experiment/flydelta-utility-gate.cpp`, orthogonal direction construction plus
its bounded ridge solve by `experiment/flydelta-orthogonal-search.cpp`, and
BootstrapZoom validation, selection and proposal construction by
`experiment/flydelta-bootstrap-zoom.cpp`.
Their existing public types, callers, thresholds, budgets and result enums
remain in the facade contract during this migration. The facade therefore
continues to provide the compatibility entry points while implementation
ownership moves into the experiment domain; this is a source-ownership
boundary, not a second algorithm or runtime path.

Coefficient search follows the same ownership rule. Basis validation,
Gram-Schmidt construction, paired-intervention basis construction and strict
semantic basis resolution are owned by
`experiment/flydelta-coefficient-basis.cpp`; strategy validation and proposal
generation by `experiment/flydelta-coefficient-proposals.cpp`; the scalar
standard adapter by `experiment/flydelta-coefficient-standard.cpp`; staged
diagnostic/full execution by `experiment/flydelta-coefficient-staged.cpp`; and
the resident batch core by `experiment/flydelta-coefficient-batched.cpp`.
TFO-lite execution is isolated in `experiment/flydelta-coefficient-tfo.cpp`.
Those execution components share only the private
`experiment/flydelta-coefficient-execution-internal.h` contract. The public
coefficient-search facade remains responsible for the reference-only
lifecycle append in `flydelta-coefficient-search.cpp`. The public coefficient
contracts, candidate order, dose behavior, host-selection boundary and
existing TFO-lite semantics are unchanged; this is a source-ownership split,
not a second execution or lifecycle path.

```text
Shallow controls
Deep basis and Deep controls
TFO-lite
orthogonal search
representation augmentation
concept capture
concept synthesis
```

Concept capture and synthesis are conditional on a shared production teaching-
material runtime and ready relation/trajectory references. Augmentation is
conditional on resolvable donor/capture references. If those host-owned
dependencies are absent, the daemon reports the phase as unavailable or
configured-but-idle; it does not invent prompts, semantic outcomes or
material references.

The daemon workflow implementation is split into two private components:

```text
tools/agent/daemon/flydelta/concept-capture.cpp
    relation/group readiness
    baseline/conditioned/control arm preparation
    existing server-context batch execution
    host verification and trajectory persistence

tools/agent/daemon/flydelta/concept-synthesis.cpp
    persisted trajectory validation
    concept specification construction
    existing residual/prototype/negative candidate builders
    candidate return to the existing evaluator
```

The public binding and callback signatures remain unchanged. The split does
not create a second capture store, evaluator, queue, lifecycle writer or model
path. Physical model interaction remains in the resident server-context host;
resource authority and durable references remain in the adaptation layer;
algorithm implementations remain in `common/agent/adaptation/flydelta`.
Capture and synthesis therefore remain one production chain while their
responsibilities are independently readable and testable.

Concept synthesis currently exposes two explicit semantic sources through the
same experimental candidate contract. `control_residualized` preserves the
existing matched trajectory calculation

```text
(conditioned - baseline) - (control - baseline)
```

while `positive_prototype` compares independently verified conditioned
captures with independently verified neutral/control captures. Positive
prototype samples may have different task anchors and fixture references; they
must still share model, tokenizer/template, capture-layout, scope and layer
identity. Both sources produce the existing raw, trimmed and
diagonal-whitened candidate estimators, remain `experimental_only`, and use
the existing diagnostics, bounded frontier and host-Oracle path. This is a
representation extension only: neither source grants learning credit,
promotion eligibility or activation authority.

An explicitly host-labelled undesired capture can additionally produce the
`negative_repulsion` semantic source. Its vector is `control - negative`, so
it describes a bounded direction away from a known bad behavior. The negative
capture must carry its own capture reference and host verification; the
baseline is never silently reinterpreted as a negative example. It reuses the
same raw/trimmed/diagonal-whitened estimators as the positive prototype, and
its provenance records the negative sample count separately from the neutral
control count. A negative candidate is a contrast/basis component only: the
ordinary rank-1 frontier never persists or runs it independently, even when
capacity remains. It remains experimental-only with no learning credit, Oracle
truth or promotion authority of its own. A future explicit low-rank/coefficient
path may evaluate the positive and negative components together without
forcing rank two when the basis is degenerate.

Negative material is now admitted from the ordinary production capture path
for a narrow deterministic dataset-operation whitelist. The flow is:

```text
model-facing call
  -> shared canonical semantic normalization
  -> Dataset Operation Oracle
  -> known violated + attributable violation kind
  -> retain the already captured baseline as negative support
  -> match verified prefer/conditioned and neutral control material
  -> existing paired rank-2 basis/coefficient search
  -> full generation and the same host Oracle
```

The first supported family is grouped aggregation. Missing required grouping,
wrong grouping field, missing/wrong measure and wrong tool are recorded with a
versioned violation kind and bounded dimensions. Generic malformed output,
unknown semantics, `UNKNOWN`, and unverified failures remain
`unattributable` and cannot create negative material. The negative capture
keeps its execution, fixture, scope, model/template, capture-layout and Oracle
provenance in the existing trajectory resource; no negative store is created.

Negative support is not a standalone answer or direction. It cannot enter the
rank-one frontier, receive `HELPED`, learning credit or promotion. It is only
usable beside an explicit verified prefer/repair or positive-prototype
component and a protected control. This permits iterative correction: one
iteration may reduce a wrong tool choice, while a later iteration captures a
narrower argument-structure violation such as missing `group_by`.

Synthesis candidates also carry a comparison descriptor that separates
semantic strategy from estimator. `contrast_repair` and
`conditioned_prototype` are the current activation-derived strategies;
`decision_output_margin` and `execution_boundary_prototype` are challenger
labels used only when their host material exists. The descriptor retains
concept/behavior identity, model/tokenizer/template/capture fingerprints,
scope and Oracle revision. It is an audit and compatibility contract, not a
second registry or evidence store. A direction builder cannot make a
candidate learning-eligible merely by assigning one of these labels.

The current concept-synthesis job returns the complete candidate set for
inspection and validation and persists a bounded frontier of at most two
directions: the first raw `control_residualized` candidate and the first raw
`positive_prototype` candidate. The first persisted reference is mirrored in
the legacy single `graft_direction_ref`; the additional references are carried
through the worker report as `graft_direction_refs`. Each reference receives
its own idempotent ordinary search-pipeline job, with the same localized
WHERE/state references and a candidate-specific queue identity. The model
smoke executes those frontier jobs serially so the comparison is between
strategies, not concurrent GPU workloads. This is a bounded model-backed
comparison seam: candidates remain experimental-only, and neither diagnostics
nor a successful search job creates learning credit, promotion evidence or
activation authority.

The production concept-capture seam now host-verifies the conditioned arm
before persisting `conditioned_host_verified`; relation admission alone is not
model evidence. The resident Qwen concept-synthesis smoke uses a versioned
`normalized_call` dataset-operation fixture with the host-grounded
`data.aggregate(dataset://local/sales, group_by=[region], sum(amount))`
decision and can run it as a `user_taught_concept` relation. Its baseline is deliberately wrong, while conditioned captures must
pass the same production dataset Oracle. This makes the positive-prototype
fixture counterfactual and model-facing without granting learning credit or
promotion. The smoke still does not provide holdout/transfer validation for
the synthesized direction. Its offline contract path also builds three
explicit negative-repulsion estimators from host-labelled undesired captures
and verifies that they are admitted only as experimental, non-standalone
contrast components. The Qwen path now exercises the same admission from an
observed missing `group_by` violation; this remains wiring and host-verification
evidence, not evidence that the paired intervention improves the model.

Oracle dispatch now has an explicit host-owned contract identity in addition to
the existing semantic decision fast path. An Oracle request may carry
`expected_contract_kind`, `expected_contract_ref`,
`expected_contract_revision` and an optional contract fingerprint. The common
registry selects exactly one evaluator by semantic kind and contract identity,
in deterministic, host-supported, model-supported order; ambiguous ownership
or an unowned contract fails closed. The default registry currently registers
the deterministic dataset-operation evaluator. Host assemblies may register
the host-supported `model_facing_tool_contract` and
`openapi_operation_contract` evaluators beside it. The former validates the
effective model-facing schema and host policy; the latter validates the
provider/operation identity on top of the same parser and argument validator.
Both accept JSONL, compact DSL and the bounded server-context
`{"name":...,"arguments":...}` observation. The OpenAPI evaluator is an
operation owner in the existing registry, does not create a second evaluator
aggregator and does not execute network calls in V0.

The production OpenAPI adapter is
`tools/agent/openapi/agent-openapi-flydelta-oracle.{h,cpp}`. It projects each
operation from the same filtered catalog used by the OpenAPI provider, applies
the same model-facing required-parameter projection, and registers one
host-owned evaluator per operation. It is a registry adapter only: HTTP
execution remains in the existing OpenAPI executor and Oracle dispatch
remains request-scoped.

Each result keeps evaluator reference/revision and bounded structured checks in
addition to the top-level verdict. Dataset violations also carry a stable
`violation_code` such as `dataset.missing_required_grouping`, while the older
violation kind and dimensions remain the coarse admission fields. The Oracle
suite persists contract identity and evaluator provenance for baseline and
candidate observations. The server-context FlyDelta host uses this registry
for `normalized_call` verification and carries the violation code through arm,
counterfactual, worker and trace JSON. These fields explain host semantics;
they do not create `HELPED`, learning credit, promotion or activation authority.

Capability reporting must follow the same boundary: a capability is available
only when its production callback and durable resolver are registered. The
presence of a common algorithm or model-free smoke callback is not sufficient.
Scalar fallback remains a backend policy; when the production batch host is
enabled, FlyDelta logical waves must use it.

### Canonical end-to-end component flow

The production FlyDelta flow is a chain of existing seams, not one monolithic
algorithm. Each component receives typed references and returns a bounded
result to the next seam. The canonical flow is:

```text
learning/runtime host
  -> host-verified relation and teaching-material group
  -> typed FlyDelta seed/job and bounded filesystem queue
  -> daemon evaluator slice
  -> scoped resource binding
  -> resident server-context arm batch
  -> diagnostic capture/teacher scoring
  -> WHERE search and bounded refinement
  -> WHAT synthesis or existing direction admission
  -> basis/coefficient/alpha search
  -> bounded full-generation frontier
  -> semantic Oracle per arm
  -> paired baseline/candidate counterfactual
  -> selected-candidate descriptor and lifecycle report
  -> typed next_action
  -> review/promotion gates
  -> optional activation
```

The component responsibilities and hand-offs are:

| Component/seam | Owns | Does not own | Output to next seam |
| --- | --- | --- | --- |
| Learning observer and host collector | Source discovery, transaction/scope references and collection policy | Semantic correctness, hidden-state capture or activation | Host-owned baseline/candidate/verifier references |
| Teaching-material runtime | Relation admission, relation grouping, provenance and readiness | Model inference, direction construction or promotion | Relation-ready group with compatible material refs |
| Experiment queue and evaluator | Typed job validation, one bounded worker slice, resume references and result routing | Prompt construction, raw model data or lifecycle approval | Worker report and typed `next_action` |
| Daemon FlyDelta workflow | Phase orchestration, callback composition, Whirlpool/Bootstrap/augmentation/concept workflows and continuation | Vector mathematics, model internals or independent Oracle truth | Arm batches, search state and candidate references |
| Scoped resource adapter | Authority-checked resource resolution, JSON/material loading and durable artifact refs | Search policy, model generation or promotion decisions | Validated prompt/material/capture references |
| Resident server-context host | Request construction, fresh inference, batch execution, capture, generation finalization and execution telemetry | Search ranking, semantic truth, lifecycle persistence or model paths in callers | Per-arm execution, capture, generation and host observations |
| Common FlyDelta search algorithms | Whirlpool/WHERE, Bootstrap, alpha/dose, coefficient/TFO, Deep and low-rank math | Prompt semantics, host authority or promotion | Diagnostic trials and bounded frontier proposals |
| Synthesis and semantic basis admission | Candidate strategy/estimator descriptors and strict identity-compatible basis admission | New evidence, learning credit or activation authority | Experimental candidates or stable verified basis inputs |
| Host semantic Oracle | Parse/canonicalize model-facing output and decide whether expected behavior is satisfied | Hidden-state geometry, search ranking or promotion | Per-arm `verifier_known`/pass result with Oracle revision |
| Counterfactual classifier | Compare executed baseline and candidate against the same Oracle fixture | Inventing a semantic result when either side is unknown | `HELPED`, `HARMED`, `NEUTRAL` or `UNKNOWN` |
| Lifecycle/review/promotion | Durable reports, evidence aggregation, review and promotion gates | Correcting model output, searching new directions or silently activating | Durable decision/review state and, only after policy, activation authority |

The hand-off rules are deliberately strict:

1. A queue job contains references and identity, not prompts, raw tool output or
   activation tensors.
2. The daemon may choose the next bounded phase, but it cannot turn geometry or
   a teacher-forced margin into semantic evidence.
3. Diagnostic arms may capture and score, but only the bounded full-generation
   frontier reaches the host Oracle and counterfactual classifier.
4. Synthesis may propose `contrast_repair`, `conditioned_prototype`,
   `negative_repulsion`, `decision_output_margin` or
   `execution_boundary_prototype`. The descriptor records identity and
   compatibility; it does not grant trust. Negative repulsion is support
   material for a paired contrast, never a standalone answer.
5. The semantic basis resolver admits only identity-matching,
   host-verified, non-experimental directions. It is an admission filter in
   front of the existing low-rank/coefficient/TFO path, not a second store or
   evaluator.
6. The same Oracle fixture must be used for baseline and candidate comparison.
   A single arm's semantic pass is represented locally as `NEUTRAL`; the
   paired classifier supplies the actual counterfactual outcome.

#### Individual arm status, semantic progress, and paired counterfactual outcome

The host finalizer first evaluates each generated arm independently. A strict
individual pass is carried separately as `verifier_passed`; the legacy host
`NEUTRAL` marker is retained only for consumers that use it as an individual
pass marker. An individual failure is never written as `HARMED`. The paired
counterfactual classifier then compares the baseline and candidate:

```text
baseline fails  + candidate passes -> HELPED
baseline passes + candidate fails  -> HARMED
baseline passes + candidate passes -> NEUTRAL
either side not executed/unknown   -> UNKNOWN
```

`HARMED` means a verified paired regression, not merely that a candidate is
incomplete. When both baseline and candidate fail strict verification, the
counterfactual outcome remains `UNKNOWN`.

The host also records a separate bounded semantic-progress observation. For
dataset operations it compares operation, dataset, grouping, measure and
contract shape. A candidate may therefore report:

```text
strict_outcome: UNKNOWN
progress:       IMPROVED
residual:        grouping, contract_shape
```

when it moved toward the right operation and dataset but still omitted a
required `group_by`. This progress is search state only: it does not create
learning credit, promotion evidence or activation authority. `SOLVED` is
emitted only when the strict host Oracle passes. Improved safe candidates may
be retained in the existing bounded Bootstrap/search state for a later
residual iteration.

The smoke summary records the strict Oracle outcome, semantic progress,
provenance and the bounded
`expected_decision`, `observed_decision` and `verifier_reason` fields, but not
the full generated text. A detailed investigation of the exact raw model
response must therefore inspect the resident content trace when enabled; the
compact summary is the authoritative explanation of the verifier result, not
a verbatim transcript of the model output.

### Execution optimization boundary

The production phase chain is wired end to end and now uses a uniform
diagnostics-first execution boundary. This is an execution/wiring optimization;
it does not change the search algorithms or their promotion semantics.

| Phase | Current execution class | Status | Remaining verification or work |
| --- | --- | --- | --- |
| Deep | Diagnostic arms use `full_execution=false` with prompt evaluation, capture and teacher-forced scoring; only selected top-K arms use fresh full generation and host verification | IMPLEMENTED | Traced runtime evidence covers the diagnostic/full-generation callback split |
| Whirlpool/region search | All probes use prompt evaluation, capture and teacher-forced scoring first; only the bounded top-K diagnostic frontier is re-run with generation and host verification | IMPLEMENTED | The daemon batch host records diagnostic probes separately from the full-generation frontier |
| Shallow/TFO-lite | Coordinate and TFO-lite populations use diagnostics first; only the bounded top-K frontier uses fresh full generation and host verification | IMPLEMENTED | The staged coefficient seam preserves proposal order and host-only HELPED selection |
| BootstrapZoom | Candidate waves use one diagnostic batch; only baseline and a bounded top-K frontier are re-run with generation and host verification | IMPLEMENTED | Full HARMED results clear continuation eligibility; frontier trace evidence remains required |
| Orthogonal controls | Control waves use one diagnostic batch; only baseline and a bounded top-K frontier are re-run with generation and host verification | IMPLEMENTED | The existing control construction and ranking remain unchanged |
| AdaptiveAlpha | Sequential adaptive probes run a diagnostic capture/margin gate first; only geometrically safe probes repeat with generation and host verification | IMPLEMENTED | The adaptive order, dose policy and outcome-dependent decisions remain unchanged |

The execution class is explicit for staged phases. Diagnostic results remain
search and safety evidence; only the full-generation frontier can produce
host-verified counterfactual outcomes for selection, lifecycle or promotion
decisions. AdaptiveAlpha remains deliberately sequential: diagnostics can stop
an unsafe probe before generation, while safe probes retain the existing full
counterfactual decision path.

BootstrapZoom and Orthogonal currently bound the full-generation frontier to
two candidate arms per wave, in addition to a fresh full baseline. The bound
is an execution budget; it does not change candidate construction, ranking
order or Whirlpool continuation semantics.

## Project reuse and automatic first canary

Session candidates remain immutable and keep their full
`namespace/project/session` identity. The candidate index is a projection over
the existing lifecycle journal. `find_project_reusable_family()` is a
read-only lookup limited to the same namespace and project and requires exact
behavior, applicability, model/tokenizer/template, capture-layout and Oracle
identities. It returns opaque candidate/relation/evidence references and
identity metadata only; prompts, captures and tool output are never exposed.

When a new host-resolved TeachingRelation arrives, compatible family members
are tried first as bounded experimental challengers. A previous WHAT direction
may be reused strongly; its WHERE and HOW MUCH are only starting priors for a
bounded local diagnostics-first probe. Failed reuse falls through to the
existing capture/synthesis/search path. Reuse is never implicit runtime
activation.

The procedure, correction and explicit user-teaching adapters use the same
reference-only index boundary through `observe_resolved_relation()`; they do
not fabricate or persist a second hypothesis object merely to make a resolved
relation reusable. Research/reflection still enter through the semantic
hypothesis/grounding path and remain subject to the same host gates.

An automatically generalized revision is a new immutable sideband manifest.
Its provenance declares `generalization.level=model`, source candidate refs,
supporting session IDs, task fingerprints and synthesis strategy. The session
parents are not mutated. Automatic admission requires a host-approved
resolved relation with contrast/control material, a compatible configured
profile binding and compatible `IMPROVED`/`SOLVED` progress without regressed
dimensions. Research or reflection may discover a hypothesis but cannot alone
generalize or stage it.

The default FlyDelta canary mode is `policy` when canary routing is enabled.
The first automatic envelope is progress-only and bounded to 500 basis points,
6 evaluated observations and 8 hours. The host writes the existing
`approve_canary` and `stage_canary` review events; it never changes an active
binding. Missing profile bindings retain the family for later reuse. A
progress-only canary cannot become `active`; active promotion still uses the
separate promotion-backed gate and explicit lifecycle path. Compatible
HARMED/disconfirmation closes or blocks only the affected candidate family and
surface, not the whole concept or unrelated revisions.

The resulting flow is:

```text
resolved TeachingRelation
  -> project family lookup
  -> bounded reusable WHAT challenger
  -> local WHERE/HOW MUCH diagnostics
  -> existing Oracle/counterfactual
  -> generalized immutable model revision
  -> approve_canary + stage_canary (progress-only)
  -> retain/expand/close; active promotion remains separate
```

## Current pre-canary status map

The current branch has the following verified chain. `IMPLEMENTED` means the
link is registered, invoked, persisted and consumed; it does not mean that
promotion or activation happens automatically.

| Link | Status | Entry point and durable consumer | Evidence |
| --- | --- | --- | --- |
| candidate -> counterfactual reports | IMPLEMENTED | `daemon_flydelta_run_counterfactual()` and the evaluator's counterfactual job path; worker reports are persisted as lifecycle records | CTest contract: `test-agent-flydelta-pipeline`; functional evidence requires traced model-free or model-backed smoke, depending on the path |
| reports -> PromotionSummary | IMPLEMENTED | `common_flydelta_promotion_summary_from_reports()`; daemon worker persists and admin lookup reloads or derives it | `test-agent-flydelta-promotion`, `test-agent-flydelta-pipeline` |
| EvaluationRunner -> EvaluationReport | IMPLEMENTED | `daemon_flydelta_run_evaluation()` registered through `run_evaluation`; each fixture reuses the existing bounded batch/counterfactual path | CTest contract: `test-agent-flydelta-evaluator`; functional model-host evidence requires traced model smoke |
| EvaluationReport persistence/load | IMPLEMENTED | evaluation lifecycle record plus `common_flydelta_load_evaluation_report()` | evaluator and candidate-lifecycle contract tests |
| EvaluationReport -> OracleSuiteReport | IMPLEMENTED | `daemon_flydelta_run_evaluation()` writes one bounded immutable `flydelta_oracle_suite_report` resource through the existing resource store; the lifecycle report carries only `oracle_suite_report_ref` plus Oracle/policy revisions | `test-agent-flydelta-oracles`, `test-agent-flydelta-contracts`, focused 53-test FlyDelta CTest sweep and traced TinyLlama Cozo/Vulkan model-smoke readback; this is artifact/lifecycle wiring evidence, not Qwen adaptation-quality evidence |
| PromotionSummary persistence/load | IMPLEMENTED | promotion-summary lifecycle record plus `common_flydelta_load_promotion_summary()` | promotion and lifecycle tests |
| review JSONL/admin operation | IMPLEMENTED | JSONL protocol parses `flydelta.evaluate_candidate`, `get_evaluation`, `get_promotion_summary` and `review_candidate`; review data uses the configured lifecycle backend | These tests evaluate production wiring: CTest covers daemon protocol/review-store; model-free smoke covers wiring; the tiny model-backed admin smoke covers resident-model execution wiring, host-verification wiring and the full admin chain; the larger Qwen 1.5B profile has separate server-context runtime and cvec batch wiring evidence but no candidate differential token |
| review -> stage_canary | IMPLEMENTED | `flydelta.stage_canary` checks durable approval and sends the complete stage review through `apply_and_append`; registry state advances only after the review journal append succeeds; replay orders review actions by lifecycle dependency rather than backend row order | These tests evaluate production lifecycle wiring: model-free and tiny model-backed admin smokes trace dispatch, worker persistence, review, canary and durable replay; the larger Qwen 1.5B admin smoke stops before staging when its production-gated probe has no candidate-only verifier token |

The generic `common_learning_evaluate_candidate` path documented in
`agent-model-adaptation.md` belongs to the separate adapter lifecycle. It is
not the production FlyDelta pre-canary runner and must not be used as proof
for this table.

## Documentation maintenance gate

At the end of every related development sweep, compare the changed code with
this status map, the daemon usage guide and the referenced examples. Update
any changed entrypoint, caller, persistence record, consumer, test command or
limitation before closing the sweep. Record the verification date, commit and
tests in the maintenance log below. Classify each result as CTest contract
evidence, model-free functional smoke evidence or model-backed functional
smoke evidence, and retain the corresponding traces. If the code changed but
the status did not, record why the existing evidence remains valid. After
several related sweeps have passed this gate, create a local commit on the
current work branch; synchronization with another branch requires an explicit
owner instruction.

Completed worker traces are appended to the same lifecycle journal before a
follow-up `next_action` is scheduled. A completed queue item therefore cannot
advance the orchestration cursor while its trace/evaluation evidence is still
only in memory; the queue remains the scheduling record and the lifecycle
journal remains the durable evidence boundary.

### Maintenance log

| Date | Commit | Sweep/result | Tests or smokes | Documentation decision |
| --- | --- | --- | --- | --- |
| 2026-09-24 | 71f3e6274 | Initial pre-canary status map; stage-canary remains PARTIAL | CTest contract evidence: six focused CTests covering pipeline, promotion, review-store, evaluator, daemon protocol and daemon JSONL protocol; no functional smoke was run in this documentation-only sweep | Added bootstrap and documented the durable stage-canary gap; no implementation change made in this documentation sweep |
| 2026-09-24 | 9c8bb9a8a | Stage-canary atomic ordering and full model-free admin smoke implemented | CTest contract evidence: the same six focused CTests passed; model-free functional smoke passed with `flydelta_admin_trace` for queue, evaluation, summary, review and canary plus `flydelta_worker_trace` and durable replay | Reclassified review -> stage_canary as IMPLEMENTED; added the standalone smoke target and retained model-free/model-backed evidence separation |
| 2026-09-24 | 5b39241c6 | Catalog-only daemon configuration accepted; Vulkan model-host batch smoke and elevated tiny-model daemon startup passed | CTest contracts: server binding, model residency, daemon config and config discovery passed; model-backed functional smoke reported `execution_path=device_batch`; daemon startup reported `flydelta_model_adapter_configured=true` with the configured Vulkan device | Made `--model` optional when a named catalog profile is configured; added reusable Vulkan smoke configurations and kept tiny-model evidence explicitly wiring-only |
| 2026-09-24 | this local commit | Full model-backed pre-canary admin chain and deterministic review replay verified | CTest contract: `test-agent-flydelta-sideband-review-store` passed; model-backed Vulkan smoke passed with 8/8 known/helped trials, all four evaluation gates, admin/worker traces, canary staging and durable replay; persistent ccache configuration used | Added the model-backed functional smoke target and documented the model-host execution; fixed replay ordering so Cozo row ordering cannot apply stage before admission/promotion |
| 2026-09-24 | this local commit | Larger Vulkan profile validated for resident generation and host/cvec activation; daemon differential admin evidence remains open | CTest contracts: 5/5 focused tests passed; model-backed `llama-agent-flydelta-model-ab-smoke` passed with Qwen2.5-1.5B, Vulkan, host-verified capture, five neutral A/B arms; daemon admin smoke exercised twelve production-gated probes but found no candidate-only verifier token and correctly stopped before seeding | Expanded the daemon probe across representative layers and documented the larger-profile limitation; no promotion or quality claim made for the neutral A/B result |
| 2026-09-25 | this local commit | Evidence wording clarified: these smokes evaluate production wiring, not learned model quality | Documentation check: daemon usage guide and status map reviewed; existing 5/5 CTests and model-free/tiny/Qwen smoke results remain the recorded evidence | Marked runtime, host, worker, lifecycle, review, canary and replay claims explicitly as wiring claims; retained the larger-profile differential evidence gap |
| 2026-09-25 | this local commit | Later dataset-question model smokes aligned with the established portable wrapper interface | `bash -n` and `--help` passed for both wrappers; no model or CTest run was required for the argument-only change | Added explicit `--model` and `--suite` overrides with existing environment/default fallback; relative paths resolve from the repository root; removed local environment details from agent documentation |
| 2026-09-25 | this local commit | Clean-architecture correction for runtime capture handoff and worker evidence durability | Serial CTests passed for capture manifest, runtime observer, job, queue, worker, candidate lifecycle, server binding, daemon readiness and JSONL protocol; the runtime CTest remained explicitly skipped; model-free admin smoke passed with `flydelta_admin_trace` and `flydelta_worker_trace` through evaluation, promotion summary, review, canary and replay | Added typed capture scope/refs, durable candidate resource materialization, job-scoped donor authority, lifecycle trace persistence, next-action gating and a needs-based recurring architecture gate; direct scope/trace assertions were added in existing contract tests |
| 2026-09-25 | this local commit | Long-running build polling made a portable workflow rule | Documentation-only review; no additional runtime test required | Added a needs-based one-shot poll routine with a short progress check, dynamic completion estimate plus measured margin, stale-poll replacement, serial follow-up validation and no local environment paths in repository documentation |
| 2026-09-25 | this local commit | Evaluation and diagnostic-arm runtime wiring corrected without changing search policy | Serial FlyDelta contract CTests passed; model-free and model-backed functional smokes passed with tracing; model-backed evaluation persisted 8/8 known/helped trials and worker trace | Evaluation references now resolve through the immutable job scope; diagnostic arms use prompt-only server evaluation (`n_predict=0`) without sampling; smoke timeout and callback failure reporting remain portable |
| 2026-09-25 | this local commit | Qwen server-context runtime and cvec batch paths rechecked serially after device-selection correction | Qwen Instruct runtime smoke passed with three activation arms; native two-slot cvec batch and one-slot scalar fallback both passed with tracing; broader repair smoke passed separately; daemon Qwen probe loaded and generated but correctly stopped when no candidate-only verifier token was found | Recorded runtime/batch wiring as functional evidence only; kept the Qwen differential/model-quality limitation explicit and documented device selection as an invocation concern rather than a project-local setting |
| 2026-09-25 | d8a6fc42b | Deep diagnostics-first execution boundary corrected; broader optimization remains partial | Code review identifies separate diagnostic/full-generation Deep callbacks; a fresh focused runtime smoke proving `n_predict=0` diagnostics and top-K full generation is still outstanding; Whirlpool/region and Shallow/TFO remain full-execution paths | Recorded Deep as implemented, kept Whirlpool/region and Shallow/TFO explicitly open, and prevented the Deep result from closing the whole execution-optimization area |
| 2026-09-25 | this local commit | Diagnostics-first execution boundary completed across Deep, Whirlpool/region, Shallow/TFO-lite, BootstrapZoom, Orthogonal and AdaptiveAlpha without changing search policy | 13 focused FlyDelta CTests passed serially; model-free admin smoke passed with admin/worker traces; Qwen server-context runtime and repair smokes passed serially with tracing; repair trace covered native batch, scalar remainder and backend-batch paths | Added separate diagnostic/full-generation callback classes, bounded top-K full-generation continuation, AdaptiveAlpha diagnostic safety gating and host-only selection; evidence is recorded as production wiring/execution evidence, not learned model-quality evidence |
| 2026-09-25 | this local commit | Daemon FlyDelta host integration split across runtime execution, adaptation materialization and daemon workflows | 7 focused CTests passed serially; model-free admin smoke and Qwen server-context runtime smoke passed with tracing; Qwen repair smoke completed serially with tracing | Moved implementation ownership only; preserved common algorithms, public contracts, queue/lifecycle authority, diagnostic/full boundary and promotion semantics |
| 2026-09-26 | this local commit | Production concept-synthesis model smoke implemented and verified end-to-end; diagnostics-only graft rejection and continuation-state identity corrected | 17 focused FlyDelta CTests passed serially; model-free admin smoke passed with admin/worker tracing; Qwen Instruct server-context smoke passed with Whirlpool/Bootstrap, 2 relations × 3 capture arms, three synthesis candidates and grafted search | Added the portable model smoke and CTest entry; documented wiring-only evidence, four agent workers/two FlyDelta workers, serialized inference, no learning/promotion, safe `no_useful_utility` stop and fresh continuation refs |
| 2026-09-26 | this local commit | Concept accumulation and semantic teaching admission implemented on the existing durable seams | 7 focused CTests passed serially; updated model-free concept-capture smoke passed with reference-only candidate index evidence; model-free admin smoke passed with admin/worker tracing | Added a thin lifecycle-backed candidate index, separate one-relation admission from multi-relation synthesis eligibility, exact `(teaching_key, behavior_key)` family transport and scope/provenance checks; no parallel store, algorithm or promotion-policy change |
| 2026-09-26 | this local commit | Safe terminal handling corrected for production BootstrapZoom slices after the concept-synthesis E2E exposed diagnostics without a selected arm being reported as callback failure | 8 focused FlyDelta CTests passed serially; model-free concept-capture and admin smokes passed with tracing; Qwen Instruct server-context E2E passed with 2 relations × 3 capture arms, three synthesis candidates and grafted search ending in `no_useful_utility` | Preserved the existing terminal-search contract: diagnostic trials without a safe promising selection now stop successfully; no algorithm, ranking, learning-credit or promotion semantics changed. This is wiring evidence, not model-quality evidence |
| 2026-09-26 | this local sweep | Oracle V0 and shared bounded proposer implemented on the existing FlyDelta seams | `test-agent-flydelta-oracles` and `llama-agent-flydelta-oracle-ctest` passed serially; host smoke used the native dataset registry to normalize and execute a grouped aggregation; Qwen Instruct agent/server-context dataset suite completed 12/12 scenarios (9 matched, 3 repair observations); Qwen FlyDelta concept smoke passed with 2 relations × 3 capture arms, 3 candidates and safe grafted-search termination; no learning credit or promotion | Added generic deterministic/host-supported/model-supported evaluator contracts, versioned provenance, probe-suite metrics and a reusable bounded A* proposer. `normalized_call` server verification now uses the dataset semantic oracle. A* is a support component only; existing FlyDelta search algorithms and promotion policy are unchanged |
| 2026-09-26 | this local sweep | Concurrent worker/admin access to the in-memory lifecycle journal corrected; the timing-dependent model-free admin-smoke hang is closed | Rebuilt all FlyDelta/lifecycle test binaries with three compile workers; 53/53 focused lifecycle/FlyDelta tests passed serially; model-free admin smoke passed five consecutive serial runs with admin/worker tracing and durable replay | Added locking around the existing in-memory lifecycle records and an explicit four-writer concurrency contract test. No store, algorithm, search-policy, learning-credit or promotion-semantics change; this remains wiring/reliability evidence |
| 2026-09-26 | this local sweep | Research hypothesis routing and the missing proposed-to-host-grounding transition implemented on existing seams | Six focused FlyDelta/teaching CTests passed serially; model-free admin tracing, concept capture/synthesis/augmentation passed; Qwen Instruct server-context runtime and repair E2E passed with no promotion | Added strict structured research-envelope admission, default daemon batch-provider wiring, reusable host contrast-provider adapter and proposed-hypothesis grounding in runtime assembly. No new store, algorithm, Oracle policy, learning-credit or promotion change |
| 2026-09-26 | this local sweep | Model-facing tool verification aligned with the shared runtime normalization boundary | 15 focused FlyDelta CTests passed serially; model-free Oracle smoke and Oracle CTest passed; Qwen Instruct repair E2E passed with host-verified baseline/candidate calls, `repair_transition_outcome=helped`, Whirlpool centre discovery and augmentation controls | The repair and Oracle fixtures now apply runtime defaults, registry normalization and schema validation before executing normalized arguments. FlyDelta algorithms, Oracle verdict policy, learning credit and promotion semantics are unchanged; evidence remains wiring/host-verification evidence, not model-quality or promotion evidence |
| 2026-09-26 | this local sweep | Legacy inspection-tool aliases removed and model-facing arm scoring corrected to the catalog contract | 63 focused CTests passed serially; model-free Oracle, tool-repair, concept and learning smokes passed with tracing; Qwen Instruct repair E2E passed on Intel GPU 0 with three model threads: baseline `statistics.describe`, repaired `dataset.inspect`, four region trials, Whirlpool centre 22/radius 1, all FlyDelta arms `UNKNOWN`, no selected overlay, augmentation retained | Removed legacy inspection/description compatibility references from scoped tests, fixtures, configs, scripts and documentation. Teacher-forced scoring now uses `dataset.inspect`, `statistics.describe` and `dataset://local/sales`; production FlyDelta algorithms, Oracle policy, learning credit and promotion semantics were not changed. Evidence is wiring and host-verification evidence, not a model-quality or promotion claim |
| 2026-09-30 | this local sweep | Repair/contrast smoke corrected to use an observed small-model baseline failure and a semantically distinct host-verifiable repair; generation-boundary capture now tolerates different prompt lengths | 53 deterministic FlyDelta CTests and 12 relevant agent/adaptation CTests passed serially; Qwen repair/contrast smoke produced baseline `dataset.inspect` (host failure), repaired `statistics.describe` with `amount,units` (host pass), 26 behavior deltas and three fresh overlay arms; all overlay arms remained `UNKNOWN`, no overlay was selected or promoted | Kept the change smoke-local. The fixture now demonstrates a real `HELPED` repair transition without turning it into causal overlay evidence. The generation-boundary alignment fix uses the existing capture contract; FlyDelta algorithms, Oracle policy, learning credit and promotion semantics are unchanged |
| 2026-09-30 | this local sweep | Server-context cvec application and prompt-cache isolation made observable and fail-closed | Focused prepared-generation, sparse-cvec and server-binding CTests passed serially; Qwen Instruct Vulkan server-context repair smoke completed with six distinct cvec hashes, six non-empty 165888-byte payloads, `applied=yes`, `cache_prompt=no` and `n_cache_reuse=0` for every overlay arm; Whirlpool/region completed with six trials and no selected overlay | Closed the stale-cache ambiguity in the resident context by clearing/disabling the cvec-unsafe global prompt cache after the first cvec task and exposing request-scoped runtime telemetry. This is execution/wiring evidence only: all arms remained `UNKNOWN`, with no learning credit or promotion |
| 2026-09-30 | this local sweep | All model-backed FlyDelta smokes now use resident server-context execution | Six affected smoke targets rebuilt with three compile workers; Qwen Vulkan0 runtime, model A/B, concept, dataset-question incremental (4/4 requests completed), dataset-repair incremental and repair-model runs all used resident server-context tracing. The full 12-scenario dataset-question run reached inference with `n_ctx=4096` but hit a reproducible long-prompt smoke segfault under `n_predict=96`; it is not claimed as passed. | Removed smoke-only CLI execution from the FlyDelta validation surface; dataset-question hosts use 4096 context for the generated model-facing contract. The generic agent CLI backend remains available outside FlyDelta; model-free contract smokes remain model-free. The long-prompt dataset smoke remains an open bounded validation issue, not a reason to reintroduce CLI execution |
| 2026-09-30 | this local sweep | Causal FlyDelta smoke aligned with the existing runtime-host/driver seam | Runtime contract CTest passed; Cozo/Vulkan targets rebuilt with three compile workers; Qwen Instruct on Intel Vulkan used resident server-context tracing for schema, overview and OpenAlex fixtures, all of which correctly stopped at `fixture_not_counterfactual` because the first planner candidate was already valid | Propagated the existing host-owned capture request through runtime host/driver request construction and removed the smoke's direct planner invocation. Residual patching remains smoke-local and planner-scoped; no production algorithm, Oracle, learning-credit or promotion semantics changed |

| 2026-10-01 | this local sweep | Model-facing tool output now has a shared `native`/`jsonl`/`compact_dsl` selection seam; compact V1 accepts both command- and call-shaped scalar DSL | Model-free codec CTest passed; Cozo/Vulkan build linked the codec, runtime host and OpenAlex compact-DSL smoke. Model-backed rerun remains a separate validation step after this wiring change | Added profile/request format selection and canonical normalization to the existing tool-call contract. Unsupported nested/array/union schemas are filtered from compact-DSL views; no provider, planner, repair, FlyDelta or promotion policy was changed |
| 2026-10-01 | this local sweep | Qwen format smoke verified the shared codec on the resident server-context path | Qwen Instruct on Intel Vulkan0 with four inference threads produced a parseable JSONL `openalex.listWorks` call; the exact fixture still lacked optional `per_page=1`. Compact DSL was not accepted because the model emitted unsupported `contains(...)`/misnamed arguments, so no model-quality pass is claimed | This is model-output evidence, not a production policy change: the host now distinguishes codec parsing from exact fixture/host validation. The compact DSL V1 remains intentionally scalar-only and the smoke exposes `--tool-output-format {jsonl,compact_dsl}` for controlled comparison |
| 2026-10-01 | this local sweep | Compact DSL comparison now normalizes explicitly declared comma-separated fields | Codec CTest passed; Phi-4 Mini Instruct on Intel Vulkan0 with four inference threads produced `openalex.listWorks(search="machine learning", select="id, display_name", per_page=1)` and the OpenAlex smoke passed after normalizing `select` token whitespace | Added a shared comma-separated-string normalizer for schema-aware callers. Ordinary scalar strings such as `search` remain unchanged; `per_page` remains optional in the OpenAPI schema but is required by this one-result smoke fixture |
| 2026-10-01 | this local sweep | Positive-prototype synthesis added beside the existing control-residualized concept path | Cozo/Vulkan daemon target and focused concept/teaching CTests passed serially; model-free concept smoke produced three residualized plus three positive-prototype candidates per family; Qwen Instruct Vulkan0 resident server-context smoke passed with two relations, six capture arms, six candidates and safe `no_useful_utility` graft termination | Added explicit synthesis semantics, independent positive/control capture validation and CPU prototype estimators. Positive prototypes remain experimental-only; the scheduler now persists and runs a bounded two-candidate frontier (control-residualized raw plus positive-prototype raw) serially, with no learning credit or promotion created |
| 2026-10-01 | this local correction sweep | Full-generation attempt is now reported independently from verifier decisiveness | Focused FlyDelta CTests passed serially; the Qwen resident server-context run completed its bounded frontier with `host_evaluated=true` and `verifier_known=false` on the generic concept fixture | Propagated explicit host-evaluation provenance through counterfactual trials and lifecycle observations, and enabled non-stream resident content tracing. `UNKNOWN` remains fail-closed: no learning credit, selection or promotion |
| 2026-10-07 | this local sweep | Compact DSL is selectable for single-tool calls and multi-step planner output, with nested object/array arguments and canonical plan normalization | Focused codec/catalog/bridge tests passed. Phi-4 Mini produced a valid compact `openalex.listWorks` call accepted by the parser and smoke fixture. Qwen 2.5 1.5B emitted unsupported/malformed compact calls; full Qwen and Phi compact planner runs did not pass. A short Phi `dataset.list` planner run also failed; the same request in JSON produced a semantically invalid plan and incomplete regeneration, so planner reliability remains unproven across formats. GPU runs retained the `n99` fit/check path. | Public format names are `json` and `dsl`; `json` maps to the internal native mode and `dsl` is the default. The planner parses DSL into canonical plan JSON and uses the same host validation path. A configured model-profile format overrides the request format; an omitted field leaves the request/CLI choice intact. This is structural/parser-path evidence, not a model-quality or end-to-end execution pass. |
| 2026-10-01 | this local correction sweep | Positive-prototype capture now requires actual host verification and the model fixture is a dataset grouped-aggregation counterfactual | Six focused contract/smoke CTests passed serially; Qwen Instruct Vulkan0 resident server-context smoke completed with a deliberately failing baseline, host-valid conditioned captures, 2 relations x 3 capture arms, six synthesis candidates and two serial grafted frontiers; both frontiers stopped at `no_useful_utility`, with no learning credit or promotion | Replaced the generic instruction-following fixture with a versioned `normalized_call` dataset-operation fixture; production trajectory provenance now records the conditioned arm's real verifier result rather than relation admission. The remaining gap is holdout/transfer validation, not the positive-prototype admission seam |
| 2026-10-01 | this local causal correction sweep | Resident planner capture and causal patch diagnostics now complete the intended request-scoped server-context path | `test-agent-flydelta*` passed 53/53 serially; prepared-generation/runtime contract tests passed; Qwen Instruct on Intel Vulkan0 with four runtime threads produced a counterfactual baseline `dataset.inspect` and repair `dataset.schema`, finite distinct layer-21 captures (`base_norm=60.7102`, `repair_norm=61.3367`, `delta_norm=9.46182`), teacher margins for all six diagnostic arms and two bounded frontier generations | Propagated capture through the runtime chat driver, forced final-response handling for capture/patch requests so JSON-schema streaming cannot drop metadata, and made teacher scoring reuse the captured planner context. The run remained diagnostic-only: exact replacement reached margin `0.0151199` from baseline `0.0114106` but did not change the tool choice; no `HELPED`, learning credit, overlay selection or promotion was produced |
| 2026-10-01 | this local synthesis-portfolio correction sweep | The existing control-residualized and positive-prototype candidates now share one tested, bounded frontier selector | Cozo/Vulkan common/daemon targets built with three compile threads; focused concept test passed after correction; serial `test-agent-flydelta*` passed 53/53 | Moved the pre-existing semantic-source selection out of the evaluator-local helper into the common concept contract. It still prefers one raw candidate per semantic source, preserves source order and frontier bounds, and continues through the existing search/Oracle path; no synthesis estimator, low-rank builder, evidence, lifecycle or promotion semantics changed |
| 2026-10-02 | 2ed3616f0 | Synthesis strategy/estimator descriptors and strict semantic basis admission added around the existing direction, low-rank and coefficient seams | Cozo/Vulkan rebuild passed with three compile threads; serial `test-agent-flydelta*` passed 53/53; model-free concept smoke passed; full Qwen Instruct server-context concept-synthesis smoke passed with six candidates and two serial frontier searches | Decision-margin and boundary-prototype builders remain challenger-capable only when host material exists. Basis resolution requires matching concept/behavior/model/capture/scope/Oracle identity plus host-verified, non-experimental directions; no new store, evaluator, learning-credit or promotion path. The complete component flow and individual-arm versus paired-counterfactual outcome semantics are documented above |
| 2026-10-02 | this local decision-margin seam sweep | Host-owned decision-pair request/provider, output-head row resolver, challenger callback and provenance fields are now wired through the resident server-context binding and evaluator result | Cozo/Vulkan full rebuild completed with three compile threads; focused direction-search/server-binding tests passed; serial `test-agent-flydelta*` passed 53/53 | The daemon callback is optional and fail-closed. Its default provider is empty because generic output-head weight rows are not exposed by the public server-context API; logits are never substituted for `U[t]`. No challenger is persisted, grafted, activated, promoted or granted `HELPED` by this seam alone |
| 2026-10-02 | this local iterative-progress correction | Individual strict verifier pass is separated from paired outcome; bounded dataset semantic progress, residual dimensions and Bootstrap resume fields now travel through reports, trace and existing state | Cozo/Vulkan focused build completed with three compile threads and the user-profile ccache; affected FlyDelta CTests passed after rebuilding all affected binaries | A partial candidate is now `UNKNOWN` plus `IMPROVED`, not individual `HARMED`; strict Oracle, learning credit, promotion and activation authority remain unchanged |
| 2026-10-03 | this local negative-support and resident-batch correction sweep | Explicit negative captures remain contrast/basis support only; the ordinary rank-1 frontier never persists them independently, and resident parallel completion uses per-request byte cells rather than bit-packed `vector<bool>` state | Focused concept/evaluator/server-binding CTests passed serially; Qwen Instruct on Vulkan0 with four runtime workers completed both serial primary frontier searches through resident server-context | The earlier false incomplete-batch failure was a concurrent completion-bit race: both slots could complete successfully while one packed flag was lost. The correction preserves native per-sequence batching, request isolation and active evidence semantics; Qwen still produced an aggregate missing `group_by`, so both candidates remained `UNKNOWN` with no learning credit or promotion |
| 2026-10-02 | this local decision-margin seam validation | The production concept-synthesis path was exercised after the seam change with Qwen Instruct on the resident server-context/Vulkan0 path | Cozo/Vulkan build and serial FlyDelta tests passed; model smoke completed with 2 relations, 6 capture arms, 6 synthesis candidates and 2 serial frontier searches. The full-verification arms in the positive-prototype frontier were `HARMED`; no `HELPED`, learning credit or promotion was produced. The default daemon had no decision-pair/output-head provider, so no decision-margin challenger was created | This is runtime/wiring evidence, not evidence that positive-prototype is useful. The Oracle correctly rejects the harmful frontier, and the missing model-facing output-head provider remains an explicit backend integration boundary |
| 2026-10-02 | this local user-concept smoke correction | The same production smoke now labels its host-grounded concept relation as `user_taught_concept` | Qwen Instruct Vulkan0 server-context run completed with 2 relations, 6 capture arms, 6 candidates and 2 serial frontier searches; positive-prototype arms were `HARMED`, with no learning credit or promotion | This validates user-concept source wiring and fail-closed Oracle behavior, not a successful positive-prototype intervention. The lambda/code-structure example remains future work because it needs a separate host Oracle |
| 2026-10-02 | this local verifier-trace correction | Full FlyDelta arm traces now preserve compact expected-versus-observed verifier observations through the common arm, trial, region and worker seams | After a complete Cozo/Vulkan rebuild, all 53 focused FlyDelta CTests passed serially. Qwen Instruct Vulkan0 resident server-context smoke completed with 2 relations, 6 capture arms, 6 candidates and 2 frontier searches; both full-verification prototype arms reported `expected_decision` as grouped `sum(amount)` by `region`, `observed_decision` as `unparsed:missing_field`, and `verifier_reason` explaining that `group_by` was missing, yielding `HARMED` and no learning credit/promotion | The fields are bounded diagnostic trace data only. They explain `HELPED/HARMED/UNKNOWN` outcomes and do not change Oracle truth, evidence, learning, selection or promotion semantics |
| 2026-10-02 | this local canary-deployment/runtime-seam sweep | Bounded canary resolution now has a factory and an explicit resident-runtime seam on top of the existing registry and review journal | Cozo/Vulkan daemon/deployment targets rebuilt with three compile threads; 11 focused FlyDelta CTests and 10 model-free FlyDelta smokes passed serially. Qwen Instruct Vulkan0 server-context causal smoke reached planner/reflection/capture with four threads; OpenAlex fixtures were already valid on the first planner attempt and therefore stopped as `fixture_not_counterfactual` | Added active-only default resolution, explicit canary authority, deterministic cohort selection, replacement/additive ordered composition, baseline/candidate deployment fingerprints, per-turn activation refresh on a reused resident runtime, and `get_binding.open_canaries` admin projection. The Qwen result is server-context wiring evidence only; no canary, learning credit, promotion or active-binding change was performed |
| 2026-10-02 | this local canary-disposition/runtime wiring sweep | Host configuration, artifact loading, observation reservation, policy disposition and deployment-aware session/KV identity now use the existing registry, review journal and resident session host | Cozo/Vulkan daemon target rebuilt with three compile threads after one compile correction; model-free deployment and sideband targets are ready for the serial correction round. Real Qwen canary routing remains unclaimed unless a host supplies explicit canary authority, applicability and sparse-code inputs | Added `disabled/manual/policy`, `retain/expand_scope/promote_active/close`, preconfigured scope-step validation, atomic exposure reservation, same-key canary conflict rejection, journaled policy close/expand/promote and CAS-preserving runtime identity. No FlyDelta search, Oracle truth, learning credit or active binding is changed implicitly |
| 2026-10-02 | this local iterative-progress correction | Individual strict verifier pass is separated from paired outcome; bounded dataset semantic progress, residual dimensions and Bootstrap resume fields now travel through reports, trace and existing state | Cozo/Vulkan focused build completed with three compile threads and the user-profile ccache; affected FlyDelta CTests passed after rebuilding all affected binaries | A partial candidate is now `UNKNOWN` plus `IMPROVED`, not individual `HARMED`; strict Oracle, learning credit, promotion and activation authority remain unchanged |
| 2026-10-03 | this local paired-intervention and negative-capture sweep | Host-verified negative support now retains its `avoid_support` role and can be paired with a prefer/repair or positive-prototype component through the existing rank-2 basis/coefficient seam | Cozo/Vulkan full build passed with three compile threads; 54/54 focused `test-agent-flydelta*` CTests and 11 relevant contract/replay smokes passed serially; Qwen Instruct resident server-context smoke produced 2 relations, 6 capture arms, 2 host-verified negative trajectories, 9 synthesis candidates and two serial production search phases | The Qwen trace showed `missing_required_grouping`, `negative_trajectories=2` and `paired_intervention=yes`; the paired frontier remained `UNKNOWN`/`no_useful_utility`, so no learning credit, promotion or active-binding change occurred. Negative support cannot form an independent rank-1 frontier; duplicate same-layer components are accepted only for explicitly tagged paired artifacts |
| 2026-10-01 | this local evaluation-contract sweep | Durable evaluation fixtures can now name the Oracle-aligned classes `paraphrase`, `transfer`, `control` and `competing` in addition to the existing lifecycle gate classes | Contract round-trip checks passed; production daemon mapping preserves the existing intended/holdout/retention/agent-regression gates; no model-backed rerun was needed for this vocabulary-only change | The new names are optional fixture identity only; they do not alter lifecycle gates |
| 2026-10-01 | this local Oracle-report sweep | Evaluation now emits a separate immutable, redacted Oracle-suite report resource and links it from the lifecycle evaluation report | Cozo/Vulkan build with three compile threads passed; 53 focused FlyDelta CTests passed serially, including Oracle-report and evaluation-provenance round trips; the TinyLlama Cozo/Vulkan daemon smoke traced resource `agent-resource://.../resource-12` through lifecycle-report reference and post-evaluation readback | The report is an audit/diagnostic artifact only. It does not create learning credit, select an overlay or change promotion semantics; no second store or evaluator path was introduced. The smoke's synthetic candidate verifies wiring only, not useful Qwen adaptation. |
| 2026-10-03 | `196c87648`, `a7209fb9f`, `d058438c7`, `bf7854c15` | Experiment-route facade plus plateau, UtilityGate and orthogonal-search implementation ownership extracted without changing policy | Cozo/Vulkan builds used three compile threads; the seven affected FlyDelta CTests passed serially after each extraction; model-free contract/replay smokes passed; Qwen Instruct resident server-context concept-synthesis and repair smokes completed with tracing | Kept the public orchestration facade, callers, thresholds, budgets, transitions, lifecycle/evidence boundaries and runtime callbacks unchanged. BootstrapZoom remains in the facade for a later isolated extraction; the Qwen results are wiring/evidence diagnostics and produced no learning credit or promotion. |
| 2026-10-03 | `efc4ddf5f` | BootstrapZoom implementation ownership extracted behind the existing orchestration facade without changing candidate validation, trial selection, budgets or continuation semantics | Cozo/Vulkan build passed with three compile threads; the seven affected FlyDelta CTests and four model-free FlyDelta smokes passed serially (runtime CTest retained its existing skip); Qwen Instruct Vulkan0 server-context smoke passed with 2 relations, 9 candidates, paired intervention and safe `no_useful_utility` termination | Moved only BootstrapZoom helpers and implementations to `experiment/flydelta-bootstrap-zoom.cpp`; no public contract, search policy, lifecycle/evidence boundary or runtime callback changed. Model-backed evidence remains wiring/diagnostic evidence, with no learning credit or promotion. |
| 2026-10-03 | this local coefficient-basis sweep | Low-rank basis construction, paired prefer/avoid composition and strict semantic basis resolution extracted behind the existing coefficient-search facade without changing basis order, rank bounds or admission rules | Cozo/Vulkan coefficient target built with three compile threads; eight affected FlyDelta CTests passed serially, including coefficient search | Moved only basis ownership to `experiment/flydelta-coefficient-basis.cpp`; standard, batched, staged and TFO-lite execution remain in the existing facade for later isolated sweeps. |
| 2026-10-03 | this local coefficient-proposals sweep | Coefficient strategy naming, configuration validation and bounded coordinate/rank-two proposal generation extracted without changing proposal order, bounds or TFO-lite execution | Cozo/Vulkan coefficient target built with three compile threads; eight affected FlyDelta CTests passed serially, including coefficient search | Moved only proposal ownership to `experiment/flydelta-coefficient-proposals.cpp`; TFO-lite and standard/batched/staged execution remain unchanged in the existing facade. |
| 2026-10-03 | this local coefficient-standard sweep | The scalar standard coefficient-search façade now lives in its own experiment component and delegates to the existing batch core without changing runner order or result semantics | Cozo/Vulkan coefficient target built with three compile threads; eight affected FlyDelta CTests passed serially, including coefficient search | Moved only the standard adapter to `experiment/flydelta-coefficient-standard.cpp`; batch, staged, TFO-lite, dose and lifecycle behavior remain unchanged. |
| 2026-10-03 | this local coefficient-staged sweep | Diagnostics-first staged coefficient execution and bounded top-K full-generation handoff extracted behind the existing batch core without changing ranking, frontier size or host-selection semantics | Cozo/Vulkan coefficient target built with three compile threads; eight affected FlyDelta CTests passed serially, including staged execution | Moved only staged execution to `experiment/flydelta-coefficient-staged.cpp`; batch core, TFO-lite, dose and lifecycle behavior remain unchanged. |
| 2026-10-03 | this local evaluator-provenance sweep | Completed search slices now emit one typed selected-candidate descriptor from the validated pipeline selection; worker results and trace carry it through the existing lifecycle boundary | Evaluator, worker, search-pipeline and orchestration CTests passed serially after adding a selected-region contract case | The descriptor contains refs, strategy/estimator, layer, score and host/verifier status. Consumers no longer need to reconstruct the selected arm from trial indexes; no search, Oracle, evidence, lifecycle transition or promotion semantics changed. |
| 2026-10-03 | this local coefficient-execution sweep | Batch execution, TFO-lite execution and the lifecycle append seam now have explicit ownership behind the existing coefficient-search facade | Cozo/Vulkan coefficient target built with three compile threads; 8/8 affected FlyDelta CTests passed serially after correcting two private helper definitions; no public contract or candidate semantics changed | Batch arm execution and TFO-lite share a private execution contract; lifecycle append remains a separate reference-only seam. No ranking, dose, diagnostic/full boundary, Oracle, evidence or promotion semantics changed. |
| 2026-10-03 | this local lifecycle-writer sweep | Daemon orchestration snapshots now use the common validated/idempotent FlyDelta lifecycle writer instead of two direct record-construction copies | Focused Cozo/Vulkan targets built with three compile threads; lifecycle, search-pipeline and orchestration CTests passed serially; full FlyDelta suite passed 54/54 and Qwen Instruct server-context smoke passed after the preceding descriptor sweep | The writer accepts only bounded JSON objects and persists `flydelta_experiment/running` state; it is resumable experiment state, not evidence, promotion or activation authority. No lifecycle backend, search policy or status semantics changed. |
| 2026-10-03 | this local resource-scope sweep | Resource authority cloning and lifecycle-scope matching now have a private implementation unit separate from JSON/material/capture handling | Cozo/Vulkan targets rebuilt with three compile threads; full FlyDelta suite passed 54/54, model-free FlyDelta smokes passed 3/4 with the existing runtime skip, and Qwen Instruct Vulkan0 server-context smoke completed with the same 4/2 worker configuration and fail-closed UNKNOWN result | Moved only scope-resolution ownership to `tools/agent/adaptation/flydelta/resource-scope.cpp`; resource authority, formats, capture semantics, search, Oracle, lifecycle and promotion behavior are unchanged. |
| 2026-10-03 | this local resource-json sweep | Bounded JSON resource reads now have a private implementation unit separate from resource scope, teaching material and capture/augmentation materialization | Cozo/Vulkan targets rebuilt with three compile threads; full FlyDelta suite passed 54/54, model-free FlyDelta smokes passed 3/4 with the existing runtime skip, and Qwen Instruct Vulkan0 server-context smoke completed with 2 relations, 6 capture arms, 9 candidates and fail-closed UNKNOWN/no promotion | Moved only reference read/size/JSON-object validation to `tools/agent/adaptation/flydelta/resource-json.cpp`; resource authority, formats, capture semantics, search, Oracle, lifecycle and promotion behavior are unchanged. |
| 2026-10-03 | this local augmentation-material sweep | Augmentation-material JSON serialization and deserialization now have a private implementation unit separate from resource scope, JSON reads and capture/persistence workflows | Cozo/Vulkan daemon target rebuilt with three compile threads; serial FlyDelta suite passed 54/54; the existing model-free and Qwen server-context smoke gates remain unchanged and are run for the completed resource domain | Moved only augmentation-material JSON ownership to `tools/agent/adaptation/flydelta/augmentation-material.cpp`; schema, resource references, bounded reads, capture semantics, search, Oracle, lifecycle and promotion behavior are unchanged. |
| 2026-10-03 | this local teaching-material sweep | Teaching-relation provenance, JSON persistence and parsing now have a private implementation unit separate from context, capture and augmentation handling | Deterministic validation remains covered by the existing teaching-material/resource callers; the previous serial FlyDelta suite was 54/54 and the Qwen rerun exposed the existing model/fixture parse-failure boundary before this ownership-only move | Moved only teaching-relation serialization/parsing ownership to `agent-flydelta-teaching-material.cpp`; relation schema, resource authority, capture planning, search, Oracle, lifecycle and promotion behavior are unchanged. |
| 2026-10-03 | this local capture-material sweep | Hidden-state capture JSON serialization, validation and persisted-capture loading now have a private implementation unit separate from teaching/resource/augmentation workflow logic | Validation is covered by the existing capture-manifest, hidden-state-hook, server-binding and representation-augmentation tests; the resource-domain model smoke remains explicitly unverified because Qwen stopped at its existing host-parse boundary | Moved only capture JSON/load ownership to `agent-flydelta-capture-material.cpp`; capture layout, deep-copy/resource behavior, server-context execution, search, Oracle, lifecycle and promotion semantics are unchanged. |
| 2026-10-03 | this local decision-margin ownership sweep | The optional host decision-margin challenger now has its own daemon implementation unit behind the existing private seam | Direction-search, evaluator, server-binding and orchestration tests remain the relevant deterministic coverage; no new model execution was introduced by this ownership-only move | Moved only challenger request/configuration and candidate construction to `tools/agent/daemon/flydelta/decision-margin.cpp`; missing host decision-pair/output-head callbacks remain a fail-closed no-challenger result, and search, Oracle, lifecycle and promotion semantics are unchanged. |
| 2026-10-03 | this local provider-factory ownership sweep | Resident host/model validation and FlyDelta resource-provider initialization now have a private helper separate from callback composition | Server-binding, orchestration and daemon smoke build paths are the relevant deterministic coverage; no new runtime or model route was introduced | Moved only provider construction/lifecycle-store initialization to `tools/agent/daemon/flydelta/provider.cpp`; host validation, model/profile/layout identity, resource authority defaults and lifecycle semantics are unchanged. |
| 2026-10-03 | this local execution-binding ownership sweep | Core resident execution callbacks (capture, arm preparation/finalization, teacher-forced scoring, decision-margin challenger and counterfactual comparison) now have a private binding helper separate from provider construction and evaluator registration | Server-binding and orchestration tests remain the deterministic coverage; the helper only assigns the existing callbacks and primitives, with no new evaluator or runtime path | Moved only callback composition to `tools/agent/daemon/flydelta/binding.cpp`; the existing server-context host, batch path, Oracle/evidence semantics and lifecycle/promotion boundaries are unchanged. |
| 2026-10-03 | this local FlyDelta path-shortening sweep | Daemon and adaptation implementation units now use domain directories with short names while their existing private seams and CMake target identities remain stable | The Cozo/Vulkan build is the first structural check; focused and full agent tests follow after all path sweeps | Current implementation paths are `tools/agent/daemon/flydelta/{binding,provider,decision-margin}.cpp` and `tools/agent/adaptation/flydelta/{resource-adapter,resource-scope,resource-json,augmentation-material,teaching-material,capture-material}.cpp`; this is an ownership/path refactor only. |

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

Run the optional model repair bridge with an explicit model argument:

```text
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-repair-model-smoke.sh \
  --model /path/to/model.gguf \
  --suite docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json
```

The wrapper also accepts `LLAMA_AGENT_MODEL` and
`LLAMA_AGENT_DATASET_QUESTION_SUITE`; explicit arguments take precedence.
Relative model and suite paths are resolved from the repository root.

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
LLAMA_AGENT_THREADS=3 \
scripts/test-agent-flydelta-dataset-question-model-smoke.sh \
  --model /path/to/model.gguf \
  --suite docs/examples/agent-flydelta-dataset-question-suite-incremental-v2.json
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

The response-search budget is counted in actual new model evaluations. In
particular, one golden-ratio refinement iteration probes one new alpha and
reuses the existing pivot/bracket point; it is not a two-arm model call hidden
behind one `max_zoom_trials` unit. This keeps the persisted budget comparable
between scalar and batched model hosts.

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

The daemon does not manufacture a model host for this lane. A production
runtime must register the existing backend-neutral `common_flydelta_model_host`
or a fully built `common_flydelta_model_adapter` before the lane becomes
consuming. The resident server-context runtime now provides
`common_agent_server_context_host_make_flydelta_model_host()` as the concrete
composition seam: it owns model residency, fresh inference contexts, the
`generate_batch()` backend call and execution telemetry, while the caller binds
opaque reference resolution, semantic request construction, finalization and
host verification. The worker must not reach into a session manager,
`llama_context`, server context or model path directly. If no registration is
present, the FlyDelta lane remains configured-but-idle and a queued job is not
consumed; this is a capability/configuration gap, not a successful search with
no utility.

Embedding daemons use
`common_agent_daemon_register_flydelta_server_binding()` to perform that
registration as one operation. It accepts the resident server-context host and
the host-owned `prepare_arm`, `finalize_arm`, `register_evaluator` and semantic
verification binding, then installs the resulting backend-neutral host and
adapter in the daemon runtime. The helper is deliberately explicit: the
daemon cannot invent a semantic verifier, and a runtime with no binding keeps
the lane idle rather than falling back to an unverified model path.

The canonical callback bundle is
`common_agent_server_flydelta_binding_callbacks`. An embedding host normally
constructs it once during runtime composition and uses
`common_agent_server_flydelta_binding_from_callbacks()` from its startup
factory:

```cpp
common_agent_server_flydelta_binding_callbacks callbacks;
callbacks.primitives = host_flydelta_capabilities;
callbacks.prepare_arm = host_resolve_and_prepare_arm;
callbacks.finalize_arm = host_finalize_and_verify_arm;
callbacks.register_evaluator = host_register_flydelta_evaluator;
callbacks.run_concept_capture = host_run_concept_capture;

common_agent_daemon_set_flydelta_server_binding_callbacks(
    runtime, std::move(callbacks), error);
```

The daemon converts this callback bundle into the existing startup factory
before acquiring the resident model. An embedding host may still assign
`runtime.flydelta_server_binding_factory` directly when construction needs the
resident server host. If neither form is supplied, FlyDelta remains explicitly
configured-but-idle; the generic daemon never fabricates semantic references or
host outcomes from JSON configuration.

`host_resolve_and_prepare_arm` remains responsible for resolving the opaque
context, fixture and intervention references. `host_finalize_and_verify_arm`
is the only place that may assign a semantic host outcome. The factory is a
composition helper, not a default semantic implementation; missing callbacks
must remain a startup/configuration failure or an idle lane, never a fabricated
`HELPED` result.

Worker evaluation resolves candidate, suite, fixture and context references
through a provider view derived from the immutable `job.seed.scope`. The
resident host is shared, but its default resource authority is not used for a
queued job; this keeps concurrent worker jobs from crossing namespace or
session boundaries. Diagnostic arms with `request_generation=false` still
evaluate their prompt and may capture hidden state, but are sent to the
server-context host with `n_predict=0` so they do not sample a throwaway token.
Full-generation arms must opt into decoding explicitly.

The current phase/authority map is:

```text
queued job
  -> runner provider view(job.seed.scope)
  -> resident server-context host (shared execution owner)
  -> scoped resource reads/writes and capture lifetime
  -> common search/orchestration policy
  -> opaque lifecycle/state references for resume
```

Concept capture/synthesis, initial search, BootstrapZoom, AdaptiveAlpha,
orthogonal search, post-Bootstrap controls, representation augmentation and
counterfactual execution all create that job-scoped provider view before
resolving job-owned material. The append-only lifecycle journal remains a
host-owned shared store. Job-aware state callbacks filter reads and writes by
the queued job's namespace, project and session; legacy opaque callbacks remain
compatibility interfaces and must not be used for scoped resume. This keeps
durable lifecycle authority host-owned without allowing one job scope to
resolve another job's state reference.

The existing runtime also exposes the lower-level pieces needed by that host
binding: the inference object accepts immutable FlyDelta activations and
capture requests, and the server-context backend supports isolated
teacher-forced choice scoring. The semantic resolver/finalizer remains host
owned because only that layer can resolve fixture and intervention references
and assign host outcomes. This is intentionally not implemented in the
generic worker or as a second inference adapter hierarchy.

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
- V0 compatibility is exact-model only: the canonical GGUF basename
  (`base_model_id`), tokenizer, template and architecture/layout must match.
  A content fingerprint is optional in V0 and, when present in the expected
  runtime profile, is an additional exact check. A sideband is not presumed
  portable between Q2/Q4/Q8/FP16 variants merely by family name. The model id
  is a basename, never an absolute path or a fuzzy family label.

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
and outcome. For arms that reached host verification, the same trace also
records compact `expected_decision`, `observed_decision` and `verifier_reason`
fields. These fields answer the operator question "expected X, got Y, so why
was this HELPED/HARMED/UNKNOWN?"; they are bounded diagnostic text, not
evidence, learning credit or promotion authority. Search-pipeline slices
additionally retain the existing Whirlpool round traces (probes, recentering,
radius and model-evaluation count). The worker bounds this material and never
includes prompts, model outputs, credentials or activation tensors.
The daemon scheduler uses the same reference-only collection boundary for a
returned `next_action`: it carries forward the claimed search-pipeline job,
updates opaque state references and enqueues at most one idempotent follow-up
job. It never evaluates that job recursively inside the worker slice.

Before that follow-up is collected, the production daemon associates a
search-pipeline job with the unique compatible teaching-material family for
its `(teaching_key, behavior_key)` when a semantic family key is present, or
for its `behavior_key` for legacy jobs. The association compares the material
runtime compatibility identity and fails closed on ambiguity; it only
transports the opaque `teaching_material_group_ref`. This is the family
binding that allows a terminal augmentation slice to schedule concept capture
or synthesis without putting family policy in the generic runtime. Existing
jobs with an explicit group reference are preserved.

For resumable rank-one BootstrapZoom/AdaptiveAlpha work, the queue transports
only `bootstrap_zoom_state_ref`. For later rank-one plateau, orthogonal-search or
augmentation slices, it transports the separate opaque `search_state_ref`;
augmentation may additionally use the explicit
`representation_augmentation_state_ref`.
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

This opaque state-reference contract is also the V0 save/resume boundary for
FlyDelta learning and search. State and `.flyd` artifacts are reusable only
when the resolved runtime model has the same `base_model_id` and the same
tokenizer/template/architecture/layout compatibility fields. A present
expected content fingerprint is checked as an additional guard; V0 does not
require one when the runtime cannot provide it consistently. This reuses the
existing artifact/state stores rather than creating a parallel training
database. Persistent registry replay and a stronger mandatory fingerprint
policy remain later lifecycle work.

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
production adapter must be constructed by the runtime host, where the selected
model profile, backend inference object, reference/fixture resolution, fresh
contexts and host verification are available. The adapter contract is
model/backend-neutral; model names used by model smokes are test configuration,
not part of the worker or adapter API. The common worker never creates that
adapter, and an advertised capability is not evidence that a correction works.

The model-facing host contract also exposes one generic
`common_flydelta_arm_request`/`common_flydelta_arm_result` bounded-arm seam.
The existing `common_flydelta_search_pipeline_runner_from_model_host()` helper
maps that result into the ordinary pipeline runner, including absolute margin,
baseline-relative margin handling, geometry, outcome and provenance. This is
deliberately one shared transport path: Whirlpool, BootstrapZoom,
AdaptiveAlpha, orthogonal probes, augmentation and coefficient search do not
get separate model APIs. The host resolves the opaque fixture/intervention
references and owns fresh inference, capture, generation, teacher scoring and
verification; the common search code owns only bounded search and transition
policy.

The bounded-arm transport is validated at the adapter boundary before it is
passed to the search runner. Request validation covers layer/coefficient shape,
finite dose values, a stable `arm_id` and fresh-context requirements for
overlays. The host must echo that `arm_id` in the result; retries of the same
job/surface arm can then be recognized without comparing activation payloads.
Result validation covers finite dose/geometry values, consistent
`executed`/`host_evaluated`/`verifier_known` flags and any returned
decision-margin or baseline-comparison contract. A host may return a lower
`executed_alpha` than requested only when it marks the result
`dose_safety_limited`; the common transport never silently changes a dose.
These checks are transport invariants only:
they do not convert an arm into utility, HELPED, learning credit or promotion.

The same contract has an optional batch form for backend acceleration:
`common_flydelta_arm_batch_request` contains independent arm requests and
`common_flydelta_arm_batch_result` contains one result per request in the same
order. A host that does not support batching uses the common CPU fallback,
which invokes the existing scalar callback once per arm. This keeps batching
an execution optimization rather than a new search policy or evidence path.
The registered capability `bounded_arm_batch` is an explicit runtime opt-in,
and must be advertised only after the concrete backend/server binding has
validated that multi-arm execution is supported. A registered callback alone
does not enable batching, so a scalar-only or otherwise unvalidated host
cannot accidentally advertise device batching. The host adapter may derive
the final capability view from its validated registration, but it must not
infer it from callback presence alone.
The scalar callback is optional when a backend exposes a validated batch
callback: a single-arm search request is wrapped as a one-arm batch, so a
device host does not need to implement a duplicate scalar path merely to
participate in the common search seam.
Per-arm `alpha`, coefficients, layer mask, fresh-context semantics, margins,
geometry and outcomes remain independent. The experimental resident-server
binding can now evaluate compatible sparse overlays as per-sequence parameters
in one resident-server model batch without changing Whirlpool, AdaptiveAlpha,
Shallow/Deep or TFO; hosts without the opt-in capability retain the scalar
fallback.

The common batch seam is consumed by the bounded Whirlpool region probes and
by rank-N coordinate/TFO coefficient populations. The baseline remains a
scalar reference; safety backoffs are submitted as a separate bounded batch,
and TFO iterations remain sequential CPU control-plane steps. Every affected
phase has separate diagnostic and full-generation execution classes: the
diagnostic wave requests prompt evaluation, capture and teacher-forced
margin/geometry only, while the pruned top-K frontier explicitly requests full
generation and host verification. The scalar public entry points remain
compatibility wrappers over the same contracts, so batching changes execution
shape but not proposal order, dose decisions, utility gates or lifecycle
semantics.

Rank-N coefficient search also accepts an explicit `max_batch_arms` wave limit.
Production callers should copy the registered model-host capacity into this
field for Shallow/Deep/TFO; zero leaves the limit to the batch callback. The
field only bounds one device-facing submission and never changes the proposed
coefficient population, retry order or CPU-side dose/utility decisions.

Resident server-context memory is a separate host concern. The project has
historically used `n_gpu_layers = 99` as an "all layers" convenience value.
When `fit_params` and `reserve_flydelta_workspace` are enabled, the resident
host translates that legacy request to the existing fit path (`-1`) before
model load. This lets the fit logic choose an offload level with its existing
device margin instead of treating an explicit 99 as non-adjustable and
leaving no proven workspace for cvecs, diagnostic reductions or transient arm
waves. Explicit smaller layer counts remain explicit, and a host may disable
the reservation for a deliberately full-offload configuration. The trace
reports requested layers, effective pre-fit layers and whether the workspace
reservation was applied; this is an execution/resource decision, not a
FlyDelta search or evidence decision.

Resident FlyDelta arm waves use the server's vector task-posting seam. The
host prepares and validates every non-streaming completion task first, then
submits the complete vector through `server_response_reader::post_tasks()`.
The server queue inserts that vector under one queue lock before waking its
loop, so the wave is visible as one posting operation before slot selection
and decode. This replaces the former timing-dependent worker post gate. The
response vector remains in request order and each result retains its own
capture, cvec and runtime metadata. Streaming is rejected explicitly for
this bounded resident-arm path; it never falls back to partially visible
posting. Hosts without a valid device batch still use the ordinary scalar
fallback, with the same search, Oracle and lifecycle semantics.

Batch capacity is owned by the registered model host, not by an individual
search algorithm. The resident server derives `max_arms_per_batch` from its
validated number of parallel sequences and the common adapter partitions a
larger request into bounded waves. The current runtime keeps
`max_inflight_batches` at one: model execution is synchronous at this seam and
the host scheduler remains responsible for cross-job admission. The effective
shape is therefore:

```text
Whirlpool round / Shallow controls / Deep controls / TFO population
        -> independent arm request set
        -> host-capacity waves (<= resident parallel slots)
        -> ordered arm results
        -> CPU policy, dose, UtilityGate and lifecycle decision
```

AlphaResponse remains sequential at the policy level because the next alpha
depends on the previous response, safety boundary and bracket. Its future
device optimization is to batch only independent refinement probes; it must
not hide a retry or backoff inside the host callback. This makes the phase
budgets comparable: Whirlpool uses its probes-per-round, Shallow/Deep use
their bounded control/top-K budgets, and TFO uses its population size, while
the model host caps the actual concurrent wave without changing any proposal
or evidence semantics.

Teacher-forced comparisons have the same backend boundary through
`common_agent_teacher_forced_choice_batch_request` and its result type. The
default inference implementation executes the entries through the existing
scalar scorer in order. The resident server inference backend now submits all
positive/negative tasks for one request through a single server response
reader, then reconstructs one isolated result per choice. The server may
still execute those tasks serially when their context-wide cvec identities
differ; the batch boundary therefore removes duplicate reader/collection
plumbing but does not claim device throughput. Prompt/KV reuse remains
disabled for every comparison task, and cvec changes still invalidate the
server slot prompt cache. A future Vulkan/device implementation may replace
only this method with a true per-sequence model batch, but must preserve one
result per choice and the context/overlay identity of every entry.

This is also the intended execution shape for bounded scale search. The CPU
may propose a small set of independent alpha arms (for example the current
bracket and up to four golden/refinement probes), while a device backend
evaluates those arms as separate sequences and returns one compact result per
arm. The CPU then owns the next bracket, UtilityGate and stop decision. This
does not move golden-section, Whirlpool or lifecycle policy to a shader. It
requires per-sequence overlay parameters and independent KV/context state;
the resident server provides that capability behind its explicit opt-in;
other hosts retain the correct scalar or serialized fallback.

The llama core exposes a deliberately small opt-in graph hook for this
backend: `llama_adapter_cvec_batch_ref` is a non-owning callback
reference carried alongside the existing scalar cvec in `llm_graph_params`.
When installed through the internal `llama_context` seam, the callback may
apply backend-owned per-sequence control vectors for the current `llama_ubatch`.
The scalar `llama_set_adapter_cvec()` path remains the default, and the core
does not allocate, persist or interpret the backend's overlay table. The
reference and all data it points to must remain valid until graph execution
has completed; changing the reference also participates in graph reuse
identity. The resident server now uses this seam for its experimental
per-sequence Vulkan binding when the explicit runtime opt-in and backend
preconditions are satisfied; other hosts may continue to use the scalar
fallback.

The core also contains `llama_adapter_cvec_batch`, a small backend-neutral
reference table behind that hook. It owns one backend-resident F32 row per
bound sequence plus an explicit zero fallback row, and uses `ggml_get_rows()`
to select the row for each token. The zero row lets a sparse single-arm
binding coexist with inactive graph sequences without inheriting another
request's overlay. A token shared by bound and unbound sequence IDs also uses
that zero row rather than silently selecting the first ID. This makes a
one-arm sparse binding safe while retaining the existing dense scalar fallback
for hosts that do not opt in. The server-side `server_task_cvec_batch` is the matching
ownership/geometry view: it keeps the immutable request cvecs alive and
materializes sequence IDs plus payload pointers for a backend. The resident
server connects this view to the core callback without moving model execution
or search policy into FlyDelta; hosts that do not enable the binding continue
through the scalar path.

The resident server receives per-sequence cvec batching through the typed
`runtime.adaptation.flydelta.batch_mode` policy: `disabled`, `auto` (the
default) or `required`. In `auto`, a FlyDelta-enabled server-context profile
without an explicit parallel capacity receives a low two-arm capacity when
GPU layers are configured. If the context, backend or overlay geometry cannot
support the binding, the same logical wave uses the isolated scalar fallback
and daemon status records that fact. `required` instead fails startup when
native batching cannot be registered. The binding remains available only when
the target context has no speculative draft context and every participating
slot has a compatible dense cvec layout and sparse layer mask. Different cvec
identities then become rows in one graph table; equal identities continue
through the scalar path. If the binding cannot be prepared, the server fails
that physical batch rather than applying one slot's overlay to another slot.
The table, selector indices and device buffers are synchronized and detached
before replacement, so asynchronous decode cannot observe freed overlay
buffers. The legacy `LLAMA_SERVER_PER_SEQUENCE_CVEC` environment setting is
retained only for direct non-agent server users; the agent host applies its
typed setting before model load. The production model host must bind both the
bounded-arm callback and the explicit `bounded_arm_batch` capability before the common
FlyDelta adapter can use the batch path; otherwise the common scalar fallback
is retained.

A backend integration is intentionally small and can be kept behind its
existing model-host abstraction. The usual flow is:

```cpp
struct backend_overlay_table {
    // Device buffers, layer lookup and the per-sequence overlay identity.
};

static ggml_tensor * apply_sequence_overlays(
        ggml_context * ctx,
        ggml_tensor * cur,
        int il,
        const llama_ubatch & ubatch,
        void * opaque) {
    auto & table = *static_cast<backend_overlay_table *>(opaque);

    // Backend-specific: map each token in ubatch to its sequence and build
    // one [hidden_size, n_tokens] overlay tensor for this layer. A typical
    // implementation uses a sequence-index tensor plus ggml_get_rows().
    ggml_tensor * overlay = table.overlay_for_layer(ctx, il, ubatch);
    return overlay != nullptr ? ggml_add(ctx, cur, overlay) : cur;
}

backend_overlay_table table = make_overlay_table(...);
llama_adapter_cvec_batch_ref ref {
    /* .apply = */ apply_sequence_overlays,
    /* .user_data = */ &table,
};

context.set_adapter_cvec_batch(&ref);
// Build/decode the bounded batch while `ref` and `table` remain alive.
context.set_adapter_cvec_batch(nullptr); // restore scalar cvec handling
```

This is pseudocode for the binding seam, not a new public C API. The backend
must decide how `ubatch.seq_id`/`seq_id_unq` select rows, how layer overlays
are stored, and how device execution is scheduled. The callback is invoked
once for each graph layer that requests cvec application; it should only build
the corresponding graph expression. It must not mutate FlyDelta state, choose
the next search arm, or perform host verification. If a backend cannot provide
per-sequence overlays, it simply does not install the reference and the
existing scalar/serialized path remains authoritative.

The compact geometry fields in `common_flydelta_arm_result` are also the first
GPU boundary. For ordinary search ranking the host only needs the reduced
scalars `cosine`, `progress`, `leakage` and `shift_norm` (plus teacher-forced
margin totals/counts). Full hidden-state captures should cross back to the CPU
only when the arm is being retained as basis, donor, orthogonal or concept
material. The CPU remains the owner of proposal, budget, utility, evidence and
lifecycle decisions; the model backend owns reductions and model execution.

The overlay transport now has the same split. `ArmRequest.layer_indices` and
`coefficients` are the sparse per-arm control description. The common overlay
composer can materialize layer-sparse data and can explicitly expand it to the
legacy dense cvec layout. Only the legacy model boundary needs that expansion;
a Vulkan backend should consume the sparse layer list and coefficients as
per-sequence device parameters. This keeps dense cvec allocation out of the
search policy and makes the sparse path an execution optimization rather than
a second overlay representation.

Activation results carry both forms during this migration: the sparse overlay
is the canonical per-arm material, while the dense overlay is its validated
compatibility expansion for the current scalar cvec server path. Legacy
dense-only activation results remain accepted so older host integrations can
migrate without changing search semantics. When the dense cvec path is active,
the runtime also disables prompt/KV reuse for that request. Changing the cvec
identity clears slot-local prompt state. Because the persistent server prompt
cache does not carry per-sequence overlay identity, the first cvec-backed task
also clears and disables that global idle prompt cache for the lifetime of the
resident context. This prevents a prompt state produced under one overlay (or
without an overlay) from being selected before the task-specific cvec is
applied. The server reports the applied cvec hash, payload size and cache
flags in the generation result; FlyDelta model smokes fail closed if an active
arm does not report a non-empty applied cvec with prompt reuse disabled.

The sparse overlay batch contract is likewise execution-only. Every enabled
entry in `common_flydelta_sparse_overlay_batch` must have a distinct artifact
identity. This is intentional: two sequences may have identical prompt tokens
but different overlays, and a backend must not alias their cvec/KV state. The
common fallback can expand each entry independently to a legacy dense cvec
through `common_flydelta_expand_sparse_overlay_batch()`; it never merges the
entries. The current server validates and executes these entries either
through the opt-in per-sequence device table or through the isolated scalar
fallback; neither path relaxes the cvec identity rule.

The four compact geometry values in an arm result have a CPU reference oracle
in `common_flydelta_representation_diagnostics_from_vectors()`. A device
reduction returns the same four scalar inputs (`dot_shift_delta`,
`shift_squared`, `delta_squared` and `residual_squared`), from which the host
uses the same CPU-parity conversion to `cosine`, `progress`, `leakage` and
`shift_norm`. Full capture transfer remains reserved for material that will
become basis, donor, orthogonal or concept evidence.

The resident server-context path supports a paired diagnostic wave: a
request-scoped no-op reference row and one or more overlay rows are submitted
to the same physical per-sequence cvec batch. The graph captures the requested
layer input for both sequence rows, computes the four reductions on the
selected non-CPU backend, and returns only the compact scalars. The reference
row is not expected to produce a reduction of its own; it is the baseline
operand for the overlay row. If the rows cannot be co-batched, the direction
or layer is invalid, or the reduction outputs land on the CPU backend, the
result remains unavailable and the host must use the existing scalar/capture
fallback rather than claim `device_reduction_used`.

`device_reduction_used` therefore means more than a registered callback: all
four reduction outputs were assigned to a non-CPU backend and were read back
as compact values. `capture_bytes_to_host=0` is expected for a geometry-only
diagnostic wave. This is execution telemetry only; it does not change search,
Oracle truth, evidence, learning credit or promotion.

Arm results may also carry optional execution telemetry: model, teacher-forced
and generation time, overlay/capture transfer bytes, and whether device
reduction or batched execution was used. This telemetry is for benchmark and
regression comparison only; it cannot make an arm safer, useful, HELPED or
learning-eligible. The resident binding records whether its overlay table was
allocated in a non-CPU backend buffer. When that fact is present, the
production adapter reports `device_batch`; otherwise it reports
`backend_batch` or the scalar fallback. This is measured graph/buffer
provenance, not an assumption based only on the batch callback. It still does
not mean compact geometry reductions or teacher-forced scoring are
device-batched: those remain separate capabilities until their backend
results carry the same proof. Benchmarks must compare scalar fallback,
backend batch and device batch on the same arm identities.

The telemetry also records an explicit execution path: `scalar`,
`scalar_fallback`, `backend_batch` or `device_batch`. The common runner marks
the fallback when a host has no batch callback, and marks an otherwise
unlabelled batch callback as `backend_batch`. A concrete host may upgrade that
label to `device_batch` only when it proves non-CPU per-sequence overlay
execution. The path is diagnostic provenance only;
it never changes search utility, evidence depth, host outcome or learning
credit. A `scalar` path may additionally report
`sparse_device_binding_used=true`: one sparse overlay row was
backend-resident, while the logical arm count remained one. It is deliberately
not relabelled `device_batch`.

The arm identity is intentionally narrower than experiment lifecycle identity.
`job_id`, fixture, intervention, layer mask, scale and baseline-vs-overlay
mode form one retry-safe model arm. Surface revision, parent-best comparison
and the immutable fixture baseline remain orchestration/lifecycle state and
must not be inferred from the arm id alone. A resumed worker therefore keeps
both the persisted state reference and the arm id in its trace.

For a runtime that owns the model-facing evaluator but does not want to expose
its callback bundle directly through daemon setup, it may instead register a
`common_flydelta_model_host`. Its registration callback fills the evaluator
configuration and host-owned callbacks exactly once at daemon startup; the
daemon then composes the ordinary `common_flydelta_model_adapter` from that
registration. This is only a composition seam: the host still owns model
residency, reference/fixture resolution, fresh inference contexts, margin
scoring and verification. The generic worker receives no raw model context and
does not recursively execute the next phase.

The host registration is fail-closed for advertised post-Bootstrap features:
when `orthogonal_search` is enabled, the callback bundle must include the
bounded `run_search_pipeline_with_search_state` runner. That runner is where
the host resolves the persisted surface, prepares the typed orthogonal input,
runs fresh model probes and returns the next opaque state reference. A capability
flag without that callback is a configuration error, not an orthogonal search
that found no direction.

Primitive capabilities (`capture`, `overlay`, `generation` and
`teacher_forced_scoring`) are facts about the registered runtime. Algorithm
capabilities are derived from those facts and from the callbacks actually
registered: Bootstrap/AdaptiveAlpha require a bounded arm plus a normal search
runner, while orthogonal search and representation augmentation additionally
require the resumable state-aware runner. A capability is therefore not
enabled merely because an algorithm exists in common code.

The daemon runtime carries the optional adapter registration into the
dispatcher and exposes two separate status facts: whether an adapter is
registered, and whether it advertises model-facing search support. A configured
FlyDelta lane without a registered adapter is therefore an observable idle
configuration, not a callback that silently captures user-session state. A
runtime host may provide either a ready adapter, a `common_flydelta_model_host`,
or a shared evaluator configuration plus host-owned evaluator callbacks; daemon
startup composes the latter two through
`common_flydelta_model_adapter_from_host()` or
`common_flydelta_model_adapter_from_evaluator()`. Missing host callbacks fail
closed and leave the lane idle rather than fabricating model execution.
When an adapter is present, daemon capability observability is copied from that
adapter at startup; manually supplied phase flags cannot override the derived
capabilities of the registered bounded-arm and search callbacks.

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
adaptive `WHERE` phase and evidence-driven `WHAT` work. There are two related
selection contracts here and they must not be conflated: the local Whirlpool
or region selector gives learning credit only to a host-known `HELPED` arm,
while the orchestration layer may return a safe, promising `UNKNOWN` or
`NEUTRAL` arm as a reference-only search continuation when no helped arm is
available. Such a continuation does not grant learning credit, promotion or
activation. The host then resolves compatible samples for the same behavior
identity and selected layer, assesses evidence depth, and uses the resulting
plan:

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
aggregated. The aggregation snapshot therefore exposes two views: the
experimental/search view retains compatible non-`HARMED` observations, while
the evidence-depth view contains only `HELPED` observations with explicit
learning eligibility. Retention is bounded by
`aggregation_max_retained_samples`; the configured bound must still allow the
deep threshold to be reached.

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
| Bootstrap | one learning-eligible `HELPED` sample or rank-one evidence; an experimental surface with no such sample remains Bootstrap-only | up to 4 region arms plus bounded rank-1 BootstrapZoom/AdaptiveAlpha arms; no coefficient search | run the smallest model experiment and collect cosine, progress, leakage, shift norm and decision margin; a useful signal may refine alpha/profile locally |
| Shallow | at least 2 compatible samples and effective rank at least 2 | up to 8 region arms, up to 4 coefficient proposals, top 1 full arm | compare a small rank-2 basis and cheap margin/geometry controls |
| Deep | at least 6 compatible samples, effective rank at least 2, stable geometry and valid condition bound | up to 32 region arms, up to 16 coefficient proposals, top 3 full arms; TFO-lite allowed | build robust aggregate/Deep basis and run Deep controls; coefficient search/TFO-lite only after positive Deep UtilityGate |

The depth result chooses a budget; it does not itself run a model or promote a
candidate. An experimental search may still continue with a Bootstrap slice
before any natural `HELPED` evidence exists, but that empty evidence view is
never treated as rank one or rank two. Bootstrap therefore does perform diagnostics when its small model
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
positive/negative coordinate stencil. The staged helper executes all
proposals diagnostically and then re-runs only the bounded top-K frontier in
the full-generation callback. The helper itself still derives
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

### Semantic basis resolution — implemented admission seam

`common_flydelta_resolve_semantic_basis()` is the semantic admission layer in
front of the existing Gram--Schmidt builder. It accepts a bounded collection
of direction candidates together with their synthesis descriptors and only
composes candidates whose concept, behavior, model, tokenizer/template,
capture-layout, scope and Oracle revisions all match the query. Candidates
must already be host-verified and non-experimental; incompatible, stale or
unverified material is skipped, and collinear candidates do not consume rank.

The resolver does not create a new basis store, evaluator or lifecycle path.
It returns the selected candidate indexes and the ordinary low-rank basis,
which then enters the existing coordinate/TFO coefficient search. Descriptor
filtering and orthogonalization are CPU-side; model work remains the existing
bounded coefficient-arm evaluation. Current concept synthesis still runs its
two activation-derived frontier candidates, while semantic basis reuse is
available when a registry contains compatible verified directions.

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
the canonical GGUF basename as `base_model_id`, an optional content
fingerprint, architecture and tokenizer/template fingerprints,
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
existing path. Registry admission has the same retry rule for an identical
manifest: an identical `id` and metadata is a successful no-op, while changed
metadata for an existing `id` is rejected. The store does not decide lifecycle
status, evaluate a candidate, or activate a sideband.

The sideband manifest now carries an explicit `namespace_id`, `project_id`,
optional `expires_at_epoch_ms` and revocation reason. Its in-memory registry
enforces the lifecycle transitions `candidate -> canary -> active -> retired`
and supports explicit `revoke(reason)`. Expired entries cannot enter canary or
be resolved; revoked entries cannot be resolved. Manifest JSON includes these
fields so the review journal can reuse the same contract. The
`common_flydelta_sideband_review_store` records operator or host-automation
decisions in the existing adaptation lifecycle backend and can replay them into
the registry after a restart. The `llama-agent-adaptation-admin` utility
exposes FlyDelta review listing and explicit review application; application
still requires an approval flag and the normal HELPED/evaluation gates. The
daemon startup binding to automatically replay this store is a separate runtime
composition step; no model-facing path can invoke it.

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
existing model-residency manager remains the owner of the resident model; a
future backend may attach a prepared, cacheable basis resource to that
resident model, but FlyDelta must not create a second model-residency cache.
The residency/cache identity may include declared sideband identity and
profile scale, while the registry additionally checks the resolved artifact
hash and compatibility metadata.

The prepared-resource and activation lifetimes are deliberately separate:

```text
resident model + compatible artifact/basis
    -> optional prepared backend resource
    -> ephemeral per-turn activation binding
    -> fresh inference context
```

The prepared resource may be reused only when model, backend, artifact, layout
and device compatibility match. Per-turn alpha, coefficients, sparse code,
gate decision and context identity remain ephemeral and are never cached as a
resident model property.

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
The server-side cvec identity includes the artifact identity, content hash,
dimensions and payload. A changed per-sequence overlay therefore clears the
slot's prompt/KV state before the new cvec is applied; this is covered by the
prepared-generation contract test. A future batched Vulkan path must preserve
the same isolation rule rather than relying on token equality alone.

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

### Host-taught concept extraction — initial contract

Some host-owned procedures and verified corrections are useful as concept
material even when they are not tool-repair deltas. The initial extraction
path is deliberately a builder on top of the existing direction contract, not
a second intervention or learning pipeline:

```text
host-approved ConceptSpec
    -> matched baseline / conditioned / control captures
    -> control-residualized concept trajectories
    -> raw / trimmed / diagonal-whitened rank-one candidates
    -> existing FlyDelta direction/search contract
    -> normal layer, dose, margin and host-verifier evaluation
```

The host must provide the behavior identity, an extraction identity, procedure
and verifier references, model/tokenizer/template/layout fingerprints, scope
and redaction attestation, and host verification for every conditioned
trajectory. `concept_key` identifies the reusable concept; `extraction_id`
identifies one immutable extraction run. A
procedure blueprint also requires a matched control capture. The builder
rejects unapproved, unaligned, mixed-layer or unverified material.

Concept candidates are emitted with `origin=host_taught_extracted`. They are
always `experimental_only` and never learning-eligible at this stage. The
conversion into the ordinary direction candidate preserves that provenance;
it does not bypass the existing utility gate, worker slice, host verifier or
promotion lifecycle. The candidate's transient vectors are builder input;
persisted state should retain references and diagnostics rather than raw
hidden-state payloads.

The first implementation intentionally stops at CPU-cheap rank-one builders.
It does not introduce gradient steering, SFT, a new concept registry or a new
model-residency path. Later concept families can reuse the same contract by
changing the explicit host-approved `behavior_key` and evidence source.

Resolved teaching relations now have a host-owned material accumulator seam.
The runtime assembly forwards each admitted procedure, user-correction or
user-taught relation to that observer; the accumulator groups only by
`teaching_key`, `behavior_key` and the complete model/tokenizer/template,
capture, context, scope and verifier compatibility identity. It deduplicates
relation IDs and exposes two separate readiness levels: a relation set is
ready when enough compatible relations exist, while concept material is ready
only after enough matched baseline/conditioned/control trajectory references
have been captured. Persistence remains in the existing host
artifact/lifecycle layer; the generic runtime does not create a second store
or retain prompts/activations. A small host-owned material-runtime façade may
be shared by the runtime assembly and model host so both inspect the same
reference index.

When a bounded augmentation branch is terminal, the evaluator may ask the
host for both readiness levels. A relation-ready but trajectory-incomplete
group schedules one `concept_capture` bounded job; that callback resolves the
opaque relation refs, runs the matched capture slice and records trajectory
refs. Only a trajectory-ready group schedules `concept_synthesis`. The
production model adapter advertises the capture/synthesis callbacks only when
they are actually registered; neither capability grants learning credit or
bypasses verification.

The concept smoke has two layers. Its default offline mode covers grouped
aggregation, validated filtering and ordered querying with synthetic matched
trajectories, explicit `seen`/`holdout`/`transfer`/`contrastive` fixture
partitions and contract rejection cases. Its optional `--model` mode uses
the existing resident model/capture path and reports unresolved families when
the model does not produce enough host-verified conditioned pairs. An
unresolved model smoke is diagnostic and carries no learning credit; it is not
allowed to fabricate concept evidence. This gives the smoke useful model
coverage without creating a private model path unavailable to production.

The production wiring is additionally covered by the separate
`llama-agent-flydelta-concept-synthesis-model-smoke`. This smoke is deliberately
not folded into the repair smoke: it starts with an ordinary resident
`search_pipeline` slice, requires its persisted Bootstrap/continuation state
as the localized `WHERE` anchor, then runs the real daemon/server-context
concept capture and synthesis callbacks. Two host-approved relations produce
three fresh model arms each (baseline, conditioned and control), so the model
phase contains six capture arms in one bounded batch. Synthesis emits an
experimental direction reference and the smoke schedules a normal grafted
search job with that reference as `WHAT`. The output reports the production
Whirlpool/Bootstrap trace, capture-arm count, trajectory count, synthesized
candidate metadata and graft handoff. It never grants learning credit,
promotion or activation. Run it with an explicit model and tracing enabled:

```bash
GGML_VK_VISIBLE_DEVICES=<device-index> \
LLAMA_AGENT_RESIDENT_TRACE=1 LLAMA_AGENT_TRACE=1 \
build-agent/bin/llama-agent-flydelta-concept-synthesis-model-smoke \
  --model /path/to/model.gguf --threads 4
```

The executable keeps model inference serialized (`inference_max_active=1`),
uses the resident server-context host and reports the configured four general
agent workers and two FlyDelta workers. The CTest entry skips when no model is
supplied and has a longer timeout because it covers the complete capture →
synthesis → grafted-search chain.

The grafted BootstrapZoom slice may safely retain no arm: when diagnostics
reject every bounded probe, the worker records `no_useful_utility` and stops
the branch without fabricating a candidate, learning credit or promotion.
Every persisted BootstrapZoom/AdaptiveAlpha/orthogonal continuation revision
also receives a new opaque state reference; the previous reference remains
the parent link and is never reused as the lifecycle idempotency key.

When a registered concept-synthesis callback emits a candidate, the evaluator
persists one immutable experimental direction reference through the same
direction registry used by ordinary FlyDelta search. The worker reports that
reference as `graft_direction_ref`; the collection seam then schedules a normal
`search_pipeline` job with the reference as its seed candidate. The originating
capture-manifest and behavior-delta references are carried through the bounded
concept jobs so the grafted search still has its ordinary model inputs. Old
BootstrapZoom state is carried into the new Bootstrap slice, while the
concept-only search/augmentation continuation refs are cleared at the graft
boundary. This preserves the localized WHERE anchor and HOW-MUCH resume
surface while replacing only WHAT with the synthesized direction. No evidence
rank, learning credit or promotion status is granted by the graft.
This is a wiring handoff, not a second concept-search engine or a learned-vector
training path.

Concept capture is likewise localized. The relation supplies semantic WHAT;
the latest durable Bootstrap/augmentation search state supplies WHERE. The
daemon creates a `common_flydelta_concept_capture_plan` per relation and uses
the persisted anchor layer for all three captures:

```text
Whirlpool / Bootstrap local surface
    -> anchor layer L*
    -> baseline @ L*, conditioned @ L*, control @ L*
    -> ConceptSynthesis candidate @ L*
    -> grafted ordinary search @ L*
```

If a multi-layer selected region has no resolvable persisted anchor, capture
fails closed rather than selecting an arbitrary layer. The relation resource
does not choose a model layer; its legacy layer field, when present, is
ignored. The concept trajectory retains the parent surface reference and
revision for comparison and restart provenance, while the evidence rank is
unchanged by synthesis or grafting.

Teacher-forced margin scoring in the model smoke is behavior-specific as well:
the positive continuation is the family target and the negative continuation
is the fixture's nearest declared alternative. It must not use one global
tool pair such as `dataset.inspect` versus `statistics.describe` for every concept
family. The margin remains search evidence only. Canonical/semantic host
verification is a separate gate, so a model response that selects the right
tool but emits a non-canonical or semantically incomplete call remains
diagnostic `CANONICALIZATION_FAILED` or `SEMANTIC_DECISION_WRONG` material
rather than a verified trajectory. Equivalent spellings are normalized before
this classification; the host still requires all semantic fields needed by
the fixture.

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

### Host teaching relation — shared source adapter seam

The host-side teaching seam is shared by all evidence sources. Existing
`common_adaptation_evidence_relation` is the host-owned relation over baseline,
conditioned candidate, verifier, scope and transaction references. FlyDelta
views it as a `TeachingRelation` with an explicit reusability status and
provenance for each side:

```text
evidence source
    -> host admission / reusability
    -> TeachingRelation
    -> existing behavior transition / concept trajectory
    -> FlyDelta capture, search and verification
```

`tool_repair` remains the strict failure/recovery convenience path. A
`procedure_blueprint` relation uses the same reference-only contract; it does
not create a second procedure learner or a second FlyDelta registry. Explicit
`user_correction` uses the same seam only when the host supplies the observed
model baseline, a corrected conditioned reference, verifier and explicit
contrast. The observed execution reference must equal the relation baseline;
this prevents a correction statement from being attached to another turn.
An explicit `user_taught_concept` is a separate source: host grounding returns
two or more verified minimal contrasts with the same teaching key before the
ordinary residualized concept builder can form an experimental direction. The
separate user-confirmation source is intentionally not part of this phase.

The V0 procedure/blueprint producer is deliberately narrow. The host passes
resolved immutable references and semantic identity through a typed request:
procedure/blueprint reference, `behavior_key`, separate `teaching_key`, scope,
task fingerprint, baseline and conditioned references, verifier/evidence
references, and optional control reference. The producer returns a typed
admission result such as `no_contrast`, `not_reusable`, `out_of_scope` or
`resolved`; it does not parse a blueprint, infer a behavior key, or construct a
counterfactual contrast. A verified successful procedure is therefore not by
itself a FlyDelta teaching relation. A resolved relation requires an explicit
behavioral contrast with provenance preserved.

`task_fingerprint` identifies the concrete execution situation. `teaching_key`
identifies the reusable semantic relation across independent tasks and is the
future aggregation key. This distinction lets several procedure executions
contribute evidence to one concept without conflating their task identities.

The procedure smoke is intentionally model-free and contract-focused: it
checks host admission, explicit contrast, optional control requirements and
the absence of learning credit/promotion. It also exercises the resolved
relation through generic evidence materialization into the existing capture
candidate queue. The existing concept smoke remains the model-facing path.
Both feed the same `TeachingRelation` and existing transition/capture/search
contracts; the procedure smoke does not create a private model or persistence
path.

The relation may be `ambiguous`, `insufficient_evidence`, `no_contrast`,
`not_reusable` or `unsupported_behavior`. Such relations can remain in the
learning/evidence journal, but they must not create captures or FlyDelta
learning credit. Only a `resolved`, host-approved relation enters the existing
trajectory/capture path. This keeps source-specific interpretation in the
host while keeping layer search, dose control, margin, worker slices and
promotion source-neutral.

The host-side contrast record contains positive, negative and optional
control references, the one changed semantic dimension, declared invariant
dimensions, per-side provenance and host verification. `baseline` is what the
model did; `control` is a matched case where the principle must not fire.
They are not interchangeable. User text supplies evidence, the host supplies
semantic grounding, and verified contrasts supply the FlyDelta learning
signal.

### Semantic preparation before representation synthesis

Research, reflection and conversation may also propose a reusable principle,
but those inputs first enter the host-side semantic preparation seam as a
`ConceptHypothesis`. A hypothesis is not evidence and is not a direction. It
contains a bounded semantic kind, preconditions, optional state transitions,
invariants, counterexamples and source references. Facts remain eligible for
the knowledge/memory path; only a host-grounded procedure, decision rule or
concept may proceed toward FlyDelta teaching material.

The host then creates a grounded verifier/fixture family and checks minimal
contrasts. One verified contrast is enough to create and persist one
`TeachingRelation`; the material group remains incomplete until its configured
independent-relation threshold is reached:

```text
ConceptHypothesis
    -> host grounding
    -> baseline / conditioned / control fixtures
    -> semantic verification
    -> 0..N independent verified TeachingRelations
    -> existing TeachingMaterialStore admission
    -> concept_capture
    -> ConceptSynthesis
```

The model may propose a hypothesis or fixture variation, but it cannot certify
its own meaning. The host owns semantic normalization, verifier admission,
independence and provenance. The semantic preparation provider therefore
returns either no hypothesis/material (a normal outcome) or already verified
relations; it must never manufacture learning credit. A single relation may be
retained as durable teaching material, while the existing material-group policy
(normally two compatible and independent relations) controls eligibility for
capture and synthesis. No new FlyDelta job or material store is introduced by
this layer: `concept_capture`, `concept_synthesis`, grafting and ordinary
FlyDelta search remain the existing bounded pipeline.

#### V0 research/user-concept input routing

The first automatic input route reuses the completed research result and the
existing explicit `user_taught_concept_relation_provider`; it does not add
another research, evidence or candidate store. Research extraction is
fail-closed: only directly observed, supporting research evidence whose
statement is an explicit structured `concept_hypothesis` JSON envelope is
admitted. Free-form research prose, model-inferred evidence, reflection and
unresolved research remain outside FlyDelta grounding.

The portable envelope is intentionally small and source-reference based:

```json
{
  "kind": "concept_hypothesis",
  "concept_key": "dataset.grouped_sum",
  "statement": "Use grouped aggregation when totals are requested per region.",
  "semantic_kind": "decision_rule",
  "preconditions": ["the request asks for totals per group"]
}
```

The daemon installs the existing batch-provider seam for this route when no
custom hypothesis provider is supplied. It records the resulting hypothesis
in the existing candidate index as `proposed`; it never marks it grounded.
Explicit user-taught relations continue through their existing authoritative
provider and are not recreated by this route.

For a host-supported dataset/tool flow, the host supplies the existing
`semantic_concept_contrast_provider`. The shared adapter uses that callback to
obtain already verified baseline/conditioned/control contrasts, checks
verifier, changed-dimension and independent-key consistency, and converts them
through the existing `common_agent_concept_contrast_to_teaching_relation`
bridge. The runtime may now pass a `proposed` hypothesis to this host seam;
only a returned host-approved relation creates a local grounded copy for
admission. This closes the missing `proposed -> host grounding` transition
without changing FlyDelta algorithms, Oracle verdicts, learning-credit rules
or promotion policy.

The ownership remains:

```text
research result / explicit user relation
        -> existing transaction boundary
        -> existing candidate index (refs only)
        -> host contrast provider / existing Oracle seam
        -> TeachingRelation
        -> existing material, capture and ConceptSynthesis path
```

No new model call, Q-learning path or implicit A* runtime route is introduced
by V0. The bounded proposer can now be registered through the existing
host-owned blueprint materializer seam; ordinary runtime keeps the existing
planner path unless that callback is explicitly supplied and Oracle-gated.

### FlyDelta Oracle layer and bounded proposer

Oracle V0 separates semantic truth from representation geometry and lifecycle
authority. The shared contract has three evaluator strengths:

```text
deterministic
    exact host-independent semantic normalization and comparison
host_supported
    native registry/host execution and structural verification
model_supported
    model-assisted judgement behind an explicit host callback
```

All three return the same versioned result shape: `satisfied`, `violated`,
`not_applicable` or `unknown`, together with oracle revision, policy revision,
confidence, evidence reference and reason. The evaluator chain tries the
cheapest available authority first and only falls through when the previous
authority cannot establish a known result. An `unknown` result does not grant
learning credit, promotion or activation.

The first concrete family is `dataset_operation`. It reuses the existing
`SemanticDecision` IR and accepts both canonical semantic decisions and
model-shaped dataset tool calls. `normalized_call` fixtures in the resident
server-context host now use this oracle, so concept grounding and runtime
verification share one semantic definition. The host-supported smoke goes one
step further: it passes a canonical aggregate call through the existing native
tool registry and executes it against the host fixture. The smoke reports
host-supported verification only; it is not model-quality evidence.

The shared probe suite supports target, paraphrase, transfer, control and
competing probes and computes baseline/candidate success, intervention gain,
false-intervention rate, control retention and transfer gain. These metrics are
diagnostic input to later synthesis/search decisions; they do not replace the
existing counterfactual `HELPED` classifier. The suite runner is an explicit
agent/server-context callback, so the Oracle layer does not create a direct
single-turn model path.

The durable evaluation path now materializes a separate, redacted
`flydelta_oracle_suite_report` resource. It records bounded fixture/probe
identity, host-known/pass flags, counterfactual outcome and aggregate metrics,
along with `oracle_ref`, `oracle_revision` and `policy_revision`. The
`common_flydelta_evaluation_report` remains the lifecycle gate summary and
stores only the resource reference and provenance metadata; it does not gain a
second promotion or learning gate. Raw prompts, model outputs, captures and
activation authority remain in their existing seams.

A bounded generic A* proposer lives beside the shared Oracle contracts. It
operates only on opaque state fingerprints, successor costs, heuristic values,
and explicit expansion/goal callbacks. The Workflow/Blueprint adapter wraps
that result as a bounded proposal with host-canonical steps; it does not
evaluate or persist the proposal. That makes it reusable for deterministic
case construction, host-supported valid-combination search and later
model-supported probe proposal. Its expansion and path bounds are mandatory,
so it is a cheap proposer rather than a replacement for FlyDelta's layer,
scale, Whirlpool or evidence-depth algorithms. The current sweep establishes
this reusable support seam; ordinary FlyDelta search and promotion semantics
remain unchanged until a later, separately verified integration uses it.

Concept accumulation uses a thin reference-only candidate index backed by the
existing learning lifecycle journal. It stores canonical statement refs,
supporting/disconfirming evidence refs, relation refs and grounding/novelty
state; it does not copy turns, research text, trajectories or activations.
Support is counted from independent evidence/relation keys, not repeated
mentions. Research and reflection may create hypotheses, but do not strengthen
the candidate until a new host-verified contrast is observed. Conflicts are
retained as disconfirming evidence and block synthesis until the host resolves
the applicability boundary.

The model-facing decision boundary uses the same generic seam. A bounded
`decision_pair` contains two host-selected alternatives plus tokenizer and
template fingerprints; it is not limited to tool names. A host may implement
the `common_flydelta_decision_pair_provider` for a request carrying relation,
fixture, Oracle, behavior and model-identity references. The provider must
return a validated pair with a bounded first divergence; otherwise the
challenger is absent and the ordinary synthesis portfolio continues unchanged.
The host may then resolve the corresponding output-head rows through the
separate `common_flydelta_output_head_row_resolver`. The existing
`token_margin_direction` then computes:

```text
normalize(U[positive_token] - U[negative_token])
```

This is an output-space decision signal, not automatically an intermediate
layer injection. The daemon exposes the optional
`run_decision_margin_challenger` callback through the resident server-context
binding. It records returned experimental directions separately from the
ordinary concept frontier; it does not persist, graft, activate or promote
them by merely existing. A missing provider/resolver, missing localized layer,
ambiguous divergence or incompatible fingerprint fails closed without
disabling the existing search. The current default daemon binding leaves the
provider empty because the public server-context API does not expose generic
output-head weight rows; logits are not substituted for `U[t]`.

The existing `execution_boundary_prototype` remains the generic
positive-minus-negative capture direction, and coefficient search can combine
it with another direction without introducing a tool-specific candidate type.

Search objective, verification objective and learning authority are separate
contracts. A model-facing margin is search evidence only; it must be bound to
the same `behavior_key`, fixture/surface revision and host-selected decision
pair as the arm being evaluated. Implementations should retain the pair
identity, positive/negative references and a typed scope such as
`tool_choice`, `normalized_call`, `selected_arguments` or
`full_continuation`. A margin improvement may guide Whirlpool, BootstrapZoom,
UtilityGate, augmentation or coefficient search, but it cannot create
`HELPED`.

### Maintenance note — repair smoke helper boundaries

The model-backed repair smoke keeps its production execution, FlyDelta search,
frontier selection and lifecycle assertions in the smoke entry point, while
two smoke-local support modules own the surrounding test mechanics. The
verification helper parses model tool calls, invokes the existing host
normalizer/executor and classifies the existing counterfactual outcome. The
report helper only writes the existing margin and representation-diagnostic
fields, including both representation and scale geometry. These helpers do
not create a second evaluator, alter thresholds, or change evidence,
promotion, activation or runtime contracts. The refactor is covered by the
full serial FlyDelta CTest suite and the model-free repair/concept smokes;
model-backed results remain reported separately from deterministic contract
verification.

The decision-margin candidate descriptor additionally retains
`strategy_revision`, `decision_pair_ref`, `decision_score_scope` and
`decision_first_divergence_index`. These fields make the challenger auditable
and prevent a direction from being reused against a different model-facing
surface. They are provenance only; `HELPED` still comes only from the normal
host counterfactual path.

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
decision, alter a resident model or make FlyDelta active by itself. A
backend-specific prepared resource, if one is added later, is a resource
attachment below this seam; it must not move artifact lifecycle or learning
authority into activation preparation.

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
wrong statistics.describe
  -> host-verified repair to dataset.inspect
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
arms, while each arm receives a fresh server-context runtime context so the
cvec affects prefill consistently. This FlyDelta smoke has no CLI backend
override; the production-like server-context path is mandatory. The local
example keeps the thread limit at three:

```bash
LD_LIBRARY_PATH=build-agent-cozo/bin \
LLAMA_AGENT_MODEL=/path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
LLAMA_AGENT_THREADS=3 \
build-agent-cozo/bin/llama-agent-flydelta-runtime-smoke \
  --threads 3 --n-predict 16
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
host-controlled failed selection (statistics.describe)
  -> host-controlled repaired selection (dataset.inspect)
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
the model continued to emit `statistics.describe`; no candidate was selected. The
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

The two-phase search helpers use separate callbacks for these phases:

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

### 4N. Revisionsbar kandidataktivering och rollback

FlyDelta sideband-artefakter är immutabla revisioner. En kandidat kan därför
testas med riktig agent/server-context-körning utan att den aktiva runtime-
konfigurationen ändras. Oracle verifierar baseline, candidate och controls;
den befintliga parvisa counterfactual-klassificeringen producerar därefter
`HELPED`, `HARMED`, `NEUTRAL` eller `UNKNOWN`. `HELPED` gör kandidaten
eligible för fortsatt lifecycle-hantering, men aktiverar den inte automatiskt.

En modellprofil kan använda en logisk `binding_key` i stället för att peka på
en fysisk sideband-revision:

```text
model profile + FlyDelta scope
        ↓
durable activation binding
        ↓
immutable sideband revision
        ↓
existing registry compatibility/applicability checks
```

Binding-händelser lagras som typade poster i den befintliga append-only
lifecycle-journalen. Registryt bygger en in-memory-projektion genom replay;
ingen separat activation-store införs. En aktivering kräver canary-gate,
host approval, kompatibilitet och ett Oracle-/evaluation-resultat. Den tidigare
revisionen förblir tillgänglig och dess artifact ändras inte.

Rollback är en ny binding-händelse med samma profil- och scope-nyckel som
pekar på den tidigare giltiga revisionen. Den gör ingen ny syntes och skriver
inte om någon artifact. Journalens `expected_current_revision_id` används som
compare-and-swap-skydd mot samtidig ändring. Detta ger följande säkra flöde:

```text
experimental candidate
        → real counterfactual evaluation
        → canary
        → explicit activation of revision B
        → optional rollback to revision A
```

Direkta fysiska `sideband_id`-profiler är bakåtkompatibla. Binding-profiler
används för revisioner som ska kunna bytas och återställas utan profiländring.

### 4O. Begränsad canary ovanpå en aktiv binding

`canary` är inte en svagare form av `active`. Normal runtime-resolution
accepterar fortfarande endast den aktiva revision som bindingen valt. En
canary får endast resolvas genom en explicit, request-scopad
`canary_evaluation`-authority efter att hosten har valt ett öppet, durabelt
review-envelope.

```text
model profile sidebands[]
        ↓ stable default bindings
active deployment composition
        ↓ matching scope + deterministic cohort
temporary canary override
        ↓
candidate deployment composition
        ↓ host Oracle / counterfactual report
close canary | retain for evaluation | later explicit active promotion
```

Envelope:t lagras som en `stage_canary`-händelse i den befintliga
sideband-review-journalen, inte i den immutabla artifacten. Det anger
`binding_key`, candidate-revision, behavior/scope, traffic i basis points,
expiry, observationsbudget, max scale, modellkompatibilitet, Oracle/policy,
baseline-deployment och rollback-target. En `close_canary`-händelse tar bort
den temporära override:n utan att ändra active-bindingen eller artifacten.

Alla matchande turns bucketas deterministiskt från en host-owned stabil
session-/konversationsnyckel, canary-event-id och binding-kontext. Saknad
nyckel, fel scope, expiry, slut budget eller inkompatibilitet ger alltid
active-only fallback. Modellen kan inte välja cohort.

Counterfactual- och evaluationrapporter kan bära både baseline- och
candidate-deployment-fingerprint. De canonicaliseras från den ordnade
resolved overlay-kompositionen och dess effektiva scales/layout. Detta gör
att ett resultat kan uttrycka `A → A+B` eller `A → B` utan att felaktigt
tolkas som ett pristine-baseline-resultat. Oracle/policyrevision är separat
audit-provenance och ingår inte i cache-identiteten.

`IMPROVED` är en host-observerad admission-signal för ett uttryckligen
godkänt bounded experiment; den ändrar inte `HELPED`/`HARMED`, learning credit,
promotion eller active-kravet. En progress-only canary kan därför inte
aktiveras till `active` förrän en senare promotion-backed evaluation har
uppfyllt den vanliga aktiveringspolicyn.

#### Runtime-söm och deployment-factory

Den konkreta runtime-sömmen är host-owned och ligger före den befintliga
`make_base_turn_request`-sömmen:

```text
daemon/session-host turn request
        ↓
common_flydelta_resolve_deployment()
        ↓
profile sidebands + replayed review envelopes
        ↓
active-only eller explicit canary cohort
        ↓
host activation-loader (artifact store + gate + code)
        ↓
ordered deployment composition + fingerprints
        ↓
immutable request.flydelta_activation
        ↓
resident runtime / server-context generation
```

`flydelta-deployment.{h,cpp}` är en factory, inte en ny store eller runtime.
Den äger resolution, cohortbeslut, replacement/additive composition och
canonical deployment-identitet. Den läser inte artifactfiler och vet inte hur
ett sparse code skapas; det levereras av hostens befintliga loader-callback.
Loadern använder därefter de existerande artifact-, compatibility-, gate- och
activation-seamsen. Den färdiga activationen skickas genom den redan befintliga
`common_agent_request::flydelta_activation` till server-context-hostens cvec.

En återanvänd resident runtime uppdaterar activation-snapshoten per turn, men
byter inte modell, kö, orchestration eller lifecycle-state. Aktiv cvec tvingar
fortsatt fresh prompt/KV enligt den befintliga server-context-regeln. Den
effektiva ordnade overlay-listan, scales, layout och modellidentitet bildar
deployment-fingerprint; Oracle-, policy- och evaluationrevisioner är separat
provenance och får inte förändra cache-identiteten.

Composition är begränsad till profilens befintliga högst fyra `sidebands[]` i
stabil ordning. En canary på samma `binding_key` ersätter bara den aktiva
revisionen i den slotten. En canary på en annan redan konfigurerad binding
resulterar i `[A, B]` och jämförs mot `[A]`. Om allocation key saknas,
envelope-baseline/rollback inte stämmer, budgeten är slut eller någon
kompatibilitetskontroll faller, returnerar factoryn active-only och lämnar
active-bindingen orörd.

#### Canary-disposition, expansion och rollback

Canary-konfigurationen ligger under `runtime.adaptation.flydelta.canary`.
`mode` är `disabled`, `manual` eller `policy`; default när FlyDelta är
aktiverat är `policy`. `disabled` stänger routing helt, `manual` kräver en
explicit host/admin-authority och `policy` gör den hostägda, journalförda
dispositionsfunktionen tillgänglig för policykörning. `policy` betyder inte
att en experimentell revision automatiskt blir active: `allow_promotion`
måste dessutom vara aktiverad och den färdiga evidensen måste passera alla
trösklar. Vanlig inference har fortfarande alltid `active_only` som default.

En policy läser endast färdiga semantic/counterfactual-observationer och
returnerar en av:

```text
retain        -> samma envelope och cohort fortsätter
expand_scope  -> nästa förkonfigurerade traffic/scope-steg
promote_active -> explicit activate + CAS mot aktuell binding
close         -> journalförd close och omedelbar active-fallback
```

`expand_scope` får inte skapa ett godtyckligt scope; `scope_step_fingerprints`
är den hostkonfigurerade tillåtna listan. Observation-reservationen före
generation är bara en atomisk exponeringsbudget. Den räknas inte som
`HELPED`, semantic evidence eller promotion evidence. Reservationerna skrivs
som journalhändelser i samma review-journal och seedas tillbaka vid omstart;
endast färdiga verifierade resultat får driva policybeslut.

Evaluation-lineage är explicit. En reservation skapar ett bounded
`observation_id` och en immutable canary-evaluation-context med event-id,
binding/candidate, allocation, scope-steg, baseline/candidate-deployment och
Oracle-revision. Runtime trace och evaluation-/counterfactual-rapporter kan
föra samma context vidare. En observation avslutas genom journalhändelsen
`complete_canary_observation` med terminalstatus (`generation_failed`,
`cancelled`, `oracle_unknown` eller `evaluated`); en verifierad rapport kopplas
sedan genom `attach_canary_evaluation`.

Policy-snapshoten projiceras vid replay från dessa händelser och innehåller
reserverade, terminala och utvärderade observationer, unika allocations,
HELPED/NEUTRAL/UNKNOWN/HARMED-utfall samt gain- och regressionsmått. Endast
utvärderade observationer räknas som `completed_observations` i
`retain`/`expand_scope`/`promote_active`. En observation får högst en terminal
completion och en evaluation attachment; retry med samma id är idempotent och
en konkurrerande attachment avvisas.

`HARMED` stänger det berörda envelope:t/den berörda strategy-revisionen och
underkänner inte automatiskt konceptet, andra strategier eller andra active
bindings. Active rollback är CAS-skyddad: den får bara byta tillbaka om den
revision som beslutet avser fortfarande är vald. Replay av samma review-journal
återskapar därför samma open/closed canary-state efter omstart.

Session-hosten bygger deployment-requesten från modellprofil, applicability,
modellidentitet, layout, host-owned allocation key, authority, gate och sparse
code. Factoryn resolverar en gång före turnens runtime-start. Om en vanlig
daemon-turn saknar explicit host-owned applicability använder factoryn den
redan konfigurerade applicabilityn från den matchande aktiva bindingen för
canary-matchningen. Det är en profilbunden host-seam, inte en gissning från
modellens output eller klientens JSONL; saknas applicability även i bindingen
faller resolutionen active-only. Den effektiva
deployment-fingerprinten ingår i session/KV-identiteten, medan Oracle-, policy-
och evaluationrevisioner endast är provenance. Samma resolution används genom
planner, tool execution, repair, reflection och final generation.

En policy- eller admin-klient kan stänga en öppen canary explicit med
`flydelta.close_canary` och host approval. Det är en journalförd close med
omedelbar active-fallback; den muterar inte active-bindingen. `expand_scope`
får endast välja det omedelbart nästa värdet i `scope_step_fingerprints`, inte
ett senare eller modellföreslaget scope.

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
3. Registry shape: the current implementation remains sideband-specific; a
   generic overlay registry is outside this change.
4. Exact redaction, scope, TTL and revocation semantics for capture manifests.

The binding path remains host-gated: documentation, model-free contract tests
and explicit model-backed evaluation remain required before a candidate can be
activated. Policy mode may automatically create only a bounded, progress-only
canary through the review journal; it never grants learning credit or active
activation by itself.

### Maintenance note — automatic project reuse

The project-reuse and first-canary implementation keeps session candidates,
review journal, registry, configured profile bindings and the existing Oracle /
promotion gates as the only sources of truth. It adds no parallel store,
runtime or evaluator. The model can be a source of a hypothesis, but host
compatibility and host-verified progress are required before generalized
provenance or canary staging.

### Maintenance note — negative contrast support

Negative examples are optional, explicitly host-verified contrast material in
the existing trajectory/prototype seam. They are not a second evidence store
and do not alter the production Oracle, learning-credit or promotion rules.
The daemon loads them only when all trajectories in a compatible group carry
the negative reference and capture; otherwise the group fails closed. The
negative builder reuses the positive prototype estimator and emits
`control - negative`. A negative-only group is retained for inspection and
cannot create a graft frontier or learning evidence. When a compatible
prefer/repair or positive-prototype candidate is available, the host keeps the
roles explicit in a paired intervention proposal:

```text
prefer / repair or positive prototype
avoid_support / negative repulsion
        -> existing rank-2 basis builder
        -> existing coefficient/TFO search
        -> the same diagnostic, full-generation and host-Oracle path
```

`avoid_support` is therefore not a rank-one frontier candidate and can never
receive independent `HELPED`, learning credit or promotion. The paired basis
is persisted, when the host elects to persist it, as one ordinary composed
experimental artifact through the existing daemon binding seam. Its proposal
retains the two immutable candidate references, roles, fixture/Oracle identity,
model/capture identity and layer. The role is not reconstructed from a generic
direction later, so a negative component cannot silently become a normal
positive arm. Coefficients may still discover that the avoid component is
useful, but only a complete host-verified pair outcome can be `HELPED`.
