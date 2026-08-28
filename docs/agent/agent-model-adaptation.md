# Agent Model Adaptation

## Status

Design and implementation plan. Corpus contracts exist, but no external
trainer integration, adapter registry, model catalog, or automatic adapter
activation exists yet.

The purpose of this work is not to make the model self-authorizing or to move
runtime reasoning into model weights. It introduces a host-supervised path for
turn evidence to become a reproducible, evaluated adapter candidate when that
is demonstrably more appropriate than memory, a procedure, or a blueprint.

The preferred name is **model adaptation**. The host owns every promotion and
activation decision.

### Current implementation status

The local branch currently contains the first contract slices:

- host-owned learning observations with signal, scope, verification, and
  idempotency fields;
- an optional runtime observer seam that is invoked for qualifying signals,
  including failed turns, without affecting the user-facing result;
- append-only in-memory and JSONL transaction stores with duplicate suppression;
- destination qualification and a bounded training-candidate contract;
- a deterministic JSONL corpus builder requiring explicitly approved
  candidates;
- an external training job/result contract and a validated adapter registry
  with explicit candidate/active/retired/rejected transitions.

The observation identity hash and corpus bundle hash are currently stable
non-cryptographic identity hashes used for local deduplication and test
reproducibility. Resource and artifact integrity must continue to use the
existing SHA-256 stores. Scope-aware bootstrap configuration, redaction and
revocation propagation, semantic replay, an actual trainer worker, adapter
artifact import, model-profile loading, and inference activation remain later
sweeps. The current registry is metadata-only and does not load an adapter.

## Review conclusion and recommended adjustments

The proposed direction is sound, provided that adaptation is treated as a
second, offline lifecycle rather than as another runtime learning mode. The
important boundary is:

```text
resident agent runtime  ->  evidence and bounded learning transactions
external worker         ->  corpus build and training
adapter registry        ->  validation, evaluation, activation, rollback
```

I recommend the following adjustments before implementation:

1. A normal successful turn must not automatically become a training record.
   Create a transaction only when a qualifying host signal exists, such as a
   failure, a verified recovery, or an explicit correction. Ordinary success
   may still contribute aggregate metrics, but must not silently grow a corpus.
2. `successful_recovery` is evidence that a repaired path completed. It is not
   by itself proof that the original model behavior was wrong, general, or
   safe to teach. The candidate gate must require host verification and must
   distinguish a model defect from a contract, policy, metadata, or missing
   evidence defect.
3. Keep one source of truth for runtime evidence. Existing plan observations,
   memory, procedures, and resources remain evidence sources; adaptation
   transactions and candidates are immutable derived records. Do not copy all
   tool output or reflection text into a second durable store.
4. Make deterministic qualification automatic but make corpus inclusion and
   adapter activation explicit at first. This preserves throughput without
   allowing an ambiguous correction to change model behavior silently.
5. Start with one narrow behavior and SFT. Tool-contract compliance on
   redacted fixture data is a better first target than free-form planning or
   project-specific facts. DPO requires same-context chosen/rejected pairs,
   which a recovery often does not provide.
6. Start with one active adapter and an adapter-free baseline profile. A full
   multi-model catalog is useful, but it should not block the first complete
   transaction-to-adapter path.

These adjustments keep the feature small enough to verify and prevent a
second, competing memory system from emerging beside the existing learning
and resource flows.

## Integration with the existing learning path

The existing learning path remains the first consumer of turn evidence. It
already receives host-produced `learning_signals`, extracts bounded memory
candidates, validates provenance and scope, stores accepted memory or
procedures, and promotes verified procedures to blueprints. Model adaptation
must be joined at the observation boundary, not by reusing the memory
candidate schema or by creating a second interpretation of the turn.

The shared lifecycle is:

```text
runtime completes or fails a turn
             |
             v
     host learning observation
        /                \
       v                  v
memory learner      adaptation observer
       |                  |
memory/procedure    transaction ledger
       |                  |
blueprint           qualification/corpus
```

The memory learner may continue to require a completed plan and final
response. The adaptation observer must also be able to record a qualifying
failure, because a failed turn can be important evidence even when no final
answer exists. It must never make the runtime turn fail, train in the session
lane, or change an active model.

### Shared observation, separate destinations

The existing `common_learning_signal` is the beginning of the shared
observation contract. The adaptation seam should reuse its plan, step, tool,
and evidence identity, while deriving namespace/project/session scope from
the host request. It should add host-owned provenance and verification rather
than asking the model to certify itself:

- relation between a failure and a later recovery;
- host classification of the suspected cause: model behavior, host contract,
  policy, missing evidence, or project knowledge;
- verification status and idempotency key;
- bounded source references and content hashes.

`common_memory_candidate` must not become the training-candidate contract.
Memory candidates describe durable scoped knowledge; training candidates must
describe an approved, redacted, potentially generalizable behavior with an
explicit input and target. The two paths may share scope, hashing, and
provenance helpers, but not their promotion policy.

The current runtime derives `successful_recovery` late, after a response has
been accepted. The future finalization seam must make that signal visible to
both consumers. It should also invoke adaptation for an error result when a
qualifying signal exists, while retaining the current rule that ordinary
successful turns do not become training data.

## Current foundations

The runtime already records several useful signals as bounded plan
observations and exposes them in the terminal result:

- `tool_failure`;
- `successful_recovery`;
- `reflection_hint`;
- `user_correction`, including the caller-supplied source turn id.

The post-turn memory learner already validates candidate shape and evidence,
applies confidence and expected-reuse thresholds, detects duplicates and
conflicts, and can promote a verified procedure to a blueprint. This is the
right conceptual precedent, but it is not a model-training system.

The current configuration has one generation model (`model.path`) and an
optional separate embedding model (`model.embedding_model`). The current
resident model-load key includes the generation-model and multimodal-projector
identity, but not an adapter set or a named model profile.

## Why a separate adaptation lifecycle is needed

Most corrections should remain outside the weights:

| Observed outcome | Preferred durable destination |
| --- | --- |
| User/project preference or current fact | scoped memory |
| Reusable ordered work pattern | procedure, then blueprint when verified |
| Missing tool metadata or a host-policy problem | repair the host contract or policy |
| Missing resource/evidence | resource retrieval or research |
| Repeated, general model behavior defect | training candidate |

Fine-tuning is therefore an escalation path, not the default learning action.
For example, a project-specific naming preference belongs in memory, while a
repeated inability to emit a valid compact tool binding may be an adaptation
candidate after host-side contract repairs have been ruled out.

## Target architecture

```text
turn / plan / tool execution
             |
             v
      learning transaction               append-only, host-owned evidence
             |
             v
       qualification policy
       /        |          \
  memory   procedure      training candidate
  or       / blueprint            |
                                curator
                                  |
                                  v
                         corpus revision + manifest
                                  |
                                  v
                    local or remote external trainer
                                  |
                                  v
                      candidate adapter + manifest
                                  |
                                  v
               offline evaluation -> canary -> active / retired
                                  |
                                  v
                     model profile used by runtime inference
```

The runtime must never run gradient training in a turn or make model-weight
changes in-place. A trainer is an external worker, whether it runs as a local
process or in another trusted instance.

## Core concepts and contracts

### Learning transaction

A learning transaction is an immutable, append-only record of host-observed
experience. It is not automatically training data. It references source data
by identity and hash, preserving the facts needed to later audit why a case
was considered.

The capture rule is deliberately sparse: no transaction is created for an
ordinary successful turn unless a configured sampling policy explicitly asks
for a metric-only observation. A qualifying transaction must identify the
signal, its bounded evidence, and the scope that authorized collection. A
duplicate signal for the same source turn and evidence hashes must be
idempotent.

Conceptual shape:

```json
{
  "schema_version": 1,
  "id": "learning://transaction/01J...",
  "created_at": "2026-08-29T12:00:00Z",
  "source": {
    "namespace_id": "local",
    "project_id": "default",
    "session_id": "web-123",
    "turn_id": "turn-9812",
    "plan_id": "plan-42"
  },
  "signals": ["tool_failure", "successful_recovery"],
  "evidence_ids": ["tool:step_2:failure", "tool:step_3:result"],
  "runtime_fingerprint": {
    "agent_commit": "...",
    "tool_catalog_hash": "sha256:...",
    "policy_hash": "sha256:...",
    "model_profile": "agent-default"
  },
  "content_policy": "redacted",
  "payload_hash": "sha256:..."
}
```

The transaction may retain host-local references to full material when policy
allows it. Its portable representation must contain only explicitly admitted,
redacted fields. Secrets, credentials, raw sensitive resources, opaque tool
payloads, and internal scratch/reasoning text are excluded by default.

### Training candidate

A training candidate is a curated derivation from one or more transactions. It
has a stated behavioral hypothesis, qualification evidence, and an approved
model-facing target. It is distinct from its raw turn history.

```json
{
  "schema_version": 1,
  "id": "learning://candidate/01J...",
  "transaction_ids": ["learning://transaction/01J..."],
  "scope": "agent_behavior",
  "hypothesis": "The model repeatedly emits invalid tool arguments for a stable contract.",
  "training_kind": "sft",
  "input": {"approved_prompt": "...", "contract_context": "..."},
  "target": {"approved_action_or_response": "..."},
  "qualification": {
    "observed_occurrences": 18,
    "verified_recoveries": 17,
    "contradictions": 0,
    "confidence": 0.91
  },
  "status": "approved_for_corpus"
}
```

The first implementation should create supervised fine-tuning (SFT) candidates
only. A successful recovery is not automatically a valid DPO pair: the repair
often happens after new tool evidence has changed the context. DPO should be
added only when rejected and chosen outputs were generated against the same
bounded input contract. KTO is later work, not a first dependency.

### Corpus revision

A corpus revision is an immutable selection of approved candidates plus a
manifest. The dataset builder is deterministic: the same candidate ids,
redaction policy, ordering seed, and builder version must produce the same
bundle hash.

The manifest records candidate ids, splits, replay selection, base-model
requirements, hashes, and the evaluation set excluded from training. A held-out
evaluation case must never also be selected as replay material.

### Adapter revision

An adapter revision is an artifact plus a manifest. The manifest must bind it
to a compatible base model and describe how it was produced:

```json
{
  "schema_version": 1,
  "id": "agent-adaptation-v1",
  "status": "candidate",
  "base": {
    "architecture": "qwen2",
    "model_fingerprint": "sha256:...",
    "tokenizer_fingerprint": "sha256:...",
    "chat_template_fingerprint": "sha256:..."
  },
  "corpus_revision": "learning-corpus@12",
  "trainer": {"kind": "qlora-sft", "seed": 42, "code_revision": "..."},
  "artifact": {"path": "adapters/agent-adaptation-v1.gguf", "sha256": "..."},
  "evaluation": {"suite_revision": "adapter-eval@3", "status": "passed"}
}
```

An adapter is never active merely because a trainer produced a file. The
registry admits only a fully validated manifest and promotion result.

## Data boundaries and privacy

Training data must be materially more restrictive than normal local runtime
state. The initial policy should be fail-closed:

- collection is disabled by default;
- each namespace/project chooses whether adaptation collection is allowed;
- raw user resources and external API responses are excluded by default;
- credentials, authorization headers, filesystem paths outside approved
  project scope, and tool diagnostics are redacted or represented by typed
  categories;
- model scratch/reasoning and reflection prose are not corpus material;
- remote export requires an explicit portable-bundle policy, not just a remote
  trainer URL;
- deleting or revoking a source scope marks dependent candidates and corpus
  revisions unusable for future training.

Local-only deployments may deliberately opt into broader retention, but this
must be visible in configuration and manifests rather than being implied by a
local file path.

## Local and remote trainer topologies

Both topologies share the same corpus and adapter contracts.

### Local worker

The daemon writes an approved immutable corpus bundle to a bounded work queue.
A separate worker process consumes it. It has independent CPU/GPU, disk,
deadline, and cancellation limits. The inference daemon does not share its
active inference context with the trainer and does not run training work on its
session lanes.

This is the preferred initial topology for private development use.

### Remote worker

The daemon exports a portable bundle; the remote worker returns an adapter
artifact, manifest, and evaluation report. Initial support should be an
explicit export/import command or MCP-like job protocol, not an always-on
implicit uploader. The receiver is trusted only for bundles allowed by the
source scope's export policy.

Remote execution does not weaken the local promotion gate: the local registry
still verifies hashes, base compatibility, evaluation provenance, and policy
before a returned adapter becomes selectable.

## Persistence boundary and deduplication

The system should not introduce a second unstructured copy of the runtime's
knowledge. The recommended split is:

| Layer | Responsibility | Durable content |
| --- | --- | --- |
| Existing runtime stores | plan observations, memories, procedures, resources | source evidence under existing scope rules |
| Adaptation ledger/index | immutable transactions and candidate state | ids, scope, hashes, signal classification, provenance, status |
| Corpus/artifact store | portable revisions and model artifacts | canonical JSONL, manifests, adapters, evaluation reports |

The adaptation interface should follow the existing scoped-store conventions.
The first backend may use a local append-only representation, but the contract
must leave room for an indexed backend so filtering by scope, status, source,
and revision does not require scanning every corpus file. Payloads should be
referenced by content hash where possible; the ledger should not duplicate
large resource or tool-result bodies merely to make export convenient.

Every write needs an idempotency key, atomic completion semantics, and a
content hash. A crash may leave a pending write, but it must not create two
different meanings for the same source event. Revocation must invalidate
derived candidates and corpus revisions without rewriting historical audit
records.

## Inference and model profiles

The current runtime supports one generation model per host configuration. The
target design introduces named model profiles. A profile selects one base
model, zero or more approved adapters, and runtime limits. It is the unit a
session uses, rather than a free model path or adapter path supplied by a
caller.

Conceptual future configuration:

```json
{
  "schema_version": 2,
  "models": {
    "bases": {
      "qwen-small": {
        "kind": "generation",
        "backend": "server-context",
        "path": "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "load": "resident"
      },
      "qwen-research": {
        "kind": "generation",
        "backend": "server-context",
        "path": "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "load": "lazy"
      },
      "nomic": {
        "kind": "embedding",
        "backend": "server-context",
        "path": "models/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "load": "resident"
      }
    },
    "profiles": {
      "agent-default": {
        "base": "qwen-small",
        "adapters": ["agent-adaptation-v1"],
        "context_size": 4096,
        "load": "resident"
      },
      "agent-baseline": {
        "base": "qwen-small",
        "adapters": [],
        "context_size": 4096,
        "load": "lazy"
      },
      "research": {
        "base": "qwen-research",
        "adapters": [],
        "context_size": 8192,
        "load": "lazy"
      }
    }
  },
  "routing": {
    "default_profile": "agent-default",
    "embedding_model": "nomic",
    "thinking_modes": {"research": "research"}
  },
  "limits": {
    "max_loaded_generation_models": 2,
    "inference_max_active": 1,
    "model_eviction": "lru"
  }
}
```

This is a proposed version-2 schema, not a supported configuration today. It
intentionally separates base-model identity, adapter identity, runnable
profile, and host-owned routing. The adapter registry owns artifact paths and
validation; production profile configuration should reference adapter ids, not
arbitrary paths.

A session is pinned to a profile for the duration of a turn. The host may
perform a bounded, visible escalation to another profile, but a model cannot
silently choose a larger model or activate an adapter. The resident load key
must include the base-model fingerprint, multimodal projector, adapter ids and
scales, tokenizer/chat-template fingerprint, and relevant context parameters.
Changing an adapter set must not reuse a KV cache or resident state built for a
different profile.

The first adaptation deployment should use one base model plus one active
adapter and an adapter-free baseline profile. Multiple adapters and dynamic
per-turn switching add cache, scheduling, memory, and evaluation complexity;
they are follow-up work.

## Training and deployment constraints

QLoRA/LoRA training normally uses a compatible training checkpoint and trainer
toolchain, then emits an adapter for inference. It should not be treated as
direct training of the deployment Q4 GGUF file. The serving runtime uses the
quantized GGUF base plus a compatible adapter overlay.

The repository already contains LoRA inference and export support. Merging is
not the initial deployment path: overlays make rollback and A/B evaluation
simple, while the current merge tool requires f16/f32 adapter tensors. A later
consolidation workflow may produce a new base generation only after explicit
evaluation and release policy approval.

## Evaluation and promotion

Loss reduction is insufficient evidence. Candidate adapters require at least:

- held-out correction cases for the intended behavior;
- replay/retention evaluation against earlier accepted cases;
- agent contract tests for JSON/tool bindings and policy boundaries;
- planning and tool-execution regression tests;
- comparison with the adapter-free baseline under the same profile and test
  fixture;
- recorded runtime-intervention metrics.

Useful runtime-oriented measures include completed turns, tool-contract
violations, failed tool selections, plan revisions, reflection repairs, and
the number of turns requiring host recovery. The latter can be reported as
**runtime intervention rate** or **correction pressure**. These measures do
not replace quality evaluation; they show whether the runtime must compensate
for the model less often.

Promotion sequence:

```text
candidate adapter
  -> offline evaluation passed
  -> canary profile
  -> bounded comparison with baseline
  -> active profile
  -> retired (reversible)
```

Automatic promotion to active is out of scope for the first implementation.

## Implementation plan and sweep gates

Each sweep ends with a local commit and contract tests. A sweep is complete
only when its exit gate passes; a later sweep must not hide a failed contract
in an end-to-end smoke. The first usable path is deliberately staged as
evidence -> candidate -> corpus -> external training -> evaluation -> manual
activation.

### Sweep 0 — contract fixtures and target definition

Deliverables:

- Choose one first behavior target, preferably valid tool-call structure and
  argument binding in redacted fixture contexts.
- Define canonical serialization, content-hash rules, scope inheritance, and
  the difference between source evidence, a learning transaction, a candidate,
  and a corpus row.
- Add minimal fixtures for a failure, a verified recovery, an explicit user
  correction, a host-contract defect, and an ordinary successful turn.

Tests and exit gate:

- model-free CTests validate schema shape, enum values, canonical hashes,
  scope checks, bounded payloads, and the rule that ordinary success creates no
  training transaction;
- fixtures show that the same source event is idempotent and that a resource
  body is referenced rather than copied by default.

This sweep prevents the later implementation from guessing what “learning”
means. It is documentation and contracts only; there is no export or model
loading.

### Sweep 1 — durable learning transactions

Deliverables:

- Add host-neutral contracts under `common/agent/adaptation/` for transactions,
  source references, payload classes, hashes, and collection policy.
- Add an append-only store interface and a local backend following existing
  scoped-store conventions.
- Capture bounded records from existing runtime signals only when the
  qualification trigger is present; preserve plan/tool provenance without
  persisting raw scratch or unrestricted tool output.
- Make writes idempotent and recoverable after interruption.

Tests and exit gate:

- CTests cover namespace/project/session isolation, append-only behavior,
  duplicate suppression, provenance, hash stability, size limits, revoked
  scope, and partial-write recovery;
- an integration fixture proves that a failed turn, a verified recovery, and a
  user correction produce distinct signal records;
- an ordinary `hi there`-style successful turn contributes metrics only.

No training data is exported in this sweep.

The runtime integration deliberately preserves the existing constructor ABI.
Adaptation observation is attached through an optional observer setter, so
existing hosts and shared-library clients do not need a new constructor
symbol. The observer is finalized after both successful and failed turns and
must never change the user-facing result. Idempotency covers the complete
signal/evidence identity for a turn; two distinct failures in the same turn
must not collapse into one transaction.

### Sweep 2 — qualification policy and training candidates

Deliverables:

- Add an explicit destination classifier: retain, memory,
  procedure/blueprint, training candidate, or reject.
- Add the defect classification seam: model behavior, host contract/policy,
  missing evidence, or project-specific knowledge.
- Require repeated compatible observations, host-verified recovery or
  user-approved correction, no unresolved contradiction, and an approved target
  shape. Deterministic rules may mark a candidate eligible; corpus inclusion
  remains explicit initially.
- Add a curator/approval seam without allowing the model to self-certify.

Tests and exit gate:

- CTests cover occurrence thresholds, contradictory examples, cross-context
  similarity, user-correction provenance, and approval transitions;
- project facts, raw resources, credentials, and a bad tool description are
  rejected as general behavior candidates or routed to the correct destination;
- a successful recovery with no host verification cannot qualify by itself.

The key tradeoff is automation versus safety. Automatic eligibility keeps the
ledger useful, while explicit corpus approval prevents a single ambiguous
correction from changing future model behavior.

### Sweep 3 — deterministic corpus builder

Deliverables:

- Build versioned JSONL bundles and manifests from approved candidates.
- Implement redaction, revocation propagation, deduplication, stable ordering,
  train/validation/test split assignment, and bounded semantic replay selection.
- Keep held-out evaluation cases out of both training and replay selection.
- Add a local inspection/export command; remote export remains opt-in.

Tests and exit gate:

- the same candidate ids, policy, builder version, and seed produce a byte-for-
  byte identical bundle and manifest;
- tests verify no secret/path leakage, no revoked candidate, no split overlap,
  stable deduplication, row/byte bounds, and correct provenance;
- a fixture can reconstruct the exact corpus revision from its manifest.

The principal choice is JSONL versus an indexed database. JSONL is portable
and reproducible; an index is better for scoped queries. The recommended
boundary is an indexed adaptation ledger plus canonical JSONL artifact bundles,
rather than two independent sources of truth.

### Sweep 4 — external trainer protocol

Deliverables:

- Define local-worker job files first, then optional remote export/import using
  the same corpus and result manifests.
- Keep Python PEFT/TRL or another trainer outside the daemon and outside agent
  session lanes. The first worker may be a deterministic fake trainer used to
  validate the protocol.
- Require returned adapter artifact, manifest, hash, trainer version, base
  training fingerprint, conversion information, and evaluation report.

Tests and exit gate:

- malformed, incomplete, duplicate, expired, mismatched-base, and
  unauthorized results are rejected;
- a local worker smoke produces a reproducible fake result without CUDA;
- import never changes an active profile automatically.

Training must bind to a compatible training checkpoint, not silently train the
deployment Q4 GGUF in place. The conversion manifest must connect the training
checkpoint to the serving base.

### Sweep 5 — adapter registry and single-profile inference

Deliverables:

- Add adapter registry validation and lifecycle states: candidate, active,
  retired, rejected.
- Extend model-load identity, status/readiness, and traces with profile and
  adapter revision.
- Support one explicitly configured active adapter and an adapter-free baseline
  against the same base model. Partition or clear KV/resident state when the
  profile identity changes.
- Pin a session to its profile for a turn and make activation/rollback host
  operations.

Tests and exit gate:

- CTests cover manifest compatibility, load/rollback, cache isolation,
  incompatible adapter rejection, session pinning, and readiness failure;
- an inference stub proves baseline and adapter profiles are distinguishable;
- a real Qwen smoke is added only after the contract tests pass.

The first deployment should use an overlay, not merge weights into a new base:
unloading the adapter provides a simple rollback and A/B comparison.

### Sweep 6 — evaluation, replay, and canary promotion

Deliverables:

- Add held-out behavior cases, replay/retention cases, existing agent contract
  tests, and planning/tool regression coverage.
- Compare candidate against the adapter-free baseline with the same tool
  catalog, resource snapshot, profile limits, and reproducibility settings.
- Record runtime intervention rate: contract violations, failed selections,
  plan revisions, reflection repairs, and host recoveries.
- Add manual canary activation and reversible rollback. Automatic promotion is
  explicitly out of scope.

Tests and exit gate:

- candidate must pass intended-behavior, old-learning retention, and agent
  regression gates;
- evaluation metadata binds to corpus, base, adapter, and test revisions;
- rollback returns to baseline without changing the base model.

Only after this gate should DPO be considered, and only for genuinely
same-context chosen/rejected pairs. Broader continual-learning strategies are
later work.

### Sweep 7 — model catalog and controlled routing

Deliverables:

- Introduce named bases and profiles only when the single-profile path is
  proven. Migrate the current single-model configuration once; do not retain
  two long-lived configuration semantics.
- Add bounded resident/lazy loading, profile-aware scheduler admission, and
  explicit host-controlled escalation between a small model, baseline, and
  research model.
- Keep the embedding model separate from generation profiles.

Tests and exit gate:

- configuration tests cover named profiles, adapter references, memory limits,
  lazy loading, eviction, and refusal of caller-supplied model overrides;
- a scheduler fixture proves profile pinning and no cache reuse across model or
  adapter identity changes;
- small-model operation remains valid when no adaptation profile is active.

This ordering is intentional: multi-model routing is valuable for production,
but it is not required to prove the adaptation lifecycle and adds a substantial
memory/scheduling surface.

## Initial non-goals

- training inside the daemon or on a live inference session;
- automatic weight mutation or automatic active-adapter promotion;
- collecting all user prompts, resources, tool payloads, or reflection text;
- multi-adapter composition or arbitrary per-turn adapter switching;
- merging adapters into a new base model;
- distributed/federated training;
- DPO, KTO, reward modeling, or reinforcement learning in the first slice.

## Decisions to make before implementation

1. Which scopes may opt into collection: installation, namespace, project,
   session, or user?
2. Does the first candidate promotion require manual approval, or may a strict
   deterministic policy approve corpus inclusion locally?
3. Which first behavior is narrow enough to train safely: compact tool
   bindings, tool-family selection, structured planning, or another target?
4. Which base checkpoint and conversion path will be the supported first
   QLoRA/LoRA training target?
5. Should remote training initially be export/import only, or is a trusted job
   service needed in the first release?
6. What retention and revocation guarantees are required for locally retained
   raw evidence and exported corpus bundles?

## Completion criteria for the first usable release

The first usable adaptation release is complete when an operator can opt in a
project, inspect why a transaction became a candidate, reproduce an immutable
redacted corpus bundle, run an external trainer, import a compatible adapter
with an evaluation report, compare it against an adapter-free baseline, and
activate or roll back the adapter without changing the base GGUF model.
