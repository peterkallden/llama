# Agent Model Adaptation

## Status

Design and implementation plan. No training corpus, trainer integration,
adapter registry, model catalog, or automatic adapter activation exists yet.

The purpose of this work is not to make the model self-authorizing or to move
runtime reasoning into model weights. It introduces a host-supervised path for
turn evidence to become a reproducible, evaluated adapter candidate when that
is demonstrably more appropriate than memory, a procedure, or a blueprint.

The preferred name is **model adaptation**. The host owns every promotion and
activation decision.

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

## Implementation plan

### Sweep 1 — durable learning transactions

- Add host-neutral contracts under `common/agent/adaptation/` for transactions,
  payload classification, hashes, and source references.
- Add an append-only store interface and a local backend following the existing
  scoped-store conventions.
- Record bounded transaction candidates from the existing runtime signals and
  completed-plan evidence.
- Add contract tests for scope isolation, append-only semantics, hashes,
  source provenance, and rejection of oversized/unclassified payloads.

No training data is exported in this sweep.

### Sweep 2 — qualification policy and training candidates

- Add an explicit destination policy: retain, memory, procedure/blueprint,
  training candidate, or reject.
- Require repeated compatible observations, verified recovery or user-approved
  correction, no unresolved contradiction, and a host-approved target shape.
- Add a curator seam; start with deterministic host rules and optional manual
  approval rather than model self-certification.
- Test that project-specific facts and raw resources cannot be promoted as
  general behavior candidates by default.

### Sweep 3 — deterministic corpus builder

- Build versioned JSONL bundles and manifests from approved candidates.
- Implement redaction, deletion/revocation propagation, deduplication, split
  assignment, and bounded semantic replay selection.
- Add a CLI/admin export command for local inspection only.
- Test deterministic reconstruction and held-out/replay separation.

### Sweep 4 — external trainer protocol

- Define local-worker job files first, then an optional remote export/import
  protocol using the same bundle and result manifests.
- Keep trainer implementation outside the daemon runtime; Python PEFT/TRL or
  another compatible worker is an implementation detail of this boundary.
- Require a returned adapter manifest, artifact hash, trainer version, base
  compatibility data, and evaluation report.
- Test malformed, mismatched, and incomplete results are rejected.

### Sweep 5 — adapter registry and single-profile inference

- Add adapter registry validation and lifecycle states: candidate, active,
  retired, rejected.
- Extend model-load identity and status/readiness/traces with profile and
  adapter revision.
- Support one configured active adapter plus an adapter-free baseline against
  the same base model. Clear or partition caches when profile identity changes.
- Add lifecycle tests for load, rollback, incompatible adapter rejection, and
  session profile pinning.

### Sweep 6 — model catalog and controlled routing

- Introduce schema version 2 with named base models and profiles.
- Migrate the existing single-model configuration once; do not retain two
  long-lived parallel configuration semantics.
- Add bounded resident/lazy load policy, profile-aware scheduler admission,
  and explicit host-controlled profile escalation.
- Test configuration validation, memory limits, lazy loading, and refusal of
  unapproved caller/model routing overrides.

### Sweep 7 — evaluation, replay, and canary promotion

- Add an adapter evaluation suite with held-out behavior cases and existing
  agent contract/smoke coverage.
- Add baseline versus candidate comparison and runtime-intervention metrics.
- Add manual canary activation and reversible rollback.
- Consider DPO only after comparable same-context rejected/chosen pairs are
  demonstrably available. Consider broader continual-learning strategies only
  after SFT, replay, and regression gates are stable.

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
