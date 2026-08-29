# Agent Model Adaptation

## Status

Design and implementation plan. The observation, transaction, candidate,
corpus, trainer-job, adapter-registry, cause-classification, runtime assembly,
artifact-import, and evaluation-report contracts now exist. There is still no
    external trainer integration, model loading/routing, or automatic adapter activation.

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
- a deterministic JSONL corpus builder requiring explicitly approved and
  redaction-attested candidates, with provenance, revocation, held-out,
  split, duplicate, byte-bound, and bounded replay handling;
- a separate one-shot adaptation worker with an atomic file queue and a
  deterministic `fake-sft` trainer for protocol testing;
- an append-only lifecycle journal for candidate, corpus, training-job,
  training-result, and adapter status events, with idempotency conflict checks;
- an external training job/result contract and a validated adapter registry
  with explicit candidate/canary/active/retired/rejected transitions;
- a host-side adapter artifact verifier that binds a manifest to a canonical
  artifact root, rejects path and symlink escape, enforces a byte bound, and
  recomputes the artifact SHA-256 before import;
- an evaluation report contract that binds a candidate to its corpus, base
  fingerprint, baseline/candidate profiles, test-suite revision, three
  promotion gates, and runtime-intervention counts. It is a report binding,
  not an evaluator or an automatic promotion mechanism;
- a host-neutral model-profile contract for a base-model identity and bounded
  adapter overlays, including a cache key that isolates model, tokenizer,
  template, context, and adapter state;
- a versioned model catalog contract bound to host configuration, with named
  generation/embedding bases, explicit profiles, profile selection, relative
  model paths, load policy, and bounded resident-model limits;
- a host-level adaptation orchestrator and two smokes: a model-free end-to-end
  lifecycle smoke using an explicit evaluator seam, and a model-agnostic
  resident-inference smoke that verifies adaptation-ledger capture can be
  reopened after a real model turn;
- conservative host-side cause classification and recovery linking;
- opt-in assembly wiring with an adaptation-store factory. The factory supports
  in-memory, Cozo, SQLite, and explicit JSONL backends with the documented
  priority order.

The observation identity hash and corpus bundle hash are currently stable
non-cryptographic identity hashes used for local deduplication and test
reproducibility. Resource and artifact integrity must continue to use the
existing SHA-256 stores. The corpus builder now enforces candidate-level
redaction attestation, candidate revocation, held-out exclusion, deterministic
splits, provenance, duplicate handling, and bounded host-selected replay.
The redaction implementation is still an explicit caller/policy seam rather
than a PII detector; semantic replay selection, an actual trainer worker,
model-profile loading, and inference activation remain later sweeps. The
current registry is metadata-only and does not load an adapter. The artifact
verifier and evaluation report only establish import and promotion evidence;
they do not train, evaluate, or activate a model by themselves.

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
- canonical tool family and provider kind (`native`, `mcp`, or `openapi`) when
  a tool signal is involved; the provider kind is provenance and the family is
  the model-facing taxonomy;
- verification status and idempotency key;
- bounded source references and content hashes.

`common_memory_candidate` must not become the training-candidate contract.
Memory candidates describe durable scoped knowledge; training candidates must
describe an approved, redacted, potentially generalizable behavior with an
explicit input and target. The two paths may share scope, hashing, and
provenance helpers, but not their promotion policy.

The current runtime derives `successful_recovery` late, after a response has
been accepted. The runtime finalization seam makes that signal visible to
both consumers. It should also invoke adaptation for an error result when a
qualifying signal exists, while retaining the current rule that ordinary
successful turns do not become training data.

## End-to-end process and provider convergence

Native, MCP, and OpenAPI tools enter adaptation through the same host-owned
execution boundary. Their transport and failure details differ, but the
learning process does not create a separate training path for each provider:

```text
model-facing compact tool contract
              |
              v
     binding, normalization, policy
              |
       +------+------+
       |             |
    native          MCP/OpenAPI adapter
       |             |
       +------+------+
              v
       common tool execution
              |
              v
       result + evidence IDs
              |
              v
       learning signals
       - tool_failure
       - successful_recovery
       - user_correction
              |
              v
   optional adaptation observer
              |
              v
   scoped learning transaction
              |
              v
   cause classification and routing
   memory / procedure / retain / candidate
              |
              v
   curator-approved training candidate
              |
              v
   deterministic SFT corpus JSONL
              |
              v
   external trainer -> evaluation -> adapter registry
```

The provider result must preserve enough bounded provenance to identify the
provider, operation/tool, plan step, scope, and evidence. An MCP transport
failure, an OpenAPI authentication failure, a policy denial, and a malformed
model tool call may all produce a `tool_failure`, but they must not receive the
same cause classification. Only a verified, repeated model-behavior defect may
proceed toward a training candidate. Binding, schema, auth, network, policy,
and server defects are retained as operational evidence and routed for code
or configuration repair instead.

No provider creates an SFT pair directly. The observer records evidence; a
curation step supplies the bounded `approved_prompt` and `approved_target`;
the corpus builder then emits training rows. This keeps MCP and OpenAPI
results reusable as evidence without allowing remote data, credentials, or a
single accidental tool failure to become model weights.

### Learning cases and redaction boundary

The intermediate `common_learning_case` is the explicit bridge between an
observation and a training candidate. It carries only bounded, curator-facing
fields: source observation, complete scope, evidence ids, a tool/schema
fingerprint, redacted input, rejected action, and preferred action. The runtime
does not construct it by copying raw tool output, reflection text, credentials,
or complete conversation history. A case must be marked `redacted` or
`approved` before export; `draft` and `revoked` cases are rejected. The case
content hash and source observation remain available for audit, while the
corpus builder consumes only the approved prompt/target projection.

The case also carries a separate redaction result: `redaction_policy_id`,
`redaction_method`, and `redaction_status`. The current implementation is a
stub seam, not a detector: `caller_asserted` records an explicit host
assertion and does not prove that secrets or personal data are absent. A real
redactor can later replace the seam and use `policy_checked`; it must preserve
the same bounded output and revocation contract.

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

The manifest records candidate ids, source transaction ids, split counts,
redaction policy, replay selection, held-out ids, and hashes. Each JSONL row
also carries its source transaction ids and redaction attestation. A held-out
candidate is excluded before row generation and cannot also be selected as
replay material. Identical duplicate candidate records are collapsed; a
conflicting record with the same id, or overlapping source transactions across
different candidates, stops the build rather than silently changing
provenance.

The builder accepts replay ids selected by a host or ledger query and enforces
`max_replay_candidates`. It does not claim to calculate semantic similarity;
that selector belongs to a later indexed-ledger/worker layer. This keeps the
corpus artifact reproducible and prevents an implicit background query from
changing a revision.

### Shared corpus and deterministic views

The corpus is shared across learning domains. Planning, tool use, research,
and procedure learning are not separate stores or separate trainer inputs;
they are candidate metadata and deterministic views over the same approved
candidate set. A view may select a `learning_domain` and, for tool-use cases,
the canonical `tool_family`. It may optionally restrict `provider_kind` when
an evaluation specifically needs native, MCP, or OpenAPI provenance.

For example, a tool-use view for `diagnostics` includes equivalent candidates
from all three transports when they describe the same model-facing family:

```json
{
  "view": {
    "learning_domain": "tool_use",
    "tool_family": "diagnostics"
  }
}
```

`provider_kind` is recorded on each JSONL row as provenance, not as a second
learning taxonomy. The host must provide the canonical family; it must not
derive `diagnostics` from a transport name such as `mcp_diagnostics`. Empty
view fields mean all values. Filtering happens before corpus qualification,
while approval, redaction, bounds, replay, and provenance rules remain
unchanged. This makes a view a reproducible projection rather than a hidden
second corpus.

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
registry admits only a fully validated manifest. A candidate must then be
staged as `canary` with a passed evaluation report whose adapter id, corpus
revision, corpus hash, training fingerprint, and evaluation revision all
match the manifest. Only a canary can be manually activated. This keeps the
canary boundary explicit without making the registry run inference or promote
automatically.

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

The first worker slice is implemented as `llama-agent-adaptation-worker`. It
processes at most one bundle per invocation:

```text
llama-agent-adaptation-worker --queue var/agent/adaptation/jobs
```

The queue has `pending`, `running`, `succeeded`, `failed`, and `cancelled`
directories. A complete bundle is written to a staging directory and exposed
by one atomic rename. Claiming is also an atomic rename, so two workers cannot
claim the same directory. The worker writes a bounded `state.json`, a
`result.json`, and a job-local artifact directory. A running state records the
worker id, update time, and lease expiry. A later invocation moves an expired
running lease to `failed` with a retryable summary, so a crashed one-shot
worker does not leave the queue permanently blocked. A long-running trainer
must refresh its lease and check cancellation during execution before this
worker is extended beyond the current one-shot fake trainer.

The worker enforces independent positive bounds for artifact bytes, corpus
bytes, and job runtime. The current runtime is the minimum of the job deadline
and the worker limit; the fake trainer is intentionally immediate. Capability
discovery is explicit and side-effect free:

```text
llama-agent-adaptation-worker --capabilities
```

The current response advertises `cuda: false`, `fake-sft` as the only trainer,
and `max_parallel_jobs: 1`. This is a truthful worker capability contract,
not a promise that QLoRA is available.

Only `trainer_kind: "fake-sft"` is enabled in this slice. It creates a
deterministic test artifact and computes its SHA-256, but the artifact is not a
loadable model adapter. The worker does not inspect the adaptation ledger,
access raw resources, start automatically from the daemon, or activate/import
an adapter. PEFT/QLoRA is a later trainer profile behind the same job/result
contract.

### Local orchestration smokes

The host-level orchestration smoke is `llama-agent-adaptation-e2e-smoke`. It
is model-free and uses an explicit fixture evaluator only to prove composition
of the host contracts:

```text
transaction ledger
  -> approved candidate
  -> deterministic corpus revision
  -> queued fake-sft job
  -> one-shot worker result (evaluation = not_run)
  -> injected passed evaluation report
  -> artifact hash verification
  -> candidate -> canary -> active overlay resolution
```

The smoke does not imply that the fake artifact is a loadable adapter or that
the fixture evaluator is a quality evaluator. A real evaluator remains an
external host-controlled step.

The separate `llama-agent-model-adaptation-smoke` runs one real resident
inference turn with adaptation capture enabled and reopens the configured
ledger afterward. It is deliberately model-agnostic:

```bash
llama-agent-model-adaptation-smoke \
  --model models/your-generation-model.gguf \
  --threads 3 --n-predict 32
```

The same model can be supplied through `LLAMA_AGENT_MODEL`; `--model` takes
precedence. Any compatible generation model may be used, including small
Qwen, Phi, Mistral, or other GGUF variants. If no model is available, the
CTest entry returns skip code 77 rather than passing falsely. The model smoke
is an inference/ledger integration check, not a training or adapter-promotion
run.

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
The persistence order is deliberate:

1. **in-memory** for the default, for tests, and for hosts without a durable
   adaptation path;
2. **Cozo** for the normal desktop/server packaging, because it is the
   project's general indexed backend outside Android;
3. **SQLite** for Android and as the final supported persistent fallback.

The first two persistent implementations must share the same transaction-store
contract and use an adaptation-specific relation/table. They must not write
adaptation transactions into the ordinary memory store: memory, procedures,
resources, and adaptation evidence have different retention and promotion
semantics. Payloads should be referenced by content hash where possible; the
ledger should not duplicate large resource or tool-result bodies merely to make
export convenient.

The offline lifecycle uses a separate append-only envelope for state changes.
`candidate`, `corpus_revision`, `training_job`, `training_result`, and `adapter`
are subjects; approval, queue, evaluation, and activation changes are new
events for the same subject. The envelope carries scope, source identity,
content hash, and canonical typed JSON payload. The implementation has
in-memory and explicit JSONL stores, plus the same factory contract for the
optional Cozo and SQLite host adapters. Each backend uses its own
`agent_learning_lifecycle` relation/table and must not invent backend-specific
status semantics.

The host-side journal can be inspected or fed an already curated envelope with
the separate `llama-agent-adaptation-admin` utility. It supports `--list` and
`--append FILE` over the same `--backend`/`--path` selection as the adaptation
storage factory. This is deliberately an operator interface: it does not
expose lifecycle mutation or approval as model-facing tools, and it does not
activate an adapter as a side effect of appending a record.

JSONL remains useful as an explicit portable export/debug representation. It is
not a competing normal persistence backend. The backend factory must fail
clearly when a requested backend is not compiled in; `auto` may select Cozo or
SQLite according to the order above, but must never silently downgrade an
explicit backend request.

### Current runtime capture configuration

The current daemon/bootstrap path exposes only capture, not training or
activation. It is disabled by default:

```json
{
  "runtime": {
    "adaptation": {
      "capture": false,
      "collection_allowed": false,
      "max_evidence": 16,
      "backend": "jsonl",
      "transaction_path": "var/agent/learning.jsonl",
      "stable_model_facing_tools": ["data.inspect"],
      "domains": {
        "planning": true,
        "tool_use": false,
        "research": true,
        "procedure_learning": true,
        "families": {"data": true}
      }
    }
  }
}
```

With an empty `transaction_path`, enabled capture uses an assembly-local
in-memory store. With a persistent path, `backend: "auto"` selects Cozo when
that backend is compiled in, then SQLite. `backend: "jsonl"` is explicit and
keeps the portable append-only representation available for export/debug.
Explicit requests for a backend that is not compiled in fail clearly; they do
not silently downgrade. `collection_allowed` is an explicit collection gate: false keeps
the observer connected but prevents durable transaction collection. The
stable-tool list is deliberately opt-in; it permits a known, validated
model-facing contract failure to be classified as model behavior, while
binding, policy, transport, and provider failures remain host-contract
evidence. This configuration does not create SFT pairs or change the active
model. The optional `domains` object filters collection by learning purpose;
family overrides use canonical model-facing families and apply equally to
native, MCP, and OpenAPI tools. It does not create separate provider corpora;
the corpus builder can later create deterministic views from approved
candidates.

The same complete host-configuration example is available at
[`examples/agent-host-config-adaptation.json`](../../examples/agent-host-config-adaptation.json).

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

The host configuration now accepts the following catalog shape:

```json
{
  "schema_version": 1,
  "model": {"profile": "agent-default"},
  "models": {
    "bases": {
      "qwen-small": {
        "kind": "generation",
        "backend": "server-context",
        "path": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "load": "resident"
      },
      "qwen-research": {
        "kind": "generation",
        "backend": "server-context",
        "path": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "load": "lazy"
      },
      "nomic": {
        "kind": "embedding",
        "backend": "server-context",
        "path": "nomic-embed-text-v1.5.Q4_K_M.gguf",
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
    },
    "routing": {
      "default_profile": "agent-default",
      "embedding_model": "nomic"
    },
    "limits": {
      "max_loaded_generation_models": 2,
      "model_eviction": "lru"
    }
  }
}
```

This is a supported host-configuration shape for catalog parsing and profile
resolution, but it is not yet a supported multi-model serving configuration.
`model.profile` overrides the catalog default; the host resolver returns the
selected base path, `mmproj`, context, load policy, and adapter references.
Bootstrap/model loading still has to consume that selection. Until that next
sweep is complete, the existing `model.path` remains the effective inference
input. It intentionally separates base-model identity, adapter identity,
runnable profile, and host-owned routing. The adapter registry owns artifact
paths and validation; production profile configuration should reference
adapter ids, not arbitrary paths.

The profile contract allows at most eight unique overlays, each with a bounded
positive scale, and supports `resident` or `lazy` loading. Its cache key
includes the ordered overlay list, so changing an adapter or scale cannot
reuse a resident context created for another profile. It does not itself
resolve adapter paths, load a model, switch profiles, or authorize a candidate.

At the host boundary, registry resolution verifies every referenced overlay
before a runtime loader sees it. Only `active` adapters are runtime-resolvable;
each one must match the profile's serving-model, tokenizer, and chat-template
fingerprints. A baseline profile with no overlays remains valid. Canary adapters
are evaluation material, not an implicit serving choice.

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

The evaluation report is the durable hand-off between an external evaluator
and the host. It does not claim that the host ran the tests; it records which
corpus revision, base fingerprint, profiles, and test-suite revision produced
the result. The three gates are deliberately separate:

```json
{
  "revision_id": "evaluation:7",
  "candidate_adapter_id": "adapter:agent-correction-v1",
  "baseline_profile_id": "agent-baseline",
  "candidate_profile_id": "agent-candidate",
  "corpus_revision_id": "corpus:12",
  "corpus_bundle_hash": "sha256:...",
  "base_training_fingerprint": "qwen-small:...",
  "test_suite_revision": "agent-eval:3",
  "gates": {
    "intended_behavior": true,
    "retention": true,
    "agent_regression": true
  },
  "evaluated_turns": 120,
  "runtime_interventions": 14,
  "status": "passed",
  "evaluated_at": "2026-08-29T00:00:00Z"
}
```

The report is valid only when the status agrees with all three gates, the
baseline and candidate profiles differ, and runtime interventions do not
exceed evaluated turns. The derived intervention rate is a diagnostic metric,
not a promotion threshold. A later host-owned policy may require additional
thresholds, signatures, or human approval before a canary or active transition.

## Verification snapshot

The adaptation contract tests currently pass as a focused group: observation,
runtime observer, transaction store, candidate qualification, corpus builder,
trainer protocol, and adapter registry. The model-backed adaptation smoke is
model-agnostic: it accepts any compatible GGUF generation model through
`--model` or `LLAMA_AGENT_MODEL`. Qwen, Phi, and similar small models are
useful local choices, but are not part of the smoke's contract. The smoke runs
one real resident turn with adaptation capture enabled and verifies that the
ledger can be reopened; it does not train or activate an adapter during
inference.

The larger data and document model smokes remain useful regression probes but
are not adaptation failures. They currently expose an existing compact-model
binding seam: Qwen sometimes emits an extra `mode` field for `data.query`, or
emits a direct `document.tables(...)` call instead of the required family
selection wrapper. Host validation correctly rejects these calls and defers
the final answer. The next repair belongs to the model-facing tool binding
path; adaptation must record such a failed/repaired turn, but must not silently
train on it until the contract defect is classified and verified.

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

### Sweep 3 — deterministic corpus builder (completed for the current phase)

Deliverables:

- Build versioned JSONL bundles and manifests from approved candidates.
- Enforce candidate-level redaction attestation, candidate revocation,
  deduplication, stable ordering, train/validation/test split assignment, and
  bounded host-selected replay metadata.
- Keep held-out evaluation cases out of both training and replay selection.
- Keep the builder as a pure local contract in this phase; an inspection/export
  command and semantic replay selector remain worker/ledger work. Remote export
  remains opt-in.

Tests and exit gate:

- the same candidate ids, policy, builder version, and seed produce a byte-for-
  byte identical bundle and manifest;
- tests verify admission redaction status, no revoked or held-out candidate,
  no split overlap, stable deduplication, replay and row/byte bounds, and
  correct provenance;
- a fixture can reconstruct the exact corpus revision from its manifest.

The current redaction check verifies an explicit policy/method/status
attestation and never treats a caller assertion as a secret scanner. A later
redactor may replace `caller_asserted` with `policy_checked` without changing
the corpus boundary.

### Sweep 3b — shared corpus views (completed)

The corpus builder now carries optional `learning_domain`, canonical
`tool_family`, and `provider_kind` metadata on training candidates. A
`common_learning_corpus_view` can deterministically project the shared corpus
for one domain/family and can optionally narrow transport provenance. The
contract test verifies that native, MCP, and OpenAPI tool-use candidates join
the same family view, while research candidates remain in a research view.

Runtime extraction now comes from the resolved tool view at the actual native,
MCP, or OpenAPI execution boundary. `tool_family` remains the canonical
model-facing family and `provider_kind` is provenance only. A provider that
cannot resolve the tool has no metadata to attach; it must not guess from a
provider-prefixed tool name. Indexed semantic replay selection remains a
later extension; the current ledger query is bounded and deterministic.

The principal choice is JSONL versus an indexed database. JSONL is portable
and reproducible; an index is better for scoped queries. The recommended
boundary is an indexed adaptation ledger plus canonical JSONL artifact bundles,
rather than two independent sources of truth.

### Sweep 3c — explicit case promotion (completed)

`common_training_candidate_from_approved_case` is the host-owned seam from a
curator-approved case and a matching verified transaction to an approved
training candidate. It checks the case status, source identity, evidence
membership, observation classification, redaction attestation, and bounded
aggregation counts. A single runtime observation cannot promote itself, and
the runtime never copies raw tool output into a training pair.

The resulting candidate preserves the case's canonical learning domain,
family, and provider provenance. It can still fail the normal corpus
qualification thresholds; approval is authorization to consider the candidate,
not proof that it belongs in a training run.

### Sweep 3d — bounded ledger replay and corpus inspection (completed)

The ledger exposes a bounded query contract over scope, canonical family,
provider provenance, and cause. The same selection can produce deterministic
transaction ids for replay input. The current common helper uses a bounded
ordered scan for all backends; indexed Cozo/SQLite implementations can replace
that internal scan without changing callers. `max_scan` and `max_results` are
part of the contract, and truncation is explicit in the query result.

Corpus revisions also have explicit host APIs for bounded inspection and
JSONL export. Inspection reports row count, byte count, domain/family/provider
counts, candidate ids, and truncation. Export writes only the already-built
revision and enforces a caller-supplied byte bound; it does not rebuild or
silently broaden a corpus.

### Sweep 4 — external trainer protocol (contract slice completed)

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

The current trainer contract now binds a job and result to the corpus bundle
hash and trainer version in addition to the corpus revision, base training
fingerprint, and trainer kind. Returned artifact paths must be relative and
normalized, and the result requires a `sha256:` artifact hash plus an
explicit evaluation status. `not_run` is valid for a trainer result but is
never sufficient for registry admission; only a separate passed evaluation
report can open the canary gate. The worker currently consumes this contract
for the deterministic fake trainer; QLoRA and automatic scheduling remain
unimplemented.

The generated job identity also includes the base fingerprint, trainer kind
and version, code revision, and seed. The same corpus can therefore produce
distinct, auditable training jobs without idempotency collisions.

The trainer-to-registry seam is also host-owned: a validated training result
can be projected into a candidate adapter manifest without manually copying
corpus, trainer, base, or artifact identity. Registry admission repeats the
artifact-path and evaluation checks, so callers cannot bypass them by
constructing a manifest directly. Activation remains a separate explicit
operation and never happens as a side effect of import. The current fake
trainer correctly reports `not_run`; it cannot manufacture a passed evaluation.

The host artifact-import helper adds the filesystem boundary that the metadata
registry intentionally does not own. It canonicalizes an operator-selected
artifact root, rejects traversal and symlink escapes, enforces a byte bound,
and recomputes the manifest's SHA-256. Successful verification returns only a
safe resolved path; the caller must still admit the adapter as `candidate` and
must perform evaluation before any explicit activation.

### Sweep 5 — adapter registry and single-profile inference (contract slice in place)

Deliverables:

- Add adapter registry validation and lifecycle states: candidate, canary,
  active, retired, rejected.
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
- a real model-backed smoke is available after the contract tests pass; the
  model path is supplied at runtime so the target is not tied to one model
  family.

The registry, profile contract, and active-overlay resolver now cover the
metadata and compatibility boundary. The CLI serving path now resolves the
approved manifest artifacts below a host-selected adapter root, loads the LoRA
objects with llama.cpp, and applies their scales to each generation context.
The session carries the profile/cache identity into the inference session so a
profile change recreates the context and cannot reuse KV state. Server-context
currently rejects non-empty overlays explicitly; it must grow an equivalent
backend adapter hook before profiles with adapters are enabled there. The first
deployment should use an overlay, not merge weights into a new base: unloading
the adapter provides a simple rollback and A/B comparison.

The runtime binding deliberately does not let a model choose an adapter. The
host selects a validated profile, the registry permits only `active` manifests,
and the artifact path is resolved under the configured root. An empty adapter
list is the explicit adapter-free baseline. Profile identity is metadata for
readiness/traces and cache isolation; it is not added to the model-facing
prompt.

### Sweep 6 — evaluation, replay, and canary promotion (report/gate slice in place)

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

The report schema, three-gate invariant, canary staging, and manual activation
boundary are now implemented. `common_learning_evaluate_candidate` is the
host-owned evaluator seam: it runs the same immutable fixture against baseline
and candidate, checks comparable turn counts, records both intervention
counts, and produces the report consumed by `stage_canary`. The fixture binds
corpus revision/bundle, base training fingerprint, and test-suite revision;
the host should additionally freeze tool/resource/profile-limit snapshots in
that suite revision. A real evaluator still has to connect held-out,
replay/retention, and agent suites to the runner. Canary activation remains a
manual runtime operation rather than an automatic registry action.

Only after this gate should DPO be considered, and only for genuinely
same-context chosen/rejected pairs. Broader continual-learning strategies are
later work.

### Blueprint hardening before Sweep 7

The blueprint path now has one host-owned structural validation boundary for
bootstrapped, learned, and imported blueprints. It rejects missing identity,
duplicate or unknown step/dependency IDs, cycles, excessive sizes, duplicate
constraint/assumption IDs, and assumptions that are not currently valid. A
blueprint instance receives the caller's scope but does not silently rewrite
assumptions to valid. Resume selection also checks the full applicable scope,
including session, project, and turn identity.

The model-facing selector view is bounded by the host, and blueprint binding
only considers pending or active reasoning steps. It cannot bind completed,
blocked, failed, skipped, tool, or final-response steps. Keyword fallback is
explicitly configurable and requires a unique minimum score. These boundaries
keep a learned blueprint inspectable and prevent stale or oversized blueprint
text from becoming an implicit execution authority.

The remaining blueprint follow-up is provenance/tool-catalog revision binding
when a persisted blueprint is resumed, plus a dedicated CLI-level regression
fixture for model-facing truncation and binding refusal. Neither is required
for the model catalog router seam below.

### Sweep 7 — model catalog and controlled routing

Status: the catalog contract, host JSON binding, validation, path resolution,
and explicit profile selection are implemented. The actual multi-model loader
and scheduler work remains.

Remaining deliverables:

- Migrate the current single-model configuration once the loader consumes the
  catalog; do not retain two long-lived configuration semantics.
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
