# Agent model residency and multi-model scheduling

This document records the design and implementation boundary for running more
than one generation model in the same `llama-agent` process. It is deliberately
separate from the model-adaptation document: adaptation produces and evaluates
adapter artifacts, while residency decides which already-approved model
profiles are loaded and which turn may use them.

## Current status

The host-neutral preparation is present:

- `common/agent/runtime/model-catalog.*` parses named generation and embedding
  bases, profiles, load policy, adapter references and resident-model limits;
- `common/agent/runtime/model-profile.*` defines the runnable profile identity;
- `common/agent/runtime/model-profile-cache.*` provides bounded admission,
  active-turn pinning and LRU eviction candidates;
- `common_agent_model_router` resolves profiles and reports cache reuse or an
  idle entry that may be evicted;
- `common_agent_inference_capacity_gate` limits active inference turns and
  provides priority-aware, cancellable admission;
- session lanes already serialize turns belonging to one conversation.

The process-wide residency manager is now implemented in
`tools/agent/runtime/agent-model-residency.*`: it consumes the catalog
selection contract, reserves and pins profiles, performs idle LRU eviction,
and calls a backend-neutral loader outside its mutex. The concrete
`agent-model-loaders.*` adapters use the existing llama.cpp file loader for
`cli` and the existing `common_agent_server_context_host` for
`server-context`. A runtime session can attach one of these resident resources
while retaining ownership of its own inference context and KV state. The
daemon/bootstrap path now constructs the manager from the `models` catalog and
passes the configured profile into each daemon turn. Legacy `model.path`
remains the explicit single-model path when no catalog profiles are configured;
it is not consulted after a catalog profile has been selected. Model loading
parameters such as GPU layers and thread count are applied by the concrete
loader adapters at manager construction time, so changing them requires a
daemon restart.

The two-model loader smoke is also available as the unregistered target
`llama-agent-two-model-smoke`. It is intentionally not part of default CTest,
because it loads real model files only when an operator supplies two paths.

## Ownership model

The intended lifetime split is:

```text
daemon process
  └── model residency manager
        ├── shared base model handle per load identity
        ├── profile/adapter state
        └── resident-model admission and eviction
              └── session lane
                    └── per-session inference context and KV state
                          └── turn generation
```

A `llama_model` may be reused when its complete load identity matches. A
`llama_context`, conversation state and KV cache must remain session-owned and
must never be reused across sessions or profiles. A profile is pinned for the
whole turn, including planning, tool calls, reflection, research and final
synthesis.

The residency manager owns model handles; the session host owns context state;
the scheduler owns admission. None of these layers owns conversation memory,
plans, tools or adaptation policy.

## Two scheduler levels

The existing inference gate and the residency manager solve different
problems:

```text
residency admission
  Which model profiles may remain loaded?
  Can an idle profile be evicted?

inference admission
  Which queued turn may execute now?
  Is the turn cancelled or past its deadline?
```

The turn lifecycle is:

```text
resolve host-selected profile
  -> acquire and pin resident profile
  -> acquire inference capacity
  -> create or reuse the session's context for that profile
  -> run the complete turn
  -> release inference capacity
  -> unpin the resident profile
```

Model loading and file I/O must not happen while holding the global scheduler
lock. Admission reserves a key, loading happens outside the lock, and the
loaded handle is published only after validation succeeds. A failed load must
release its reservation and leave the previous active profile untouched.

## Backend-neutral contract, backend-specific loaders

Both supported inference backends must consume the same profile and lifecycle
contract:

| Backend | Residency unit | Context rule | First limitation |
| --- | --- | --- | --- |
| `cli` | `llama_model` plus chat templates and approved overlays | one context per session/profile | simplest first loader; rejects `mmproj` today |
| `server-context` | one server-context host per model/load identity | server context and session state remain isolated | runtime adapter overlays are not supported yet |

The scheduler must not know these backend details. It receives an abstract
loaded-profile handle and a release operation. The CLI and server-context
loaders translate that handle into their own inference session.

The catalog currently permits backend metadata that the concrete loader may not
yet support. In particular, a CLI profile with `mmproj` is rejected by the
runtime today; this combination should be rejected during catalog validation
once the catalog is wired into serving. A server-context profile containing
adapters must likewise fail at profile resolution unless server-context adapter
support has been implemented. It must not silently fall back to another
backend or to the adapter-free profile.

## Profile identity and cache safety

The resident/load identity must include every property that can change serving
behavior:

- base model identity and resolved path;
- backend and multimodal projector;
- model-load parameters such as GPU layers and fitting policy;
- tokenizer and chat-template fingerprints;
- context-size and other context parameters;
- ordered adapter ids and scales.

The host may share a base model handle between profiles when that is safe, but
it must not share KV state. Changing profile, adapter set, tokenizer, template
or context size requires a new inference context. Continuation checkpoints must
carry the profile/cache identity and be rejected when resumed against a
different identity.

The model is never allowed to select a profile, supply a model path, activate
an adapter or increase the residency limit. Profile selection is derived by the
host from request mode, policy and authenticated/session context.

## Memory and eviction policy

`max_loaded_generation_models` is the initial bound. The manager must also be
designed so a later implementation can enforce byte and GPU-memory limits.
The current policy is:

- both `resident` and `lazy` profiles are loaded on demand at the first turn;
- `resident` is an explicit host policy label reserved for keeping a profile
  warm across turns, while the initial manager applies the same hard cache
  bound to both labels;
- `lazy` profiles may be loaded on demand;
- active, loading, or cleanup-pending profiles are not evictable;
- if every resident profile is pinned, the turn waits or fails with an
  observable capacity error;
- eviction releases the actual backend handle before the cache entry is
  removed from the published resident set.

Embedding models are separate generation profiles and have their own semantic
role. They may share the eventual physical memory budget, but they must not be
selected as generation models or accidentally evicted as if they were ordinary
agent profiles.

## Configuration and bootstrap

The catalog is the single source of truth for multi-model serving. The existing
single-model `model.path` configuration remains a separate, explicit fallback
for hosts that do not configure `models.profiles`. Once a catalog has profiles,
the daemon constructs the residency manager and requests are host-bound to the
configured profile (or the catalog default); the legacy path is not used for
those requests. A catalog change, profile change, model path change, or loader
setting change is restart-required because resident resources and session
contexts cannot be swapped behind an active daemon.

Conceptually:

```json
{
  "models": {
    "directory": "/models",
    "bases": {
      "agent-small": {
        "kind": "generation",
        "backend": "cli",
        "path": "qwen-small.gguf",
        "load": "resident"
      },
      "agent-research": {
        "kind": "generation",
        "backend": "server-context",
        "path": "qwen-research.gguf",
        "load": "lazy"
      },
      "nomic": {
        "kind": "embedding",
        "backend": "server-context",
        "path": "nomic.gguf",
        "load": "resident"
      }
    },
    "profiles": {
      "agent-default": {"base": "agent-small", "context_size": 4096},
      "research": {"base": "agent-research", "context_size": 8192}
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

### Optional two-GGUF smoke

Build the target with the normal agent build, then run it only when two
compatible GGUF files are available:

```bash
./bin/llama-agent-two-model-smoke \
  --model-1 /models/model-a.gguf \
  --model-2 /models/model-b.gguf \
  --backend server-context \
  --threads 3 \
  --n-gpu-layers 0
```

The equivalent environment variables are `LLAMA_AGENT_MODEL_1` and
`LLAMA_AGENT_MODEL_2`; explicit arguments take precedence. The test loads the
two files through the selected concrete loader and verifies two distinct
resident resources. It does not make Qwen, Phi, Nomic, CUDA, or a particular
model family a contract requirement. Missing paths return CTest-style skip
code 77, and the test is not registered as an automatic CTest so ordinary
agent suites never load operator-owned model files.

The model profile is a host input, not a model-facing tool or prompt field.
Chat, deliberate, research and adaptation evaluation may request a named
profile through host policy, but a client cannot override it with an arbitrary
path. An explicit escalation should be visible in events and traces.

## Required implementation sweeps

1. Add a profile id and resolved profile identity to host turn preparation,
   continuation/checkpoint validation and readiness/status output. Validate
   backend/profile combinations early.
2. Add a backend-neutral residency manager with an abstract loader, reference
   counts, loading state, eviction, failure cleanup and fake-loader tests.
3. Connect the manager to the session host and session manager. Acquire and
   release profiles around the complete turn while retaining per-session
   contexts.
4. Connect residency admission to the existing inference scheduler. Preserve
   priority, cancellation, deadlines and per-session ordering; do not create a
   second queue.
5. Implement the concrete CLI loader, including model fingerprints and
   adapter compatibility checks. Then implement the equivalent server-context
   loader and its explicit adapter capability boundary.
6. Add model-free tests for reuse, profile isolation, eviction, pinned-model
   refusal, cancellation and failed loads. Add model-backed smokes with two
   small generation models only after the contract tests pass.

The first useful milestone is not unrestricted parallel inference. It is two
profiles that can be resident in one process, with one or a small configured
number of active inference turns, correct context isolation and observable
eviction. Batching and backend-specific GPU scheduling remain later work.

## Non-goals

- changing `llama` or `ggml` core APIs;
- letting the model select or load arbitrary files;
- sharing KV state between sessions;
- silently switching profile after a failed tool call;
- automatically activating adaptation artifacts;
- treating model count as a complete CUDA/VRAM budget;
- adding a second scheduler inside a backend adapter.
