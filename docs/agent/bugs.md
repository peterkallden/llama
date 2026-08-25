# Agent Bugs

This file records agent-specific bugs that affected runtime behavior and the
commits that fixed them.

## Structured agent grammar failure before tool-family selection

- Status: Fixed locally; contract tests and Qwen web smoke verified
- Affected area: daemon/web agent turns with small local models such as Qwen
- Symptom: a simple prompt such as `hi there` enters agent mode, llama.cpp
  reports `Unexpected empty grammar stack after accepting piece`, and the web
  request can remain suspended until its turn timeout

### Description

The web client correctly submits an agent turn, but the runtime previously
reached the full structured planner/tool grammar before it had established
whether the request needed tools at all. Small models could fail that grammar
on an ordinary conversational prompt. A retry implemented as a nested chat
call inside the failed resident turn was not a safe general fix: the failed
structured-generation attempt could still own session-lane or
inference-capacity state, so the nested retry could reuse the same runtime
state and wait forever.

### Why this was a bug

Tool selection and ordinary conversation were coupled too late. The host must
decide whether a turn needs external tools before exposing the larger planner
contract. The daemon and CLI now both connect `agent_plan=auto` to
`enable_tool_family_routing`, so the family selection phase is actually run
before the full planner. This daemon wiring is important because the CLI and
daemon have separate configuration adapters.

### Chosen solution

Use a bounded, host-owned family-selection phase before full planning:

```text
request
  -> family selection
       needs_tools=false -> ordinary chat
       needs_tools=true  -> selected family contracts -> plan -> tools
```

The family view should contain only compact family descriptions. Individual
tool contracts are exposed only after the model has selected the relevant
families. A `needs_tools=false` result must never enter the structured tool
planner. If family selection itself fails, ordinary chat is allowed only when
the request has no attached resources and no plan/tool event has been created.
Resource-bearing or already-planned turns fail closed and retain their agent
semantics.

Reflection, deliberation, and internal memory phases do not by themselves mean
that an external tool is required. Explicit memory/resource/tool operations do
count as host-owned work and must not be silently downgraded to chat.

The implementation uses a bounded plain-text family contract rather than a
JSON tool-call grammar for this first small-model decision:

```text
NO_TOOLS
```

or:

```text
TOOLS: dataset, data
```

`NO_TOOLS` routes to ordinary chat with no tool grammar at all. A selected
family is filtered against the host-owned tool view before full planning. The
empty-tool preparation path also deliberately leaves the grammar empty; an
empty tool view is ordinary chat, not an empty structured tool-call grammar.
The web client must not classify debugger/model error strings or implement its
own semantic fallback.

The implementation is covered by `test-tool-family-index`,
`test-agent-prepared-generation`, the runtime/inference contract tests, and a
Qwen-backed web smoke with resident tracing. The trace confirms the intended
sequence: `tool_family_selection` -> `conversation` for `hi there`, with no
planner grammar and no timeout.

## Research checkpoint missing turn identity for project-scoped plans

- Status: Fixed and verified
- Fix commit: `2ba75c1ff` (`Assign turn identity to agent runtime runs`)
- Affected area: CLI agent-runtime research runs with `project` or `session` plan scope
- Failure: `research workspace checkpoint requires bounded identity and sequence`

### Description

The CLI generated an implicit `memory_turn` only when the persisted plan scope
was `turn`. Research workspaces are execution state for one operation and
require a turn identity even when the plan itself is persisted at project or
session scope. A model-backed research run could therefore generate a valid
plan and then fail while creating its first research checkpoint, before any
planned tool call was dispatched.

### Why this was a bug

Persisted plan scope and active operation identity are separate dimensions. The
runtime correctly required bounded checkpoint identity, but the CLI incorrectly
treated a non-turn plan scope as permission to omit the operation turn ID. The
CLI now assigns an implicit bounded turn ID to every agent-runtime invocation
when the caller did not provide one. The plan remains in its requested scope;
only the active operation identity is filled in.

## Resident server-context initialization before model loading

- Status: Fixed and verified
- Fix commit: `4813c990a` (`Initialize resident server context backend`)
- Affected area: resident agent daemon inference on the server-context path
- Verified with: Qwen and Phi model-backed daemon smoke runs, plus the
  `runtime-server-context-host-invalid-model-paths` model-free smoke

### Description

The resident agent server-context host called `server_context::load_model()`
before initializing the llama backend. This could make a daemon inference
request hang during model loading, and it could also make invalid model paths
fail inside the lower-level loading path instead of returning a bounded agent
error.

### Why this was a bug

The server-context load path reports model-loading progress through code that
uses the ggml timer. The timer is initialized by the llama backend bootstrap.
Because the resident host had not performed that bootstrap, the load path used
an uninitialized backend/timer state. This violated the runtime's initialization
ordering contract and made a valid model-backed request fail to reach inference
reliably.

The host now performs the one-time `common_init()`, `llama_backend_init()`, and
`llama_numa_init()` sequence before loading a model. It also validates that the
configured model path is non-empty, exists, and is a regular file before
entering the lower-level load path. Invalid paths therefore produce a bounded
`turn.failed` result rather than hanging or crashing.

The fix is intentionally contained in the agent resident host; it does not
change the shared `tools/server/server-context.cpp` implementation.
