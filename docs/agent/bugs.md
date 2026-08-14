# Agent Bugs

This file records agent-specific bugs that affected runtime behavior and the
commits that fixed them.

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
