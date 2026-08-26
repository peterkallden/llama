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

## Dataset attachment binding after compact planning

- Status: Fixed locally; runtime JSON regression coverage passes
- Affected area: dataset inspection of small-model turns with attached or
  previously scoped resources
- Symptom: a CSV upload succeeded, but a subsequent `dataset.inspect` could
  report that the dataset reference was unavailable.

The model-facing resource catalog uses compact handles: `rN` means a
current-turn attachment and `sN` means a host-listed scoped resource. The
normalizer previously selected the right `sN` candidate but accidentally
copied the URI from the `rN` list. It also handled `resource:"r1"` for dataset
tools but not the equivalent `dataset:"r1"` that a small model can emit when
it follows the `dataset_ref` schema annotation. Both forms now resolve to the
host-owned resource URI before the dataset adapter runs. With exactly one
current attachment, a non-canonical model alias is treated as the implicit
attachment default; explicit `dataset://...` references remain unchanged.

## Family selection allowed an ungrounded answer fallback

- Status: Fixed locally; deterministic contract tests and a Qwen web smoke are
  required for verification
- Affected area: daemon/web agent turns using `agent_plan=auto`
- Symptom: a request such as `what time is it?` selected the `time` family, but
  the planner could still produce a reasoning/answer step without completing
  `time_now`. The turn then returned a placeholder-like answer instead of the
  UTC value from the tool.

### Cause and fix

Family selection is a host-owned decision that says external evidence is
needed. It must therefore not only narrow the model-facing tool view; it must
also set the existing `require_tool_execution` request contract. The normal
runtime guard then rejects answer-only fallback until a mandatory tool step has
completed. This keeps the behavior aligned with ordinary CLI requests that
already use the same contract, without adding a second family-specific
execution path.

The `time` family is described semantically as `Return the current time`. The
family view remains compact and does not expose `time_now` until the model has
selected the family; the later planner view contains the exact tool name and
contract.

Singleton tools with no required arguments now use the existing chat/tool
driver directly after family selection. This is a fast path, not a second
execution model: multi-tool and dataflow requests still use the normal plan
compiler/runtime. The selected request remains marked as requiring tool
execution, so the fast path also fails closed if the model emits no tool call.

The host/bootstrap value `limits.max_tool_rounds: 0` is treated as “use the
default budget” and resolves to 16 rounds. This avoids a misleading situation
where required tool selection is enabled but the execution budget is
accidentally zero; explicit positive limits remain authoritative.

## Resource upload scope drift at the daemon boundary

- Status: Fixed locally; daemon scope regression test added
- Affected area: authenticated TCP and Unix resource upload/read/list commands
- Symptom: an upload could succeed, but the following turn could not read the
  returned URI when the client omitted namespace/project fields.

The JSONL parser supplies protocol defaults for omitted scope fields. A turn is
then rebound to the authenticated token policy, so the resource and turn could
end up in different authorities. The daemon now binds `put_resource`,
`read_resource`, and `list_resources` to the authenticated namespace/project,
regardless of whether the client omitted them or supplied conflicting values.
`run_turn` and `execute_tool` use the same shared binding helper. This is an
authorization consistency fix, not a permission expansion: the token policy
still determines the effective scope. `test-agent-daemon-scope` covers omitted
fields, conflicting fields, and the implicit caller session.

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
