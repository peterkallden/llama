# Agent Runtime Architecture

This note describes the current resident-agent runtime direction. The main completed step is the inference/runtime abstraction that now sits between agent behavior and the old CLI-local generation path, with a small foreground daemon layered on top as an admin/test transport.

The goal is to make `llama-agent` able to run the same agent turn from different hosts. The CLI is the first host adapter. A future resident process, service, or MCP-facing application should build the same runtime contracts directly instead of pretending to be CLI arguments.

## Current Shape

The current implementation is still in-process and synchronous.

```text
llama-agent CLI / llama-memory chat compatibility
        |
        v
CLI argument parsing and validation
        |
        v
CLI host adapter
        |
        v
runtime host
        |
        +--> inference session backend
        |       +--> local CLI llama generation
        |       +--> in-process server_context smoke backend
        |
        +--> chat runtime driver
        |
        +--> mini agent runtime driver
                +--> planner
                +--> scheduler
                +--> registered tools
                +--> reasoning / draft / reflection
                +--> memory learning
```

What exists today is a narrow foreground daemon, not a production service lifecycle. It speaks a minimal JSONL protocol over stdin/stdout and is intentionally narrow: one foreground process, keyed session routing for admin/test turns, one worker-driven execution lane behind a small in-process command queue, explicit shutdown, and no detached lifetime management. The daemon suppresses routine info-level model logs in this admin/test path so stdout stays protocol-oriented, while stderr remains available for warnings and errors.

The daemon ready event now advertises a small protocol version plus capability list, and turn results now expose a few host-relevant runtime signals such as runtime reuse, reflection/revision flags, event count and memory-learning summary. That keeps admin/test clients from having to infer runtime behavior from stderr.

Daemon command results now also carry a small internal daemon event list plus `daemon_event_count`. The current JSONL path is still request/response rather than streamed, but admin/test callers can now distinguish queueing, dispatch start, status reporting, session lifecycle actions, shutdown requests, queued-turn cancellation, and active-turn cancellation rejection without scraping diagnostics.

The daemon can now open the same store backends as the CLI path. In addition to the default in-memory stores, a build with Cozo support can use `--backend cozo --memory-db PATH` and `--plan-backend cozo --plan-db PATH` so daemon-based runs exercise the same memory/plan persistence layer.

## Layer Responsibilities

### CLI Adapter

The CLI remains responsible for local command-line concerns:

- Parse and validate `args`.
- Resolve profiles and defaults that are meaningful only to CLI users.
- Translate CLI backend flags and store paths into host-owned runtime/store configuration.
- Bootstrap, import, export, and blueprint package setup.
- Build the tool context for the selected profile and host-owned scope/policy.
- Retrieve memory context and render any CLI debug output.

The CLI should not own the agent loop. It should build runtime inputs and call the runtime host.

### CLI Host Adapter

`agent-cli-host-adapter` is the bridge between CLI state and the runtime host contract.

It currently owns argument-derived wiring that is still local to CLI behavior:

- Build chat and mini host inputs from CLI-owned state.
- Attach the post-run episode-recording hook.
- Resolve provider-backed tool exposure and execution for the selected tool profile.
- Print the final response and decoded-token summary.

This adapter is allowed to know about CLI `args`. The runtime/session host below it should not need to. The current daemon path now follows the same rule for policy/config assembly, even though its own option parsing is still local.

The older explicit memory-tool flags have now been removed from `llama-agent`. Agent-side memory tool exposure goes through the catalog/provider path only: the CLI chooses a tool profile, the host resolves a scoped `agent_tool_view`, and execution stays behind the same provider boundary used by the rest of the runtime.

`agent-cli-run` now also has a small adapter helper beside it. That helper owns CLI-only validation, default stamping, and mini/bootstrap/export setup so the top-level run function can stay focused on retrieval, tool wiring, and dispatch into the runtime host.

The CLI runtime and CLI selection paths now also share one small generation-helper utility for trace IDs, request envelopes, generation options, and failure formatting. That keeps the resident/runtime contract shaping in one place instead of duplicating it across two CLI-facing files.

The CLI tool path is now also shaped the same way. `agent-cli-run.cpp` no longer carries separate inlined assembly blocks for different tool wiring paths. A small CLI resolver now returns one `tools + tool_view + profile_tools_active` bundle, which lets the top-level run function stay focused on retrieval and runtime dispatch while the CLI adapter owns host-specific tool wiring.

### Runtime Host

The runtime host owns one prepared agent turn.

It is responsible for:

- Receiving already-built host inputs.
- Selecting and initializing the inference backend.
- Owning the runtime session lifecycle for the turn, including whether a prepared session is reused or reset after completion.
- Dispatching to chat mode or mini-planning mode.
- Running completion hooks and session reset policy.

The runtime host does not parse CLI arguments. It should remain small enough that a future resident host can provide equivalent inputs directly.

The host inputs now carry a CLI-free runtime turn request: request payload, scope, inference options, runtime policy, runtime config, orchestration config, generation options, and memory authority. The CLI adapter translates `args` into that request at the edge.

The host/runtime path now also carries one small tooling contract instead of threading separate `tools + profile_tools_active + tool_view` fields through each layer. That keeps the provider-facing shape more explicit: one host-owned tooling bundle contains the model-visible `common_chat_tool` list plus the resolved `agent_tool_view` used for execution.

A thin resident-host wrapper now exists above this layer. It owns a runtime session and can run multiple turns against the same host contract without forcing session reset after each turn. That keeps the resident path small: it reuses the same runtime host and turn request instead of introducing a second agent loop.

There is now a small resident runtime layer on top of that wrapper. It owns the reusable resident host session plus the base runtime turn contract, and it can run either ordinary chat turns or mini planning turns against the same keepalive-backed model session. The thinner resident chat and mini helpers now delegate to that layer. Their job remains deliberately narrow: stamp per-turn prompt and turn identity onto the base request, run the turn, and in mini mode keep track of the active plan identity after completion.

The resident path also now has small builder contracts above the raw runtime types: one for constructing a base resident turn request from host-owned model/session/scope settings, one for constructing the resident runtime config itself, and one lightweight daemon-facing turn request/result shape. That keeps the first daemon step focused on process and transport concerns instead of rediscovering how to assemble runtime state.

There is now also a small generic resident session host above that builder layer. It owns the reusable resident runtime plus one small host-owned runtime-reuse key for session/scope matching, and can execute repeated chat or mini turns from a prepared host contract without carrying several parallel `active_*` identity fields.

The `server-context` resident backend now also has its own extracted host layer instead of being assembled inline inside the generic runtime assembly file. That layer still uses the current coarse `server_context` API, but it now owns the backend-specific load key, context key, host config, derived load params, loop lifetime and inference-session construction in one place. It also now distinguishes host configuration from the active running instance that owns the live `server_context`, derived params and loop thread. The active host now materializes the inference session directly from that running instance instead of rebuilding backend-specific details out in the generic assembly layer. Runtime session ownership has also been tightened a little further: for the `server-context` backend, the long-lived resident host now sits in the session's loaded-model state, while the active inference-context state only owns the currently built inference session that uses that host. That gives the next model-versus-context lifetime split a more concrete home instead of leaving it buried inside generic assembly code.

On top of that sits the first explicit session manager for the daemon/admin path. It now keys resident session hosts by namespace and session, while the currently bound project and scope remain part of the runtime state inside that session host. That means the foreground daemon can now treat the resident lane as `session A`, `session B`, `session A again` without baking project ownership directly into the manager key, while still rebuilding the resident runtime if a session changes project or scope.

The foreground daemon entrypoint is now also split a little more cleanly. `agent-daemon.cpp` is mostly the process loop, while a small daemon adapter layer owns daemon-only argument parsing, store and host assembly, and JSONL request/response translation.

The daemon now also routes requests through explicit daemon commands plus a small daemon service layer. On top of that sits a very small dispatcher: stdin/JSON parsing still happens on the transport thread, but command execution now runs through one worker thread and a small bounded in-process queue before reaching the runtime service. The shape is intentionally modest: it separates transport from execution without yet introducing a richer async protocol, streaming events, cancellation, or multiple workers.

The daemon adapter is now also slightly less CLI-shaped in its host construction path. It still uses the existing store-opening helpers, but it no longer has to synthesize a temporary full CLI `args` object just to build runtime policy, runtime config, orchestration config, or the resident session-host contract.

That service layer now understands a slightly broader host-oriented command surface:

- `run_turn`
- `cancel_turn`
- `status`
- `reset_session`
- `close_session`
- `shutdown`

`status` reports a narrow readiness/liveness snapshot plus the currently tracked session keys and queued-command count. It now also exposes a few small lifecycle signals from the dispatcher itself, such as whether the worker thread is running, whether the daemon is still accepting new commands, whether shutdown has been requested, and the current queue capacity. `reset_session` and `close_session` go through the same keyed session manager as ordinary turns, which gives the admin/test path an explicit place to manage resident session state before a fuller queued daemon lifecycle exists.

`cancel_turn` now exists as a first dispatcher-level contract, and the daemon result contract distinguishes two cases explicitly: queued-turn cancellation succeeds and emits a `turn.cancelled` daemon event, while attempts to cancel the currently active turn are rejected with a `turn.cancel_rejected` daemon event plus the active request/turn identity. That is still intentionally narrow. The current runtime/inference/tool stack does not yet have a full end-to-end cancellation token or safe active-turn abort semantics, and the current foreground JSONL transport is still request/response serial from one stdin stream.

On top of that, the CLI now has two thin child-process adapters. `daemon-chat` starts the foreground daemon, sends one turn, reads one response, and shuts the child down. `daemon-session` keeps the same foreground child alive across multiple prompts in the same admin/test session. Both paths still go through the same runtime request/result contracts rather than delegating multi-turn state to a backend conversation loop, and the CLI reads protocol from stdout while relaying daemon diagnostics from stderr separately.

Those child-process adapters now also pass through the same daemon-owned tool configuration surface as the direct JSONL admin/test path: `--tool-profile`, `--repository-root`, and `--mcp-tool-command` all reach the foreground daemon when present. The chat-oriented daemon client path also now defaults its plan scope more conservatively when planning is off, so a simple session- or project-scoped admin/test chat turn does not accidentally force a synthetic turn-scoped contract.

The daemon-facing request shape now carries host-owned scope data such as namespace, session, project, memory scope and plan scope. The current session manager treats namespace plus session as the live resident lane, while status responses still report the currently bound project/scope for that lane. That is still intentionally modest: it is enough to drive multi-turn resident smoke and integration tests, while keeping the future service-owned session model explicit.

Each keyed session host still manages one active resident runtime at a time and still matches reuse from the current host-owned session/scope contract. That is sufficient for the current foreground daemon and smoke coverage, but it is still an early manager shape rather than the final host/service session model.

The runtime surface is now split more explicitly in code as well:

- turn contracts
- resident runtime contracts
- session-host contracts
- host input/build contracts

The backend-specific inference-session selection now also lives with runtime-session initialization rather than in the generic runtime-assembly layer. That keeps runtime assembly focused on agent behavior wiring while session initialization owns backend choice and resident inference-session reuse.

That split is still structural rather than behavioral, but it makes the next service-facing step easier because the current single-active session host is no longer buried inside the same header as every other runtime layer.

Inside the resident runtime session, ownership is now also expressed a little more explicitly. The session keeps separate state for:

- loaded model ownership
- active inference-context ownership

In practice that means a resident session no longer presents one flat bag of `model + templates + inference session + reuse flags`. The first step toward a cleaner resident host is now visible in code: model-loading state and active inference-context state are separate sub-objects, runtime execution asks the session for its active inference session instead of reaching straight into one merged structure, and the `server-context` host itself now lives with the loaded-state side of that split rather than being hidden only inside the active session object.

### Runtime Drivers

The runtime drivers contain the agent behavior.

The chat driver handles ordinary chat generation plus bounded synchronous tool follow-up rounds. The mini runtime driver handles planning, step scheduling, registered tool execution, reasoning, draft synthesis, reflection, and memory learning.

These drivers should not know whether the caller was CLI, a resident process, or a future MCP-facing host.

### Inference Backend

`common_agent_inference` is the abstraction for model generation.

The current backends are:

- `cli`: local llama-backed generation using the existing CLI-style path.
- `server-context`: an in-process resident smoke backend using `server_context`.

The generation request/result contract is narrower than top-level CLI state. Requests carry purpose, trace metadata, scope, messages, tools, optional schema, and generation options. Results return content, decoded-token counts, status, stop reason, parser metadata, and errors in one shared envelope.

Today the runtime session can also be reused when the host keeps the same backend and inference options. The current CLI adapter still chooses to reset after each completed turn, but a resident host no longer needs a different core contract to keep the model session alive across turns.

The reuse logic is now split a little more explicitly inside the resident session as well. There is a small model-load key for properties that really affect model loading, and a separate inference-context key for properties that still require rebuilding the active inference session. In the current resident backends that means turn-shaped settings such as `n_predict` no longer look like model or context identity changes.

The current split is still a pragmatic first slice rather than the final shape. It is good enough for resident smoke and the foreground daemon, but the longer-term split should distinguish:

- model-load options
- context/inference-context options
- per-turn generation options

That split now exists structurally for both the CLI-backed and `server-context` resident sessions, and the daemon/admin path now treats `n_predict` as a turn-level override instead of a resident-runtime identity change. The `server-context` path is still coarser in a deeper sense: its current host object still combines model load and inference-context lifetime, so the next cleanup there is to separate those lifetimes more explicitly rather than just removing turn-shaped fields from the reuse key.

The resident `server-context` host no longer bakes `n_predict` into its own resident host config either. The decode limit is now only stamped onto per-turn server tasks, while the long-lived host keeps only load/context identity plus baseline runtime settings. That makes the reuse boundary line up better with the actual runtime behavior exercised by the resident and daemon `n_predict` reuse smokes.

### Stores and Scope

Memory and plan stores are runtime dependencies, not global singletons.

Scope values are caller-provided authority:

- namespace
- session
- project
- turn
- memory scope
- plan scope
- global-memory opt-in

The model cannot choose these values. A future server or MCP host must derive them from authenticated caller/session context before constructing runtime inputs.

For the current daemon/admin path, store location and backend are still chosen by the host process at startup. Clients can provide session/scope identifiers, but they should never be able to choose arbitrary persistence paths at turn time.

### Tools

Tools currently have three layers:

- Catalog: declares versioned metadata and profile membership.
- Registry: owns executable handlers.
- Adapter bindings: bind catalog definitions to local runtime resources such as memory store, plan id, repository root, and embedding provider.

There is now also a small provider/view boundary above those native pieces:

- `agent_tool_provider`: resolve tools for one host-owned runtime context.
- `agent_tool_view`: expose model-facing `common_chat_tool` values and execute one validated tool call.

The first implementation started native-only, and it still uses the existing catalog, registry, and adapter bindings underneath. The chat runtime no longer dispatches profile tools directly through a runtime-owned registry pointer. Instead, the host resolves a policy-bound, scope-bound `agent_tool_view`, passes `chat_tools()` into generation, and routes parsed assistant tool calls back through that view.

There is now also a first MCP-shaped provider slice beside the native one. `mcp_agent_tool_provider` sits behind a narrow `agent_mcp_tool_client` interface, resolves model-visible `common_chat_tool` values from listed MCP-style tool definitions, namespaces exposed tool names, applies the same host-owned policy gate at exposure time, and normalizes tool-call results back into the shared `agent_tool_result` shape.

That seam now has two concrete test paths:

- an in-process fake MCP client used to exercise provider filtering and result normalization
- a first stdio-based MCP client adapter that speaks JSON-RPC-style `Content-Length` framed messages to a subprocess

The stdio client is still deliberately small. It is enough to prove the provider boundary through a real child process with `initialize`, `tools/list` and `tools/call`, but it is not yet a production MCP lifecycle: there is still no reconnect logic, approval model, streaming event path, or broader capability surface.

Even in this small slice, the stdio client now does a little more than the original smoke seam: it switches Windows stdio pipes to binary framing for `Content-Length` transport, attempts a best-effort `shutdown` plus `exit` sequence before tearing down the child process, and can map structured MCP-side tool error metadata back into the shared failure contract used by native tools.

The stdio client now also keeps a small stderr tail from the MCP child and appends it to transport-level failures when the subprocess exits early or emits malformed JSON-RPC data. That keeps the first real subprocess smoke debuggable without pulling in a larger async lifecycle or logging subsystem yet.

There is now also a first thin integration into the ordinary CLI host-adapter path. When `llama-agent run` is given `--mcp-tool-command`, the CLI tool-selection layer resolves an MCP-backed `agent_tool_view` through the same host/runtime tooling contract used by native tools. This first slice is intentionally narrow: it covers the direct CLI host/runtime path, not the daemon adapters yet.

That CLI path can now also compose native and MCP-backed tools in the same resolved view. If both `--tool-profile` and `--mcp-tool-command` are present, the host resolves them through one small composite provider surface and keeps model-visible tool names unique at the merge boundary.

The foreground daemon can now also resolve the same native and MCP-backed tool surface that the CLI host adapter already uses. Session-host execution now has a per-turn tooling resolver hook, and the daemon uses it to rebuild tooling from the current host-owned session/scope turn contract before each turn. In practice that means `--tool-profile`, `--repository-root`, and `--mcp-tool-command` can now participate in the same resolved daemon tool view instead of the daemon having a narrower MCP-only wiring path.

The modern CLI profile-tool path also no longer builds a second parallel native registry just to derive model-facing tools. For profile-driven tools it now resolves one provider view and reuses that single host-owned surface for exposure and execution wiring.

That keeps host authority in one place:

- the model sees only `common_chat_tool`
- the model requests tools only through parsed `tool_calls`
- the runtime owns bindings, scope, repository root, memory authority, and policy
- disallowed native tools are filtered before exposure instead of being shown and rejected later

The first mini/planning migration is now also in place for blueprint binding. Auto-selected blueprint bindings no longer inspect the native registry directly; they validate against the resolved `agent_tool_view`, which means blueprint-binding now follows the same host-owned exposure and read-only policy surface as chat tool dispatch.

The planning/orchestration edge is also a little less ad hoc now. Plan auto-selection and blueprint auto-selection no longer take a longer loose parameter list for inference, generation config, scope, plan state and tool-view details separately. Instead they can be driven from one small orchestration runtime context that carries the current host-owned planning inputs plus the optional tooling bundle used for blueprint binding.

The older common agent runtime is now also structurally decoupled from the concrete registry type. Planned tool-step execution runs through a small `common_agent_tool_runtime` interface in `common/agent`, and the PoC runtime assembles either a view-backed adapter or a registry-backed fallback adapter underneath it during migration. That means the common planning/runtime core no longer needs to know whether tool execution ultimately comes from the native registry, a host-resolved provider view, or a future MCP-backed adapter.

What still remains is behavioral convergence: the runtime path still uses a registry-backed fallback when no resolved provider view is available, and provider-backed execution still needs broader smoke coverage once more of the mini/runtime flow is exercised through resident/daemon tests.

There is now also a focused smoke for the planned-tool-step path itself. It runs a tiny `common_agent_runtime` scenario where the planner emits a calculator tool step and execution goes through a provider-backed tool-runtime adapter rather than a raw registry pointer. That gives the current refactor one concrete end-to-end proof point before broader resident/daemon mini smokes are added.

The resident/session-host layer has also been trimmed a bit further: it now carries `agent_tool_view` for the modern profile-tool path, but no longer keeps threading a separate `tool_registry` field through its own configs just to pass it onward unused. That makes the resident host/session contracts slightly closer to the intended provider-first shape.

The mini/runtime assembly path has also dropped its registry-backed tool-runtime fallback. Planned tool-step execution in that path now expects the modern provider-backed `agent_tool_view` when profile tools are active, and fails explicitly if a caller tries to run profile-tool planning without that resolved view. That narrows the remaining legacy surface and keeps the planned-tool runtime aligned with the provider-first direction already covered by smoke tests.

The host/chat contracts have now been trimmed in the same direction. The runtime host, chat driver, CLI host adapter and resident host path no longer carry a parallel `tool_registry` field through their modern provider-backed contracts. Chat dispatch itself also no longer carries a separate legacy tool-handler branch. Modern agent-side tool execution now goes through `agent_tool_view`, with the host deciding which scoped provider view to resolve for the turn.

The first shared memory-tool migration now sits underneath that provider path as well. Native `memory_search` and `memory_remember` execution both go through `common_memory_tool_service`, which keeps host-owned scope, embedding, store and policy bindings in one place while preserving the current synchronous behavior and result shapes.

Tool execution is synchronous in this slice. That is deliberate: it preserves current behavior while the runtime boundary stabilizes. A future worker model needs explicit semantics for cancellation, timeouts, ordering, result delivery, and shared-state access.

## MCP Direction

An MCP integration should be built on top of the runtime host, not inside the core agent loop.

The natural mapping is:

```text
MCP-facing llama-agent host
        |
        +--> runtime host
        |       +--> inference backend
        |       +--> stores
        |       +--> scope and policy
        |
        +--> tool providers
                +--> native registry provider
                +--> future MCP client provider
```

For this codebase, "MCP support" should mean the application can act as an MCP host: it discovers tools/resources/prompts from MCP servers through MCP clients, applies local policy, exposes allowed capabilities to the model, dispatches model-requested tool calls, and feeds results back into the runtime.

The runtime should first grow an internal tool-provider boundary before adding MCP transport details.

A useful provider shape is:

- List tools available to this runtime scope and policy.
- Return model-visible schemas for allowed tools.
- Call one tool with validated JSON arguments.
- Return structured success/failure results.

The native tool registry is the first concrete provider. There is now also a first MCP-shaped provider seam built on top of an abstract MCP client contract, plus a first stdio-based MCP client adapter that already exercises `initialize`, `tools/list` and `tools/call` against a subprocess. A later fuller MCP-client implementation can extend that path without changing the runtime loop.

Resources and prompts can follow the same pattern later. They should not be added directly to the agent loop as transport-specific concepts.

## What Not To Build Yet

These are intentionally deferred:

- Detached or OS-managed daemon/service lifecycle.
- Named pipes, Unix sockets, or HTTP transport.
- Concurrent session management.
- Full `llama-server` integration.
- MCP stdio or Streamable HTTP clients.
- JSON-RPC lifecycle and capability negotiation.
- Background tool workers or parallel tool execution.
- A richer streamed event protocol.

The current code should remain useful without any of these. The next steps should keep tightening the runtime/session contract so these pieces have somewhere clean to attach.

## Next Steps

1. Keep moving host construction away from CLI-shaped state.

   The host builders now accept a CLI-free runtime turn request and CLI-free policy/runtime/orchestration contracts. The next cleanup is to move more callers onto those contracts directly, so non-CLI hosts can build prompt/messages, scope, policy, inference options, generation options, plan identity and hooks without routing through CLI-shaped helpers.

2. Keep separating session ownership from model/inference-context ownership.

   The first ownership split now exists inside `common_agent_runtime_session`, where loaded-model state and active inference-context state are tracked separately and the `server-context` resident host now lives with the loaded-state side of that split. The next meaningful step is to carry that separation upward as well, so keyed agent sessions and resident hosts do not implicitly own more model/context lifetime than they need to.

3. Keep shrinking the remaining CLI-shaped adapters around host construction.

   The host/runtime contracts now carry a shared tooling bundle, and the runtime/orchestration modules no longer expose `args`-shaped builders. The next cleanup is to keep pushing that edge outward so more callers can assemble host turns, plan identity and hooks directly from host-owned contracts.

4. Finish moving the remaining tool-exposure and tool-execution decisions behind the provider-facing host surface.

   The runtime host now carries one tooling bundle instead of separate `tools + profile_tools_active + tool_view` fields, and chat/runtime dispatch uses that bundle consistently. The remaining cleanup is mostly convergence work: keep trimming older helper signatures and make sure blueprint/planning-related tool decisions continue to depend on the same host-owned provider view rather than drifting back toward registry-era wiring.

5. Extend cancellation, timeout and event contracts before async tools.

   Synchronous tools are acceptable for the current slice. The daemon now has a small internal event list and queued-turn cancellation contract, but workers should still wait until active-turn cancellation, timeout, retry, ordering and richer failure reporting semantics are explicit.

6. Split model lifetime, inference context lifetime and agent-session lifetime more explicitly.

   The current `server-context` path is a good resident smoke backend, but a real host should be able to keep models loaded while resetting or expiring individual agent sessions.

7. Extend the first stdio MCP client into a fuller MCP host/client path.

   The first provider slice now exists behind an abstract MCP client contract and already has a small stdio transport adapter with basic shutdown and structured tool-error mapping. The next step is to harden that path further: clearer diagnostics, less forceful teardown, better request/response correlation under notifications, and then broader MCP capability support. Add resources and prompts only after tool discovery, session ownership, and policy are stable.

## Backlog Notes

- Keep tightening the session-versus-project split above the current manager key.

  The current daemon/session layer now treats `namespace + session` as the resident lane key, while `project` remains part of the host-owned runtime scope bound within that lane. That is closer to the intended model, but it is still only a first cut: project-scoped memory and plans are still assembled through the same session-host contract, and there is not yet a richer host-owned project object above the resident runtime.

  In practice the intended direction is still the same: `namespace` stays the tenant/authority boundary, `project` is the longer-lived shared work container, and `session` is the shorter-lived live runtime/conversation lane inside that project. That becomes more important once MCP-facing host state, external tool providers, and richer multi-session project workflows sit above the current daemon/admin path.

- Revisit activity order after the current runtime/session cleanup wave.

  The branch has now completed several structural extractions in a row: provider-first tools, resident runtime/session layering, daemon service/dispatcher split, CLI-free runtime builders, and removal of older daemon/session aliases. Before taking another deep refactor in the same area, it is reasonable to pause and choose between a new activity track such as event/cancellation contracts, deeper `server-context` lifetime splitting, or MCP/provider-facing host work.

## Current Verification Baseline

The resident-inference branch has been validated with:

- `test-agent-inference`
- `test-tool-adapters`
- ordinary chat smoke with local Qwen plus Nomic embedding
- mini planning smoke with `--agent-inference-backend server-context`
- resident host multi-turn smoke with `llama-agent-resident-smoke`, verifying the same `server_context` keepalive across two turns
- resident host `n_predict` keepalive smoke, verifying the same `server_context` keepalive survives when the second turn raises the decode limit
- foreground daemon smoke with `llama-agent-daemon`, verifying ready/turn/reuse/shutdown over JSONL
- Cozo-backed foreground daemon smoke, verifying the same daemon path can open persistent memory/plan stores through the existing Cozo store factories
- daemon `n_predict` reuse smoke, verifying a resident session still reports runtime reuse when only the per-turn decode limit changes
- CLI-to-daemon smoke with `llama-agent daemon-chat`, verifying the CLI can drive the same resident backend through the foreground child-process adapter
- multi-turn CLI-to-daemon smoke with `llama-agent daemon-session`, verifying the same child daemon can answer multiple prompts inside one session and scope envelope
- multi-turn CLI-to-daemon tooling smoke with `daemon-session`, verifying the child-process adapter can carry `--tool-profile`, `--repository-root`, and MCP stdio tool wiring through to the same resident daemon session
- direct daemon integration harness with a live foreground process, verifying status, multi-turn chat reuse, reset/close-session lifecycle, project rebinding, native and MCP-configured tooling paths, and mini memory-learning flows against locally available models
- multi-turn daemon `mini` smoke, verifying runtime reuse plus stable `plan_id` reuse across two planning turns in the same resident session
- daemon `mini` learning smoke, verifying resident planning plus post-turn memory-learning summary over the same daemon session when an embedding model is supplied

The foreground daemon `mini` path is now part of the smoke baseline as well. One stabilization issue in this layer was contract drift across wrappers: the daemon request builder was correctly seeded with `server-context`, but a later policy overwrite silently fell back to the default CLI backend, and a second host-execution scope duplication made resident `mini` fragile. The current shape keeps the daemon/backend wiring explicit and reuses `turn_request.scope` as the single host-execution scope source for mini turns.

This baseline verifies that the runtime host and CLI adapter refactors preserve the existing synchronous behavior while making the next host boundary easier to grow.
