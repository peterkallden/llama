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

The daemon can now open the same store backends as the CLI path. In addition to the default in-memory stores, a build with Cozo support can use `--backend cozo --memory-db PATH` and `--plan-backend cozo --plan-db PATH` so daemon-based runs exercise the same memory/plan persistence layer.

## Layer Responsibilities

### CLI Adapter

The CLI remains responsible for local command-line concerns:

- Parse and validate `args`.
- Resolve profiles and defaults that are meaningful only to CLI users.
- Translate CLI backend flags and store paths into host-owned runtime/store configuration.
- Bootstrap, import, export, and blueprint package setup.
- Build the native tool catalog, registry, and adapter bindings for the selected profile.
- Retrieve memory context and render any CLI debug output.

The CLI should not own the agent loop. It should build runtime inputs and call the runtime host.

### CLI Host Adapter

`agent-cli-host-adapter` is the bridge between CLI state and the runtime host contract.

It currently owns argument-derived wiring that is still local to CLI behavior:

- Build chat and mini host inputs from CLI-owned state.
- Attach the post-run episode-recording hook.
- Attach the legacy synchronous memory tool handler for old `--memory-search-tool` and `--memory-remember-tool` flows.
- Print the final response and decoded-token summary.

This adapter is allowed to know about CLI `args`. The runtime/session host below it should not need to. The current daemon path now follows the same rule for policy/config assembly, even though its own option parsing is still local.

`agent-cli-run` now also has a small adapter helper beside it. That helper owns CLI-only validation, default stamping, and mini/bootstrap/export setup so the top-level run function can stay focused on retrieval, tool wiring, and dispatch into the runtime host.

The CLI runtime and CLI selection paths now also share one small generation-helper utility for trace IDs, request envelopes, generation options, and failure formatting. That keeps the resident/runtime contract shaping in one place instead of duplicating it across two CLI-facing files.

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

A thin resident-host wrapper now exists above this layer. It owns a runtime session and can run multiple turns against the same host contract without forcing session reset after each turn. That keeps the resident path small: it reuses the same runtime host and turn request instead of introducing a second agent loop.

There is now a small resident runtime layer on top of that wrapper. It owns the reusable resident host session plus the base runtime turn contract, and it can run either ordinary chat turns or mini planning turns against the same keepalive-backed model session. The thinner resident chat and mini helpers now delegate to that layer. Their job remains deliberately narrow: stamp per-turn prompt and turn identity onto the base request, run the turn, and in mini mode keep track of the active plan identity after completion.

The resident path also now has small builder contracts above the raw runtime types: one for constructing a base resident turn request from host-owned model/session/scope settings, one for constructing the resident runtime config itself, and one lightweight daemon-facing turn request/result shape. That keeps the first daemon step focused on process and transport concerns instead of rediscovering how to assemble runtime state.

There is now also a small generic resident session host above that builder layer. It owns the reusable resident runtime plus host-owned session/scope matching rules and can execute repeated chat or mini turns from a prepared host contract.

The `server-context` resident backend now also has its own extracted host layer instead of being assembled inline inside the generic runtime assembly file. That layer still uses the current coarse `server_context` API, but it now owns the backend-specific load key, context key, host config, derived load params, loop lifetime and inference-session construction in one place. It also now distinguishes host configuration from the active running instance that owns the live `server_context`, derived params and loop thread. The active host now materializes the inference session directly from that running instance instead of rebuilding backend-specific details out in the generic assembly layer. That gives the next model-versus-context lifetime split a concrete home instead of leaving it buried inside generic assembly code.

On top of that sits the first explicit session manager for the daemon/admin path. It keys resident session hosts by namespace, session and project so the foreground daemon can now handle `session A`, `session B`, `session A again` and return to the prior resident state for `A` instead of pretending there is only one active slot.

The foreground daemon entrypoint is now also split a little more cleanly. `agent-daemon.cpp` is mostly the process loop, while a small daemon adapter layer owns daemon-only argument parsing, store and host assembly, and JSONL request/response translation.

The daemon now also routes requests through explicit daemon commands plus a small daemon service layer. On top of that sits a very small dispatcher: stdin/JSON parsing still happens on the transport thread, but command execution now runs through one worker thread and a small bounded in-process queue before reaching the runtime service. The shape is intentionally modest: it separates transport from execution without yet introducing a richer async protocol, streaming events, cancellation, or multiple workers.

That service layer now understands a slightly broader host-oriented command surface:

- `run_turn`
- `cancel_turn`
- `status`
- `reset_session`
- `close_session`
- `shutdown`

`status` reports a narrow readiness/liveness snapshot plus the currently tracked session keys and queued-command count. It now also exposes a few small lifecycle signals from the dispatcher itself, such as whether the worker thread is running, whether the daemon is still accepting new commands, whether shutdown has been requested, and the current queue capacity. `reset_session` and `close_session` go through the same keyed session manager as ordinary turns, which gives the admin/test path an explicit place to manage resident session state before a fuller queued daemon lifecycle exists.

`cancel_turn` now exists as a first dispatcher-level contract, but the current support is deliberately narrow: it can cancel a turn that is already sitting in the daemon's internal queue, before execution begins. It does not yet interrupt an actively running turn, because the current runtime/inference/tool stack still lacks a full end-to-end cancellation token and safe active-turn abort semantics. The current foreground JSONL transport is also still request/response serial from one stdin stream, so the first meaningful cancel smoke lives one layer lower at the dispatcher boundary rather than in the top-level stdio protocol.

On top of that, the CLI now has two thin child-process adapters. `daemon-chat` starts the foreground daemon, sends one turn, reads one response, and shuts the child down. `daemon-session` keeps the same foreground child alive across multiple prompts in the same admin/test session. Both paths still go through the same runtime request/result contracts rather than delegating multi-turn state to a backend conversation loop, and the CLI reads protocol from stdout while relaying daemon diagnostics from stderr separately.

The daemon-facing request shape now carries host-owned scope data such as namespace, session, project, memory scope and plan scope. That is still intentionally modest: it is enough to drive multi-turn resident smoke and integration tests, while keeping the future service-owned session model explicit.

Each keyed session host still manages one active resident runtime at a time and still matches reuse from the current host-owned session/scope/inference contract. That is sufficient for the current foreground daemon and smoke coverage, but it is still an early manager shape rather than the final host/service session model.

The runtime surface is now split more explicitly in code as well:

- turn contracts
- resident runtime contracts
- session-host contracts
- host input/build contracts

The backend-specific inference-session selection now also lives with runtime-session initialization rather than in the generic runtime-assembly layer. That keeps runtime assembly focused on agent behavior wiring while session initialization owns backend choice and resident inference-session reuse.

That split is still structural rather than behavioral, but it makes the next service-facing step easier because the current single-active session host is no longer buried inside the same header as every other runtime layer.

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

The native tool registry can be the first provider. A later MCP-client provider can translate MCP `tools/list` and `tools/call` into the same internal shape.

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

2. Separate session management from model/context ownership more explicitly.

   The daemon/admin path now has a keyed session manager, but each keyed entry still owns a resident session host with its own resident runtime lifetime. The next meaningful runtime step is to keep the keyed agent-session model while separating it more clearly from model-loading and inference-context ownership.

3. Relax the current runtime reuse key where it is too turn-shaped.

   The current reuse match still includes values such as `n_predict`. Before a multi-session manager hardens around it, the host should split model/context reuse concerns from per-turn generation options more clearly.

4. Make tool provider discovery explicit.

   Keep the existing catalog/registry/adapters, but introduce a provider-facing contract for listing and calling tools. The native registry should implement it first. MCP can then become another provider rather than a special runtime mode.

5. Add cancellation, timeout and event contracts before async tools.

   Synchronous tools are acceptable for the current slice. Workers should wait until timeout, cancellation, retry, ordering and failure reporting semantics are explicit. The same is true for daemon output: a richer internal event model should exist before transport-specific streaming grows.

6. Split model lifetime, inference context lifetime and agent-session lifetime more explicitly.

   The current `server-context` path is a good resident smoke backend, but a real host should be able to keep models loaded while resetting or expiring individual agent sessions.

7. Create MCP client/provider support after the internal provider contract exists.

   Start with `initialize`, `tools/list`, and `tools/call` for one local stdio server. Add resources and prompts only after tool discovery, session ownership, and policy are stable.

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
- multi-turn daemon `mini` smoke, verifying runtime reuse plus stable `plan_id` reuse across two planning turns in the same resident session
- daemon `mini` learning smoke, verifying resident planning plus post-turn memory-learning summary over the same daemon session when an embedding model is supplied

The foreground daemon `mini` path is now part of the smoke baseline as well. One stabilization issue in this layer was contract drift across wrappers: the daemon request builder was correctly seeded with `server-context`, but a later policy overwrite silently fell back to the default CLI backend, and a second host-execution scope duplication made resident `mini` fragile. The current shape keeps the daemon/backend wiring explicit and reuses `turn_request.scope` as the single host-execution scope source for mini turns.

This baseline verifies that the runtime host and CLI adapter refactors preserve the existing synchronous behavior while making the next host boundary easier to grow.
