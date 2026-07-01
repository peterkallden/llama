# Agent Runtime Architecture

This note describes the current resident-agent runtime direction. It is intentionally focused on the runtime boundary before any daemon, socket, MCP transport, or server lifecycle work.

The goal is to make `llama-agent` able to run the same agent turn from different hosts. The CLI is the first host adapter. A future resident process or MCP-facing application should build the same runtime contracts directly instead of pretending to be CLI arguments.

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

There is still no real daemon lifecycle yet. There are no named pipes, Unix sockets, HTTP endpoints, MCP transports, async workers, or background tool queues in this slice.

A small foreground daemon smoke now exists on top of the resident runtime builders. It speaks a minimal JSONL protocol over stdin/stdout and is intentionally narrow: one process, one host-owned model config, synchronous turn handling, explicit shutdown, and no detached lifetime management. The daemon now suppresses routine info-level model logs in this admin/test path so stdout stays protocol-oriented, while stderr remains available for warnings and errors.

The daemon ready event now advertises a small protocol version plus capability list, and turn results now expose a few host-relevant runtime signals such as runtime reuse, reflection/revision flags, event count and memory-learning summary. That keeps admin/test clients from having to infer runtime behavior from stderr.

## Layer Responsibilities

### CLI Adapter

The CLI remains responsible for local command-line concerns:

- Parse and validate `args`.
- Resolve profiles and defaults that are meaningful only to CLI users.
- Open memory and plan stores from CLI paths or backend flags.
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

This adapter is allowed to know about CLI `args`. The runtime host should not need to.

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

There is now also a small generic resident session host above that builder layer. It owns the reusable resident runtime plus host-owned session/scope matching rules and can execute repeated chat or mini turns from a prepared host contract. The foreground daemon uses that host through a thin compatibility alias, and direct resident smoke can exercise the same host without going through daemon protocol code.

On top of that, the CLI now has two thin child-process adapters. `daemon-chat` starts the foreground daemon, sends one turn, reads one response, and shuts the child down. `daemon-session` keeps the same foreground child alive across multiple prompts in the same admin/test session. Both paths still go through the same runtime request/result contracts rather than delegating multi-turn state to a backend conversation loop, and the CLI reads protocol from stdout while relaying daemon diagnostics from stderr separately.

The daemon-facing request shape now carries host-owned scope data such as namespace, session, project, memory scope and plan scope. That is still intentionally modest: it is enough to drive multi-turn resident smoke and integration tests, while keeping the future service-owned session model explicit.

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

- Daemon lifecycle.
- Named pipes, Unix sockets, or HTTP transport.
- Full `llama-server` integration.
- MCP stdio or Streamable HTTP clients.
- JSON-RPC lifecycle and capability negotiation.
- Background tool workers or parallel tool execution.
- Long-lived multi-turn session protocol.

The current code should remain useful without any of these. The next steps should keep tightening the runtime contract so these pieces have somewhere clean to attach.

## Next Steps

1. Use the runtime turn request end-to-end outside CLI.

   The host builders now accept a CLI-free runtime turn request. The next cleanup is to move more callers onto that contract directly, so non-CLI hosts can build prompt/messages, scope, policy, inference options, generation options, plan identity and hooks without routing through CLI-shaped helpers.

2. Make tool provider discovery explicit.

   Keep the existing catalog/registry/adapters, but introduce a provider-facing contract for listing and calling tools. The native registry should implement it first. MCP can then become another provider rather than a special runtime mode.

3. Define host-owned policy boundaries.

   Document and enforce where scope authority, tool allowlists, write permissions, global memory opt-in, plan authority, and sensitive operations are checked.

4. Add cancellation and timeout policy before async tools.

   Synchronous tools are acceptable for the current slice. Workers should wait until timeout, cancellation, retry, ordering and failure reporting semantics are explicit.

5. Split resident host lifecycle from inference backend.

   The current `server-context` path is an in-process smoke backend. A real resident host should own model lifetime, session reuse policy, cancellation, and resource shutdown separately from CLI process lifetime.

6. Create MCP client/provider support after the internal provider contract exists.

   Start with `initialize`, `tools/list`, and `tools/call` for one local stdio server. Add resources and prompts only after tool discovery and policy are stable.

## Current Verification Baseline

The resident-inference branch has been validated with:

- `test-agent-inference`
- `test-tool-adapters`
- ordinary chat smoke with local Qwen plus Nomic embedding
- mini planning smoke with `--agent-inference-backend server-context`
- resident host multi-turn smoke with `llama-agent-resident-smoke`, verifying the same `server_context` keepalive across two turns
- foreground daemon smoke with `llama-agent-daemon`, verifying ready/turn/reuse/shutdown over JSONL
- CLI-to-daemon smoke with `llama-agent daemon-chat`, verifying the CLI can drive the same resident backend through the foreground child-process adapter
- multi-turn CLI-to-daemon smoke with `llama-agent daemon-session`, verifying the same child daemon can answer multiple prompts inside one session and scope envelope
- multi-turn daemon `mini` smoke, verifying runtime reuse plus stable `plan_id` reuse across two planning turns in the same resident session
- daemon `mini` learning smoke, verifying resident planning plus post-turn memory-learning summary over the same daemon session when an embedding model is supplied

The foreground daemon `mini` path is now part of the smoke baseline as well. One stabilization issue in this layer was contract drift across wrappers: the daemon request builder was correctly seeded with `server-context`, but a later policy overwrite silently fell back to the default CLI backend, and a second host-execution scope duplication made resident `mini` fragile. The current shape keeps the daemon/backend wiring explicit and reuses `turn_request.scope` as the single host-execution scope source for mini turns.

This baseline verifies that the runtime host and CLI adapter refactors preserve the existing synchronous behavior while making the next host boundary easier to grow.
