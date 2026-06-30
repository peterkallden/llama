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

There is no daemon lifecycle yet. There are no named pipes, Unix sockets, HTTP endpoints, MCP transports, async workers, or background tool queues in this slice.

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

This baseline verifies that the runtime host and CLI adapter refactors preserve the existing synchronous behavior while making the next host boundary easier to grow.
