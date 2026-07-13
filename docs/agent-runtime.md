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

That foreground daemon now also has a first explicit lifecycle-state contract above the worker/queue slice. The current state model is still intentionally small, but `starting`, `ready`, `draining`, `stopping`, `stopped`, and `failed` are now named service states instead of being inferred only from scattered booleans and transport-local status shaping.

The same daemon path is now also a little less transport-shaped around status reporting. Readiness/liveness, queue state, active request identity, and session descriptors now sit behind one daemon-status object first, and the current JSONL protocol mainly serializes that host-owned status surface rather than inventing it inline.

The response side is now slightly less string-shaped too. The daemon command result carries an explicit response kind for `turn`, `status`, or lifecycle/session actions, so the current JSONL adapter no longer has to infer the response shape only from ad hoc event-string conventions before serializing it.

The command side is now also a little less ad hoc. The transport still speaks the same JSONL command fields, but the host-owned daemon command contract now carries small typed payloads for turn execution, session actions and queued-turn cancellation instead of relying only on a flat bag of optional top-level fields.

The JSONL client/transport seam is now moving in the same direction. It no longer only has a special turn-request helper plus an ad hoc shutdown helper; it now has named request builders for ordinary turn execution, `status`, queued-turn cancellation, session actions and shutdown, so the current stdio/JSONL adapter is a little less likely to grow one-off inline command JSON as the service surface expands.

The daemon ready event now advertises a small protocol version plus capability list, and turn results now expose a few host-relevant runtime signals such as runtime reuse, reflection/revision flags, event count and memory-learning summary. That keeps admin/test clients from having to infer runtime behavior from stderr.

Turn results now also carry a first structured trace history. The current slice is intentionally modest: the trace is still a bounded execution summary rather than a streamed event protocol, but it already records host-safe facts such as plan creation/resume, step activation/completion, observation recording, tool success/failure, reflection decisions, memory-learning outcomes, and final response completion.

Daemon command results now also carry a small internal daemon event list plus `daemon_event_count`. The current JSONL path is still request/response rather than streamed, but admin/test callers can now distinguish queueing, dispatch start, status reporting, session lifecycle actions, shutdown requests, queued-turn cancellation, and active-turn cancellation rejection without scraping diagnostics.

The daemon can now open the same store backends as the CLI path. In addition to the default in-memory stores, a build with Cozo support can use `--backend cozo --memory-db PATH` and `--plan-backend cozo --plan-db PATH` so daemon-based runs exercise the same memory/plan persistence layer.

There is now also a first shared host-config slice above those flags. The foreground daemon and the real MCP stdio server can both accept `--config PATH` and load one small JSON host-owned configuration model for model/backend settings, runtime defaults, stores, resources, tool profile, MCP subprocess providers, and a few coarse limits. CLI flags still exist and still matter, but this is the first path where daemon/runtime construction does not have to start from a full CLI-style `args` object.

The resource-store slice now follows the same host-owned backend pattern. The CLI and daemon argument surfaces accept:

- `--resource-blob-backend auto|in-memory|fs|s3`
- `--resource-blob-root PATH`
- `--resource-metadata-backend auto|in-memory|cozo`
- `--resource-metadata-db PATH`

The current implementation supports `fs` and `in-memory` for blob storage, and `in-memory` and `cozo` for metadata. `s3` remains deferred. In the current default shape, blob storage resolves to `fs` and derives a default root if one is not supplied, while metadata resolves to `cozo` when a metadata DB path is present and otherwise stays `in-memory`.

The resource path is also now split more cleanly internally:

- blob storage owns raw bytes and content-addressed persistence
- resource catalog owns descriptor/authority metadata and lookup
- the composed resource store binds those two responsibilities together for runtime and tool callers

## Design Constraints

The runtime direction depends on keeping the layer boundaries boring and explicit.

- `common/memory` should not depend on agent/runtime host code.
- `common/plan` may depend on memory contracts and stores, but not on agent PoC host flow.
- `common/agent` may depend on plan and memory, but should still prefer neutral runtime-facing contracts when a type does not need full agent semantics.
- `pocs/agent` may depend on all lower layers and is where host adapters, daemon experiments, MCP-shaped seams, and resident runtime assembly can live.
- `pocs/memory` should not become the owner of agent orchestration or resident host flow.

In practice that means "almost production" shared types such as lightweight runtime DTOs, resource references, trace envelopes, or host/service contracts should move toward neutral common headers instead of being trapped inside one PoC adapter. The goal is to keep reusable contracts below the PoC host layer and keep the PoC layer focused on assembly rather than ownership of core abstractions.

The practical JSON rule is now the same: if JSON crosses a subsystem boundary and is not just a short-lived local implementation detail, it should move behind a named parse/serialize/validate helper. JSON as a wire or storage format is fine; raw `ordered_json` plus string `.dump()` should not silently become the contract.

The first concrete example of that constraint is now in place for tracing: the structured trace envelope lives in a neutral `common/runtime-trace.h` header, while `common/agent` populates it and `pocs/agent` only adapts or serializes it for CLI/daemon surfaces.

The same cleanup has now started for tool execution contracts. The lightweight tool-call and execution-result DTOs used by runtime-side tool validation/execution no longer need to live in the heavier native registry header; they now have their own smaller neutral contract header. The native registry still exists and still owns handlers, but the runtime-facing contract is starting to separate from the older registry-era shape.

The same direction has now been reinforced around the biggest JSON-heavy runtime edges:

- native tool payloads now serialize through named result-contract helpers instead of ad hoc JSON literals at each return site
- the older native chat-bridge path now also serializes tool success/failure payloads through named common-agent helpers instead of inlined JSON snippets
- the runtime core now also routes its remaining bounded observation/default JSON seams through named runtime-contract helpers instead of open-coding request-tool default stamping and observation payload `.dump()` calls in `agent-runtime.cpp`
- those runtime helpers now also expose one JSON-level safe-default shaping seam for tool arguments, plus a named reasoning-observation serializer, so tests and future adapters can validate the contract without going through a full runtime turn
- daemon JSONL request/response shaping now goes through explicit daemon protocol helpers
- the foreground daemon and its child-process client now also share named JSONL ready/event parsers instead of validating those protocol messages inline
- MCP stdio transport still owns framing, but JSON-RPC request/notification construction and tool result parsing now live behind extracted protocol helpers
- plan-step tool arguments now have a small named contract wrapper even though stored compatibility still remains `arguments_json`
- host config now has an explicit `schema_version`, a validator, and a roundtrip JSON helper instead of only a one-way parse path

That does not mean every JSON surface is now formalized. It means the highest-value runtime seams now have a named contract boundary, so later daemon/host/MCP work is less likely to hard-code behavior into scattered `.dump()` or `parse()` sites.

The same direction has now started for host-owned resource references. The neutral `common/runtime-resource.h` contract no longer stops at lightweight resource refs; it also carries the first blob/resource store interfaces plus authority/descriptor DTOs. The current implementation remains intentionally local to `pocs/agent`, but it is no longer only an in-memory proof: resource blobs can now be stored on the filesystem through a content-addressed `fs` blob backend, and resource metadata can now be persisted through a first Cozo-backed metadata store. In the current default shape, resource blob storage prefers `fs`, while metadata remains `in-memory` unless a Cozo metadata database is selected explicitly or implied by `--resource-metadata-db`.

That contract is now also a little less ad hoc for tools. Native tool bindings no longer thread a raw `resource_store` plus separate namespace/session/project/turn fields through the host path. Instead they carry one scoped `agent_resource_runtime`, and helper functions derive read authority or stamp put-requests from that host-owned runtime scope.

That resource metadata is also starting to become more than a storage note. The current descriptor shape now has room for host-authored purpose, short content summary, usage hint, limitations, and lightweight semantic tags such as keywords or entity names. The intended meaning is practical rather than decorative: a resource row should answer what it is, why it was created, what it contains in short, how a later step can use it, and what its limits are.

The same contract-cleanup pattern now also covers two older JSON-heavy seams that were still carrying more history than necessary:

- plan tool-argument parsing/materialization now goes through the named `common_plan_tool_arguments_contract` path instead of reparsing and redumping ad hoc JSON inside `plan-bindings`
- the plan argument serializer now re-applies the safe integer normalization pass, so wrapped or legacy small-model shapes cannot leak `"limit":"2"` style control fields back out after normalization
- memory tool `search` and `remember` execution now parse JSON through named argument-contract helpers before business logic runs, instead of letting the service implementation read raw `ordered_json` directly
- those memory tool contracts can now also be parsed from an already-owned JSON value, so host/runtime adapters do not need to round-trip through a string just to validate or normalize bounded memory-tool arguments

That is still intentionally modest. The stored compatibility format remains `arguments_json` where older plan/runtime paths expect it, but the contract boundary is now explicit at parse/materialize/serialize time.

The same "almost boring" rule now applies to symbolic memory. The clean first fit is not a second symbolic-memory subsystem beside the current memory path; it is the existing host-owned memory model with a few more durable kinds and a later typed overlay above it. `procedure` already fills one symbolic role today, so the next incremental expansion is to let `constraint` and `decision` travel through the same scope, retrieval, proposal, provenance, MCP, and store contracts. Overlay compaction for planning, reflection, and reasoning should sit above that durable store rather than replace it.

That first overlay slice now exists in a deliberately narrow form. `common/memory` can render a compact symbolic overlay from already retrieved memories, grouping bounded `constraint`, `decision`, `procedure`, and a small amount of supporting `fact` context. The current planner, reasoning, draft, reflection, and memory-learning prompts still keep the older full memory context available, but they can now prepend this typed overlay so the model gets a cleaner symbolic summary without changing persistence, authority, or retrieval ownership.

The next small refinement is now also in place: overlay selection is stage-aware before rendering. The current selector is still deterministic and intentionally simple, but it no longer treats every prompt phase the same. Planning prefers `constraint` and `decision`; reasoning still favors `procedure` while allowing relevant symbolic guidance to follow; reflection emphasizes `constraint` and `decision`; and memory-learning can bias back toward `decision` plus reusable `procedure` context. That keeps retrieval and persistence unchanged while giving each runtime phase a cleaner symbolic slice from the same authorized memory set.

There is now also a first compacting pass inside that same host-owned rendering layer. It stays deliberately local to prompt shaping: before symbolic overlay or policy-pack text is rendered, repeated normalized items are deduplicated, oversized entries are trimmed to bounded per-item limits, and the retained set is capped by the existing section budgets. This is not a new store or a second compaction database. It is prompt-budget hygiene that fits the current model: keep durable memory unchanged, compact the rendered overlay/policy view when repetition or prompt pressure would otherwise waste context.

There is now also a narrow post-reflection hook for session-owned policy packs. It is intentionally conservative: if a turn actually ran reflection and the active session policy-pack still shows obvious duplication or section overflow according to the existing host-owned compaction heuristic, the session host compacts that pack and re-seeds the resident runtime with the compacted version for later turns. If there is no sign of duplication or pressure, nothing happens.

The next adjacent layer is now also explicit: a small host-owned policy-pack contract can be rendered ahead of the retrieval overlay. This follows the same broad shape as bootstrap blueprints and plan templates, but it is intentionally not a plan and not a second memory store. It is a compact declarative pack for purpose, goal, success criteria, constraints, decisions, and preferred procedures that a host or session can supply directly. The current prompt builders can derive a lightweight pack from caller-owned objective data or from the active plan's blueprint-like constraints, then prepend it before stage-selected symbolic memory.

That policy-pack seam is now present in the runtime/session contracts as well, not only in prompt rendering. Resident/session configuration can carry a stable pack across turns, the mini runtime path now preserves that host-owned pack instead of dropping it while rebuilding `common_agent_request`, and the daemon resident-request builder can seed one small session-level pack from host configuration. Request/objective and active-plan derivation still remain as fallbacks, but the host path no longer depends on those fallbacks alone.

The session layer now owns that seam a little more explicitly too. A session-host turn can provide a policy-pack override once, the host retains it as session state, later turns can omit it without losing the active pack, and daemon/session status can report the current `policy_pack_id` for diagnostics. The intent is still conservative: policy-pack identity is session state, not part of the resident model/inference reuse key.

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

That provider assembly is now a little less CLI-shaped internally as well. The CLI-facing resolver still exists, but it now sits on top of a smaller host-owned tool-selection request: resolved `agent_tool_context`, repository root, resource-store config, and optional MCP stdio provider command. The foreground daemon now uses that host-owned request directly instead of first synthesizing a temporary CLI `args` object just to reach the tool provider seam.

The new host-config slice is intentionally modest. It currently models:

- model backend/path and optional embedding model
- runtime defaults such as context size, `n_predict`, planning/reflection toggles, and trace/learning flags
- memory/plan store backend and path
- resource blob/metadata backend and path
- tool profile, repository root, and a list of configured MCP providers
- a few daemon-style limits such as queue capacity and max turn seconds

In the current slice, the daemon and the real MCP stdio server can both carry a list of enabled stdio MCP subprocess providers from that host-config path into the provider/view seam. Non-stdio transports are still deferred, but the host-owned config model no longer has to collapse back down to a single external tool provider before runtime assembly.

A thin resident-host wrapper now exists above this layer. It owns a runtime session and can run multiple turns against the same host contract without forcing session reset after each turn. That keeps the resident path small: it reuses the same runtime host and turn request instead of introducing a second agent loop.

There is now a small resident runtime layer on top of that wrapper. It owns the reusable resident host session plus the base runtime turn contract, and it can run either ordinary chat turns or mini planning turns against the same keepalive-backed model session. The thinner resident chat and mini helpers now delegate to that layer. Their job remains deliberately narrow: stamp per-turn prompt and turn identity onto the base request, run the turn, and in mini mode keep track of the active plan identity after completion.

The resident path also now has small builder contracts above the raw runtime types: one for constructing a base resident turn request from host-owned model/session/scope settings, one for constructing the resident runtime config itself, and one lightweight daemon-facing turn request/result shape. That keeps the first daemon step focused on process and transport concerns instead of rediscovering how to assemble runtime state.

There is now also a small generic resident session host above that builder layer. It owns the reusable resident runtime plus one small host-owned runtime-reuse key for session/scope matching, and can execute repeated chat or mini turns from a prepared host contract without carrying several parallel `active_*` identity fields.

The `server-context` resident backend now also has its own extracted host layer instead of being assembled inline inside the generic runtime assembly file. That layer still uses the current coarse `server_context` API, but it now owns the backend-specific load key, context key, host config, derived load params, loop lifetime and inference-session construction in one place. It also now distinguishes host configuration from the active running instance that owns the live `server_context`, derived params and loop thread. The active host now materializes the inference session directly from that running instance instead of rebuilding backend-specific details out in the generic assembly layer. Runtime session ownership has also been tightened a little further: for the `server-context` backend, the long-lived resident host now sits in the session's loaded-model state, while the active inference-context state only owns the currently built inference session that uses that host. That gives the next model-versus-context lifetime split a more concrete home instead of leaving it buried inside generic assembly code.

On top of that sits the first explicit session manager for the daemon/admin path. It now keys resident session hosts by namespace and session, while the currently bound project and scope remain part of the runtime state inside that session host. That means the foreground daemon can now treat the resident lane as `session A`, `session B`, `session A again` without baking project ownership directly into the manager key, while still rebuilding the resident runtime if a session changes project or scope.

The foreground daemon entrypoint is now also split a little more cleanly. `agent-daemon.cpp` is mostly the process loop, while a small daemon adapter layer owns daemon-only argument parsing, store and host assembly, and JSONL request/response translation.

The daemon now also routes requests through explicit daemon commands plus a small daemon service layer. On top of that sits a very small dispatcher: stdin/JSON parsing still happens on the transport thread, but command execution now runs through one worker thread and a small bounded in-process queue before reaching the runtime service. The shape is intentionally modest: it separates transport from execution without yet introducing a richer async protocol, streaming events, cancellation, or multiple workers.

The foreground JSONL transport loop is now also explicitly owned by the daemon adapter rather than living inline inside `agent-daemon.cpp`. That is still a small step, but it matters: `main` is now closer to pure process bootstrap plus environment wiring, while the JSONL request/response loop has a named adapter seam that later transports can mirror without reintroducing daemon lifecycle logic into the entrypoint.

That adapter loop is now a little thinner too. The outer loop still owns stream framing and lifetime, but one small helper now owns the "parse one JSONL request, run one daemon command, serialize one response" path. It is still synchronous and intentionally modest, but later foreground/socket/pipe adapters now have a cleaner seam above raw stdio framing.

There is now also a first explicit foreground request/response contract above that helper. The adapter no longer treats "one foreground admin request" as only a transient local combination of parsed JSON plus immediate writeback. It now has a named host-owned foreground request/result seam that can later be reused by a socket/pipe/HTTP adapter without first inheriting the stdio loop structure itself.

The daemon adapter is now also slightly less CLI-shaped in its host construction path. It still uses the existing store-opening helpers, but it no longer has to synthesize a temporary full CLI `args` object just to build runtime policy, runtime config, orchestration config, or the resident session-host contract.

That cleanup now extends one step further down the daemon path. The daemon runtime no longer synthesizes temporary CLI-style `args` just to open stores, build resource-store config, or resolve provider-backed tooling. Memory/plan store selection now follows the same host-owned backend/path values directly, including the old `auto` resolution rules, while tooling resolution receives a host-built tool-selection request instead of a CLI object.

That service layer now understands a slightly broader host-oriented command surface:

- `run_turn`
- `cancel_turn`
- `status`
- `reset_session`
- `close_session`
- `shutdown`

`status` reports a narrow readiness/liveness snapshot plus the currently tracked session keys and queued-command count. It now also exposes a few small lifecycle signals from the dispatcher itself, such as whether the worker thread is running, whether the daemon is still accepting new commands, whether shutdown has been requested, and the current queue capacity. `reset_session` and `close_session` go through the same keyed session manager as ordinary turns, which gives the admin/test path an explicit place to manage resident session state before a fuller queued daemon lifecycle exists.

That lifecycle surface is now also enforced inside the service itself. Once shutdown or draining has been requested, `run_turn` is rejected with a host-owned lifecycle error instead of relying only on the dispatcher's outer acceptance window. That closes the small gap where a late turn could otherwise slip in after shutdown had conceptually started but before the queue had fully stopped accepting work.

The dispatcher/protocol path now also carries a small status snapshot on non-status responses, including lifecycle replies such as `shutdown`. That means the current foreground/admin path can observe `draining` directly from the shutdown response instead of only inferring it later from booleans or stderr timing.

`cancel_turn` now exists as a first dispatcher-level contract, and the daemon result contract distinguishes two cases explicitly: queued-turn cancellation succeeds and emits a `turn.cancelled` daemon event, while cancellation against the currently active turn now records a host-owned cancel request against that turn's execution-control state and returns `turn_cancel_requested` plus the active request/turn identity. That is still intentionally narrow. The current runtime/inference/tool stack does not yet have a full end-to-end safe abort path, but the daemon/session seam now has one shared place for cancellation identity, turn deadlines, and later timeout propagation.

One concrete "full current functionality" foreground run looks like this on Windows/PowerShell:

```powershell
@'
{
  "schema_version": 1,
  "model": {
    "backend": "server-context",
    "path": "C:\\Users\\kalld\\models\\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "embedding_model": "C:\\Users\\kalld\\models\\nomic-embed-text-v1.5.Q4_K_M.gguf"
  },
  "runtime": {
    "planning_mode": "mini",
    "reflection_mode": "always",
    "memory_learn": "post-turn",
    "agent_plan": "auto",
    "n_predict": 96
  },
  "stores": {
    "memory": {
      "backend": "cozo",
      "path": ".\\work\\agent-memory.cozo"
    },
    "plan": {
      "backend": "cozo",
      "path": ".\\work\\agent-plan.cozo"
    }
  },
  "resources": {
    "blob_backend": "fs",
    "blob_root": ".\\work\\agent-resources",
    "metadata_backend": "cozo",
    "metadata_db": ".\\work\\agent-resources.cozo"
  },
  "tools": {
    "profile": "minimal",
    "repository_root": "C:\\Users\\kalld\\Documents\\Codex\\llama-dyn",
    "providers": [
      {
        "type": "mcp",
        "id": "local-mcp",
        "enabled": true,
        "transport": "stdio",
        "command": [
          ".\\build-plan-resident-cozo-debug-3\\bin\\Release\\llama-agent-mcp-stdio-fake-server.exe"
        ],
        "prefix": "local",
        "server_name": "local"
      }
    ]
  },
  "limits": {
    "queue_capacity": 8,
    "max_tool_rounds": 2,
    "max_turn_seconds": 120,
    "turn_timeout_ms": 120000,
    "inference_step_timeout_ms": 30000,
    "tool_timeout_ms": 5000,
    "mcp_connect_timeout_ms": 3000,
    "mcp_request_timeout_ms": 10000,
    "mcp_shutdown_timeout_ms": 1000
  }
}
'@ | Set-Content .\work\agent-host.json -Encoding utf8

@(
  '{"request_id":"status-1","command":"status"}',
  '{"request_id":"turn-1","mode":"mini","prompt":"Plan how to inspect the repository tooling path.","session_id":"demo-session","namespace_id":"local","project_id":"llama-dyn","memory_scope":"project","plan_scope":"project"}',
  '{"request_id":"shutdown-1","command":"shutdown"}'
) | Set-Content .\work\agent-requests.jsonl -Encoding ascii

Get-Content .\work\agent-requests.jsonl |
  .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe --config .\work\agent-host.json
```

That example exercises the current end-to-end foreground daemon shape: resident `server-context` inference, Cozo-backed memory/plan stores, filesystem+Cozo resource storage, mini planning with reflection and memory learning, repository/native tools, and one MCP stdio provider under the same host-owned config.

On top of that, the CLI now has two thin child-process adapters. `daemon-chat` starts the foreground daemon, sends one turn, reads one response, and shuts the child down. `daemon-session` keeps the same foreground child alive across multiple prompts in the same admin/test session. Both paths still go through the same runtime request/result contracts rather than delegating multi-turn state to a backend conversation loop, and the CLI reads protocol from stdout while relaying daemon diagnostics from stderr separately.

That CLI session path is now a little less turn-only as well. The child-process adapter has explicit request helpers for daemon `status`, `reset_session`, `close_session`, and `shutdown`, and `daemon-session` exposes a small admin/test command set over stdin: `/help`, `/status`, `/reset`, `/close`, and `/quit`. The implementation also normalizes Windows-style stdin a bit more carefully, including a first-line UTF-8 BOM edge that showed up in PowerShell piping during smoke verification.

The CLI-side diagnostics seam is a little cleaner now too. The child-process adapter still reads daemon stderr separately from protocol stdout, but it suppresses the high-volume routine model/bootstrap chatter in the ordinary admin/test path and only forwards warnings, errors, and unexpected lines by default. That keeps `daemon-chat` and `daemon-session` usable as foreground integration tools without making the protocol consumer scrape through several hundred lines of model-loader noise. For deeper debugging, the adapter still becomes more permissive when agent tracing is enabled.

That foreground client path is now also slightly less ad hoc internally. It has a tiny JSONL transport wrapper around the child-process stdio pipes, and the admin/test command handlers no longer peel turn/session/status results straight out of raw `ordered_json` with repeated `response.value(...)` calls. Instead they consume a few small protocol-shaped result parsers for turn, status, and lifecycle/session events while still speaking the same external JSONL wire format.

The `daemon-session` admin/test surface is now a little friendlier too. Its `/status` command no longer echoes the raw daemon JSON object back to stdout; it renders one compact typed summary of lifecycle state, queue health, active work, and bound sessions while still relying on the same parsed JSONL status contract underneath.

That rendering is now its own small CLI-facing seam rather than another helper hidden in the wire-protocol file. The JSONL parser still owns the transport/status DTOs, while the foreground client owns the tiny status-summary contract and rendering policy that turns those DTOs into a stable human-facing admin/test line.

The same client path now also uses the lifecycle snapshot actively instead of only carrying it through the protocol. `reset`, `close`, and `shutdown` now parse lifecycle responses through the richer DTO and use the embedded state snapshot for rendering and shutdown validation rather than treating those replies as event strings alone.

Those child-process adapters now also pass through the same daemon-owned tool configuration surface as the direct JSONL admin/test path: `--tool-profile`, `--repository-root`, and `--mcp-tool-command` all reach the foreground daemon when present. The chat-oriented daemon client path also now defaults its plan scope more conservatively when planning is off, so a simple session- or project-scoped admin/test chat turn does not accidentally force a synthetic turn-scoped contract.

The daemon-facing request shape now carries host-owned scope data such as namespace, session, project, memory scope and plan scope. The current session manager treats namespace plus session as the live resident lane, while status responses still report the currently bound project/scope for that lane. That is still intentionally modest: it is enough to drive multi-turn resident smoke and integration tests, while keeping the future service-owned session model explicit.

There is now also a focused smoke for that foreground client seam itself: `scripts/test-agent-daemon-client-clean-io-smoke.ps1` checks that `daemon-session` keeps protocol-oriented output on stdout, keeps routine loader/bootstrap chatter off stderr in the ordinary path, preserves the small admin/test command surface (`/help`, `/status`, `/reset`, `/close`, `/quit`), and renders the typed `/status` summary instead of raw JSON.

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

## Remaining Informal JSON Surfaces

The recent contract work removed several of the highest-friction JSON seams, but a number of important "still mostly implicit" JSON shapes remain. The most relevant near-term backlog looks roughly like this:

1. `common/agent/agent-runtime.cpp`

   This file is much cleaner now: request-tool safe defaults plus the bounded user-correction, failure-observation, reflection-hint, and reasoning-observation payload seams all go through named runtime JSON helpers. The remaining runtime-core cleanup is more about reducing mixed responsibilities than about raw JSON literals.

2. `pocs/agent/agent-cli-selection.cpp`

   This seam is narrower now too: selection schemas are requested through named schema-string helpers, and blueprint-binding tool arguments travel as a named plan tool-arguments contract instead of raw nested JSON until the final plan serializer step. The remaining work there is mostly around separating more assembly responsibility out of the CLI adapter, not about ad hoc nested selection payloads.

3. `pocs/agent/agent-daemon.cpp` plus `pocs/agent/agent-daemon-client.cpp`

   The daemon wire payload and the JSONL transport framing are now both explicit seams. The daemon protocol still owns the command/response objects, while the transport endpoints now reuse one small JSONL helper for line-oriented parsing and emission instead of open-coding request/shutdown JSON in each caller.

4. `pocs/agent/agent-tool-provider.cpp`

   The provider/view boundary is structurally much better now, and the final success/failure payload shaping has been pulled behind named helper contracts. The remaining work here is mostly around richer typed normalization and eventual schema validation parity for MCP, rather than hand-built wrapper JSON at each provider return site.

5. `common/memory/memory-tool-service.cpp`

   Memory tool `search` and `remember` now parse through named argument-contract helpers before the service logic runs. The follow-up work here is mostly about extending that pattern to any additional memory tool surfaces rather than first extraction.

6. `common/plan/cozo/plan-cozo.cpp`

   Plan persistence and event rows still serialize and deserialize larger JSON payloads inline for Cozo storage. Some of this is legitimate persistence encoding, but the stored row/document shapes are still only implicit.

7. `common/memory/cozo/memory-cozo.cpp`

   The same pattern exists on the memory side: several metadata/result/row shapes are persisted and reloaded as raw JSON without a named storage contract layer.

8. `common/chat.cpp`

   This file intentionally owns a large amount of OpenAI-compatible chat/tool JSON mapping, so not all JSON use here is debt. Even so, there are still some model-facing parsed/normalized shapes here that could benefit from more explicit contract helpers over time, especially where argument strings are parsed and re-emitted.

9. `pocs/agent/agent-server-inference.cpp`

   A few resident `server_context` result paths still parse response JSON blobs directly to interpret structured output. The scope is smaller than the files above, but it is still a live runtime seam.

If we count "production-relevant, non-test JSON shapes that are still at least partly informal", there are about 8 to 9 meaningful clusters left. If we count every single `.dump()` or `parse()` site in persistence, chat compatibility, diagnostics, and tests, the raw number is much larger. The clustered backlog above is the useful planning unit.

### Tools

Tools currently have three layers:

- Catalog: declares versioned metadata and profile membership.
- Registry: owns executable handlers.
- Adapter bindings: bind catalog definitions to local runtime resources such as memory store, plan id, repository root, and embedding provider.

There is now also a small provider/view boundary above those native pieces:

- `agent_tool_provider`: resolve tools for one host-owned runtime context.
- `agent_tool_view`: expose model-facing `common_chat_tool` values and execute one validated tool call.

The first implementation started native-only, and it still uses the existing catalog, registry, and adapter bindings underneath. The chat runtime no longer dispatches profile tools directly through a runtime-owned registry pointer. Instead, the host resolves a policy-bound, scope-bound `agent_tool_view`, passes `chat_tools()` into generation, and routes parsed assistant tool calls back through that view.

The embedding callback that sits underneath memory-oriented native tools is also one step less CLI-bound now. There is still a CLI wrapper for convenience, but the underlying helper can now work from a host-owned model path plus `n_gpu_layers`, which lets the daemon and real MCP stdio server construct their own embedding providers without carrying full CLI option objects into that path.

That provider result path now also carries host-owned resource refs for native tools, not only MCP-shaped ones. In the first concrete slice, larger `web_search` payloads can now be externalized into the configured resource store and returned as `resources` alongside a shorter inline result. The full search payload remains host-addressable by resource URI, while the inline tool result can stay bounded for the model.

That evidence path now continues one step further into planning state. Tool observations can now retain resolved `resource_refs` alongside their bounded inline summary, and the read-only `resource_read` native tool gives later steps a host-owned way to load the deferred payload back by opaque URI instead of forcing the earlier tool to keep everything inline.

There is now also a first MCP-shaped provider slice beside the native one. `mcp_agent_tool_provider` sits behind a narrow `agent_mcp_tool_client` interface, resolves model-visible `common_chat_tool` values from listed MCP-style tool definitions, namespaces exposed tool names, applies the same host-owned policy gate at exposure time, and normalizes tool-call results back into the shared `agent_tool_result` shape.

That seam now has two concrete test paths:

- an in-process fake MCP client used to exercise provider filtering and result normalization
- a first stdio-based MCP client adapter that speaks JSON-RPC-style `Content-Length` framed messages to a subprocess

The stdio client is still deliberately small. It is enough to prove the provider boundary through a real child process with `initialize`, `tools/list`, `tools/call`, `resources/list`, and `resources/read`, but it is not yet a production MCP lifecycle: there is still no reconnect logic, approval model, streaming event path, or broader capability surface.

On the server side, the subprocess path is now also a little more explicit. There is a small reusable stdio MCP server core in the PoC layer: JSON-RPC framing helpers, a tiny server-side tool registry, and a stdio server loop that dispatches `initialize`, `tools/list`, `tools/call`, `shutdown`, and `exit`. The older fake subprocess now reuses that same core, and the first real PoC MCP stdio server binary now exports a host-resolved tool surface through the same catalog/provider/bindings path used by the host runtime. That keeps the current subprocess path from drifting into a second ad hoc server shape while still staying much smaller than a full agent daemon or broader MCP host surface.

That real MCP stdio server can now also be bootstrapped through the same `--config PATH` host-config entrypoint as the daemon. In practice that means the tool export seam is no longer tied only to per-process CLI flags; it can already be described through a shared host-owned config file, including enabled stdio MCP subprocess providers, even though the current server still exports tools rather than a full resident agent runtime.

Its internal tool assembly is now also closer to the rest of the host/runtime stack. The server no longer hand-builds a separate native provider wiring path for export; instead it resolves its tool surface through the same host-owned tool-selection seam used by the CLI/daemon side, while still applying its own export policy to keep the MCP-visible tool surface intentionally narrower than a fully bound local runtime.

That current subprocess story now has two intentionally different shapes:

- fake/smoke MCP server: small hand-authored stub tools used to exercise protocol, namespacing, error mapping and resource-link normalization
- real MCP stdio server: a host-resolved tool surface exported through the same provider/bindings path the host runtime already uses

The important implication is that "available through MCP" now means "available through the real stdio server's resolved tool surface and bindings", not "everything mentioned anywhere in MCP smokes". The fake server remains a protocol/regression harness, not the exported host surface.

### MCP Export Surface Today

The current real MCP stdio server exports a host-resolved tool surface, not the whole agent runtime loop. In practice that means native tools from the selected profile can appear there, and configured external stdio MCP providers can also be forwarded into that same exported surface when the host config enables them.

| Capability | Available through real MCP stdio server | What it requires today |
| --- | --- | --- |
| `calculator`, `time_now` | Yes | `--tool-profile minimal` or any broader native profile |
| Repository tools such as `repository_list`, `repository_search`, `repository_read`, `repository_diff`, `repository_log` | Yes | A profile that includes them, typically `research`, plus `--repository-root PATH` |
| Web tools such as `web_search` and `web_fetch` | Yes | A profile that includes them, currently `research`; host policy still decides whether network tools are exposed |
| `resource_read` | Yes | A profile that includes it, such as `memory-read`, `memory`, or `research`; the server always opens a host-owned resource store |
| `resources/list` and `resources/read` | Yes | The real MCP stdio server owns a scoped host resource store and exposes host-authorized resource descriptors and reads through the MCP resource capability |
| Memory read tools such as `memory_search`, `memory_get`, `memory_inspect`, `memory_conflict_check` | Yes, when bound | A profile that includes them plus any host-owned memory store, including the default in-memory store or an explicit persistent backend such as `--memory-db PATH` |
| External MCP subprocess tools such as prefixed `github_search_issues` | Yes, when configured | A host config or equivalent host-owned request that enables stdio MCP subprocess providers with a prefix and command |
| Memory proposal tools such as `memory_remember`, `memory_propose_update`, `memory_propose_forget`, `memory_link`, `memory_compact_propose` | Yes, when bound and allowed | A profile such as `memory` or `research`, a host-owned memory store, and host policy that allows proposal-style writes |
| Planning tools such as `plan_get` and `plan_propose` | Partly | A profile that includes them plus a real plan store; `plan_get` is only meaningful when a bound `--plan-id ID` exists |
| Resource refs returned from native tools | Yes | The underlying native tool must materialize them through the host-owned resource store; `web_search`, `web_fetch` and `resource_read` are the first concrete examples |
| Full mini/planning runtime, reflection loop, memory learning loop | No | Those are runtime behaviors, not MCP tools in the current slice |
| Fake server tools such as `search_issues`, `search_recent_failures`, `create_issue` | No, not from the real server | Those remain test-only tools from the fake MCP server path |

This is also the cleanest way to think about memory, planning and resources through MCP right now:

- memory can be exported as native MCP-visible tools when the server process has any host-owned memory store, including the default in-memory store
- planning can expose its native plan tools, but that is not the same thing as exporting the whole mini/planning runtime as an MCP tool surface
- resources are the strongest fit so far because the real server always owns the resource store contract, can now expose `resources/list` and `resources/read`, and native tools can already return opaque resource refs for deferred payloads

So the current MCP stdio server is best understood as "native tool export through an MCP transport seam", not yet as "the agent runtime itself exposed as an MCP server".

Even in this small slice, the stdio client now does a little more than the original smoke seam: it switches Windows stdio pipes to binary framing for `Content-Length` transport, attempts a best-effort `shutdown` plus `exit` sequence before tearing down the child process, and can map structured MCP-side tool error metadata back into the shared failure contract used by native tools.

The stdio client now also keeps a small stderr tail from the MCP child and appends it to transport-level failures when the subprocess exits early or emits malformed JSON-RPC data. That keeps the first real subprocess smoke debuggable without pulling in a larger async lifecycle or logging subsystem yet.

There is now also a first thin integration into the ordinary CLI host-adapter path. When `llama-agent run` is given `--mcp-tool-command`, the CLI tool-selection layer resolves an MCP-backed `agent_tool_view` through the same host/runtime tooling contract used by native tools. This first slice is intentionally narrow: it covers the direct CLI host/runtime path, not the daemon adapters yet.

That CLI path can now also compose native and MCP-backed tools in the same resolved view. If both `--tool-profile` and `--mcp-tool-command` are present, the host resolves them through one small composite provider surface and keeps model-visible tool names unique at the merge boundary.

The subprocess-based MCP smokes now also declare their helper binaries explicitly at the build layer. Targets that launch `llama-agent-mcp-stdio-fake-server` or the real `llama-agent-mcp-stdio-server` now depend on those helper executables directly, which avoids a subtle stale-binary failure mode where a smoke would exercise an older helper build and appear to hang or regress in protocol behavior even though the caller target itself had rebuilt cleanly.

The foreground daemon can now also resolve the same native and MCP-backed tool surface that the CLI host adapter already uses. Session-host execution now has a per-turn tooling resolver hook, and the daemon uses it to rebuild tooling from the current host-owned session/scope turn contract before each turn. In practice that means `--tool-profile`, `--repository-root`, and `--mcp-tool-command` can now participate in the same resolved daemon tool view instead of the daemon having a narrower MCP-only wiring path.

The modern CLI profile-tool path also no longer builds a second parallel native registry just to derive model-facing tools. For profile-driven tools it now resolves one provider view and reuses that single host-owned surface for exposure and execution wiring.

That keeps host authority in one place:

- the model sees only `common_chat_tool`
- the model requests tools only through parsed `tool_calls`
- the runtime owns bindings, scope, repository root, memory authority, and policy
- disallowed native tools are filtered before exposure instead of being shown and rejected later

The first mini/planning migration is now also in place for blueprint binding. Auto-selected blueprint bindings no longer inspect the native registry directly; they validate against the resolved `agent_tool_view`, which means blueprint-binding now follows the same host-owned exposure and read-only policy surface as chat tool dispatch.

The planning/orchestration edge is also a little less ad hoc now. Plan auto-selection and blueprint auto-selection no longer take a longer loose parameter list for inference, generation config, scope, plan state and tool-view details separately. Instead they can be driven from one small orchestration runtime context that carries the current host-owned planning inputs plus the optional tooling bundle used for blueprint binding.

The older common agent runtime is now also structurally decoupled from the concrete registry type. Planned tool-step execution runs through a small `common_agent_tool_runtime` interface in `common/agent`, and the PoC runtime now adapts the resolved `agent_tool_view` into that interface through one shared provider-backed bridge. That means the common planning/runtime core no longer needs to know whether tool execution ultimately comes from native tools, a host-resolved provider view, or a future MCP-backed adapter.

What still remains is mostly coverage and convergence work rather than contract churn. The provider-backed path is now the intended modern shape; the next cleanup is to keep broadening smoke and integration coverage around the same interface as more mini/runtime and daemon flows exercise it.

There is now also a focused smoke for the planned-tool-step path itself. It runs a tiny `common_agent_runtime` scenario where the planner emits a calculator tool step and execution goes through a provider-backed tool-runtime adapter rather than a raw registry pointer. That gives the current refactor one concrete end-to-end proof point before broader resident/daemon mini smokes are added.

The resident/session-host layer has also been trimmed a bit further: it now carries `agent_tool_view` for the modern profile-tool path, but no longer keeps threading a separate `tool_registry` field through its own configs just to pass it onward unused. That makes the resident host/session contracts slightly closer to the intended provider-first shape.

The mini/runtime assembly path has also dropped its registry-backed tool-runtime fallback. Planned tool-step execution in that path now expects the modern provider-backed `agent_tool_view` when profile tools are active, and fails explicitly if a caller tries to run profile-tool planning without that resolved view. That narrows the remaining legacy surface and keeps the planned-tool runtime aligned with the provider-first direction already covered by smoke tests.

The host/chat contracts have now been trimmed in the same direction. The runtime host, chat driver, CLI host adapter and resident host path no longer carry a parallel `tool_registry` field through their modern provider-backed contracts. Chat dispatch itself also no longer carries a separate legacy tool-handler branch. Modern agent-side tool execution now goes through `agent_tool_view`, with the host deciding which scoped provider view to resolve for the turn.

The first shared memory-tool migration now sits underneath that provider path as well. Native `memory_search` and `memory_remember` execution both go through `common_memory_tool_service`, which keeps host-owned scope, embedding, store and policy bindings in one place while preserving the current synchronous behavior and result shapes.

The remaining CLI coupling around embeddings has also been narrowed further. The CLI path still supplies one concrete embedding implementation, but profile-tool bindings and runtime memory-learning hooks now reach it through a small `agent_embedding_provider` seam rather than capturing CLI options directly inside the native tool bindings.

Tool execution is synchronous in this slice. That is deliberate: it preserves current behavior while the runtime boundary stabilizes. A future worker model needs explicit semantics for cancellation, timeouts, ordering, result delivery, and shared-state access.

The cancellation/timeout seam now also reaches a little deeper into the runtime-owned tool and inference paths. Resolved `agent_tool_context` values now carry the shared execution-control contract, native and MCP-backed tool views fail early when a host cancellation or turn deadline has already fired, and the current runtime checks the same seam again immediately after synchronous tool execution. That is still cooperative rather than preemptive, but it means the provider-backed tool surface now has one host-owned place to report `tool_call_cancelled`, `tool_call_deadline_exceeded`, or timeout-class failures instead of only treating those conditions as outer daemon concerns.

The runtime now also starts mapping inference-step budgets into the existing generation request contract. CLI and resident/session-host assembly set `t_max_predict_ms` from `inference_step_timeout_ms` when configured, so the `server-context` path can begin enforcing bounded generation time through the same generation-options seam it already uses for `n_predict`. This is still only a first slice: active cancellation does not yet safely interrupt every in-flight inference or native tool body, and MCP stdio transport timeouts are not yet enforced as a fully isolated subprocess lifecycle.

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

That MCP-facing tool surface has now also been tightened slightly around naming and policy. Runtime filters treat the resolved model-visible name as the authority surface for exposed MCP tools, so prefixed MCP names are filtered the same way the model actually sees them rather than by an internal pre-prefix identifier.

One intentional gap remains in the current MCP tool path: validation currently enforces that tool arguments parse as a JSON object, but it does not yet validate those arguments against the advertised `inputSchema`. Native tools already get stronger contract validation through the registry path; fuller MCP schema validation is still an explicit follow-up item rather than something hidden behind the current PoC surface.

Resources and prompts can follow the same pattern later. They should not be added directly to the agent loop as transport-specific concepts.

## Service Direction

The next daemon-facing step should be designed as a host/service core that can later be run as a Unix-style service even though current development happens on Windows. That should affect the shape of the code now, but not force immediate platform-specific daemonization work.

The long-term target should look more like this:

```text
process host / transport adapter
        |
        +--> stdio foreground host
        +--> future unix-socket host
        +--> future named-pipe host
        +--> future HTTP host
                |
                v
agent daemon service
        |
        +--> lifecycle state
        +--> command ingress
        +--> scheduler / worker lanes
        +--> session manager
        +--> event/result sink
                |
                v
agent runtime host
        |
        +--> inference provider
        +--> tool provider
        +--> stores
        +--> policy and scope resolution
                |
                v
resident inference backend
        +--> local CLI generation adapter
        +--> server_context host
        +--> later remote or pooled backends
```

The important rule is that the service core should not know whether it is hosted from foreground stdio, a Unix service wrapper, a Windows process wrapper, or a later HTTP listener. Those are transport/process-host choices around the same command/session/runtime core, not separate daemon implementations.

`llama-server` remains useful inspiration here, but mostly for responsibility boundaries rather than for direct reuse of its public REST surface. In practice the reusable ideas are:

- transport threads should not run agent turns directly
- requests should become commands/tasks that move through a queue or scheduler
- readiness, drain and shutdown should be explicit service concerns
- inference/context ownership should stay inside the inference side of the host rather than leaking into transport code

For this branch that means `server_context` should be treated as an inference/backend building block under `llama-agent`, not as the outer service model. The agent daemon should own routing, session lifetime, tool policy and stores; the inference backend should own model/context execution.

### Service States

The service lifecycle should become explicit before new transports or real background hosting are added. A small target state model is:

- `starting`
- `ready`
- `draining`
- `stopping`
- `stopped`
- `failed`

The corresponding host-facing operations should stay equally small:

- `start()`
- `request_shutdown(drain|cancel)`
- `status()`
- `accepting_commands()`

Foreground stdio can keep using this lifecycle first. A later Unix-service or detached-process wrapper should only host the same service object and react to the same lifecycle/status contracts.

### Scheduler Direction

The daemon is still intentionally single-lane today, but the next design should already make room for asynchronous and multi-session execution.

The intended shape is:

```text
transport adapters
        |
        v
bounded command queue
        |
        v
scheduler / dispatcher
        |
        +--> worker lane 1
        +--> worker lane 2
        +--> ...
```

The first implementation does not need multiple workers yet. One worker lane remains the right small step. The important part is that queueing, scheduling and worker ownership become explicit, so later concurrency is an extension of the same contract rather than a rewrite.

That scheduler should eventually be able to enforce rules such as:

- different sessions may run in parallel when capacity exists
- the same session should normally serialize turns
- cancellation and deadlines should target queued or active commands by request/turn identity
- inference-capacity limits should be separate from logical session ownership

### Session and Lifetime Model

The long-term host model should keep three identities distinct:

- `namespace` as tenant/authority boundary
- `project` as longer-lived shared work container
- `session` as live runtime/conversation lane inside that project

That aligns with the current daemon/session direction, but it should become more explicit over time. A future multi-session host should be able to keep several sessions alive inside one project without treating each session as a separate model owner.

The same principle applies to runtime lifetime:

- process lifetime
- model lifetime
- inference-context lifetime
- agent-session lifetime
- turn lifetime

The current runtime/session split is already moving in that direction. The next cleanup should keep carrying that split upward so a host can unload one session, reset one context, or expire one lane without forcing a model reload.

### Beta-Oriented Step Plan

If the goal is a first beta that still feels structurally safe, the natural order is:

1. Keep the current foreground/stdin path as the reference host.

   It is already useful for smoke, integration and admin/test work. The goal is not to replace it yet, but to make it one transport/process-host around a cleaner service core.

2. Make the daemon service lifecycle explicit.

   Promote the current dispatcher/service shape into a clearer service object with lifecycle state, readiness, drain/shutdown intent and status reporting that does not depend on the JSONL transport.

3. Keep JSONL/stdin as a transport adapter, not as the service definition.

   The current JSONL protocol is still the right first transport. The next cleanup is to keep command parsing/emission outside the long-lived service core, so later Unix socket or HTTP adapters can reuse the same commands and results.

4. Keep the scheduler contract ahead of real concurrency.

   Preserve one worker lane first, but make queue, dispatcher and active-command ownership explicit enough that multi-worker scheduling can be added later without moving session logic again.

5. Harden keyed session ownership before detached hosting.

   Session reset/close/status already exist. The next step is to keep session state, project binding, policy-pack state and resident runtime reuse under the session manager instead of letting transport-facing code grow new side state.

6. Add a second process-host mode only after the core above is clear.

   A later detached process, Unix-service wrapper or socket listener should only host the same daemon service. It should not introduce a second runtime/session path.

### Minimum Prep Before A Beta

Before trying to present the daemon as a first beta, a few minimum seams should be in place so later async/multi-session/service work does not force a large redesign:

- one explicit service lifecycle/state contract above the current dispatcher
- one transport-neutral command/result surface
- one transport-neutral status/readiness surface
- one clear owner for live sessions and their resident runtime state
- one clear split between model lifetime, inference-context lifetime and session lifetime
- one bounded command queue with explicit capacity and shutdown behavior
- one place for cancellation/deadline identity, even if active-turn abort stays narrow at first
- one event/result seam that can later grow from bounded summaries into streaming without changing the service core

Those are the minimum structural preparations, not a request to build the full production service now. The point is to avoid creating a "foreground daemon architecture" and then later a separate "real service architecture". There should be one daemon/service core, with foreground stdio as the first host around it.

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

   Synchronous tools are still acceptable for the current slice. The daemon/session seam now has a shared execution-control contract plus active-turn cancellation requests and configured timeout budgets, but workers should still wait until those controls propagate all the way into inference, tool execution, MCP requests, retry policy, ordering, and richer failure reporting.

6. Split model lifetime, inference context lifetime and agent-session lifetime more explicitly.

   The current `server-context` path is a good resident smoke backend, but a real host should be able to keep models loaded while resetting or expiring individual agent sessions.

7. Extend the first stdio MCP client into a fuller MCP host/client path.

   The first provider slice now exists behind an abstract MCP client contract and already has a small stdio transport adapter with basic shutdown and structured tool-error mapping. The next step is to harden that path further: clearer diagnostics, less forceful teardown, better request/response correlation under notifications, and then broader MCP capability support. Add resources and prompts only after tool discovery, session ownership, and policy are stable.

## Backlog Notes

- Keep symbolic memory inside the existing memory/store model, then add overlays above it.

  The intended first symbolic slice is deliberately small:

  - continue using the current host-owned memory stores, scopes, provenance, proposal policy, MCP exposure, and retrieval path
  - treat `procedure`, `constraint`, and `decision` as the first symbolic memory kinds
  - keep inferred post-turn learning conservative at first; new symbolic kinds should start as explicit proposal-backed memory rather than immediate auto-learn targets
  - build later project/session overlays by compacting retrieved symbolic memories plus resource/evidence links instead of inventing a parallel persistence model

  The next follow-up after this kind-level slice is a typed overlay builder for planning, reflection, and reasoning. That overlay should assemble a bounded block such as constraints, decisions, procedures, relevant facts, and evidence/resource references from already authorized memory retrieval. In other words: memory remains the durable source of truth, while overlays become a host-owned context-shaping view of that memory.

  The current implementation is still intentionally conservative: the overlay is only a renderer over already retrieved memories, not a new retrieval path, scoring policy, or compaction store. The next useful step is therefore to make symbolic retrieval/selection a little smarter per stage, especially so planning and reflection can emphasize project-scoped constraints and decisions while reasoning can keep favoring procedure memories plus any directly relevant symbolic guidance.

  The current stage-aware selector is still only the first cut. It uses bounded kind-based weighting on the already retrieved hit set rather than a new retrieval pass. A later refinement can add project/session affinity, tool/step metadata boosts, evidence reuse hints, and more explicit cross-links, but that should stay above the same host-owned memory authority and below prompt rendering.

  The same principle applies to session policy. Blueprints are still the right inspiration for the declarative shape, but not the right runtime type to reuse directly. A later host/session layer can own stable policy packs per project or per resident lane, while retrieval overlays stay ephemeral and evidence-driven. That keeps executable plan structure, durable symbolic memory, and host/session policy related but distinct.

- Keep tightening the session-versus-project split above the current manager key.

  The current daemon/session layer now treats `namespace + session` as the resident lane key, while `project` remains part of the host-owned runtime scope bound within that lane. That is closer to the intended model, but it is still only a first cut: project-scoped memory and plans are still assembled through the same session-host contract, and there is not yet a richer host-owned project object above the resident runtime.

  In practice the intended direction is still the same: `namespace` stays the tenant/authority boundary, `project` is the longer-lived shared work container, and `session` is the shorter-lived live runtime/conversation lane inside that project. That becomes more important once MCP-facing host state, external tool providers, and richer multi-session project workflows sit above the current daemon/admin path.

- Revisit activity order after the current runtime/session cleanup wave.

  The branch has now completed several structural extractions in a row: provider-first tools, resident runtime/session layering, daemon service/dispatcher split, CLI-free runtime builders, and removal of older daemon/session aliases. Before taking another deep refactor in the same area, it is reasonable to pause and choose between a new activity track such as event/cancellation contracts, deeper `server-context` lifetime splitting, or MCP/provider-facing host work.

- Add host-owned resource references with turn/session/project lifetime.

  Large tool outputs do not need to stay inline forever. The first host-owned resource-store shape now exists with content-addressed filesystem blobs and a first Cozo-backed metadata option, and `web_search` is now the first native tool that can materialize a full result payload there while returning a shorter inline answer plus `resources`. The next real step is to use that same path more broadly from runtime/tool flows and to deepen the metadata side with TTL cleanup, provenance links, blob reference management and richer lookup operations. The intended lifetime split remains `turn` for short-lived tool artifacts, `session` for live working-set reuse across turns, and `project` for longer-lived shared artifacts tied to the work container rather than one conversation lane. As with memory and plan scope, the model should never choose arbitrary resource URIs or storage locations directly.

- Let planning and tool binding treat resource metadata as first-class evidence shape.

  The next planning-facing slice should not be "blob store first, meaning later". Resource metadata should become one of the ways observations explain what a deferred payload is for: purpose, short content summary, usage hint, limitations, and lightweight semantic tags such as keywords or entity names. That gives later planner/tool-binding steps a host-owned bridge between a compact inline observation and a richer off-context payload, and it sets up later cross-references such as `observation -> resource`, `resource -> derived observation`, or project-scoped semantic lookup without forcing the model to carry the full content inline.

- Add structured execution history that explains "why this answer" without debug logs.

  A later trace/event slice should make one turn inspectable through structured execution facts rather than C++ diagnostics. The useful shape is along the lines of `plan -> step -> observation -> tool result -> reflection decision -> final answer`, with enough host-owned metadata that an operator can understand what happened without exposing chain-of-thought.

- Add a file-backed host configuration model above CLI flags.

  CLI flags remain useful for PoC and admin overrides, but a near-production daemon should be constructible from a host config file that describes server settings, model load settings, profiles, storage backends, and safety/runtime limits. The runtime and daemon service layers should consume that host-owned configuration directly rather than depending on CLI `args`.

- Keep the daemon target shape centered on a resident runtime service, not a bigger CLI.

  The intended direction is still a host/service process that owns the loaded-model manager, session manager, store manager, tool execution surface, policy decisions, and trace sink, with the CLI eventually acting as a client/admin tool against that daemon rather than a parallel owner of the agent loop.

## Current Verification Baseline

The resident-inference branch has been validated with:

- `test-agent-inference`
- `test-tool-adapters`
- `llama-agent-tool-provider-smoke`
- `llama-agent-mcp-tool-provider-smoke`
- `llama-agent-mcp-stdio-client-smoke`, verifying the client/provider path against the reusable fake stdio MCP server core, including malformed `tools/list` diagnostics
- `llama-agent-mcp-stdio-server-smoke`, verifying the same client/provider path against the first real PoC stdio MCP server binary while exporting host-resolved tool surfaces such as `minimal`, `research`, and `research` plus configured external MCP subprocess tools
- MCP-related subprocess smokes now also rebuild their helper server targets explicitly before execution, which closes the stale-helper regression that previously surfaced as a misleading `resources/list` hang in the client smoke
- `llama-agent-tool-runtime-smoke`, verifying structured trace history across plan creation, tool execution, and final response completion
- `llama-agent-resource-store-smoke`, verifying the first host-owned resource/blob store contract for scoped reads, size limits, content-addressed filesystem blob reuse, and Cozo-backed resource metadata in a Cozo-enabled build
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
