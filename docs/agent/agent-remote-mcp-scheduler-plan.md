# Remote MCP and Global Inference Scheduler

This document records the next architecture phase after the resident runtime,
operation manager, MCP validation, stdio hard-timeout handling, and beta smoke
pack slices.

## Current implementation status

The current branch has:

- a host-owned MCP provider configuration model;
- a generation model with an optional separate embedding-model override;
  when omitted, the generation model is reused for embeddings;
- stdio MCP client/server support;
- MCP input-schema validation for native tools, outbound MCP providers, and
  inbound MCP server tool calls through the shared bounded validator;
- hard request timeout handling for a hanging stdio server;
- typed runtime operations with timeout, cancellation, polling and cleanup;
- session lanes and a daemon dispatcher with queued and active cancellation;
- an inbound Streamable HTTP MCP listener on the existing MCP server entrypoint,
  with local binding, bearer authentication, Origin checking, bounded request
  and response bodies, session IDs, and DELETE session cleanup;
- a caller-policy contract for opaque bearer tokens, including caller identity,
  audience, namespace, project, tool profile, tool allowlists and write authority;
- host-owned capability/profile resolution with an immutable resolved tool view,
  startup `tooling` diagnostics, and protocol rejection of client profile or
  write-authority overrides;
- a transport-neutral `execute_tool` daemon command and dispatcher bridge used
  by the inbound HTTP end-to-end smoke;
- daemon host assembly that starts inbound MCP HTTP beside the JSONL adapter and
  routes both transports through the same dispatcher/runtime tool executor,
- a transport-neutral JSONL stream seam with an optional authenticated TCP
  adapter for container/private-network administration,
- a POSIX Unix-socket JSONL adapter and systemd `Type=simple` service example
  for local background hosting,
  including the resolved tool-profile catalog and MCP boundary validation;
- a beta smoke pack covering the above.

This branch adds the configuration contract for remote providers and an
outbound HTTP client behind the existing MCP tool-client interface. It accepts
`url`, `token_env`, `allowed_tools`, transport-specific timeout values, and
`max_result_bytes`. The local HTTP smoke proves initialize, Bearer auth,
session-id retention, tool listing/call, resource listing/read, and bounded
JSON responses. HTTPS still requires an OpenSSL-enabled cpp-httplib build.
The current client uses configured connection/read timeouts and bounds those
values by the caller operation deadline for tool calls. Active cancellation
through the HTTP library is still a required follow-up. The inbound listener is now wired into the daemon
host and dispatcher. It derives caller identity, namespace, project, tool
profile, allowlists and write authority from the authenticated token profile
or JWT policy template, then projects that policy onto the host-resolved tool
catalog. Remaining remote-service work is limited to later hardening such as
full operation-deadline propagation, active cancellation through the HTTP
library, production TLS/service-host integration, and broader Streamable HTTP
streaming support.

## Configuration examples

The inbound endpoint also has an explicit opt-in agent-to-agent MCP surface
with `delegate_task`, `summarize`, and `review_plan`. These are host-adapter
tools, not additions to the native tool catalog: calls reuse the daemon
dispatcher/session manager and are constrained by the authenticated caller's
MCP allowlist. Chained delegation-depth propagation and a richer dedicated
delegation profile remain follow-up hardening.

The complete examples are kept as JSON files so they can be copied and adapted:

- [stdio host configuration](../examples/agent-host-config-stdio.json)
- [capability/profile host configuration](../examples/agent-host-config-capabilities.json)
- [remote Streamable HTTP host configuration](../examples/agent-host-config-remote-http.json)
- [inbound MCP home/single-secret configuration](../examples/agent-host-config-inbound-mcp-home.json)
- [inbound MCP multi-token configuration](../examples/agent-host-config-inbound-mcp-auth.json)
- [inbound MCP enterprise/JWT configuration](../examples/agent-host-config-inbound-mcp-jwt.json)
- [inbound MCP agent-to-agent configuration](../examples/agent-host-config-inbound-mcp-agent.json)
- [JSONL TCP administration configuration](../examples/agent-host-config-jsonl-tcp.json)
- [JSONL Unix-socket configuration](../examples/agent-host-config-jsonl-unix.json)
- [systemd service unit](../examples/llama-agent-daemon.service)
- [Visual Studio Code stdio MCP configuration](../examples/vscode-mcp.json)
- [Visual Studio Code inbound HTTP MCP configuration](../examples/vscode-mcp-inbound.json)

The mode and scheduler settings are shown explicitly in the examples:

| Use case | `default_mode` | `thinking_mode` | Tool profile | Example |
| --- | --- | --- | --- | --- |
| Local bounded agent | `agent` | `reflective` | `minimal` | [home/inbound MCP](../examples/agent-host-config-inbound-mcp-home.json) |
| Delegated agent work | `agent` | `deliberate` | `minimal` | [agent-to-agent MCP](../examples/agent-host-config-inbound-mcp-agent.json) |
| Local research provider | `agent` | `research` | `research` | [stdio MCP](../examples/agent-host-config-stdio.json) |
| Remote research provider | `agent` | `research` | `research` | [Streamable HTTP MCP](../examples/agent-host-config-remote-http.json) |
| JSONL administration | `agent` | `reflective` | `minimal` | [TCP](../examples/agent-host-config-jsonl-tcp.json) / [Unix](../examples/agent-host-config-jsonl-unix.json) |

All host examples use `limits.inference_max_active: 1` to make the initial
single-lane inference behavior explicit. `limits.worker_count` controls
daemon dispatch concurrency independently; increasing it does not enable
parallel model execution.

For the remote example, provide the token before starting the agent. The token
is intentionally referenced by environment name and is not stored in the JSON
file:

```powershell
$env:REMOTE_GITHUB_MCP_TOKEN = "replace-with-a-token"
llama-agent-daemon.exe --config docs/examples/agent-host-config-remote-http.json
```

The inbound HTTP listener can now be started by the daemon host itself. It
binds to localhost by default and requires an opaque bearer token supplied
through an environment variable. JSONL administration and HTTP tool calls
share the daemon dispatcher and worker pool:

```powershell
$env:LLAMA_AGENT_MCP_TOKEN = "replace-with-a-local-secret"
llama-agent-daemon.exe --model MODEL.gguf --tool-profile minimal `
  --http-listen 127.0.0.1 --http-port 8080 --http-path /mcp `
  --http-token-env LLAMA_AGENT_MCP_TOKEN --worker-count 2
```

If a browser-originated client is needed, configure one exact allowed origin:

```powershell
llama-agent-daemon.exe --model MODEL.gguf --http-listen 127.0.0.1 `
  --http-token-env LLAMA_AGENT_MCP_TOKEN `
  --http-allowed-origin http://localhost:3000
```

The remote example requires a server endpoint that supports MCP Streamable
HTTP. `streamable_http` is the preferred transport value; `http` and `https`
remain accepted aliases in the provider validator. Do not expose the token in
command-line arguments, URLs, logs, event payloads or checked-in config.

For the complete daemon startup, flag and JSONL command walkthrough, see
[agent-daemon-usage.md](agent-daemon-usage.md).

The intended deployment profiles are:

| Profile | Authentication | Typical policy |
| --- | --- | --- |
| Home | One opaque secret via token_env | Localhost, minimal, read-only |
| Lab | Multiple opaque secrets via mcp.inbound.tokens | Separate callers and tool profiles |
| Enterprise | JWT access tokens plus JWKS | External issuer, audience and scopes |
| Hybrid | Future composite authenticator | Local admin secret plus enterprise JWT |

Hybrid is a design target rather than a current configuration. The current
listener selects either opaque or jwt for one running endpoint.

## Standard transport and authorization

Remote MCP should use MCP Streamable HTTP: one MCP endpoint supporting POST,
with optional GET/SSE for server-to-client messages and optional
`Mcp-Session-Id` state. The client must send the negotiated
`MCP-Protocol-Version` on HTTP requests. The older HTTP+SSE transport should
only be retained as an explicit compatibility mode.

HTTP servers must validate Origin, bind locally by default, and require
authentication before being exposed beyond localhost. TLS termination should
initially live in a reverse proxy or service host rather than in the runtime
core.

The long-term interoperable auth profile is OAuth 2.1-style resource-server
authorization with Protected Resource Metadata, issuer/audience validation,
short-lived Bearer tokens, and scopes. A local beta may use a configured
opaque Bearer token, but it must go through the same authenticator interface so
it cannot become a permanent parallel auth model.

## Remote provider requirements

An enabled remote provider must define:

- `transport`: `streamable_http` (with `http`/`https` accepted as aliases);
- `url`: the canonical MCP endpoint;
- optional `token_env` for an access token supplied by the process environment;
- optional `allowed_tools` allowlist;
- connect, request, and shutdown deadlines;
- a maximum result size;
- an optional exposed-name prefix.

The HTTP client must preserve the existing MCP client/tool-provider contract.
Stdio and HTTP must be interchangeable below `agent_mcp_tool_client`; no
transport-specific logic belongs in the runtime session or scheduler.

The client must support:

- initialize/initialized negotiation;
- `MCP-Protocol-Version` and `Accept` headers;
- `Mcp-Session-Id` when returned by the server;
- JSON-RPC errors and HTTP status errors as typed tool failures;
- request deadline, cancellation and bounded response bodies;
- safe token handling with no token in URLs, logs, events or diagnostics;
- optional DELETE session cleanup.

## Inbound MCP server requirements

The agent-facing listener is a host adapter around the current MCP host and is
being moved toward the daemon/service,
not a second runtime. Its request path is:

```text
HTTP -> Origin check -> authentication -> caller policy
     -> namespace/session binding -> MCP validation
     -> daemon dispatcher -> session lane -> operation manager
```

The current beta authenticates opaque configured token profiles, binds the session to
the resulting caller policy, and tracks MCP session IDs. The caller policy is
projected onto the native tool surface: `allowed_tools` filters both
`tools/list` and `tools/call`, while `allow_writes=false` hides and rejects
non-read-only/confirmation-gated tools and is also passed into the daemon's
native tool view. The native profile remains the source of tool definitions and
execution bindings; MCP auth supplies caller identity and narrower authority.
The token profiles are now loaded from `mcp.inbound.tokens`; each profile uses
`token_env`, so raw bearer secrets remain outside the config file. The legacy
single-token CLI path remains available for compatibility.
The same inbound authorization block now accepts `mode: "jwt"` with
`issuer`, `audience`, `jwks_uri`, allowed algorithms, required scopes and a
native policy template. JWT access tokens are parsed and claim-checked, and
RS256 signatures are verified when the build has OpenSSL support enabled.
Without a crypto-enabled build, JWT requests are rejected explicitly; opaque
token mode remains usable.
The authenticated caller, not the request body, must determine the default
namespace, project and allowed tool profile. The daemon host profile is chosen
by configuration/startup and is not client-selectable. Public turn requests
that attempt to provide `tool_profile`, `allowed_tools`, `allow_writes` or
`enable_shell` are rejected; authenticated caller policy is only a narrowing
intersection. Every request must have bounded body/result sizes and a
deadline. Write-capable tools require the existing confirmation/policy path.

## Global session/inference scheduler

The model-residency part of this boundary is specified in
[Agent model residency and multi-model scheduling](agent-model-residency.md).
The scheduler plan owns queueing and inference admission; it must not become a
second owner of model handles or session conversation state.

The daemon dispatcher now has a shared, configurable worker pool. The
foreground JSONL host, inbound MCP HTTP host and future service hosts construct
the same dispatcher; host lifetime and transport must not decide worker
semantics. The default is one worker until a deployment opts into more.

Session ordering remains local to each session lane. A global scheduler should
control inference capacity, not own conversation state:

```text
daemon dispatcher
    -> session lane mailbox
    -> inference operation
    -> global inference scheduler
    -> model/backend pool
    -> GPU or CPU
```

The first scheduler version provides admission control, priority-aware fair
queueing, capacity limits, deadlines, cancellation, pending status and
inference admission events. It runs without batching and may run one job at a
time initially. This keeps scheduler correctness testable before backend slot
or batching support is introduced. Multi-model residency adds a separate
profile acquire/pin/release step around the complete turn; it does not add a
second turn queue.

Inference jobs should be backend-neutral and carry at least:

- operation/session/model/backend identity;
- prompt or decode step kind;
- KV/sequence-state handle;
- sampling profile;
- token limit;
- deadline and cancellation token;
- per-job progress/result channel.

Only compatible inference steps should be microbatched. Tool calls, planning,
resource writes and session transitions remain lane-local. Batching is an
optimization inside the inference backend, not a new session protocol.

## Verification activity: beta smoke pack

The beta smoke pack establishes evidence for the existing operation, transport,
policy and lane contracts before scheduler work begins.

The latest recorded Windows/Cozo verification run is documented in the
[Agent Assurance record](agent-assurance.md): the `agent` label passed 16/16
tests. The exact count belongs to the build and test configuration, so this
architecture document does not treat it as a permanent test-count contract.
The result covers the current contract, memory, plan, lifecycle and tooling
seams; it does not imply that the future worker-pool acceptance criteria are
complete.

The verification entry points are intentionally layered: CMake provides
`llama-agent-build-pack`, `llama-agent-ctest-pack`, and
`llama-agent-beta-test-pack`; the platform wrappers provide the process-level
PowerShell and Bash execution. The wrappers share suite names, timeout/failure
semantics, temporary-log cleanup, and machine-readable `suite=...` output.

When adding a new test or smoke, follow the checklist in
`agent-runtime.md` under “Adding a new test or smoke”. In particular, add
the target to its CMake category and to both platform beta runners; otherwise
the test can exist and build successfully while remaining absent from the
actual verification pack.

The smoke pack should cover:

1. a hanging MCP stdio provider, including deadline expiry, subprocess
   termination, reader-thread cleanup and daemon survival;
2. remote HTTP MCP initialize, discovery, tool call, timeout and result bounds;
3. inbound HTTP, TCP and Unix-socket authentication, scope binding, tool
   allowlists, read-only write rejection, admin-command authorization and
   credential non-leakage;
4. same-session ordering, cross-session parallelism, `running_with_waiters`,
   queued/active cancellation, reload and graceful shutdown;
5. operation-manager transitions, deadlines, events and cleanup/reap.

## Backlog activity: lane-aware worker pool

The worker-pool implementation should extend the current dispatcher and use
the following ownership model:

```text
request -> lane mailbox -> ready-lane queue -> bounded workers
                                      ^             |
                                      |             v
                              pending wakeup <- one lane step
```

Required invariants are one active turn per lane, parallel execution across
different lanes, no worker occupied by pending external I/O, fair lane
selection, prioritized cancellation/shutdown, bounded admission and explicit
capacity metrics. `worker_count` controls scheduler concurrency; GPU/CPU
inference capacity is a separate resource limit.

The first scheduler must run without batching. It is complete when fairness,
cross-session isolation, deadline expiry, cancellation, queue rejection,
capacity recovery and deterministic shutdown pass with batching disabled.
Microbatching is a later inference-backend activity and must not change lane
ownership or the daemon command protocol.

## Required implementation order

1. Transport-neutral MCP client seam.
2. Streamable HTTP outbound client and remote-provider smoke.
3. Inbound MCP HTTP endpoint on localhost.
4. Authenticator, caller identity, scope and tool-profile binding.
5. HTTPS/reverse-proxy deployment contract and remote-safe policy smoke.
6. Global inference admission scheduler without batching. Implemented in the agent runtime with host-owned capacity admission.
7. Scheduler fairness, timeout, cancellation and capacity smokes. Implemented with priority-aware FIFO admission, aging, cancellation and deadline cleanup.
8. Process-wide model residency manager and backend loaders with per-session sequence/KV state; this is the next model-serving phase.
9. Microbatch prototype and benchmark coverage.
10. Service host, metrics, audit and production lifecycle.

## Explicit non-goals for the first slices

- distributed scheduling across processes or machines;
- passing inbound caller tokens through to downstream MCP providers;
- making HTTP transport own session or conversation state;
- batching tool execution or complete agent turns;
- enabling public network binding without an explicit service/auth policy;
- dynamic OAuth client registration before the basic resource-server path is
  verified.

## Acceptance criteria

The remote MCP slice is ready when the beta smoke pack can prove successful
initialize/list-tools/call-tool/read-resource over HTTP, invalid token and
wrong-audience rejection, allowlist enforcement, response-limit enforcement,
request timeout, cancellation, session cleanup and no credential leakage.

The scheduler slice is ready when multiple session lanes demonstrate bounded
fairness, no cross-session state mixing, active cancellation, deadline expiry,
queue rejection, capacity recovery and deterministic shutdown. Batching is
ready only after those tests pass both with batching disabled and enabled.
