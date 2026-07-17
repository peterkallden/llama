# Remote MCP and Global Inference Scheduler

This document records the next architecture phase after the resident runtime,
operation manager, MCP validation, stdio hard-timeout handling, and beta smoke
pack slices.

## Current implementation status

The current branch has:

- a host-owned MCP provider configuration model;
- stdio MCP client/server support;
- MCP input-schema validation;
- hard request timeout handling for a hanging stdio server;
- typed runtime operations with timeout, cancellation, polling and cleanup;
- session lanes and a daemon dispatcher with queued and active cancellation;
- a beta smoke pack covering the above.

This branch adds the configuration contract for remote providers and an
outbound HTTP client behind the existing MCP tool-client interface. It accepts
`url`, `token_env`, `allowed_tools`, transport-specific timeout values, and
`max_result_bytes`. The local HTTP smoke proves initialize, Bearer auth,
session-id retention, tool listing/call, resource listing/read, and bounded
JSON responses. HTTPS still requires an OpenSSL-enabled cpp-httplib build.
The current client uses configured connection/read timeouts; propagation of the
caller operation deadline and active cancellation through the HTTP library is
still a required follow-up. The inbound HTTP listener is deliberately not
claimed complete yet.

## Configuration examples

The complete examples are kept as JSON files so they can be copied and adapted:

- [stdio host configuration](examples/agent-host-config-stdio.json)
- [remote Streamable HTTP host configuration](examples/agent-host-config-remote-http.json)

For the remote example, provide the token before starting the agent. The token
is intentionally referenced by environment name and is not stored in the JSON
file:

```powershell
$env:REMOTE_GITHUB_MCP_TOKEN = "replace-with-a-token"
llama-agent-daemon.exe --config docs/examples/agent-host-config-remote-http.json
```

The remote example requires a server endpoint that supports MCP Streamable
HTTP. `streamable_http` is the preferred transport value; `http` and `https`
remain accepted aliases in the provider validator. Do not expose the token in
command-line arguments, URLs, logs, event payloads or checked-in config.

For the complete daemon startup, flag and JSONL command walkthrough, see
[agent-daemon-usage.md](agent-daemon-usage.md).

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

The agent-facing listener should be a host adapter around the daemon/service,
not a second runtime. Its request path is:

```text
HTTP -> Origin check -> authentication -> caller policy
     -> namespace/session binding -> MCP validation
     -> daemon dispatcher -> session lane -> operation manager
```

The authenticated caller, not the request body, must determine the default
namespace, project and allowed tool profile. Every request must have bounded
body/result sizes and a deadline. Write-capable tools require the existing
confirmation/policy path.

## Global session/inference scheduler

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

The first scheduler version should provide admission control, fair queueing,
capacity limits, deadlines, cancellation, pending status and metrics. It may
run one job at a time initially. This makes scheduler correctness testable
before batching is introduced.

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

## Required implementation order

1. Transport-neutral MCP client seam.
2. Streamable HTTP outbound client and remote-provider smoke.
3. Inbound MCP HTTP endpoint on localhost.
4. Authenticator, caller identity, scope and tool-profile binding.
5. HTTPS/reverse-proxy deployment contract and remote-safe policy smoke.
6. Global inference admission scheduler without batching.
7. Scheduler fairness, timeout, cancellation and capacity smokes.
8. Inference backend adapter with per-job sequence/KV state.
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
