# Agent daemon: configuration and startup

The current daemon is a foreground JSONL process. It reads one JSON command
per line from stdin and writes one JSON response per line to stdout. The first
response is `ready`; diagnostics and warnings go to stderr. It can also host
the inbound MCP HTTP listener in the same foreground process. It is not yet a
detached Windows service or supervised production host.

The same JSONL protocol can optionally be served over TCP. TCP is a transport
adapter for the existing dispatcher, not a second daemon implementation. The
stdin/stdout mode remains the simplest local administration mode. A copyable
TCP configuration is [agent-host-config-jsonl-tcp.json](examples/agent-host-config-jsonl-tcp.json).

For a background Linux service, use the Unix domain socket configuration
[agent-host-config-jsonl-unix.json](examples/agent-host-config-jsonl-unix.json)
and the systemd example [llama-agent-daemon.service](examples/llama-agent-daemon.service).
The daemon remains a foreground process under systemd (`Type=simple`); systemd
owns restart, logging and stop timeout while the daemon owns runtime shutdown.
The socket file is created with the configured Unix mode, so its owner/group
provides an additional local access boundary. Authentication policy still
applies after the OS accepts the connection.

## Start from a host configuration file

The recommended starting point is a config file rather than a long command
line:

```powershell
$env:REMOTE_GITHUB_MCP_TOKEN = "replace-with-a-token"

& .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-remote-http.json
```

For a local stdio MCP provider, use
[agent-host-config-stdio.json](examples/agent-host-config-stdio.json). For a
remote provider, use
[agent-host-config-remote-http.json](examples/agent-host-config-remote-http.json).
For inbound MCP authentication, the home, lab and enterprise examples are
[agent-host-config-inbound-mcp-home.json](examples/agent-host-config-inbound-mcp-home.json),
[agent-host-config-inbound-mcp-auth.json](examples/agent-host-config-inbound-mcp-auth.json)
and [agent-host-config-inbound-mcp-jwt.json](examples/agent-host-config-inbound-mcp-jwt.json).
The model path and MCP command/URL in those files are examples and must be
changed for the local machine.

## Start with flags only

The equivalent minimal shape is:

```powershell
& .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe `
    --model C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf `
    --default-mode chat `
    --n-predict 64 `
    --n-gpu-layers 0 `
    --queue-capacity 8 `
    --max-turn-seconds 120
```

Useful overrides are:

| Area | Flags |
| --- | --- |
| model | `--model`, `--embedding-model`, `--n-predict`, `--n-gpu-layers` |
| runtime | `--default-mode`, `--planning-mode`, `--reflection-mode`, `--agent-plan`, `--agent-trace` |
| stores | `--backend`, `--memory-db`, `--plan-backend`, `--plan-db` |
| resources | `--resource-blob-backend`, `--resource-blob-root`, `--resource-metadata-backend`, `--resource-metadata-db` |
| tools | `--tool-profile`, `--repository-root`, `--mcp-tool-command`, `--mcp-tool-arg`, `--mcp-tool-server-name`, `--mcp-tool-prefix` |
| HTTP host | `--http-listen`, `--http-port`, `--http-path`, `--http-token-env`, `--http-allowed-origin`, `--http-agent-tools`, `--http-max-delegation-depth` |
| JSONL TCP host | `--tcp-listen`, `--tcp-port`, `--tcp-max-line-bytes` |
| JSONL Unix host | `--unix-socket`, `--unix-socket-mode` |
| workers/limits | `--worker-count`/`--workers`, `--queue-capacity`, `--max-turn-seconds`, `--max-tool-rounds` |

The config file is loaded first and explicit flags are the appropriate place
for a one-run override. Secrets should remain in environment variables, not
in flags or checked-in JSON.

## JSONL commands

The copyable request sequence is in
[agent-daemon-requests.jsonl](examples/agent-daemon-requests.jsonl). It
demonstrates status, one chat turn, and graceful shutdown.

```powershell
$requests = Get-Content .\docs\examples\agent-daemon-requests.jsonl
$requests | & .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-stdio.json
```

Important protocol behavior:

- keep stdout reserved for JSONL protocol messages;
- consume stderr separately for diagnostics;
- wait for `event: "ready"` before sending turns;
- use stable `session_id`, `namespace_id`, `project_id`, and `turn_id` values;
- send `{"command":"shutdown"}` for graceful foreground shutdown;
- do not treat this stdin/stdout process as a production network service yet.

Configuration can be reloaded by the local JSONL administration channel:

```json
{"request_id":"reload-1","command":"reload_config","path":"docs/examples/agent-host-config-stdio.json"}
```

The daemon validates the complete candidate configuration before applying it.
Timeouts, tool profile, repository root, bounded tool limits and MCP providers
can be applied to new operations. A repository root must resolve to a
directory. Provider IDs are stable: new IDs are added, missing IDs
are removed for future operations, and changed IDs are replaced. Existing
operations keep their provider clients until their tooling is destroyed.
Model/backend, stores, resource roots, worker/queue sizing and runtime
assembly are restart-required for now. A
rejected reload returns `event: "config.reload.rejected"`, a
`restart_required` array and a warning; it never partially applies a
candidate. In-flight operations retain their existing configuration snapshot.
The daemon stores the active configuration as a shared immutable snapshot;
reload publishes a new snapshot for subsequent JSONL, HTTP and tool-resolution
work without mutating the snapshot already observed by an operation.

`--worker-count N` enables a shared worker pool. The default remains `1` for
compatibility. Multiple workers may process different session lanes in
parallel, while the session manager keeps turns within one session ordered.
Workers are independent of the foreground/service-host choice: the current
inbound MCP HTTP host and future named-pipe or supervised hosts should
construct the same dispatcher with the same worker count.

## JSONL over TCP

Start the TCP adapter with the configuration example:

```powershell
$env:LLAMA_AGENT_TCP_READ_TOKEN = "replace-with-a-long-random-read-secret"
$env:LLAMA_AGENT_TCP_ADMIN_TOKEN = "replace-with-a-long-random-admin-secret"
& .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-jsonl-tcp.json
```

The first line after `ready` must authenticate the connection:

```json
{"authorization":"Bearer replace-with-a-long-random-read-secret"}
```

The daemon answers with a status response after successful authentication.
Subsequent lines use the same JSONL commands as stdin/stdout. The authenticated
caller policy supplies namespace and project binding; clients cannot override
those fields. `shutdown`, `drain` and `reload_config` additionally require
`allow_admin: true`. `allowed_tools` and `allow_writes` are projected into the
turn's tool view, so a read-only caller cannot use policy-gated writes. Multiple
TCP connections are handled concurrently and share the configured daemon worker
pool. Use a separate port from inbound MCP HTTP.

The current TCP adapter is intended for trusted container or private-network
deployment. It does not provide TLS; use a TLS sidecar or service mesh for
untrusted networks. TCP listener address, port and framing limits require a
restart. Authentication and policy remain connection-scoped.

Unix sockets are POSIX-only in this slice. Windows continues to use stdio or
TCP; named pipes can be added later using the same JSONL stream boundary.

The focused beta smoke can be run with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test-agent-daemon-beta-smoke.ps1 `
    -ChatModel C:\Users\you\models\model.gguf
```

It covers MCP/config policy projection, restricted versus admin TCP callers,
TCP status and graceful shutdown.

## Inbound MCP HTTP in the daemon host

The daemon can expose the inbound Streamable HTTP MCP endpoint while retaining
JSONL on stdin/stdout for local administration. Both transports use the same
dispatcher, worker pool and runtime tool executor:

For multiple inbound callers, use the host configuration example
[agent-host-config-inbound-mcp-auth.json](examples/agent-host-config-inbound-mcp-auth.json)
and export its `token_env` variables before starting the daemon:

```powershell
$env:LLAMA_AGENT_READ_TOKEN = "replace-with-read-token"
$env:LLAMA_AGENT_ADMIN_TOKEN = "replace-with-admin-token"
llama-agent-daemon.exe --config docs/examples/agent-host-config-inbound-mcp-auth.json
```

For a single home/lab caller:

```powershell
$env:LLAMA_AGENT_HOME_TOKEN = "replace-with-a-long-random-secret"
llama-agent-daemon.exe --config docs/examples/agent-host-config-inbound-mcp-home.json
```

For enterprise JWT, replace the issuer, audience and JWKS URL in
[agent-host-config-inbound-mcp-jwt.json](examples/agent-host-config-inbound-mcp-jwt.json).
The authorization server issues the access token; the daemon only verifies it
and maps it to the configured native policy. Use an OpenSSL-enabled build.

```powershell
$env:LLAMA_AGENT_MCP_TOKEN = "replace-with-a-local-secret"

& .\build-plan-resident-cozo-debug-3\bin\Release\llama-agent-daemon.exe `
    --model C:\Users\kalld\models\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf `
    --tool-profile minimal `
    --worker-count 2 `
    --http-listen 127.0.0.1 `
    --http-port 8080 `
    --http-path /mcp `
    --http-token-env LLAMA_AGENT_MCP_TOKEN
```

The HTTP listener is stopped as part of foreground shutdown. Keep the default
local binding until a service host supplies TLS termination and a production
authentication/policy configuration. The daemon projects the resolved
tool-profile catalog for `tools/list` and validates tool names and input
schemas before dispatching calls. The authenticated caller policy narrows that
surface further: `allowed_tools` applies to both listing and calls, and
`allow_writes: false` removes confirmation/write tools and is propagated to the
native runtime gate. A successful provider reload atomically replaces this
catalog for new HTTP requests; existing requests keep their registry/policy
snapshot. Configuration-driven token profiles are supported through
`mcp.inbound.tokens`. JWT mode is available through
`mcp.inbound.authorization` with `issuer`, `audience`, `jwks_uri`,
`required_scopes`, `allowed_algorithms` and a native `tool_profile`. It
requires an OpenSSL-enabled build for signature verification; otherwise the
daemon rejects JWT authentication rather than accepting an unverified token.
The current listener selects one authorization mode at a time. A future
composite authenticator may combine a local admin secret with enterprise JWT;
that hybrid configuration is intentionally not presented as a working JSON
example yet.

### Agent-to-agent MCP (explicit opt-in)

The daemon can expose a small agent-to-agent MCP surface with
`--http-agent-tools`. It is separate from the native tool profiles and is
never enabled by default:

```powershell
llama-agent-daemon.exe --model MODEL.gguf --tool-profile minimal `
  --http-listen 127.0.0.1 --http-port 8080 `
  --http-token-env LLAMA_AGENT_MCP_TOKEN --http-agent-tools `
  --http-max-delegation-depth 1
```

The first bounded surface contains `delegate_task`, `summarize` and
`review_plan`. Calls reuse the existing daemon dispatcher and session manager,
inherit the authenticated caller namespace/project, and use the existing
timeout and result contracts. `allowed_tools` still applies to these names.
Propagation of depth across chained outbound MCP clients, richer delegation
profiles and dedicated audit events remain follow-up work.
The copyable JSON variant is
[agent-host-config-inbound-mcp-agent.json](examples/agent-host-config-inbound-mcp-agent.json).

## Daemon lifecycle

```text
starting -> ready -> draining -> stopping -> stopped
                         \-> failed
```

`status` exposes readiness, liveness, queue capacity, active turn and session
lane information. A later service host may wrap the same daemon service and
dispatcher contracts with HTTP, named pipes or a supervisor without changing
the JSON/runtime ownership model.

Every status-shaped JSONL response also contains a small `metrics` object with
dispatcher counters: `commands_accepted`, `commands_completed`,
`commands_failed`, `turns_completed`, and `tools_completed`. This is the first
host metrics slice; a later service host can project the same counters to
Prometheus text without making metrics an MCP tool.

## Current limitations

- foreground process only;
- stdin/stdout remains the local administrative transport; JSONL TCP is
  available as an explicitly enabled transport adapter;
- no automatic restart or supervisor;
- config reload is currently available only through local JSONL administration;
- config reload does not restart the process or rebuild model, stores, workers,
  HTTP listeners or already-running MCP provider clients;
- HTTPS support depends on an OpenSSL-enabled build for the current HTTP
  client/listener path; TLS termination is not part of the foreground daemon.
- Unix socket mode is a foreground service transport, not an internal
  daemonization mechanism; systemd or another supervisor owns process lifetime.
