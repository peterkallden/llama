# Agent daemon: configuration and startup

The current daemon is a foreground JSONL process. It reads one JSON command
per line from stdin and writes one JSON response per line to stdout. The first
response is `ready`; diagnostics and warnings go to stderr. It can also host
the inbound MCP HTTP listener in the same foreground process. It is not yet a
detached Windows service or supervised production host.

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
| HTTP host | `--http-listen`, `--http-port`, `--http-path`, `--http-token-env`, `--http-allowed-origin` |
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

`--worker-count N` enables a shared worker pool. The default remains `1` for
compatibility. Multiple workers may process different session lanes in
parallel, while the session manager keeps turns within one session ordered.
Workers are independent of the foreground/service-host choice: a future HTTP,
named-pipe or supervised host should construct the same dispatcher with the
same worker count.

## Inbound MCP HTTP in the daemon host

The daemon can expose the inbound Streamable HTTP MCP endpoint while retaining
JSONL on stdin/stdout for local administration. Both transports use the same
dispatcher, worker pool and runtime tool executor:

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
authentication/policy configuration. The current daemon assembly has the
dispatcher-backed call path; projecting the daemon's complete tool catalog and
supporting multiple token profiles remain follow-up work.

## Daemon lifecycle

```text
starting -> ready -> draining -> stopping -> stopped
                         \-> failed
```

`status` exposes readiness, liveness, queue capacity, active turn and session
lane information. A later service host may wrap the same daemon service and
dispatcher contracts with HTTP, named pipes or a supervisor without changing
the JSON/runtime ownership model.

## Current limitations

- foreground process only;
- JSONL remains the only administrative transport;
- no automatic restart or supervisor;
- no config reload;
- HTTPS support depends on an OpenSSL-enabled build for the current HTTP
  client/listener path; TLS termination is not part of the foreground daemon.
