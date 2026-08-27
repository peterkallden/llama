# Agent daemon: configuration and startup

The current daemon is a foreground JSONL process. It reads one JSON command
per line from stdin and writes one JSON response per line to stdout. The first
JSONL response is `ready`, which confirms protocol availability; use the
`status` command to verify operational readiness. Diagnostics and warnings go
to stderr. It can also host
the inbound MCP HTTP listener in the same foreground process. It is not yet a
detached Windows service or supervised production host.

The same JSONL protocol can optionally be served over TCP. TCP is a transport
adapter for the existing dispatcher, not a second daemon implementation. The
stdin/stdout mode remains the simplest local administration mode. A copyable
TCP configuration is [agent-host-config-jsonl-tcp.json](../examples/agent-host-config-jsonl-tcp.json).

For a background Linux service, use the Unix domain socket configuration
[agent-host-config-jsonl-unix.json](../examples/agent-host-config-jsonl-unix.json)
and the systemd example [llama-agent-daemon.service](../examples/llama-agent-daemon.service).
The daemon remains a foreground process under systemd (`Type=simple`); systemd
owns restart, logging and stop timeout while the daemon owns runtime shutdown.
The socket file is created with the configured Unix mode, so its owner/group
provides an additional local access boundary. Authentication policy still
applies after the OS accepts the connection.

## Configuration discovery

The daemon, CLI and MCP host accept an explicit `--config PATH`. When no
explicit path is supplied, the shared host resolver checks these locations in
order:

1. `LLAMA_AGENT_CONFIG`
2. `/etc/llama-agent/config.json` on POSIX installations
3. `$XDG_CONFIG_HOME/llama-agent/config.json`
4. `%APPDATA%/llama-agent/config.json` on Windows
5. `~/.config/llama-agent/config.json`

An explicit path or `LLAMA_AGENT_CONFIG` value that does not exist is an error;
missing optional system and user configuration simply leaves the existing
command-line defaults in place. Systemd units and Docker entrypoints should
continue to pass their configuration path explicitly so service startup is
deterministic.

## Bootstrap a first configuration

The repository includes
[`scripts/agent-config-bootstrap.sh`](../../scripts/agent-config-bootstrap.sh)
and [`scripts/agent-config-bootstrap.ps1`](../../scripts/agent-config-bootstrap.ps1)
for creating a complete starting configuration. They use Cozo for the memory,
plan and structured-data stores, enable the normal agent deliberation defaults,
and keep network transports disabled unless they are selected explicitly. The
two scripts expose the same daemon transport, authentication, sandbox and
processor-policy controls.

Run it from the repository root in Bash:

```bash
./scripts/agent-config-bootstrap.sh \
  --model models/model.gguf \
  --embedding-model models/embedding.gguf \
  --cozo-root data \
  --repository-root . \
  --threads 4 \
  --gpu-layers 0 \
  --output agent-daemon-config.json
```

On Windows, the equivalent command is:

```powershell
.\scripts\agent-config-bootstrap.ps1 `
  -Model models\model.gguf `
  -EmbeddingModel models\embedding.gguf `
  -CozoRoot data `
  -RepositoryRoot . `
  -Threads 4 `
  -GpuLayers 0 `
  -Output agent-daemon-config.json
```

The default sandbox backend is `none`. For the Docker-compatible Podman
backend, keep the contract backend name as `docker` and select Podman as the
host executable:

```bash
./scripts/agent-config-bootstrap.sh \
  --sandbox docker \
  --sandbox-executable podman \
  --output agent-daemon-config.json
```

On Linux, LXC/Incus can be selected as the sandbox backend when Docker,
Podman or Kubernetes is unavailable:

```bash
./scripts/agent-config-bootstrap.sh \
  --sandbox lxc \
  --lxc-executable lxc \
  --lxc-image ubuntu:24.04 \
  --lxc-network-profile llama-agent-network-none \
  --lxc-network-profile-scope none \
  --output agent-daemon-config.json
```

LXC requires an operator-managed profile on every execution. The profile name
does not prove what the profile enforces, so its declared scope is explicit.
Use `none` for a networkless profile. For a deliberately restricted network,
use `--lxc-network-mode profile` together with a profile and one of
`dns_only`, `allowlisted`, `package_registry` or `research_web`. The host
advertises only the declared scope and rejects requests that it cannot prove
the profile enforces; CPU, memory and process limits are applied with LXC
instance limits and failure to set any requested limit aborts the operation.
Ready-to-import profile examples and the restricted-network template are in
[`docs/examples/lxc-profiles`](../examples/lxc-profiles/README.md).

Resource processors remain sandboxed unless explicitly selected. For trusted
hosts with MuPDF and Tesseract installed locally, local execution can be
enabled as follows:

```bash
./scripts/agent-config-bootstrap.sh \
  --pdf-page-image-execution local_preferred \
  --pdf-page-image-backend auto \
  --ocr-tesseract-execution local_preferred \
  --ocr-tesseract-backend auto \
  --pandoc-execution local_preferred \
  --pandoc-backend auto \
  --output agent-daemon-config.json
```

The backend values are `auto`, `local`, `docker`, `kubernetes` and `lxc`.
The PowerShell names are `-PdfPageImageExecution`,
`-PdfPageImageBackend`, `-OcrTesseractExecution` and
`-OcrTesseractBackend`, `-PandocExecution` and `-PandocBackend`. Use
`local_required` when silently falling back to a
sandbox is not acceptable.

The generated file can then be used by the daemon:

```bash
./build-agent/bin/llama-agent-daemon \
  --config agent-daemon-config.json
```

The script also provides a compact view of the current external tool catalog:

```bash
./scripts/agent-config-bootstrap.sh --list-tools
```

Memory, planning, deliberation, reflection, research and resource handling are
internal agent capabilities. They are not disabled by the external MCP/JSONL
tool allowlist. Use `--enable-tools` to restrict the tools exposed to an
authenticated caller:

```bash
./scripts/agent-config-bootstrap.sh \
  --transport jsonl-tcp \
  --auth-mode opaque \
  --token-env LLAMA_AGENT_TOKEN \
  --token-profile minimal \
  --enable-tools repository.read,repository.search \
  --output agent-daemon-config.json
```

Use `--enable-tools none` for an authenticated endpoint with no externally
callable tools. The allowlist is written to the inbound caller authorization;
it does not widen the host-owned tool profile.

## Start from a host configuration file

The recommended starting point is a config file rather than a long command
line:

```powershell
$env:REMOTE_GITHUB_MCP_TOKEN = "replace-with-a-token"

& .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-remote-http.json
```

For a local stdio MCP provider, use
[agent-host-config-stdio.json](../examples/agent-host-config-stdio.json). For a
remote provider, use
[agent-host-config-remote-http.json](../examples/agent-host-config-remote-http.json).
For inbound MCP authentication, the home, lab and enterprise examples are
[agent-host-config-inbound-mcp-home.json](../examples/agent-host-config-inbound-mcp-home.json),
[agent-host-config-inbound-mcp-auth.json](../examples/agent-host-config-inbound-mcp-auth.json)
and [agent-host-config-inbound-mcp-jwt.json](../examples/agent-host-config-inbound-mcp-jwt.json).
The model path and MCP command/URL in those files are examples and must be
changed for the local machine.
For a minimal custom capability/profile setup, use
[agent-host-config-capabilities.json](../examples/agent-host-config-capabilities.json).

## Container persistence and model mounts

The feature Docker image uses explicit paths for data that must survive a
container replacement:

| Container path | Purpose | Recommended mount |
| --- | --- | --- |
| `/models` | Generation and optional embedding GGUF files | Read-only bind mount or model volume |
| `/etc/llama-agent` | Generated daemon configuration | Persistent named volume or bind mount |
| `/var/lib/llama-agent/data` | Memory, plan, structured data and resource stores | Persistent named volume or bind mount |
| `/var/log/llama-agent` | Daemon diagnostics | Persistent named volume or bind mount |
| `/opt/llama-agent-web` | Built Vue example client served by Nginx | Image content; override only for development |

The release image includes
`agent-config.docker.example.json`, which points to these paths. The dev image
uses a separate development configuration with inbound MCP enabled, the
`all-configured` tool profile, agent tools enabled, and
`memory_learn: post-turn`. It expects
`/models/model.gguf` and `/models/embedding.gguf`; omit or edit the embedding
model entry when a separate embedding model is not used. The entrypoint keeps
diagnostics visible through the container runtime and also appends them to
`/var/log/llama-agent/daemon.log`. JSONL protocol output remains on stdout.

For example, with Docker-managed named volumes:

```bash
docker volume create llama-agent-models
docker volume create llama-agent-config
docker volume create llama-agent-data
docker volume create llama-agent-logs

docker run --rm \
  --name llama-agent-dev \
  --publish 127.0.0.1:8080:8080 \
  --publish 127.0.0.1:8081:8081 \
  --mount source=llama-agent-models,target=/models,readonly \
  --mount source=llama-agent-config,target=/etc/llama-agent \
  --mount source=llama-agent-data,target=/var/lib/llama-agent/data \
  --mount source=llama-agent-logs,target=/var/log/llama-agent \
  ghcr.io/peterkallden/llama/llama-agent-dev:cpu-amd64-latest
```

The container defaults to `LLAMA_AGENT_MODE=all`: it starts the daemon, the
JSONL/TCP web adapter and Nginx serving the Vue example client. Open
`http://localhost:8080`. The dev image also serves inbound MCP at
`http://localhost:8081/mcp`; clients must send the development bearer token.
The default Docker publish bindings above are loopback-only. To deliberately
make the service reachable from other hosts, change the host-side bind to
`0.0.0.0` and put it behind the network and authentication controls appropriate
for that environment. The daemon TCP listener remains internal on port `8091`.
Use `LLAMA_AGENT_MODE=daemon` for a daemon-only container.

For the dev image, the token is `dev-token` by default and is persisted at
`/etc/llama-agent/mcp-dev-token`; the entrypoint prints it once at startup so
local MCP clients can be configured quickly. Set `LLAMA_AGENT_DEV_TOKEN` or
`LLAMA_AGENT_MCP_TOKEN` to use another token. Keep the `/etc/llama-agent`
volume when you want the generated token and configuration to survive a
container replacement. The release image does not enable inbound MCP or
generate a token automatically.

Set `LLAMA_AGENT_WEB_TLS=true` to make Nginx serve HTTPS on port `8443` and
redirect HTTP from `8080`. If
`/etc/llama-agent/tls/server.crt` and `server.key` are absent, the entrypoint
generates and persists a development self-signed certificate. Mount a
certificate and key at those paths for deployment use:

```bash
docker run --rm --name llama-agent-dev \
  -e LLAMA_AGENT_WEB_TLS=true \
  -p 8443:8443 \
  -v llama-agent-tls:/etc/llama-agent/tls \
  ghcr.io/peterkallden/llama/llama-agent-dev:cpu-amd64-latest
```

TLS termination remains an Nginx/container concern; the daemon and web
adapter continue to use their internal loopback JSONL/TCP boundary.

The model volume must contain the files named by the configuration. For a
host bind mount, ensure that the container user can read `/models` and write
the configuration, data and log directories. The entrypoint generates
`/etc/llama-agent/config.json` with `llama-agent-config-bootstrap` when the
file is missing, and leaves an existing configuration unchanged. The model,
embedding model, store root, workspace root and selected runtime defaults can
be overridden through the corresponding `LLAMA_AGENT_*` environment variables
before first startup. A deployment-specific configuration can also be mounted
at `/etc/llama-agent/config.json` and is used automatically.

The same workflow also publishes verified release images from successful
`Agent release` runs. Release images use a separate package name and backend
tag, so they do not overwrite the feature image:

```text
ghcr.io/peterkallden/llama/llama-agent:cpu-amd64-latest
ghcr.io/peterkallden/llama/llama-agent:cuda13-amd64-latest
ghcr.io/peterkallden/llama/llama-agent:vulkan1.4-amd64-latest
```

Versioned release tags use the backend label and release identity, for example
`cpu-amd64-0.4.0-build17-llama-b6321`,
`cuda13-amd64-0.4.0-build17-llama-b6321`, or
`vulkan1.4-amd64-0.4.0-build17-llama-b6321`. The CUDA image supplies the CUDA runtime libraries; the
CUDA driver remains a host requirement. The Vulkan image supplies the Vulkan
loader, while the host remains responsible for the device driver and device
access.

The host may use a separate embedding model for semantic memory or resource
search. It is optional and is configured as an override alongside the
generation model:

```json
{
  "model": {
    "backend": "server-context",
    "path": "MODEL.gguf",
    "embedding_model": "EMBEDDING_MODEL.gguf"
  }
}
```

`model.path` is the generation model used for chat and agent turns.
`model.embedding_model`, when present, is used for embedding-backed search
instead of the generation model. When it is omitted, the implementation uses
`model.path` for embeddings as well. Omitting it is therefore valid and keeps
the configuration small; it is useful when the generation model also supports
embeddings and a second model would add unnecessary memory use. A separate
embedding model is useful when the generation model is not suitable for
embeddings, or when retrieval quality, model size or GPU/CPU resource usage
should be tuned independently from generation.

## Remote MCP tools

Remote tools are the client side of MCP: llama-agent connects outward to an
MCP server and adds its approved tools to the host tool view. This is different
from inbound delegation, where another agent connects inward and calls
`delegate_task` on llama-agent.

The copyable remote provider example is
[agent-host-config-remote-http.json](../examples/agent-host-config-remote-http.json).
The bootstrap script can import the `tools.providers` array from a separate
JSON file, which keeps provider configuration independent from the general
daemon settings:

```bash
./scripts/agent-config-bootstrap.sh \
  --providers-file docs/examples/agent-providers.json \
  --output agent-daemon-config.json
```

The same input can be supplied through stdin. This is useful for Docker
entrypoints and deployment templating:

```bash
cat providers.json | \
  ./scripts/agent-config-bootstrap.sh \
    --providers-file - \
    --output agent-daemon-config.json
```

The provider input must be a JSON array containing the existing provider
objects. It may contain `token_env` names, but must not contain secret values;
provide those values through the process environment at runtime. Use
`--providers-file none` to explicitly generate a configuration without remote
providers.

For a persistent deployment, provider objects can instead be split into
`tools.include_dir`; see [Agent configuration fragments](agent-config-fragments.md).
The directory is rescanned when the main configuration is reloaded.

The provider allowlist and the caller allowlist are independent: provider
`allowed_tools` limits what the daemon imports from a remote MCP server, while
`--enable-tools` limits what an authenticated MCP/JSONL caller can invoke.

Its essential shape is:

```json
{
  "runtime": {
    "default_mode": "agent",
    "thinking_mode": "research"
  },
  "tools": {
    "profile": "research",
    "providers": [
      {
        "type": "mcp",
        "id": "remote-github",
        "enabled": true,
        "transport": "streamable_http",
        "url": "https://mcp.example.com/mcp",
        "token_env": "REMOTE_GITHUB_MCP_TOKEN",
        "allowed_tools": ["search_issues", "read_issue"],
        "prefix": "github",
        "server_name": "github"
      }
    ]
  }
}
```

Start it with the token outside the configuration file:

```powershell
$env:REMOTE_GITHUB_MCP_TOKEN = "replace-with-a-token"
& .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-remote-http.json
```

The host performs the MCP handshake and tool discovery, then filters the
provider through the selected tool profile and `allowed_tools`. A remote tool
does not bypass the agent policy: writes, network access, resource references,
timeouts and result-size limits remain host-controlled. In research mode,
remote results are input to the research runner and must still become sources
or evidence before they can support a conclusion.

Turn responses normally contain the response, events and trace fields already
defined by the protocol. An admin or orchestration client can additionally set
`"include_summary": true` on a `run_turn` request to receive a compact
`turn_summary`; omitting the field preserves the ordinary response shape.
The CLI equivalent for `daemon-chat` and `daemon-session` is
`--include-summary`; it prints the answer normally and writes a compact
`turn-summary` line to stderr.

The built-in native profiles are `minimal`, `memory-read`, `memory`,
`analysis`, `research`, and `all-configured`. `analysis` is the default
bounded profile for explicit deliberate turns: it can inspect local resources,
document tables, datasets, statistics and (when host network policy permits)
web sources, but it does not include workspace writes, development execution
or memory proposals. `research` keeps controller-owned acquisition and
evidence progression. `all-configured` includes every currently
catalogued native tool; network access, proposal-style writes, confirmation,
scope and provider allowlists remain host-controlled.

The active profile is host-owned. A client may submit an objective, mode,
resources and bounded limits, but it cannot select a profile or widen the
tool surface. Public `run_turn` requests containing `tool_profile`,
`allowed_tools`, `allow_writes` or `enable_shell` are rejected. Authenticated
caller policy is a separate, narrower intersection applied by the host.

Profiles can be defined in host configuration through capabilities:

```json
{
  "tools": {
    "profile": "local-developer",
    "capabilities": {
      "workspace.read": ["repository.list", "repository.read", "repository.search"],
      "research.read": ["web_search", "web_fetch"]
    },
    "profiles": {
      "local-developer": {
        "include_capabilities": ["workspace.read"],
        "allow_network": false,
        "allow_policy_gated_writes": false
      }
    }
  }
}
```

The daemon `ready` response includes a `tooling` object with the active
profile, configured capabilities, resolved tool names and effective
network/write policy. It also reports `effective_capabilities`, the semantic
capabilities supplied by the resolved tools, and the selected
`agent_blueprint` mode. This is the authoritative startup diagnostic for what
the instance exposes before blueprint eligibility and ranking run.

For automatic blueprint reuse, set `runtime.agent_blueprint` to `auto` (or
use `--agent-blueprint auto`). A fixed blueprint id can be selected explicitly.
The daemon still filters persisted candidates by scope, assumptions, hard
constraints, and the resolved capability set before any model ranking. A
requirement such as `development.build` is therefore checked against the
host-owned profile; callers cannot add capabilities by choosing a blueprint.

The read-only developer profile also exposes `diagnostics.symbol` and
`diagnostics.references`. Their schemas contain a symbol plus an optional path
hint/definition path and result limit. A host may bind a semantic implementation
such as clangd/LSP; otherwise the native adapter uses the controlled repository
root as a bounded text fallback and marks the result with
`backend: text-fallback` and `semantic: false`. `diagnostics.test_failures`
does not execute tests: it groups an existing bounded result by normalized
message and classification, so it can be used by reflective and deliberate
flows as well as research.

The optional clang integration is enabled at build time with
`LLAMA_AGENT_TOOLS_CLANG=ON`. This only includes the integration seam; the host
still owns executable discovery, `compile_commands.json` selection and the
semantic provider binding. If `clangd` is unavailable, symbol and reference
tools retain the explicit text fallback for backward compatibility.

The tool catalog includes semantic `development.build` and `development.test` contracts for
the developer path. They are confirmation-gated and describe a
`developer-build` sandbox policy. When Docker is configured as the host
backend, the host resolves `sandbox.defaults` and the class overrides before
execution. Clients should send a target and bounded options, not container
commands or shell strings. With `backend: none`, sandbox-backed tools are
omitted from the effective tooling view at startup. A direct sandbox execution
request still fails with `sandbox.backend_unavailable`; the runtime never
falls back to an unsandboxed process.

For a local subprocess provider, use the same model with
`"transport": "stdio"` and a `command` array; see
[agent-host-config-stdio.json](../examples/agent-host-config-stdio.json).

The intended failure behavior is bounded: authentication or handshake errors
keep the provider unavailable, tool timeouts become failed tool outcomes, and
partial or non-authoritative results must not be presented as verified
evidence. The daemon should be started with a read-only allowlist first, then
expanded only when the provider and caller policy are understood.

## Start with flags only

The equivalent minimal shape is:

```powershell
& .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
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
| model | `--model`, optional `--embedding-model`, `--n-predict`, `--n-gpu-layers` |
| runtime | `--default-mode`, `--thinking-mode`, `--memory-learn`, `--max-reflection-rounds`, `--max-plan-revisions`, `--max-research-iterations`, `--agent-plan`, `--agent-blueprint`, `--agent-trace` |
| stores | `--backend`, `--memory-db`, `--plan-backend`, `--plan-db`, `--data-backend`, `--data-db` |
| resources | `--resource-blob-backend`, `--resource-blob-root`, `--resource-metadata-backend`, `--resource-metadata-db` |
| tools | `--tool-profile`, `--repository-root`, `--mcp-tool-command`, `--mcp-tool-arg`, `--mcp-tool-server-name`, `--mcp-tool-prefix` |
| HTTP host | `--http-listen`, `--http-port`, `--http-path`, `--http-token-env`, `--http-allowed-origin`, `--http-agent-tools`, `--http-max-delegation-depth` |
| JSONL TCP host | `--tcp-listen`, `--tcp-port`, `--tcp-max-line-bytes` |
| JSONL Unix host | `--unix-socket`, `--unix-socket-mode` |
| workers/limits | `--worker-count`/`--workers`, `--inference-max-active`, `--queue-capacity`, `--max-turn-seconds`, `--max-tool-rounds` |

Structured data storage is configured independently from memory and plan:

```json
{
  "stores": {
    "data": {
      "backend": "cozo",
      "path": ".\\work\\agent-data.cozo"
    }
  }
}
```

`backend: auto` disables the data store when no path is configured. A Cozo
path requires a build with `LLAMA_MEMORY_COZO=ON`. When the data store is not
available, store-backed data tools are removed from the model-visible resolved
tool snapshot rather than exposed as tools that fail on every call.

## Agent runtime modes

The agent runtime has three thinking modes. They describe how much control and
verification the agent applies; they do not select tools or MCP providers.

`reflective` is the default agent level. It performs bounded execution or
reasoning, produces a draft, reviews the result, and may revise it once. A
persistent plan is optional and evidence cross-checking is not required.

`deliberate` is for multi-step work, several constraints, alternatives, or
uncertainty. It normally requires a plan, reviews relevant steps, permits
bounded plan revisions, and performs final answer review.

`research` builds on deliberate work. It adds explicit knowledge-gap
tracking, source acquisition, evidence extraction, provenance and
evidence-based stopping. The current branch contains the bounded research
controller and workspace slice: objectives, gaps, tasks, sources, evidence,
coverage and a normalized result. User-supplied resources and memory hits can
be selected as bounded research inputs. The result preserves source and
evidence provenance, and the runtime includes bounded source comparison,
synthesis context, gap assessment and answer verification. Tool completion is
not treated as proof that a gap is answered; the assessment is a separate
typed decision.

The remaining research work is model-driven rather than structural: query and
gap reformulation, semantic assessment, structured claim extraction and a
verification path can now reopen the same turn-local research workspace once
after synthesis. That reopen is capped at one additional bounded research task
and one draft verification pass. The current runner remains sequential and
bounded and delegates actual tool calls to the host tool runtime.

The intended host configuration shape is:

```json
{
  "runtime": {
    "default_mode": "agent",
    "thinking_mode": "reflective",
    "n_threads": 4,
    "n_gpu_layers": 0,
    "max_reflection_rounds": 1,
    "max_plan_revisions": 0,
    "max_research_iterations": 0
  },
  "limits": {
    "worker_count": 2,
    "inference_max_active": 1,
    "max_tool_rounds": 16
  }
}
```

`runtime.n_threads` controls the CPU thread count used by daemon-owned model
generation. It is propagated to the resident turn request and to the shared
runtime generation configuration, so planning, reflection, tool-call repair
and final response generation use the same configured value. It defaults to
`2` and must be greater than zero. The setting is restart-required when the
daemon configuration is reloaded.

`runtime.n_gpu_layers` controls how many model layers are offloaded to the GPU
for the daemon's resident inference context. `0` keeps inference on the CPU;
the exact useful value depends on the model and available device memory. This
is a GPU-offload setting, not a separate GPU-thread count, and it is also
restart-required when the daemon configuration is reloaded.

The earlier two PoC mode flags are no longer accepted. Use
`default_mode=chat|agent`
together with `thinking_mode=reflective|deliberate|research`;
`direct` remains a chat runtime behavior, not an agent thinking mode.

The requested `thinking_mode` is a starting point. The host may resolve one
bounded upward escalation when the request contains multiple constraints,
external uncertainty, resource-comparison requirements or an explicit
verification request. Escalation is controlled by host policy and is reported
through typed `thinking_mode_resolved`, `thinking_escalation_allowed` and
`thinking_escalation_denied` events. A denied escalation leaves the requested
mode in effect and includes a reason code.

The config file is loaded first and explicit flags are the appropriate place
for a one-run override. Secrets should remain in environment variables, not
in flags or checked-in JSON.

## JSONL commands

The copyable request sequence is in
[agent-daemon-requests.jsonl](../examples/agent-daemon-requests.jsonl). It
demonstrates status, one chat turn, and graceful shutdown.

```powershell
$requests = Get-Content .\docs\examples\agent-daemon-requests.jsonl
$requests | & .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-stdio.json
```

Important protocol behavior:

- keep stdout reserved for JSONL protocol messages;
- consume stderr separately for diagnostics;
- wait for `event: "ready"` before sending turns;
- treat `message_type: "event"` lines as the live daemon event channel;
- treat `message_type: "response"` lines as terminal result-only responses;
- use stable `session_id`, `namespace_id`, `project_id`, and `turn_id` values;
- send `{"command":"shutdown"}` for graceful foreground shutdown;
- do not treat this stdin/stdout process as a production network service yet.

Configuration can be reloaded by the local JSONL administration channel:

```json
{"request_id":"reload-1","command":"reload_config","path":"docs/examples/agent-host-config-stdio.json"}
```

The daemon validates the complete candidate configuration before applying it.
Timeouts, repository root, bounded tool limits and MCP/OpenAPI providers can be applied
to new operations. A repository root must resolve to a directory. Provider IDs
are stable: new IDs are added, missing IDs are removed for future operations,
and changed IDs are replaced. Existing operations keep their provider clients
until their tooling is destroyed. The active tool profile, capability map and
profile definitions are restart-required so an instance never changes its
host-owned tool snapshot halfway through a running session. Model/backend,
stores, resource roots, worker/queue sizing and runtime assembly are also
restart-required for now. A
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

`--inference-max-active N` controls the maximum number of active resident
`host->run_turn()` executions. It is independent of `--worker-count` and
defaults to `1`, so increasing daemon workers does not increase GPU
concurrency. Turns waiting for capacity remain visible as
`awaiting_inference`/`wait_for_inference` through the existing operation and
event model. The configuration equivalent is:

```json
{
  "limits": {
    "worker_count": 2,
    "inference_max_active": 1
  }
}
```

The admission scheduler uses one shared inference capacity across daemon
workers. A turn that cannot acquire capacity is represented as a pending
operation and follows this event order:

```text
inference.queued
  -> turn.waiting_for_inference
  -> inference.capacity_granted
  -> turn.waiting_for_inference
  -> turn.completed | turn.failed | turn.cancelled
```

The first waiting event means admission waiting; the second means that the
resident host execution has started. Chat turns are currently interactive,
normal agent turns normal, and research turns background. Waiters are FIFO
within priority, with bounded aging after five seconds. There is no separate
scheduler-thread flag: `worker_count` controls daemon dispatchers and
`inference_max_active` controls model execution capacity.

## JSONL over TCP

Start the TCP adapter with the configuration example:

```powershell
$env:LLAMA_AGENT_TCP_READ_TOKEN = "replace-with-a-long-random-read-secret"
$env:LLAMA_AGENT_TCP_ADMIN_TOKEN = "replace-with-a-long-random-admin-secret"
& .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
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

## Minimal web adapter: HTTP commands and SSE events

`llama-agent-web` is a protocol adapter for a running JSONL/TCP daemon. It is
not another agent host: sessions, turns, planning, tools, MCP, resources,
policy and capabilities remain owned by `llama-agent-daemon`.

The adapter translates only the transport boundary:

```text
HTTP request                  JSONL request
HTTP response / SSE event  <- JSONL response / event
```

Start the daemon with a private TCP endpoint first, then start the web adapter:

```bash
./build-agent/bin/llama-agent-daemon \
  --config docs/examples/agent-host-config-jsonl-tcp.json

./build-agent/bin/llama-agent-web \
  --daemon-address 127.0.0.1 \
  --daemon-port 8091 \
  --listen 127.0.0.1 \
  --port 8090 \
  --web-bearer-token replace-with-a-web-secret \
  --daemon-authorization "Bearer replace-with-a-daemon-secret"
```

The daemon authorization value must match the TCP caller policy. Keep both
listeners on loopback or behind a TLS/authentication proxy; the adapter does
not provide TLS.

The initial web surface is intentionally small:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/v1/turns` | Forward a JSON turn request |
| `POST /api/v1/turns/{id}/cancel` | Forward `cancel_turn` |
| `POST /api/v1/resources` | Forward `put_resource` for text or bounded `bytes_base64` payloads |
| `GET /api/v1/resources/download?uri=...` | Read an authenticated bounded resource download through the daemon |
| `GET /api/v1/status` | Return the daemon status response |
| `GET /api/v1/events` | Stream unchanged daemon JSONL event objects as SSE |

The browser uses ordinary HTTP for commands and SSE for events. The example
Vue client uses fetch-based SSE so it can send a bearer token in the
`Authorization` header; a client using native `EventSource` must instead use
same-origin cookie authentication or an authenticated proxy. SSE adds only
`id`, `event` and `data` framing; the JSON object in `data` remains the common
JSONL event payload. Event IDs mirror the daemon sequence for client-side
correlation. Replay/resume from `Last-Event-ID` is deliberately left for a
later protocol extension; the initial adapter does not pretend to provide
replay when a client reconnects.

Resource uploads preserve the existing daemon contract. Text files use the
`text` field; binary files use `bytes_base64` and are decoded into the
byte-oriented resource store before the turn receives the resulting opaque
resource URI. Both forms are bounded by the daemon's 1 MiB resource limit.

For authenticated TCP and Unix clients, resource upload, read and list
requests are always bound to the namespace/project in the authenticated token
policy. Omitting those fields is supported; sending different values does not
change the authority. This keeps an uploaded URI readable by the subsequent
turn, whose scope is bound by the same policy.

For agent planning, uploaded URIs are rendered to the model as `r1`, `r2`, ...;
other visible scoped resources are rendered as `s1`, `s2`, .... These are
model-facing handles, not values to send back over the wire. A single current
attachment is the default dataset-inspection source unless the model selects
an explicit canonical dataset reference; multiple attachments require an
explicit choice. See [Agent Runtime](agent-runtime.md#resource-handles-and-attachment-defaults).

Tool-generated resources are announced as `tool.artifact_created` events. The
web client may turn the resource URI from that event into a download action,
but the web adapter must read the bytes through the daemon's scoped
`read_resource` command. It must not expose the resource-store filesystem or
construct direct filesystem URLs. Download responses are independently
bounded by the web adapter configuration.

The adapter owns HTTP routing, SSE connections, request framing, browser
authentication and CORS. The example client and the optional Nginx config are
documented in [`examples/llama-agent.web`](../../examples/llama-agent.web/README.md).
Nginx should serve the built `dist/` directory and proxy `/api/` without
buffering; it is also the natural place for TLS and deployment-specific edge
authentication. Static files and richer binary upload/download routes can be
added around this seam later. Neither layer may implement planning, tool
selection, MCP semantics, resource policy or a second session manager.

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
[agent-host-config-inbound-mcp-auth.json](../examples/agent-host-config-inbound-mcp-auth.json)
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
[agent-host-config-inbound-mcp-jwt.json](../examples/agent-host-config-inbound-mcp-jwt.json).
The authorization server issues the access token; the daemon only verifies it
and maps it to the configured native policy. Use an OpenSSL-enabled build.

```powershell
$env:LLAMA_AGENT_MCP_TOKEN = "replace-with-a-local-secret"

& .\build-agent-inbound-debug\bin\Debug\llama-agent-daemon.exe `
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
[agent-host-config-inbound-mcp-agent.json](../examples/agent-host-config-inbound-mcp-agent.json).

Start that example with its caller token in the environment:

```powershell
$env:LLAMA_AGENT_AGENT_TOKEN = "replace-with-a-long-random-secret"
& .\build-agent-current-ninja-debug\bin\llama-agent-daemon.exe `
    --config .\docs\examples\agent-host-config-inbound-mcp-agent.json
```

The endpoint is then `http://127.0.0.1:8080/mcp`. An MCP client sends the
token as `Authorization: Bearer ...`; it should first perform the normal MCP
initialize handshake and call `tools/list` before invoking a tool.

### Connecting another agent

An MCP client discovers the available thinking modes from `tools/list`; the
client does not need a separately maintained list. The `delegate_task` schema
advertises `reflective`, `deliberate` and `research`, with `reflective` as the
default. A client can then make a normal MCP `tools/call`:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "delegate_task",
    "arguments": {
      "task": "Inspect the repository and report the remaining cancellation gaps.",
      "thinking_mode": "research",
      "resource_refs": ["resource://submitted/design-notes"],
      "max_research_iterations": 4
    }
  }
}
```

`resource_refs` are input references; the daemon resolves them through the
configured resource boundary and makes them available to the delegated turn.
The caller cannot widen its namespace, project, tool allowlist or write
policy. Keep the listener on localhost unless an authenticated private-network
deployment is deliberately configured.

The default inbound call remains synchronous at the MCP boundary and waits for
the bounded dispatcher result. When `--http-agent-tools` is enabled, the HTTP
server also advertises the experimental MCP Tasks capability using protocol
version `2025-11-25`. A caller may add `params.task` to `tools/call`; the
response contains a task handle, which can be inspected with `tasks/get` and
completed with `tasks/result`. The task adapter still delegates execution to
the existing daemon queue, session lane, operation and typed-event lifecycle;
it does not create a second agent runtime. Task retention, cancellation and
tool execution remain bounded by the daemon host policy.

## VS Code and Visual Studio

VS Code and recent Visual Studio versions are MCP clients. They can connect to
llama-agent in either of two different roles:

| Use case | llama-agent endpoint | Role |
| --- | --- | --- |
| Give an IDE tools and resources | `llama-agent-mcp-stdio-server` over stdio, or the MCP HTTP server | llama-agent is the MCP server |
| Let an IDE/agent delegate work to llama-agent | `llama-agent-daemon --http-agent-tools` over inbound MCP HTTP | llama-agent receives `delegate_task` |
| Local administration and test control | JSONL stdin/stdout or TCP | daemon control plane, not an IDE tool surface |

These are two related but distinct host shapes. `llama-agent-daemon` is the
resident agent runtime: it owns sessions, turns, queueing, cancellation,
status, and the JSONL control plane. It can also host inbound MCP over HTTP
and expose `delegate_task` when `mcp.inbound` and agent tools are enabled.
`llama-agent-mcp-stdio-server` is a standalone MCP transport adapter for a
client that launches an MCP subprocess over stdin/stdout. It is not a second
daemon runtime. Both binaries are included in the agent package because they
serve different integration directions.

The local stdio example is
[vscode-mcp.json](../examples/vscode-mcp.json). Copy it to `.vscode/mcp.json`,
replace the executable and repository paths, then use **MCP: List Servers**
and **Chat: Configure Tools** in VS Code. The same `servers` shape can be
used in the supported `mcp.json` locations in Visual Studio. These clients
perform the MCP handshake and query `tools/list` before exposing tools.

For inbound delegation, use the HTTP server configuration above and use the
copyable [vscode-mcp-inbound.json](../examples/vscode-mcp-inbound.json) instead
of the stdio entry:

```json
{
  "servers": {
    "llama-agent-inbound": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${input:llamaAgentToken}"
      }
    }
  },
  "inputs": [
    {
      "type": "promptString",
      "id": "llamaAgentToken",
      "description": "Bearer token for the local llama-agent MCP endpoint",
      "password": true
    }
  ]
}
```

This is a client configuration, not a daemon configuration. The daemon must
already be running with `--http-agent-tools`, and its token policy must permit
the caller's `delegate_task` access. Do not commit real tokens or model paths.

Official client references:

- [VS Code MCP servers](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
- [VS Code MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)
- [Visual Studio MCP servers](https://learn.microsoft.com/en-us/visualstudio/ide/mcp-servers?view=visualstudio)

The VS Code extension API is not required for this integration. MCP is the
stable boundary; an IDE-specific extension can be added later for UI,
approval and task/polling conveniences.

### Codex integration

Codex can use llama-agent as an MCP server in the same two roles:

1. Codex uses llama-agent tools and resources.
2. Codex delegates a bounded task to llama-agent through inbound MCP.

For a local stdio connection, add a server entry to the Codex MCP
configuration:

```toml
[mcp_servers.llama_agent]
command = "C:\\path\\to\\llama-agent-mcp-stdio-server.exe"
args = [
  "--tool-profile",
  "research",
  "--repository-root",
  "C:\\path\\to\\repository"
]
startup_timeout_sec = 30
```

The exact Codex configuration location depends on the Codex surface in use;
the CLI uses its own `config.toml` MCP section and does not consume
`.vscode/mcp.json` automatically. Keep executable paths and secrets local.

For delegation, start the daemon with the inbound agent configuration above
and connect Codex to:

```text
http://127.0.0.1:8080/mcp
```

Codex then discovers `delegate_task`, `summarize` and `review_plan` through
`tools/list`. A delegated request can select a mode and pass resource refs:

```json
{
  "name": "delegate_task",
  "arguments": {
    "task": "Review the current runtime design and list unresolved scheduler gaps.",
    "thinking_mode": "deliberate",
    "resource_refs": ["resource://project/runtime-notes"],
    "max_plan_revisions": 2
  }
}
```

This is a delegation boundary, not a second Codex runtime. The receiving
llama-agent owns the thinking policy, tool allowlist, scope, plan, events and
result. Until MCP Tasks and polling are implemented, Codex must treat the
request as a bounded synchronous call and handle timeout or incomplete-result
responses explicitly.

## Integration checklist

Before connecting another agent or IDE:

1. Start with `127.0.0.1` and a token environment variable.
2. Run `tools/list` and verify the expected tool profile and schemas.
3. Confirm that `delegate_task` exposes the three thinking modes and defaults
   to `reflective`.
4. Test a read-only delegated task before enabling any write-capable profile.
5. Preserve `request_id`, `session_id`, `turn_id`, namespace and project values
   in the caller-side trace where the client supports them.
6. Treat long-running research as bounded synchronous work until MCP Tasks and
   polling are implemented.

## Daemon lifecycle

```text
starting -> ready -> draining -> stopping -> stopped
                         \-> failed
```

`status` exposes readiness, liveness, queue capacity, active turn and session
lane information. A later service host may wrap the same daemon service and
dispatcher contracts with HTTP, named pipes or a supervisor without changing
the JSON/runtime ownership model.

The interactive daemon client supports both `/status` for a compact summary and
`/status --verbose` for the complete JSONL status payload, including readiness,
provider details, warnings, sessions and metrics. Both commands use the same
JSONL `{"command":"status"}` request; verbose mode only changes client-side
rendering.

Every status-shaped JSONL response also contains a small `metrics` object with
dispatcher counters: `commands_accepted`, `commands_completed`,
`commands_failed`, `turns_completed`, and `tools_completed`. This is the first
host metrics slice; a later service host can project the same counters to
Prometheus text without making metrics an MCP tool.

### Readiness is an operational gate

The daemon keeps lifecycle state (`starting`, `ready`, `draining`, `stopping`,
`stopped` or `failed`) separate from operational readiness. Status responses
include a `readiness` object with the core runtime and store checks:

```json
"readiness": {
  "health": "ready",
  "model": "loaded",
  "inference": "available",
  "stores": {
    "memory": "ready",
    "plan": "ready",
    "resource": "ready"
  },
  "tool_profile": "research",
  "providers": [
    {
      "id": "local-mcp",
      "status": "ready",
      "required": false
    }
  ],
  "warnings": []
}
```

`ready: true` is a serving gate: the daemon has an initialized runtime,
required stores, an active worker that can accept commands, and no failed
required MCP provider. At startup, configured MCP providers are probed through
the standard MCP initialize and tool-discovery path. There is no portable MCP
health method, so this probe is deliberately bounded to handshake and
`tools/list`.

The provider probe is configured by the shared daemon environment initializer,
so foreground startup and daemon lifecycle smokes exercise the same readiness
path rather than maintaining separate probe implementations.

An optional provider that fails its probe produces `health: "degraded"`, keeps
`ready: true`, and includes a warning. A required provider that fails produces
`health: "failed"` and keeps `ready: false`. `required` is host policy, not a
client-selected capability. A daemon may be `live` while it is starting,
draining or stopping, but clients must not use `live` as a substitute for
readiness.

Provider probes retain their established client resources for the daemon
lifetime so a successful startup probe does not immediately destroy the MCP
session. Configuration reload currently reports provider changes for future
operations; provider readiness is evaluated at daemon startup and a reload
that changes providers requires a restart for a fresh probe.

## Current limitations

- foreground process only;
- stdin/stdout remains the local administrative transport; JSONL TCP is
  available as an explicitly enabled transport adapter;
- no automatic restart or supervisor;
- config reload is currently available only through local JSONL administration;
- config reload does not restart the process or rebuild model, stores, workers,
  HTTP listeners or already-running MCP provider clients;
- changing `tools.profile`, `tools.capabilities` or `tools.profiles` through
  reload requires a daemon restart;
- HTTPS support depends on an OpenSSL-enabled build for the current HTTP
  client/listener path; TLS termination is not part of the foreground daemon.
- Unix socket mode is a foreground service transport, not an internal
  daemonization mechanism; systemd or another supervisor owns process lifetime.
