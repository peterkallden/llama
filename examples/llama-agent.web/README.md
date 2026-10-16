# llama-agent web example

This is a small Vue 3/TypeScript test client for `llama-agent-web`. It is not a
second agent host: the daemon owns sessions, turns, planning, tools, MCP,
resources, policy and capabilities.

The client uses:

- `POST /api/v1/turns` for questions;
- `POST /api/v1/turns/{id}/cancel` for cancellation;
- `POST /api/v1/resources` for text and binary attachments;
- `GET /api/v1/resources/download?uri=...` for authenticated artifact downloads;
- `GET /api/v1/status` for status;
- `GET /api/v1/events` for JSONL events over SSE.

The JSON payload is the same as the daemon's JSONL event payload. Only the
transport framing changes from JSONL to SSE. SSE is connected with `fetch`
instead of `EventSource`, so a bearer token can be sent in the
`Authorization` header.

Text attachments use the existing `text` field. Binary attachments use the
existing byte-oriented resource-store path and are sent as bounded
`bytes_base64` JSONL payloads. The daemon applies the same 1 MiB resource
limit to both forms. The **Record audio** button uses the browser
`MediaRecorder` API, uploads the resulting WebM/Ogg/M4A recording through the
same binary path, and adds it to the next turn as a `resource_ref`. The
browser asks for microphone permission when recording starts. A prompt such as
`Transcribe the attached recording` can be sent together with the audio
resource; the web client does not perform speech recognition itself.

The example submits turns in agent mode. Small models may reject the initial
structured grammar before planning starts, especially for ordinary prompts
that do not need tools. The planned host-side solution is to select tool
families before exposing the full planner contract and route `needs_tools=false`
to ordinary chat. The web client does not interpret model error strings or
perform semantic fallback itself.

Tools already emit `tool.artifact_created` events for generated resources. The
example client renders a **Download** action for those events. The web adapter
reads the resource through the daemon's scoped resource API; it never serves
the resource-store filesystem directly. Download responses are bounded by the
adapter's `max_download_bytes` limit.

## Build and local development

Build the web adapter and client separately. The C++ adapter connects the web
client to the daemon's JSONL/TCP port; the Vue client is a set of static files.

```bash
cmake --build build-agent --target llama-agent-web -j4

npm install
npm test
npm run build
```

The last command creates `dist/`.

Start the daemon and `llama-agent-web` as described in
[`agent-daemon-usage.md`](../../docs/agent/agent-daemon-usage.md). Then install
the frontend dependencies and run:

```bash
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8090` for a native web adapter. To
develop the UI while the daemon and web adapter run in the local test
container on port 8080, use:

```bash
VITE_AGENT_DEV_PROXY=http://127.0.0.1:8080 npm run dev -- --host 127.0.0.1
```

The browser then uses `http://127.0.0.1:5173`; Vite serves the Vue source with
hot reload and proxies authenticated `/api` requests to the adapter.

## Minimal Nginx setup

For a local or internal Linux installation, Nginx can serve `dist/` and proxy
`/api/` to `llama-agent-web`:

```bash
sudo apt install nginx
sudo install -d /opt/llama-agent-web
sudo cp -a dist /opt/llama-agent-web/
sudo cp nginx/llama-agent-web.conf /etc/nginx/sites-available/llama-agent-web
sudo ln -s /etc/nginx/sites-available/llama-agent-web \
    /etc/nginx/sites-enabled/llama-agent-web
sudo nginx -t
sudo systemctl reload nginx
```

The example listens on `http://127.0.0.1:8080`, serves the client from
`/opt/llama-agent-web/dist` and forwards `/api/` to `llama-agent-web` at
`127.0.0.1:8090`. Run the daemon's TCP port and the web adapter on loopback as
described in the daemon documentation. For network access, put TLS and
deployment-specific authentication in front of Nginx; the supplied
configuration is not a production security profile.

The configuration disables proxy buffering and uses a long timeout for SSE. It
forwards the browser's `Authorization` header to the web adapter. The client
uses fetch-based SSE instead of the built-in `EventSource`, which allows bearer
authentication in same-origin and reverse-proxy deployments.

These are three separate process roles:

```text
llama-agent-daemon  --TCP-->  llama-agent-web  --HTTP/SSE-->  Nginx/browser
```

Nginx and the web adapter must not take over sessions, planning, tool
selection, MCP, resource policy or capability decisions from the daemon.

The client is intentionally an example and does not yet provide production
features such as login, event replay or full routing. Artifact downloads are
supported through the authenticated daemon endpoint, but deployment-specific
authorization and retention remain outside this example.

## Data and MCP Apps

The example client includes a read-only Data Explorer (`dataset.schema` and
bounded `dataset.sample`) and a Workflow Inspector over the daemon's structured
runtime events. Direct data calls go through `POST /api/v1/tools/call`; the web
adapter asks the daemon to resolve and execute the selected tool and forces
read-only policy. Tool input/output schemas and MCP UI metadata are carried in
`/api/v1/status`.

Tools that advertise an MCP App UI can be opened in the app tabs. The browser
host fetches the UI resource through `POST /api/v1/mcp-apps/resource` and brokers
app requests through `/api/v1/mcp-apps/call`. The host negotiates
`io.modelcontextprotocol/ui` with the HTML app MIME type and follows the
`ui/initialize` / `ui/notifications/initialized` handshake. A data-origin
sandbox proxy separates the host page from the app view; the app view itself is
sandboxed without same-origin privileges, has a restrictive CSP, and receives
no bearer token or provider credentials. The daemon verifies that the requested
URI belongs to the advertised app tool, and that app calls target app-visible
read-only tools owned by the same provider and named in that provider's
host-controlled `read_only_tools` list. The list is empty by default; remote
`readOnlyHint` is metadata, not authorization. App-only tools appear in status
for app discovery without being exposed to model tool selection. Links and
arbitrary network access are not brokered. OpenAPI operations remain visible to
model selection by default. App invocation context is isolated by provider and
UI resource and uses the newest matching invocation; automatic navigation
reacts only to a newly received matching invocation. HTTPS CSP wildcards are
normalized before they reach the sandbox; stale RPC responses are dropped
after app reload or switching.

Changing session remounts the app and Data Explorer, invalidates pending app
replies and excludes invocation events from other sessions. The bundled MCP
Apps populate their input and display results from invocation notifications,
including tool errors. Result notification parameters carry the standard MCP
result directly (`content`, `structuredContent`, `isError`). Initialization must
complete before the host accepts app tool requests.
Each completed invocation gets its own app instance using the host event
identity, so repeated calls with identical arguments and output are still
delivered. Event-driven navigation is processed synchronously before a later
turn event in the same SSE burst can replace the newest tool event.

The bounded web host supports initialization, scoped `tools/list` and
`tools/call`, reads of the app's own declared UI resource, ping, and a displayed
`ui/message`. Tool execution failures are MCP tool results with `isError: true`;
protocol failures remain JSON-RPC errors. When opened from a model invocation,
the app receives the actual bounded input and result. Standalone tabs do not
fabricate empty input. Unsupported app RPC methods are explicitly rejected.
The daemon's MCP server also publishes
first-party Data Explorer and Workflow Inspector resources
(`ui://llama-agent/apps/...`) and attaches them to the
corresponding dataset and `plan.get` tools. `resources/read` serves only those
known first-party URIs for these bundled apps; external app resources remain
provider-owned and are fetched through that provider.
