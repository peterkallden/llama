# llama-agent web example

This is a small Vue 3/TypeScript test client for `llama-agent-web`. It is not a
second agent host: the daemon owns sessions, turns, planning, tools, MCP,
resources, policy and capabilities.

The client uses:

- `GET/POST /api/v1/projects` for project navigation;
- `GET/POST /api/v1/chats` and `GET /api/v1/chats/{id}/messages` for chat navigation and persisted transcript history;
- `POST /api/v1/chats/migrate-legacy` for one-time import of an existing browser session;
- `GET /api/v1/resources` for a bounded, scope-filtered resource browser;
- `POST /api/v1/chats/{id}/resources` to attach or detach a resource from a chat;
- `POST /api/v1/turns` for questions;
- `POST /api/v1/chats/{id}/turns/{turn_id}/cancel` for chat-bound cancellation;
- `POST /api/v1/resources` for text and binary attachments;
- `GET /api/v1/resources/download?uri=...` for authenticated artifact downloads;
- `GET /api/v1/status` for status;
- `GET /api/v1/events` for JSONL events over SSE.

The JSON payload is the same as the daemon's JSONL event payload. Only the
transport framing changes from JSONL to SSE. SSE is connected with `fetch`
instead of `EventSource`, so a bearer token can be sent in the
`Authorization` header.

The adapter sends an initial SSE comment immediately, followed by events and
heartbeats. Reconnect callbacks are tied to the active connection so an older
stream cannot reset a newer connection's status. Daemon JSONL responses are
read in buffered blocks; the response bound includes JSON escaping overhead
for configured downloads and bundled MCP Apps. Turn generation limits come
from the daemon configuration rather than a fixed browser token limit.

Text attachments use the existing `text` field. Binary attachments use the
existing byte-oriented resource-store path and are sent as bounded
`bytes_base64` JSONL payloads. The daemon applies the same 1 MiB resource
limit to both forms. The **Record audio** button uses the browser
`MediaRecorder` API, uploads the resulting WebM/Ogg/M4A recording through the
same binary path, and adds it to the next turn as a `resource_ref`. The
browser asks for microphone permission when recording starts. A prompt such as
`Transcribe the attached recording` can be sent together with the audio
resource; the web client does not perform speech recognition itself.

The example submits turns through the existing agent runtime. When tool-family
routing is enabled, the host invokes the model-based family selector over the
authorized catalog before exposing individual tool contracts. A valid
`needs_tools=false` selection uses the ordinary chat path when no host-bound
operation requires tools. Invalid family selections take the explicit normal
tool-route fallback with the authorized catalog and existing requirements;
they do not imply a tool-free answer. Routing uses conversation context and
host bindings rather than a browser keyword list. The web client does not
interpret model error strings or perform semantic fallback itself. See
[runtime seams](../../docs/agent/agent-runtime-seams.md) for prompt, selection
and required-tool boundaries.

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

The client is intentionally an example and does not provide production login
or SSE event replay. Project and chat navigation metadata is stored through the
daemon resource API under a reserved host-owned project in the configured
`web` namespace. A project is available to callers authorized for that
namespace; this metadata is not a user directory or a per-user ACL system.
Transcript messages are rebuilt from the daemon's persisted session-context
resources. If older context was not retained, the history endpoint reports that
history is unavailable instead of reconstructing it from a model summary.
Chat and transcript listings are paginated; transcript pages are ordered by
stored sequence and the UI loads older pages on demand. The adapter uses the
daemon's session snapshot to report active turns and supplements it with turns
currently being forwarded by this adapter. The client refreshes the selected
chat's authoritative history after reconnect; it does not resend the turn. Accepted work remains daemon-owned when the browser
changes chats, while a browser reload marks uncertain in-flight local queue
items for review instead of replaying them automatically.

Existing browser sessions that used the former `web-example` project are
imported once through the explicit migration endpoint. New clients do not use
that project identifier. Resource and transcript reads continue through the
daemon's scope checks; navigation metadata does not grant access to resources.

## Data and MCP Apps

The example client includes a read-only Data Explorer (`dataset.inspect`,
`dataset.schema` and bounded `dataset.sample`) and a Workflow Inspector over the daemon's structured
runtime events. Direct data calls go through `POST /api/v1/tools/call`; the web
adapter asks the daemon to resolve and execute the selected tool and forces
read-only policy. Tool input/output schemas and MCP UI metadata are carried in
`/api/v1/status`.

The Chat menu lists recent chats or chats in a selected project, starts new
chats, and creates projects. The Resources workspace browses chat/project
resources, registered datasets, available MCP App workspaces and saved
investigation cases through existing host operations. The browser passes only
the selected context; each resource or tool operation is independently
authorized by the adapter and daemon. Resources can be previewed, opened in
Data Explorer, or selected as chat references. Large resources remain bounded
and should be inspected through Data Explorer rather than loaded into the
navigation list.

Both Data Explorer views (the web panel and the sandboxed MCP App) accept exactly
one source: a dataset reference, an authorized resource reference, or a path
relative to the configured repository root. Inspection metadata, schema and sample
rows are displayed as tables. Click a column heading to toggle ascending and
descending sorting; numbers sort numerically and missing values remain last.
Sorting applies only to the rows displayed, not to the entire remote dataset.
The **Show raw JSON** button displays the structured result without its transport
envelope. Truncated samples are labelled explicitly. Source changes discard old
results and pending replies cannot replace a newer source's view. These operations
do not start a model turn.

Project and chat titles are displayed in the conversation header, separate from
workspace navigation buttons. New chats initially use the first question. After a
turn, the adapter resolves that turn's final persisted plan and uses its latest
non-empty goal as the chat title, including when a previous title exists. This
reuses the plan evaluated during reflection; reflection has no separate chat-goal
field, and naming starts no additional inference. The catalog records the source
plan and version. A latest-turn binding prevents older completions from renaming
a chat; absent goals preserve the existing title. Host-only attachment
orientation does not rename the conversation. Metadata naming errors are
reported separately as `chat_title_error` and do not fail the completed turn.
The HTTP/JSONL naming smoke is
`pocs/agent/smoke/daemon/agent-web-chat-title-smoke.py --web-binary <path>`;
it uses a synthetic daemon peer and does not verify model or Cozo behavior.

Embedded apps grow with their content through
validated `ui/notifications/size-changed` messages relayed by the sandbox proxy.
The host also measures content changes, media loading and width reflow inside
the isolated app. Height-only host resizing does not trigger another measurement.
Bundled app layouts use content height rather than viewport-relative page heights
to avoid a resize feedback loop. Heights are bounded to 320–20,000 CSS pixels;
large data tables can retain their own paginated or virtualized scrolling views.

Tools that advertise an MCP App UI can be opened from the right-hand status
floater. Applications use the main workspace with chat alongside it. The browser
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
UI resource and uses the newest matching invocation. Tool events update saved
invocation context without changing the user's selected workspace. HTTPS CSP wildcards are
normalized before they reach the sandbox; stale RPC responses are dropped
after app reload or switching.

Changing session remounts the app and Data Explorer, invalidates pending app
replies and excludes invocation events from other sessions. The bundled MCP
Apps populate their input and display results from invocation notifications,
including tool errors. Result notification parameters carry the standard MCP
result directly (`content`, `structuredContent`, `isError`). Initialization must
complete before the host accepts app tool requests.
Invocation identity is preserved when repeated calls have identical arguments
and output. The selected app stays mounted while chat updates; a document
editor accepts subsequent invocation context only for its bound document.

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

## Research, investigation and code views

The Resources workspace exposes project/session research resources and saved
work products. Research evidence graphs and report-block evidence graphs use
`visualization.research_graph`; the host resolves either the selected active
research resource or the report's exact frozen snapshot. Selecting a current
research resource does not change an older report's saved evidence. Research identity,
chat drafts and compare-and-swap publication are described in
[persistent research](../../docs/agent/agent-runtime.md#persistent-research-and-visualization).

The Investigation Workspace accepts documents, retained structured logs and
configured search sources in the same case. Users explicitly select sources,
parser mappings and analysis profiles; uploading does not start forensic
analysis. Actor mentions remain separate candidates, and the app's analyst
review path can confirm/withdraw associations without merging original source
records. See [Investigation Workspace](../../docs/agent/agent-investigation.md)
for the nine workflow families, import limits, permissions and frozen reports.

Saved report cards load retained PNG figures and their visualization references
through the scoped resource API. Figures remain tied to the saved report
revision. A local layout or a user-supplied snapshot is presentation data and
does not create independent evidence.

Completed `code.calls` and `code.impact` tool events carry a host-validated relationship projection
that is displayed in the corresponding assistant turn. Delivery sequences are
deduplicated and graph events must match the current chat session. The generic
`RelationshipGraph` component is loaded on demand and is also used by the
research wrapper. These code graphs are transient and their source locations
are not pinned to a commit. See [Source Code Analysis](../../docs/agent/code-analysis.md)
for provider configuration and current limits.

## Shared document and chat workspace

Use the pen icon on a supported TXT, Markdown, or DOCX attachment to import and
open it in the document editor. The editor stays in the main pane while the
same chat, including its unsent text and attachments, moves to the right. At
screen widths below 1,000 px the panes stack. Page width follows the viewport;
the editor receives the remaining width after the compact chat column. Web
client, status, metrics, tools, application launchers and events live in an
expandable floater aligned with the workspace toolbar. Authentication expands
when required and collapses after a successful connection. Anonymous local
connections do not display a missing-token warning.

Published report/document resources appear as read-only cards with their
assistant turn. Discovery uses host-owned resource URI, name and media type
in events, not filenames or links parsed from model text. Completed HTTP turn
responses recover events missed by SSE. Reading uses `document.open`, which
also returns host-rendered Markdown for downloading the same immutable revision.
Its result limit is 32 MiB to accommodate Markdown and structured content;
underlying document/import and transport limits still apply.

Hovering or focusing a report block shows a saved-provenance preview. Source
and tool buttons open a scrollable modal. Dataset references reuse Data
Explorer, including bounded samples and sorting. Tool observations come from
the report's frozen snapshot and are never rerun for presentation. Ordinary
sources offer their original web URL and, when authorized, a bounded text
preview or saved file in a separate tab. Sources without a saved relationship
are not inferred from prose. Linked references and host support/review status
remain distinct from factual verification.

Reading uses the existing pinned TOAST UI viewer and DOMPurify. Active HTML,
embedded images and runtime media loading are disabled in report cards. The
web build includes `third-party-notices.txt`; the floater links to these notices.

Before a chat message is sent from this layout, the editor saves its current
draft and attaches the resulting immutable document revision as typed context.
A failed save keeps the message draft and prevents submission. This save does
not trigger a separate review. The editor's × action saves and queues review of
unreviewed changes, then returns to the previous view; chat shows the queued,
running or failed state and offers a retry. If another turn is active, the
review waits in session-scoped browser storage until the turn finishes. A retry
reuses its stable turn ID. Model edit proposals still require explicit
acceptance in the editor. Starting a new session is blocked while a turn or
document review is pending so the session-bound review queue cannot be orphaned.
Development hot reload closes the previous session module's SSE connection;
otherwise repeated edits can leave stale streams occupying browser connections.

Queue registration succeeds only after browser storage accepts the entry. A
storage failure keeps the document open or the chat draft intact. Uploads and
audio recordings retain their originating session; late results cannot attach
files to a different session. Selecting another document first completes the
current editor's save-and-close flow.

Review requests carry `document_context.base_document_uri` alongside the saved
revision. The host authorizes both revisions of the same document and makes
their Markdown comparison available through the normal resource/chunk path.
Diff previews are not accumulated in the browser or embedded in model prompts.
Selecting the complete document sends an empty block-ID list, which means all
blocks; partial selections are limited to 128 blocks.

## Verification handoff for the workspace redesign

This implementation checkpoint includes tests but has not been built, run or
browser-verified. Verification is deliberately deferred to the next model.

- CTests: `test-agent-web-presentation` follows resource and MCP invocation
  metadata through daemon JSONL and SSE; `test-tool-adapters` checks the saved
  revision and rendered Markdown; `test-tool-catalog` checks the larger reading
  result contract. Existing `test-agent-mcp-app-bridge` covers app authorization.
- When Node and installed web dependencies are available at configuration time,
  `test-agent-web-workspace` registers the relevant Vitest suites with CTest.
  C++ configuration does not install dependencies or build the frontend.
- Browser checks: authenticate or connect anonymously; expand/collapse the
  floater; open Data Explorer and another MCP App; generate a report; inspect
  a linked dataset and saved tool observation; open an original source in a
  separate tab; download the report; edit and send a queued message; close and
  review. Repeat below 1,000 px, with keyboard navigation, during an active
  response, and after a reconnect. Check that drafts, selection, source scope
  and the editor iframe survive these transitions.
