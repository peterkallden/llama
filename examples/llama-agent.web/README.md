# llama-agent web example

This is a small Vue 3/TypeScript test client for `llama-agent-web`. It is not a
second agent host: the daemon owns sessions, turns, planning, tools, MCP,
resources, policy and capabilities.

The client uses:

- `POST /api/v1/turns` for questions;
- `POST /api/v1/turns/{id}/cancel` for cancellation;
- `POST /api/v1/resources` for text and binary attachments;
- `GET /api/v1/status` for status;
- `GET /api/v1/events` for JSONL events over SSE.

The JSON payload is the same as the daemon's JSONL event payload. Only the
transport framing changes from JSONL to SSE. SSE is connected with `fetch`
instead of `EventSource`, so a bearer token can be sent in the
`Authorization` header.

Text attachments use the existing `text` field. Binary attachments use the
existing byte-oriented resource-store path and are sent as bounded
`bytes_base64` JSONL payloads. The daemon applies the same 1 MiB resource
limit to both forms.

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

Vite proxies `/api` to `http://127.0.0.1:8090`. Set
`VITE_AGENT_WEB_BASE_URL` to use another base URL.

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
features such as login, event replay, binary downloads or full routing.
