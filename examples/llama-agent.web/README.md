# llama-agent web example

Det här är en liten Vue 3/TypeScript-testklient för `llama-agent-web`. Den är
inte en andra agenthost: daemonen äger sessioner, turns, planering, tools, MCP,
resources, policy och capabilities.

Klienten använder:

- `POST /api/v1/turns` för frågor;
- `POST /api/v1/turns/{id}/cancel` för avbrytning;
- `POST /api/v1/resources` för textbilagor;
- `GET /api/v1/status` för status;
- `GET /api/v1/events` för JSONL-event som SSE.

JSON-payloaden är densamma som daemonens JSONL-event. Endast transportens
framing ändras från JSONL till SSE. SSE ansluts med `fetch` i stället för
`EventSource`, så en bearer-token kan skickas i `Authorization`-headern.

## Bygg och lokal utveckling

Bygg web-adaptern och klienten separat. C++-adaptern kopplar webben till
daemonens JSONL/TCP-port; Vue-klienten är statiska filer.

```bash
cmake --build build-agent --target llama-agent-web -j4

npm install
npm test
npm run build
```

Det sista kommandot skapar `dist/`.

Starta daemonen och `llama-agent-web` enligt
[`agent-daemon-usage.md`](../../docs/agent/agent-daemon-usage.md). Installera
sedan frontend-beroenden och kör:

```bash
npm install
npm run dev
```

Vite proxar `/api` till `http://127.0.0.1:8090`. För annan adress används
`VITE_AGENT_WEB_BASE_URL`.

## Enkel Nginx-uppsättning

För en lokal eller intern Linux-installation kan Nginx servera `dist/` och
proxy:a `/api/` till `llama-agent-web`:

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

Exemplet lyssnar på `http://127.0.0.1:8080`, serverar klienten från
`/opt/llama-agent-web/dist` och vidarebefordrar `/api/` till
`llama-agent-web` på `127.0.0.1:8090`. Kör daemonens TCP-port och web-adaptern
på loopback enligt daemon-dokumentationen. För nätverksåtkomst ska Nginx få
TLS och autentisering framför sig; den medföljande konfigurationen är inte en
produktionssäkerhetsprofil.

Konfigurationen stänger av proxy-buffering och använder lång timeout för
SSE. Den vidarebefordrar browserns `Authorization`-header till web-adaptern.
Klienten använder fetch-baserad SSE i stället för inbyggd `EventSource`, vilket
gör bearer-token möjlig även i samma-origin- och reverse-proxy-installationer.

Det här är tre separata processroller:

```text
llama-agent-daemon  --TCP-->  llama-agent-web  --HTTP/SSE-->  Nginx/browser
```

Nginx och web-adaptern får inte ta över sessioner, planering, toolval, MCP,
resource-policy eller capability-beslut från daemonen.

Klienten är avsiktligt ett exempel och saknar ännu produktionsegenskaper som
login, event-replay, binära uploads/downloads och fullständig routing.
