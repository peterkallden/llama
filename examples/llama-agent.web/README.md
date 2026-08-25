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

## Lokal utveckling

Starta daemonen och `llama-agent-web` enligt
[`agent-daemon-usage.md`](../../docs/agent/agent-daemon-usage.md). Installera
sedan frontend-beroenden och kör:

```bash
npm install
npm run dev
```

Vite proxar `/api` till `http://127.0.0.1:8090`. För annan adress används
`VITE_AGENT_WEB_BASE_URL`.

Klienten är avsiktligt ett exempel och saknar ännu produktionsegenskaper som
login, event-replay, binära uploads/downloads och fullständig routing.
