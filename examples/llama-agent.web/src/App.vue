<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import { useAgentSession } from "./session";
import type { AgentEvent, DaemonProviderStatus } from "./types/agent";
import DataExplorer from "./components/DataExplorer.vue";
import WorkflowInspector from "./components/WorkflowInspector.vue";
import McpAppHost from "./components/McpAppHost.vue";
import { findAppForInvocationEvent, findLatestAppInvocation, mcpAppInvocationKey, mcpAppKey } from "./components/app-invocation";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());
const toolsExpanded = ref(false);
const expandedTools = ref<Set<string>>(new Set());
const sessionsExpanded = ref(false);
const webClientExpanded = ref(true);
const eventsViewport = ref<HTMLElement | null>(null);
const activeView = ref<"chat" | "data" | "workflow" | "app">("chat");
const activeAppTool = ref("");
const localDocumentEvent = ref<AgentEvent>();
const mcpApps = () => {
  const apps = new Map<string, (typeof session.tools)[number]>();
  for (const tool of session.tools) {
    const ui = tool._meta?.ui;
    const uri = ui?.resourceUri;
    const key = mcpAppKey(tool);
    if (uri && (ui?.visibility ?? []).includes("app") && !apps.has(key)) apps.set(key, tool);
  }
  return [...apps.values()];
};
const activeMcpApp = () => mcpApps().find((tool) => mcpAppKey(tool) === activeAppTool.value);
const activeAppInvocation = () => {
  const app = activeMcpApp();
  if (!app) return undefined;
  const event = localDocumentEvent.value?.provider_id === app.provider_id
    ? localDocumentEvent.value
    : findLatestAppInvocation(session.events, session.tools, app, session.sessionId);
  if (!event) return undefined;
  return {
    id: mcpAppInvocationKey(event),
    input: event.tool_input as Record<string, unknown>,
    ...(event.tool_result !== undefined ? { result: event.tool_result } : {}),
  };
};

watch(() => session.events[0], (event) => {
  const targetApp = findAppForInvocationEvent(event, session.tools, session.sessionId);
  if (targetApp) {
    activeAppTool.value = mcpAppKey(targetApp);
    activeView.value = "app";
  }
}, { flush: "sync" });

const connectionLabel = {
  connecting: "connecting",
  connected: "connected",
  disconnected: "disconnected",
  error: "connection failed",
  "auth-error": "authentication failed",
} as const;

onMounted(() => { void session.connect(); void session.refreshStatus().catch(() => undefined); });

watch(() => session.connectionState, (state) => {
  webClientExpanded.value = state !== "connected";
});

watch(() => session.events.length, () => {
  void nextTick(() => {
    if (eventsViewport.value) eventsViewport.value.scrollTop = 0;
  });
});

function send() {
  const value = prompt.value.trim();
  if ((!value && !session.attachments.length) || session.busy) return;
  prompt.value = "";
  void session.submit(value || "Please process the attached audio.");
}

function structuredToolResult(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object") throw new Error("The daemon returned no import result.");
  const envelope = payload as Record<string, unknown>;
  const tool = envelope.tool && typeof envelope.tool === "object" ? envelope.tool as Record<string, unknown> : envelope;
  if (tool.ok === false || tool.error) throw new Error(String((tool.error as Record<string, unknown> | undefined)?.safe_summary ?? tool.content_summary ?? "Document import failed."));
  const structured = tool.structured_content && typeof tool.structured_content === "object"
    ? tool.structured_content as Record<string, unknown> : tool;
  return structured;
}

async function openAttachmentInEditor(file: (typeof session.attachments)[number]) {
  if (!file.uri || !session.tools.some((tool) => tool.name === "document.import" && tool.app_write_authorized)) return;
  try {
    const imported = structuredToolResult(await session.importDocument(file.uri));
    const documentUri = String(imported.document_uri ?? "");
    if (!documentUri) throw new Error("Import completed without a document revision reference.");
    const opened = structuredToolResult(await session.callMcpAppTool("document.open", "document.open", { document_uri: documentUri }));
    const openTool = session.tools.find((tool) => tool.name === "document.open");
    if (!openTool) throw new Error("The document editor app is unavailable.");
    localDocumentEvent.value = {
      session_id: session.sessionId, request_id: crypto.randomUUID(), turn_id: crypto.randomUUID(),
      sequence: Date.now(), provider_id: openTool.provider_id, server_tool_name: openTool.server_tool_name ?? openTool.name,
      app_context_available: true, tool_input: { document_uri: documentUri },
      tool_result: { tool: { ok: true, structured_content: opened } },
    };
    activeAppTool.value = mcpAppKey(openTool);
    activeView.value = "app";
  } catch (cause) {
    session.error = cause instanceof Error ? cause.message : String(cause);
  }
}

function attachFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  for (const file of Array.from(input.files ?? [])) void session.upload(file);
  input.value = "";
}

function toggle(index: number) {
  const next = new Set(expanded.value);
  next.has(index) ? next.delete(index) : next.add(index);
  expanded.value = next;
}

function toggleTool(name: string) {
  const next = new Set(expandedTools.value);
  next.has(name) ? next.delete(name) : next.add(name);
  expandedTools.value = next;
}

function statusTone(value: unknown) {
  const normalized = String(value ?? "unknown").toLowerCase();
  if (["ready", "available", "loaded", "healthy", "ok", "active", "true"].includes(normalized)) return "good";
  if (["failed", "unavailable", "degraded", "error", "false"].includes(normalized)) return "bad";
  return "warn";
}

function displayLabel(value: string) {
  return value.replaceAll("_", " ");
}

const readiness = () => session.status.readiness ?? {};
const readinessComponents = () => {
  const current = readiness();
  const stores = current.stores && typeof current.stores === "object" ? current.stores as Record<string, unknown> : {};
  return ([
    ["health", current.health],
    ["model", current.model],
    ["inference", current.inference],
    ...Object.entries(stores).map(([name, value]) => [`store: ${name}`, value]),
  ].filter(([, value]) => value !== undefined && value !== null)) as Array<[string, unknown]>;
};

const readinessWarnings = (): string[] => readiness().warnings ?? [];
const readinessProviders = (): DaemonProviderStatus[] => readiness().providers ?? [];
const readinessToolProfile = () => readiness().tool_profile ?? "";
const metrics = (): Record<string, unknown> => session.status.metrics ?? {};
const sessionKeys = () => Array.isArray(session.status.session_keys) ? session.status.session_keys : [];

function eventText(event: Record<string, unknown>) {
  return String(event.detail ?? event.error ?? event.tool_name ?? "");
}

function eventCategory(event: Record<string, unknown>) {
  return String(event.event_category ?? "").trim();
}

function eventTypeLabel(event: Record<string, unknown>) {
  const value = String(event.event_type ?? event.type ?? "").trim();
  return value === "event" && event.event_type == null ? "" : value;
}

function eventTime(event: Record<string, unknown>) {
  const value = event.timestamp ?? event.created_at ?? event.client_received_at;
  if (typeof value === "number") return new Date(value < 10_000_000_000 ? value * 1000 : value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  if (typeof value === "string" && value) {
    const parsed = new Date(value);
    if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  return "—:—:—";
}

function isAgentError(message: { role: string; content: string }) {
  return message.role === "assistant" && message.content.trim().toLowerCase().startsWith("agent runtime failed:");
}

function isArtifactEvent(event: Record<string, unknown>) {
  return session.eventType(event) === "tool.artifact_created" && Boolean(event.detail || event.resource_uri);
}
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div><p class="eyebrow">EXAMPLE APPLICATION</p><h1>llama-agent</h1></div>
      <div class="connection" :class="`connection-${session.connectionState}`">
        <span class="dot" /> {{ connectionLabel[session.connectionState] }}
        <button class="quiet" @click="session.connect">reconnect</button><button class="quiet" @click="session.newSession">new session</button>
      </div>
    </header>

    <section class="notice">This is an example client for the daemon's HTTP/SSE adapter. Agent planning, tools, MCP and policy remain in the daemon.</section>

    <nav class="workspace-nav" aria-label="Agent workspace">
      <button :class="{ selected: activeView === 'chat' }" @click="activeView = 'chat'">Chat</button>
      <button :class="{ selected: activeView === 'data' }" @click="activeView = 'data'">Data Explorer</button>
      <button :class="{ selected: activeView === 'workflow' }" @click="activeView = 'workflow'">Workflow Inspector</button>
      <button v-for="tool in mcpApps()" :key="mcpAppKey(tool)" :class="{ selected: activeAppTool === mcpAppKey(tool) }" @click="activeView = 'app'; activeAppTool = mcpAppKey(tool)">{{ tool.name }} app</button>
    </nav>

    <DataExplorer v-if="activeView === 'data'" :key="session.sessionId" :tools="session.tools" :call-tool="session.callReadOnlyTool" @handoff="(text) => { activeView = 'chat'; prompt = text; }" />
    <WorkflowInspector v-else-if="activeView === 'workflow'" :events="session.events" />
    <McpAppHost v-else-if="activeView === 'app' && activeMcpApp()" :key="JSON.stringify([session.sessionId, mcpAppKey(activeMcpApp()!)])" :tool="activeMcpApp()!" :tools="session.tools" :session-id="session.sessionId" :read-resource="session.readMcpAppResource" :call-tool="session.callMcpAppTool" :submit-review="session.submitReview" :invocation="activeAppInvocation()" />

    <section v-else class="grid">
      <div class="primary">
        <section class="card chat">
          <div class="card-title"><h2>Chat</h2></div>
          <div v-if="!session.messages.length" class="empty">Send a question to the connected agent.</div>
          <div v-for="message in session.messages" :key="message.id" class="message" :class="[message.role, { 'agent-error': isAgentError(message) }]"><span class="message-role">{{ message.role === "user" ? "You" : "Agent" }}</span><div class="message-body">{{ message.content || (message.pending ? "..." : "") }}</div></div>
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><div class="attachments-spacer" /><div class="input-actions"><label class="file-button" title="Attach files" aria-label="Attach files"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg><input type="file" multiple @change="attachFiles" /></label><button type="button" class="mic-button" :class="{ recording: session.recording }" :disabled="session.busy" :title="session.recording ? 'Stop recording' : 'Record audio'" :aria-label="session.recording ? 'Stop recording' : 'Record audio'" @click="session.toggleRecording"><svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="12" rx="4" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" /></svg></button><span v-if="session.recording" class="recording-status"><span class="recording-dot" /> recording…</span><button class="primary-button" :disabled="session.busy || (!prompt.trim() && !session.attachments.length)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 11 18-8-8 18-2.5-7.5L3 11Z" /><path d="M10.5 13.5 21 3" /></svg><span>Send</span></button></div></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments</strong><div v-for="(file, index) in session.attachments" :key="`${file.uri ?? file.name}-${index}`" class="attachment"><label><input type="checkbox" checked @change="session.removeAttachment(index)" />{{ file.name }}</label><button v-if="/\.(txt|md|markdown|docx)$/i.test(file.name)" type="button" class="quiet" :disabled="!file.uri || !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized)" @click="openAttachmentInEditor(file)">Open in editor</button></div></div>
          <div v-if="session.busy" class="chat-cancel"><button class="cancel-button" @click="session.cancel">Cancel</button></div>
        </section>

        <section class="card">
          <div class="card-title"><h2>Events</h2><span class="muted">{{ session.events.length }}</span></div>
          <div v-if="!session.events.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div ref="eventsViewport" class="events-viewport">
          <div v-for="(event, index) in session.events" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div class="event-line"><time class="event-time">{{ eventTime(event) }}</time><strong v-if="eventCategory(event) || eventTypeLabel(event)" class="event-kind"><span v-if="eventCategory(event)">{{ eventCategory(event) }}: </span><span v-if="eventTypeLabel(event)">{{ eventTypeLabel(event) }}:</span></strong><span class="event-detail">{{ eventText(event) }}</span><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span><button v-if="isArtifactEvent(event)" class="download-button" @click="session.downloadArtifact(event)">Download</button></div></div>
            <pre v-if="expanded.has(index)">{{ JSON.stringify(event, null, 2) }}</pre>
          </div>
          </div>
        </section>
      </div>

      <aside class="side">
        <section class="card web-client-panel"><button class="section-toggle" :disabled="session.connectionState !== 'connected'" @click="webClientExpanded = !webClientExpanded"><h2>Web client</h2><span v-if="session.connectionState === 'connected'">{{ webClientExpanded ? "−" : "+" }}</span></button><div v-if="webClientExpanded"><p class="muted">HTTP commands go to the daemon. SSE is used only for the server event stream.</p><label class="field token-field" :class="{ 'token-missing': !session.token.trim(), 'token-unconnected': session.token.trim() && !session.connected }">Bearer token<input v-model="session.token" type="password" autocomplete="off" placeholder="enter token" :aria-invalid="!session.token.trim() || undefined" /><small v-if="!session.token.trim()">No token entered. An authenticated daemon may reject requests.</small><small v-else-if="session.connectionState === 'auth-error'">The token was rejected. Check it and reconnect.</small><small v-else-if="!session.connected">Token entered, but the event connection is not established yet. Try reconnect.</small><small v-else>Token accepted and event connection is active.</small></label><p v-if="session.connectionError" class="connection-error">{{ session.connectionError }}</p></div></section>
        <section class="card status-card"><div class="card-title"><h2>Status and capabilities</h2><button class="quiet" @click="session.refreshStatus">refresh</button></div><dl class="status-summary"><template v-for="key in ['state', 'live', 'ready', 'worker_running', 'accepting_commands']" :key="key"><template v-if="session.status[key] !== undefined"><dt>{{ displayLabel(key) }}</dt><dd>{{ String(session.status[key]) }}</dd></template></template></dl><div class="readiness"><h3>Readiness</h3><div class="readiness-list"><div v-for="([name, value]) in readinessComponents()" :key="name" class="readiness-row"><span>{{ name }}</span><span class="status-value"><span class="status-badge" :class="`status-${statusTone(value)}`" />{{ value }}</span></div></div><div v-if="readinessToolProfile()" class="readiness-meta"><span class="meta-label">Tool profile</span><span>{{ readinessToolProfile() }}</span></div><div v-if="readinessProviders().length" class="readiness-meta provider-list"><span class="meta-label">Providers</span><span v-for="provider in readinessProviders()" :key="String(provider.id ?? provider.name ?? 'provider')" class="provider-item"><span>{{ provider.id ?? provider.name }}</span><span class="status-value"><span class="status-badge" :class="`status-${statusTone(provider.status)}`" />{{ provider.status }}</span></span></div><div v-if="readinessWarnings().length" class="readiness-warnings"><span class="meta-label">Warnings</span><p v-for="warning in readinessWarnings()" :key="warning">{{ warning }}</p></div></div><div v-if="Object.keys(metrics()).length" class="metrics"><h3>Metrics</h3><div v-for="(value, key) in metrics()" :key="key" class="metric-row"><span>{{ displayLabel(key) }}</span><strong>{{ value }}</strong></div></div><div v-if="session.capabilities.length" class="capabilities"><strong>Capabilities</strong><span v-for="capability in session.capabilities" :key="capability" class="tag">{{ capability }}</span></div><div class="session-keys"><button class="section-toggle" @click="sessionsExpanded = !sessionsExpanded"><strong>Session keys</strong><span>{{ sessionsExpanded ? "−" : "+" }}</span></button><div v-if="sessionsExpanded" class="session-key-list"><div v-if="!sessionKeys().length" class="muted">No active sessions.</div><pre v-for="(item, index) in sessionKeys()" :key="index">{{ JSON.stringify(item, null, 2) }}</pre></div></div></section>
        <section class="card tool-panel"><button class="tool-panel-toggle" @click="toolsExpanded = !toolsExpanded"><span><strong>MCP tools</strong><small>{{ session.tools.length }} active</small></span><span class="collapse-icon">{{ toolsExpanded ? "−" : "+" }}</span></button><p class="muted">Host-approved tools exposed to the agent.</p><div v-if="toolsExpanded" class="tool-list"><div v-if="!session.tools.length" class="empty">No tool snapshot is available.</div><div v-for="tool in session.tools" :key="tool.name" class="tool-entry"><button class="tool-name" @click="toggleTool(tool.name)"><span class="tool-dot" :class="`tool-dot-${tool.state || 'active'}`" /><span>{{ tool.name }}</span><span class="tool-chevron">{{ expandedTools.has(tool.name) ? "−" : "+" }}</span></button><div v-if="expandedTools.has(tool.name)" class="tool-details"><p>{{ tool.description || "No description provided." }}</p><span v-if="tool.source" class="tool-source">source: {{ tool.source }}</span></div></div></div></section>
      </aside>
    </section>
  </main>
</template>
