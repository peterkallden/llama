<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useAgentSession } from "./session";
import type { DaemonProviderStatus } from "./types/agent";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());
const toolsExpanded = ref(false);
const expandedTools = ref<Set<string>>(new Set());
const sessionsExpanded = ref(false);
const webClientExpanded = ref(true);

const connectionLabel = {
  connecting: "connecting",
  connected: "connected",
  disconnected: "disconnected",
  error: "connection failed",
  "auth-error": "authentication failed",
} as const;

onMounted(() => { void session.connect(); void session.refreshStatus().catch(() => undefined); });

watch(() => session.connectionState, (state) => {
  if (state !== "connected") webClientExpanded.value = true;
});

function send() {
  const value = prompt.value.trim();
  if ((!value && !session.attachments.length) || session.busy) return;
  prompt.value = "";
  void session.submit(value || "Please process the attached audio.");
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
  return String(event.event_type ?? event.type ?? "event").trim();
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

    <section class="grid">
      <div class="primary">
        <section class="card chat">
          <div class="card-title"><h2>Chat</h2><button v-if="session.busy" class="cancel-button" @click="session.cancel">Cancel</button></div>
          <div v-if="!session.messages.length" class="empty">Send a question to the connected agent.</div>
          <div v-for="message in session.messages" :key="message.id" class="message" :class="[message.role, { 'agent-error': isAgentError(message) }]"><span class="message-role">{{ message.role === "user" ? "You" : "Agent" }}</span><div class="message-body">{{ message.content || (message.pending ? "..." : "") }}</div></div>
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><div class="attachments-spacer" /><div class="input-actions"><label class="file-button" title="Attach files"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m8.5 12.5 6.2-6.2a3.5 3.5 0 0 1 5 5l-7.8 7.8a5 5 0 0 1-7.1-7.1l7.4-7.4a2.5 2.5 0 0 1 3.5 3.5l-7.1 7.1a1.5 1.5 0 1 1-2.1-2.1l6.1-6.1" /></svg><span>Attach files</span><input type="file" multiple @change="attachFiles" /></label><button type="button" class="mic-button" :class="{ recording: session.recording }" :disabled="session.busy" :title="session.recording ? 'Stop recording' : 'Record audio'" @click="session.toggleRecording"><svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="12" rx="4" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" /></svg><span>{{ session.recording ? "Stop recording" : "Record audio" }}</span></button><span v-if="session.recording" class="recording-status"><span class="recording-dot" /> recording…</span><button class="primary-button" :disabled="session.busy || (!prompt.trim() && !session.attachments.length)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 11 18-8-8 18-2.5-7.5L3 11Z" /><path d="M10.5 13.5 21 3" /></svg><span>Send</span></button></div></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments</strong><label v-for="(file, index) in session.attachments" :key="file.name" class="attachment"><input type="checkbox" checked @change="session.removeAttachment(index)" />{{ file.name }}</label></div>
        </section>

        <section class="card">
          <div class="card-title"><h2>Events</h2><span class="muted">{{ session.events.length }}</span></div>
          <div v-if="!session.events.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div class="events-viewport">
          <div v-for="(event, index) in session.events" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div class="event-line"><time class="event-time">{{ eventTime(event) }}</time><strong class="event-kind"><span v-if="eventCategory(event)">{{ eventCategory(event) }}: </span>{{ eventTypeLabel(event) }}:</strong><span class="event-detail">{{ eventText(event) }}</span><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span><button v-if="isArtifactEvent(event)" class="download-button" @click="session.downloadArtifact(event)">Download</button></div></div>
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
