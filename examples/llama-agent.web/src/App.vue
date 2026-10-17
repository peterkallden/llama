<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import { useAgentSession } from "./session";
import type { AgentEvent, DaemonProviderStatus } from "./types/agent";
import DataExplorer from "./components/DataExplorer.vue";
import WorkflowInspector from "./components/WorkflowInspector.vue";
import McpAppHost from "./components/McpAppHost.vue";
import { findAppForInvocationEvent, findLatestAppInvocation, mcpAppInvocationKey, mcpAppKey } from "./components/app-invocation";
import { structuredToolResult } from "./components/structured-tool-result";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());
const toolsExpanded = ref(false);
const expandedTools = ref<Set<string>>(new Set());
const sessionsExpanded = ref(false);
const webClientExpanded = ref(true);
const statusExpanded = ref(true);
const eventsExpanded = ref(true);
const eventsViewport = ref<HTMLElement | null>(null);
const activeView = ref<"chat" | "data" | "workflow" | "app">("chat");
const activeAppTool = ref("");
const documentMode = ref(false);
const chatSending = ref(false);
const workspaceNavigating = ref(false);
const documentStatus = ref<{ title: string; revision: number; dirty: boolean; message?: string; error?: boolean }>({ title: "Document", revision: 0, dirty: false });
const editorHost = ref<InstanceType<typeof McpAppHost> | null>(null);
const savedWorkspaceState = ref<{ view: typeof activeView.value; app: string; web: boolean; status: boolean; events: boolean } | null>(null);
const localDocumentEvent = ref<AgentEvent>();
const openingAttachmentUri = ref("");
const attachmentEditorError = ref("");
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
  const event = findLatestAppInvocation(session.events, session.tools, app, session.sessionId, localDocumentEvent.value);
  if (!event) return undefined;
  return {
    id: mcpAppInvocationKey(event),
    input: event.tool_input as Record<string, unknown>,
    ...(event.tool_result !== undefined ? { result: event.tool_result } : {}),
  };
};

watch(() => session.events[0], (event) => {
  if (documentMode.value) return;
  const targetApp = findAppForInvocationEvent(event, session.tools, session.sessionId);
  if (targetApp) {
    localDocumentEvent.value = undefined;
    if (targetApp._meta?.ui?.resourceUri === "ui://llama-agent/apps/document-editor") enterDocumentMode();
    else activeView.value = "app";
    activeAppTool.value = mcpAppKey(targetApp);
  }
}, { flush: "sync" });

watch(() => session.sessionId, () => {
  localDocumentEvent.value = undefined;
  restoreDocumentMode();
  activeView.value = "chat";
  activeAppTool.value = "";
  attachmentEditorError.value = "";
});

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

async function send() {
  const draftAtSubmit = prompt.value;
  const value = draftAtSubmit.trim();
  if ((!value && !session.selectedResourceRefs.length) || chatSending.value) return;
  chatSending.value = true;
  const sendingSession = session.sessionId;
  const sendingResources = [...session.selectedResourceRefs];
  try {
    if (documentMode.value && !editorHost.value) throw new Error("The document editor is not ready. Your message is preserved.");
    const context = documentMode.value ? await editorHost.value!.prepareForChat() : undefined;
    if (session.sessionId !== sendingSession) throw new Error("The session changed before the message could be sent.");
    const sent = await session.submit(value || "Please process the attached audio.", [], false, context,
      undefined, sendingResources, undefined);
    if (sent && session.sessionId === sendingSession) {
      if (prompt.value === draftAtSubmit) prompt.value = "";
      session.selectedResourceRefs = session.selectedResourceRefs.filter((uri) => !sendingResources.includes(uri));
    }
  } catch (cause) {
    if (session.sessionId === sendingSession) session.error = cause instanceof Error ? cause.message : String(cause);
  } finally { chatSending.value = false; }
}

function enterDocumentMode() {
  if (!savedWorkspaceState.value) savedWorkspaceState.value = {
    view: activeView.value, app: activeAppTool.value, web: webClientExpanded.value, status: statusExpanded.value, events: eventsExpanded.value,
  };
  documentMode.value = true; activeView.value = "app";
  webClientExpanded.value = false; statusExpanded.value = false; eventsExpanded.value = false;
}

function restoreDocumentMode() {
  if (!documentMode.value && !savedWorkspaceState.value) return;
  documentMode.value = false;
  const previous = savedWorkspaceState.value;
  if (previous) {
    activeView.value = previous.view;
    activeAppTool.value = previous.app;
    webClientExpanded.value = previous.web; statusExpanded.value = previous.status; eventsExpanded.value = previous.events;
  } else activeView.value = "chat";
  savedWorkspaceState.value = null;
}

async function navigateWorkspace(view: typeof activeView.value, appKey = "") {
  if (workspaceNavigating.value) return;
  workspaceNavigating.value = true;
  try {
    if (documentMode.value) {
      try {
        if (!editorHost.value) throw new Error("The document editor is not ready to save and close.");
        await editorHost.value.closeDocument();
      }
      catch (cause) { session.error = cause instanceof Error ? cause.message : String(cause); return; }
      restoreDocumentMode();
    }
    activeView.value = view;
    if (appKey) activeAppTool.value = appKey;
  } finally { workspaceNavigating.value = false; }
}

async function startNewSession() {
  if (workspaceNavigating.value) return;
  workspaceNavigating.value = true;
  try {
    if (documentMode.value) {
      try {
        if (!editorHost.value) throw new Error("The document editor is not ready to save and close.");
        await editorHost.value.closeDocument();
      }
      catch (cause) { session.error = cause instanceof Error ? cause.message : String(cause); return; }
      restoreDocumentMode();
    }
    if (session.busy || session.documentReviews.some((item) => item.state !== "completed")) {
      session.error = "The current session still has an active turn or document review. It will remain available here; start a new session after it finishes.";
      return;
    }
    session.newSession();
  } finally { workspaceNavigating.value = false; }
}

function onDocumentClosed() { restoreDocumentMode(); }
function onDocumentStatus(value: Record<string, unknown>) {
  documentStatus.value = {
    title: String(value.title ?? "Document"), revision: Number(value.revision ?? 0),
    dirty: Boolean(value.dirty), message: String(value.message ?? ""), error: Boolean(value.error),
  };
}

async function openAttachmentInEditor(file: (typeof session.attachments)[number]) {
  attachmentEditorError.value = "";
  if (!file.uri) {
    attachmentEditorError.value = "The attachment has not finished uploading yet.";
    return;
  }
  if (!session.tools.some((tool) => tool.name === "document.import" && tool.app_write_authorized)) {
    attachmentEditorError.value = "The connected daemon has not authorized document import for the editor app.";
    return;
  }
  const sourceUri = file.uri;
  if (openingAttachmentUri.value || workspaceNavigating.value) return;
  openingAttachmentUri.value = sourceUri;
  const importingSession = session.sessionId;
  try {
    if (documentMode.value) {
      if (!editorHost.value) throw new Error("The document editor is not ready to save and close.");
      await editorHost.value.closeDocument();
      if (session.sessionId !== importingSession) return;
      restoreDocumentMode();
    }
    const imported = structuredToolResult(await session.importDocument(sourceUri));
    if (session.sessionId !== importingSession) return;
    const documentUri = String(imported.document_uri ?? "");
    if (!documentUri) throw new Error("Import completed without a document revision reference.");
    const opened = structuredToolResult(await session.callMcpAppTool("document.open", "document.open", { document_uri: documentUri }));
    if (session.sessionId !== importingSession) return;
    const openTool = session.tools.find((tool) => tool.name === "document.open");
    if (!openTool) throw new Error("The document editor app is unavailable.");
    localDocumentEvent.value = {
      session_id: session.sessionId, request_id: crypto.randomUUID(), turn_id: crypto.randomUUID(),
      sequence: Date.now(), client_received_at: Date.now(), provider_id: openTool.provider_id ?? "", server_tool_name: openTool.server_tool_name ?? openTool.name,
      app_context_available: true, tool_input: { document_uri: documentUri },
      tool_result: { isError: false, structuredContent: opened },
    };
    enterDocumentMode();
    activeAppTool.value = mcpAppKey(openTool);
  } catch (cause) {
    if (session.sessionId === importingSession) attachmentEditorError.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (openingAttachmentUri.value === sourceUri) openingAttachmentUri.value = "";
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
        <button class="quiet" @click="session.connect">reconnect</button><button class="quiet" @click="startNewSession">new session</button>
      </div>
    </header>

    <section class="notice">This is an example client for the daemon's HTTP/SSE adapter. Agent planning, tools, MCP and policy remain in the daemon.</section>

    <nav class="workspace-nav" aria-label="Agent workspace">
      <button :class="{ selected: activeView === 'chat' && !documentMode }" @click="navigateWorkspace('chat')">Chat</button>
      <button :class="{ selected: activeView === 'data' }" @click="navigateWorkspace('data')">Data Explorer</button>
      <button :class="{ selected: activeView === 'workflow' }" @click="navigateWorkspace('workflow')">Workflow Inspector</button>
      <button v-for="tool in mcpApps()" :key="mcpAppKey(tool)" :class="{ selected: activeAppTool === mcpAppKey(tool) }" @click="navigateWorkspace('app', mcpAppKey(tool))">{{ tool.name }} app</button>
    </nav>

    <section class="grid" :class="{ 'document-layout': documentMode, 'chat-layout': activeView === 'chat' && !documentMode }">
      <div v-show="activeView !== 'chat' || documentMode" class="workspace-main">
        <DataExplorer v-if="activeView === 'data'" :key="session.sessionId" :tools="session.tools" :call-tool="session.callReadOnlyTool" @handoff="(text) => { void navigateWorkspace('chat'); prompt = text; }" />
        <WorkflowInspector v-else-if="activeView === 'workflow'" :events="session.events" />
        <McpAppHost v-else-if="activeView === 'app' && activeMcpApp()" ref="editorHost" :key="JSON.stringify([session.sessionId, mcpAppKey(activeMcpApp()!)])" :tool="activeMcpApp()!" :tools="session.tools" :session-id="session.sessionId" :read-resource="session.readMcpAppResource" :call-tool="session.callMcpAppTool" :submit-review="session.submitReview" :invocation="activeAppInvocation()" @document-status="onDocumentStatus" @document-closed="onDocumentClosed" />
      </div>
      <div class="primary">
        <section v-show="activeView === 'chat' || documentMode" class="card chat chat-panel">
          <div class="card-title chat-heading"><h2>Chat</h2><div v-if="documentMode" class="document-chat-status" :class="{ dirty: documentStatus.dirty, 'status-error': documentStatus.error }" :title="documentStatus.message"><strong>{{ documentStatus.title }}</strong><span>revision {{ documentStatus.revision }}</span><span>{{ documentStatus.dirty ? 'unsaved changes' : 'saved' }}</span></div></div>
          <div v-if="!session.messages.length" class="empty">Send a question to the connected agent.</div>
          <div v-for="message in session.messages" :key="message.id" class="message" :class="[message.role, { 'agent-error': isAgentError(message) }]"><span class="message-role">{{ message.role === "user" ? "You" : "Agent" }}</span><div class="message-body">{{ message.content || (message.pending ? "..." : "") }}</div></div>
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><div class="attachments-spacer" /><div class="input-actions"><label class="file-button" title="Attach files" aria-label="Attach files"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg><input type="file" multiple @change="attachFiles" /></label><button type="button" class="mic-button" :class="{ recording: session.recording }" :disabled="session.busy" :title="session.recording ? 'Stop recording' : 'Record audio'" :aria-label="session.recording ? 'Stop recording' : 'Record audio'" @click="session.toggleRecording"><svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="12" rx="4" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" /></svg></button><span v-if="session.recording" class="recording-status"><span class="recording-dot" /> recording…</span><button class="primary-button" :disabled="chatSending || (!prompt.trim() && !session.selectedResourceRefs.length)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 11 18-8-8 18-2.5-7.5L3 11Z" /><path d="M10.5 13.5 21 3" /></svg><span>{{ chatSending ? 'Saving…' : session.busy ? 'Queue' : 'Send' }}</span></button></div></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments — select for this message</strong><div v-for="(file, index) in session.attachments" :key="`${file.uri ?? file.name}-${index}`" class="attachment"><label><input v-model="session.selectedResourceRefs" type="checkbox" :value="file.uri" :disabled="!file.uri" />{{ file.name }}</label><button type="button" class="quiet" :aria-label="`Remove ${file.name}`" @click="session.removeAttachment(index)">Remove</button><button v-if="/\.(txt|md|markdown|docx)$/i.test(file.name)" type="button" class="attachment-editor-button" :disabled="!file.uri || !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized) || Boolean(openingAttachmentUri)" :title="!file.uri ? 'Upload is still in progress.' : !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized) ? 'The daemon has not authorized document import for the editor app.' : 'Redigera dokument'" aria-label="Redigera dokument" @click="openAttachmentInEditor(file)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m14.5 6.5 3 3M4 20l4.4-.9L19 8.5a2.1 2.1 0 0 0-3-3L5.4 16.1 4 20Z"/><path d="M3 22h18"/></svg></button><small v-if="file.uri && !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized)" class="muted">Document editor import is not authorized by the daemon profile.</small><small v-if="openingAttachmentUri === file.uri" class="muted">Importing the document and opening its saved revision…</small><small v-if="attachmentEditorError" class="error">{{ attachmentEditorError }}</small></div></div>
          <div v-if="session.busy" class="chat-cancel"><button class="cancel-button" @click="session.cancel">Cancel</button></div>
          <section v-if="session.documentReviews.some((item) => item.state !== 'completed')" class="document-review-queue" aria-label="Document review status">
            <div v-for="review in session.documentReviews.filter((item) => item.state !== 'completed')" :key="review.operation_id" class="review-queue-item" :class="`review-${review.state}`">
              <span>{{ review.state === 'queued' ? 'Review waiting for the current turn' : review.state === 'running' ? 'Document review running in chat' : `Review failed: ${review.error ?? 'unknown error'}` }}</span>
              <button v-if="review.state === 'failed'" class="quiet" @click="session.retryDocumentReview(review.operation_id)">Retry review</button>
            </div>
          </section>
          <section v-if="session.pendingUserTurns.length" class="document-review-queue" aria-label="Pending chat messages">
            <div v-for="turn in session.pendingUserTurns" :key="turn.turn_id" class="review-queue-item" :class="`review-${turn.state}`">
              <span>{{ turn.state === 'failed' ? `Queued message failed: ${turn.error ?? 'unknown error'}` : turn.state === 'running' ? 'Queued message running' : 'Message queued' }} — {{ turn.display_prompt }}</span>
              <button v-if="turn.state === 'failed'" class="quiet" @click="session.retryPendingUserTurn(turn.turn_id)">Retry</button>
              <button v-if="turn.state === 'failed'" class="quiet" @click="session.dismissPendingUserTurn(turn.turn_id)">Dismiss</button>
            </div>
          </section>
        </section>

        <section v-show="activeView === 'chat' || documentMode" class="card events-panel">
          <button class="section-toggle" @click="eventsExpanded = !eventsExpanded"><h2>Events</h2><span>{{ eventsExpanded ? '−' : '+' }} · {{ session.events.length }}</span></button>
          <template v-if="eventsExpanded">
          <div v-if="!session.events.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div ref="eventsViewport" class="events-viewport">
          <div v-for="(event, index) in session.events" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div class="event-line"><time class="event-time">{{ eventTime(event) }}</time><strong v-if="eventCategory(event) || eventTypeLabel(event)" class="event-kind"><span v-if="eventCategory(event)">{{ eventCategory(event) }}: </span><span v-if="eventTypeLabel(event)">{{ eventTypeLabel(event) }}:</span></strong><span class="event-detail">{{ eventText(event) }}</span><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span><button v-if="isArtifactEvent(event)" class="download-button" @click="session.downloadArtifact(event)">Download</button></div></div>
            <pre v-if="expanded.has(index)">{{ JSON.stringify(event, null, 2) }}</pre>
          </div>
          </div>
          </template>
        </section>
      </div>

      <aside class="side">
        <section class="card web-client-panel"><button class="section-toggle" :disabled="session.connectionState !== 'connected'" @click="webClientExpanded = !webClientExpanded"><h2>Web client</h2><span v-if="session.connectionState === 'connected'">{{ webClientExpanded ? "−" : "+" }}</span></button><div v-if="webClientExpanded"><p class="muted">HTTP commands go to the daemon. SSE is used only for the server event stream.</p><label class="field token-field" :class="{ 'token-missing': !session.token.trim(), 'token-unconnected': session.token.trim() && !session.connected }">Bearer token<input v-model="session.token" type="password" autocomplete="off" placeholder="enter token" :aria-invalid="!session.token.trim() || undefined" /><small v-if="!session.token.trim()">No token entered. An authenticated daemon may reject requests.</small><small v-else-if="session.connectionState === 'auth-error'">The token was rejected. Check it and reconnect.</small><small v-else-if="!session.connected">Token entered, but the event connection is not established yet. Try reconnect.</small><small v-else>Token accepted and event connection is active.</small></label><p v-if="session.connectionError" class="connection-error">{{ session.connectionError }}</p></div></section>
        <section class="card status-card"><button class="section-toggle status-heading" @click="statusExpanded = !statusExpanded"><h2>Status and capabilities</h2><span>{{ statusExpanded ? "−" : "+" }}</span></button><div v-if="statusExpanded"><div class="card-title status-refresh"><span></span><button class="quiet" @click="session.refreshStatus">refresh</button></div><dl class="status-summary"><template v-for="key in ['state', 'live', 'ready', 'worker_running', 'accepting_commands']" :key="key"><template v-if="session.status[key] !== undefined"><dt>{{ displayLabel(key) }}</dt><dd>{{ String(session.status[key]) }}</dd></template></template></dl><div class="readiness"><h3>Readiness</h3><div class="readiness-list"><div v-for="([name, value]) in readinessComponents()" :key="name" class="readiness-row"><span>{{ name }}</span><span class="status-value"><span class="status-badge" :class="`status-${statusTone(value)}`" />{{ value }}</span></div></div><div v-if="readinessToolProfile()" class="readiness-meta"><span class="meta-label">Tool profile</span><span>{{ readinessToolProfile() }}</span></div><div v-if="readinessProviders().length" class="readiness-meta provider-list"><span class="meta-label">Providers</span><span v-for="provider in readinessProviders()" :key="String(provider.id ?? provider.name ?? 'provider')" class="provider-item"><span>{{ provider.id ?? provider.name }}</span><span class="status-value"><span class="status-badge" :class="`status-${statusTone(provider.status)}`" />{{ provider.status }}</span></span></div><div v-if="readinessWarnings().length" class="readiness-warnings"><span class="meta-label">Warnings</span><p v-for="warning in readinessWarnings()" :key="warning">{{ warning }}</p></div></div><div v-if="Object.keys(metrics()).length" class="metrics"><h3>Metrics</h3><div v-for="(value, key) in metrics()" :key="key" class="metric-row"><span>{{ displayLabel(key) }}</span><strong>{{ value }}</strong></div></div><div v-if="session.capabilities.length" class="capabilities"><strong>Capabilities</strong><span v-for="capability in session.capabilities" :key="capability" class="tag">{{ capability }}</span></div><div class="session-keys"><button class="section-toggle" @click="sessionsExpanded = !sessionsExpanded"><strong>Session keys</strong><span>{{ sessionsExpanded ? "−" : "+" }}</span></button><div v-if="sessionsExpanded" class="session-key-list"><div v-if="!sessionKeys().length" class="muted">No active sessions.</div><pre v-for="(item, index) in sessionKeys()" :key="index">{{ JSON.stringify(item, null, 2) }}</pre></div></div></div></section>
        <section class="card tool-panel"><button class="tool-panel-toggle" @click="toolsExpanded = !toolsExpanded"><span><strong>MCP tools</strong><small>{{ session.tools.length }} active</small></span><span class="collapse-icon">{{ toolsExpanded ? "−" : "+" }}</span></button><p class="muted">Host-approved tools exposed to the agent.</p><div v-if="toolsExpanded" class="tool-list"><div v-if="!session.tools.length" class="empty">No tool snapshot is available.</div><div v-for="tool in session.tools" :key="tool.name" class="tool-entry"><button class="tool-name" @click="toggleTool(tool.name)"><span class="tool-dot" :class="`tool-dot-${tool.state || 'active'}`" /><span>{{ tool.name }}</span><span class="tool-chevron">{{ expandedTools.has(tool.name) ? "−" : "+" }}</span></button><div v-if="expandedTools.has(tool.name)" class="tool-details"><p>{{ tool.description || "No description provided." }}</p><span v-if="tool.source" class="tool-source">source: {{ tool.source }}</span></div></div></div></section>
      </aside>
    </section>
  </main>
</template>
