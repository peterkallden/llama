<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useAgentSession } from "./session";
import type { AgentEvent } from "./types/agent";
import DataExplorer from "./components/DataExplorer.vue";
import WorkflowInspector from "./components/WorkflowInspector.vue";
import McpAppHost from "./components/McpAppHost.vue";
import { findLatestAppInvocation, mcpAppInvocationKey, mcpAppKey } from "./components/app-invocation";
import { structuredToolResult } from "./components/structured-tool-result";
import StatusFloater from "./components/StatusFloater.vue";
import ReportCard from "./components/ReportCard.vue";
import SourceInspector from "./components/SourceInspector.vue";
import { reportReferences, type InspectionTarget } from "./components/report-view";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());
const eventsExpanded = ref(false);
const eventsViewport = ref<HTMLElement | null>(null);
const activeView = ref<"chat" | "data" | "workflow" | "app">("chat");
const activeAppTool = ref("");
const documentMode = ref(false);
const chatSending = ref(false);
const workspaceNavigating = ref(false);
const documentStatus = ref<{ title: string; revision: number; dirty: boolean; uri?: string; message?: string; error?: boolean }>({ title: "Document", revision: 0, dirty: false });
const editorHost = ref<InstanceType<typeof McpAppHost> | null>(null);
const savedWorkspaceState = ref<{ view: typeof activeView.value; app: string } | null>(null);
const localDocumentEvent = ref<AgentEvent>();
const openingAttachmentUri = ref("");
const attachmentEditorError = ref("");
const inspection = ref<InspectionTarget>();
const workspaceActive = computed(() => activeView.value !== "chat");
const reports = computed(() => reportReferences(session.events, session.sessionId));
const orphanReports = computed(() => reports.value.filter((report) => !session.messages.some((message) => message.role === "assistant" && message.turnId === report.turnId)));
const canEdit = computed(() => !documentMode.value && session.tools.some((tool) => tool.name === "document.save" && tool.app_write_authorized));
const navigationLocked = computed(() => documentMode.value || workspaceNavigating.value || Boolean(openingAttachmentUri.value));
const sessionEvents = computed(() => session.events.filter((event) => !event.session_id || event.session_id === session.sessionId));
function reportsForTurn(turnId: string) { return reports.value.filter((report) => report.turnId === turnId); }
function openReportRead(uri: string) { return session.callReadOnlyTool("document.open", { document_uri: uri }); }
function handoff(text: string) { prompt.value = text; void navigateWorkspace("chat"); }
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
let cachedInvocation: { id: string; input: Record<string, unknown>; result?: unknown } | undefined;
const activeAppInvocation = computed(() => {
  const app = activeMcpApp();
  if (!app) return undefined;
  const events = documentMode.value && localDocumentEvent.value
    ? session.events.filter((event) => event.tool_input?.document_uri === documentStatus.value.uri ||
        event.tool_input?.document_uri === localDocumentEvent.value?.tool_input?.document_uri)
    : session.events;
  const event = findLatestAppInvocation(events, session.tools, app, session.sessionId, localDocumentEvent.value);
  if (!event) return undefined;
  const id = mcpAppInvocationKey(event);
  if (cachedInvocation?.id === id) return cachedInvocation;
  cachedInvocation = {
    id,
    input: event.tool_input as Record<string, unknown>,
    ...(event.tool_result !== undefined ? { result: event.tool_result } : {}),
  };
  return cachedInvocation;
});

watch(() => session.sessionId, () => {
  localDocumentEvent.value = undefined;
  restoreDocumentMode();
  activeView.value = "chat";
  activeAppTool.value = "";
  attachmentEditorError.value = "";
  openingAttachmentUri.value = "";
  inspection.value = undefined;
});

onMounted(() => { void session.connect(); void session.refreshStatus().catch((cause) => {
  session.error = cause instanceof Error ? cause.message : String(cause);
}); });

watch(() => session.events.length, () => {
  void nextTick(() => {
    if (eventsViewport.value) eventsViewport.value.scrollTop = 0;
  });
});

let sendGeneration = 0;
async function send() {
  const draftAtSubmit = prompt.value;
  const value = draftAtSubmit.trim();
  if ((!value && !session.selectedResourceRefs.length) || chatSending.value) return;
  const generation = ++sendGeneration;
  chatSending.value = true;
  const sendingSession = session.sessionId;
  const sendingResources = [...session.selectedResourceRefs];
  try {
    if (documentMode.value && !editorHost.value) throw new Error("The document editor is not ready. Your message is preserved.");
    const context = documentMode.value ? await editorHost.value!.prepareForChat() : undefined;
    if (session.sessionId !== sendingSession) throw new Error("The session changed before the message could be sent.");
    const turnId = crypto.randomUUID();
    const response = session.submit(value || "Please inspect the selected attachments.", [], false, context,
      turnId, sendingResources, undefined);
    // submit registers the message/queue synchronously before awaiting HTTP.
    // Only the save/registration operation owns the composer, not model runtime.
    if (session.messages.some((message) => message.id === `${turnId}-user`)) {
      if (prompt.value === draftAtSubmit) prompt.value = "";
      session.selectedResourceRefs = session.selectedResourceRefs.filter((uri) => !sendingResources.includes(uri));
    }
    chatSending.value = false;
    const sent = await response;
    if (!sent && generation === sendGeneration && session.sessionId === sendingSession && !prompt.value) prompt.value = draftAtSubmit;
  } catch (cause) {
    if (session.sessionId === sendingSession) session.error = cause instanceof Error ? cause.message : String(cause);
  } finally { if (generation === sendGeneration) chatSending.value = false; }
}

function enterDocumentMode() {
  if (!savedWorkspaceState.value) savedWorkspaceState.value = {
    view: activeView.value, app: activeAppTool.value,
  };
  documentMode.value = true; activeView.value = "app";
}

function restoreDocumentMode() {
  if (!documentMode.value && !savedWorkspaceState.value) return;
  documentMode.value = false;
  const previous = savedWorkspaceState.value;
  if (previous) {
    activeView.value = previous.view;
    activeAppTool.value = previous.app;
  } else activeView.value = "chat";
  savedWorkspaceState.value = null;
}

async function navigateWorkspace(view: typeof activeView.value, appKey = "", closing = false) {
  if (workspaceNavigating.value || openingAttachmentUri.value) return;
  if (documentMode.value && !closing) return;
  if (view === activeView.value && (view !== "app" || appKey === activeAppTool.value)) return;
  workspaceNavigating.value = true;
  try {
    if (documentMode.value) {
      try {
        if (!editorHost.value) throw new Error("The document editor is not ready to save and close.");
        await editorHost.value.closeDocument();
      }
      catch (cause) { session.error = cause instanceof Error ? cause.message : String(cause); return; }
      restoreDocumentMode();
      if (closing) return;
    }
    activeView.value = view;
    if (appKey) {
      activeAppTool.value = appKey;
      documentMode.value = activeMcpApp()?._meta?.ui?.resourceUri === "ui://llama-agent/apps/document-editor";
    }
  } finally { workspaceNavigating.value = false; }
}

async function startNewSession() {
  if (navigationLocked.value) return;
  workspaceNavigating.value = true;
  try {
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
    title: String(value.title ?? "Document"), revision: Number(value.revision ?? 0), uri: String(value.document_uri ?? ""),
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
  if (navigationLocked.value) return;
  openingAttachmentUri.value = sourceUri;
  const importingSession = session.sessionId;
  try {
    const imported = structuredToolResult(await session.importDocument(sourceUri));
    if (session.sessionId !== importingSession) return;
    const documentUri = String(imported.document_uri ?? "");
    if (!documentUri) throw new Error("Import completed without a document revision reference.");
    const opened = structuredToolResult(await session.callMcpAppTool("document.open", "document.open", { document_uri: documentUri }));
    if (session.sessionId !== importingSession) return;
    showOpenedDocument(opened, documentUri);
  } catch (cause) {
    if (session.sessionId === importingSession) attachmentEditorError.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (openingAttachmentUri.value === sourceUri) openingAttachmentUri.value = "";
  }
}

function showOpenedDocument(opened: Record<string, unknown>, documentUri: string) {
  const openTool = session.tools.find((tool) => tool.name === "document.open" && tool._meta?.ui?.resourceUri);
  if (!openTool) throw new Error("The document editor app is unavailable.");
  localDocumentEvent.value = {
    session_id: session.sessionId, request_id: crypto.randomUUID(), turn_id: crypto.randomUUID(),
    sequence: Date.now(), client_received_at: Date.now(), provider_id: openTool.provider_id ?? "",
    server_tool_name: openTool.server_tool_name ?? openTool.name, app_context_available: true,
    tool_input: { document_uri: documentUri }, tool_result: { isError: false, structuredContent: opened },
  };
  documentStatus.value = { title: String(opened.title || "Document"), revision: Number(opened.revision || 0), dirty: false, uri: documentUri };
  enterDocumentMode();
  activeAppTool.value = mcpAppKey(openTool);
}

async function editReport(uri: string) {
  if (navigationLocked.value) return;
  workspaceNavigating.value = true;
  const sessionId = session.sessionId;
  try {
    const opened = structuredToolResult(await openReportRead(uri));
    if (session.sessionId === sessionId) showOpenedDocument(opened, uri);
  } catch (cause) { if (session.sessionId === sessionId) session.error = cause instanceof Error ? cause.message : String(cause); }
  finally { workspaceNavigating.value = false; }
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
    <header class="topbar"><div><p class="eyebrow">WORKSPACE</p><h1>llama-agent</h1></div><span class="session-caption">{{ !session.connected ? 'Disconnected' : session.busy ? 'Agent working' : 'Ready for your next question' }}</span></header>
    <div class="workspace-toolbar">
      <nav class="workspace-nav" aria-label="Workspace">
        <button type="button" :class="{ selected: activeView === 'chat' }" :disabled="navigationLocked" :title="documentMode ? 'Close the document tab to return to chat' : undefined" @click="navigateWorkspace('chat')">Chat</button>
        <div v-if="workspaceActive" class="workspace-tab" :aria-busy="workspaceNavigating">
          <span class="workspace-title">{{ documentMode ? documentStatus.title : activeView === 'data' ? 'Data Explorer' : activeView === 'workflow' ? 'Workflow Inspector' : activeMcpApp()?.name || 'Application' }}</span>
          <button type="button" class="workspace-close" :disabled="workspaceNavigating || Boolean(openingAttachmentUri)" :title="documentMode ? 'Save, review and close' : 'Close workspace'" :aria-label="documentMode ? 'Save, review and close document' : 'Close workspace'" @click="navigateWorkspace('chat', '', true)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg></button>
        </div>
      </nav>
      <StatusFloater :active-view="activeView" :active-app="activeAppTool" :navigation-locked="navigationLocked" @navigate="navigateWorkspace" @new-session="startNewSession"><template #events>
        <section class="events-panel">
          <button class="section-toggle" @click="eventsExpanded = !eventsExpanded"><h2>Events</h2><span>{{ eventsExpanded ? '−' : '+' }} · {{ sessionEvents.length }}</span></button>
          <template v-if="eventsExpanded">
          <div v-if="!sessionEvents.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div ref="eventsViewport" class="events-viewport">
          <div v-for="(event, index) in sessionEvents" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div class="event-line"><time class="event-time">{{ eventTime(event) }}</time><strong v-if="eventCategory(event) || eventTypeLabel(event)" class="event-kind"><span v-if="eventCategory(event)">{{ eventCategory(event) }}: </span><span v-if="eventTypeLabel(event)">{{ eventTypeLabel(event) }}:</span></strong><span class="event-detail">{{ eventText(event) }}</span><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span><button v-if="isArtifactEvent(event)" class="download-button" @click="session.downloadArtifact(event)">Download</button></div></div>
            <pre v-if="expanded.has(index)">{{ JSON.stringify(event, null, 2) }}</pre>
          </div>
          </div>
          </template>
        </section>
      </template></StatusFloater>
    </div>
    <section class="grid" :class="{ 'workspace-layout': workspaceActive, 'chat-layout': !workspaceActive }">
      <div v-show="workspaceActive" class="workspace-main">
        <DataExplorer v-if="activeView === 'data'" :key="session.sessionId" :tools="session.tools" :call-tool="session.callReadOnlyTool" @handoff="handoff" />
        <WorkflowInspector v-else-if="activeView === 'workflow'" :events="session.events" />
        <McpAppHost v-else-if="activeView === 'app' && activeMcpApp()" ref="editorHost" :key="JSON.stringify([session.sessionId, mcpAppKey(activeMcpApp()!)])" :tool="activeMcpApp()!" :tools="session.tools" :session-id="session.sessionId" :read-resource="session.readMcpAppResource" :call-tool="session.callMcpAppTool" :submit-review="session.submitReview" :invocation="activeAppInvocation" @document-status="onDocumentStatus" @document-closed="onDocumentClosed" />
      </div>
      <div class="primary">
        <section class="card chat chat-panel">
          <div class="card-title chat-heading"><h2>Chat</h2><div v-if="documentMode" class="document-chat-status" :class="{ dirty: documentStatus.dirty, 'status-error': documentStatus.error }" :title="documentStatus.message"><strong>{{ documentStatus.title }}</strong><span>revision {{ documentStatus.revision }}</span><span>{{ documentStatus.dirty ? 'unsaved changes' : 'saved' }}</span></div></div>
          <div v-if="!session.messages.length" class="empty">Send a question to the connected agent.</div>
          <div v-for="message in session.messages" :key="message.id" class="chat-entry">
            <div class="message" :class="[message.role, { 'agent-error': isAgentError(message) }]"><span class="message-role">{{ message.role === "user" ? "You" : "Agent" }}</span><div class="message-body">{{ message.content || (message.pending ? "..." : "") }}</div></div>
            <template v-if="message.role === 'assistant'"><ReportCard v-for="report in reportsForTurn(message.turnId)" :key="report.uri" :uri="report.uri" :name="report.name" :session-id="session.sessionId" :editable="canEdit" :open-document="openReportRead" :read-json="session.readJsonResource" @edit="editReport" @inspect="inspection = $event" /></template>
          </div>
          <ReportCard v-for="report in orphanReports" :key="report.uri" :uri="report.uri" :name="report.name" :session-id="session.sessionId" :editable="canEdit" :open-document="openReportRead" :read-json="session.readJsonResource" @edit="editReport" @inspect="inspection = $event" />
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><div class="attachments-spacer" /><div class="input-actions"><label class="file-button" title="Attach files" aria-label="Attach files"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg><input type="file" multiple @change="attachFiles" /></label><button type="button" class="mic-button" :class="{ recording: session.recording }" :disabled="session.busy && !session.recording" :title="session.recording ? 'Stop recording' : 'Record audio'" :aria-label="session.recording ? 'Stop recording' : 'Record audio'" @click="session.toggleRecording"><svg aria-hidden="true" viewBox="0 0 24 24"><rect x="8" y="3" width="8" height="12" rx="4" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8 21h8" /></svg></button><span v-if="session.recording" class="recording-status"><span class="recording-dot" /> recording…</span><button class="primary-button" :disabled="chatSending || (!prompt.trim() && !session.selectedResourceRefs.length)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 11 18-8-8 18-2.5-7.5L3 11Z" /><path d="M10.5 13.5 21 3" /></svg><span>{{ chatSending ? 'Saving…' : session.busy ? 'Queue' : 'Send' }}</span></button></div></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments — select for this message</strong><div v-for="(file, index) in session.attachments" :key="`${file.uri ?? file.name}-${index}`" class="attachment"><label><input v-model="session.selectedResourceRefs" type="checkbox" :value="file.uri" :disabled="!file.uri" />{{ file.name }}</label><button type="button" class="attachment-editor-button" :aria-label="`Remove ${file.name} from session`" title="Remove attachment from this session; the original file is kept" @click="session.removeAttachment(index)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M3 6h18M9 6V4h6v2M5 6l1 14h12l1-14M10 10v6M14 10v6"/></svg></button><button v-if="/\.(txt|md|markdown|docx)$/i.test(file.name)" type="button" class="attachment-editor-button" :disabled="documentMode || !file.uri || !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized) || Boolean(openingAttachmentUri)" :title="documentMode ? 'Close the current document tab before opening another document' : !file.uri ? 'Upload is still in progress.' : !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized) ? 'The daemon has not authorized document import for the editor app.' : 'Redigera dokument'" aria-label="Redigera dokument" @click="openAttachmentInEditor(file)"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m14.5 6.5 3 3M4 20l4.4-.9L19 8.5a2.1 2.1 0 0 0-3-3L5.4 16.1 4 20Z"/><path d="M3 22h18"/></svg></button><small v-if="file.uri && !session.tools.some((tool) => tool.name === 'document.import' && tool.app_write_authorized)" class="muted">Document editor import is not authorized by the daemon profile.</small><small v-if="openingAttachmentUri === file.uri" class="muted">Importing the document and opening its saved revision…</small><small v-if="attachmentEditorError" class="error">{{ attachmentEditorError }}</small></div></div>
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


      </div>


    </section>
    <SourceInspector v-if="inspection" :key="JSON.stringify([session.sessionId, inspection.id])" :target="inspection" :tools="session.tools" :call-tool="session.callReadOnlyTool" :read-resource="session.readResourceBlob" @close="inspection = undefined" />
  </main>
</template>
