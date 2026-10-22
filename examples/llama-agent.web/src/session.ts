import { computed, reactive, ref } from "vue";
import { AgentApi, AgentApiError, type WebChat, type WebProject } from "./api/agent-api";
import type { AgentEvent, ChatMessage, DaemonStatus, DaemonToolStatus, PutResourceResult, TurnRequest, TurnResult, UploadedResource } from "./types/agent";

const apiBase = import.meta.env.VITE_AGENT_WEB_BASE_URL || "/api/v1";
const token = ref(import.meta.env.VITE_AGENT_WEB_TOKEN || "");
const api = computed(() => new AgentApi(apiBase, token.value));
const events = ref<AgentEvent[]>([]);
const status = ref<DaemonStatus>({ state: "unknown" });
const capabilities = computed(() => Array.isArray(status.value.capabilities)
  ? status.value.capabilities.map(String)
  : []);
const tools = computed<DaemonToolStatus[]>(() => Array.isArray(status.value.readiness?.tools)
  ? status.value.readiness.tools.filter((tool): tool is DaemonToolStatus => Boolean(tool && typeof tool.name === "string"))
  : []);
const connected = ref(false);
const connectionState = ref<"connecting" | "connected" | "disconnected" | "error" | "auth-error">("disconnected");
const connectionError = ref("");
const busy = ref(false);
const error = ref("");
const messages = ref<ChatMessage[]>([]);
const attachments = ref<UploadedResource[]>([]);
const resourceRefs = ref<string[]>([]);
const selectedResourceRefs = ref<string[]>([]);
const activeTurnId = ref("");
const recording = ref(false);
const abortController = ref<AbortController | null>(null);
let mediaRecorder: MediaRecorder | null = null;
let mediaStream: MediaStream | null = null;
const sessionStorageKey = "llama-agent-web.session-id";
const navigationSchemaKey = "llama-agent-web.navigation-schema";
const storedSessionId = sessionStorage.getItem(sessionStorageKey);
let legacyMigrationPending = Boolean(storedSessionId && sessionStorage.getItem(navigationSchemaKey) !== "1");
const sessionId = ref(storedSessionId || `web-${crypto.randomUUID()}`);
sessionStorage.setItem(sessionStorageKey, sessionId.value);
const projectId = ref(sessionStorage.getItem("llama-agent-web.project-id") || "");
const chatTitle = ref(sessionStorage.getItem("llama-agent-web.chat-title") || "New chat");
const historyAvailable = ref(true);
const historyCursor = ref("");
const projects = ref<WebProject[]>([]);
const chats = ref<WebChat[]>([]);
const chatCursor = ref("");
const navigationLoading = ref(false);
const runningChats = ref<Record<string, string>>({});
export type DocumentReviewState = "queued" | "running" | "failed" | "completed";
export type PendingDocumentReview = {
  operation_id: string;
  turn_id: string;
  session_id: string;
  prompt: string;
  document_context: NonNullable<TurnRequest["document_context"]>;
  state: DocumentReviewState;
  error?: string;
};
const documentReviews = ref<PendingDocumentReview[]>(restoreDocumentReviews());
let drainingDocumentReviews = false;
let attachmentOrientationTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
export type AttachmentOrientationState = "queued" | "running" | "failed" | "completed";
export type PendingAttachmentOrientation = {
  operation_id: string;
  turn_id: string;
  session_id: string;
  resource_uri: string;
  name: string;
  state: AttachmentOrientationState;
  error?: string;
};
const attachmentOrientations = ref<PendingAttachmentOrientation[]>(restoreAttachmentOrientations());
let turnRequestInFlight = false;
let turnRequestController: AbortController | null = null;
let turnRequestFailure = "";
let sessionEpoch = 0;
let disposed = false;
let navigationGeneration = 0;
let chatOpenGeneration = 0;

function reviewQueueKey(id = sessionId.value) { return `llama-agent-web.document-reviews:${id}`; }
function restoreDocumentReviews(): PendingDocumentReview[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(reviewQueueKey()) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is PendingDocumentReview => item && typeof item === "object" &&
      typeof item.operation_id === "string" && typeof item.turn_id === "string" && item.session_id === sessionId.value &&
      typeof item.prompt === "string" && item.document_context && typeof item.document_context.document_uri === "string" &&
      Number.isSafeInteger(item.document_context.revision) && Array.isArray(item.document_context.block_ids))
      .map((item) => item.state === "completed" || item.state === "failed"
        ? item
        : item.state === "running"
          ? { ...item, state: "failed" as const, error: "This review may have reached the daemon. Check the chat history before retrying." }
          : { ...item, state: "queued" as const });
  } catch { return []; }
}
function persistDocumentReviews(): boolean {
  try {
    sessionStorage.setItem(reviewQueueKey(), JSON.stringify(documentReviews.value));
    return true;
  } catch {
    error.value = "A pending document review could not be persisted in this browser session.";
    return false;
  }
}
function orientationQueueKey(id = sessionId.value) { return `llama-agent-web.attachment-orientations:${id}`; }
function restoreAttachmentOrientations(): PendingAttachmentOrientation[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(orientationQueueKey()) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is PendingAttachmentOrientation => item && typeof item === "object" &&
      typeof item.operation_id === "string" && typeof item.turn_id === "string" && item.session_id === sessionId.value &&
      typeof item.resource_uri === "string" && typeof item.name === "string")
      .map((item) => item.state === "completed" || item.state === "failed"
        ? item
        : item.state === "running"
          ? { ...item, state: "failed" as const, error: "This orientation may have reached the daemon. Check the chat history before retrying." }
          : { ...item, state: "queued" as const });
  } catch { return []; }
}
function persistAttachmentOrientations() {
  try {
    sessionStorage.setItem(orientationQueueKey(), JSON.stringify(attachmentOrientations.value));
  } catch { error.value = "A pending attachment orientation could not be persisted in this browser session."; }
}
export type PendingUserTurn = {
  turn_id: string;
  session_id: string;
  prompt: string;
  resource_refs: string[];
  display_prompt: string;
  document_context?: TurnRequest["document_context"];
  investigation_context?: TurnRequest["investigation_context"];
  attachment_context_required?: true;
  attachment_orientation?: true;
  state: "queued" | "running" | "failed";
  error?: string;
};
function copyInvestigationContext(context: NonNullable<TurnRequest["investigation_context"]>) {
  return { case_id: context.case_id, revision: context.revision,
    entity_ids: [...context.entity_ids], relation_ids: [...context.relation_ids], event_ids: [...context.event_ids] };
}
function userTurnQueueKey(id = sessionId.value) { return `llama-agent-web.pending-user-turns:${id}`; }
function restorePendingUserTurns(): PendingUserTurn[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(userTurnQueueKey()) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is PendingUserTurn => item && typeof item === "object" &&
      typeof item.turn_id === "string" && item.session_id === sessionId.value && typeof item.prompt === "string" &&
      Array.isArray(item.resource_refs) && item.resource_refs.every((uri: unknown) => typeof uri === "string") &&
      typeof item.display_prompt === "string" &&
      (!item.document_context || (typeof item.document_context.document_uri === "string" &&
        Number.isSafeInteger(item.document_context.revision) && Array.isArray(item.document_context.block_ids))) &&
      (!item.investigation_context || (typeof item.investigation_context.case_id === "string" &&
        Number.isSafeInteger(item.investigation_context.revision) && Array.isArray(item.investigation_context.entity_ids) &&
        Array.isArray(item.investigation_context.relation_ids) && Array.isArray(item.investigation_context.event_ids))))
      .map((item) => item.state === "failed"
        ? item
        : item.state === "running"
          ? { ...item, state: "failed" as const, error: "This message may have reached the daemon. Check the chat history before retrying." }
          : { ...item, state: "queued" as const });
  } catch { return []; }
}
function persistPendingUserTurns(): boolean {
  try {
    sessionStorage.setItem(userTurnQueueKey(), JSON.stringify(pendingUserTurns.value));
    return true;
  } catch {
    error.value = "A queued chat message could not be persisted in this browser session.";
    return false;
  }
}
const pendingUserTurns = ref<PendingUserTurn[]>(restorePendingUserTurns());

function base64Encode(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function eventType(event: AgentEvent): string {
  return String(event.event_type || event.type || "event");
}

function eventResourceUri(event: AgentEvent): string {
  return typeof event.resource_uri === "string" && event.resource_uri
    ? event.resource_uri
    : eventType(event) === "tool.artifact_created" && typeof event.detail === "string"
      ? event.detail
      : "";
}

function observe(event: AgentEvent) {
  // Daemon events do not consistently carry a presentation timestamp. Keep
  // the server payload intact while recording when the browser received it.
  events.value.unshift({ ...event, client_received_at: Date.now() });
  const kind = eventType(event);
  const turnId = String(event.turn_id ?? activeTurnId.value);
  const eventSession = String(event.session_id ?? sessionId.value);
  if (kind === "turn.started" || kind === "turn.queued") runningChats.value[eventSession] = turnId;
  if (kind === "turn.completed" || kind === "turn.failed" || kind === "turn.cancelled") {
    if (runningChats.value[eventSession] === turnId) {
      const next = { ...runningChats.value };
      delete next[eventSession];
      runningChats.value = next;
    }
  }
  if (eventSession !== sessionId.value) return;
  if (event.turn_id && turnId !== activeTurnId.value && kind.startsWith("turn.")) return;
  const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
  if (kind === "response.delta" && assistant) assistant.content += String(event.text ?? event.response ?? "");
  if (kind === "turn.completed" && assistant && typeof event.response === "string") assistant.content = event.response;
  if (kind === "turn.failed") error.value = String(event.error ?? event.detail ?? "Turn failed");
  if (kind === "turn.completed" || kind === "turn.failed" || kind === "turn.cancelled") {
    if (turnId === activeTurnId.value && kind !== "turn.completed") {
      turnRequestFailure = String(event.error ?? event.detail ?? (kind === "turn.cancelled" ? "Turn cancelled" : "Turn failed"));
      turnRequestController?.abort();
    }
    if (!turnRequestInFlight) {
      busy.value = false;
      if (turnId === activeTurnId.value) activeTurnId.value = "";
      void drainDocumentReviews();
    }
  } else if (kind.startsWith("turn.")) busy.value = true;
}

function finishTrackedTurn(chatId: string, turnId: string) {
  if (runningChats.value[chatId] !== turnId) return;
  const next = { ...runningChats.value };
  delete next[chatId];
  runningChats.value = next;
}

async function refreshStatus() {
  const requestApi = api.value;
  const requestEpoch = sessionEpoch;
  const result = await requestApi.status();
  if (!disposed && requestApi === api.value && requestEpoch === sessionEpoch) status.value = result;
}

async function refreshCurrentHistory() {
  const requestSession = sessionId.value;
  const requestProject = projectId.value;
  const requestEpoch = sessionEpoch;
  const history = await api.value.chatMessages(requestSession, requestProject);
  if (disposed || requestEpoch !== sessionEpoch || requestSession !== sessionId.value || requestProject !== projectId.value) return;
  chatTitle.value = history.chat.title;
  historyAvailable.value = history.history_available;
  const currentById = new Map(messages.value.map((message) => [message.id, message]));
  const restored = new Map(messages.value.filter((message) => !message.pending).map((message) => [message.id, message]));
  for (const item of history.messages) {
    const current = currentById.get(item.id);
    restored.set(item.id, current?.pending ? current : {
      id: item.id, turnId: item.turn_id, role: item.role, content: item.content, sequence: item.sequence,
    });
  }
  for (const item of messages.value) if (item.pending) restored.set(item.id, item);
  messages.value = [...restored.values()].sort((a, b) => (a.sequence ?? Number.MAX_SAFE_INTEGER) - (b.sequence ?? Number.MAX_SAFE_INTEGER));
  if (!historyCursor.value) historyCursor.value = history.next_cursor ?? "";
  sessionStorage.setItem("llama-agent-web.chat-title", chatTitle.value);
}

async function loadOlderMessages() {
  const cursor = historyCursor.value;
  if (!cursor || navigationLoading.value) return;
  const requestSession = sessionId.value;
  const requestProject = projectId.value;
  const requestEpoch = sessionEpoch;
  navigationLoading.value = true;
  try {
    const page = await api.value.chatMessages(requestSession, requestProject, cursor);
    if (disposed || requestEpoch !== sessionEpoch || requestSession !== sessionId.value || requestProject !== projectId.value) return;
    const known = new Map(messages.value.map((message) => [message.id, message]));
    for (const item of page.messages) if (!known.has(item.id)) known.set(item.id, {
      id: item.id, turnId: item.turn_id, role: item.role, content: item.content, sequence: item.sequence,
    });
    messages.value = [...known.values()].sort((a, b) => (a.sequence ?? Number.MAX_SAFE_INTEGER) - (b.sequence ?? Number.MAX_SAFE_INTEGER));
    historyCursor.value = page.next_cursor ?? "";
  } catch (cause) {
    const message = cause instanceof Error ? cause.message : String(cause);
    if (/stale/i.test(message)) {
      historyCursor.value = "";
      await refreshCurrentHistory();
    } else throw cause;
  } finally { if (requestEpoch === sessionEpoch) navigationLoading.value = false; }
}

async function callReadOnlyTool(toolName: string, args: Record<string, unknown>) {
  if (!tools.value.some((tool) => tool.name === toolName && tool.read_only && !tool.policy_gated)) {
    throw new Error(`${toolName} is not exposed as an approved read-only tool.`);
  }
  return api.value.callReadOnlyTool({
    tool_name: toolName,
    arguments: args,
    namespace_id: "web",
    session_id: sessionId.value,
    project_id: projectId.value,
  });
}

async function readMcpAppResource(appToolName: string, uri: string) {
  return api.value.readMcpAppResource({ tool_name: appToolName, resource_uri: uri, session_id: sessionId.value, project_id: projectId.value });
}

async function callMcpAppTool(appToolName: string, toolName: string, args: Record<string, unknown>) {
  return api.value.callMcpAppTool({ app_tool_name: appToolName, tool_name: toolName, arguments: args, session_id: sessionId.value, project_id: projectId.value });
}

async function importDocument(sourceUri: string) {
  return callMcpAppTool("document.open", "document.import", {
    source_uri: sourceUri,
    operation_id: `web-import-${crypto.randomUUID()}`,
  });
}

async function readResourceBlob(uri: string): Promise<Blob> {
  const requestSession = sessionId.value;
  const requestProject = projectId.value;
  const requestEpoch = sessionEpoch;
  const blob = await api.value.downloadResource(uri, {
    namespace_id: "web", project_id: requestProject, session_id: requestSession,
  });
  if (requestEpoch !== sessionEpoch || requestSession !== sessionId.value || requestProject !== projectId.value) throw new Error("The document session changed.");
  return blob;
}

async function readJsonResource(uri: string): Promise<Record<string, unknown>> {
  const blob = await readResourceBlob(uri);
  if (blob.size > 16 * 1024 * 1024) throw new Error("The saved report context exceeds the browser's 16 MiB reading limit.");
  const value: unknown = JSON.parse(await blob.text());
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("The saved context is not a structured object.");
  return value as Record<string, unknown>;
}

async function connect() {
  if (disposed) return;
  abortController.value?.abort();
  const controller = new AbortController();
  abortController.value = controller;
  const current = () => !disposed && abortController.value === controller && !controller.signal.aborted;
  connected.value = false;
  connectionState.value = "connecting";
  connectionError.value = "";
  try {
    await api.value.connectEvents((event) => { if (current()) observe(event); }, controller.signal, () => {
      if (!current()) return;
      connected.value = true;
      connectionState.value = "connected";
      void refreshStatus().then(() => { if (current()) void drainDocumentReviews(); })
        .catch((cause) => { if (current()) connectionError.value = cause instanceof Error ? cause.message : String(cause); });
      void refreshNavigation(projectId.value).catch((cause) => { if (current()) connectionError.value = cause instanceof Error ? cause.message : String(cause); });
      void refreshCurrentHistory().catch(() => { /* A chat created during initial connection is loaded by ensureActiveChat. */ });
    });
  } catch (cause) {
    if (current()) {
      const message = cause instanceof Error ? cause.message : String(cause);
      connected.value = false;
      connectionError.value = message;
      connectionState.value = /401|403|unauthor/i.test(message) ? "auth-error" : "error";
    }
  } finally {
    if (current()) {
      connected.value = false;
      // connectEvents may update this ref from its connection callback, which
      // TypeScript cannot observe across the callback boundary.
      const finalState = connectionState.value as string;
      if (finalState === "connected" || finalState === "connecting") connectionState.value = "disconnected";
    }
    if (current() && connectionState.value !== "auth-error") {
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
      if (current()) void connect();
    }
  }
}

async function submit(prompt: string, additionalResourceRefs: string[] = [], propagateFailure = false,
    documentContext?: TurnRequest["document_context"], stableTurnId?: string,
    resourceRefsOverride?: string[], displayPrompt?: string, attachmentOrientation = false,
    investigationContext?: TurnRequest["investigation_context"]) {
  const turnId = stableTurnId ?? crypto.randomUUID();
  const visiblePrompt = displayPrompt ?? prompt;
  const turnResourceRefs = [...new Set(resourceRefsOverride ??
    (additionalResourceRefs.length ? additionalResourceRefs : selectedResourceRefs.value))];
  const attachmentContextRequired = !attachmentOrientation &&
    turnResourceRefs.some((uri) => uri !== documentContext?.document_uri);
  if (!attachmentOrientation && !documentContext && turnResourceRefs.length) {
    const supplied = new Set(turnResourceRefs);
    const previousLength = attachmentOrientations.value.length;
    attachmentOrientations.value = attachmentOrientations.value.filter((orientation) =>
      orientation.state !== "queued" || !supplied.has(orientation.resource_uri));
    if (attachmentOrientations.value.length !== previousLength) {
      persistAttachmentOrientations();
      if (attachmentOrientationTimer !== undefined) globalThis.clearTimeout(attachmentOrientationTimer);
      attachmentOrientationTimer = undefined;
    }
  }
  if (busy.value || turnRequestInFlight) {
    if (propagateFailure) throw new Error("An agent turn is already running. Retry the review when it finishes.");
    if (pendingUserTurns.value.length >= 32) {
      error.value = "The pending chat queue is full. Wait for a turn to finish before sending another message.";
      return false;
    }
    pendingUserTurns.value.push({ turn_id: turnId, session_id: sessionId.value, prompt,
      resource_refs: turnResourceRefs, display_prompt: visiblePrompt,
      ...(documentContext ? { document_context: { ...documentContext, block_ids: [...documentContext.block_ids] } } : {}),
      ...(investigationContext ? { investigation_context: copyInvestigationContext(investigationContext) } : {}),
      ...(attachmentContextRequired ? { attachment_context_required: true as const } : {}),
      ...(attachmentOrientation ? { attachment_orientation: true as const } : {}), state: "queued" });
    if (!persistPendingUserTurns()) {
      pendingUserTurns.value = pendingUserTurns.value.filter((turn) => turn.turn_id !== turnId);
      return false;
    }
    if (!messages.value.some((message) => message.id === `${turnId}-user`)) {
      messages.value.push({ id: `${turnId}-user`, turnId, role: "user", content: visiblePrompt });
      messages.value.push({ id: `${turnId}-assistant`, turnId, role: "assistant",
        content: "Queued until the current turn finishes.", pending: true });
    }
    return true;
  }
  const requestEpoch = sessionEpoch;
  const requestSessionId = sessionId.value;
  error.value = ""; busy.value = true; turnRequestInFlight = true;
  turnRequestController = new AbortController(); turnRequestFailure = "";
  activeTurnId.value = turnId;
  runningChats.value = { ...runningChats.value, [sessionId.value]: turnId };
  if (!messages.value.some((message) => message.id === `${turnId}-user`)) {
    messages.value.push({ id: `${turnId}-user`, turnId, role: "user", content: visiblePrompt });
    messages.value.push({ id: `${turnId}-assistant`, turnId, role: "assistant", content: "", pending: true });
  } else {
    const assistant = messages.value.find((message) => message.id === `${turnId}-assistant`);
    if (assistant) { assistant.content = ""; assistant.pending = true; }
  }
  const turn: TurnRequest = {
    prompt, session_id: sessionId.value, namespace_id: "web", project_id: projectId.value,
    turn_id: turnId, mode: "agent", n_predict: 0, include_summary: true,
    ...(turnResourceRefs.length ? { resource_refs: [...new Set(turnResourceRefs)] } : {}),
    ...(documentContext ? { document_context: documentContext } : {}),
    ...(investigationContext ? { investigation_context: copyInvestigationContext(investigationContext) } : {}),
    ...(attachmentOrientation ? { attachment_orientation: true } : {}),
    ...(attachmentContextRequired
      ? { attachment_context_required: true } : {}),
  };
  try {
    const result = await api.value.submitTurn(turn, turnRequestController.signal) as TurnResult;
    finishTrackedTurn(requestSessionId, turnId);
    if (requestEpoch !== sessionEpoch || requestSessionId !== sessionId.value) {
      if (requestSessionId === sessionId.value) {
        busy.value = false;
        activeTurnId.value = "";
        void refreshCurrentHistory().catch((cause) => {
          if (requestSessionId === sessionId.value) error.value = String(cause);
        });
      }
      return false;
    }
    // Recover command-owned events when SSE was disconnected or overflowed.
    // A repeated streamed event must not create another report card or effect.
    if (Array.isArray(result.events)) for (const value of result.events) {
      if (!value || typeof value !== "object" || Array.isArray(value)) continue;
      const event = value as AgentEvent;
      if (event.sequence !== undefined && events.value.some((known) =>
        known.sequence === event.sequence && known.request_id === event.request_id)) continue;
      observe({ ...event, session_id: event.session_id ?? requestSessionId, turn_id: event.turn_id ?? turnId });
    }
    const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
    if (assistant && typeof result.response === "string") assistant.content = result.response;
    if (assistant) assistant.pending = false;
    if (result.ok === false) {
      error.value = result.error || "Turn failed";
      if (assistant && !assistant.content) assistant.content = error.value;
      busy.value = false;
      activeTurnId.value = "";
      if (propagateFailure) throw new Error(error.value);
      return false;
    }
    busy.value = false;
    activeTurnId.value = "";
  } catch (cause) {
    if (requestEpoch !== sessionEpoch || requestSessionId !== sessionId.value) return false;
    busy.value = false; activeTurnId.value = "";
    const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
    if (assistant) assistant.pending = false;
    error.value = turnRequestFailure || (cause instanceof Error ? cause.message : String(cause));
    if (assistant && !assistant.content) assistant.content = error.value;
    if (propagateFailure) throw new Error(error.value);
    return false;
  } finally {
    if (requestEpoch === sessionEpoch && requestSessionId === sessionId.value) {
      turnRequestInFlight = false;
      turnRequestController = null; turnRequestFailure = "";
      void drainDocumentReviews();
    }
  }
  return true;
}

async function submitReview(prompt: string, documentContext: NonNullable<TurnRequest["document_context"]>, operationId: string = crypto.randomUUID()) {
  const existing = documentReviews.value.find((item) => item.operation_id === operationId && item.session_id === sessionId.value);
  if (existing) {
    if (existing.prompt !== prompt || JSON.stringify(existing.document_context) !== JSON.stringify(documentContext)) {
      throw new Error("The review operation ID already belongs to a different document review.");
    }
    return { queued: true, operation_id: operationId, state: existing.state };
  }
  documentReviews.value.push({
    operation_id: operationId, turn_id: operationId, session_id: sessionId.value,
    prompt, document_context: { ...documentContext, block_ids: [...documentContext.block_ids] }, state: "queued",
  });
  if (!persistDocumentReviews()) {
    documentReviews.value = documentReviews.value.filter((item) => item.operation_id !== operationId);
    throw new Error(error.value);
  }
  void drainDocumentReviews();
  return { queued: true, operation_id: operationId, state: "queued" as const };
}

async function drainDocumentReviews() {
  if (disposed || drainingDocumentReviews || busy.value || turnRequestInFlight) return;
  const drainEpoch = sessionEpoch;
  const item = documentReviews.value.find((review) => review.state === "queued" && review.session_id === sessionId.value);
  if (!item) {
    const queuedTurn = pendingUserTurns.value.find((turn) => turn.state === "queued" && turn.session_id === sessionId.value);
    if (!queuedTurn) { void drainAttachmentOrientations(); return; }
    drainingDocumentReviews = true;
    queuedTurn.state = "running";
    persistPendingUserTurns();
    try {
      const sent = await submit(queuedTurn.prompt, [], false, queuedTurn.document_context,
        queuedTurn.turn_id, queuedTurn.resource_refs, queuedTurn.display_prompt,
        Boolean(queuedTurn.attachment_orientation), queuedTurn.investigation_context);
      if (drainEpoch !== sessionEpoch) return;
      if (sent) {
        pendingUserTurns.value = pendingUserTurns.value.filter((turn) => turn.turn_id !== queuedTurn.turn_id);
      } else {
        queuedTurn.state = "failed";
        queuedTurn.error = error.value || "The queued turn could not be submitted.";
        const assistant = messages.value.find((message) => message.id === `${queuedTurn.turn_id}-assistant`);
        if (assistant) { assistant.pending = false; assistant.content = `Queued message failed: ${queuedTurn.error}`; }
      }
    } finally {
      if (drainEpoch === sessionEpoch) {
        persistPendingUserTurns();
        drainingDocumentReviews = false;
        if (!busy.value) void drainDocumentReviews();
      }
    }
    return;
  }
  drainingDocumentReviews = true;
  item.state = "running"; item.error = undefined; persistDocumentReviews();
  try {
    await submit(item.prompt, [item.document_context.document_uri], true, item.document_context, item.turn_id);
    if (drainEpoch !== sessionEpoch) return;
    item.state = error.value ? "failed" : "completed";
    if (item.state === "failed") item.error = error.value;
  } catch (cause) {
    if (drainEpoch !== sessionEpoch) return;
    item.state = "failed"; item.error = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (drainEpoch === sessionEpoch) {
      drainingDocumentReviews = false;
      persistDocumentReviews();
      if (!busy.value) void drainDocumentReviews();
    }
  }
}

async function drainAttachmentOrientations() {
  if (disposed || attachmentOrientationTimer !== undefined || drainingDocumentReviews || busy.value || turnRequestInFlight) return;
  const drainEpoch = sessionEpoch;
  const item = attachmentOrientations.value.find((orientation) => orientation.state === "queued" && orientation.session_id === sessionId.value);
  if (!item) return;
  drainingDocumentReviews = true;
  item.state = "running"; item.error = undefined; persistAttachmentOrientations();
  try {
    const prompt = "Orient me to the attached file. Give a short overview of its subject and structure, explain any extraction or coverage limitations, do not edit it or search externally, and ask what I would like to do next.";
    await submit(prompt, [], true, undefined, item.turn_id, [item.resource_uri], `Initial orientation: ${item.name}`, true);
    if (drainEpoch !== sessionEpoch) return;
    item.state = error.value ? "failed" : "completed";
    if (item.state === "failed") item.error = error.value;
  } catch (cause) {
    if (drainEpoch !== sessionEpoch) return;
    item.state = "failed";
    item.error = cause instanceof Error ? cause.message : String(cause);
    const assistant = messages.value.find((message) => message.turnId === item.turn_id && message.role === "assistant");
    if (assistant) assistant.content = `I couldn't create an initial orientation for ${item.name}: ${item.error}`;
  } finally {
    if (drainEpoch === sessionEpoch) {
      drainingDocumentReviews = false;
      persistAttachmentOrientations();
      if (!busy.value) void drainDocumentReviews();
    }
  }
}

function retryDocumentReview(operationId: string) {
  const item = documentReviews.value.find((review) => review.operation_id === operationId && review.session_id === sessionId.value);
  if (!item || item.state !== "failed") return;
  item.state = "queued"; item.error = undefined; persistDocumentReviews();
  void drainDocumentReviews();
}

function retryPendingUserTurn(turnId: string) {
  const item = pendingUserTurns.value.find((turn) => turn.turn_id === turnId && turn.session_id === sessionId.value);
  if (!item || item.state !== "failed") return;
  item.state = "queued"; item.error = undefined; persistPendingUserTurns();
  void drainDocumentReviews();
}

function dismissPendingUserTurn(turnId: string) {
  pendingUserTurns.value = pendingUserTurns.value.filter((turn) => turn.turn_id !== turnId);
  persistPendingUserTurns();
}

async function cancel() {
  if (!activeTurnId.value) return;
  try { await api.value.cancelTurn(activeTurnId.value, sessionId.value, projectId.value); } catch (cause) { error.value = String(cause); }
}

async function upload(file: File, options: { scope?: "session" | "project"; orient?: boolean; attach?: boolean } = {}): Promise<string | undefined> {
  const uploadEpoch = sessionEpoch;
  const uploadSessionId = sessionId.value;
  const uploadProjectId = projectId.value;
  try {
    const extensionIsMarkdown = /\.(md|markdown)$/i.test(file.name);
    const extensionIsDocx = /\.docx$/i.test(file.name);
    const isText = file.type.startsWith("text/") || /\.(csv|json|jsonl|md|markdown|txt|xml|html?)$/i.test(file.name);
    const mimeType = !file.type || file.type === "application/octet-stream"
      ? extensionIsDocx ? "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        : extensionIsMarkdown ? "text/markdown" : isText ? "text/plain" : "application/octet-stream"
      : file.type;
    const resource: UploadedResource = {
      name: file.name, mime_type: mimeType, scope: options.scope ?? "session",
      description: `Web attachment: ${file.name}`, session_id: uploadSessionId,
      namespace_id: "web", project_id: uploadProjectId,
      ...(isText
        ? { text: await file.text() }
        : { bytes_base64: base64Encode(new Uint8Array(await file.arrayBuffer())) }),
    };
    if (disposed || uploadEpoch !== sessionEpoch || uploadSessionId !== sessionId.value || uploadProjectId !== projectId.value) return;
    const result = await api.value.putResource(resource) as PutResourceResult;
    if (disposed || uploadEpoch !== sessionEpoch || uploadSessionId !== sessionId.value || uploadProjectId !== projectId.value) return;
    const uri = result.resource?.uri;
    if (!uri) throw new Error("The daemon returned no resource URI for the attachment");
    // The resource store owns the content after upload. Keep only attachment
    // metadata in reactive UI state rather than a second full file copy.
    if (options.attach !== false) {
      attachments.value.push({ name: resource.name, mime_type: resource.mime_type, scope: resource.scope,
        description: resource.description, session_id: uploadSessionId,
        namespace_id: resource.namespace_id, project_id: resource.project_id, uri });
      resourceRefs.value.push(uri);
    }
    if (options.orient !== false) {
      const operationId = `orient-${crypto.randomUUID()}`;
      attachmentOrientations.value.push({ operation_id: operationId, turn_id: operationId,
        session_id: sessionId.value, resource_uri: uri, name: file.name, state: "queued" });
      persistAttachmentOrientations();
      if (attachmentOrientationTimer === undefined) attachmentOrientationTimer = globalThis.setTimeout(() => {
        attachmentOrientationTimer = undefined;
        void drainDocumentReviews();
      }, 1500);
    }
    return uri;
  }
  catch (cause) {
    if (!disposed && uploadEpoch === sessionEpoch && uploadSessionId === sessionId.value) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
    return undefined;
  }
}

function recordingMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((mime) => MediaRecorder.isTypeSupported(mime)) || "";
}

async function startRecording() {
  if (recording.value) return;
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    error.value = "Audio recording is not supported by this browser.";
    return;
  }
  const recordingEpoch = sessionEpoch;
  const recordingSessionId = sessionId.value;
  try {
    const mimeType = recordingMimeType();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    if (disposed || recordingEpoch !== sessionEpoch || recordingSessionId !== sessionId.value) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    mediaStream = stream;
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecorder = recorder;
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onstop = () => {
      const type = recorder.mimeType || mimeType || "audio/webm";
      const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      const file = new File(chunks, `recording-${new Date().toISOString().replaceAll(/[:.]/g, "-")}.${extension}`, { type });
      stream.getTracks().forEach((track) => track.stop());
      if (mediaRecorder === recorder) { mediaStream = null; mediaRecorder = null; }
      if (file.size && !disposed && recordingEpoch === sessionEpoch && recordingSessionId === sessionId.value) void upload(file);
    };
    mediaRecorder.start();
    recording.value = true;
    error.value = "";
  } catch (cause) {
    if (disposed || recordingEpoch !== sessionEpoch || recordingSessionId !== sessionId.value) return;
    mediaStream?.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    mediaRecorder = null;
    recording.value = false;
    error.value = cause instanceof Error ? cause.message : "Microphone access was denied.";
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;
  recording.value = false;
  mediaRecorder.stop();
}

function toggleRecording() {
  if (recording.value) stopRecording();
  else void startRecording();
}

async function removeAttachment(index: number) {
  const removed = attachments.value[index];
  if (!removed) return;
  const epoch = sessionEpoch;
  const requestSession = sessionId.value;
  if (removed.uri) {
    try { await api.value.setChatResource({ sessionId: requestSession, projectId: projectId.value, uri: removed.uri, scope: removed.scope, action: "detach" }); }
    catch (cause) {
      if (epoch === sessionEpoch) error.value = cause instanceof Error ? cause.message : String(cause);
      return;
    }
  }
  if (epoch !== sessionEpoch || requestSession !== sessionId.value) return;
  const currentIndex = attachments.value.indexOf(removed);
  if (currentIndex < 0) return;
  attachments.value.splice(currentIndex, 1);
  resourceRefs.value = resourceRefs.value.filter((uri) => uri !== removed.uri);
  if (removed?.uri) selectedResourceRefs.value = selectedResourceRefs.value.filter((uri) => uri !== removed.uri);
  if (removed?.uri) {
    const previousLength = attachmentOrientations.value.length;
    attachmentOrientations.value = attachmentOrientations.value.filter((item) =>
      item.state !== "queued" || item.resource_uri !== removed.uri);
    if (attachmentOrientations.value.length !== previousLength) persistAttachmentOrientations();
    if (!attachmentOrientations.value.some((item) => item.state === "queued") && attachmentOrientationTimer !== undefined) {
      globalThis.clearTimeout(attachmentOrientationTimer);
      attachmentOrientationTimer = undefined;
    }
  }
}

async function downloadArtifact(event: AgentEvent) {
  const uri = eventResourceUri(event);
  if (!uri) {
    error.value = "The artifact event did not contain a resource URI.";
    return;
  }
  try {
    const blob = await api.value.downloadResource(uri, {
      namespace_id: String(event.namespace_id ?? "web"),
      project_id: String(event.project_id ?? projectId.value),
      session_id: String(event.session_id ?? sessionId.value),
      turn_id: String(event.turn_id ?? ""),
    });
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = uri.split("/").at(-1) || "artifact";
    anchor.click();
    URL.revokeObjectURL(objectUrl);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function refreshNavigation(forProject = projectId.value) {
  const request = ++navigationGeneration;
  navigationLoading.value = true;
  try {
    const [projectResult, chatResult] = await Promise.all([
      api.value.projects(), api.value.chats(forProject),
    ]);
    if (request !== navigationGeneration) return;
    projects.value = projectResult.projects;
    chats.value = chatResult.chats;
    chatCursor.value = chatResult.next_cursor || "";
    for (const chat of chatResult.chats) {
      if (chat.active_turn_id) runningChats.value[chat.id] = chat.active_turn_id;
      else if (runningChats.value[chat.id]) {
        const next = { ...runningChats.value };
        delete next[chat.id];
        runningChats.value = next;
      }
    }
  } finally { if (request === navigationGeneration) navigationLoading.value = false; }
}

async function loadMoreChats(forProject = projectId.value) {
  const cursor = chatCursor.value;
  if (!cursor || navigationLoading.value) return;
  const request = ++navigationGeneration;
  navigationLoading.value = true;
  try {
    const result = await api.value.chats(forProject, cursor);
    if (request !== navigationGeneration) return;
    const known = new Set(chats.value.map((chat) => chat.id));
    chats.value = [...chats.value, ...result.chats.filter((chat) => !known.has(chat.id))];
    chatCursor.value = result.next_cursor || "";
  } finally { if (request === navigationGeneration) navigationLoading.value = false; }
}

async function createProject(name: string) {
  const result = await api.value.createProject(name);
  projects.value = [result.project, ...projects.value];
  return result.project;
}

function restoreActiveQueues() {
  documentReviews.value = restoreDocumentReviews();
  attachmentOrientations.value = restoreAttachmentOrientations();
  pendingUserTurns.value = restorePendingUserTurns();
}

async function activateChat(chat: WebChat) {
  const request = ++chatOpenGeneration;
  navigationLoading.value = true;
  try {
    const history = await api.value.chatMessages(chat.id, chat.project_id);
    if (request !== chatOpenGeneration || disposed) return;
    ++sessionEpoch;
    drainingDocumentReviews = false;
    turnRequestController = null;
    turnRequestInFlight = false;
    if (attachmentOrientationTimer !== undefined) globalThis.clearTimeout(attachmentOrientationTimer);
    attachmentOrientationTimer = undefined;
    if (recording.value) stopRecording();
    sessionId.value = chat.id;
    projectId.value = chat.project_id;
    chatTitle.value = history.chat.title || chat.title;
    historyAvailable.value = history.history_available;
    sessionStorage.setItem(sessionStorageKey, sessionId.value);
    sessionStorage.setItem(navigationSchemaKey, "1");
    sessionStorage.setItem("llama-agent-web.project-id", projectId.value);
    sessionStorage.setItem("llama-agent-web.chat-title", chatTitle.value);
    messages.value = history.messages.map((item) => ({ id: item.id, turnId: item.turn_id, role: item.role, content: item.content, sequence: item.sequence }));
    historyCursor.value = history.next_cursor ?? "";
    events.value = [];
    attachments.value = [];
    resourceRefs.value = [];
    selectedResourceRefs.value = [];
    activeTurnId.value = runningChats.value[chat.id] || chat.active_turn_id || "";
    busy.value = Boolean(activeTurnId.value);
    error.value = "";
    restoreActiveQueues();
    attachments.value = (history.chat.attachments ?? []).map((item) => ({
      name: item.name, mime_type: item.mime_type ?? "application/octet-stream", scope: "session" as const,
      session_id: chat.id, namespace_id: "web", project_id: chat.project_id, uri: item.uri,
    }));
    resourceRefs.value = attachments.value.map((item) => item.uri ?? "").filter(Boolean);
  } finally { if (request === chatOpenGeneration) navigationLoading.value = false; }
}

async function newSession(forProject = projectId.value) {
  const request = ++chatOpenGeneration;
  const nextSessionId = `web-${crypto.randomUUID()}`;
  const priorEpoch = sessionEpoch;
  error.value = "";
  try {
    await api.value.createChat(nextSessionId, forProject);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
    return;
  }
  if (priorEpoch !== sessionEpoch || request !== chatOpenGeneration) return;
  sessionEpoch++;
  drainingDocumentReviews = false;
  // A session switch detaches the browser view; it does not cancel an accepted daemon turn.
  turnRequestController = null;
  turnRequestInFlight = false;
  if (attachmentOrientationTimer !== undefined) globalThis.clearTimeout(attachmentOrientationTimer);
  attachmentOrientationTimer = undefined;
  projectId.value = forProject;
  chatTitle.value = "New chat";
  sessionStorage.setItem("llama-agent-web.project-id", projectId.value);
  sessionStorage.setItem("llama-agent-web.chat-title", chatTitle.value);
  sessionId.value = nextSessionId;
  sessionStorage.setItem(sessionStorageKey, sessionId.value);
  legacyMigrationPending = false;
  sessionStorage.setItem(navigationSchemaKey, "1");
  messages.value = [];
  historyCursor.value = "";
  historyAvailable.value = true;
  events.value = [];
  attachments.value = [];
  resourceRefs.value = [];
  selectedResourceRefs.value = [];
  if (recording.value) stopRecording();
  recording.value = false;
  activeTurnId.value = "";
  busy.value = false;
  error.value = "";
  restoreActiveQueues();
  await refreshNavigation(forProject);
}

async function ensureActiveChat() {
  try {
    const history = await api.value.chatMessages(sessionId.value, projectId.value);
    chatTitle.value = history.chat.title;
    historyAvailable.value = history.history_available;
    messages.value = history.messages.map((item) => ({ id: item.id, turnId: item.turn_id, role: item.role, content: item.content, sequence: item.sequence }));
    historyCursor.value = history.next_cursor ?? "";
    attachments.value = (history.chat.attachments ?? []).map((item) => ({ name: item.name,
      mime_type: item.mime_type ?? "application/octet-stream", scope: "session" as const,
      session_id: sessionId.value, namespace_id: "web", project_id: projectId.value, uri: item.uri }));
    resourceRefs.value = attachments.value.map((item) => item.uri ?? "").filter(Boolean);
    sessionStorage.setItem("llama-agent-web.chat-title", chatTitle.value);
    legacyMigrationPending = false;
    sessionStorage.setItem(navigationSchemaKey, "1");
  } catch (cause) {
    if (!(cause instanceof AgentApiError) || cause.status !== 404) throw cause;
    if (legacyMigrationPending) {
      const migrated = await api.value.migrateLegacyChat(sessionId.value);
      projectId.value = migrated.chat.project_id;
      const history = await api.value.chatMessages(sessionId.value, projectId.value);
      chatTitle.value = history.chat.title;
      historyAvailable.value = history.history_available;
      messages.value = history.messages.map((item) => ({ id: item.id, turnId: item.turn_id, role: item.role, content: item.content, sequence: item.sequence }));
      historyCursor.value = history.next_cursor ?? "";
      attachments.value = (history.chat.attachments ?? []).map((item) => ({ name: item.name,
        mime_type: item.mime_type ?? "application/octet-stream", scope: "session" as const,
        session_id: sessionId.value, namespace_id: "web", project_id: projectId.value, uri: item.uri }));
      resourceRefs.value = attachments.value.map((item) => item.uri ?? "").filter(Boolean);
      sessionStorage.setItem("llama-agent-web.project-id", projectId.value);
      sessionStorage.setItem("llama-agent-web.chat-title", chatTitle.value);
      sessionStorage.setItem(navigationSchemaKey, "1");
      legacyMigrationPending = false;
    } else {
      const result = await api.value.createChat(sessionId.value, projectId.value, chatTitle.value);
      chatTitle.value = result.chat.title;
      sessionStorage.setItem(navigationSchemaKey, "1");
    }
  }
}

async function listResources(input: { scope: "project" | "session"; query?: string; mimeType?: string; cursor?: string; limit?: number; projectId?: string; sessionId?: string }) {
  const selectedProject = input.projectId ?? projectId.value;
  const selectedSession = input.sessionId ?? sessionId.value;
  return api.value.resources({ ...input, projectId: selectedProject,
    ...(input.scope === "session" ? { sessionId: selectedSession } : {}) });
}

async function cancelChat(chatId: string) {
  const turnId = runningChats.value[chatId];
  if (!turnId) return;
  try { await api.value.cancelTurn(turnId, chatId, chats.value.find((chat) => chat.id === chatId)?.project_id ?? ""); }
  catch (cause) { if (chatId === sessionId.value) error.value = cause instanceof Error ? cause.message : String(cause); }
}

export function useAgentSession() {
  return reactive({ apiBase, token, sessionId, projectId, chatTitle, historyAvailable, historyCursor, projects, chats, chatCursor, navigationLoading, runningChats, messages, events, status, capabilities, tools, connected, connectionState, connectionError, busy, error, attachments, resourceRefs, selectedResourceRefs, activeTurnId, recording, documentReviews, attachmentOrientations, pendingUserTurns, refreshStatus, refreshCurrentHistory, loadOlderMessages, refreshNavigation, loadMoreChats, createProject, activateChat, ensureActiveChat, listResources, cancelChat, connect, submit, submitReview, retryDocumentReview, retryPendingUserTurn, dismissPendingUserTurn, callReadOnlyTool, readMcpAppResource, callMcpAppTool, importDocument, readJsonResource, readResourceBlob, cancel, upload, removeAttachment, downloadArtifact, newSession, eventType, toggleRecording });
}

// Replacing this module during Vite development must release its long-lived
// stream; otherwise abandoned modules keep browser connection slots occupied.
import.meta.hot?.dispose(() => {
  disposed = true;
  if (attachmentOrientationTimer !== undefined) globalThis.clearTimeout(attachmentOrientationTimer);
  abortController.value?.abort();
  turnRequestController?.abort();
  if (mediaRecorder?.state !== "inactive") mediaRecorder?.stop();
  mediaStream?.getTracks().forEach((track) => track.stop());
});
