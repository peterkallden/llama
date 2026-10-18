import { computed, reactive, ref } from "vue";
import { AgentApi } from "./api/agent-api";
import type { AgentEvent, ChatMessage, DaemonStatus, DaemonToolStatus, PutResourceResult, TurnRequest, TurnResult, UploadedResource } from "./types/agent";

const apiBase = import.meta.env.VITE_AGENT_WEB_BASE_URL || "/api/v1";
const token = ref("");
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
const sessionId = ref(sessionStorage.getItem(sessionStorageKey) || `web-${crypto.randomUUID()}`);
sessionStorage.setItem(sessionStorageKey, sessionId.value);
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

function reviewQueueKey(id = sessionId.value) { return `llama-agent-web.document-reviews:${id}`; }
function restoreDocumentReviews(): PendingDocumentReview[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(reviewQueueKey()) ?? "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((item): item is PendingDocumentReview => item && typeof item === "object" &&
      typeof item.operation_id === "string" && typeof item.turn_id === "string" && item.session_id === sessionId.value &&
      typeof item.prompt === "string" && item.document_context && typeof item.document_context.document_uri === "string" &&
      Number.isSafeInteger(item.document_context.revision) && Array.isArray(item.document_context.block_ids))
      .map((item) => ({ ...item, state: item.state === "completed" || item.state === "failed" ? item.state : "queued" }));
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
      .map((item) => ({ ...item, state: item.state === "completed" || item.state === "failed" ? item.state : "queued" }));
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
  attachment_context_required?: true;
  attachment_orientation?: true;
  state: "queued" | "running" | "failed";
  error?: string;
};
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
        Number.isSafeInteger(item.document_context.revision) && Array.isArray(item.document_context.block_ids))))
      .map((item) => ({ ...item, state: item.state === "failed" ? "failed" as const : "queued" as const }));
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
  if (event.session_id && event.session_id !== sessionId.value) return;
  const kind = eventType(event);
  const turnId = String(event.turn_id ?? activeTurnId.value);
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

async function refreshStatus() {
  const requestApi = api.value;
  const requestEpoch = sessionEpoch;
  const result = await requestApi.status();
  if (!disposed && requestApi === api.value && requestEpoch === sessionEpoch) status.value = result;
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
    project_id: "web-example",
  });
}

async function readMcpAppResource(appToolName: string, uri: string) {
  return api.value.readMcpAppResource({ tool_name: appToolName, resource_uri: uri, session_id: sessionId.value });
}

async function callMcpAppTool(appToolName: string, toolName: string, args: Record<string, unknown>) {
  return api.value.callMcpAppTool({ app_tool_name: appToolName, tool_name: toolName, arguments: args, session_id: sessionId.value });
}

async function importDocument(sourceUri: string) {
  return callMcpAppTool("document.open", "document.import", {
    source_uri: sourceUri,
    operation_id: `web-import-${crypto.randomUUID()}`,
  });
}

async function readResourceBlob(uri: string): Promise<Blob> {
  const requestSession = sessionId.value;
  const requestEpoch = sessionEpoch;
  const blob = await api.value.downloadResource(uri, {
    namespace_id: "web", project_id: "web-example", session_id: requestSession,
  });
  if (requestEpoch !== sessionEpoch || requestSession !== sessionId.value) throw new Error("The document session changed.");
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
    resourceRefsOverride?: string[], displayPrompt?: string, attachmentOrientation = false) {
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
  if (!messages.value.some((message) => message.id === `${turnId}-user`)) {
    messages.value.push({ id: `${turnId}-user`, turnId, role: "user", content: visiblePrompt });
    messages.value.push({ id: `${turnId}-assistant`, turnId, role: "assistant", content: "", pending: true });
  } else {
    const assistant = messages.value.find((message) => message.id === `${turnId}-assistant`);
    if (assistant) { assistant.content = ""; assistant.pending = true; }
  }
  const turn: TurnRequest = {
    prompt, session_id: sessionId.value, namespace_id: "web", project_id: "web-example",
    turn_id: turnId, mode: "agent", n_predict: 0, include_summary: true,
    ...(turnResourceRefs.length ? { resource_refs: [...new Set(turnResourceRefs)] } : {}),
    ...(documentContext ? { document_context: documentContext } : {}),
    ...(attachmentOrientation ? { attachment_orientation: true } : {}),
    ...(attachmentContextRequired
      ? { attachment_context_required: true } : {}),
  };
  try {
    const result = await api.value.submitTurn(turn, turnRequestController.signal) as TurnResult;
    if (requestEpoch !== sessionEpoch || requestSessionId !== sessionId.value) return false;
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
        Boolean(queuedTurn.attachment_orientation));
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
  try { await api.value.cancelTurn(activeTurnId.value); } catch (cause) { error.value = String(cause); }
}

async function upload(file: File) {
  const uploadEpoch = sessionEpoch;
  const uploadSessionId = sessionId.value;
  try {
    const isText = file.type.startsWith("text/") || /\.(csv|json|jsonl|md|txt|xml|html?)$/i.test(file.name);
    const resource: UploadedResource = {
      name: file.name, mime_type: file.type || (isText ? "text/plain" : "application/octet-stream"), scope: "session",
      description: `Web attachment: ${file.name}`, session_id: uploadSessionId,
      namespace_id: "web", project_id: "web-example",
      ...(isText
        ? { text: await file.text() }
        : { bytes_base64: base64Encode(new Uint8Array(await file.arrayBuffer())) }),
    };
    if (disposed || uploadEpoch !== sessionEpoch || uploadSessionId !== sessionId.value) return;
    const result = await api.value.putResource(resource) as PutResourceResult;
    if (disposed || uploadEpoch !== sessionEpoch || uploadSessionId !== sessionId.value) return;
    const uri = result.resource?.uri;
    if (!uri) throw new Error("The daemon returned no resource URI for the attachment");
    // The resource store owns the content after upload. Keep only attachment
    // metadata in reactive UI state rather than a second full file copy.
    attachments.value.push({ name: resource.name, mime_type: resource.mime_type, scope: resource.scope,
      description: resource.description, session_id: uploadSessionId,
      namespace_id: resource.namespace_id, project_id: resource.project_id, uri });
    resourceRefs.value.push(uri);
    const operationId = `orient-${crypto.randomUUID()}`;
    attachmentOrientations.value.push({ operation_id: operationId, turn_id: operationId,
      session_id: sessionId.value, resource_uri: uri, name: file.name, state: "queued" });
    persistAttachmentOrientations();
    if (attachmentOrientationTimer === undefined) attachmentOrientationTimer = globalThis.setTimeout(() => {
      attachmentOrientationTimer = undefined;
      void drainDocumentReviews();
    }, 1500);
  }
  catch (cause) {
    if (!disposed && uploadEpoch === sessionEpoch && uploadSessionId === sessionId.value) {
      error.value = cause instanceof Error ? cause.message : String(cause);
    }
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

function removeAttachment(index: number) {
  const removed = attachments.value[index];
  attachments.value.splice(index, 1);
  resourceRefs.value.splice(index, 1);
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
      namespace_id: String(event.namespace_id ?? ""),
      project_id: String(event.project_id ?? ""),
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

function newSession() {
  sessionEpoch++;
  drainingDocumentReviews = false;
  turnRequestController?.abort();
  turnRequestController = null;
  turnRequestInFlight = false;
  if (attachmentOrientationTimer !== undefined) globalThis.clearTimeout(attachmentOrientationTimer);
  attachmentOrientationTimer = undefined;
  sessionId.value = `web-${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionStorageKey, sessionId.value);
  messages.value = [];
  events.value = [];
  attachments.value = [];
  resourceRefs.value = [];
  selectedResourceRefs.value = [];
  if (recording.value) stopRecording();
  activeTurnId.value = "";
  busy.value = false;
  error.value = "";
  documentReviews.value = restoreDocumentReviews();
  attachmentOrientations.value = restoreAttachmentOrientations();
  pendingUserTurns.value = restorePendingUserTurns();
}

export function useAgentSession() {
  return reactive({ apiBase, token, sessionId, messages, events, status, capabilities, tools, connected, connectionState, connectionError, busy, error, attachments, resourceRefs, selectedResourceRefs, activeTurnId, recording, documentReviews, attachmentOrientations, pendingUserTurns, refreshStatus, connect, submit, submitReview, retryDocumentReview, retryPendingUserTurn, dismissPendingUserTurn, callReadOnlyTool, readMcpAppResource, callMcpAppTool, importDocument, readJsonResource, readResourceBlob, cancel, upload, removeAttachment, downloadArtifact, newSession, eventType, toggleRecording });
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
