import { computed, reactive, ref } from "vue";
import { AgentApi } from "./api/agent-api";
import type { AgentEvent, ChatMessage, DaemonStatus, PutResourceResult, TurnRequest, TurnResult, UploadedResource } from "./types/agent";

const apiBase = import.meta.env.VITE_AGENT_WEB_BASE_URL || "/api/v1";
const token = ref("");
const api = computed(() => new AgentApi(apiBase, token.value));
const events = ref<AgentEvent[]>([]);
const status = ref<DaemonStatus>({ state: "unknown" });
const capabilities = computed(() => Array.isArray(status.value.capabilities)
  ? status.value.capabilities.map(String)
  : []);
const connected = ref(false);
const busy = ref(false);
const error = ref("");
const messages = ref<ChatMessage[]>([]);
const attachments = ref<UploadedResource[]>([]);
const resourceRefs = ref<string[]>([]);
const activeTurnId = ref("");
const abortController = ref<AbortController | null>(null);
const sessionStorageKey = "llama-agent-web.session-id";
const sessionId = ref(sessionStorage.getItem(sessionStorageKey) || `web-${crypto.randomUUID()}`);
sessionStorage.setItem(sessionStorageKey, sessionId.value);

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

function observe(event: AgentEvent) {
  events.value.unshift(event);
  const kind = eventType(event);
  const turnId = String(event.turn_id ?? activeTurnId.value);
  const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
  if (kind === "response.delta" && assistant) assistant.content += String(event.text ?? event.response ?? "");
  if (kind === "turn.completed" && assistant && typeof event.response === "string") assistant.content = event.response;
  if (kind === "turn.failed") error.value = String(event.error ?? event.detail ?? "Turn failed");
  if (kind === "turn.completed" || kind === "turn.failed" || kind === "turn.cancelled") {
    busy.value = false;
    if (turnId === activeTurnId.value) activeTurnId.value = "";
  } else if (kind.startsWith("turn.")) busy.value = true;
}

async function refreshStatus() { status.value = await api.value.status(); }

async function connect() {
  abortController.value?.abort();
  const controller = new AbortController();
  abortController.value = controller;
  connected.value = false;
  try {
    connected.value = true;
    await api.value.connectEvents(observe, controller.signal);
  } catch (cause) {
    if (!controller.signal.aborted) error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    connected.value = false;
    if (!controller.signal.aborted) {
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
      if (!controller.signal.aborted) void connect();
    }
  }
}

async function submit(prompt: string) {
  error.value = ""; busy.value = true;
  const turnId = crypto.randomUUID();
  activeTurnId.value = turnId;
  messages.value.push({ id: `${turnId}-user`, turnId, role: "user", content: prompt });
  messages.value.push({ id: `${turnId}-assistant`, turnId, role: "assistant", content: "", pending: true });
  const turn: TurnRequest = {
    prompt, session_id: sessionId.value, namespace_id: "web", project_id: "web-example",
    turn_id: turnId, mode: "agent", n_predict: 384, include_summary: true,
    ...(resourceRefs.value.length ? { resource_refs: [...resourceRefs.value] } : {}),
  };
  try {
    const result = await api.value.submitTurn(turn) as TurnResult;
    const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
    if (assistant && typeof result.response === "string") assistant.content = result.response;
    if (assistant) assistant.pending = false;
    if (result.ok === false) error.value = result.error || "Turn failed";
    busy.value = false;
    activeTurnId.value = "";
  } catch (cause) {
    busy.value = false; activeTurnId.value = "";
    const assistant = messages.value.find((message) => message.turnId === turnId && message.role === "assistant");
    if (assistant) assistant.pending = false;
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function cancel() {
  if (!activeTurnId.value) return;
  try { await api.value.cancelTurn(activeTurnId.value); } catch (cause) { error.value = String(cause); }
}

async function upload(file: File) {
  const isText = file.type.startsWith("text/") || /\.(csv|json|jsonl|md|txt|xml|html?)$/i.test(file.name);
  const resource: UploadedResource = {
    name: file.name, mime_type: file.type || (isText ? "text/plain" : "application/octet-stream"), scope: "session",
    description: `Web attachment: ${file.name}`, session_id: sessionId.value,
    namespace_id: "web", project_id: "web-example",
    ...(isText
      ? { text: await file.text() }
      : { bytes_base64: base64Encode(new Uint8Array(await file.arrayBuffer())) }),
  };
  try {
    const result = await api.value.putResource(resource) as PutResourceResult;
    const uri = result.resource?.uri;
    if (!uri) throw new Error("The daemon returned no resource URI for the attachment");
    attachments.value.push({ ...resource, uri });
    resourceRefs.value.push(uri);
  }
  catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
}

function removeAttachment(index: number) {
  attachments.value.splice(index, 1);
  resourceRefs.value.splice(index, 1);
}

function newSession() {
  sessionId.value = `web-${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionStorageKey, sessionId.value);
  messages.value = [];
  events.value = [];
  attachments.value = [];
  resourceRefs.value = [];
  activeTurnId.value = "";
  busy.value = false;
  error.value = "";
}

export function useAgentSession() {
  return reactive({ apiBase, token, sessionId, messages, events, status, capabilities, connected, busy, error, attachments, activeTurnId, refreshStatus, connect, submit, cancel, upload, removeAttachment, newSession, eventType });
}
