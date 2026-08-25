import { computed, reactive, ref } from "vue";
import { AgentApi } from "./api/agent-api";
import type { AgentEvent, DaemonStatus, PutResourceResult, TurnRequest, UploadedResource } from "./types/agent";

const apiBase = import.meta.env.VITE_AGENT_WEB_BASE_URL || "/api/v1";
const token = ref("");
const api = computed(() => new AgentApi(apiBase, token.value));
const events = ref<AgentEvent[]>([]);
const status = ref<DaemonStatus>({ state: "unknown" });
const connected = ref(false);
const busy = ref(false);
const error = ref("");
const responseText = ref("");
const attachments = ref<UploadedResource[]>([]);
const resourceRefs = ref<string[]>([]);
const abortController = ref<AbortController | null>(null);
const sessionId = `web-${crypto.randomUUID()}`;

function eventType(event: AgentEvent): string {
  return String(event.event_type || event.type || "event");
}

function observe(event: AgentEvent) {
  events.value.unshift(event);
  const kind = eventType(event);
  if (kind === "response.delta") responseText.value += String(event.text ?? event.response ?? "");
  if (kind === "turn.completed" && typeof event.response === "string") responseText.value = event.response;
  if (kind === "turn.failed") error.value = String(event.error ?? event.detail ?? "Turn failed");
  if (kind.startsWith("turn.")) busy.value = !["turn.completed", "turn.failed", "turn.cancelled"].includes(kind);
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
  } finally { connected.value = false; }
}

async function submit(prompt: string) {
  error.value = ""; responseText.value = ""; busy.value = true;
  const turn: TurnRequest = {
    prompt, session_id: sessionId, namespace_id: "web", project_id: "web-example",
    turn_id: crypto.randomUUID(), mode: "agent", n_predict: 384, include_summary: true,
    ...(resourceRefs.value.length ? { resource_refs: [...resourceRefs.value] } : {}),
  };
  try { await api.value.submitTurn(turn); } catch (cause) {
    busy.value = false; error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function cancel(turnId: string) {
  try { await api.value.cancelTurn(turnId); } catch (cause) { error.value = String(cause); }
}

async function upload(file: File) {
  const text = await file.text();
  const resource: UploadedResource = {
    name: file.name, mime_type: file.type || "text/plain", text, scope: "session",
    description: `Web attachment: ${file.name}`, session_id: sessionId,
    namespace_id: "web", project_id: "web-example",
  };
  try {
    const result = await api.value.putResource(resource) as PutResourceResult;
    const uri = result.resource?.uri;
    if (!uri) throw new Error("Daemonen returnerade ingen resource_uri för bilagan");
    attachments.value.push(resource);
    resourceRefs.value.push(uri);
  }
  catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
}

function removeAttachment(index: number) {
  attachments.value.splice(index, 1);
  resourceRefs.value.splice(index, 1);
}

export function useAgentSession() {
  return reactive({ apiBase, token, events, status, connected, busy, error, responseText, attachments, refreshStatus, connect, submit, cancel, upload, removeAttachment, eventType });
}
