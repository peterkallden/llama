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
const recording = ref(false);
const abortController = ref<AbortController | null>(null);
let mediaRecorder: MediaRecorder | null = null;
let mediaStream: MediaStream | null = null;
let recordedChunks: Blob[] = [];
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

function eventResourceUri(event: AgentEvent): string {
  return typeof event.resource_uri === "string" && event.resource_uri
    ? event.resource_uri
    : eventType(event) === "tool.artifact_created" && typeof event.detail === "string"
      ? event.detail
      : "";
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
    let finalResult = result;
    const grammarFallback = result.ok === false &&
      resourceRefs.value.length === 0 &&
      /grammar|empty grammar stack|json schema/i.test(result.error || "");
    if (grammarFallback) {
      finalResult = await api.value.submitTurn({ ...turn, mode: "chat" }) as TurnResult;
    }
    if (assistant && typeof finalResult.response === "string") assistant.content = finalResult.response;
    if (assistant) assistant.pending = false;
    if (finalResult.ok === false) {
      error.value = grammarFallback
        ? `Agent grammar fallback failed: ${finalResult.error || "chat turn failed"}`
        : (finalResult.error || "Turn failed");
    }
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
  try {
    const mimeType = recordingMimeType();
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
    recordedChunks = [];
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size) recordedChunks.push(event.data);
    };
    mediaRecorder.onstop = () => {
      const type = mediaRecorder?.mimeType || mimeType || "audio/webm";
      const extension = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
      const file = new File(recordedChunks, `recording-${new Date().toISOString().replaceAll(/[:.]/g, "-")}.${extension}`, { type });
      mediaStream?.getTracks().forEach((track) => track.stop());
      mediaStream = null;
      mediaRecorder = null;
      recordedChunks = [];
      if (file.size) void upload(file);
    };
    mediaRecorder.start();
    recording.value = true;
    error.value = "";
  } catch (cause) {
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
  attachments.value.splice(index, 1);
  resourceRefs.value.splice(index, 1);
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
  sessionId.value = `web-${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionStorageKey, sessionId.value);
  messages.value = [];
  events.value = [];
  attachments.value = [];
  resourceRefs.value = [];
  if (recording.value) stopRecording();
  activeTurnId.value = "";
  busy.value = false;
  error.value = "";
}

export function useAgentSession() {
  return reactive({ apiBase, token, sessionId, messages, events, status, capabilities, connected, busy, error, attachments, activeTurnId, recording, refreshStatus, connect, submit, cancel, upload, removeAttachment, downloadArtifact, newSession, eventType, toggleRecording });
}
