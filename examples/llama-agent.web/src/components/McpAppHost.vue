<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { authorizedInvestigationReportReference, canonicalToolStructuredContent, cloneJsonMessage, invocationNotifications, isCurrentAppRequest, normalizeMcpAppCsp, mcpAppHeight, sandboxProxyDocument } from "./mcp-app-host";
import { DocumentEditorAccess } from "./document-editor-access";

const props = defineProps<{
  tool: DaemonToolStatus;
  tools: DaemonToolStatus[];
  sessionId: string;
  projectId?: string;
  readResource: (toolName: string, uri: string) => Promise<unknown>;
  listResources?: (input: { scope: "project" | "session"; query?: string; cursor?: string; limit?: number; projectId?: string; sessionId?: string }) => Promise<{ resources: Array<Record<string, unknown>>; next_cursor: string }>;
  uploadResource?: (file: File, options: { scope: "session"; orient: false; attach: false }) => Promise<string | undefined>;
  callTool: (appTool: string, toolName: string, args: Record<string, unknown>) => Promise<unknown>;
  submitReview?: (prompt: string, documentContext: { document_uri: string; revision: number; block_ids: string[]; base_document_uri?: string }, operationId?: string) => Promise<unknown>;
  invocation?: { input?: Record<string, unknown>; result?: unknown };
}>();
const emit = defineEmits<{
  documentStatus: [status: Record<string, unknown>];
  documentClosed: [];
  openReport: [uri: string];
  startInvestigation: [prompt: string];
}>();
const frame = ref<HTMLIFrameElement | null>(null);
const proxyUrl = ref("");
const appHeight = ref(620);
const error = ref("");
const busy = ref(false);
const appMessage = ref("");
const uri = computed(() => String(props.tool._meta?.ui?.resourceUri ?? ""));
const appVisible = computed(() => (props.tool._meta?.ui?.visibility ?? []).includes("app"));
let initialized = false;
let initializeAccepted = false;
let hostRequestId = 1_000_000_000;
const hostRequests = new Map<number, { method: string; resolve: (value: unknown) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }>();
let lastInvocationSignature = "";
let documentAccess = new DocumentEditorAccess(props.sessionId, props.tool.provider_id ?? "");

function unwrap(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object") return {};
  const outer = payload as Record<string, unknown>;
  const nested = outer.tool && typeof outer.tool === "object" ? outer.tool as Record<string, unknown> : outer;
  const body = nested.structured_content && typeof nested.structured_content === "object"
    ? nested.structured_content as Record<string, unknown> : nested;
  return body;
}

function toolEnvelope(payload: unknown): Record<string, unknown> {
  if (!payload || typeof payload !== "object") return {};
  const outer = payload as Record<string, unknown>;
  return outer.tool && typeof outer.tool === "object" ? outer.tool as Record<string, unknown> : outer;
}

let loadGeneration = 0;
async function load() {
  const generation = ++loadGeneration;
  for (const request of hostRequests.values()) {
    clearTimeout(request.timer);
    request.reject(new Error("The editor app was reloaded; its local draft is retained."));
  }
  hostRequests.clear();
  initialized = false; initializeAccepted = false; proxyUrl.value = ""; error.value = "";
  busy.value = false; appMessage.value = "";
  lastInvocationSignature = "";
  documentAccess = new DocumentEditorAccess(props.sessionId, props.tool.provider_id ?? "");
  pendingHtml = ""; pendingCsp = {}; appHeight.value = 620;
  if (!uri.value || !appVisible.value) { error.value = "This tool does not advertise an app-visible UI resource."; return; }
  busy.value = true;
  try {
    const result = unwrap(await props.readResource(props.tool.name, uri.value));
    if (generation !== loadGeneration) return;
    if (!String(result.mime_type ?? "").toLowerCase().startsWith("text/html")) throw new Error("The provider resource is not an HTML MCP App.");
    let html = typeof result.text === "string" ? result.text : "";
    if (!html && typeof result.blob_base64 === "string") {
      const bytes = Uint8Array.from(atob(result.blob_base64), (character) => character.charCodeAt(0));
      html = new TextDecoder().decode(bytes);
    }
    if (!html) throw new Error("The provider returned no HTML content.");
    if (new TextEncoder().encode(html).byteLength > 2 * 1024 * 1024) throw new Error("The MCP App resource exceeds the 2 MiB web-host limit.");
    const proxy = sandboxProxyDocument();
    proxyUrl.value = `data:text/html;charset=utf-8,${encodeURIComponent(proxy)}`;
    pendingHtml = html;
    pendingCsp = normalizeMcpAppCsp(result._meta);
  } catch (cause) {
    if (generation === loadGeneration) error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (generation === loadGeneration) busy.value = false;
  }
}

let pendingHtml = "";
let pendingCsp: Record<string, string[]> = {};
function send(message: Record<string, unknown>) {
  frame.value?.contentWindow?.postMessage(cloneJsonMessage(message), "*");
}

function publishInvocation() {
  if (!initialized || !props.invocation) return;
  const signature = JSON.stringify(props.invocation);
  if (!signature || signature === lastInvocationSignature) return;
  lastInvocationSignature = signature;
  for (const notification of invocationNotifications(props.invocation)) {
    if (notification.method === "ui/notifications/tool-result") {
      const params = notification.params as Record<string, unknown>;
      const structured = canonicalToolStructuredContent(params.structuredContent);
      if (structured) params.structuredContent = structured;
    }
    if (notification.method === "ui/notifications/tool-result" && uri.value === "ui://llama-agent/apps/document-editor") {
      const params = notification.params as Record<string, unknown>;
      if (!params.isError && params.structuredContent && typeof params.structuredContent === "object") {
        documentAccess.observe("document.open", props.invocation.input ?? {}, params.structuredContent as Record<string, unknown>);
      }
    }
    send(notification);
  }
}

async function onMessage(event: MessageEvent) {
  const targetWindow = frame.value?.contentWindow;
  if (!targetWindow || event.source !== targetWindow || event.origin !== "null") return;
  const requestGeneration = loadGeneration;
  const message = event.data as Record<string, unknown>;
  if (!message || typeof message !== "object" || message.jsonrpc !== "2.0") return;
  if (typeof message.id === "number" && hostRequests.has(message.id)) {
    const request = hostRequests.get(message.id)!; hostRequests.delete(message.id); clearTimeout(request.timer);
    if (message.error && typeof message.error === "object") request.reject(new Error(String((message.error as Record<string, unknown>).message ?? "Editor request failed.")));
    else request.resolve(message.result);
    return;
  }
  const method = typeof message.method === "string" ? message.method : "";
  const id = message.id;
  const reply = (result: unknown, errorText?: string) => {
    if (id === undefined || !isCurrentAppRequest(
      { generation: requestGeneration, target: targetWindow }, loadGeneration, frame.value?.contentWindow ?? null)) return;
    targetWindow.postMessage(cloneJsonMessage({
      jsonrpc: "2.0", id,
      ...(errorText ? { error: { code: -32000, message: errorText } } : { result }),
    }), "*");
  };
  if (method === "ui/notifications/sandbox-proxy-ready") {
    send({ jsonrpc: "2.0", method: "ui/notifications/sandbox-resource-ready", params: { html: pendingHtml, csp: pendingCsp } });
    return;
  }
  if (method === "ui/notifications/size-changed") {
    const height = mcpAppHeight(message.params);
    if (height !== null) appHeight.value = height;
    return;
  }
  if (method === "ui/initialize") {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    if (id === undefined || !params.appInfo || !params.appCapabilities || params.protocolVersion !== "2026-01-26") {
      reply(undefined, "Invalid MCP App initialization request.");
      return;
    }
    initializeAccepted = true;
    reply({
      protocolVersion: "2026-01-26",
      hostInfo: { name: "llama-agent-web", version: "0.1" },
      hostCapabilities: { serverTools: { listChanged: false }, serverResources: { listChanged: false }, sandbox: {} },
      hostContext: { toolInfo: { tool: { name: props.tool.name, description: props.tool.description ?? "", inputSchema: props.tool.input_schema ?? { type: "object" } } } },
    });
    return;
  }
  if (method === "ui/notifications/initialized") {
    if (!initializeAccepted) return;
    initialized = true;
    publishInvocation();
    return;
  }
  if (!initialized && method !== "ui/notifications/sandbox-proxy-ready") {
    reply(undefined, "The MCP App has not completed initialization.");
    return;
  }
  if (method === "ping") { reply({}); return; }
  if (method === "notifications/message") return;
  if (method === "ui/llama-agent/editor-status") {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    emit("documentStatus", params);
    return;
  }
  if (method === "ui/message") {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    const content = params.content && typeof params.content === "object" ? params.content as Record<string, unknown> : {};
    appMessage.value = typeof content.text === "string" ? content.text : "App sent a message.";
    const action = String(params.action ?? "");
    if (action === "start-investigation") {
      if (uri.value !== "ui://llama-agent/apps/investigation-workspace" ||
          typeof content.text !== "string" || content.text.length > 2000) {
        reply(undefined, "The investigation workspace supplied an invalid analysis request.");
        return;
      }
      emit("startInvestigation", content.text);
      reply({ queued: true });
      return;
    }
    if (action === "open-investigation-report") {
      const uriValue = authorizedInvestigationReportReference(uri.value, content.text);
      if (!uriValue) {
        reply(undefined, "The investigation workspace supplied an invalid frozen report reference.");
        return;
      }
      emit("openReport", uriValue);
      reply({ opened: true });
      return;
    }
    if (action === "close-document") {
      if (uri.value !== "ui://llama-agent/apps/document-editor") { reply(undefined, "Only the document editor may close its document view."); return; }
      reply({ closed: true });
      // Navigation awaits its own editor RPC before unmounting. Only an
      // editor-initiated close emits here; otherwise unmount rejects that RPC.
      if (![...hostRequests.values()].some((request) => request.method === "ui/llama-agent/close-document")) {
        window.setTimeout(() => {
          if (isCurrentAppRequest({ generation: requestGeneration, target: targetWindow }, loadGeneration, frame.value?.contentWindow ?? null)) emit("documentClosed");
        }, 0);
      }
      return;
    }
    if (typeof content.text === "string" && uri.value === "ui://llama-agent/apps/document-editor") {
      const rawBinding = params.document_context && typeof params.document_context === "object"
        ? params.document_context as Record<string, unknown> : {};
      try {
        if (!props.submitReview) throw new Error("Agent review is unavailable in this host.");
        if (typeof rawBinding.document_uri !== "string" || typeof rawBinding.revision !== "number" || !Number.isSafeInteger(rawBinding.revision) ||
            rawBinding.revision <= 0 || !Array.isArray(rawBinding.block_ids) || rawBinding.block_ids.length > 128 ||
            !rawBinding.block_ids.every((item) => typeof item === "string") ||
            (rawBinding.base_document_uri !== undefined &&
              (typeof rawBinding.base_document_uri !== "string" || !rawBinding.base_document_uri || rawBinding.base_document_uri.length > 512))) {
          throw new Error("The editor did not provide a valid typed document binding.");
        }
        documentAccess.authorize("document.propose_patch", {
          document_uri: rawBinding.document_uri, expected_revision: rawBinding.revision,
        });
        const queued = await props.submitReview(content.text, {
          document_uri: rawBinding.document_uri,
          revision: rawBinding.revision as number,
          block_ids: rawBinding.block_ids as string[],
          ...(typeof rawBinding.base_document_uri === "string" ? { base_document_uri: rawBinding.base_document_uri } : {}),
        }, typeof params.operation_id === "string" ? params.operation_id : undefined);
        reply({ queued: true, ...(queued && typeof queued === "object" ? queued as Record<string, unknown> : {}) });
        if (action === "close-review") window.setTimeout(() => {
          if (isCurrentAppRequest({ generation: requestGeneration, target: targetWindow }, loadGeneration, frame.value?.contentWindow ?? null)) emit("documentClosed");
        }, 0);
      } catch (cause) { reply(undefined, cause instanceof Error ? cause.message : String(cause)); }
    } else {
      reply({});
    }
    return;
  }
  if (method === "ui/open-link") { reply({ opened: false, message: "Links are not opened automatically by this host." }); return; }
  try {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    if (method === "ui/llama-agent/document-download") {
      if (uri.value !== "ui://llama-agent/apps/document-editor") throw new Error("Only the document editor may request a document download.");
      const file = documentAccess.authorizeDownload(params);
      const objectUrl = URL.createObjectURL(new Blob([file.markdown], { type: "text/markdown;charset=utf-8" }));
      const anchor = document.createElement("a"); anchor.href = objectUrl; anchor.download = file.filename;
      document.body.append(anchor); anchor.click(); anchor.remove(); window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
      reply({ downloaded: true, filename: file.filename });
    } else if (method === "ui/llama-agent/document-draft") {
      if (uri.value !== "ui://llama-agent/apps/document-editor") throw new Error("Draft storage is reserved for the first-party editor.");
      let storage: Storage | undefined;
      try { storage = window.localStorage; } catch { /* host can still retain drafts in memory */ }
      reply(documentAccess.draft(params, storage));
    } else if (method === "ui/llama-agent/list-investigation-resources") {
      if (uri.value !== "ui://llama-agent/apps/investigation-workspace") throw new Error("Resource selection is available only to the investigation workspace.");
      if (params.scope !== undefined && params.scope !== "project" && params.scope !== "session") throw new Error("Invalid resource scope.");
      if (params.scope === "project" && !props.projectId) throw new Error("A project scope is not active for this chat.");
      const scopeValue = params.scope === "project" ? "project" : "session";
      const query = typeof params.query === "string" ? params.query.slice(0, 256) : "";
      const cursor = typeof params.cursor === "string" ? params.cursor.slice(0, 128) : "";
      const limit = typeof params.limit === "number" && Number.isSafeInteger(params.limit) ? Math.max(1, Math.min(50, params.limit)) : 30;
      if (!props.listResources) throw new Error("The host resource browser is unavailable.");
      const result = await props.listResources({ scope: scopeValue, query, cursor, limit,
        projectId: props.projectId, sessionId: props.sessionId });
      reply({ resources: result.resources, next_cursor: result.next_cursor, project_available: Boolean(props.projectId) });
    } else if (method === "ui/llama-agent/upload-investigation-resource") {
      if (uri.value !== "ui://llama-agent/apps/investigation-workspace") throw new Error("Uploads are available only to the investigation workspace.");
      const file = params.file;
      if (!(file instanceof File) || file.size === 0 || file.size > 4 * 1024 * 1024) throw new Error("Choose a non-empty file no larger than 4 MiB.");
      if (!props.uploadResource) throw new Error("The host upload service is unavailable.");
      const resourceUri = await props.uploadResource(file, { scope: "session", orient: false, attach: false });
      if (!resourceUri) throw new Error("The upload did not produce a resource reference.");
      reply({ resource_uri: resourceUri });
    } else if (method === "tools/list") {
      const listed = props.tools.filter((tool) => (tool.provider_id ?? "") === (props.tool.provider_id ?? "") &&
        (tool.app_read_only || tool.app_write_authorized || (!tool.provider_id && tool.read_only)) &&
        (tool._meta?.ui?.visibility ?? []).includes("app") && tool._meta?.ui?.resourceUri === uri.value);
      reply({ tools: listed.map((tool) => ({
        name: tool.server_tool_name || tool.name,
        description: tool.description ?? "",
        inputSchema: tool.input_schema ?? { type: "object" },
        ...(tool.output_schema ? { outputSchema: tool.output_schema } : {}),
        annotations: { readOnlyHint: !tool.app_write_authorized },
      })) });
    } else if (method === "tools/call") {
      const name = String(params.name ?? "");
      const args = params.arguments && typeof params.arguments === "object" && !Array.isArray(params.arguments)
        ? params.arguments as Record<string, unknown> : {};
      const exposed = props.tools.find((tool) => (tool.provider_id ?? "") === (props.tool.provider_id ?? "") &&
        (tool.server_tool_name || tool.name) === name && (tool.app_read_only || tool.app_write_authorized || (!tool.provider_id && tool.read_only)) &&
        (tool._meta?.ui?.visibility ?? []).includes("app") && tool._meta?.ui?.resourceUri === uri.value);
      if (!exposed) {
        reply({ isError: true, content: [{ type: "text", text: "The requested tool is not host-authorized for this app." }] });
      } else {
        try {
          if (uri.value === "ui://llama-agent/apps/document-editor") documentAccess.authorize(name, args);
          const result = toolEnvelope(await props.callTool(props.tool.name, name, args));
          const structured = canonicalToolStructuredContent(result.structured_content ?? result.structuredContent);
          const nestedError = result.structured_content && typeof result.structured_content === "object" && !Array.isArray(result.structured_content)
            ? (result.structured_content as Record<string, unknown>).error : undefined;
          const errorInfo = result.error && typeof result.error === "object" ? result.error as Record<string, unknown>
            : nestedError && typeof nestedError === "object" ? nestedError as Record<string, unknown> : {};
          const failed = result.ok === false || Boolean(result.error) || !structured;
          if (!failed && uri.value === "ui://llama-agent/apps/document-editor" && structured &&
              isCurrentAppRequest({ generation: requestGeneration, target: targetWindow }, loadGeneration, frame.value?.contentWindow ?? null)) {
            documentAccess.observe(name, args, structured);
          }
          reply({
            isError: failed,
            ...(structured ? { structuredContent: structured } : {}),
            content: [{ type: "text", text: String(result.content_summary ?? errorInfo.safe_summary ?? "Tool call completed.") }],
          });
        } catch (cause) {
          const payload = cause && typeof cause === "object" && "payload" in cause ? (cause as { payload: unknown }).payload : undefined;
          const outer = payload && typeof payload === "object" ? payload as Record<string, unknown> : {};
          const nested = outer.tool && typeof outer.tool === "object" ? outer.tool as Record<string, unknown> : outer;
          const err = nested.error && typeof nested.error === "object" ? nested.error as Record<string, unknown> : {};
          const safe = String(err.safe_summary ?? (cause instanceof Error ? cause.message : "Tool execution failed."));
          reply({ isError: true, content: [{ type: "text", text: safe }] });
        }
      }
    } else if (method === "resources/read") {
      const requested = String(params.uri ?? "");
      if (requested !== uri.value) throw new Error("The app may read only its own UI resource through this host.");
      reply({ contents: [{ uri: requested, mimeType: "text/html;profile=mcp-app", text: pendingHtml }] });
    } else if (method) {
      throw new Error(`Unsupported MCP App host method: ${method}`);
    }
  } catch (cause) { reply(undefined, cause instanceof Error ? cause.message : String(cause)); }
}

function requestEditor(method: "ui/llama-agent/prepare-chat" | "ui/llama-agent/prepare-document-switch" | "ui/llama-agent/close-document" | "ui/llama-agent/prepare-investigation-context"): Promise<unknown> {
  const target = frame.value?.contentWindow;
  const editorRequest = method !== "ui/llama-agent/prepare-investigation-context";
  if (!target || !initialized || (editorRequest
      ? uri.value !== "ui://llama-agent/apps/document-editor"
      : uri.value !== "ui://llama-agent/apps/investigation-workspace")) {
    return Promise.reject(new Error(editorRequest ? "The document editor is not ready." : "The investigation workspace is not ready."));
  }
  const id = hostRequestId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      hostRequests.delete(id); reject(new Error("The editor did not respond; its local draft is still preserved."));
    }, 210000);
    hostRequests.set(id, { method, resolve, reject, timer });
    target.postMessage({ jsonrpc: "2.0", id, method, params: {} }, "*");
  });
}

async function prepareForChat(): Promise<{ document_uri: string; revision: number; block_ids: string[] }> {
  const result = await requestEditor("ui/llama-agent/prepare-chat");
  if (!result || typeof result !== "object") throw new Error("The editor returned no saved document context.");
  const value = result as Record<string, unknown>;
  if (typeof value.document_uri !== "string" || !Number.isSafeInteger(value.revision) ||
      !Array.isArray(value.block_ids) || !value.block_ids.every((item) => typeof item === "string")) {
    throw new Error("The editor returned an invalid saved document context.");
  }
  return { document_uri: value.document_uri, revision: value.revision as number, block_ids: value.block_ids as string[] };
}

async function closeDocument() { await requestEditor("ui/llama-agent/close-document"); }
async function prepareDocumentSwitch(): Promise<boolean> {
  const result = await requestEditor("ui/llama-agent/prepare-document-switch");
  return Boolean(result && typeof result === "object" && (result as Record<string, unknown>).ready === true);
}
async function prepareInvestigationContext(): Promise<NonNullable<import("../types/agent").TurnRequest["investigation_context"]>> {
  const result = await requestEditor("ui/llama-agent/prepare-investigation-context");
  if (!result || typeof result !== "object") throw new Error("The workspace returned no case selection.");
  const value = result as Record<string, unknown>;
  const refs = (key: string) => {
    const list = value[key];
    const max = key === "source_ids" ? 32 : 64;
    if (!Array.isArray(list) || list.length > max || !list.every((item) => typeof item === "string" && item.length > 0 && item.length <= 512)) {
      throw new Error(`The workspace returned an invalid ${key} selection.`);
    }
    return [...new Set(list as string[])];
  };
  if (typeof value.case_id !== "string" || !value.case_id || value.case_id.length > 512 ||
      !Number.isSafeInteger(value.revision) || Number(value.revision) <= 0) throw new Error("The workspace returned an invalid case binding.");
  return { case_id: value.case_id, revision: Number(value.revision), entity_ids: refs("entity_ids"),
    relation_ids: refs("relation_ids"), event_ids: refs("event_ids"), source_ids: refs("source_ids"),
    ...(typeof value.analysis_profile === "string" && value.analysis_profile.length <= 64
      ? { analysis_profile: value.analysis_profile } : {}) };
}
defineExpose({ prepareForChat, prepareDocumentSwitch, closeDocument, prepareInvestigationContext });

watch(() => [props.tool.name, props.tool.provider_id, uri.value, props.sessionId], () => { initialized = false; void load(); }, { flush: "sync" });
watch(() => props.invocation, publishInvocation, { deep: true });
if (typeof window !== "undefined") window.addEventListener("message", onMessage);
void load();
onBeforeUnmount(() => { ++loadGeneration; window.removeEventListener("message", onMessage); for (const request of hostRequests.values()) { clearTimeout(request.timer); request.reject(new Error("The editor host was closed.")); } hostRequests.clear(); });
</script>

<template>
  <section class="card mcp-app-card">
    <div class="card-title"><h2>{{ tool.name }} · MCP App</h2><button class="quiet" :disabled="busy" @click="load">reload</button></div>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="appMessage" class="muted">App: {{ appMessage }}</p>
    <p v-if="busy" class="muted">Loading UI resource through the provider…</p>
    <iframe v-if="proxyUrl" ref="frame" :style="{ height: `${appHeight}px` }" title="MCP App sandbox" sandbox="allow-scripts allow-same-origin" referrerpolicy="no-referrer" :src="proxyUrl" />
  </section>
</template>
