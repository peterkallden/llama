<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { canonicalToolStructuredContent, cloneJsonMessage, invocationNotifications, isCurrentAppRequest, normalizeMcpAppCsp, sandboxProxyDocument } from "./mcp-app-host";
import { DocumentEditorAccess } from "./document-editor-access";

const props = defineProps<{
  tool: DaemonToolStatus;
  tools: DaemonToolStatus[];
  sessionId: string;
  readResource: (toolName: string, uri: string) => Promise<unknown>;
  callTool: (appTool: string, toolName: string, args: Record<string, unknown>) => Promise<unknown>;
  submitReview?: (prompt: string, documentContext: { document_uri: string; revision: number; block_ids: string[] }) => Promise<unknown>;
  invocation?: { input?: Record<string, unknown>; result?: unknown };
}>();
const frame = ref<HTMLIFrameElement | null>(null);
const proxyUrl = ref("");
const error = ref("");
const busy = ref(false);
const appMessage = ref("");
const uri = computed(() => String(props.tool._meta?.ui?.resourceUri ?? ""));
const appVisible = computed(() => (props.tool._meta?.ui?.visibility ?? []).includes("app"));
let initialized = false;
let initializeAccepted = false;
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
  initialized = false; initializeAccepted = false; proxyUrl.value = ""; error.value = "";
  busy.value = false; appMessage.value = "";
  lastInvocationSignature = "";
  documentAccess = new DocumentEditorAccess(props.sessionId, props.tool.provider_id ?? "");
  pendingHtml = ""; pendingCsp = {};
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
  if (method === "ui/notifications/size-changed") return;
  if (method === "ping") { reply({}); return; }
  if (method === "notifications/message") return;
  if (method === "ui/message") {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    const content = params.content && typeof params.content === "object" ? params.content as Record<string, unknown> : {};
    appMessage.value = typeof content.text === "string" ? content.text : "App sent a message.";
    if (typeof content.text === "string" && uri.value === "ui://llama-agent/apps/document-editor") {
      const rawBinding = params.document_context && typeof params.document_context === "object"
        ? params.document_context as Record<string, unknown> : {};
      try {
        if (!props.submitReview) throw new Error("Agent review is unavailable in this host.");
        if (typeof rawBinding.document_uri !== "string" || typeof rawBinding.revision !== "number" || !Number.isSafeInteger(rawBinding.revision) ||
            !Array.isArray(rawBinding.block_ids) || !rawBinding.block_ids.every((item) => typeof item === "string")) {
          throw new Error("The editor did not provide a valid typed document binding.");
        }
        documentAccess.authorize("document.propose_patch", {
          document_uri: rawBinding.document_uri, expected_revision: rawBinding.revision,
        });
        await props.submitReview(content.text, {
          document_uri: rawBinding.document_uri,
          revision: rawBinding.revision as number,
          block_ids: rawBinding.block_ids as string[],
        });
        reply({});
      } catch (cause) { reply(undefined, cause instanceof Error ? cause.message : String(cause)); }
    } else {
      reply({});
    }
    return;
  }
  if (method === "ui/open-link") { reply({ opened: false, message: "Links are not opened automatically by this host." }); return; }
  try {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    if (method === "ui/llama-agent/document-draft") {
      if (uri.value !== "ui://llama-agent/apps/document-editor") throw new Error("Draft storage is reserved for the first-party editor.");
      let storage: Storage | undefined;
      try { storage = window.localStorage; } catch { /* host can still retain drafts in memory */ }
      reply(documentAccess.draft(params, storage));
    } else if (method === "tools/list") {
      const listed = props.tools.filter((tool) => tool.provider_id === props.tool.provider_id &&
        (tool.app_read_only || tool.app_write_authorized || (tool.provider_id === "" && tool.read_only)) &&
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
      const exposed = props.tools.find((tool) => tool.provider_id === props.tool.provider_id &&
        (tool.server_tool_name || tool.name) === name && (tool.app_read_only || tool.app_write_authorized || (tool.provider_id === "" && tool.read_only)) &&
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

watch(() => [props.tool.name, props.tool.provider_id, uri.value, props.sessionId], () => { initialized = false; void load(); }, { flush: "sync" });
watch(() => props.invocation, publishInvocation, { deep: true });
if (typeof window !== "undefined") window.addEventListener("message", onMessage);
void load();
onBeforeUnmount(() => { ++loadGeneration; window.removeEventListener("message", onMessage); });
</script>

<template>
  <section class="card mcp-app-card">
    <div class="card-title"><h2>{{ tool.name }} · MCP App</h2><button class="quiet" :disabled="busy" @click="load">reload</button></div>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="appMessage" class="muted">App: {{ appMessage }}</p>
    <p v-if="busy" class="muted">Loading UI resource through the provider…</p>
    <iframe v-if="proxyUrl" ref="frame" title="MCP App sandbox" sandbox="allow-scripts allow-same-origin" referrerpolicy="no-referrer" :src="proxyUrl" />
  </section>
</template>
