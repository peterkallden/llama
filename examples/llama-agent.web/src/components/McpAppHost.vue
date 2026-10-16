<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";

const props = defineProps<{
  tool: DaemonToolStatus;
  tools: DaemonToolStatus[];
  sessionId: string;
  readResource: (toolName: string, uri: string) => Promise<unknown>;
  callTool: (appTool: string, toolName: string, args: Record<string, unknown>) => Promise<unknown>;
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
let lastInvocationSignature = "";

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

function cspMeta(meta: unknown): Record<string, string[]> {
  const outer = meta && typeof meta === "object" ? meta as Record<string, unknown> : {};
  const ui = outer.ui && typeof outer.ui === "object" ? outer.ui as Record<string, unknown> : outer;
  const csp = ui.csp && typeof ui.csp === "object" ? ui.csp as Record<string, unknown> : {};
  const domains = (field: string) => Array.isArray(csp[field])
    ? (csp[field] as unknown[]).flatMap((value) => {
      if (typeof value !== "string") return [];
      try {
        const candidate = value.startsWith("https://*.") ? `https://${value.slice("https://*".length)}` : value.includes("://") ? value : `https://${value}`;
        const url = new URL(candidate);
        if (url.protocol !== "https:" || url.username || url.password || url.pathname !== "/" || url.search || url.hash) return [];
        const hostname = value.includes("*.") ? `*.${url.hostname}` : url.hostname;
        return [/^(\*\.)?[a-z0-9.-]+$/i.test(hostname) ? `${url.protocol}//${hostname}${url.port ? `:${url.port}` : ""}` : ""].filter(Boolean);
      } catch { return []; }
    })
    : [];
  return {
    connectDomains: [...new Set(domains("connectDomains"))],
    resourceDomains: [...new Set(domains("resourceDomains"))],
    frameDomains: [...new Set(domains("frameDomains"))],
    baseUriDomains: [...new Set(domains("baseUriDomains"))],
  };
}

// The top-level sandbox is a data: document (opaque origin, different from the
// web host). It creates a separate opaque-origin view iframe and relays only
// postMessage traffic between that view and its parent host.
function sandboxProxyDocument(): string {
  return `<!doctype html><meta charset="utf-8"><script>
    let view;
    function sendHost(message){ parent.postMessage(message, '*'); }
    addEventListener('message', event => {
      if (event.source === parent && event.data?.method === 'ui/notifications/sandbox-resource-ready') {
        const { html, csp } = event.data.params || {};
        if (typeof html !== 'string' || !csp || typeof csp !== 'object') return;
        const domains = key => Array.isArray(csp[key]) ? csp[key].filter(value => typeof value === 'string' && /^(https:\/\/)?(\*\.)?[a-z0-9.-]+(:[0-9]+)?$/i.test(value)) : [];
        const resource = domains('resourceDomains').join(' ');
        const connect = domains('connectDomains').join(' ') || "'none'";
        const frames = domains('frameDomains').join(' ') || "'none'";
        const base = domains('baseUriDomains').join(' ') || "'none'";
        const policy = "default-src 'none'; script-src 'unsafe-inline' " + resource + "; style-src 'unsafe-inline' " + resource + "; img-src data: blob: " + resource + "; font-src " + resource + "; connect-src " + connect + "; frame-src " + frames + "; object-src 'none'; base-uri " + base + "; form-action 'none'";
        view = document.createElement('iframe');
        view.setAttribute('sandbox', 'allow-scripts');
        view.setAttribute('referrerpolicy', 'no-referrer');
        view.style.cssText = 'border:0;width:100%;height:100%;min-height:560px';
        view.srcdoc = html.replace(/<head\\b[^>]*>/i, head => head + '<meta http-equiv="Content-Security-Policy" content="' + policy.replaceAll('&','&amp;').replaceAll('"','&quot;') + '">');
        if (!/<head\\b/i.test(html)) view.srcdoc = '<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="' + policy + '"></head><body>' + html + '</body></html>';
        view.onload = () => {};
        document.body.replaceChildren(view);
        return;
      }
      if (event.source === parent && view?.contentWindow && !String(event.data?.method || '').includes('/sandbox-')) {
        view.contentWindow.postMessage(event.data, '*');
      } else if (view?.contentWindow === event.source && event.origin === 'null' && !String(event.data?.method || '').includes('/sandbox-')) {
        sendHost(event.data);
      }
    });
    sendHost({jsonrpc:'2.0', method:'ui/notifications/sandbox-proxy-ready'});
  </scr${"ipt"}>`;
}

let loadGeneration = 0;
async function load() {
  const generation = ++loadGeneration;
  initialized = false; proxyUrl.value = ""; error.value = "";
  lastInvocationSignature = "";
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
    if (html.length > 2 * 1024 * 1024) throw new Error("The MCP App resource exceeds the 2 MiB web-host limit.");
    const proxy = sandboxProxyDocument();
    proxyUrl.value = `data:text/html;charset=utf-8,${encodeURIComponent(proxy)}`;
    pendingHtml = html;
    pendingCsp = cspMeta(result._meta);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { busy.value = false; }
}

let pendingHtml = "";
let pendingCsp: Record<string, string[]> = {};
function send(message: Record<string, unknown>) {
  frame.value?.contentWindow?.postMessage(message, "*");
}

function publishInvocation() {
  if (!initialized || !props.invocation) return;
  const signature = JSON.stringify(props.invocation);
  if (!signature || signature === lastInvocationSignature) return;
  lastInvocationSignature = signature;
  if (props.invocation.input) {
    send({ jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: props.invocation.input } });
  }
  if ("result" in props.invocation) {
    send({ jsonrpc: "2.0", method: "ui/notifications/tool-result", params: { result: props.invocation.result } });
  }
}

async function onMessage(event: MessageEvent) {
  if (!frame.value?.contentWindow || event.source !== frame.value.contentWindow || event.origin !== "null") return;
  const message = event.data as Record<string, unknown>;
  if (!message || typeof message !== "object" || message.jsonrpc !== "2.0") return;
  const method = typeof message.method === "string" ? message.method : "";
  const id = message.id;
  const reply = (result: unknown, errorText?: string) => {
    if (id === undefined) return;
    send({
      jsonrpc: "2.0", id,
      ...(errorText ? { error: { code: -32000, message: errorText } } : { result }),
    });
  };
  if (method === "ui/notifications/sandbox-proxy-ready") {
    send({ jsonrpc: "2.0", method: "ui/notifications/sandbox-resource-ready", params: { html: pendingHtml, csp: pendingCsp } });
    return;
  }
  if (method === "ui/initialize") {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    if (!params.appInfo || !params.appCapabilities || params.protocolVersion !== "2026-01-26") {
      reply(undefined, "Invalid MCP App initialization request.");
      return;
    }
    reply({
      protocolVersion: "2026-01-26",
      hostInfo: { name: "llama-agent-web", version: "0.1" },
      hostCapabilities: { serverTools: { listChanged: false }, serverResources: { listChanged: false }, sandbox: {} },
      hostContext: { toolInfo: { tool: { name: props.tool.name, description: props.tool.description ?? "", inputSchema: props.tool.input_schema ?? { type: "object" } } } },
    });
    return;
  }
  if (method === "ui/notifications/initialized") {
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
    reply({});
    return;
  }
  if (method === "ui/open-link") { reply({ opened: false, message: "Links are not opened automatically by this host." }); return; }
  try {
    const params = message.params && typeof message.params === "object" ? message.params as Record<string, unknown> : {};
    if (method === "tools/list") {
      const listed = props.tools.filter((tool) => tool.provider_id === props.tool.provider_id &&
        (tool.app_read_only || (tool.provider_id === "" && tool.read_only)) &&
        (tool._meta?.ui?.visibility ?? []).includes("app"));
      reply({ tools: listed.map((tool) => ({
        name: tool.server_tool_name || tool.name,
        description: tool.description ?? "",
        inputSchema: tool.input_schema ?? { type: "object" },
        ...(tool.output_schema ? { outputSchema: tool.output_schema } : {}),
        annotations: { readOnlyHint: true },
      })) });
    } else if (method === "tools/call") {
      const name = String(params.name ?? "");
      const args = params.arguments && typeof params.arguments === "object" && !Array.isArray(params.arguments)
        ? params.arguments as Record<string, unknown> : {};
      const exposed = props.tools.find((tool) => tool.provider_id === props.tool.provider_id &&
        (tool.server_tool_name || tool.name) === name && (tool.app_read_only || (tool.provider_id === "" && tool.read_only)) &&
        (tool._meta?.ui?.visibility ?? []).includes("app"));
      if (!exposed) {
        reply({ isError: true, content: [{ type: "text", text: "The requested tool is not host-authorized for this app." }] });
      } else {
        try {
          const result = toolEnvelope(await props.callTool(props.tool.name, name, args));
          const failed = result.ok === false || Boolean(result.error);
          const errorInfo = result.error && typeof result.error === "object" ? result.error as Record<string, unknown> : {};
          reply({
            isError: failed,
            ...(result.structured_content !== undefined ? { structuredContent: result.structured_content } : {}),
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

watch(() => [props.tool.name, props.tool.provider_id, uri.value], () => { initialized = false; void load(); });
watch(() => props.invocation, publishInvocation, { deep: true });
if (typeof window !== "undefined") window.addEventListener("message", onMessage);
void load();
onBeforeUnmount(() => window.removeEventListener("message", onMessage));
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
