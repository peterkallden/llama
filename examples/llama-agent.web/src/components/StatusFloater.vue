<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useAgentSession } from "../session";
import { mcpAppKey } from "./app-invocation";
import type { DaemonToolStatus } from "../types/agent";
const props = defineProps<{ activeView: string; activeApp: string }>();
const emit = defineEmits<{ navigate: [view: "chat" | "data" | "workflow" | "app", app?: string]; newSession: [] }>();
const session = useAgentSession();
const root = ref<HTMLElement | null>(null);
const open = ref(false);
const authExpanded = ref(false);
const statusExpanded = ref(false);
const toolsExpanded = ref(false);
const expandedTool = ref("");
const refreshing = ref(false);
const refreshError = ref("");
const readiness = computed(() => session.status.readiness ?? {});
const entries = computed(() => Object.entries(session.status.metrics ?? {}));
const apps = computed(() => {
  const grouped = new Map<string, DaemonToolStatus>();
  for (const tool of session.tools) {
    const uri = tool._meta?.ui?.resourceUri;
    if (!uri || !(tool._meta?.ui?.visibility ?? []).includes("app") ||
        uri === "ui://llama-agent/apps/data-explorer" || uri === "ui://llama-agent/apps/workflow-inspector") continue;
    if (!grouped.has(mcpAppKey(tool))) grouped.set(mcpAppKey(tool), tool);
  }
  return [...grouped.values()];
});
function label(tool: DaemonToolStatus) {
  return tool._meta?.ui?.resourceUri === "ui://llama-agent/apps/document-editor" ? "Document editor" : tool.name;
}
function navigate(view: "data" | "workflow" | "app", app = "") { emit("navigate", view, app); open.value = false; }
function outside(event: PointerEvent) { if (event.target instanceof Node && !root.value?.contains(event.target)) open.value = false; }
async function refresh() {
  refreshing.value = true; refreshError.value = "";
  try { await session.refreshStatus(); }
  catch (cause) { refreshError.value = cause instanceof Error ? cause.message : String(cause); }
  finally { refreshing.value = false; }
}
watch(() => session.connectionState, (state) => {
  if (state === "auth-error") { open.value = true; authExpanded.value = true; }
  else if (state === "connected") { authExpanded.value = false; open.value = false; }
});
onMounted(() => document.addEventListener("pointerdown", outside));
onBeforeUnmount(() => document.removeEventListener("pointerdown", outside));
</script>
<template>
  <aside ref="root" class="status-floater" aria-label="Connection, status and applications" @keydown.esc="open = false">
    <button type="button" class="floater-trigger" :class="`connection-${session.connectionState}`" :aria-expanded="open" aria-controls="workspace-controls" @click="open = !open"><span class="dot" /><span>{{ session.connectionState === 'auth-error' ? 'Authentication required' : session.connectionState }}</span><span aria-hidden="true">{{ open ? '▴' : '▾' }}</span></button>
    <div v-show="open" id="workspace-controls" class="floater-content">
      <section class="floater-section"><button type="button" class="section-toggle" :aria-expanded="authExpanded" @click="authExpanded = !authExpanded"><h2>Web client</h2><span>{{ authExpanded ? '−' : '+' }}</span></button>
        <div v-show="authExpanded"><label class="field token-field" :class="{ 'token-missing': session.connectionState === 'auth-error' }">API / bearer token<input v-model="session.token" type="password" autocomplete="off" :aria-invalid="session.connectionState === 'auth-error' || undefined" @keydown.enter.prevent="session.connect()" /><small>{{ session.connectionState === 'auth-error' ? 'Enter the required token and reconnect.' : session.connected && !session.token ? 'Connected without a browser token.' : 'Credentials remain in this browser tab.' }}</small></label></div>
        <p v-if="session.connectionError" class="connection-error" role="alert">{{ session.connectionError }}</p><div class="floater-actions"><button type="button" class="quiet" @click="session.connect()">Reconnect</button><button type="button" class="quiet" @click="emit('newSession')">New session</button></div>
      </section>
      <section class="floater-section"><button type="button" class="section-toggle" :aria-expanded="statusExpanded" @click="statusExpanded = !statusExpanded"><h2>Status and capabilities</h2><span>{{ statusExpanded ? '−' : '+' }}</span></button>
        <div v-show="statusExpanded"><button type="button" class="quiet" :disabled="refreshing" @click="refresh">{{ refreshing ? 'Refreshing…' : 'Refresh' }}</button><p v-if="refreshError" class="error" role="alert">{{ refreshError }}</p>
          <dl><template v-for="key in ['state', 'live', 'ready', 'worker_running', 'accepting_commands']" :key="key"><template v-if="session.status[key] !== undefined"><dt>{{ key.replaceAll('_', ' ') }}</dt><dd>{{ session.status[key] }}</dd></template></template><dt>Tool profile</dt><dd>{{ readiness.tool_profile || 'unknown' }}</dd></dl>
          <details><summary>Readiness and stores</summary><dl><template v-for="(value, key) in readiness.stores || {}" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template><dt>Model</dt><dd>{{ readiness.model }}</dd><dt>Inference</dt><dd>{{ readiness.inference }}</dd></dl></details>
          <p v-for="warning in readiness.warnings || []" :key="warning" class="error">{{ warning }}</p>
          <details v-if="readiness.providers?.length"><summary>Providers</summary><p v-for="provider in readiness.providers" :key="String(provider.id || provider.name)">{{ provider.id || provider.name }} · {{ provider.status }}</p></details>
          <dl v-if="entries.length"><template v-for="[key, value] in entries" :key="key"><dt>{{ key.replaceAll('_', ' ') }}</dt><dd>{{ value }}</dd></template></dl>
          <div class="capabilities"><span v-for="capability in session.capabilities" :key="capability" class="tag">{{ capability }}</span></div>
          <details><summary>Session keys</summary><pre>{{ JSON.stringify(session.status.session_keys ?? [], null, 2) }}</pre></details>
        </div>
      </section>
      <section class="floater-section"><h2>Applications</h2><nav class="app-launcher" aria-label="Open an application in the main workspace"><button type="button" :class="{ selected: activeView === 'data' }" @click="navigate('data')">Data Explorer ↗</button><button type="button" :class="{ selected: activeView === 'workflow' }" @click="navigate('workflow')">Workflow Inspector ↗</button><button v-for="tool in apps" :key="mcpAppKey(tool)" type="button" :class="{ selected: activeView === 'app' && activeApp === mcpAppKey(tool) }" @click="navigate('app', mcpAppKey(tool))">{{ label(tool) }} ↗</button></nav></section>
      <section class="floater-section"><button type="button" class="section-toggle" :aria-expanded="toolsExpanded" @click="toolsExpanded = !toolsExpanded"><h2>MCP tools · {{ session.tools.length }}</h2><span>{{ toolsExpanded ? '−' : '+' }}</span></button><div v-if="toolsExpanded" class="tool-list"><div v-for="tool in session.tools" :key="tool.name" class="tool-entry"><button type="button" class="tool-name" :aria-expanded="expandedTool === tool.name" @click="expandedTool = expandedTool === tool.name ? '' : tool.name"><span class="tool-dot" :class="`tool-dot-${tool.state || 'active'}`" />{{ tool.name }}</button><p v-if="expandedTool === tool.name" class="tool-details">{{ tool.description || 'No description provided.' }}</p></div></div></section>
      <section class="floater-section"><slot name="events" /></section>
      <section class="floater-section"><a href="./third-party-notices.txt" target="_blank" rel="noopener noreferrer">Third-party licenses</a></section>
    </div>
  </aside>
</template>
