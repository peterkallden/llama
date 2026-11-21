<script setup lang="ts">
import { computed, defineAsyncComponent, onBeforeUnmount, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { mcpAppLabel } from "./app-invocation";
import { structuredToolResult } from "./structured-tool-result";
const ResearchEvidenceGraph = defineAsyncComponent(() => import("./ResearchEvidenceGraph.vue"));
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";
import type { RelationshipSelection } from "../visualization/relationship-renderer";

type ResourceRow = { uri: string; name: string; mime_type: string; description: string; size_bytes: number; scope: string };
const props = defineProps<{
  projectId: string;
  sessionId: string;
  tools: DaemonToolStatus[];
  listResources: (input: { scope: "project" | "session"; query?: string; mimeType?: string; cursor?: string; limit?: number }) => Promise<{ resources: Array<Record<string, unknown>>; next_cursor: string }>;
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  readResource: (uri: string) => Promise<Blob>;
}>();
const emit = defineEmits<{
  openData: [uri: string, kind: "resource" | "dataset"];
  attach: [uri: string];
  openApp: [tool: DaemonToolStatus, input?: Record<string, unknown>];
}>();
const tab = ref<"resources" | "datasets" | "workspaces" | "work-products">("resources");
const scope = ref<"session" | "project">("session");
const query = ref("");
const rows = ref<ResourceRow[]>([]);
const nextCursor = ref("");
const busy = ref(false);
const error = ref("");
const selected = ref<ResourceRow>();
const preview = ref("");
const cases = ref<Array<Record<string, unknown>>>([]);
const datasets = ref<string[]>([]);
const casesLoading = ref(false);
const datasetsLoading = ref(false);
const casesLoaded = ref(false);
const datasetsLoaded = ref(false);
const casesUnavailable = ref(false);
const datasetsUnavailable = ref(false);
const researchGraph = ref<RelationshipVisualization>();
const graphBusy = ref(false);
const graphError = ref("");
const graphSelection = ref<RelationshipSelection>();
const apps = computed(() => {
  const unique = new Map<string, DaemonToolStatus>();
  for (const tool of props.tools) {
    const ui = tool._meta?.ui;
    if (ui?.resourceUri && (ui.visibility ?? []).includes("app")) unique.set(String(ui.resourceUri), tool);
  }
  return [...unique.values()];
});
let generation = 0;
let scopeGeneration = 0;
let previewGeneration = 0;
let searchTimer: ReturnType<typeof setTimeout> | undefined;

function toRow(item: Record<string, unknown>): ResourceRow {
  return { uri: String(item.uri ?? ""), name: String(item.name ?? "Resource"), mime_type: String(item.mime_type ?? "application/octet-stream"),
    description: String(item.description ?? ""), size_bytes: Number(item.size_bytes ?? 0), scope: String(item.scope ?? scope.value) };
}

async function load(append = false) {
  const request = ++generation;
  busy.value = true; error.value = "";
  try {
    const result = await props.listResources({ scope: scope.value, query: query.value.trim(), cursor: append ? nextCursor.value : "", limit: 30 });
    if (request !== generation) return;
    rows.value = append ? [...rows.value, ...result.resources.map(toRow)] : result.resources.map(toRow);
    nextCursor.value = result.next_cursor || "";
  } catch (cause) {
    if (request === generation) error.value = cause instanceof Error ? cause.message : String(cause);
  } finally { if (request === generation) busy.value = false; }
}

async function selectResource(item: ResourceRow) {
  const request = ++previewGeneration;
  selected.value = item; preview.value = ""; error.value = ""; researchGraph.value = undefined; graphError.value = ""; graphSelection.value = undefined; graphBusy.value = false;
  try {
    const blob = await props.readResource(item.uri);
    if (request !== previewGeneration) return;
    if (blob.size > 512 * 1024) { preview.value = `Preview omitted (${blob.size.toLocaleString()} bytes). Open the resource in Data Explorer to inspect it in bounded pages.`; return; }
    if (item.mime_type.startsWith("text/") || /json|csv|xml|yaml/.test(item.mime_type)) {
      const text = await blob.text();
      if (request === previewGeneration) preview.value = text.slice(0, 12_000);
    }
    else preview.value = `${item.mime_type} · ${blob.size.toLocaleString()} bytes`;
  } catch (cause) { if (request === previewGeneration) error.value = cause instanceof Error ? cause.message : String(cause); }
}

async function showResearchGraph(item: ResourceRow) {
  if (graphBusy.value) return;
  const request = previewGeneration;
  if (!props.tools.some((tool) => tool.name === "visualize.research_graph" && tool.read_only && !tool.policy_gated)) {
    graphError.value = "Research graph projection is not available as an approved read-only tool in this runtime.";
    return;
  }
  graphBusy.value = true; graphError.value = ""; graphSelection.value = undefined;
  try {
    const result = structuredToolResult(await props.callTool("visualize.research_graph", { source_uri: item.uri }));
    if (request !== previewGeneration) return;
    researchGraph.value = parseRelationshipVisualization(result.visualization);
  } catch (cause) { if (request === previewGeneration) graphError.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (request === previewGeneration) graphBusy.value = false; }
}

async function loadCases() {
  const request = scopeGeneration;
  if (casesLoaded.value || casesLoading.value) return;
  if (!props.tools.some((tool) => tool.name === "investigate.case.list" && tool.read_only && !tool.policy_gated)) {
    casesUnavailable.value = true;
    casesLoaded.value = true;
    return;
  }
  casesUnavailable.value = false;
  casesLoading.value = true;
  try {
    const result = structuredToolResult(await props.callTool("investigate.case.list", { limit: 50 }));
    if (request !== scopeGeneration) return;
    cases.value = Array.isArray(result.cases) ? result.cases.filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object")) : [];
    casesLoaded.value = true;
  } catch (cause) { if (request === scopeGeneration) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (request === scopeGeneration) casesLoading.value = false; }
}

function openCase(item: Record<string, unknown>) {
  const app = apps.value.find((candidate) => candidate._meta?.ui?.resourceUri === "ui://llama-agent/apps/investigation-workspace");
  const caseId = String(item.case_id ?? item.id ?? "");
  if (app && caseId) emit("openApp", app, { case_id: caseId });
}

async function loadDatasets() {
  const request = scopeGeneration;
  if (datasetsLoaded.value || datasetsLoading.value) return;
  if (!props.tools.some((tool) => tool.name === "dataset.list" && tool.read_only && !tool.policy_gated)) {
    datasetsUnavailable.value = true;
    datasetsLoaded.value = true;
    return;
  }
  datasetsUnavailable.value = false;
  datasetsLoading.value = true;
  try {
    const result = structuredToolResult(await props.callTool("dataset.list", { max_results: 256 }));
    if (request !== scopeGeneration) return;
    datasets.value = Array.isArray(result.datasets) ? result.datasets.map((item) =>
      typeof item === "string" ? item : String((item as Record<string, unknown>)?.ref ?? (item as Record<string, unknown>)?.uri ?? "")).filter(Boolean) : [];
    datasetsLoaded.value = true;
  } catch (cause) { if (request === scopeGeneration) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (request === scopeGeneration) datasetsLoading.value = false; }
}

watch([scope, () => props.projectId, () => props.sessionId], () => {
  ++scopeGeneration; ++generation; ++previewGeneration;
  if (searchTimer) clearTimeout(searchTimer);
  researchGraph.value = undefined; graphError.value = ""; graphSelection.value = undefined; graphBusy.value = false;
  rows.value = []; nextCursor.value = ""; selected.value = undefined; preview.value = "";
  cases.value = []; datasets.value = []; casesLoaded.value = false; datasetsLoaded.value = false;
  casesUnavailable.value = false; datasetsUnavailable.value = false; casesLoading.value = false; datasetsLoading.value = false;
  if (!props.projectId && scope.value === "project") { scope.value = "session"; return; }
  void load();
  if (tab.value === "work-products") void loadCases();
  if (tab.value === "datasets") void loadDatasets();
}, { immediate: true });
watch(query, () => {
  ++generation;
  rows.value = []; nextCursor.value = ""; busy.value = true;
  if (searchTimer) clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { rows.value = []; nextCursor.value = ""; void load(); }, 200);
});
watch(tab, (value) => { if (value === "work-products") void loadCases(); if (value === "datasets") void loadDatasets(); });
onBeforeUnmount(() => {
  ++generation; ++scopeGeneration; ++previewGeneration;
  if (searchTimer) clearTimeout(searchTimer);
});

function toggleAttach(item: ResourceRow) { emit("attach", item.uri); }
</script>

<template>
  <section class="resource-browser card">
    <div class="card-title"><h2>Resources and workspaces</h2><span class="muted">Host-authorized</span></div>
    <nav class="resource-tabs" aria-label="Browse scope">
      <button type="button" :class="{ active: tab === 'resources' }" @click="tab = 'resources'">Resources</button>
      <button type="button" :class="{ active: tab === 'datasets' }" @click="tab = 'datasets'">Datasets</button>
      <button type="button" :class="{ active: tab === 'workspaces' }" @click="tab = 'workspaces'">Workspaces</button>
      <button type="button" :class="{ active: tab === 'work-products' }" @click="tab = 'work-products'">Work products</button>
    </nav>
    <template v-if="tab === 'resources'">
      <div class="resource-controls">
        <label>Scope<select v-model="scope"><option value="session">This chat</option><option v-if="projectId" value="project">This project</option></select></label>
        <label class="resource-search">Search<input v-model="query" type="search" placeholder="Name or description" /></label>
      </div>
      <div class="resource-results" aria-live="polite">
        <button v-for="item in rows" :key="item.uri" type="button" class="resource-row" :class="{ chosen: selected?.uri === item.uri }" @click="selectResource(item)">
          <span class="resource-name">{{ item.name }}</span><span class="resource-type">{{ item.mime_type }}</span><small>{{ item.description || item.scope }}</small>
        </button>
        <p v-if="!busy && !rows.length" class="muted">No resources in this scope.</p>
      </div>
      <button v-if="nextCursor" type="button" class="resource-more" :disabled="busy" @click="load(true)">{{ busy ? 'Loading…' : 'Load more' }}</button>
      <aside v-if="selected" class="resource-preview">
        <div><strong>{{ selected.name }}</strong><small>{{ selected.uri }}</small></div>
        <pre v-if="preview">{{ preview }}</pre>
        <div class="resource-actions">
          <button type="button" @click="emit('openData', selected!.uri, /dataset/i.test(selected!.uri) ? 'dataset' : 'resource')">Open in Data Explorer</button>
          <button type="button" @click="toggleAttach(selected!)">Use in chat</button>
          <button v-if="selected.mime_type === 'application/vnd.llama-agent.research-workspace+json' || selected.mime_type === 'application/vnd.llama-agent.report+json' || selected.mime_type === 'application/vnd.llama-agent.document+json'" type="button" :disabled="graphBusy" @click="showResearchGraph(selected)">{{ graphBusy ? 'Building graph…' : 'Open evidence graph' }}</button>
        </div>
      </aside>
      <p v-if="graphError" class="error" role="alert">{{ graphError }}</p>
      <p v-if="graphSelection" class="muted" role="status">Selected {{ graphSelection.kind }} {{ graphSelection.id }}</p>
      <ResearchEvidenceGraph v-if="researchGraph" :visualization="researchGraph" @select="graphSelection = $event" />
    </template>
    <template v-else-if="tab === 'datasets'">
      <p class="muted">Registered datasets available to this chat.</p>
      <div class="resource-results">
        <button v-for="item in datasets" :key="item" type="button" class="resource-row" @click="emit('openData', item, 'dataset')"><span class="resource-name">{{ item }}</span><small>Dataset</small></button>
        <p v-if="datasetsLoading" class="muted">Loading datasets…</p>
        <p v-else-if="datasetsUnavailable" class="muted">Dataset listing is not available as an approved read-only tool in this runtime.</p>
        <p v-else-if="datasetsLoaded && !datasets.length" class="muted">No datasets are registered for this scope.</p>
      </div>
    </template>
    <template v-else-if="tab === 'workspaces'">
      <p class="muted">Workspace types open the existing tools with this chat's authorized scope.</p>
      <button v-for="tool in apps" :key="tool.name" type="button" class="workspace-row" @click="emit('openApp', tool)"><span>{{ mcpAppLabel(tool) }}</span><small>{{ tool.description }}</small></button>
      <p v-if="!apps.length" class="muted">No workspace apps are available in this runtime.</p>
    </template>
    <template v-else>
      <p class="muted">Saved investigation cases are work products; opening one uses the existing Investigation Workspace.</p>
      <button v-for="item in cases" :key="String(item.case_id ?? item.id)" type="button" class="workspace-row" @click="openCase(item)">
        <span>{{ String(item.title ?? item.case_id ?? item.id ?? 'Investigation case') }}</span><small>{{ String(item.status ?? '') }}</small>
      </button>
      <p v-if="casesLoading" class="muted">Loading saved cases…</p>
      <p v-else-if="casesUnavailable" class="muted">Investigation case listing is not available as an approved read-only tool in this runtime.</p>
      <p v-else-if="casesLoaded && !cases.length" class="muted">No saved cases are available in this scope.</p>
    </template>
    <p v-if="busy && tab === 'resources'" class="muted">Loading resources…</p>
    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>

<style scoped>
.resource-browser{padding:18px}.resource-tabs{display:flex;gap:7px;border-bottom:1px solid #30445f;padding-bottom:9px;margin:14px 0}.resource-tabs button,.resource-actions button,.resource-more{border:1px solid #344a64;background:#122035;color:#bfd0e4;border-radius:6px;padding:7px 10px}.resource-tabs button.active{background:#7dd3fc;color:#07111d;font-weight:700}.resource-controls{display:flex;gap:12px;align-items:end;margin:10px 0}.resource-controls label{display:grid;gap:5px;color:#9db1ca;font-size:.82rem}.resource-controls input,.resource-controls select{background:#0b1523;color:#e5edf7;border:1px solid #30445f;border-radius:6px;padding:8px}.resource-search{flex:1}.resource-results{max-height:38vh;overflow:auto;border:1px solid #263b55;border-radius:8px}.resource-row,.workspace-row{display:grid;width:100%;text-align:left;gap:3px;padding:10px 12px;border:0;border-bottom:1px solid #263b55;background:transparent;color:#e5edf7}.resource-row:hover,.resource-row.chosen,.workspace-row:hover{background:#172941}.resource-name,.workspace-row span{font-weight:650}.resource-type,.resource-row small,.workspace-row small,.resource-preview small{color:#92a9c3;font-size:.76rem}.resource-preview{margin-top:12px;padding:12px;background:#0e1928;border-radius:8px}.resource-preview>div:first-child{display:grid;gap:4px}.resource-preview pre{max-height:220px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;color:#cbd7e6;font-size:.8rem}.resource-actions{display:flex;gap:8px;flex-wrap:wrap}.resource-more{margin:10px 0}.workspace-row{margin:8px 0;border:1px solid #263b55;border-radius:8px}
</style>
