<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";
import type { RelationshipSelection } from "../visualization/relationship-renderer";
import { structuredToolResult } from "./structured-tool-result";
import RelationshipGraph from "./RelationshipGraph.vue";

const props = defineProps<{ visualization: RelationshipVisualization; toolName: string; tools: DaemonToolStatus[];
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown> }>();
const graph = ref(props.visualization);
const selection = ref<RelationshipSelection>();
const item = ref<Record<string, unknown>>();
const source = ref<Record<string, unknown>>();
const error = ref("");
const busy = ref(false);
let generation = 0;
const authorized = (name: string) => props.tools.some((tool) => tool.name === name && tool.read_only && !tool.policy_gated);
const isProjectGraph = computed(() => props.toolName === "graph.expand");
const path = computed(() => typeof item.value?.path === "string" ? item.value.path : "");
watch(() => props.visualization, (value) => { ++generation; graph.value = value; selection.value = undefined; item.value = undefined; source.value = undefined; error.value = ""; busy.value = false; });
onBeforeUnmount(() => { ++generation; });

async function select(value: RelationshipSelection) {
  selection.value = value; item.value = undefined; source.value = undefined; error.value = "";
  const current = ++generation;
  if (!isProjectGraph.value || !authorized("graph.get")) return;
  busy.value = true;
  try {
    const result = structuredToolResult(await props.callTool("graph.get", { id: value.id }));
    const selected = result[value.kind];
    if (current === generation && selected && typeof selected === "object" && !Array.isArray(selected)) item.value = selected as Record<string, unknown>;
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
async function expand() {
  if (!selection.value || selection.value.kind !== "node" || !authorized("graph.expand")) return;
  const current = ++generation; busy.value = true; error.value = "";
  try {
    const result = structuredToolResult(await props.callTool("graph.expand", { id: selection.value.id, depth: 1 }));
    const next = parseRelationshipVisualization(result.visualization);
    if (current === generation) { graph.value = next; selection.value = undefined; item.value = undefined; source.value = undefined; }
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
async function readSource() {
  if (!path.value || !authorized("repository.read")) return;
  const current = ++generation; busy.value = true; error.value = "";
  const line = typeof item.value?.line === "number" && item.value.line > 0 ? item.value.line : 1;
  try {
    const result = structuredToolResult(await props.callTool("repository.read", { path: path.value, start_line: Math.max(1, line - 5), end_line: Math.min(1000000, line + 15) }));
    if (current === generation) source.value = result;
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
</script>

<template>
  <div class="report-heading"><strong>{{ graph.title }}</strong><span>{{ isProjectGraph ? 'Project graph' : 'Source analysis' }}</span></div>
  <RelationshipGraph :visualization="graph" @select="select" />
  <p v-if="graph.limitations?.length" class="muted">{{ graph.limitations.join(' ') }}</p>
  <template v-if="isProjectGraph && selection">
    <p class="muted">Selected {{ selection.kind }}: {{ selection.id }}</p>
    <pre v-if="item" class="explorer-result">{{ JSON.stringify(item, null, 2) }}</pre>
    <button v-if="selection.kind === 'node' && authorized('graph.expand')" type="button" :disabled="busy" @click="expand">Expand selected node</button>
    <button v-if="path && authorized('repository.read')" type="button" :disabled="busy" @click="readSource">Read source around location</button>
    <pre v-if="source" class="explorer-result">{{ JSON.stringify(source, null, 2) }}</pre>
    <p v-if="busy" role="status">Loading graph data…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
  </template>
</template>
