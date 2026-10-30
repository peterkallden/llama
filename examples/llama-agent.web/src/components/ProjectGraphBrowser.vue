<script setup lang="ts">
import { onBeforeUnmount, ref } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";
import { structuredToolResult } from "./structured-tool-result";
import ContextGraph from "./ContextGraph.vue";
import { projectGraphSearchArguments } from "./project-graph-search";

const props = defineProps<{ tools: DaemonToolStatus[]; callTool: (name: string, args: Record<string, unknown>) => Promise<unknown> }>();
const query = ref("");
const nodes = ref<Array<{ id: string; name: string; kind: string }>>([]);
const graph = ref<RelationshipVisualization>();
const relation = ref("");
const direction = ref<"from" | "to">("from");
const relationships = ref<Array<{ id: string; from: string; to: string; relation: string; status: string }>>([]);
const relationNodes = ref<Record<string, string>>({});
const relationSearched = ref(false);
const relationTruncated = ref(false);
const error = ref("");
const busy = ref(false);
const searched = ref(false);
const truncated = ref(false);
let generation = 0;
onBeforeUnmount(() => { ++generation; });
async function find() {
  if (!query.value.trim()) return;
  const current = ++generation; busy.value = true; error.value = ""; graph.value = undefined;
  nodes.value = []; searched.value = false; truncated.value = false;
  try {
    const result = structuredToolResult(await props.callTool("graph.find", { query: query.value.trim() }));
    if (!Array.isArray(result.nodes)) throw new Error("The graph search returned invalid nodes.");
    const valid = result.nodes.filter((node): node is { id: string; name: string; kind: string } =>
      !!node && typeof node === "object" && typeof node.id === "string" && typeof node.name === "string" && typeof node.kind === "string");
    if (current === generation) { nodes.value = valid; searched.value = true; truncated.value = result.truncated === true; }
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
async function open(id: string) {
  const current = ++generation; busy.value = true; error.value = "";
  try {
    const result = structuredToolResult(await props.callTool("graph.expand", { id, depth: 1 }));
    const next = parseRelationshipVisualization(result.visualization);
    if (current === generation) graph.value = next;
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
async function searchRelations(id: string) {
  const current = ++generation; busy.value = true; error.value = ""; graph.value = undefined;
  relationships.value = []; relationSearched.value = false; relationTruncated.value = false;
  try {
    const args = projectGraphSearchArguments(id, direction.value, relation.value);
    const result = structuredToolResult(await props.callTool("graph.search", args));
    if (!Array.isArray(result.nodes) || !Array.isArray(result.edges)) throw new Error("The relation search returned invalid data.");
    const labels: Record<string, string> = {};
    for (const node of result.nodes) if (node && typeof node.id === "string" && typeof node.name === "string") labels[node.id] = node.name;
    const edges = result.edges.filter((edge): edge is { id: string; from: string; to: string; relation: string; status: string } =>
      !!edge && typeof edge.id === "string" && typeof edge.from === "string" && typeof edge.to === "string" &&
      typeof edge.relation === "string" && typeof edge.status === "string");
    if (current === generation) { relationships.value = edges; relationNodes.value = labels; relationSearched.value = true; relationTruncated.value = result.truncated === true; }
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
</script>

<template>
  <form class="explorer-actions" @submit.prevent="find">
    <label>Find project graph nodes <input v-model="query" maxlength="512" placeholder="Name, path or source reference" /></label>
    <button type="submit" :disabled="busy || !query.trim()">Find</button>
  </form>
  <div v-if="nodes.length && tools.some((tool) => tool.name === 'graph.search' && tool.read_only && !tool.policy_gated)" class="explorer-actions">
    <label>Relation <input v-model="relation" maxlength="64" placeholder="Any relation" /></label>
    <label>Direction <select v-model="direction"><option value="from">From selected node</option><option value="to">To selected node</option></select></label>
  </div>
  <ul v-if="nodes.length"><li v-for="node in nodes" :key="node.id"><button type="button" :disabled="busy" @click="open(node.id)">{{ node.name }} · {{ node.kind }}</button>
    <button v-if="tools.some((tool) => tool.name === 'graph.search' && tool.read_only && !tool.policy_gated)" type="button" :disabled="busy" @click="searchRelations(node.id)">Find relations</button></li></ul>
  <p v-else-if="searched && !busy" class="muted">No matching project graph nodes.</p>
  <p v-if="truncated" class="muted">The search returned a bounded subset; refine the query for more specific results.</p>
  <ul v-if="relationships.length"><li v-for="edge in relationships" :key="edge.id">
    <button type="button" :disabled="busy" @click="open(edge.from)">{{ relationNodes[edge.from] ?? edge.from }}</button>
    — {{ edge.relation }} ({{ edge.status }}) →
    <button type="button" :disabled="busy" @click="open(edge.to)">{{ relationNodes[edge.to] ?? edge.to }}</button>
  </li></ul>
  <p v-else-if="relationSearched && !busy" class="muted">No matching relations.</p>
  <p v-if="relationTruncated" class="muted">The relation search was truncated; refine the filters.</p>
  <p v-if="busy" role="status">Loading project graph…</p>
  <p v-if="error" class="error" role="alert">{{ error }}</p>
  <ContextGraph v-if="graph" :visualization="graph" tool-name="graph.expand" :tools="tools" :call-tool="callTool" />
</template>
