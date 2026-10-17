<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { AgentApiError } from "../api/agent-api";
import { structuredToolResult } from "./structured-tool-result";
import { datasetTable, sortedDatasetRows, tableCell } from "./dataset-table";

const props = defineProps<{
  tools: DaemonToolStatus[];
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
}>();
const emit = defineEmits<{ handoff: [prompt: string] }>();
const dataset = ref("");
const source = ref<"dataset" | "resource" | "path">("dataset");
const rows = ref(8);
const result = ref<Record<string, unknown> | null>(null);
const sortColumn = ref("");
const descending = ref(false);
const raw = ref(false);
const table = computed(() => datasetTable(result.value ?? {}));
const sortedRows = computed(() => sortedDatasetRows(table.value.rows, sortColumn.value, descending.value));
const busy = ref(false);
const error = ref("");
const available = computed(() => new Set(props.tools.filter((tool) => tool.read_only).map((tool) => tool.name)));
let requestGeneration = 0;
watch([dataset, source], () => { requestGeneration++; result.value = null; error.value = ""; busy.value = false; }, { flush: "sync" });

function sort(column: string) {
  descending.value = sortColumn.value === column ? !descending.value : false;
  sortColumn.value = column;
}

async function inspect(name: "dataset.inspect" | "dataset.schema" | "dataset.sample") {
  if (!dataset.value.trim()) { error.value = "Enter a source first."; return; }
  if (!available.value.has(name)) { error.value = `${name} is not available as a read-only tool.`; return; }
  const generation = ++requestGeneration;
  const requestedDataset = dataset.value.trim();
  const requestedSource = source.value;
  busy.value = true; error.value = ""; result.value = null;
  sortColumn.value = ""; descending.value = false;
  try {
    const count = Number.isFinite(rows.value) ? Math.max(1, Math.min(100, Math.trunc(rows.value))) : 8;
    const current = await props.callTool(name, name === "dataset.sample"
      ? { [requestedSource]: requestedDataset, rows: count }
      : { [requestedSource]: requestedDataset });
    if (generation === requestGeneration) result.value = structuredToolResult(current);
  } catch (cause) {
    if (generation === requestGeneration) {
      result.value = null;
      let failure = cause;
      if (cause instanceof AgentApiError) {
        try { structuredToolResult(cause.payload); } catch (safeError) { failure = safeError; }
      }
      error.value = failure instanceof Error ? failure.message : String(failure);
    }
  } finally { if (generation === requestGeneration) busy.value = false; }
}
</script>

<template>
  <section class="card explorer-card">
    <div class="card-title"><h2>Data Explorer</h2><span class="muted">read-only</span></div>
    <p class="muted">Browse a dataset, uploaded resource or repository file.</p>
    <label class="field">Source type<select v-model="source"><option value="dataset">Dataset reference</option><option value="resource">Resource reference</option><option value="path">Repository-relative path</option></select></label>
    <label class="field">Source<input v-model="dataset" :placeholder="source === 'path' ? 'datasets/example.csv' : source === 'resource' ? 'agent-resource://resource/…' : 'dataset://local/example'" /></label>
    <div class="explorer-actions">
      <button :disabled="busy || !available.has('dataset.inspect')" @click="inspect('dataset.inspect')">Inspect</button>
      <button :disabled="busy || !available.has('dataset.schema')" @click="inspect('dataset.schema')">Schema</button>
      <label>Sample rows <input v-model.number="rows" type="number" min="1" max="100" /></label>
      <button :disabled="busy || !available.has('dataset.sample')" @click="inspect('dataset.sample')">Sample</button>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <template v-if="result !== null">
      <p class="muted">{{ table.rows.length }} displayed rows. Click a column heading to sort the displayed rows.</p>
      <p v-if="result.scan_truncated || result.result_truncated" class="muted">This is a bounded sample; more rows exist outside this view.</p>
      <button type="button" class="explorer-toggle" :aria-pressed="raw" @click="raw = !raw">{{ raw ? 'Show table' : 'Show raw JSON' }}</button>
      <pre v-if="raw" class="explorer-result">{{ JSON.stringify(result, null, 2) }}</pre>
      <div v-else class="explorer-result table-scroll"><table class="dataset-table">
        <thead><tr><th v-for="column in table.columns" :key="column" scope="col" :aria-sort="sortColumn === column ? (descending ? 'descending' : 'ascending') : 'none'"><button type="button" @click="sort(column)">{{ column }}{{ sortColumn === column ? (descending ? ' ↓' : ' ↑') : '' }}</button></th></tr></thead>
        <tbody><tr v-for="(row, index) in sortedRows" :key="index"><td v-for="column in table.columns" :key="column">{{ tableCell(row[column]) }}</td></tr></tbody>
      </table><p v-if="!table.rows.length" class="muted">No rows returned.</p></div>
    </template>
    <button v-if="result !== null" class="handoff-button" @click="emit('handoff', `Analyze this dataset result for ${dataset}. The host returned this schema/sample: ${JSON.stringify(result)}`)">Continue with agent</button>
  </section>
</template>
