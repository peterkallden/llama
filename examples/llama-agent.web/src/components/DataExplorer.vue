<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import { AgentApiError } from "../api/agent-api";
import { structuredToolResult } from "./structured-tool-result";
import ResultTable from "./ResultTable.vue";

const props = defineProps<{
  tools: DaemonToolStatus[];
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  initialSource?: string;
  initialKind?: "dataset" | "resource" | "path";
  hideHandoff?: boolean;
}>();
const emit = defineEmits<{ handoff: [prompt: string] }>();
const dataset = ref("");
const source = ref<"dataset" | "resource" | "path">("dataset");
const rows = ref(8);
const result = ref<Record<string, unknown> | null>(null);
const busy = ref(false);
const error = ref("");
const available = computed(() => new Set(props.tools.filter((tool) => tool.read_only).map((tool) => tool.name)));
let requestGeneration = 0;
watch([dataset, source], () => { requestGeneration++; result.value = null; error.value = ""; busy.value = false; }, { flush: "sync" });

async function inspect(name: "dataset.inspect" | "dataset.schema" | "dataset.sample") {
  if (!dataset.value.trim()) { error.value = "Enter a source first."; return; }
  if (!available.value.has(name)) { error.value = `${name} is not available as a read-only tool.`; return; }
  const generation = ++requestGeneration;
  const requestedDataset = dataset.value.trim();
  const requestedSource = source.value;
  busy.value = true; error.value = ""; result.value = null;
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
watch(() => [props.initialSource, props.initialKind], () => {
  if (!props.initialSource) return;
  source.value = props.initialKind ?? "dataset";
  dataset.value = props.initialSource;
  void inspect("dataset.sample");
}, { immediate: true });
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
      <p v-if="result.scan_truncated || result.result_truncated" class="muted">This is a bounded sample; more rows exist outside this view.</p>
      <ResultTable :result="result" />
    </template>
    <button v-if="result !== null && !hideHandoff" class="handoff-button" @click="emit('handoff', source === 'dataset' ? `Analyze the already selected dataset ${dataset}. Use this exact dataset reference directly with dataset.schema and dataset.sample or data.query; do not look it up again with dataset.select.` : `Analyze the ${source} source ${dataset}. Read its schema and relevant rows through the dataset tools.`)">Continue with agent</button>
  </section>
</template>
