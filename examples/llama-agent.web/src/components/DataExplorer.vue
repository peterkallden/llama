<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { DaemonToolStatus } from "../types/agent";

const props = defineProps<{
  tools: DaemonToolStatus[];
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
}>();
const emit = defineEmits<{ handoff: [prompt: string] }>();
const dataset = ref("");
const rows = ref(8);
const result = ref<unknown>(null);
const busy = ref(false);
const error = ref("");
const available = computed(() => new Set(props.tools.filter((tool) => tool.read_only).map((tool) => tool.name)));
let requestGeneration = 0;
watch(dataset, () => { requestGeneration++; result.value = null; error.value = ""; busy.value = false; });

async function inspect(name: "dataset.schema" | "dataset.sample") {
  if (!dataset.value.trim()) { error.value = "Enter a dataset reference first."; return; }
  if (!available.value.has(name)) { error.value = `${name} is not available as a read-only tool.`; return; }
  const generation = ++requestGeneration;
  const requestedDataset = dataset.value.trim();
  busy.value = true; error.value = ""; result.value = null;
  try {
    const current = await props.callTool(name, name === "dataset.sample"
      ? { dataset: requestedDataset, rows: Math.max(1, Math.min(100, rows.value)) }
      : { dataset: requestedDataset });
    if (generation === requestGeneration && dataset.value.trim() === requestedDataset) result.value = current;
  } catch (cause) {
    if (generation === requestGeneration) { result.value = null; error.value = cause instanceof Error ? cause.message : String(cause); }
  } finally { if (generation === requestGeneration) busy.value = false; }
}
</script>

<template>
  <section class="card explorer-card">
    <div class="card-title"><h2>Data Explorer</h2><span class="muted">read-only</span></div>
    <p class="muted">Inspect a host-authorized dataset without asking the model to serialize a tool call.</p>
    <label class="field">Dataset reference<input v-model="dataset" placeholder="dataset://local/example" /></label>
    <div class="explorer-actions">
      <button :disabled="busy || !available.has('dataset.schema')" @click="inspect('dataset.schema')">Schema</button>
      <label>Sample rows <input v-model.number="rows" type="number" min="1" max="100" /></label>
      <button :disabled="busy || !available.has('dataset.sample')" @click="inspect('dataset.sample')">Sample</button>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <pre v-if="result !== null" class="explorer-result">{{ JSON.stringify(result, null, 2) }}</pre>
    <button v-if="result !== null" class="handoff-button" @click="emit('handoff', `Analyze this dataset result for ${dataset}. The host returned this schema/sample: ${JSON.stringify(result)}`)">Continue with agent</button>
  </section>
</template>
