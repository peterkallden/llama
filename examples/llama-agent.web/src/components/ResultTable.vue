<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { datasetTable, sortedDatasetRows, tableCell } from "./dataset-table";
const props = defineProps<{ result: Record<string, unknown> }>();
const column = ref("");
const descending = ref(false);
const raw = ref(false);
const table = computed(() => datasetTable(props.result));
const rows = computed(() => sortedDatasetRows(table.value.rows, column.value, descending.value));
function sort(name: string) { descending.value = column.value === name ? !descending.value : false; column.value = name; }
watch(() => props.result, () => { column.value = ""; descending.value = false; raw.value = false; });
</script>
<template>
  <div class="result-table-view">
    <button type="button" class="explorer-toggle" :aria-pressed="raw" @click="raw = !raw">{{ raw ? 'Show table' : 'Show raw JSON' }}</button>
    <pre v-if="raw" class="explorer-result">{{ JSON.stringify(result, null, 2) }}</pre>
    <div v-else class="explorer-result table-scroll"><table class="dataset-table">
      <caption>{{ rows.length }} displayed rows. Column headings sort this displayed view.</caption>
      <thead><tr><th v-for="name in table.columns" :key="name" scope="col" :aria-sort="column === name ? (descending ? 'descending' : 'ascending') : 'none'"><button type="button" @click="sort(name)">{{ name }}{{ column === name ? (descending ? ' ↓' : ' ↑') : '' }}</button></th></tr></thead>
      <tbody><tr v-for="(row, index) in rows" :key="index"><td v-for="name in table.columns" :key="name">{{ tableCell(row[name]) }}</td></tr></tbody>
    </table><p v-if="!rows.length" class="muted">No rows returned.</p></div>
  </div>
</template>
