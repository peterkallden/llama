<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { DaemonToolStatus } from "../types/agent";
import DataExplorer from "./DataExplorer.vue";
import ResultTable from "./ResultTable.vue";
import { externalUri, savedToolPayload, type InspectionTarget } from "./report-view";
import { structuredToolResult } from "./structured-tool-result";
const props = defineProps<{ target: InspectionTarget; tools: DaemonToolStatus[];
  callTool: (name: string, args: Record<string, unknown>) => Promise<unknown>;
  readResource: (uri: string) => Promise<Blob>;
}>();
const emit = defineEmits<{ close: [] }>();
const modal = ref<HTMLDialogElement | null>(null);
const error = ref("");
const busy = ref(false);
const preview = ref<Record<string, unknown>>();
const external = computed(() => externalUri(props.target.originalUri));
const toolPayload = computed(() => savedToolPayload(props.target.detail ?? {}));
let disposed = false;
async function readPreview() {
  if (!props.target.uri || busy.value) return;
  busy.value = true; error.value = "";
  try {
    const result = structuredToolResult(await props.callTool("resource.read", { uri: props.target.uri, representation: "text", max_bytes: 32768 }));
    if (!disposed) preview.value = result;
  } catch (cause) { if (!disposed) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (!disposed) busy.value = false; }
}
async function openResource() {
  if (!props.target.uri || busy.value) return;
  const tab = window.open("about:blank", "_blank");
  if (!tab) { error.value = "The browser blocked the new tab. Allow it and try again."; return; }
  tab.opener = null;
  busy.value = true; error.value = "";
  try {
    const blob = await props.readResource(props.target.uri);
    if (disposed) { tab.close(); return; }
    // Never execute uploaded HTML or SVG under the web client's origin.
    const safe = blob.type.split(";")[0] === "application/pdf" ? blob
      : new Blob([blob], { type: blob.type.startsWith("text/") || blob.type.includes("json") ? "text/plain" : "application/octet-stream" });
    const url = URL.createObjectURL(safe);
    // The separate tab owns its URL until it closes, not the inspection modal.
    const release = () => { window.clearInterval(timer); URL.revokeObjectURL(url); window.removeEventListener("pagehide", release); };
    const timer = window.setInterval(() => { if (tab.closed) release(); }, 1000);
    window.addEventListener("pagehide", release, { once: true });
    tab.location.replace(url);
  } catch (cause) { tab.close(); if (!disposed) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (!disposed) busy.value = false; }
}
onMounted(() => modal.value?.showModal());
onBeforeUnmount(() => { disposed = true; });
</script>
<template>
  <Teleport to="body"><dialog ref="modal" class="source-dialog" aria-labelledby="source-dialog-title" @close="emit('close')" @click="(event) => { if (event.target === modal) modal?.close(); }">
    <header class="source-dialog-heading"><div><span class="eyebrow">{{ target.kind === 'tool' ? 'Saved tool observation' : 'Source material' }}</span><h2 id="source-dialog-title">{{ target.title }}</h2></div><button type="button" class="icon-button" title="Close source" aria-label="Close source" @click="modal?.close()">×</button></header>
    <div class="source-dialog-content">
      <p v-if="external"><a :href="external" target="_blank" rel="noopener noreferrer">Open original source in a new tab ↗</a></p>
      <p v-if="target.uri" class="source-uri">{{ target.uri }}</p>
      <DataExplorer v-if="target.kind === 'dataset' && target.uri" :tools="tools" :call-tool="callTool" :initial-source="target.uri" initial-kind="dataset" :hide-handoff="true" />
      <template v-else-if="target.kind === 'tool'"><p class="muted">This is the observation retained in the frozen report snapshot. The tool has not been rerun; its original output may have been projected or bounded.</p><ResultTable :result="toolPayload" /></template>
      <template v-else-if="target.uri"><div class="explorer-actions"><button type="button" :disabled="busy || !tools.some((tool) => tool.name === 'resource.read' && tool.read_only)" @click="readPreview">Read bounded text preview</button><button type="button" :disabled="busy" @click="openResource">Open saved file in a new tab ↗</button></div><p v-if="preview" class="muted">Bounded preview, at most 32 KiB. This does not establish full reading coverage.</p><ResultTable v-if="preview" :result="preview" /></template>
      <p v-if="busy" role="status">Loading authorized source…</p><p v-if="error" class="error" role="alert">{{ error }}</p>
      <details v-if="target.detail"><summary>Saved source metadata</summary><ResultTable :result="target.detail" /></details>
    </div>
  </dialog></Teleport>
</template>
