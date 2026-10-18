<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import MarkdownView from "./MarkdownView.vue";
import { blockTargets, externalUri, object, objects, savedToolPayload, sourceTargets, strings, type Fields, type InspectionTarget } from "./report-view";
import { structuredToolResult } from "./structured-tool-result";
const props = defineProps<{
  uri: string; name: string; sessionId: string; editable: boolean;
  openDocument: (uri: string) => Promise<unknown>;
  readJson: (uri: string) => Promise<Fields>;
}>();
const emit = defineEmits<{ edit: [uri: string]; inspect: [target: InspectionTarget] }>();
const opened = ref<Fields>({});
const snapshot = ref<Fields>({});
const error = ref("");
const provenanceError = ref("");
const busy = ref(false);
const downloading = ref(false);
const document = computed(() => object(opened.value.document));
const sections = computed(() => objects(document.value.sections));
const targetIndex = computed(() => new Map<Fields, InspectionTarget[]>(sections.value.flatMap((section) =>
  objects(section.blocks).map((block) => [block, blockTargets(block, document.value, snapshot.value)] as [Fields, InspectionTarget[]]))));
let generation = 0;
async function load() {
  const current = ++generation;
  busy.value = true; error.value = ""; provenanceError.value = ""; opened.value = {}; snapshot.value = {};
  try {
    const result = structuredToolResult(await props.openDocument(props.uri));
    if (current !== generation) return;
    if (result.document_uri !== props.uri) throw new Error("The returned document does not match the selected revision.");
    opened.value = result;
    const uri = object(result.document).snapshot_uri;
    if (typeof uri === "string" && uri) {
      try {
        const saved = await props.readJson(uri);
        if (current !== generation) return;
        if (saved.fingerprint !== document.value.snapshot_fingerprint) throw new Error("The report and saved snapshot identities differ.");
        snapshot.value = saved;
      } catch (cause) { if (current === generation) provenanceError.value = cause instanceof Error ? cause.message : String(cause); }
    }
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
function targets(block: Fields) { return targetIndex.value.get(block) ?? []; }
function toolExcerpt(target: InspectionTarget) {
  const text = JSON.stringify(savedToolPayload(target.detail ?? {}), null, 2);
  return text.length > 600 ? text.slice(0, 600) + "…" : text;
}
function link(uri: string, block: Fields) {
  const external = externalUri(uri);
  const target = targets(block).find((item) => item.id === uri || item.uri === uri || (external && item.originalUri === external));
  if (target) emit("inspect", target);
  else {
    if (external) emit("inspect", { id: external, title: external, kind: "source", originalUri: external });
    else provenanceError.value = "This link has no resolvable source in the saved report context.";
  }
}
async function download() {
  if (downloading.value) return;
  downloading.value = true;
  try {
    if (typeof opened.value.rendered_markdown !== "string") throw new Error("The host did not provide a rendered Markdown export for this revision.");
    const blob = new Blob([opened.value.rendered_markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = window.document.createElement("a");
    anchor.href = url;
    anchor.download = `${String(opened.value.title || "document").replace(/[/\\\u0000-\u001f]/g, "_")}.md`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }
  catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { downloading.value = false; }
}
watch(() => [props.uri, props.sessionId], () => { void load(); }, { immediate: true });
onBeforeUnmount(() => { generation++; });
</script>
<template>
  <article class="report-card" aria-label="Saved document reading view">
    <header class="report-heading"><div><span class="eyebrow">Saved {{ opened.kind || 'report' }} · revision {{ opened.revision || '…' }}</span><h3>{{ opened.title || name }}</h3></div><div class="report-actions"><button type="button" :disabled="busy || downloading || !opened.document_uri" @click="download">Download</button><button type="button" :disabled="busy || !editable || !opened.document_uri" @click="emit('edit', uri)">Edit</button></div></header>
    <p v-if="busy" role="status">Opening saved document…</p>
    <p v-if="error" class="error" role="alert">{{ error }} <button type="button" @click="load">Retry</button></p>
    <p v-if="document.kind === 'report' && document.coverage_complete === false" class="report-warning">Incomplete coverage. {{ strings(document.unresolved_gaps).join(' · ') }}</p>
    <p v-if="provenanceError" class="muted">Saved source detail unavailable: {{ provenanceError }}</p>
    <section v-for="section in sections" :key="String(section.id)" class="report-section">
      <h4 v-if="document.kind === 'report'">{{ section.title }}</h4>
      <p v-if="section.no_findings" class="muted">No findings were produced in this section.</p>
      <div v-for="block in objects(section.blocks)" :key="String(block.id)" class="report-block" tabindex="0">
        <MarkdownView :text="String(block.text ?? '')" @link="(uri) => link(uri, block)" />
        <div class="report-block-meta"><span class="support-state">{{ block.support || 'unverified' }} · {{ block.review_state || 'unreviewed' }}</span><button v-for="target in targets(block)" :key="target.id" type="button" class="citation-button" :title="`Inspect saved ${target.kind}: ${target.title}`" @click="emit('inspect', target)">{{ target.kind === 'tool' ? 'Tool: ' : '' }}{{ target.title }}</button><span v-if="!targets(block).length && strings(block.evidence_ids).length" class="muted">Evidence references retained; no saved source mapping available.</span></div>
        <div v-if="targets(block).length" class="provenance-peek" role="note"><strong>Saved supporting material</strong><div v-for="target in targets(block)" :key="target.id"><span>{{ target.kind }} · {{ target.title }}</span><pre v-if="target.kind === 'tool'">{{ toolExcerpt(target) }}</pre></div><small>Choose a source or tool above to inspect it. A linked reference is not itself factual verification.</small></div>
      </div>
    </section>
    <footer v-if="sourceTargets(document).length" class="report-sources"><h4>Sources</h4><ul><li v-for="source in sourceTargets(document)" :key="source.id"><button type="button" class="citation-button" @click="emit('inspect', source)">{{ source.title }}</button><span v-if="source.originalUri" class="muted"> {{ source.originalUri }}</span></li></ul></footer>
  </article>
</template>
