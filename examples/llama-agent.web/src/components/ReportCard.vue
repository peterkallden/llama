<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import MarkdownView from "./MarkdownView.vue";
import { blockTargets, externalUri, object, objects, savedToolPayload, sourceTargets, strings, type Fields, type InspectionTarget } from "./report-view";
import { structuredToolResult } from "./structured-tool-result";
import { loadReportSnapshots, type ReportSnapshot } from "./report-snapshots";
import ResearchEvidenceGraph from "./ResearchEvidenceGraph.vue";
import { parseRelationshipVisualization, type RelationshipVisualization } from "../visualization/contract";
import type { RelationshipSelection } from "../visualization/relationship-renderer";
const props = defineProps<{
  uri: string; name: string; sessionId: string; editable: boolean;
  openDocument: (uri: string) => Promise<unknown>;
  readJson: (uri: string) => Promise<Fields>;
  readBlob?: (uri: string) => Promise<Blob>;
  callTool?: (toolName: string, args: Record<string, unknown>) => Promise<unknown>;
}>();
const emit = defineEmits<{ edit: [uri: string]; inspect: [target: InspectionTarget] }>();
const opened = ref<Fields>({});
const snapshot = ref<Fields>({});
const error = ref("");
const provenanceError = ref("");
const busy = ref(false);
const downloading = ref(false);
const graph = ref<RelationshipVisualization>();
const graphBusy = ref(false);
const graphError = ref("");
const graphSelection = ref<RelationshipSelection>();
const figures = ref<(ReportSnapshot & { url?: string })[]>([]);
function clearFigures() {
  for (const figure of figures.value) if (figure.url) URL.revokeObjectURL(figure.url);
  figures.value = [];
}
const document = computed(() => object(opened.value.document));
const canShowEvidenceGraph = computed(() => Boolean(props.callTool &&
  typeof snapshot.value.research_workspace_id === "string" && snapshot.value.research_workspace_id));
const sections = computed(() => objects(document.value.sections));
const targetIndex = computed(() => new Map<Fields, InspectionTarget[]>(sections.value.flatMap((section) =>
  objects(section.blocks).map((block) => [block, blockTargets(block, document.value, snapshot.value)] as [Fields, InspectionTarget[]]))));
let generation = 0;
async function load() {
  const current = ++generation;
  clearFigures();
  busy.value = true; error.value = ""; provenanceError.value = ""; opened.value = {}; snapshot.value = {};
  graph.value = undefined; graphError.value = ""; graphSelection.value = undefined; graphBusy.value = false;
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
        if (saved.schema === "investigation-report-source.v1") {
          if (!props.readBlob) throw new Error("Snapshot image reading is unavailable.");
          const assets = await loadReportSnapshots(document.value, props.readBlob);
          if (current !== generation) return;
          figures.value = assets.map((asset) => ({ ...asset, url: asset.blob ? URL.createObjectURL(asset.blob) : undefined }));
        } else {
          if (saved.fingerprint !== document.value.snapshot_fingerprint) throw new Error("The report and saved snapshot identities differ.");
          snapshot.value = saved;
        }
      } catch (cause) { if (current === generation) provenanceError.value = cause instanceof Error ? cause.message : String(cause); }
    }
  } catch (cause) { if (current === generation) error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) busy.value = false; }
}
async function showEvidenceGraph(block?: Fields) {
  if (!props.callTool || !canShowEvidenceGraph.value || !props.uri || graphBusy.value) return;
  const current = generation;
  graphBusy.value = true; graphError.value = ""; graphSelection.value = undefined;
  try {
    const args: Record<string, unknown> = { source_uri: props.uri };
    if (block && typeof block.id === "string") args.block_ids = [block.id];
    const result = structuredToolResult(await props.callTool("visualization.research_graph", args));
    if (current !== generation) return;
    graph.value = parseRelationshipVisualization(result.visualization);
  } catch (cause) { if (current === generation) graphError.value = cause instanceof Error ? cause.message : String(cause); }
  finally { if (current === generation) graphBusy.value = false; }
}
function targets(block: Fields) { return targetIndex.value.get(block) ?? []; }
function toolExcerpt(target: InspectionTarget) {
  const text = JSON.stringify(savedToolPayload(target.detail ?? {}), null, 2);
  return text.length > 600 ? text.slice(0, 600) + "…" : text;
}
function link(uri: string, block: Fields) {
  const figure = figures.value.find((item) => item.uri === uri || item.sourceUri === uri);
  if (figure) {
    emit("inspect", { id: uri, uri, title: uri === figure.uri ? figure.caption : "Saved visualization projection",
      kind: "resource", mimeType: uri === figure.uri ? "image/png" : "application/vnd.llama-agent.visualization+json" });
    return;
  }
  const external = externalUri(uri);
  const target = targets(block).find((item) => item.id === uri || item.uri === uri || (external && item.originalUri === external));
  if (target) emit("inspect", target);
  else {
    if (external) emit("inspect", { id: external, title: external, kind: "source", originalUri: external });
    else provenanceError.value = "This link has no resolvable source in the saved report context.";
  }
}
function imageFailed(figure: ReportSnapshot & { url?: string }) {
  if (figure.url) URL.revokeObjectURL(figure.url);
  figure.url = undefined;
  figure.error = "The saved PNG could not be decoded by the browser.";
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
onBeforeUnmount(() => { generation++; clearFigures(); });
</script>
<template>
  <article class="report-card" aria-label="Saved document reading view">
    <header class="report-heading"><div><span class="eyebrow">Saved {{ opened.kind || 'report' }} · revision {{ opened.revision || '…' }}</span><h3>{{ opened.title || name }}</h3></div><div class="report-actions"><button v-if="canShowEvidenceGraph" type="button" :disabled="busy || graphBusy" @click="showEvidenceGraph()">{{ graphBusy ? 'Building graph…' : 'Evidence graph' }}</button><button type="button" :disabled="busy || downloading || !opened.document_uri" @click="download">Download</button><button type="button" :disabled="busy || !editable || !opened.document_uri" @click="emit('edit', uri)">Edit</button></div></header>
    <p v-if="busy" role="status">Opening saved document…</p>
    <p v-if="error" class="error" role="alert">{{ error }} <button type="button" @click="load">Retry</button></p>
    <p v-if="document.kind === 'report' && document.coverage_complete === false" class="report-warning">Incomplete coverage. {{ strings(document.unresolved_gaps).join(' · ') }}</p>
    <p v-if="provenanceError" class="muted">Saved source detail unavailable: {{ provenanceError }}</p>
    <section v-for="section in sections" :key="String(section.id)" class="report-section">
      <h4 v-if="document.kind === 'report'">{{ section.title }}</h4>
      <p v-if="section.no_findings" class="muted">No findings were produced in this section.</p>
      <div v-for="block in objects(section.blocks)" :key="String(block.id)" class="report-block" tabindex="0">
        <MarkdownView :text="String(block.text ?? '')" @link="(uri) => link(uri, block)" />
        <div class="report-block-meta"><span class="support-state">{{ block.support || 'unverified' }} · {{ block.review_state || 'unreviewed' }}</span><button v-for="target in targets(block)" :key="target.id" type="button" class="citation-button" :title="`Inspect saved ${target.kind}: ${target.title}`" @click="emit('inspect', target)">{{ target.kind === 'tool' ? 'Tool: ' : '' }}{{ target.title }}</button><button v-if="canShowEvidenceGraph" type="button" class="citation-button" :disabled="graphBusy" @click="showEvidenceGraph(block)">Trace evidence</button><span v-if="!targets(block).length && strings(block.evidence_ids).length" class="muted">Evidence references retained; no saved source mapping available.</span></div>
        <div v-if="targets(block).length" class="provenance-peek" role="note"><strong>Saved supporting material</strong><div v-for="target in targets(block)" :key="target.id"><span>{{ target.kind }} · {{ target.title }}</span><pre v-if="target.kind === 'tool'">{{ toolExcerpt(target) }}</pre></div><small>Choose a source or tool above to inspect it. A linked reference is not itself factual verification.</small></div>
      </div>
    </section>
    <p v-if="graphError" class="error" role="alert">Evidence graph unavailable: {{ graphError }}</p>
    <p v-if="graphSelection" class="muted" role="status">Selected {{ graphSelection.kind }} {{ graphSelection.id }}</p>
    <ResearchEvidenceGraph v-if="graph" :visualization="graph" @select="graphSelection = $event" />
    <section v-if="figures.length" class="report-snapshots" aria-label="Saved visualization snapshots">
      <h4>Visualization snapshots</h4>
      <figure v-for="figure in figures" :key="figure.uri">
        <img v-if="figure.url" :src="figure.url" :alt="figure.caption" loading="lazy" @error="imageFailed(figure)" />
        <p v-else class="error" role="status">Snapshot unavailable: {{ figure.error }}</p>
        <figcaption>{{ figure.caption }}
          <button type="button" class="citation-button" @click="emit('inspect', { id: figure.uri, uri: figure.uri, title: figure.caption, kind: 'resource', mimeType: 'image/png' })">Inspect snapshot</button>
          <button type="button" class="citation-button" @click="emit('inspect', { id: figure.sourceUri, uri: figure.sourceUri, title: 'Saved visualization projection', kind: 'resource', mimeType: 'application/vnd.llama-agent.visualization+json' })">Inspect visualization</button>
        </figcaption>
      </figure>
    </section>
    <footer v-if="sourceTargets(document).length" class="report-sources"><h4>Sources</h4><ul><li v-for="source in sourceTargets(document)" :key="source.id"><button type="button" class="citation-button" @click="emit('inspect', source)">{{ source.title }}</button><span v-if="source.originalUri" class="muted"> {{ source.originalUri }}</span></li></ul></footer>
  </article>
</template>
