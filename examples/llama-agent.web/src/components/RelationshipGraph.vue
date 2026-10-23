<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { RelationshipRenderer, type RelationshipSelection } from "../visualization/relationship-renderer";
import type { RelationshipVisualization } from "../visualization/contract";

const props = defineProps<{ visualization: RelationshipVisualization; description?: string }>();
const emit = defineEmits<{ select: [selection: RelationshipSelection] }>();
const container = ref<HTMLDivElement>();
const selection = ref<RelationshipSelection>();
let renderer: RelationshipRenderer | undefined;
const selectedRelation = computed(() => selection.value?.kind === "edge"
  ? props.visualization.edges.find((edge) => edge.id === selection.value?.id) : undefined);

onMounted(() => {
  if (!container.value) return;
  renderer = new RelationshipRenderer(container.value, { onSelect: (value) => { selection.value = value; emit("select", value); } });
  renderer.update(props.visualization, { replace: true });
});
watch(() => props.visualization, (value) => { selection.value = undefined; renderer?.update(value, { replace: true }); }, { deep: true });
onBeforeUnmount(() => { renderer?.dispose(); renderer = undefined; });
</script>

<template>
  <section class="relationship-graph" :aria-label="visualization.title">
    <div ref="container" class="relationship-graph-canvas" />
    <div v-if="selectedRelation" class="relationship-selection" role="status">
      <strong>{{ selectedRelation.relation || 'relation' }}</strong><span>{{ selectedRelation.label }}</span>
      <small v-if="selectedRelation.provenance_refs?.length">Source references: {{ selectedRelation.provenance_refs.join(', ') }}</small>
      <small v-else>No source references are attached to this relation.</small>
    </div>
    <small v-if="description" class="muted">{{ description }}</small>
  </section>
</template>
