<script setup lang="ts">
import RelationshipGraph from "./RelationshipGraph.vue";
import type { RelationshipSelection } from "../visualization/relationship-renderer";
import type { RelationshipVisualization } from "../visualization/contract";

defineProps<{ visualization: RelationshipVisualization }>();
const emit = defineEmits<{ select: [selection: RelationshipSelection] }>();
</script>

<template>
  <RelationshipGraph :visualization="visualization"
    description="Relations show saved evidence references and their declared assessment; visual proximity does not imply causation."
    @select="emit('select', $event)" />
</template>
