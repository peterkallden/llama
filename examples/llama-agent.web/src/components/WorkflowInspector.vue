<script setup lang="ts">
import { computed } from "vue";
import type { AgentEvent } from "../types/agent";
import { classifyWorkflowEvent } from "./workflow-event";

const props = defineProps<{ events: AgentEvent[] }>();
const rows = computed(() => props.events.map((event) => ({ event, ...classifyWorkflowEvent(event) })));
</script>

<template>
  <section class="card workflow-card">
    <div class="card-title"><h2>Workflow Inspector</h2><span class="muted">{{ rows.length }} events</span></div>
    <p class="muted">A structured view of events already received from the daemon. “Observed” is not a verification claim.</p>
    <div v-if="!rows.length" class="empty">Workflow and tool events will appear here.</div>
    <ol v-else class="workflow-list">
      <li v-for="({ event, type, text, phase, execution, verdict }, index) in rows" :key="`${event.sequence ?? index}-${index}`" :class="`workflow-${execution}`">
        <span class="workflow-state">{{ execution }}</span><div><strong>{{ phase }} · {{ type }}<template v-if="verdict"> · verdict {{ verdict }}</template></strong><p>{{ text || "No detail in event payload." }}</p>
          <details><summary>Raw event</summary><pre>{{ JSON.stringify(event, null, 2) }}</pre></details>
        </div>
      </li>
    </ol>
  </section>
</template>
