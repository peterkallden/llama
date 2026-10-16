<script setup lang="ts">
import { computed } from "vue";
import type { AgentEvent } from "../types/agent";

const props = defineProps<{ events: AgentEvent[] }>();
const rows = computed(() => props.events.map((event) => {
  const type = String(event.event_type ?? event.type ?? "event");
  const text = String(event.detail ?? event.error ?? event.tool_name ?? "");
  const explicit = String(event.verification_status ?? event.verdict ?? event.status ?? "");
  const status = /failed|rejected|violated/i.test(type + " " + explicit) ? "failed"
    : /verified|satisfied|completed/i.test(type + " " + explicit) ? "completed"
      : "observed";
  const phase = String(event.phase ?? event.cognitive_mode ?? event.route_id ?? event.event_category ?? "runtime");
  return { event, type, text, phase, status };
}));
</script>

<template>
  <section class="card workflow-card">
    <div class="card-title"><h2>Workflow Inspector</h2><span class="muted">{{ rows.length }} events</span></div>
    <p class="muted">A structured view of events already received from the daemon. “Observed” is not a verification claim.</p>
    <div v-if="!rows.length" class="empty">Workflow and tool events will appear here.</div>
    <ol v-else class="workflow-list">
      <li v-for="({ event, type, text, phase, status }, index) in rows" :key="`${event.sequence ?? index}-${index}`" :class="`workflow-${status}`">
        <span class="workflow-state">{{ status }}</span><div><strong>{{ phase }} · {{ type }}</strong><p>{{ text || "No detail in event payload." }}</p>
          <details><summary>Raw event</summary><pre>{{ JSON.stringify(event, null, 2) }}</pre></details>
        </div>
      </li>
    </ol>
  </section>
</template>
