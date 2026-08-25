<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useAgentSession } from "./session";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());

onMounted(() => { void session.connect(); void session.refreshStatus().catch(() => undefined); });

function send() {
  const value = prompt.value.trim();
  if (!value || session.busy) return;
  prompt.value = "";
  void session.submit(value);
}

function toggle(index: number) {
  const next = new Set(expanded.value);
  next.has(index) ? next.delete(index) : next.add(index);
  expanded.value = next;
}

function eventText(event: Record<string, unknown>) {
  return String(event.detail ?? event.error ?? event.tool_name ?? "");
}
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div><p class="eyebrow">EXAMPLE APPLICATION</p><h1>llama-agent</h1></div>
      <div class="connection" :class="{ online: session.connected }">
        <span class="dot" /> {{ session.connected ? "connected" : "disconnected" }}
        <button class="quiet" @click="session.connect">reconnect</button>
      </div>
    </header>

    <section class="notice">This is an example client for the daemon's HTTP/SSE adapter. Agent planning, tools, MCP and policy remain in the daemon.</section>

    <section class="grid">
      <div class="primary">
        <section class="card chat">
          <div class="card-title"><h2>Chat</h2><span v-if="session.busy" class="working">working…</span></div>
          <div v-if="session.responseText" class="assistant">{{ session.responseText }}</div>
          <div v-else class="empty">Send a question to the connected agent.</div>
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><label class="file-button">Attach text file<input type="file" @change="(e) => { const file = (e.target as HTMLInputElement).files?.[0]; if (file) void session.upload(file); }" /></label><button class="primary-button" :disabled="session.busy || !prompt.trim()">Send</button></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments</strong><label v-for="(file, index) in session.attachments" :key="file.name" class="attachment"><input type="checkbox" checked @change="session.removeAttachment(index)" />{{ file.name }}</label></div>
        </section>

        <section class="card">
          <div class="card-title"><h2>Händelser</h2><span class="muted">{{ session.events.length }}</span></div>
          <div v-if="!session.events.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div v-for="(event, index) in session.events" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div><strong>{{ session.eventType(event) }}</strong><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span></div><p>{{ eventText(event) }}</p></div>
            <pre v-if="expanded.has(index)">{{ JSON.stringify(event, null, 2) }}</pre>
          </div>
        </section>
      </div>

      <aside class="side">
        <section class="card"><div class="card-title"><h2>Status</h2><button class="quiet" @click="session.refreshStatus">refresh</button></div><dl><template v-for="(value, key) in session.status" :key="String(key)"><dt>{{ key }}</dt><dd>{{ typeof value === 'object' ? JSON.stringify(value) : value }}</dd></template></dl></section>
        <section class="card"><div class="card-title"><h2>Web client</h2></div><p class="muted">HTTP commands go to the daemon. SSE is used only for the server event stream.</p><label class="field">Bearer token<input v-model="session.token" type="password" autocomplete="off" placeholder="optional for local development" /></label></section>
      </aside>
    </section>
  </main>
</template>
