<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useAgentSession } from "./session";

const session = useAgentSession();
const prompt = ref("");
const expanded = ref<Set<number>>(new Set());

const connectionLabel = {
  connecting: "connecting",
  connected: "connected",
  disconnected: "disconnected",
  error: "connection failed",
  "auth-error": "authentication failed",
} as const;

onMounted(() => { void session.connect(); void session.refreshStatus().catch(() => undefined); });

function send() {
  const value = prompt.value.trim();
  if ((!value && !session.attachments.length) || session.busy) return;
  prompt.value = "";
  void session.submit(value || "Please process the attached audio.");
}

function attachFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  for (const file of Array.from(input.files ?? [])) void session.upload(file);
  input.value = "";
}

function toggle(index: number) {
  const next = new Set(expanded.value);
  next.has(index) ? next.delete(index) : next.add(index);
  expanded.value = next;
}

function eventText(event: Record<string, unknown>) {
  return String(event.detail ?? event.error ?? event.tool_name ?? "");
}

function isArtifactEvent(event: Record<string, unknown>) {
  return session.eventType(event) === "tool.artifact_created" && Boolean(event.detail || event.resource_uri);
}
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div><p class="eyebrow">EXAMPLE APPLICATION</p><h1>llama-agent</h1></div>
      <div class="connection" :class="`connection-${session.connectionState}`">
        <span class="dot" /> {{ connectionLabel[session.connectionState] }}
        <button class="quiet" @click="session.connect">reconnect</button><button class="quiet" @click="session.newSession">new session</button>
      </div>
    </header>

    <section class="notice">This is an example client for the daemon's HTTP/SSE adapter. Agent planning, tools, MCP and policy remain in the daemon.</section>

    <section class="grid">
      <div class="primary">
        <section class="card chat">
          <div class="card-title"><h2>Chat</h2><button v-if="session.busy" class="cancel-button" @click="session.cancel">Cancel</button></div>
          <div v-if="!session.messages.length" class="empty">Send a question to the connected agent.</div>
          <div v-for="message in session.messages" :key="message.id" class="message" :class="message.role"><span class="message-role">{{ message.role === "user" ? "You" : "Agent" }}</span><div class="message-body">{{ message.content || (message.pending ? "working…" : "") }}</div></div>
          <p v-if="session.error" class="error">{{ session.error }}</p>
          <form class="composer" @submit.prevent="send">
            <textarea v-model="prompt" rows="3" placeholder="Write a message…" @keydown.ctrl.enter="send" />
            <div class="composer-actions"><div class="input-actions"><label class="file-button">Attach files<input type="file" multiple @change="attachFiles" /></label><button type="button" class="mic-button" :class="{ recording: session.recording }" :disabled="session.busy" :title="session.recording ? 'Stop recording' : 'Record audio'" @click="session.toggleRecording">{{ session.recording ? "Stop recording" : "Record audio" }}</button><span v-if="session.recording" class="recording-status"><span class="recording-dot" /> recording…</span></div><button class="primary-button" :disabled="session.busy || (!prompt.trim() && !session.attachments.length)">Send</button></div>
          </form>
          <div v-if="session.attachments.length" class="attachments"><strong>Attachments</strong><label v-for="(file, index) in session.attachments" :key="file.name" class="attachment"><input type="checkbox" checked @change="session.removeAttachment(index)" />{{ file.name }}</label></div>
        </section>

        <section class="card">
          <div class="card-title"><h2>Events</h2><span class="muted">{{ session.events.length }}</span></div>
          <div v-if="!session.events.length" class="empty">Runtime events will appear here while the daemon is working.</div>
          <div v-for="(event, index) in session.events" :key="`${event.sequence ?? index}-${index}`" class="event-row">
            <button class="expand" @click="toggle(index)">{{ expanded.has(index) ? "−" : "+" }}</button>
            <div class="event-main"><div><strong>{{ session.eventType(event) }}</strong><span v-if="event.tool_name" class="tag">{{ event.tool_name }}</span><button v-if="isArtifactEvent(event)" class="download-button" @click="session.downloadArtifact(event)">Download</button></div><p>{{ eventText(event) }}</p></div>
            <pre v-if="expanded.has(index)">{{ JSON.stringify(event, null, 2) }}</pre>
          </div>
        </section>
      </div>

      <aside class="side">
        <section class="card"><div class="card-title"><h2>Status and capabilities</h2><button class="quiet" @click="session.refreshStatus">refresh</button></div><dl><template v-for="(value, key) in session.status" :key="String(key)"><dt>{{ key }}</dt><dd>{{ typeof value === 'object' ? JSON.stringify(value) : value }}</dd></template></dl><div v-if="session.capabilities.length" class="capabilities"><strong>Capabilities</strong><span v-for="capability in session.capabilities" :key="capability" class="tag">{{ capability }}</span></div></section>
        <section class="card"><div class="card-title"><h2>Web client</h2></div><p class="muted">HTTP commands go to the daemon. SSE is used only for the server event stream.</p><label class="field token-field" :class="{ 'token-missing': !session.token.trim(), 'token-unconnected': session.token.trim() && !session.connected }">Bearer token<input v-model="session.token" type="password" autocomplete="off" placeholder="enter token" :aria-invalid="!session.token.trim() || undefined" /><small v-if="!session.token.trim()">No token entered. An authenticated daemon may reject requests.</small><small v-else-if="session.connectionState === 'auth-error'">The token was rejected. Check it and reconnect.</small><small v-else-if="!session.connected">Token entered, but the event connection is not established yet. Try reconnect.</small><small v-else>Token accepted and event connection is active.</small></label><p v-if="session.connectionError" class="connection-error">{{ session.connectionError }}</p></section>
      </aside>
    </section>
  </main>
</template>
