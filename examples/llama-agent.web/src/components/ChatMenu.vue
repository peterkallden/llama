<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useAgentSession } from "../session";
import type { WebChat } from "../api/agent-api";

const session = useAgentSession();
const props = defineProps<{ locked?: boolean }>();
const open = ref(false);
const selectedProject = ref("");
const creatingProject = ref(false);
const projectName = ref("");
const busy = ref(false);
const error = ref("");
const selectedProjectName = computed(() => session.projects.find((item) => item.id === selectedProject.value)?.name ?? "Projects");

onMounted(() => { void session.refreshNavigation("").catch((cause) => { error.value = cause instanceof Error ? cause.message : String(cause); }); });
watch(open, (value) => {
  if (!value) return;
  if (!selectedProject.value && session.projectId) selectedProject.value = session.projectId;
  void session.refreshNavigation(selectedProject.value).catch((cause) => { error.value = cause instanceof Error ? cause.message : String(cause); });
});

async function selectProject(id: string) {
  selectedProject.value = id;
  error.value = "";
  try { await session.refreshNavigation(id); }
  catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
}

async function newChat() {
  busy.value = true; error.value = "";
  session.error = "";
  try {
    await session.newSession(selectedProject.value);
    if (!session.error) open.value = false;
  } catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { busy.value = false; }
}

async function openChat(chat: WebChat) {
  busy.value = true; error.value = "";
  try { await session.activateChat(chat); selectedProject.value = chat.project_id; await session.refreshNavigation(chat.project_id); open.value = false; }
  catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { busy.value = false; }
}

async function addProject() {
  const name = projectName.value.trim();
  if (!name) return;
  busy.value = true; error.value = "";
  try {
    const project = await session.createProject(name);
    projectName.value = ""; creatingProject.value = false;
    await selectProject(project.id);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : String(cause); }
  finally { busy.value = false; }
}

function title(chat: WebChat) { return chat.title || "New chat"; }
</script>

<template>
  <div class="chat-menu-root">
    <button type="button" class="chat-menu-trigger" aria-haspopup="true" :aria-expanded="open" :disabled="props.locked" @click="open = !open">
      <span>Chat</span><svg aria-hidden="true" viewBox="0 0 16 16"><path d="m3 6 5 5 5-5" /></svg>
    </button>
    <section v-if="open" class="chat-menu-panel" aria-label="Chat navigation">
      <template v-if="selectedProject">
        <div class="chat-menu-heading"><button type="button" class="chat-menu-back" aria-label="All projects" @click="selectProject('')">←</button><strong>{{ selectedProjectName }}</strong></div>
        <button type="button" class="chat-menu-action" :disabled="busy" @click="newChat">＋ New chat</button>
        <div class="chat-menu-list-title">Project chats</div>
      </template>
      <template v-else>
        <button type="button" class="chat-menu-action" :disabled="busy" @click="newChat">＋ New chat</button>
        <div class="chat-menu-list-title">Projects</div>
        <div class="chat-menu-scroll projects-list">
          <button v-for="project in session.projects" :key="project.id" type="button" class="chat-menu-row" @click="selectProject(project.id)">{{ project.name }}</button>
          <div v-if="!session.projects.length" class="chat-menu-empty">No projects yet.</div>
        </div>
        <form v-if="creatingProject" class="new-project-form" @submit.prevent="addProject">
          <input v-model="projectName" autofocus maxlength="120" aria-label="Project name" placeholder="Project name" />
          <button type="submit" :disabled="busy || !projectName.trim()">Create</button>
        </form>
        <button v-else type="button" class="chat-menu-action secondary" :disabled="busy" @click="creatingProject = true">＋ New project</button>
        <div class="chat-menu-list-title">Recent chats</div>
      </template>
      <div class="chat-menu-scroll chats-list">
        <div v-for="chat in session.chats" :key="chat.id" class="chat-row-container">
          <button type="button" class="chat-menu-row chat-row" :class="{ current: chat.id === session.sessionId }" :disabled="busy" @click="openChat(chat)">
            <span class="chat-row-indicator" :class="{ working: session.runningChats[chat.id] }" :aria-label="session.runningChats[chat.id] ? 'Working' : undefined" />
            <span class="chat-row-title">{{ title(chat) }}</span>
            <span v-if="chat.id === session.sessionId" class="chat-row-current">Open</span>
          </button>
          <button v-if="session.runningChats[chat.id]" type="button" class="chat-row-cancel" aria-label="Cancel this chat turn" title="Cancel this chat turn" @click="session.cancelChat(chat.id)">×</button>
        </div>
        <div v-if="!session.chats.length" class="chat-menu-empty">No chats in this view.</div>
      </div>
      <button v-if="session.chatCursor" type="button" class="chat-menu-action secondary" :disabled="busy || session.navigationLoading" @click="session.loadMoreChats(selectedProject)">Load more chats</button>
      <p v-if="session.navigationLoading || busy" class="chat-menu-status">{{ busy ? "Opening chat…" : "Loading…" }}</p>
      <p v-if="error || session.error" class="chat-menu-error">{{ error || session.error }}</p>
    </section>
  </div>
</template>

<style scoped>
.chat-menu-root{position:relative}.chat-menu-trigger{display:inline-flex;align-items:center;gap:.55rem}.chat-menu-trigger svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:1.7}.chat-menu-panel{position:absolute;z-index:30;top:calc(100% + 8px);left:0;width:min(340px,calc(100vw - 36px));padding:12px;background:#111d2d;border:1px solid #2b405a;border-radius:12px;box-shadow:0 16px 42px #0008;color:#e5edf7}.chat-menu-heading{display:flex;align-items:center;gap:8px;padding:2px 4px 10px}.chat-menu-back{border:0;background:transparent;color:#83d5f7;font-size:1.15rem;padding:3px 8px}.chat-menu-action,.chat-menu-row{width:100%;text-align:left;border:0;border-radius:7px;background:transparent;color:inherit;padding:9px 10px}.chat-menu-action{background:#1a2a40;color:#89d9f7;font-weight:650;margin:2px 0}.chat-menu-action.secondary{background:transparent;color:#afc4dc}.chat-menu-action:hover,.chat-menu-row:hover{background:#20334c}.chat-menu-list-title{margin:12px 8px 5px;color:#90a7c1;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}.chat-menu-scroll{overflow-y:auto;overscroll-behavior:contain}.projects-list{max-height:155px}.chats-list{max-height:min(38vh,350px)}.chat-menu-row{display:flex;align-items:center;gap:9px;min-height:36px}.chat-menu-row.current{background:#183149}.chat-row-container{display:flex;align-items:center}.chat-row-container>.chat-row{flex:1;min-width:0}.chat-row-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}.chat-row-indicator{width:8px;height:8px;border-radius:50%;background:#52647b;flex:none}.chat-row-indicator.working{background:#7dd3fc;box-shadow:0 0 0 3px #7dd3fc25;animation:pulse 1.4s ease-in-out infinite}.chat-row-current{font-size:.7rem;color:#83d5f7}.chat-row-cancel{flex:none;border:0;background:#2a3d54;color:#d9e8f7;border-radius:5px;width:24px;height:24px}.chat-menu-empty,.chat-menu-status{padding:7px 9px;color:#91a5bd;font-size:.82rem}.chat-menu-error{padding:7px 9px;color:#ff9aa6;font-size:.82rem}.new-project-form{display:flex;gap:6px;margin:5px 2px}.new-project-form input{min-width:0;flex:1;background:#0c1624;color:#e5edf7;border:1px solid #354a63;border-radius:6px;padding:8px}.new-project-form button{background:#7dd3fc;color:#07111c;border:0;border-radius:6px;padding:0 10px}@keyframes pulse{50%{opacity:.5}}
</style>
