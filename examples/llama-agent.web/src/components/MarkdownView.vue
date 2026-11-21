<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import Viewer from "@toast-ui/editor/viewer";
import DOMPurify from "dompurify";
import "@toast-ui/editor/dist/toastui-editor-viewer.css";
import "@toast-ui/editor/dist/theme/toastui-editor-dark.css";
const props = defineProps<{ text: string }>();
const emit = defineEmits<{ link: [uri: string] }>();
const mount = ref<HTMLElement | null>(null);
let viewer: Viewer | undefined;
let observer: IntersectionObserver | undefined;
function click(event: MouseEvent) {
  const link = event.target instanceof Element ? event.target.closest("a") : null;
  if (!link) return;
  if ((link.getAttribute("href") ?? "").startsWith("#")) return;
  event.preventDefault();
  emit("link", link.getAttribute("href") ?? "");
}
function createViewer() {
  if (viewer) return;
  if (mount.value) viewer = new Viewer({ el: mount.value, initialValue: props.text, theme: "dark",
    usageStatistics: false, linkAttributes: { target: "_blank", rel: "noopener noreferrer" },
    customHTMLSanitizer: (html) => DOMPurify.sanitize(html, {
      FORBID_TAGS: ["img", "svg", "math", "iframe", "object", "embed", "video", "audio", "source", "style", "link", "form", "input"],
      FORBID_ATTR: ["style", "src", "srcset"],
      ALLOWED_URI_REGEXP: /^(?:(?:https?|agent-resource|resource|dataset):\/\/|#|[A-Za-z0-9_.-]+$)/i,
    }),
  });
}
onMounted(() => {
  if (!mount.value || typeof IntersectionObserver === "undefined") { createViewer(); return; }
  observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) { createViewer(); observer?.disconnect(); }
  }, { rootMargin: "300px" });
  observer.observe(mount.value);
});
watch(() => props.text, (text) => viewer?.setMarkdown(text));
onBeforeUnmount(() => { observer?.disconnect(); viewer?.destroy(); });
</script>
<template><div ref="mount" class="markdown-view" @click="click" /></template>
