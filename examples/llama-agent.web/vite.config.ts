import { defineConfig, loadEnv, type Plugin } from "vite";
import vue from "@vitejs/plugin-vue";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// Use the notices already shipped with the editor for the reading-view bundle.
function thirdPartyNotices() {
  const packages = ["dompurify", "vue", "prosemirror-commands", "prosemirror-history", "prosemirror-inputrules", "prosemirror-keymap", "prosemirror-model", "prosemirror-state", "prosemirror-transform", "prosemirror-view", "orderedmap", "rope-sequence", "w3c-keyname"];
  const files = ["../../docs/third-party/NOTICE.md", "../../docs/third-party/TOAST-UI-MIT.txt",
    ...packages.map((name) => `node_modules/${name}/LICENSE`)];
  return files.map((file) => `${file}\n\n${readFileSync(resolve(import.meta.dirname, file), "utf8")}`).join("\n\n---\n\n");
}

const noticesPlugin: Plugin = {
  name: "llama-agent-third-party-notices",
  generateBundle() { this.emitFile({ type: "asset", fileName: "third-party-notices.txt", source: thirdPartyNotices() }); },
  configureServer(server) {
    server.middlewares.use("/third-party-notices.txt", (_request, response) => {
      response.setHeader("Content-Type", "text/plain; charset=utf-8");
      response.end(thirdPartyNotices());
    });
  },
};

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [vue(), noticesPlugin],
    server: {
      port: 5173,
      proxy: {
        "/api": env.VITE_AGENT_DEV_PROXY || "http://127.0.0.1:8090",
      },
    },
  };
});
