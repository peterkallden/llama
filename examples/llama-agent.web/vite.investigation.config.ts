import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, "src/mcp-app/investigation-workspace.ts"),
      formats: ["iife"],
      name: "LlamaAgentInvestigationWorkspace",
      fileName: () => "investigation-workspace.js",
      cssFileName: "investigation-workspace.css",
    },
    rollupOptions: { output: { inlineDynamicImports: true } },
    minify: "esbuild",
  },
});
