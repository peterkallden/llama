import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, "src/mcp-app/document-editor.ts"),
      formats: ["iife"],
      name: "LlamaAgentDocumentEditor",
      fileName: () => "document-editor.js",
      cssFileName: "document-editor.css",
    },
    rollupOptions: { output: { inlineDynamicImports: true } },
    minify: "esbuild",
  },
});
