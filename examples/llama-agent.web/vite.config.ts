import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [vue()],
    server: {
      port: 5173,
      proxy: {
        "/api": env.VITE_AGENT_DEV_PROXY || "http://127.0.0.1:8090",
      },
    },
  };
});
