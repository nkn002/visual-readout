import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Reads VITE_* from the repo-root .env and from the environment (CI sets VITE_BASE).
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, path.resolve(import.meta.dirname, ".."), "VITE_"), ...process.env };
  return {
    base: env.VITE_BASE ?? "/",
    plugins: [react()],
    server: {
      host: true,
      proxy: { "/api": { target: env.VITE_API_TARGET ?? "http://localhost:8000", changeOrigin: true } },
    },
  };
});
