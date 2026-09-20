import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Reads VITE_* from the repo-root .env and from the environment (CI sets VITE_BASE).
// The dev proxy forwards `${base}api/*` to the backend, so the app works under a sub-path too
// (GitHub Pages uses /visual-readout/, code-server on Open OnDemand uses /absproxy/5173/).
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, path.resolve(import.meta.dirname, ".."), "VITE_"), ...process.env };
  const base = env.VITE_BASE ?? "/";
  const target = env.VITE_API_TARGET ?? "http://localhost:8000";
  const proxy = { [`${base}api`]: { target, changeOrigin: true, rewrite: (p: string) => p.slice(base.length - 1) } };
  return {
    base,
    plugins: [react()],
    server: { host: true, allowedHosts: true, proxy },
    preview: { host: true, allowedHosts: true, proxy },
  };
});
