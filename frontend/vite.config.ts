import { defineConfig, loadEnv } from "vite";
import type { Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Reads VITE_* from the repo-root .env and from the environment (CI sets VITE_BASE).
//
// VITE_BASE          the path the browser sees, e.g. /visual-readout/ on GitHub Pages
// VITE_SERVER_PREFIX the path requests carry when they reach this server, when a reverse proxy
//                    strips part of the public prefix (code-server on Open OnDemand: the browser
//                    uses /rnode/<host>/<port>/absproxy/5173/, the server receives /absproxy/5173/).
//                    Defaults to VITE_BASE, i.e. no rewriting.
// VITE_API_TARGET    where `${VITE_BASE}api/*` is proxied in dev (default http://localhost:8000)
export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, path.resolve(import.meta.dirname, ".."), "VITE_"), ...process.env };
  const base = env.VITE_BASE ?? "/";
  const serverPrefix = env.VITE_SERVER_PREFIX ?? base;
  const target = env.VITE_API_TARGET ?? "http://localhost:8000";
  const proxy = { [`${base}api`]: { target, changeOrigin: true, rewrite: (p: string) => p.slice(base.length - 1) } };

  const mapPrefix = (req: { url?: string }) => {
    if (!req.url) return;
    if (req.url === serverPrefix.slice(0, -1)) req.url = base;
    else if (req.url.startsWith(serverPrefix)) req.url = base + req.url.slice(serverPrefix.length);
  };
  const stripProxyPrefix: Plugin = {
    name: "map-stripped-proxy-prefix",
    configureServer(server) {
      if (serverPrefix === base) return;
      server.middlewares.use((req, _res, next) => (mapPrefix(req), next()));
      // The HMR websocket upgrade arrives with the stripped path too; map it before Vite checks it.
      server.httpServer?.prependListener("upgrade", (req) => mapPrefix(req));
    },
    configurePreviewServer(server) {
      if (serverPrefix !== base) server.middlewares.use((req, _res, next) => (mapPrefix(req), next()));
    },
  };

  return {
    base,
    plugins: [react(), stripProxyPrefix],
    server: { host: true, allowedHosts: true, proxy },
    preview: { host: true, allowedHosts: true, proxy },
  };
});
