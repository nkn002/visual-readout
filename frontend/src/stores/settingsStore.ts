import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Mode = "static" | "api";

interface SettingsState {
  mode: Mode;
  apiBase: string;
  setMode(mode: Mode): void;
  setApiBase(base: string): void;
}

// In dev the Vite proxy forwards `${BASE_URL}api` to the backend, so the default stays same-origin.
// The built page on GitHub Pages talks to a backend on the viewer's own machine.
export const DEFAULT_API_BASE = import.meta.env.DEV ? import.meta.env.BASE_URL.replace(/\/$/, "") : "http://localhost:8000";

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      mode: "static",
      apiBase: DEFAULT_API_BASE,
      setMode: (mode) => set({ mode }),
      setApiBase: (base) => set({ apiBase: base.trim().replace(/\/+$/, "") }),
    }),
    {
      name: "visual-readout:settings",
      // A remembered relative address ("/rnode/<node>/<port>/absproxy/5173") belongs to a previous
      // dev session behind a different proxy; only an absolute URL the user typed is worth keeping.
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<SettingsState>;
        const keep = p.apiBase && /^https?:\/\//.test(p.apiBase) ? p.apiBase : DEFAULT_API_BASE;
        return { ...current, ...p, apiBase: keep };
      },
    },
  ),
);
