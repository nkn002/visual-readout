import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Mode = "static" | "api";

interface SettingsState {
  mode: Mode;
  apiBase: string;
  setMode(mode: Mode): void;
  setApiBase(base: string): void;
}

// In dev the Vite proxy forwards /api to the backend, so an empty base means "same origin".
export const DEFAULT_API_BASE = import.meta.env.DEV ? "" : "http://localhost:8000";

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      mode: "static",
      apiBase: DEFAULT_API_BASE,
      setMode: (mode) => set({ mode }),
      setApiBase: (base) => set({ apiBase: base.trim().replace(/\/+$/, "") }),
    }),
    { name: "visual-readout:settings" },
  ),
);
