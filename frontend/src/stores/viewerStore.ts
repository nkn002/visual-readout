import { create } from "zustand";
import type { ConditionKey } from "../types/api";

export type Tool = "select" | "box";

interface ViewerState {
  selectedId: string | null;
  condition: ConditionKey;
  tool: Tool;
  showRegion: boolean;
  hiddenArms: string[];
  draftLabel: string;
  settingsOpen: boolean;
  select(id: string | null): void;
  setCondition(c: ConditionKey): void;
  setTool(t: Tool): void;
  toggleRegion(): void;
  toggleArm(key: string): void;
  setDraftLabel(s: string): void;
  setSettingsOpen(open: boolean): void;
}

export const useViewer = create<ViewerState>()((set) => ({
  selectedId: null,
  condition: "orig",
  tool: "select",
  showRegion: true,
  hiddenArms: [],
  draftLabel: "",
  settingsOpen: false,
  select: (selectedId) => set({ selectedId }),
  setCondition: (condition) => set({ condition }),
  setTool: (tool) => set({ tool }),
  toggleRegion: () => set((s) => ({ showRegion: !s.showRegion })),
  toggleArm: (key) =>
    set((s) => ({ hiddenArms: s.hiddenArms.includes(key) ? s.hiddenArms.filter((k) => k !== key) : [...s.hiddenArms, key] })),
  setDraftLabel: (draftLabel) => set({ draftLabel }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
}));
