import { create } from "zustand";

export type Tool = "select" | "box";

interface ViewerState {
  selectedId: string | null;
  tool: Tool;
  showRegion: boolean;
  draftLabel: string;
  settingsOpen: boolean;
  select(id: string | null): void;
  setTool(t: Tool): void;
  toggleRegion(): void;
  setDraftLabel(s: string): void;
  setSettingsOpen(open: boolean): void;
}

export const useViewer = create<ViewerState>()((set) => ({
  selectedId: null,
  tool: "box",
  showRegion: true,
  draftLabel: "",
  settingsOpen: false,
  select: (selectedId) => set({ selectedId }),
  setTool: (tool) => set({ tool }),
  toggleRegion: () => set((s) => ({ showRegion: !s.showRegion })),
  setDraftLabel: (draftLabel) => set({ draftLabel }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
}));
