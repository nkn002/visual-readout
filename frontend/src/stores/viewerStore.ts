import { create } from "zustand";

export type Tool = "select" | "box";

interface ViewerState {
  selectedId: string | null;
  tool: Tool;
  showRegion: boolean; // the dataset's own region box, revealed by the Default region button
  draftLabel: string;
  settingsOpen: boolean;
  select(id: string | null): void;
  setTool(t: Tool): void;
  toggleRegion(): void;
  setShowRegion(on: boolean): void;
  setDraftLabel(s: string): void;
  setSettingsOpen(open: boolean): void;
}

export const useViewer = create<ViewerState>()((set) => ({
  selectedId: null,
  tool: "box",
  showRegion: false,
  draftLabel: "",
  settingsOpen: false,
  select: (selectedId) => set({ selectedId }),
  setTool: (tool) => set({ tool }),
  toggleRegion: () => set((s) => ({ showRegion: !s.showRegion })),
  setShowRegion: (showRegion) => set({ showRegion }),
  setDraftLabel: (draftLabel) => set({ draftLabel }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
}));
