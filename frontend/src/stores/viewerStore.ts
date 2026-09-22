import { create } from "zustand";

export type Tool = "select" | "box";

interface ViewerState {
  selectedId: string | null;
  tool: Tool;
  showRegion: boolean; // the dataset's own region box, revealed by the Default region button
  heatRegionId: string | null; // the drawn region whose activation heatmap is shown
  draftLabel: string;
  settingsOpen: boolean;
  select(id: string | null): void;
  setTool(t: Tool): void;
  toggleRegion(): void;
  setShowRegion(on: boolean): void;
  setHeatRegion(id: string | null): void;
  setDraftLabel(s: string): void;
  setSettingsOpen(open: boolean): void;
}

export const useViewer = create<ViewerState>()((set) => ({
  selectedId: null,
  tool: "box",
  showRegion: false,
  heatRegionId: null,
  draftLabel: "",
  settingsOpen: false,
  select: (selectedId) => set({ selectedId }),
  setTool: (tool) => set({ tool }),
  toggleRegion: () => set((s) => ({ showRegion: !s.showRegion })),
  setShowRegion: (showRegion) => set({ showRegion }),
  setHeatRegion: (heatRegionId) => set({ heatRegionId }),
  setDraftLabel: (draftLabel) => set({ draftLabel }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
}));
