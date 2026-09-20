import type { ComponentType } from "react";
import type { ViewerProps } from "./types";
import { RegionViewer } from "./RegionViewer";

export const viewers: Record<string, ComponentType<ViewerProps>> = {
  region: RegionViewer,
  // image: ImageViewer, timeseries: TimeSeriesViewer, ...
};
