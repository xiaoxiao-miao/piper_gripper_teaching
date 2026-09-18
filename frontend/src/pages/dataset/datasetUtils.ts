import type { TakeJointsData } from "../../types";

export type ViewMode = "compact" | "thumbnail" | "grid";
export type InspectorTab = "preview" | "trajectory" | "debug" | "postprocess" | "export";

export const COMPLETENESS: Array<{ key: string; label: string; title: string }> = [
  { key: "rgb", label: "RGB", title: "RealSense RGB video" },
  { key: "depth", label: "D", title: "Depth video" },
  { key: "cam0", label: "C0", title: "Cam0 ego video" },
  { key: "cam1", label: "C1", title: "Cam1 overview video" },
  { key: "joints", label: "J", title: "Joint logs" },
  { key: "traj", label: "T", title: "Trajectory data" },
  { key: "label", label: "L", title: "Success/failure label" },
];

export function fmtMb(mb?: number): string {
  if (mb === undefined || mb === null) return "-";
  if (mb < 1) return `${Math.round(mb * 1024)} KB`;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  return `${(mb / 1024).toFixed(2)} GB`;
}

export function fmtDuration(seconds?: number): string {
  if (seconds === undefined || seconds === null) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(0).padStart(2, "0");
  return `${m}m ${s}s`;
}

export function fmtDate(value?: string): string {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value.slice(0, 16).replace("T", " ");
  return d.toLocaleString(undefined, {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function takeDuration(data: TakeJointsData | null): number | undefined {
  if (!data) return undefined;
  for (const src of ["teach", "executed", "command"] as const) {
    const samples = data[src];
    if (samples.length > 1) return samples[samples.length - 1].t;
  }
  return undefined;
}

export function statusLabel(success: boolean | null | undefined): string {
  if (success === true) return "SUCCESS";
  if (success === false) return "FAILURE";
  return "UNLABELED";
}
