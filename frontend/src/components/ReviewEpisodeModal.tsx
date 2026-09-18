import React, { useEffect, useState } from "react";
import type { AppMode, SaveStep, SaveStepStatus } from "../types";

interface Props {
  mode: AppMode;
  availableActions: string[];
  episodeId?: string;
  saveSteps?: SaveStep[];
  onClose?: () => void;
}

export default function ReviewEpisodeModal({
  mode, availableActions, episodeId, saveSteps = [], onClose,
}: Props) {
  const [label, setLabel] = useState<"success" | "failure" | null>(null);
  const [reason, setReason] = useState("");
  const [saving, setSaving] = useState(false);
  const [videoTab, setVideoTab] = useState("color");
  const [partialReplay, setPartialReplay] = useState(false);
  const [partialReason, setPartialReason] = useState("");
  const [abortClass, setAbortClass] = useState("");
  const [cameraVideos, setCameraVideos] = useState<Array<{ id: string; label: string; url: string }>>([]);

  const can = (a: string) => availableActions.includes(a);

  useEffect(() => {
    setLabel(null);
    setReason("");
    setPartialReplay(false);
    setPartialReason("");
    setAbortClass("");
  }, [episodeId]);

  useEffect(() => {
    if (!episodeId) return;
    fetch(`/api/episodes/${episodeId}`)
      .then(r => r.json())
      .then(ep => {
        const replay = ep.recording?.last_replay;
        setPartialReplay(Boolean(replay?.partial));
        setPartialReason(String(replay?.reason ?? ""));
        setAbortClass(String(replay?.abort_class ?? ""));
        const takes: Array<{
          take: string;
          has_video?: boolean;
          has_webcam_0?: boolean;
          has_webcam_1?: boolean;
          cameras?: Array<{
            id: string;
            label: string;
            streams?: Array<{ kind: string; label: string; url: string }>;
          }>;
        }> = ep.takes ?? [];
        if (takes.length > 0) {
          const latest = takes[takes.length - 1];
          if (latest.cameras?.length) {
            const tabs = latest.cameras.flatMap(c => {
              const streams = c.streams?.length
                ? c.streams
                : [{ kind: "color", label: "Color", url: `/api/episodes/${episodeId}/video_camera/${c.id}/color` }];
              const multi = streams.length > 1;
              return streams.map(s => ({
                id: `${c.id}:${s.kind}`,
                label: multi ? `${c.label || c.id} ${s.label || s.kind}` : c.label || c.id,
                url: s.url,
              }));
            });
            setCameraVideos(tabs);
            setVideoTab(tabs[0]?.id ?? "color");
          } else {
            setCameraVideos([
              ...(latest.has_video ? [{ id: "color", label: "RGB", url: `/api/episodes/${episodeId}/video` }] : []),
              ...(latest.has_video ? [{ id: "depth", label: "Depth", url: `/api/episodes/${episodeId}/video_depth` }] : []),
              ...(latest.has_webcam_0 ? [{ id: "cam0", label: "Cam0", url: `/api/episodes/${episodeId}/video_camera/cam0/color` }] : []),
              ...(latest.has_webcam_1 ? [{ id: "cam1", label: "Cam1", url: `/api/episodes/${episodeId}/video_camera/cam1/color` }] : []),
            ]);
          }
        }
      })
      .catch(() => {});
  }, [episodeId]);

  const postAndClose = (path: string, body?: object) => {
    fetch(path, {
      method: "POST",
      ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
    }).finally(() => onClose?.());
  };

  const handleSave = async () => {
    if (!label) return;
    setSaving(true);
    await fetch("/api/episodes/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ success: label === "success", reason }),
    });
    setSaving(false);
    onClose?.();
  };

  const videoTabs = cameraVideos;
  const videoUrl = videoTabs.find(t => t.id === videoTab)?.url ?? null;

  if (mode === "PROCESSING") {
    return (
      <ModalFrame>
        <div style={styles.processingBox}>
          <div style={styles.processingTitle}>Saving episode</div>
          <div style={styles.episodeId}>{episodeId}</div>
          <div style={stepListStyles.container}>
            {saveSteps.map(step => (
              <div key={step.key} style={stepListStyles.row}>
                <span style={stepListStyles.icon(step.status)}>
                  {step.status === "waiting" ? "○"
                    : step.status === "running" ? "◐"
                    : step.status === "ok" ? "✓"
                    : "✕"}
                </span>
                <span style={stepListStyles.label}>{step.label}</span>
                {step.detail && <span style={stepListStyles.detail}>{step.detail}</span>}
              </div>
            ))}
          </div>
        </div>
      </ModalFrame>
    );
  }

  return (
    <ModalFrame onClose={onClose}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>Review Episode</div>
          <div style={styles.episodeId}>{episodeId}</div>
        </div>
        <button style={styles.closeBtn} onClick={onClose}>×</button>
      </div>
      <div style={styles.body}>
        {partialReplay && (
          <div style={styles.partialNotice}>
            {abortClass
              ? `Replay stopped safely (${abortClass}): ${partialReason || "motion aborted"}. This take is excluded from the default LeRobot export.`
              : "Partial replay — recording was stopped early. The captured portion can still be reviewed and saved."}
          </div>
        )}
        <div style={styles.previewColumn}>
          <div style={styles.tabs}>
            {videoTabs.map(({ id, label }) => (
              <button
                key={id}
                style={{
                  ...styles.tabBtn,
                  background: videoTab === id ? "#6366f1" : "#f8fafc",
                  color: videoTab === id ? "#fff" : "#374151",
                }}
                onClick={() => setVideoTab(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div style={styles.videoBox}>
            {videoUrl ? (
              <video key={videoUrl} controls src={videoUrl} style={styles.video} />
            ) : (
              <div style={styles.noVideo}>Video not available yet</div>
            )}
          </div>
        </div>

        <div style={styles.labelPanel}>
          <div style={styles.sectionTitle}>Label this episode</div>
          <div style={styles.labelBtns}>
            <LabelBtn label="Success" selected={label === "success"} color="#16a34a" onClick={() => setLabel("success")} />
            <LabelBtn label="Failure" selected={label === "failure"} color="#dc2626" onClick={() => setLabel("failure")} />
          </div>

          {label === "failure" && (
            <>
              <label style={styles.fieldLabel}>Failure reason</label>
              <select style={styles.select} value={reason} onChange={e => setReason(e.target.value)}>
                <option value="">select reason</option>
                <option value="grasp_fail">Grasp failed</option>
                <option value="object_slip">Object slipped</option>
                <option value="collision">Collision</option>
                <option value="camera_issue">Camera issue</option>
                <option value="trajectory_error">Trajectory error</option>
                <option value="other">Other</option>
              </select>
            </>
          )}

          <div style={styles.actionBtns}>
            <button
              style={{ ...styles.primaryBtn, opacity: (!label || saving) ? 0.5 : 1 }}
              onClick={handleSave}
              disabled={!label || saving || !can("save_episode")}
            >
              {saving ? "Saving..." : "Save Episode"}
            </button>
            {can("retake_replay") && (
              <button style={styles.secondaryBtn} onClick={() => postAndClose("/api/episodes/retake_replay")}>
                Retake Replay
              </button>
            )}
            {can("retake_teach") && (
              <button style={styles.secondaryBtn} onClick={() => postAndClose("/api/episodes/retake_teach")}>
                Retake Teaching
              </button>
            )}
            {can("discard_episode") && (
              <button style={styles.dangerBtn} onClick={() => postAndClose("/api/episodes/discard")}>
                Discard
              </button>
            )}
          </div>
        </div>
      </div>
    </ModalFrame>
  );
}

function ModalFrame({ children, onClose }: { children: React.ReactNode; onClose?: () => void }) {
  return (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

function LabelBtn({
  label, selected, color, onClick,
}: { label: string; selected: boolean; color: string; onClick: () => void }) {
  return (
    <button
      style={{
        flex: 1,
        padding: "12px 0",
        border: `2px solid ${selected ? color : "#e5e7eb"}`,
        borderRadius: 8,
        background: selected ? color + "18" : "#fff",
        color: selected ? color : "#6b7280",
        fontWeight: 700,
        fontSize: 14,
        cursor: "pointer",
      }}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

const STEP_COLORS: Record<SaveStepStatus, string> = {
  waiting: "#94a3b8",
  running: "#6366f1",
  ok: "#16a34a",
  failed: "#dc2626",
};

const stepListStyles = {
  container: {
    background: "#fff",
    borderRadius: 8,
    border: "1px solid #e5e7eb",
    padding: "14px 18px",
    minWidth: 360,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  } as React.CSSProperties,
  row: { display: "flex", alignItems: "center", gap: 10, fontSize: 14 } as React.CSSProperties,
  icon: (status: SaveStepStatus): React.CSSProperties => ({
    width: 20,
    textAlign: "center",
    fontWeight: 700,
    color: STEP_COLORS[status],
  }),
  label: { flex: 1, color: "#1e293b" } as React.CSSProperties,
  detail: { fontSize: 11, color: "#94a3b8", fontFamily: "monospace" } as React.CSSProperties,
};

const styles: Record<string, React.CSSProperties> = {
  backdrop: {
    position: "fixed",
    inset: 0,
    background: "rgba(15,23,42,0.58)",
    zIndex: 1000,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  modal: {
    width: "min(980px, 96vw)",
    maxHeight: "92vh",
    overflow: "auto",
    background: "#fff",
    borderRadius: 10,
    boxShadow: "0 24px 70px rgba(15,23,42,0.28)",
    padding: 20,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 16,
    marginBottom: 14,
  },
  title: { fontSize: 18, fontWeight: 800, color: "#1e293b" },
  episodeId: { fontSize: 12, color: "#94a3b8", fontFamily: "monospace", marginTop: 3 },
  closeBtn: {
    width: 30,
    height: 30,
    borderRadius: 6,
    border: "1px solid #e5e7eb",
    background: "#fff",
    color: "#64748b",
    fontSize: 20,
    cursor: "pointer",
    lineHeight: 1,
  },
  body: { display: "grid", gridTemplateColumns: "1fr 280px", gap: 18 },
  partialNotice: {
    gridColumn: "1 / -1",
    border: "1px solid #fbbf24",
    borderRadius: 8,
    background: "#fffbeb",
    color: "#92400e",
    padding: "9px 12px",
    fontSize: 12,
    fontWeight: 700,
  },
  previewColumn: { minWidth: 0, display: "flex", flexDirection: "column", gap: 8 },
  tabs: { display: "flex", gap: 4, flexWrap: "wrap" },
  tabBtn: {
    padding: "4px 12px",
    borderRadius: 6,
    border: "1px solid #e2e8f0",
    fontWeight: 700,
    fontSize: 12,
    cursor: "pointer",
  },
  videoBox: {
    minHeight: 340,
    background: "#0f172a",
    borderRadius: 8,
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  video: { width: "100%", height: "100%", maxHeight: 520, objectFit: "contain" },
  noVideo: { color: "#64748b", fontSize: 13 },
  labelPanel: {
    border: "1px solid #e5e7eb",
    borderRadius: 8,
    padding: 16,
    display: "flex",
    flexDirection: "column",
    gap: 12,
    alignSelf: "start",
  },
  sectionTitle: { fontWeight: 800, fontSize: 14, color: "#1e293b" },
  labelBtns: { display: "flex", gap: 8 },
  fieldLabel: { fontSize: 12, color: "#64748b" },
  select: { width: "100%", border: "1px solid #cbd5e1", borderRadius: 6, padding: "7px 8px", fontSize: 13 },
  actionBtns: { display: "flex", flexDirection: "column", gap: 8, marginTop: 4 },
  primaryBtn: {
    padding: "10px 0",
    background: "#6366f1",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontWeight: 800,
    fontSize: 14,
    cursor: "pointer",
    width: "100%",
  },
  secondaryBtn: {
    padding: "8px 0",
    background: "#f1f5f9",
    color: "#374151",
    border: "none",
    borderRadius: 8,
    fontSize: 13,
    cursor: "pointer",
  },
  dangerBtn: {
    padding: "8px 0",
    background: "#fee2e2",
    color: "#dc2626",
    border: "none",
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 700,
    cursor: "pointer",
  },
  processingBox: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 14,
    padding: "16px 10px",
  },
  processingTitle: { color: "#6366f1", fontSize: 18, fontWeight: 800 },
};
