import React, { useMemo } from "react";
import type { AppMode, RobotState, CameraState, LoggerState, EventLog } from "../types";

interface Props {
  page?: "capture" | "dataset" | "screenshots" | "diagnostics" | "setup";
  mode: AppMode;
  robot: RobotState;
  camera: CameraState;
  logger: LoggerState;
  events: EventLog[];
  availableActions: string[];
  nextAction: string;
}

// ─── Joint Bar ────────────────────────────────────────────────────────────────

const DEFAULT_MIN = -Math.PI;
const DEFAULT_MAX = Math.PI;
const NEAR_LIMIT_FRAC = 0.1;   // range의 10% 이내면 "limit 근접"
const MOVING_VEL_THRESH = 0.05; // rad/s

const JointBar = React.memo(function JointBar({
  label, sublabel, value, min, max, isMoving, isNearLimit,
}: {
  label: string;
  sublabel?: string;
  value: number | undefined;
  min: number;
  max: number;
  isMoving: boolean;
  isNearLimit: boolean;
}) {
  const pct =
    value !== undefined
      ? Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
      : 50;
  const deg = value !== undefined ? (value * 180 / Math.PI).toFixed(0) + "°" : "—";
  const dotColor = isNearLimit ? "#ef4444" : isMoving ? "#6366f1" : "#94a3b8";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5, marginBottom: 4 }}>
      <span style={{ width: 32, flexShrink: 0, display: "flex", flexDirection: "column" }}>
        <span style={{ fontSize: 10, color: "#94a3b8", fontWeight: 700, lineHeight: 1.2 }}>{label}</span>
        {sublabel && <span style={{ fontSize: 8, color: "#cbd5e1", lineHeight: 1.1 }}>{sublabel}</span>}
      </span>
      <div style={{ flex: 1, height: 5, background: "#e2e8f0", borderRadius: 3, position: "relative" }}>
        {/* Limit danger zones — only shown when near limit */}
        {isNearLimit && (
          <>
            <div style={{
              position: "absolute", left: 0, top: 0,
              width: "10%", height: "100%",
              background: "rgba(239,68,68,0.28)",
              borderRadius: "3px 0 0 3px",
            }} />
            <div style={{
              position: "absolute", right: 0, top: 0,
              width: "10%", height: "100%",
              background: "rgba(239,68,68,0.28)",
              borderRadius: "0 3px 3px 0",
            }} />
          </>
        )}
        {/* Position dot */}
        {value !== undefined && (
          <div style={{
            position: "absolute",
            left: `${pct}%`,
            top: "50%",
            transform: "translate(-50%, -50%)",
            width: 9, height: 9,
            borderRadius: "50%",
            background: dotColor,
            boxShadow: isMoving ? `0 0 5px ${dotColor}` : "none",
            transition: "left 0.12s ease",
            zIndex: 1,
          }} />
        )}
      </div>
      <span style={{
        width: 34, fontSize: 10, textAlign: "right",
        fontFamily: "monospace", flexShrink: 0,
        color: isNearLimit ? "#ef4444" : "#64748b",
        fontWeight: isNearLimit ? 700 : 400,
      }}>
        {deg}
      </span>
    </div>
  );
});

// ─── Camera Warning ───────────────────────────────────────────────────────────

function CameraWarning({ camera }: { camera: CameraState }) {
  const warnings = Object.entries(camera.cameras ?? {})
    .filter(([, c]) => (c.dropped_frames ?? 0) > 0)
    .map(([id, c]) => `${id} ${c.dropped_frames} dropped`);
  if (warnings.length === 0) return null;
  return (
    <div style={{
      background: "#fff7ed", borderLeft: "3px solid #f59e0b",
      padding: "4px 8px", fontSize: 10, color: "#92400e",
    }}>
      ⚠ {warnings.join("  ·  ")}
    </div>
  );
}

// ─── Safety Checklist ─────────────────────────────────────────────────────────

function SafeChecklist({
  robot, camera, logger, mode,
}: {
  robot: RobotState;
  camera: CameraState;
  logger: LoggerState;
  mode: AppMode;
}) {
  const cameraStreaming =
    camera.cameras?.realsense?.streaming ?? camera.streaming ?? false;
  const diskOk = (logger.disk_free_gb ?? 999) > 20;
  const poseSet = mode !== "IDLE" && mode !== "CONNECTING";

  const checks = [
    { label: "Joint limits loaded", ok: robot.joint_limits != null },
    { label: "Ready pose confirmed", ok: poseSet },
    { label: "Camera stream", ok: cameraStreaming },
    {
      label: `${logger.disk_free_gb !== undefined ? logger.disk_free_gb + " GB" : "—"} free`,
      ok: diskOk,
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      {checks.map(c => (
        <div key={c.label} style={{ display: "flex", gap: 5, alignItems: "center", fontSize: 11 }}>
          <span style={{ color: c.ok ? "#22c55e" : "#f59e0b", fontWeight: 700, flexShrink: 0 }}>
            {c.ok ? "✓" : "⚠"}
          </span>
          <span style={{ color: c.ok ? "#374151" : "#92400e" }}>{c.label}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Panel ───────────────────────────────────────────────────────────────

export default function StatusPanel({
  page = "capture", mode, robot, camera, logger, events, availableActions, nextAction,
}: Props) {
  const joints = robot.position ?? [];
  const velocities = robot.velocity ?? [];
  const limits = robot.joint_limits;
  const JOINT_LABELS: [string, string][] = [
    ["J1", "Base"], ["J2", "Shld"], ["J3", "Elbow"],
    ["J4", "W1"],   ["J5", "W2"],   ["J6", "W3"],
  ];

  const getMin = (i: number) => limits?.min[i] ?? DEFAULT_MIN;
  const getMax = (i: number) => limits?.max[i] ?? DEFAULT_MAX;
  const isMoving = (i: number) => Math.abs(velocities[i] ?? 0) > MOVING_VEL_THRESH;
  const isNearLimit = (i: number, v?: number) => {
    if (v === undefined) return false;
    const mn = getMin(i), mx = getMax(i);
    const margin = (mx - mn) * NEAR_LIMIT_FRAC;
    return v < mn + margin || v > mx - margin;
  };

  const recentEvents = useMemo(() => [...events].reverse().slice(0, 20), [events]);

  const gripperPct =
    robot.gripper !== undefined
      ? Math.round((robot.gripper / 0.07) * 100)
      : null;
  const gripperLabel =
    robot.gripper !== undefined
      ? robot.gripper < 0.02
        ? `Closed (${gripperPct}%)`
        : robot.gripper > 0.055
          ? `Open (${gripperPct}%)`
          : `Partial (${gripperPct}%)`
      : "—";

  if (page === "dataset") {
    return (
      <aside style={styles.panel}>
        <section style={styles.hint}>
          <div style={styles.hintLabel}>DATASET</div>
          <div style={styles.hintText}>Browse saved episodes, inspect quality, edit tasks, postprocess, and export batches.</div>
        </section>
        <section style={styles.section}>
          <div style={styles.sectionTitle}>Storage</div>
          <Row label="Disk" value={logger.disk_free_gb !== undefined ? `${logger.disk_free_gb} GB` : "—"} ok={(logger.disk_free_gb ?? 999) > 10} />
          <Row label="Logger" value={mode === "REPLAY_RECORDING" || mode === "PROCESSING" ? "Active" : "Idle"} ok={mode !== "REPLAY_RECORDING"} />
        </section>
        <section style={styles.section}>
          <div style={styles.sectionTitle}>Inspector</div>
          <div style={styles.contextNote}>Use the center inspector for selected episode details. Multi-select shows batch status and export actions.</div>
        </section>
      </aside>
    );
  }

  if (page === "setup") {
    return (
      <aside style={styles.panel}>
        <section style={styles.hint}>
          <div style={styles.hintLabel}>SYSTEM SETUP</div>
          <div style={styles.hintText}>Configure safe return, calibration, cameras, and storage.</div>
        </section>
        <section style={styles.section}>
          <div style={styles.sectionTitle}>Current Pose</div>
          {JOINT_LABELS.map(([j], i) => (
            <Row key={j} label={j} value={joints[i] !== undefined ? joints[i].toFixed(3) : "—"} mono />
          ))}
        </section>
        <section style={styles.section}>
          <div style={styles.sectionTitle}>Calibration</div>
          <Row label="Joint limits" value={robot.joint_limits ? "Loaded" : "Missing"} ok={robot.joint_limits != null} />
          <Row label="Robot" value={robot.connected ? "Connected" : "Disconnected"} ok={robot.connected} />
        </section>
      </aside>
    );
  }

  if (page === "diagnostics") {
    return (
      <aside style={styles.panel}>
        <section style={styles.hint}>
          <div style={styles.hintLabel}>DIAGNOSTICS</div>
          <div style={styles.hintText}>Raw robot, camera, logger, and event details live here.</div>
        </section>
        <section style={styles.section}>
          <div style={styles.sectionTitle}>Raw Summary</div>
          <Row label="Robot Hz" value={robot.hz ? `${robot.hz} Hz` : "—"} ok={(robot.hz ?? 0) >= 40} />
          <Row label="Cameras" value={camera.connected ? "Connected" : "Disconnected"} ok={camera.connected} />
          <Row label="Episode" value={logger.episode_id?.slice(-12) ?? "—"} mono />
        </section>
        <section style={{ ...styles.section, flex: 1 }}>
          <div style={styles.sectionTitle}>Recent Events</div>
          <div style={styles.eventLog}>
            {recentEvents.slice(0, 8).map((e, i) => (
              <div key={i} style={styles.eventRow}>
                <span style={styles.eventTime}>{new Date(e.t * 1000).toLocaleTimeString()}</span>
                <span>{e.message}</span>
              </div>
            ))}
          </div>
        </section>
      </aside>
    );
  }

  return (
    <aside style={styles.panel}>

      {/* Next action hint */}
      {nextAction && (
        <section style={styles.hint}>
          <div style={styles.hintLabel}>NEXT</div>
          <div style={styles.hintText}>{nextAction}</div>
        </section>
      )}

      {/* Robot overview */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>Robot</div>
        <Row label="Connected" value={robot.connected ? "Yes" : "No"} ok={robot.connected} />
        <Row label="Mode" value={robot.mode ?? "—"} />
        <Row
          label="Feedback"
          value={robot.hz ? `${robot.hz} Hz` : "—"}
          ok={(robot.hz ?? 0) >= 40}
        />
        <Row
          label="Motion"
          value={robot.is_moving ? "Moving" : "Idle"}
          ok={!robot.is_moving}
        />
      </section>

      {/* Joint bars */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>Joint Angles</div>
        {JOINT_LABELS.map(([j, sub], i) => (
          <JointBar
            key={j}
            label={j}
            sublabel={sub}
            value={joints[i]}
            min={getMin(i)}
            max={getMax(i)}
            isMoving={isMoving(i)}
            isNearLimit={isNearLimit(i, joints[i])}
          />
        ))}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
          <span style={{ fontSize: 10, color: "#94a3b8" }}>Gripper</span>
          <span style={{ fontSize: 11, fontFamily: "monospace", color: "#374151", fontWeight: 600 }}>
            {gripperLabel}
          </span>
        </div>
      </section>

      {/* Camera state (multi-camera) */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>Cameras</div>
        <CameraWarning camera={camera} />
        <Row label="Connected" value={camera.connected ? "Yes" : "No"} ok={camera.connected} />
        {Object.entries(camera.cameras ?? {}).map(([camId, info]: [string, any]) => (
          <div key={camId} style={{ marginTop: 5 }}>
            <div style={{ fontSize: 10, color: "#94a3b8", textTransform: "uppercase", marginBottom: 2, fontWeight: 700 }}>
              {camId}
            </div>
            <Row label="Avail" value={info.available ? "Yes" : "No"} ok={info.available} />
            {info.available && (
              <>
                <Row label="FPS" value={info.fps !== undefined ? `${info.fps}` : "—"} ok={(info.fps ?? 0) >= 25} />
                <Row label="Dropped" value={String(info.dropped_frames ?? 0)} ok={(info.dropped_frames ?? 0) === 0} />
              </>
            )}
          </div>
        ))}
      </section>

      {/* Logger */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>Logger</div>
        <Row label="Episode" value={logger.episode_id?.slice(-12) ?? "—"} mono />
        <Row label="Disk" value={logger.disk_free_gb !== undefined ? `${logger.disk_free_gb} GB` : "—"} ok={(logger.disk_free_gb ?? 999) > 10} />
        <Row label="Teach" value={String(logger.teach_samples ?? 0)} />
        <Row label="Exec" value={String(logger.exec_samples ?? 0)} />
        <Row label="Frames" value={String(logger.cam_frames ?? 0)} />
      </section>

      {/* Safety checklist */}
      <section style={styles.section}>
        <div style={styles.sectionTitle}>Safety</div>
        <SafeChecklist robot={robot} camera={camera} logger={logger} mode={mode} />
      </section>

      {/* Events log */}
      <section style={{ ...styles.section, flex: 1 }}>
        <div style={styles.sectionTitle}>Events</div>
        <div style={styles.eventLog}>
          {recentEvents.map((e, i) => (
            <div key={i} style={styles.eventRow}>
              <span style={styles.eventTime}>{new Date(e.t * 1000).toLocaleTimeString()}</span>
              <span>{e.message}</span>
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}

function Row({
  label, value, ok, mono,
}: { label: string; value: string; ok?: boolean; mono?: boolean }) {
  const color = ok === undefined ? "#374151" : ok ? "#15803d" : "#dc2626";
  return (
    <div style={styles.row}>
      <span style={styles.rowLabel}>{label}</span>
      <span style={{ ...styles.rowValue, color, fontFamily: mono ? "monospace" : undefined }}>
        {value}
      </span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 240,
    background: "#fff",
    borderLeft: "1px solid #e5e7eb",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 0,
    flexShrink: 0,
  },
  hint: {
    background: "#eff6ff",
    borderBottom: "1px solid #bfdbfe",
    padding: "10px 14px",
  },
  hintLabel: {
    fontSize: 10,
    fontWeight: 700,
    color: "#3b82f6",
    letterSpacing: 1,
    marginBottom: 4,
  },
  hintText: {
    fontSize: 12,
    color: "#1e40af",
    lineHeight: 1.5,
  },
  section: {
    borderBottom: "1px solid #f1f5f9",
    padding: "10px 14px",
  },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    color: "#94a3b8",
    letterSpacing: 1,
    marginBottom: 8,
    textTransform: "uppercase",
  },
  row: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "2px 0",
    fontSize: 11,
  },
  rowLabel: { color: "#6b7280" },
  rowValue: { fontWeight: 600 },
  eventLog: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    maxHeight: 160,
    overflowY: "auto",
  },
  eventRow: {
    display: "flex",
    gap: 6,
    fontSize: 11,
    color: "#374151",
    lineHeight: 1.4,
  },
  eventTime: {
    color: "#9ca3af",
    fontFamily: "monospace",
    flexShrink: 0,
  },
  contextNote: {
    fontSize: 12,
    color: "#64748b",
    lineHeight: 1.5,
  },
};
