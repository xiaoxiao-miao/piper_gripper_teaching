import React, { useState, useEffect } from "react";
import type { RobotState, CameraState } from "../types";

interface Props {
  devMode: boolean;
  robot: RobotState;
  camera: CameraState;
}

export default function DiagnosticsPage({ devMode, robot, camera }: Props) {
  const [diag, setDiag] = useState<Record<string, unknown>>({});

  const fetchDiag = () => {
    fetch("/api/diagnostics").then(r => r.json()).then(setDiag).catch(() => {});
  };

  useEffect(() => {
    fetchDiag();
    const id = setInterval(fetchDiag, 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Diagnostics</h2>
      {!devMode && (
        <p style={styles.warn}>Developer mode is OFF. Enable DEV in the top bar to see all details.</p>
      )}

      <div style={styles.grid}>
        <Card title="Controller">
          <KV k="State" v={String(diag.state ?? "—")} />
          <KV k="Piper connected" v={String(diag.piper_connected ?? "—")} />
          <KV k="RealSense connected" v={String(diag.realsense_connected ?? "—")} />
          <KV k="ROS OK" v={String(diag.ros_ok ?? "—")} />
          <KV k="Episode" v={String(diag.episode_id ?? "—")} />
          <KV k="Teach buffer" v={String(diag.teach_buffer_len ?? "—")} />
          <KV k="Event log len" v={String(diag.event_log_len ?? "—")} />
        </Card>

        {devMode && (
          <>
            <Card title="Robot raw state">
              <KV k="Connected" v={String(robot.connected)} />
              <KV k="Hz" v={String(robot.hz ?? "—")} />
              <KV k="Mode" v={robot.mode ?? "—"} />
              <KV k="is_moving" v={String(robot.is_moving ?? "—")} />
              <KV k="Transition" v={String(robot.transition_state ?? "—")} />
              <KV k="Motor enable" v={(robot.motor_enable_status ?? []).map(v => v ? "1" : "0").join(",") || "—"} />
              <KV k="Feedback age" v={robot.feedback_age_s == null ? "—" : `${robot.feedback_age_s.toFixed(3)} s`} />
              <KV k="Transition error" v={String(robot.transition_error ?? "—")} />
              <KV k="Safety state" v={String(robot.abort_class ?? "READY")} />
              <KV k="Safety latched" v={String(robot.safety_latched ?? false)} />
              <KV k="Safety reason" v={String(robot.safety_reason ?? "—")} />
              <KV k="CAN healthy" v={String(robot.can_healthy ?? "—")} />
              <KV k="Motor faults" v={(robot.motor_fault_codes ?? []).map((v, i) => `J${i + 1}=0x${(v & ~(1 << 6)).toString(16)}`).join(", ") || "—"} />
              {(robot.position ?? []).map((v, i) => (
                <KV key={i} k={`q${i + 1}`} v={v.toFixed(6)} />
              ))}
              <KV k="gripper" v={String(robot.gripper?.toFixed(6) ?? "—")} />
            </Card>

            <Card title="Camera raw state">
              <KV k="Available" v={String(camera.available)} />
              <KV k="FPS" v={String(camera.fps ?? "—")} />
              <KV k="Recording" v={String(camera.recording ?? "—")} />
              <KV k="Captured" v={String(camera.captured_frames ?? "—")} />
              <KV k="Written" v={String(camera.written_frames ?? "—")} />
              <KV k="Dropped" v={String(camera.dropped_frames ?? "—")} />
              <KV k="Queue" v={String(camera.queue_len ?? "—")} />
            </Card>

            <Card title="ROS Topics">
              <KV k="/joint_states_single" v="subscribe" />
              <KV k="/joint_ctrl_single" v="publish" />
              <p style={{ fontSize: 11, color: "#9ca3af", marginTop: 8 }}>
                Note: run `ros2 topic hz /joint_states_single` to verify feedback rate.
              </p>
            </Card>
          </>
        )}
      </div>

      <button style={styles.refreshBtn} onClick={fetchDiag}>Refresh</button>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardTitle}>{title}</div>
      {children}
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={styles.kv}>
      <span style={styles.kvKey}>{k}</span>
      <span style={styles.kvVal}>{v}</span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 900 },
  title: { fontSize: 20, fontWeight: 700, marginBottom: 8, color: "#1e293b" },
  warn: {
    background: "#fffbeb",
    border: "1px solid #fde68a",
    borderRadius: 6,
    padding: "8px 12px",
    fontSize: 13,
    color: "#92400e",
    marginBottom: 16,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 12,
    marginBottom: 16,
  },
  card: {
    background: "#fff",
    borderRadius: 10,
    padding: 14,
    border: "1px solid #e5e7eb",
  },
  cardTitle: {
    fontWeight: 700,
    fontSize: 13,
    color: "#374151",
    marginBottom: 8,
    borderBottom: "1px solid #f1f5f9",
    paddingBottom: 6,
  },
  kv: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    padding: "2px 0",
  },
  kvKey: { color: "#6b7280" },
  kvVal: { fontFamily: "monospace", color: "#1e293b", fontWeight: 600 },
  refreshBtn: {
    background: "#f1f5f9",
    border: "none",
    borderRadius: 6,
    padding: "6px 16px",
    fontSize: 12,
    cursor: "pointer",
  },
};
