import React, { useEffect, useState } from "react";
import type { RobotState } from "../types";

export default function SafetyBanner({ robot, mobile = false }: { robot: RobotState; mobile?: boolean }) {
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const locked = Boolean(robot.safety_latched) || robot.abort_class === "CAN_LOST";
  const canLost = robot.abort_class === "CAN_LOST" || robot.can_healthy === false;
  const motionAborted = robot.abort_class === "MOTION_ABORTED" && !locked;
  const gripperWarning = robot.gripper_responsive === false;

  useEffect(() => {
    if (!locked) {
      setConfirmed(false);
      setBusy(false);
    }
  }, [locked]);

  if (!locked && !motionAborted && !gripperWarning) return null;

  if (!locked && !motionAborted && gripperWarning) {
    return (
      <div style={{ ...styles.banner, ...styles.gripperWarn, ...(mobile ? styles.mobile : {}) }}>
        <div style={styles.title}>⚠ Gripper not responding</div>
        <div style={styles.reason}>
          마지막 그리퍼 명령이 실제로 반영되지 않았습니다 — 케이블 연결을 확인하세요.
        </div>
      </div>
    );
  }

  const recover = async () => {
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch("/api/robot/motors/recover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmed: true }),
      });
      const result = await response.json();
      setMessage(result.ok
        ? (result.motors_enabled
          ? "Recovered at the current pose. Previous motion was not resumed."
          : "CAN recovered. Motors remain disabled; the next motion will enable safely from the current pose.")
        : (result.reason || "Recovery failed"));
    } catch {
      setMessage("Recovery request failed");
    } finally {
      setBusy(false);
    }
  };

  const faultLabels = (robot.motor_fault_codes ?? [])
    .map((code, index) => ({ code: code & ~(1 << 6), index }))
    .filter(item => item.code)
    .map(item => `J${item.index + 1} (motor ${item.index + 1} / index ${item.index}): 0x${item.code.toString(16).toUpperCase()}`);
  const reason = robot.safety_reason || robot.last_motion_abort?.reason || robot.transition_error || "Motion stopped";

  return (
    <div style={{
      ...styles.banner,
      ...(locked ? styles.locked : styles.aborted),
      ...(mobile ? styles.mobile : {}),
    }}>
      <div style={styles.title}>{locked ? "⚠ SAFETY LOCK" : "⚠ Replay stopped safely"}</div>
      <div style={styles.reason}>{reason}</div>
      {canLost && (
        <div style={styles.canHint}>
          CAN 연결을 자동 재시도하고 있습니다. 통신이 복구된 뒤 팔을 지지하고 Recover Motors를 실행하세요.
        </div>
      )}
      {faultLabels.map(label => <div key={label} style={styles.fault}>{label}</div>)}
      {locked && (
        <div style={styles.controls}>
          <label style={styles.confirm}>
            <input type="checkbox" checked={confirmed} onChange={event => setConfirmed(event.target.checked)} />
            팔을 지지했고 작업 공간을 비웠습니다
          </label>
          <button
            type="button"
            onClick={recover}
            disabled={!confirmed || busy || canLost}
            style={styles.button}
          >
            {busy ? "Recovering…" : "Recover Motors"}
          </button>
        </div>
      )}
      {message && <div style={styles.message}>{message}</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  banner: { padding: "10px 16px", borderBottom: "1px solid", fontSize: 13 },
  locked: { background: "#450a0a", borderColor: "#ef4444", color: "#fee2e2" },
  aborted: { background: "#fffbeb", borderColor: "#f59e0b", color: "#78350f" },
  gripperWarn: { background: "#fffbeb", borderColor: "#f59e0b", color: "#78350f" },
  mobile: { borderRadius: 10, margin: "8px 10px", border: "1px solid" },
  title: { fontWeight: 800, marginBottom: 3 },
  reason: { fontFamily: "monospace", overflowWrap: "anywhere" },
  fault: { fontFamily: "monospace", marginTop: 2 },
  controls: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10, marginTop: 8 },
  confirm: { display: "flex", alignItems: "center", gap: 6, fontWeight: 600 },
  button: { padding: "7px 12px", border: 0, borderRadius: 6, fontWeight: 800, cursor: "pointer" },
  message: { marginTop: 6, fontWeight: 600 },
  canHint: { marginTop: 6, fontWeight: 600, color: "#fecaca" },
};
