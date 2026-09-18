/**
 * MobilePage — 모바일 Teaching 조작 UI
 * /mobile 경로에서 렌더링됨 (index.tsx에서 pathname 분기)
 *
 * 기존 MOBILE_HTML(Python 문자열 임베딩)을 대체하는 React 구현.
 * - 다크 테마, 터치 최적화
 * - Teaching Mode: Freedrive / Phone Mouse (TEACH_READY에서 선택)
 * - TEACH_RECORDING: 선택 모드만 표시, 모드 전환 없음
 */
import React, { useState, useEffect, useRef, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import type { AppMode, ModeChangePayload, RobotState, LoggerState, CameraState } from "../types";
import SafetyBanner from "../components/SafetyBanner";

// Production mobile UI is served by Flask on :5002 or HTTPS :5003. Keep
// Socket.IO on the page origin to avoid mixed-content blocking on HTTPS.
// React's development server still connects directly to the Flask port.
const FLASK_ORIGIN = window.location.port === "3000"
  ? `http://${window.location.hostname}:5002`
  : window.location.origin;

// ── 타입 ─────────────────────────────────────────────────────────────

type TeachMode = "freedrive" | "phone_mouse";

// ── 메인 컴포넌트 ─────────────────────────────────────────────────────

export default function MobileApp() {
  const [mode, setMode] = useState<AppMode>("IDLE");
  const [actions, setActions] = useState<string[]>([]);
  const [robot, setRobot] = useState<RobotState>({ connected: false });
  const [camera, setCamera] = useState<CameraState>({ available: false });
  const [logger, setLogger] = useState<LoggerState>({});

  // Teaching mode (TEACH_READY에서 선택, TEACH_RECORDING에서 고정)
  const [teachMode, setTeachMode] = useState<TeachMode>("freedrive");

  // Replay speed
  const [speedScale, setSpeedScale] = useState(0.5);
  const [replayStarting, setReplayStarting] = useState(false);
  const [replayMessage, setReplayMessage] = useState("");

  // Freedrive
  const [motorsFree, setMotorsFree] = useState(false);
  const [disableCountdown, setDisableCountdown] = useState(0);

  // Phone Mouse / Cartesian Jog
  const [pmActive, setPmActive] = useState(false);
  const [pmDeadman, setPmDeadman] = useState(false);
  const [gyro, setGyro] = useState({ beta: 0, gamma: 0 });
  const gyroRef = useRef({ beta: 0, gamma: 0 });
  // neutral calibration: deadman 눌린 순간의 기준 방향
  const gyroNeutralRef = useRef({ beta: 0, gamma: 0 });
  const pmIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pmDeadmanRef = useRef(false);

  const socketRef = useRef<Socket | null>(null);

  // ── Socket.IO ───────────────────────────────────────────────────────

  useEffect(() => {
    // Service Worker 해제 (React build SW가 socket.io polling 가로챌 수 있음)
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.getRegistrations().then((regs) => {
        if (regs.length > 0) {
          Promise.all(regs.map((r) => r.unregister())).then(() => {
            // SW 해제 후 페이지 새로고침 불필요 — 이미 React에서 연결
          });
        }
      });
    }

    const s = io(FLASK_ORIGIN, {
      transports: ["polling", "websocket"],
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
    });
    socketRef.current = s;

    s.on("mode_change", (data: ModeChangePayload) => {
      setMode(data.mode);
      setActions(data.available_actions);

      // TEACH_RECORDING 이탈 시 phone mouse 정리
      if (data.mode !== "TEACH_RECORDING" && data.mode !== "TEACH_STARTING") {
        setPmActive(false);
        pmDeadmanRef.current = false;
        setPmDeadman(false);
        if (pmIntervalRef.current) {
          clearInterval(pmIntervalRef.current);
          pmIntervalRef.current = null;
        }
        setMotorsFree(false);
        setDisableCountdown(0);
      }
    });

    s.on("robot_state", (data: RobotState) => setRobot(data));
    s.on("logger_state", (data: LoggerState) => setLogger(data));
    s.on("camera_state", (data: CameraState) => setCamera(data));

    return () => {
      s.disconnect();
    };
  }, []);

  // ── Freedrive 카운트다운 ────────────────────────────────────────────

  useEffect(() => {
    if (disableCountdown <= 0) return;
    const t = setTimeout(() => {
      const next = disableCountdown - 1;
      if (next === 0) {
        fetch("/api/robot/motors/disable", { method: "POST" })
          .then(r => r.json())
          .then(data => {
            setMotorsFree(Boolean(data.ok));
            setDisableCountdown(0);
          });
      } else {
        setDisableCountdown(next);
      }
    }, 1000);
    return () => clearTimeout(t);
  }, [disableCountdown]);

  // ── Gyro ────────────────────────────────────────────────────────────

  useEffect(() => {
    const handler = (e: DeviceOrientationEvent) => {
      const val = { beta: e.beta ?? 0, gamma: e.gamma ?? 0 };
      gyroRef.current = val;
      setGyro(val);
    };
    window.addEventListener("deviceorientation", handler);
    return () => window.removeEventListener("deviceorientation", handler);
  }, []);

  // ── Cartesian Jog 명령 전송 (ee_twist_cmd) ──────────────────────────

  const sendCartesianJogCmd = useCallback(() => {
    if (!pmDeadmanRef.current || !socketRef.current) return;
    const DEAD_ZONE = 5;       // deg — 미세 진동 차단
    const SCALE = 0.003;       // deg → m/s 변환 계수 (튜닝 가능)
    const DT = 0.05;           // 50ms 간격
    const applyDZ = (v: number) =>
      Math.abs(v) < DEAD_ZONE ? 0 : (v > 0 ? v - DEAD_ZONE : v + DEAD_ZONE);

    const { beta, gamma } = gyroRef.current;
    const { beta: betaRef, gamma: gammaRef } = gyroNeutralRef.current;

    // 중립 보정: deadman 눌린 순간 기준으로 delta 계산
    const dbeta  = beta  - betaRef;   // pitch delta (전후)
    const dgamma = gamma - gammaRef;  // roll delta  (좌우)

    socketRef.current.emit("ee_twist_cmd", {
      vx: applyDZ(dgamma) * SCALE,       // roll → X (좌우)
      vy: applyDZ(dbeta)  * SCALE * -1,  // pitch → Y (전후, 부호 반전)
      vz: 0,
      wx: 0, wy: 0, wz: 0,
      dt: DT,
    });
  }, []);

  // Deadman 상태 변경 → 기준점 캡처 + interval 시작/중지
  const setDeadman = useCallback(
    (active: boolean) => {
      pmDeadmanRef.current = active;
      setPmDeadman(active);
      if (active) {
        // 버튼 누른 순간의 폰 방향을 중립 기준으로 캡처
        gyroNeutralRef.current = {
          beta:  gyroRef.current.beta,
          gamma: gyroRef.current.gamma,
        };
        if (!pmIntervalRef.current) {
          pmIntervalRef.current = setInterval(sendCartesianJogCmd, 50);
        }
      } else {
        if (pmIntervalRef.current) {
          clearInterval(pmIntervalRef.current);
          pmIntervalRef.current = null;
        }
      }
    },
    [sendCartesianJogCmd],
  );

  // ── 액션 핸들러 ─────────────────────────────────────────────────────

  const can = (action: string) => actions.includes(action);
  const calibrationReady =
    robot.calibration_valid?.length === 6 && robot.calibration_valid.every(Boolean);
  const calibrationHeld = Boolean(robot.motor_enable_status?.some(Boolean));
  const post = (path: string) => fetch(path, { method: "POST" });

  const handleStartTeach = async () => {
    // 1. teach/start → TEACH_RECORDING 진입 (freedrive 5초 타이머 시작)
    await post("/api/teach/start");
    if (teachMode === "phone_mouse") {
      // 2. phone_mouse/enable → 타이머 취소 + motors ON + 즉시 recording
      setPmActive(true);
      await post("/api/robot/phone_mouse/enable");
    }
  };

  const handleStopTeach = () => post("/api/teach/stop");

  const handleStartReplay = async () => {
    setReplayStarting(true);
    setReplayMessage("");
    try {
      const response = await fetch("/api/replay/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed_scale: speedScale }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.ok === false) {
        const recommended = Number(data.recommended_max_speed);
        if (Number.isFinite(recommended) && recommended > 0) {
          setSpeedScale(recommended);
        }
        setReplayMessage(data.reason || `Replay start failed (${response.status})`);
      } else if (data.speed_adjusted) {
        setReplayMessage(`Safety limit applied: ${Math.round(Number(data.speed_scale_applied) * 100)}%`);
      }
    } catch {
      setReplayMessage("Replay start request failed");
    } finally {
      setReplayStarting(false);
    }
  };

  const handleEnableMotors = () =>
    post("/api/robot/motors/enable")
      .then(r => r.json())
      .then(data => { if (data.ok) setMotorsFree(false); });

  const handleStartFreedriveCountdown = () => setDisableCountdown(3);
  const handleCancelFreedriveCountdown = () => setDisableCountdown(0);

  // ── 렌더 ────────────────────────────────────────────────────────────

  const modeColor: Record<string, string> = {
    TEACH_STARTING: "#2563eb",
    TEACH_RECORDING: "#d97706",
    TEACH_STOPPING: "#7c3aed",
    REPLAY_RECORDING: "#dc2626",
    REVIEW: "#6366f1",
    SAVED: "#22c55e",
    DISCARDED: "#6b7280",
    RETURN_HOME: "#0ea5e9",
    DEBUG_RETURN: "#f59e0b",
    DEBUG_REPLAY: "#dc2626",
  };

  return (
    <div style={S.root}>
      {/* ── 상단 상태 바 ── */}
      <div style={S.topBar}>
        <span style={{ ...S.modeBadge, color: modeColor[mode] ?? "#f1f5f9" }}>
          {mode}
        </span>
        <span style={S.episodeLabel}>{logger.episode_id ?? "—"}</span>
        <span style={S.connDot} title={robot.connected ? "Connected" : "Disconnected"}>
          {robot.connected ? "🟢" : "🔴"}
        </span>
      </div>
      <SafetyBanner robot={robot} mobile />

      {/* Same-origin URLs keep the streams working on both HTTP :5002 and HTTPS :5003. */}
      <MobileCameraGrid mode={mode} camera={camera} />

      {/* ── 메인 컨텐츠 ── */}
      <div style={S.content}>
        {/* IDLE */}
        {mode === "IDLE" && (
          <MSection title="Setup">
            <MBtn label="Connect System" onClick={() => post("/api/connect")} primary />
          </MSection>
        )}

        {/* CONNECTING */}
        {mode === "CONNECTING" && (
          <MSection title="Connecting...">
            <div style={S.hint}>Initializing robot and camera…</div>
          </MSection>
        )}

        {/* READY */}
        {mode === "READY" && (
          <MSection title="System Ready">
            <MBtn label="Confirm Ready Pose" onClick={() => post("/api/ready")} primary disabled={!can("confirm_ready_pose")} />
            <MBtn label="Calibrate Joint Range" onClick={() => post("/api/robot/calibrate/start")} disabled={!can("calibrate")} />
            <GripperRow can={can} gripperRad={robot.gripper} />
            <FreedriveSetting motorsFree={motorsFree} disableCountdown={disableCountdown}
              onStartCountdown={handleStartFreedriveCountdown}
              onCancelCountdown={handleCancelFreedriveCountdown}
              onEnableMotors={handleEnableMotors} />
          </MSection>
        )}

        {/* CALIBRATING */}
        {mode === "CALIBRATING" && (
          <MSection title="Calibrating…">
            <div style={{ ...S.badge, background: "#052e16", color: "#4ade80" }}>
              {calibrationHeld ? "ARM IS HELD — resume to continue" : "ARM IS FREE — guide through full range"}
            </div>
            <div style={S.hint}>
              {(robot.calibration_span ?? []).map((span, i) =>
                `J${i + 1}: ${span.toFixed(3)}${robot.calibration_valid?.[i] ? " ✓" : " ✗"}`
              ).join(" · ")}
            </div>
            <MBtn label="Finish Calibration" onClick={() => post("/api/robot/calibrate/stop")}
              primary large disabled={!can("stop_calibration") || !calibrationReady} />
            {calibrationHeld ? (
              <MBtn label="Resume Calibration (Release Arm)"
                onClick={() => post("/api/robot/calibrate/resume")} primary />
            ) : (
              <MBtn label="Hold Position" onClick={() => post("/api/hold")}
                danger disabled={!can("hold_position")} />
            )}
          </MSection>
        )}

        {/* TEACH_READY */}
        {mode === "TEACH_READY" && (
          <MSection title="Ready to Teach">
            {/* Teaching Mode 선택 */}
            <div style={S.sectionLabel}>Teaching Mode</div>
            <div style={S.row}>
              <MBtn
                label="Freedrive"
                onClick={() => setTeachMode("freedrive")}
                active={teachMode === "freedrive"}
                style={{ flex: 1 }}
              />
              <MBtn
                label="Cartesian Jog"
                onClick={() => setTeachMode("phone_mouse")}
                active={teachMode === "phone_mouse"}
                warning
                style={{ flex: 1 }}
              />
            </div>
            <MBtn label="Start Teaching" onClick={handleStartTeach} primary disabled={!can("start_teach")} />
            <MBtn label="Calibrate Joint Range" onClick={() => post("/api/robot/calibrate/start")} disabled={!can("calibrate")} />
            <GripperRow can={can} gripperRad={robot.gripper} />
            <FreedriveSetting motorsFree={motorsFree} disableCountdown={disableCountdown}
              onStartCountdown={handleStartFreedriveCountdown}
              onCancelCountdown={handleCancelFreedriveCountdown}
              onEnableMotors={handleEnableMotors} />
          </MSection>
        )}

        {/* TEACH_RECORDING */}
        {mode === "TEACH_STARTING" && (
          <MSection title="Preparing Freedrive…">
            <div style={S.hint}>Waiting for all six arm motors to report disabled. Do not guide the arm yet.</div>
          </MSection>
        )}

        {mode === "TEACH_RECORDING" && (
          <MSection title="Teaching in Progress">
            <GripperRow can={can} gripperRad={robot.gripper} />

            {teachMode === "phone_mouse" && pmActive ? (
              /* ── Cartesian Jog 모드 ── */
              <>
                <div style={S.gyroDisplay}>
                  γ(roll): {gyro.gamma.toFixed(1)}°&nbsp;&nbsp;
                  β(pitch): {gyro.beta.toFixed(1)}°
                  {pmDeadman && (
                    <span style={{ color: "#22c55e", marginLeft: 8 }}>● ACTIVE</span>
                  )}
                </div>
                <DeadmanButton active={pmDeadman} onActiveChange={setDeadman} />
              </>
            ) : (
              /* ── Freedrive 모드 ── */
              <FreedriveSetting
                motorsFree={motorsFree}
                disableCountdown={disableCountdown}
                onStartCountdown={handleStartFreedriveCountdown}
                onCancelCountdown={handleCancelFreedriveCountdown}
                onEnableMotors={handleEnableMotors}
              />
            )}

            <MBtn label="STOP TEACHING" onClick={handleStopTeach} primary large disabled={!can("stop_teach")} />
            <MBtn label="Hold Position" onClick={() => post("/api/hold")} danger disabled={!can("hold_position")} />
          </MSection>
        )}

        {mode === "TEACH_STOPPING" && (
          <MSection title="Stopping Teaching…">
            <div style={S.hint}>Re-arming all axes and saving the trajectory. Please wait.</div>
          </MSection>
        )}

        {/* TRAJECTORY_CHECK */}
        {mode === "TRAJECTORY_CHECK" && (
          <MSection title="Trajectory Recorded">
            <MBtn label="↩ Return (Backtrace)" onClick={() => post("/api/home")} primary disabled={!can("return_home")} />
            <MBtn label="⚡ Return Direct" onClick={() => post("/api/home/direct")} disabled={!can("return_home_direct")} />
            <MBtn label="Discard & Retake" onClick={() => post("/api/episodes/discard")} disabled={!can("discard_episode")} />
          </MSection>
        )}

        {/* RETURN_HOME */}
        {mode === "RETURN_HOME" && (
          <MSection title="Returning to Home…">
            <div style={S.hint}>Please wait…</div>
            <MBtn label="Hold Position" onClick={() => post("/api/hold")} danger disabled={!can("hold_position")} />
          </MSection>
        )}

        {/* REPLAY_READY */}
        {mode === "REPLAY_READY" && (
          <MSection title="Ready for Replay">
            <SpeedControl speedScale={speedScale} onChange={setSpeedScale} />
            {replayMessage && <div style={{ ...S.hint, color: "#fca5a5" }}>{replayMessage}</div>}
            <MBtn
              label={replayStarting ? "Starting…" : `Start Replay & Record (${Math.round(speedScale * 100)}%)`}
              onClick={handleStartReplay}
              primary
              disabled={!can("start_replay") || replayStarting}
            />
            <GripperRow can={can} gripperRad={robot.gripper} />
            <MBtn label="Hold Position" onClick={() => post("/api/hold")} danger disabled={!can("hold_position")} />
            <MBtn label="Discard Without Replay" onClick={() => post("/api/episodes/discard")}
              danger disabled={!can("discard_episode")} />
          </MSection>
        )}

        {/* REPLAY_RECORDING */}
        {mode === "REPLAY_RECORDING" && (
          <MSection title="Recording in Progress">
            <div style={{ ...S.hint, color: "#dc2626", fontWeight: 700 }}>
              Stay away from the robot's workspace.
            </div>
            <GripperRow can={can} gripperRad={robot.gripper} />
            <MBtn label="STOP REPLAY" onClick={() => post("/api/replay/stop")} primary large disabled={!can("stop_replay")} />
            <MBtn label="Hold Position" onClick={() => post("/api/hold")} danger disabled={!can("hold_position")} />
          </MSection>
        )}

        {mode === "DEBUG_RETURN" && (
          <MSection title="Dataset Debug Return">
            <div style={S.hint}>Moving to the selected saved trajectory start. Stay clear.</div>
            <MBtn label="Hold at Current Position" onClick={() => post("/api/hold")} danger large />
          </MSection>
        )}

        {mode === "DEBUG_REPLAY" && (
          <MSection title="Dataset Debug Replay">
            <div style={{ ...S.hint, color: "#fca5a5", fontWeight: 700 }}>
              Robot is moving without camera recording.
            </div>
            <MBtn label="STOP DEBUG REPLAY"
              onClick={() => post("/api/episodes/debug_replay/stop")} danger large />
            <MBtn label="Hold Position" onClick={() => post("/api/hold")} danger />
          </MSection>
        )}

        {/* REVIEW */}
        {mode === "REVIEW" && (
          <MSection title="Episode Review">
            <div style={S.hint}>Review on the main screen, then save or discard.</div>
            <MBtn
              label="✓ Save — Success"
              onClick={() => fetch("/api/episodes/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ success: true, reason: "" }),
              })}
              primary
              disabled={!can("save_episode")}
            />
            <MBtn
              label="✗ Save — Failure"
              onClick={() => fetch("/api/episodes/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ success: false, reason: "" }),
              })}
              disabled={!can("save_episode")}
            />
            <MBtn label="Discard & Retake (Replay)" onClick={() => post("/api/episodes/retake_replay")} disabled={!can("retake_replay")} />
            <MBtn label="Discard & Retake (Teach)" onClick={() => post("/api/episodes/retake_teach")} disabled={!can("retake_teach")} />
            <MBtn label="Discard" onClick={() => post("/api/episodes/discard")} danger disabled={!can("discard_episode")} />
          </MSection>
        )}

        {/* SAVED */}
        {mode === "SAVED" && (
          <MSection title="Saved">
            <div style={{ ...S.badge, background: "#052e16", color: "#4ade80", marginBottom: 16 }}>
              Episode saved successfully
            </div>
            <MBtn label="New Episode" onClick={() => post("/api/episodes/new")} primary disabled={!can("new_episode")} />
            <MBtn label="Add Take" onClick={() => post("/api/episodes/add_take")} disabled={!can("add_take")} />
          </MSection>
        )}

        {/* DISCARDED */}
        {mode === "DISCARDED" && (
          <MSection title="Discarded">
            <div style={{ ...S.badge, background: "#1c1917", color: "#94a3b8", marginBottom: 16 }}>
              Episode discarded
            </div>
            <MBtn label="New Episode" onClick={() => post("/api/episodes/new")} primary disabled={!can("new_episode")} />
          </MSection>
        )}

        {/* PROCESSING */}
        {mode === "PROCESSING" && (
          <MSection title="Processing…">
            <div style={S.hint}>Generating preview video…</div>
          </MSection>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────

function MobileCameraGrid({ mode, camera }: { mode: AppMode; camera: CameraState }) {
  const polarizerInfo = camera.cameras?.["polarizer"];
  const hasPolarizer = polarizerInfo?.type === "spinnaker";
  const [polarizerExpanded, setPolarizerExpanded] = useState(false);
  const [streamEpoch, setStreamEpoch] = useState(0);
  const [pageActive, setPageActive] = useState(() => !document.hidden);
  const lastRestartRef = useRef(0);

  const restartStreams = useCallback((force = false) => {
    if (document.hidden) return;
    setPageActive(true);
    const now = Date.now();
    // visibilitychange/pageshow/focus often arrive as one burst.
    if (!force && now - lastRestartRef.current < 500) return;
    lastRestartRef.current = now;
    setStreamEpoch(value => value + 1);
  }, []);

  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) setPageActive(false);
      else restartStreams();
    };
    const onPageHide = () => setPageActive(false);
    const onPageShow = () => restartStreams();
    const onFocus = () => restartStreams();
    const onOnline = () => restartStreams(true);

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pagehide", onPageHide);
    window.addEventListener("pageshow", onPageShow);
    window.addEventListener("focus", onFocus);
    window.addEventListener("online", onOnline);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pagehide", onPageHide);
      window.removeEventListener("pageshow", onPageShow);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("online", onOnline);
    };
  }, [restartStreams]);

  return (
    <div style={S.cameraPanel}>
      <div style={S.cameraPanelHeader}>
        <span>LIVE CAMERAS</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {(mode === "TEACH_RECORDING" || mode === "REPLAY_RECORDING") && (
            <span style={S.cameraRecBadge}>● REC</span>
          )}
          <button type="button" onClick={() => restartStreams(true)} style={S.cameraReconnectButton}>
            ↻ Reconnect
          </button>
        </div>
      </div>
      <div style={S.cameraGrid}>
        {hasPolarizer && (
          <MobileCamera
            cameraId="polarizer" label="POLARIZER" active={pageActive} streamEpoch={streamEpoch}
            mode="rgb" onExpand={() => setPolarizerExpanded(true)}
          />
        )}
        <MobileCamera cameraId="cam0" label="CAM0 · EGO" active={pageActive} streamEpoch={streamEpoch} />
        <MobileCamera cameraId="cam1" label="CAM1 · OVERVIEW" active={pageActive} streamEpoch={streamEpoch} />
      </div>
      {polarizerExpanded && <PolarizerExpandedModal onClose={() => setPolarizerExpanded(false)} />}
    </div>
  );
}

function PolarizerExpandedModal({ onClose }: { onClose: () => void }) {
  const [viewMode, setViewMode] = useState<"rgb" | "aolp" | "dolp" | "grid" | "stokes">("rgb");
  const [shotFlash, setShotFlash] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const streamUrl = `/stream/camera/polarizer?mode=${viewMode}`;

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    img.src = "";       // 이전 MJPEG 커넥션 확실히 종료
    img.src = streamUrl; // 재연결
  }, [streamUrl]);

  const handleShot = () => {
    fetch("/api/screenshots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: "polarizer", mode: viewMode }),
    }).then(() => {
      setShotFlash(true);
      setTimeout(() => setShotFlash(false), 900);
    });
  };

  return (
    <div style={S.polarizerModalRoot}>
      <img ref={imgRef} alt="Polarizer live" style={S.polarizerModalImg} />
      <button type="button" onClick={onClose} style={S.polarizerModalClose}>✕</button>
      <button type="button" onClick={handleShot} style={S.polarizerModalShot}>
        {shotFlash ? "✓ Saved" : "📷 Screenshot"}
      </button>
      <div style={S.polarizerModalModes}>
        {(["rgb", "aolp", "dolp", "grid", "stokes"] as const).map(m => (
          <button
            key={m}
            type="button"
            onClick={() => setViewMode(m)}
            style={{
              ...S.polarizerModalModeBtn,
              background: viewMode === m ? "#0891b2" : "rgba(15,23,42,0.7)",
            }}
          >
            {m === "grid" ? "2×2" : m === "stokes" ? "S1/S2" : m.toUpperCase()}
          </button>
        ))}
      </div>
    </div>
  );
}

function MobileCamera({
  cameraId, label, active, streamEpoch, mode: viewMode, onExpand,
}: {
  cameraId: string; label: string; active: boolean; streamEpoch: number; mode?: string; onExpand?: () => void;
}) {
  const [retry, setRetry] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [shotFlash, setShotFlash] = useState(false);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamUrl = viewMode
    ? `/stream/camera/${cameraId}?mode=${viewMode}&mobile=${streamEpoch}-${retry}`
    : `/stream/camera/${cameraId}?mobile=${streamEpoch}-${retry}`;

  const handleShot = () => {
    fetch("/api/screenshots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id: cameraId, mode: viewMode ?? "rgb" }),
    }).then(() => {
      setShotFlash(true);
      setTimeout(() => setShotFlash(false), 900);
    });
  };

  useEffect(() => {
    if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    setLoaded(false);
    setRetry(0);
    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
    };
  }, [active, streamEpoch]);

  return (
    <div style={S.cameraCard}>
      <div style={S.cameraLabel}>
        <span>{label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: loaded ? "#4ade80" : "#fbbf24" }}>
            {loaded ? "● LIVE" : "● WAIT"}
          </span>
          {onExpand && (
            <button type="button" onClick={onExpand} style={S.cameraExpandButton}>⤢</button>
          )}
        </div>
      </div>
      <div style={S.cameraFrame}>
        {!loaded && <div style={S.cameraPlaceholder}>
          {active ? "Reconnecting camera…" : "Camera paused in background"}
        </div>}
        {loaded && (
          <button type="button" onClick={handleShot} style={S.cameraShotButton}>
            {shotFlash ? "✓" : "📷"}
          </button>
        )}
        {active && (
          <img
            key={`${cameraId}-${streamEpoch}-${retry}`}
            src={streamUrl}
            alt={`${label} live stream`}
            draggable={false}
            style={{ ...S.cameraImage, opacity: loaded ? 1 : 0 }}
            onLoad={() => setLoaded(true)}
            onError={() => {
              setLoaded(false);
              if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
              retryTimerRef.current = setTimeout(() => setRetry(value => value + 1), 2000);
            }}
          />
        )}
      </div>
    </div>
  );
}

function MSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={S.sectionTitle}>{title}</div>
      {children}
    </div>
  );
}

const GRIPPER_OPEN_RAD_M = 0.07;
const GRIPPER_SNAPS_M = [0, 50, 70, 100];

function radToSnapPctM(rad: number): number {
  const raw = Math.round((rad / GRIPPER_OPEN_RAD_M) * 100);
  return GRIPPER_SNAPS_M.reduce((prev, curr) =>
    Math.abs(curr - raw) < Math.abs(prev - raw) ? curr : prev
  );
}

function GripperRow({ can, gripperRad }: { can: (a: string) => boolean; gripperRad?: number }) {
  const [pct, setPct] = React.useState(() =>
    gripperRad !== undefined ? radToSnapPctM(gripperRad) : 70
  );
  const canAny = can("open_gripper") || can("close_gripper") || can("gripper_set");

  React.useEffect(() => {
    if (gripperRad !== undefined) {
      setPct(radToSnapPctM(gripperRad));
    }
  }, [gripperRad]);

  const sendGripper = (val: number) => {
    if (val === 0) {
      fetch("/api/robot/gripper/close", { method: "POST" });
    } else {
      fetch("/api/robot/gripper/set", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pct: val }),
      });
    }
  };

  const handleChange = (val: number) => {
    setPct(val);
    sendGripper(val);
  };

  const SNAPS = GRIPPER_SNAPS_M;

  return (
    <div style={{ background: "#1e293b", borderRadius: 12, padding: "12px 14px", marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 10 }}>
        Gripper — {pct}%
      </div>
      {/* 슬라이더 */}
      <input
        type="range"
        min={0} max={100} step={5}
        value={pct}
        disabled={!canAny}
        onChange={e => {
          handleChange(Number(e.target.value));
        }}
        style={{ width: "100%", accentColor: "#22c55e", height: 32, cursor: canAny ? "pointer" : "default" }}
      />
      {/* 눈금 레이블 */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2 }}>
        {SNAPS.map(v => (
          <button key={v} type="button" disabled={!canAny} onClick={() => handleChange(v)} style={{
            border: 0, background: "transparent", padding: "4px", cursor: canAny ? "pointer" : "default",
            fontSize: 10, color: v === pct ? "#22c55e" : "#475569", fontWeight: v === pct ? 700 : 400,
          }}>
            {v}%
          </button>
        ))}
      </div>
    </div>
  );
}

function FreedriveSetting({
  motorsFree,
  disableCountdown,
  onStartCountdown,
  onCancelCountdown,
  onEnableMotors,
}: {
  motorsFree: boolean;
  disableCountdown: number;
  onStartCountdown: () => void;
  onCancelCountdown: () => void;
  onEnableMotors: () => void;
}) {
  return (
    <div style={S.freedriveBox}>
      <div style={S.sectionLabel}>Freedrive (Motor Off)</div>
      {motorsFree ? (
        <>
          <div style={{ ...S.badge, background: "#052e16", color: "#4ade80" }}>
            ARM IS FREE — guide by hand
          </div>
          <MBtn label="Re-enable Motors" onClick={onEnableMotors} primary />
        </>
      ) : disableCountdown > 0 ? (
        <>
          <div style={{ ...S.badge, background: "#7c2d12", color: "#fcd34d", fontSize: 18, fontWeight: 800 }}>
            Releasing in {disableCountdown}s…
          </div>
          <MBtn label="Cancel" onClick={onCancelCountdown} />
        </>
      ) : (
        <MBtn label="Release Motors (Freedrive)" onClick={onStartCountdown} />
      )}
    </div>
  );
}

const SPEED_PRESETS = [0.3, 0.5, 0.7, 1.0];

function SpeedControl({ speedScale, onChange }: { speedScale: number; onChange: (v: number) => void }) {
  return (
    <div style={{ background: "#1e293b", borderRadius: 12, padding: "12px 14px", marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, textTransform: "uppercase" as const, letterSpacing: 0.8, marginBottom: 10 }}>
        Replay Speed — {Math.round(speedScale * 100)}%
      </div>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        {SPEED_PRESETS.map(v => (
          <button
            key={v}
            onClick={() => onChange(v)}
            style={{
              flex: 1, padding: "10px 0", borderRadius: 8, border: "none",
              background: speedScale === v ? "#4f46e5" : "#0f172a",
              color: speedScale === v ? "#fff" : "#94a3b8",
              fontWeight: 700, fontSize: 13, cursor: "pointer",
              touchAction: "manipulation",
            }}
          >
            {Math.round(v * 100)}%
          </button>
        ))}
      </div>
    </div>
  );
}

function DeadmanButton({
  active,
  onActiveChange,
}: {
  active: boolean;
  onActiveChange: (v: boolean) => void;
}) {
  return (
    <button
      style={{
        ...S.deadman,
        background: active ? "#dc2626" : "#7f1d1d",
        boxShadow: active ? "0 0 0 4px #fca5a5" : "none",
      }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onActiveChange(true);
      }}
      onPointerUp={() => onActiveChange(false)}
      onPointerCancel={() => onActiveChange(false)}
    >
      {active ? "MOVING…" : "HOLD TO MOVE"}
    </button>
  );
}

function MBtn({
  label,
  onClick,
  primary,
  danger,
  warning,
  active,
  disabled,
  large,
  style,
}: {
  label: string;
  onClick: () => void;
  primary?: boolean;
  danger?: boolean;
  warning?: boolean;
  active?: boolean;
  disabled?: boolean;
  large?: boolean;
  style?: React.CSSProperties;
}) {
  const bg = disabled
    ? "#1e293b"
    : active
      ? (warning ? "#d97706" : "#4ade80")
      : danger
        ? "#7f1d1d"
        : primary
          ? "#4f46e5"
          : warning
            ? "#92400e"
            : "#1e293b";
  const color = disabled ? "#475569" : active ? "#fff" : danger ? "#fca5a5" : "#f1f5f9";

  return (
    <button
      style={{
        ...S.btn,
        background: bg,
        color,
        ...(large ? { padding: "20px 14px", fontSize: 18, fontWeight: 800 } : {}),
        ...style,
      }}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
}

// ── Styles ──────────────────────────────────────────────────────────────

const S: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100dvh",
    background: "#0f172a",
    color: "#f1f5f9",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    overflowX: "hidden",
  },
  topBar: {
    display: "flex",
    alignItems: "center",
    padding: "10px 16px",
    background: "#1e293b",
    borderBottom: "1px solid #334155",
    gap: 8,
    flexShrink: 0,
  },
  modeBadge: {
    fontWeight: 800,
    fontSize: 16,
    flex: 1,
    letterSpacing: 0.5,
  },
  episodeLabel: {
    fontSize: 11,
    color: "#64748b",
    fontFamily: "monospace",
  },
  connDot: {
    fontSize: 14,
    marginLeft: 4,
  },
  cameraPanel: {
    padding: "8px 10px 10px",
    background: "#111827",
    borderBottom: "1px solid #334155",
    flexShrink: 0,
  },
  cameraPanelHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    minHeight: 16,
    marginBottom: 6,
    color: "#64748b",
    fontSize: 10,
    fontWeight: 800,
    letterSpacing: 1,
  },
  cameraRecBadge: {
    color: "#f87171",
    fontSize: 10,
    letterSpacing: 0.5,
  },
  cameraReconnectButton: {
    border: "1px solid #475569",
    borderRadius: 6,
    background: "#1e293b",
    color: "#cbd5e1",
    padding: "3px 7px",
    fontSize: 9,
    fontWeight: 700,
    cursor: "pointer",
  },
  cameraGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: 8,
  },
  cameraCard: {
    minWidth: 0,
    overflow: "hidden",
    borderRadius: 9,
    border: "1px solid #334155",
    background: "#020617",
  },
  cameraLabel: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 4,
    padding: "5px 7px",
    color: "#cbd5e1",
    fontSize: 9,
    fontWeight: 700,
    whiteSpace: "nowrap",
  },
  cameraFrame: {
    position: "relative",
    width: "100%",
    aspectRatio: "4 / 3",
    overflow: "hidden",
    background: "#020617",
  },
  cameraImage: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    display: "block",
    objectFit: "contain",
    transition: "opacity 0.2s ease",
  },
  cameraExpandButton: {
    background: "transparent",
    color: "#94a3b8",
    border: "none",
    fontSize: 13,
    lineHeight: 1,
    padding: 2,
  } as React.CSSProperties,
  polarizerModalRoot: {
    position: "fixed",
    inset: 0,
    zIndex: 1000,
    background: "#000",
  } as React.CSSProperties,
  polarizerModalImg: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    display: "block",
  } as React.CSSProperties,
  polarizerModalClose: {
    position: "absolute",
    top: 12, left: 12, zIndex: 2,
    background: "rgba(15,23,42,0.7)",
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 8,
    width: 36, height: 36,
    fontSize: 16,
  } as React.CSSProperties,
  polarizerModalShot: {
    position: "absolute",
    top: 12, right: 12, zIndex: 2,
    background: "rgba(15,23,42,0.7)",
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 8,
    padding: "8px 14px",
    fontSize: 13,
    fontWeight: 700,
  } as React.CSSProperties,
  polarizerModalModes: {
    position: "absolute",
    bottom: 24, left: 0, right: 0,
    display: "flex",
    justifyContent: "center",
    gap: 8,
    zIndex: 2,
  } as React.CSSProperties,
  polarizerModalModeBtn: {
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 8,
    padding: "10px 18px",
    fontSize: 13,
    fontWeight: 700,
  } as React.CSSProperties,
  cameraShotButton: {
    position: "absolute",
    top: 4, right: 4, zIndex: 2,
    background: "rgba(2,6,23,0.65)",
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 5,
    padding: "3px 6px",
    fontSize: 12,
    lineHeight: 1,
  } as React.CSSProperties,
  cameraPlaceholder: {
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 6,
    color: "#475569",
    fontSize: 9,
    textAlign: "center",
  },
  content: {
    flex: 1,
    overflowY: "auto",
    padding: 16,
  },
  sectionTitle: {
    fontWeight: 700,
    fontSize: 13,
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 12,
    marginTop: 4,
  },
  sectionLabel: {
    fontSize: 11,
    color: "#64748b",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 6,
    marginTop: 12,
  },
  hint: {
    fontSize: 13,
    color: "#64748b",
    marginBottom: 12,
    lineHeight: 1.5,
  },
  badge: {
    borderRadius: 10,
    padding: "10px 14px",
    fontSize: 14,
    fontWeight: 700,
    textAlign: "center",
    marginBottom: 8,
  },
  row: {
    display: "flex",
    gap: 8,
    marginBottom: 8,
  },
  btn: {
    display: "block",
    width: "100%",
    padding: "14px 14px",
    borderRadius: 12,
    border: "none",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
    marginBottom: 8,
    textAlign: "center",
    touchAction: "manipulation",
    WebkitTapHighlightColor: "transparent",
  },
  freedriveBox: {
    background: "#1e293b",
    borderRadius: 12,
    padding: "12px 14px",
    marginBottom: 8,
    marginTop: 4,
  },
  gyroDisplay: {
    background: "#1e3a5f",
    borderRadius: 10,
    padding: "10px 14px",
    fontFamily: "monospace",
    fontSize: 13,
    color: "#93c5fd",
    marginBottom: 10,
    textAlign: "center",
  },
  deadman: {
    display: "block",
    width: "100%",
    padding: "40px 14px",
    borderRadius: 16,
    border: "none",
    cursor: "pointer",
    fontSize: 22,
    fontWeight: 800,
    color: "#fff",
    marginBottom: 10,
    touchAction: "none",
    userSelect: "none",
    WebkitUserSelect: "none",
    transition: "background 0.1s, box-shadow 0.1s",
  },
};
