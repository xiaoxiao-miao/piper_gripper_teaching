/**
 * CapturePage — 카메라 중심 3-row 레이아웃
 *
 * Row A: Main Camera  — 선택된 카메라 대형 + 셀렉터 탭 + 썸네일 3개
 * Row B: Step Card    — 현재 단계별 가이드 + 액션 버튼 (좌: 정보, 우: 버튼)
 * Row C: Event Log    — 최근 이벤트 스트립 (최대 5줄)
 */
import React, { useState, useEffect, useMemo, useRef } from "react";
import ReviewEpisodeModal from "../components/ReviewEpisodeModal";
import type { AppMode, RobotState, CameraState, EventLog, SaveStep } from "../types";

interface Props {
  mode: AppMode;
  availableActions: string[];
  robot: RobotState;
  camera: CameraState;
  events: EventLog[];
  episodeId?: string;
  saveSteps?: SaveStep[];
}

// Flask HTTP stream (webpack 프록시는 MJPEG multipart를 버퍼링해 끊길 수 있음)
const FLASK_HTTP_ORIGIN = `http://${window.location.hostname}:5002`;

type CamId = string;

const TEACH_DISABLE_DELAY = 5; // seconds

const fmtUs = (us?: number) => {
  if (us === undefined || us === null || Number.isNaN(us)) return "—";
  return us >= 1000 ? `${(us / 1000).toFixed(1)}ms` : `${Math.round(us)}µs`;
};

const post = (path: string, body?: object) =>
  fetch(path, {
    method: "POST",
    ...(body
      ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : {}),
  });

// ── Step descriptor map ────────────────────────────────────────────────────
const STEP_INFO: Record<AppMode, { badge: string; title: string; guide: string }> = {
  IDLE:             { badge: "1 · Ready Pose", title: "Connect System",       guide: "Check all connections, then connect the robot and cameras." },
  CONNECTING:       { badge: "1 · Ready Pose", title: "Connecting…",          guide: "Initializing robot and camera. Please wait." },
  READY:            { badge: "1 · Ready Pose", title: "System Ready",         guide: "Confirm the current pose as the teaching start position." },
  CALIBRATING:      { badge: "Calibration", title: "Joint Range Calibration", guide: "Move each joint to its min and max safe position, then finish." },
  TEACH_READY:      { badge: "2 · Teaching", title: "Ready to Teach",         guide: "Describe the task, then guide the arm along the desired path." },
  TEACH_STARTING:   { badge: "2 · Teaching", title: "Preparing Freedrive",    guide: "Waiting for verified six-axis motor release." },
  TEACH_RECORDING:  { badge: "2 · Teaching", title: "Teaching in Progress",   guide: "Motors release in 5 s. Guide the arm along the path." },
  TEACH_STOPPING:   { badge: "2 · Teaching", title: "Stopping Teaching",      guide: "Re-arming all axes and preserving the recorded trajectory." },
  TRAJECTORY_CHECK: { badge: "3 · Return Home", title: "Trajectory Recorded", guide: "Return to start position before replay recording." },
  RETURN_HOME:      { badge: "3 · Return Home", title: "Returning Home",      guide: "Robot is moving to the start position. Stay clear." },
  REPLAY_READY:     { badge: "4 · Replay & Record", title: "Ready to Record", guide: "Set replay speed, then start the capture." },
  REPLAY_RECORDING: { badge: "4 · Replay & Record", title: "Recording in Progress", guide: "Stay away from the robot's workspace." },
  PROCESSING:       { badge: "4 · Replay & Record", title: "Encoding Videos…", guide: "Replay done. Encoding and saving data, please wait." },
  REVIEW:           { badge: "5 · Review",  title: "Episode Complete",        guide: "Review the new take, then save, retake, or discard it." },
  SAVED:            { badge: "5 · Review",  title: "Episode Saved",           guide: "Start a new episode or add another take." },
  DISCARDED:        { badge: "5 · Review",  title: "Episode Discarded",       guide: "Start a new episode to continue." },
  DEBUG_RETURN:     { badge: "Dataset Debug", title: "Returning to Debug Start", guide: "Robot is moving to the selected saved trajectory start." },
  DEBUG_REPLAY:     { badge: "Dataset Debug", title: "Debug Replay Running", guide: "No cameras are recording. Results are saved under debug_replays." },
};

// ─────────────────────────────────────────────────────────────────────────────
export default function CapturePage({
  mode, availableActions, robot, camera, events, episodeId, saveSteps = [],
}: Props) {
  const [selectedCam, setSelectedCam] = useState<CamId>("polarizer");  // 디버깅 편의상 기본 선택
  const [polarizerMode, setPolarizerMode] = useState<"rgb" | "aolp" | "dolp" | "grid" | "stokes">("rgb");
  const [autoExposure, setAutoExposureUI] = useState(true);
  const [exposureEv, setExposureEv] = useState(0);
  const [draggingExposure, setDraggingExposure] = useState(false);
  const [task, setTask] = useState("pick red block and place in bowl");
  const [speedScale, setSpeedScale] = useState(0.5);
  const [confirmReplay, setConfirmReplay] = useState(false);
  const [replayStarting, setReplayStarting] = useState(false);
  const [replayMessage, setReplayMessage] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [reviewDismissed, setReviewDismissed] = useState(false);
  const [calibLimits, setCalibLimits] = useState<Record<
    string, { min: number; max: number; measured?: boolean }
  > | null>(null);

  useEffect(() => { setIsSaving(false); }, [mode]);
  useEffect(() => {
    if (mode === "REVIEW" || mode === "PROCESSING") setReviewDismissed(false);
    if (mode !== "REPLAY_READY") setReplayMessage("");
  }, [mode]);

  const can = (a: string) => availableActions.includes(a);

  const handleConnect     = () => post("/api/connect");
  const handleReady       = () => post("/api/ready");
  const handleStartTeach  = () => post("/api/teach/start", { task });
  const handleStopTeach   = () => { setIsSaving(true); post("/api/teach/stop"); };
  const handleReturnHome       = () => post("/api/home");
  const handleReturnHomeDirect = () => post("/api/home/direct");
  const handleStartReplay = async () => {
    setConfirmReplay(false);
    setReplayStarting(true);
    setReplayMessage("");
    try {
      const response = await post("/api/replay/start", { speed_scale: speedScale });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.ok === false) {
        const recommended = Number(data.recommended_max_speed);
        if (Number.isFinite(recommended) && recommended > 0) {
          setSpeedScale(recommended);
        }
        setReplayMessage(data.reason || `Replay start failed (${response.status})`);
      } else if (data.speed_adjusted) {
        setReplayMessage(
          `Safety limit applied: ${Math.round(Number(data.speed_scale_applied) * 100)}%`,
        );
      }
    } catch {
      setReplayMessage("Replay start request failed. Check the backend connection.");
    } finally {
      setReplayStarting(false);
    }
  };
  const handleStopReplay  = () => { setIsSaving(true); post("/api/replay/stop"); };
  const handleStopDebugReplay = () => post("/api/episodes/debug_replay/stop");
  const handleHold        = () => post("/api/hold");
  const handleResumeCalibration = () => post("/api/robot/calibrate/resume");
  const handleGripSet     = (pct: number) => post("/api/robot/gripper/set", { pct });
  const handleGripClose   = () => post("/api/robot/gripper/close");
  const handleCalibStart  = () => post("/api/robot/calibrate/start");
  const handleCalibStop   = () =>
    post("/api/robot/calibrate/stop")
      .then(r => r.json())
      .then((d: any) => { if (d.ok) setCalibLimits(d.limits); });
  const handleDiscard     = () => post("/api/episodes/discard");
  const handleNewEpisode  = () => post("/api/episodes/new");
  const handleAddTake     = () => post("/api/episodes/add_take");
  const handleOpenReview  = () => setReviewDismissed(false);
  const handleSpeedChange = (v: number) => {
    setSpeedScale(v);
    post("/api/replay/speed", { speed_scale: v });
  };

  const recentEvents = useMemo(() => [...events].reverse().slice(0, 5), [events]);
  const calibrationReady =
    robot.calibration_valid?.length === 6 && robot.calibration_valid.every(Boolean);
  const calibrationHeld = Boolean(robot.motor_enable_status?.some(Boolean));

  const showStream = mode !== "IDLE" && mode !== "CONNECTING";
  const polarizerInfo = camera.cameras?.[selectedCam];
  const isPolarizerCam = polarizerInfo?.type === "spinnaker";
  const mainUrl    = isPolarizerCam
    ? `${FLASK_HTTP_ORIGIN}/stream/camera/${selectedCam}?mode=${polarizerMode}`
    : `${FLASK_HTTP_ORIGIN}/stream/camera/${selectedCam}`;
  const info       = STEP_INFO[mode];

  // MJPEG(multipart/x-mixed-replace)는 <img key>로 리마운트해도 브라우저가 이전
  // 커넥션을 안 닫는 경우가 있어 origin당 6개 연결 제한에 걸려 다른 API 요청까지
  // 막히는 문제가 있었음 — 같은 <img> 노드를 재사용하며 src=""로 확실히 끊고 재연결.
  const mainImgRef = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const img = mainImgRef.current;
    if (!img || !showStream) return;
    img.src = "";
    img.src = mainUrl;
  }, [mainUrl, showStream]);

  // ── Polarizer exposure control (EV 스케일 슬라이더) ──────────────────
  // 하드웨어 최대(최대 30초)까지 열어뒀더니 AcquisitionFrameRate와 정면 충돌해
  // 프레임이 깨지는 사고가 있었음 — 15fps 힌트의 4배로 다시 제한 (힌트 자체보다는
  // 훨씬 넓게 다룰 수 있으면서, 초 단위로 폭주하진 않게).
  const expMinUs = polarizerInfo?.exposure_us_min ?? 20;
  const expHintUs = polarizerInfo?.exposure_us_max_hint_for_target_fps;
  const expMaxUs = Math.max(
    expMinUs * 2,
    Math.min(polarizerInfo?.exposure_us_max_hw ?? Infinity, (expHintUs ?? expMinUs * 2) * 4)
  );
  const evMax = Math.log2(expMaxUs / expMinUs);
  const previewExposureUs = Math.round(expMinUs * Math.pow(2, exposureEv));

  // 카메라당 한 번만 서버 값으로 동기화 — 매 폴링(10Hz)마다 하면 사용자가 막
  // 조작한 로컬 상태를 곧바로 덮어써서 "버튼이 반응 안 하는" 것처럼 보임.
  const syncedExposureCamRef = React.useRef<string | null>(null);
  useEffect(() => {
    if (!isPolarizerCam || draggingExposure) return;
    if (syncedExposureCamRef.current === selectedCam) return;
    const curUs = polarizerInfo?.exposure_us;
    if (curUs === undefined) return; // camera_state 아직 안 옴 — 다음 폴링 때 재시도
    if (polarizerInfo?.exposure_auto) setAutoExposureUI(polarizerInfo.exposure_auto !== "Off");
    if (expMinUs > 0 && curUs > 0) setExposureEv(Math.log2(curUs / expMinUs));
    syncedExposureCamRef.current = selectedCam;
  }, [selectedCam, isPolarizerCam, polarizerInfo?.exposure_us, polarizerInfo?.exposure_auto, draggingExposure, expMinUs]);

  const putCameraControls = (camId: string, body: object) =>
    fetch(`/api/cameras/${camId}/controls`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  const handleAutoExposureToggle = () => {
    const next = !autoExposure;
    setAutoExposureUI(next);
    putCameraControls(selectedCam, { exposure_auto: next });
  };

  const commitExposureEv = (ev: number) => {
    putCameraControls(selectedCam, { exposure_us: Math.round(expMinUs * Math.pow(2, ev)) });
  };

  // ── Screenshot ────────────────────────────────────────────────────
  const [screenshotFlash, setScreenshotFlash] = useState(false);
  const handleScreenshot = () => {
    const body: { cam_id: string; mode?: string } = { cam_id: selectedCam };
    if (isPolarizerCam) body.mode = polarizerMode;
    fetch("/api/screenshots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(() => {
      setScreenshotFlash(true);
      setTimeout(() => setScreenshotFlash(false), 900);
    });
  };
  const cameraTabs = useMemo(() => {
    const cams = Object.values(camera.cameras ?? {}).flatMap(c => {
      if (c.enabled === false) return [];
      const baseId = c.id ?? c.name ?? "";
      if (!baseId) return [];
      const label = c.label ?? c.name ?? c.id ?? "Camera";
      const streams = c.streams ?? ["color"];
      const tabs: Array<{ id: string; label: string; info: typeof c; stream: string }> = [];
      if (streams.includes("color")) tabs.push({ id: baseId, label, info: c, stream: "color" });
      if (streams.includes("depth")) tabs.push({ id: `${baseId}_depth`, label: `${label} Depth`, info: c, stream: "depth" });
      return tabs;
    });
    if (!cams.length) {
      return [
        { id: "polarizer", label: "Polarizer", info: undefined, stream: "color" },
        { id: "cam0", label: "Cam0", info: undefined, stream: "color" },
        { id: "cam1", label: "Cam1", info: undefined, stream: "color" },
        { id: "realsense", label: "RealSense", info: undefined, stream: "color" },
        { id: "realsense_depth", label: "RealSense Depth", info: undefined, stream: "depth" },
      ];
    }
    return cams;
  }, [camera.cameras]);

  useEffect(() => {
    if (!cameraTabs.some(c => c.id === selectedCam)) {
      setSelectedCam(cameraTabs[0]?.id ?? "cam0");
    }
  }, [cameraTabs, selectedCam]);

  return (
    <div style={styles.root}>
      {mode === "PROCESSING" && (
        <ReviewEpisodeModal
          mode={mode}
          availableActions={availableActions}
          episodeId={episodeId}
          saveSteps={saveSteps}
        />
      )}
      {mode === "REVIEW" && !reviewDismissed && (
        <ReviewEpisodeModal
          mode={mode}
          availableActions={availableActions}
          episodeId={episodeId}
          saveSteps={saveSteps}
          onClose={() => setReviewDismissed(true)}
        />
      )}

      {/* Saving overlay */}
      {isSaving && (
        <div style={styles.overlay}>
          <div style={styles.overlayBox}>
            <div style={styles.spinner} />
            <span style={{ color: "#94a3b8", fontSize: 15, fontWeight: 600 }}>Saving…</span>
          </div>
        </div>
      )}

      {/* ── Row A: Main Camera ──────────────────────────────────────── */}
      <div style={styles.cameraArea}>
        {showStream ? (
          <>
            <img
              ref={mainImgRef}
              alt="Camera"
              style={styles.mainImg}
              onError={() => {
                setTimeout(() => {
                  const img = mainImgRef.current;
                  if (!img) return;
                  img.src = "";       // 이전 커넥션 확실히 종료 후
                  img.src = mainUrl;  // 재연결
                }, 3000);
              }}
            />
            {mode === "REPLAY_RECORDING" && <div style={styles.recBadge}>● REC</div>}
            {mode === "TEACH_RECORDING"  && <div style={styles.teachBadge}>TEACH</div>}
            <button onClick={handleScreenshot} style={styles.screenshotBtn}>
              {screenshotFlash ? "✓ Saved" : "📷 Screenshot"}
            </button>
            {isPolarizerCam && (
              <div style={styles.polarizerModeOverlay}>
                <div style={styles.polarizerInfoBox}>
                  <div>{polarizerInfo?.resolution ?? "—"}</div>
                  <div>{polarizerInfo?.fps ?? "—"} fps</div>
                  <div>{polarizerInfo?.pixel_format ?? "—"}</div>
                  <div>shutter {fmtUs(polarizerInfo?.exposure_us)}</div>
                  <div style={{ opacity: 0.65, fontSize: 10 }}>
                    min {fmtUs(polarizerInfo?.exposure_us_min)} / max {fmtUs(polarizerInfo?.exposure_us_max_hw)}
                  </div>
                </div>

                <div style={styles.polarizerInfoBox}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                    <input type="checkbox" checked={autoExposure} onChange={handleAutoExposureToggle} />
                    Auto Exposure
                  </label>
                  {!autoExposure && (
                    <div style={{ marginTop: 6 }}>
                      <input
                        type="range"
                        min={0}
                        max={evMax}
                        step={0.05}
                        value={exposureEv}
                        onMouseDown={() => setDraggingExposure(true)}
                        onTouchStart={() => setDraggingExposure(true)}
                        onChange={e => setExposureEv(parseFloat(e.target.value))}
                        onMouseUp={e => {
                          setDraggingExposure(false);
                          commitExposureEv(parseFloat((e.target as HTMLInputElement).value));
                        }}
                        onTouchEnd={e => {
                          setDraggingExposure(false);
                          commitExposureEv(parseFloat((e.target as HTMLInputElement).value));
                        }}
                        style={{ width: "100%" }}
                      />
                      <div style={{ textAlign: "center" }}>{fmtUs(previewExposureUs)}</div>
                      {expHintUs !== undefined && (
                        <div style={{ fontSize: 9, opacity: 0.6, textAlign: "center" }}>
                          15fps 안전 한계 ≈ {fmtUs(expHintUs)} (실측 아님, 위 fps로 실제 확인)
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {([
                  { id: "rgb", label: "RGB" },
                  { id: "aolp", label: "AoLP" },
                  { id: "dolp", label: "DoLP" },
                  { id: "grid", label: "2×2 Grid" },
                  { id: "stokes", label: "S1/S2" },
                ] as const).map(m => (
                  <button
                    key={m.id}
                    onClick={() => setPolarizerMode(m.id)}
                    style={{
                      ...styles.polarizerModeBtn,
                      background: polarizerMode === m.id ? "#0891b2" : "rgba(15,23,42,0.6)",
                      border: polarizerMode === m.id ? "1px solid #22d3ee" : "1px solid rgba(255,255,255,0.25)",
                    }}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            )}
          </>
        ) : (
          <div style={styles.placeholder}>Camera stream will appear after connecting</div>
        )}
      </div>

      {/* ── Camera tab row (below the card) ────────────────────────── */}
      <div style={styles.cameraTabs}>
        {cameraTabs.map(c => {
          const camInfo = c.info;
          const isStreaming = camInfo?.streaming ?? false;
          const fps = camInfo?.fps;
          const isActive = selectedCam === c.id;
          return (
            <button
              key={c.id}
              onClick={() => setSelectedCam(c.id)}
              style={{
                ...styles.camTab,
                background: isActive ? "#6366f1" : "#f1f5f9",
                color: isActive ? "#fff" : "#374151",
                border: isActive ? "1px solid #6366f1" : "1px solid #e2e8f0",
              }}
            >
              <span style={{
                width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
                background: isStreaming ? "#22c55e" : "#9ca3af",
                display: "inline-block",
              }} />
              {c.label}
              {c.stream === "depth" && (
                <span style={{ fontSize: 10, opacity: 0.75 }}>depth</span>
              )}
              {fps !== undefined && (
                <span style={{ fontSize: 10, opacity: 0.7 }}>{fps}fps</span>
              )}
            </button>
          );
        })}
      </div>

      {/* ── Row B: Step Card ────────────────────────────────────────── */}
      <div style={styles.stepCard}>
        <EpisodeFlowStepper mode={mode} />

        {/* Left: step info + optional extra input */}
        <div style={styles.stepLeft}>
          <span style={styles.stepBadge}>{info.badge}</span>
          <div style={styles.stepTitle}>{info.title}</div>
          <div style={styles.stepGuide}>{info.guide}</div>

          {/* TEACH_READY: task input */}
          {mode === "TEACH_READY" && (
            <input
              style={styles.taskInput}
              value={task}
              onChange={e => setTask(e.target.value)}
              placeholder="Describe the task…"
            />
          )}

          {/* REPLAY_READY: speed slider */}
          {mode === "REPLAY_READY" && (
            <div style={{ marginTop: 8 }}>
              <div style={styles.sliderLabel}>
                Speed: <strong>{Math.round(speedScale * 100)}%</strong>
              </div>
              <input
                type="range" min={0.1} max={2.0} step={0.1} value={speedScale}
                onChange={e => handleSpeedChange(parseFloat(e.target.value))}
                style={{ width: "100%", marginBottom: 2 }}
              />
              <div style={styles.sliderHint}>30% recommended · 100% = as recorded · 200% = 2× speed</div>
            </div>
          )}

          {/* TEACH_RECORDING: countdown / free indicator */}
          {mode === "TEACH_STARTING" && <TeachIndicator />}

          {/* REPLAY_RECORDING: progress circle */}
          {(mode === "REPLAY_RECORDING" || mode === "DEBUG_REPLAY") && (
            <ReplayProgressCircle progress={robot.replay_progress ?? 0} />
          )}
        </div>

        {/* Right: action buttons */}
        <div style={styles.stepRight}>
          {mode === "IDLE" && (
            <Btn label="Connect System" onClick={handleConnect} primary />
          )}

          {mode === "READY" && (
            <>
              <Btn label="Confirm Ready Pose" onClick={handleReady} primary disabled={!can("confirm_ready_pose")} />
              <Btn label="Calibrate Joint Range" onClick={handleCalibStart} disabled={!can("calibrate")} />
              <GripperRow onSet={handleGripSet} onClose={handleGripClose} can={can} gripperRad={robot.gripper} />
              {calibLimits && <CalibResultPanel limits={calibLimits} />}
            </>
          )}

          {mode === "CALIBRATING" && (
            <>
              <div style={styles.freeAlert}>
                {calibrationHeld ? "⏸ ARM IS HELD — resume to continue" : "✋ ARM IS FREE — guide freely"}
              </div>
              <div style={{ fontSize: 11, fontFamily: "monospace", color: "#64748b" }}>
                {(robot.calibration_span ?? []).map((span, i) =>
                  `J${i + 1}: ${span.toFixed(3)}${robot.calibration_valid?.[i] ? " ✓" : " ✗"}`
                ).join(" · ")}
              </div>
              <Btn label="Finish Calibration" onClick={handleCalibStop} primary
                disabled={!can("stop_calibration") || !calibrationReady} />
            </>
          )}

          {mode === "TEACH_READY" && (
            <>
              <Btn label="Start Teaching" onClick={handleStartTeach} primary disabled={!can("start_teach") || !task.trim()} />
              <Btn label="Calibrate Joint Range" onClick={handleCalibStart} disabled={!can("calibrate")} />
              <GripperRow onSet={handleGripSet} onClose={handleGripClose} can={can} gripperRad={robot.gripper} />
              {calibLimits && <CalibResultPanel limits={calibLimits} />}
            </>
          )}

          {mode === "TEACH_RECORDING" && (
            <>
              <GripperRow onSet={handleGripSet} onClose={handleGripClose} can={can} gripperRad={robot.gripper} />
              <Btn label="Stop Teaching" onClick={handleStopTeach} primary disabled={!can("stop_teach")} />
            </>
          )}

          {mode === "TEACH_STOPPING" && (
            <div style={styles.replayMessage}>Re-arming robot and saving teaching data…</div>
          )}

          {mode === "TRAJECTORY_CHECK" && (
            <>
              <Btn label="↩ Return (Backtrace)" onClick={handleReturnHome} primary disabled={!can("return_home")} />
              <Btn label="⚡ Return Direct" onClick={handleReturnHomeDirect} disabled={!can("return_home_direct")} />
              <Btn label="Discard & Retake" onClick={handleDiscard} disabled={!can("discard_episode")} />
            </>
          )}

          {mode === "REPLAY_READY" && (
            <>
              {replayMessage && (
                <div style={styles.replayMessage}>{replayMessage}</div>
              )}
              {!confirmReplay ? (
                <Btn
                  label="Start Replay & Record"
                  onClick={() => setConfirmReplay(true)}
                  primary
                  disabled={!can("start_replay") || replayStarting}
                />
              ) : (
                <div style={styles.confirmBox}>
                  <div style={{ fontSize: 12, color: "#92400e", marginBottom: 8 }}>
                    Robot will move. Stand clear of the workspace.
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <Btn label="Cancel" onClick={() => setConfirmReplay(false)} />
                    <Btn label={replayStarting ? "Starting…" : "Confirm — Start"} onClick={handleStartReplay} primary disabled={replayStarting} />
                  </div>
                </div>
              )}
              <GripperRow onSet={handleGripSet} onClose={handleGripClose} can={can} gripperRad={robot.gripper} />
              <Btn label="Discard & Retake" onClick={handleDiscard} disabled={!can("discard_episode")} />
            </>
          )}

          {mode === "REPLAY_RECORDING" && (
            <>
              <GripperRow onSet={handleGripSet} onClose={handleGripClose} can={can} gripperRad={robot.gripper} />
              <Btn label="Stop Recording" onClick={handleStopReplay} primary disabled={!can("stop_replay")} />
            </>
          )}

          {mode === "DEBUG_RETURN" && (
            <Btn label="Stop at Current Position" onClick={handleHold} danger />
          )}

          {mode === "DEBUG_REPLAY" && (
            <Btn label="Stop Debug Replay" onClick={handleStopDebugReplay} danger />
          )}

          {(mode === "REVIEW" || mode === "SAVED" || mode === "DISCARDED") && (
            <>
              {mode === "REVIEW" && (
                <Btn label="Open Review" onClick={handleOpenReview} primary disabled={!can("save_episode")} />
              )}
              {mode === "SAVED" && (
                <>
                  <Btn label="New Episode" onClick={handleNewEpisode} primary disabled={!can("new_episode")} />
                  <Btn label="Add Another Take" onClick={handleAddTake} disabled={!can("add_take")} />
                </>
              )}
              {mode === "DISCARDED" && (
                <Btn label="New Episode" onClick={handleNewEpisode} primary disabled={!can("new_episode")} />
              )}
            </>
          )}

          {/* Advanced panel (freedrive) for relevant states */}
          {(mode === "READY" || mode === "TEACH_READY" || mode === "REPLAY_READY") && (
            <AdvancedPanel onHold={handleHold} />
          )}

          {/* Safety zone — hold always accessible when available */}
          {can("hold_position") && (
            <div style={styles.safetyZone}>
              {mode === "CALIBRATING" && calibrationHeld ? (
                <Btn label="▶ Resume Calibration (Release Arm)"
                  onClick={handleResumeCalibration} size="sm" />
              ) : (
                <Btn label="⚠ Hold" onClick={handleHold} danger size="sm" />
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Row C: Event Log ────────────────────────────────────────── */}
      <div style={styles.eventLog}>
        {recentEvents.map((e, i) => (
          <div key={i} style={styles.eventRow}>
            <span style={styles.eventTime}>{new Date(e.t * 1000).toLocaleTimeString()}</span>
            <span style={styles.eventMsg}>{e.message}</span>
          </div>
        ))}
        {events.length === 0 && (
          <span style={{ color: "#475569", fontSize: 11 }}>No events yet.</span>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

const FLOW_STEPS: { label: string; modes: AppMode[] }[] = [
  { label: "Ready Pose", modes: ["IDLE", "CONNECTING", "READY", "CALIBRATING"] },
  { label: "Teaching", modes: ["TEACH_READY", "TEACH_STARTING", "TEACH_RECORDING", "TEACH_STOPPING"] },
  { label: "Return Home", modes: ["TRAJECTORY_CHECK", "RETURN_HOME"] },
  { label: "Replay & Record", modes: ["REPLAY_READY", "REPLAY_RECORDING", "PROCESSING", "DEBUG_RETURN", "DEBUG_REPLAY"] },
  { label: "Review", modes: ["REVIEW", "SAVED", "DISCARDED"] },
];

function flowIndex(mode: AppMode): number {
  const idx = FLOW_STEPS.findIndex(step => step.modes.includes(mode));
  return idx >= 0 ? idx : 0;
}

function EpisodeFlowStepper({ mode }: { mode: AppMode }) {
  const cur = flowIndex(mode);
  return (
    <div style={flowStyles.wrap}>
      <div style={flowStyles.label}>Episode Flow</div>
      <div style={flowStyles.steps}>
        {FLOW_STEPS.map((step, idx) => {
          const status: "done" | "active" | "pending" =
            idx < cur ? "done" : idx === cur ? "active" : "pending";
          return (
            <div key={step.label} style={{ ...flowStyles.step, ...flowStepStyle(status) }}>
              <span style={flowStyles.dot}>
                {status === "done" ? "✓" : status === "active" ? "●" : "○"}
              </span>
              <span>{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * TeachIndicator — TEACH_RECORDING 상태에서 countdown → "ARM IS FREE" 표시
 */
function TeachIndicator() {
  const [cnt, setCnt] = React.useState(TEACH_DISABLE_DELAY);
  const [free, setFree] = React.useState(false);

  React.useEffect(() => {
    if (cnt <= 0) { setFree(true); return; }
    const t = setTimeout(() => setCnt(c => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cnt]);

  return free ? (
    <div style={teachStyles.freeBox}>
      <span style={{ fontSize: 22 }}>✋</span>
      <span style={teachStyles.freeText}>ARM IS FREE — guide by hand</span>
    </div>
  ) : (
    <div style={teachStyles.countdownBox}>
      <span style={teachStyles.countdownNum}>{cnt}</span>
      <span style={teachStyles.countdownLabel}>Motor release in {cnt}s…</span>
    </div>
  );
}

const teachStyles: Record<string, React.CSSProperties> = {
  countdownBox: {
    display: "flex", alignItems: "center", gap: 8, marginTop: 8,
    background: "#eff6ff", borderRadius: 8, padding: "8px 12px",
  },
  countdownNum: {
    fontSize: 36, fontWeight: 800, color: "#2563eb",
    fontFamily: "monospace", lineHeight: 1, flexShrink: 0,
  },
  countdownLabel: { fontSize: 12, color: "#1e40af", lineHeight: 1.4 },
  freeBox: {
    display: "flex", alignItems: "center", gap: 8, marginTop: 8,
    background: "#f0fdf4", border: "1.5px solid #86efac",
    borderRadius: 8, padding: "8px 12px",
  },
  freeText: { fontSize: 13, fontWeight: 700, color: "#15803d" },
};

/**
 * GripperRow — 자주 쓰는 opening preset을 즉시 전송
 */
const GRIPPER_OPEN_RAD = 0.07;
const GRIPPER_SNAPS = [0, 50, 70, 100];

function radToSnapPct(rad: number): number {
  const raw = Math.round((rad / GRIPPER_OPEN_RAD) * 100);
  return GRIPPER_SNAPS.reduce((prev, curr) =>
    Math.abs(curr - raw) < Math.abs(prev - raw) ? curr : prev
  );
}

function GripperRow({
  onSet, onClose, can, gripperRad,
}: { onSet: (pct: number) => void; onClose: () => void; can: (a: string) => boolean; gripperRad?: number }) {
  const [pct, setPct] = React.useState(() =>
    gripperRad !== undefined ? radToSnapPct(gripperRad) : 70
  );
  const canOpen  = can("open_gripper");
  const canClose = can("close_gripper");
  const canAny   = canOpen || canClose || can("gripper_set");
  const SNAPS    = GRIPPER_SNAPS;

  // 실제 그리퍼 상태 반영 (슬라이더 동기화)
  useEffect(() => {
    if (gripperRad !== undefined) {
      setPct(radToSnapPct(gripperRad));
    }
  }, [gripperRad]);

  const handleChange = (val: number) => {
    setPct(val);
    if (val === 0) onClose();
    else onSet(val);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, marginBottom: 5 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input
          type="range"
          min={0} max={100} step={5}
          value={pct}
          disabled={!canAny}
          onChange={e => handleChange(Number(e.target.value))}
          style={{ flex: 1, accentColor: "#22c55e", cursor: canAny ? "pointer" : "default" }}
        />
        <span style={{ fontSize: 11, fontWeight: 700, color: canAny ? "#15803d" : "#9ca3af", width: 30, textAlign: "right" }}>
          {pct}%
        </span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: 2, paddingRight: 36 }}>
        {SNAPS.map(v => (
          <button key={v} type="button" disabled={!canAny} onClick={() => handleChange(v)} style={{
            border: 0, background: "transparent", padding: "2px 3px", cursor: canAny ? "pointer" : "default",
            fontSize: 10, color: v === pct ? "#15803d" : "#9ca3af", fontWeight: v === pct ? 700 : 400,
          }}>
            {v}%
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Btn — 범용 버튼
 * size="sm": inline, auto-width, compact padding (gripper/utility용)
 */
function Btn({
  label, onClick, primary, danger, disabled, size,
}: {
  label: string; onClick: () => void;
  primary?: boolean; danger?: boolean; disabled?: boolean;
  size?: "sm";
}) {
  const bg    = disabled ? "#e5e7eb" : danger ? "#fef2f2" : primary ? "#6366f1" : "#f1f5f9";
  const color = disabled ? "#9ca3af" : danger ? "#dc2626"  : primary ? "#fff"    : "#374151";
  const isSm  = size === "sm";
  return (
    <button
      style={{
        display: isSm ? "inline-block" : "block",
        width: isSm ? "auto" : "100%",
        padding: isSm ? "4px 10px" : "7px 12px",
        borderRadius: 7, cursor: disabled ? "not-allowed" : "pointer",
        fontSize: isSm ? 11 : 12, fontWeight: 600,
        marginBottom: isSm ? 0 : 5, textAlign: "center",
        background: bg, color,
        border: danger ? "1px solid #fca5a5" : "1px solid transparent",
      }}
      onClick={onClick}
      disabled={disabled}
    >
      {label}
    </button>
  );
}

/**
 * AdvancedPanel — Freedrive (모터 해제) 제어
 */
function AdvancedPanel({ onHold }: { onHold: () => void }) {
  const [open, setOpen]       = React.useState(false);
  const [motorsFree, setFree] = React.useState(false);
  const [countdown, setCntdwn] = React.useState(0);

  React.useEffect(() => {
    if (countdown <= 0) return;
    const t = setTimeout(() => {
      if (countdown === 1) {
        fetch("/api/robot/motors/disable", { method: "POST" }).then(() => {
          setFree(true);
          setCntdwn(0);
        });
      } else {
        setCntdwn(c => c - 1);
      }
    }, 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  const enableMotors = () =>
    fetch("/api/robot/motors/enable", { method: "POST" }).then(() => setFree(false));

  return (
    <div style={{ borderTop: "1px solid #e5e7eb", marginTop: 8, paddingTop: 8 }}>
      <button
        style={{ background: "none", border: "none", fontSize: 11, color: "#6366f1", cursor: "pointer", fontWeight: 600, padding: 0 }}
        onClick={() => setOpen(v => !v)}
      >
        Advanced Robot Control {open ? "▾" : "▸"}
      </button>

      {open && (
        <div style={{ marginTop: 6, background: "#f8fafc", borderRadius: 7, padding: 8 }}>
          <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
            <Btn
              label="Go Home"
              onClick={() => post("/api/robot/preset", { index: 0 })}
            />
            <Btn label="Hold" onClick={onHold} danger />
          </div>
          <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 6 }}>
            Freedrive — release torque to set start pose by hand
          </div>
          {motorsFree ? (
            <>
              <div style={{ background: "#dcfce7", color: "#166534", borderRadius: 5, padding: "5px 8px", fontSize: 11, fontWeight: 700, marginBottom: 5 }}>
                ✋ ARM IS FREE — guide by hand
              </div>
              <Btn label="Re-enable Motors" onClick={enableMotors} primary />
            </>
          ) : countdown > 0 ? (
            <>
              <div style={{ background: "#fff7ed", color: "#c2410c", borderRadius: 5, padding: "5px 8px", fontSize: 11, fontWeight: 700, marginBottom: 5 }}>
                Releasing in {countdown}s…
              </div>
              <Btn label="Cancel" onClick={() => setCntdwn(0)} />
            </>
          ) : (
            <Btn label="Release Motors (Freedrive)" onClick={() => setCntdwn(3)} />
          )}
        </div>
      )}
    </div>
  );
}

/**
 * CalibResultPanel — calibration 완료 후 측정된 limits 표시
 */
function CalibResultPanel({
  limits,
}: { limits: Record<string, { min: number; max: number; measured?: boolean }> }) {
  const labels = ["J1 (base)", "J2 (shoulder)", "J3 (elbow)", "J4 (wrist pitch)", "J5 (wrist roll)", "J6 (wrist yaw)"];
  return (
    <div style={{ background: "#f0fdf4", border: "1px solid #86efac", borderRadius: 7, padding: "8px 10px", marginBottom: 5 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#15803d", marginBottom: 5 }}>
        Calibrated Limits (saved)
      </div>
      {Object.entries(limits).map(([key, v], i) => (
        <div key={key} style={{ fontSize: 10, fontFamily: "monospace", color: v.measured ? "#166534" : "#6b7280", marginBottom: 2 }}>
          {labels[i] ?? key}: [{v.min.toFixed(2)}, {v.max.toFixed(2)}]{v.measured ? "" : " (default)"}
        </div>
      ))}
    </div>
  );
}

/**
 * ReplayProgressCircle — SVG 원형 진행 표시기 (compact)
 */
function ReplayProgressCircle({ progress }: { progress: number }) {
  const r    = 20;
  const circ = 2 * Math.PI * r;
  const pct  = Math.min(1, Math.max(0, progress));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
      <svg width={48} height={48} style={{ transform: "rotate(-90deg)", flexShrink: 0 }}>
        <circle cx={24} cy={24} r={r} fill="none" stroke="#1e293b" strokeWidth={5} />
        <circle
          cx={24} cy={24} r={r}
          fill="none" stroke="#3b82f6" strokeWidth={5}
          strokeDasharray={`${pct * circ} ${circ - pct * circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 0.3s ease" }}
        />
      </svg>
      <span style={{ fontSize: 13, color: "#374151", fontWeight: 600 }}>
        {Math.round(pct * 100)}% replayed
      </span>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

function flowStepStyle(status: "done" | "active" | "pending"): React.CSSProperties {
  if (status === "done") return { color: "#15803d", background: "#f0fdf4", borderColor: "#bbf7d0" };
  if (status === "active") return { color: "#3730a3", background: "#eef2ff", borderColor: "#c7d2fe" };
  return { color: "#94a3b8", background: "#f8fafc", borderColor: "#e5e7eb" };
}

const flowStyles: Record<string, React.CSSProperties> = {
  wrap: {
    width: "100%",
    borderBottom: "1px solid #eef2f7",
    paddingBottom: 10,
    marginBottom: 2,
  },
  label: {
    fontSize: 10,
    fontWeight: 800,
    color: "#94a3b8",
    letterSpacing: 0.8,
    textTransform: "uppercase" as const,
    marginBottom: 6,
  },
  steps: {
    display: "grid",
    gridTemplateColumns: "repeat(5, minmax(0, 1fr))",
    gap: 6,
  },
  step: {
    border: "1px solid",
    borderRadius: 7,
    padding: "6px 8px",
    fontSize: 11,
    fontWeight: 800,
    display: "flex",
    alignItems: "center",
    gap: 5,
    minWidth: 0,
    whiteSpace: "nowrap" as const,
    overflow: "hidden",
  },
  dot: { flexShrink: 0, width: 14 },
};

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    gap: 8,
  },

  // Row A
  cameraArea: {
    flex: 1,
    minHeight: 240,
    background: "#0f172a",
    borderRadius: 12,
    overflow: "hidden",
    position: "relative",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  mainImg: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "contain",
  } as React.CSSProperties,
  recBadge: {
    position: "absolute",
    top: 12, left: 12, zIndex: 2,
    background: "#dc2626", color: "#fff",
    borderRadius: 6, padding: "4px 10px",
    fontWeight: 700, fontSize: 13, letterSpacing: 1,
  },
  teachBadge: {
    position: "absolute",
    top: 12, left: 12, zIndex: 2,
    background: "#6366f1", color: "#fff",
    borderRadius: 6, padding: "4px 10px",
    fontWeight: 700, fontSize: 13, letterSpacing: 1,
  },
  screenshotBtn: {
    position: "absolute",
    top: 12, right: 12, zIndex: 3,
    background: "rgba(15,23,42,0.7)",
    color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)",
    borderRadius: 6,
    padding: "6px 12px",
    fontSize: 12,
    fontWeight: 700,
    cursor: "pointer",
  } as React.CSSProperties,
  polarizerModeOverlay: {
    position: "absolute",
    top: "50%", right: 12, zIndex: 2,
    transform: "translateY(-50%)",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  } as React.CSSProperties,
  polarizerInfoBox: {
    background: "rgba(15,23,42,0.6)",
    border: "1px solid rgba(255,255,255,0.25)",
    borderRadius: 6,
    padding: "8px 12px",
    color: "#e2e8f0",
    fontSize: 11,
    fontWeight: 600,
    lineHeight: 1.6,
    marginBottom: 4,
    minWidth: 120,
  } as React.CSSProperties,
  polarizerModeBtn: {
    padding: "6px 14px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 12,
    fontWeight: 700,
    color: "#fff",
    backdropFilter: "blur(2px)",
  } as React.CSSProperties,
  cameraTabs: {
    flexShrink: 0,
    display: "flex",
    gap: 6,
  },
  camTab: {
    display: "flex",
    alignItems: "center",
    gap: 5,
    padding: "5px 10px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 11,
    fontWeight: 600,
    flexShrink: 0,
  },
  placeholder: {
    color: "#475569",
    fontSize: 14,
  },

  // Row B: Step Card
  stepCard: {
    flexShrink: 0,
    background: "#fff",
    borderRadius: 10,
    padding: "12px 16px",
    display: "flex",
    flexWrap: "wrap",
    gap: 16,
    alignItems: "flex-start",
  },
  stepLeft: {
    flex: 1,
    minWidth: 0,
  },
  stepBadge: {
    display: "inline-block",
    background: "#eff6ff",
    color: "#3b82f6",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 0.5,
    padding: "2px 8px",
    borderRadius: 4,
    marginBottom: 5,
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: 700,
    color: "#1e293b",
    marginBottom: 3,
  },
  stepGuide: {
    fontSize: 12,
    color: "#6b7280",
    lineHeight: 1.4,
  },
  taskInput: {
    width: "100%",
    border: "1px solid #d1d5db",
    borderRadius: 6,
    padding: "6px 10px",
    fontSize: 12,
    outline: "none",
    marginTop: 7,
    boxSizing: "border-box",
  } as React.CSSProperties,
  sliderLabel: {
    fontSize: 11,
    color: "#6b7280",
    marginBottom: 4,
  },
  sliderHint: {
    fontSize: 10,
    color: "#9ca3af",
  },
  stepRight: {
    minWidth: 190,
    maxWidth: 240,
    display: "flex",
    flexDirection: "column",
    gap: 0,
    flexShrink: 0,
  },
  freeAlert: {
    background: "#f0fdf4",
    border: "1.5px solid #86efac",
    color: "#15803d",
    borderRadius: 6,
    padding: "6px 10px",
    fontSize: 12,
    fontWeight: 700,
    marginBottom: 6,
    textAlign: "center",
  },
  confirmBox: {
    background: "#fff7ed",
    border: "1px solid #fed7aa",
    borderRadius: 7,
    padding: "10px 12px",
    marginBottom: 5,
  },
  replayMessage: {
    width: "100%", boxSizing: "border-box", border: "1px solid #fca5a5",
    borderRadius: 7, background: "#fef2f2", color: "#b91c1c",
    padding: "8px 10px", fontSize: 11, fontWeight: 700,
  },
  safetyZone: {
    borderTop: "1px solid #fee2e2",
    marginTop: 6,
    paddingTop: 6,
  },

  // Row C: Event Log
  eventLog: {
    flexShrink: 0,
    background: "#1e293b",
    borderRadius: 8,
    padding: "7px 12px",
    display: "flex",
    flexDirection: "column",
    gap: 2,
    maxHeight: 82,
    overflowY: "auto",
  },
  eventRow: {
    display: "flex",
    gap: 8,
    fontSize: 11,
    lineHeight: 1.5,
  },
  eventTime: {
    color: "#64748b",
    fontFamily: "monospace",
    flexShrink: 0,
  },
  eventMsg: {
    color: "#cbd5e1",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },

  // Saving overlay
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.55)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 9999,
  },
  overlayBox: {
    background: "#1e293b",
    borderRadius: 16,
    padding: "32px 48px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
  },
  spinner: {
    width: 40, height: 40,
    border: "4px solid #334155",
    borderTop: "4px solid #3b82f6",
    borderRadius: "50%",
    animation: "spin 0.9s linear infinite",
  },
};

// Spinner keyframe (once)
if (typeof document !== "undefined" && !document.getElementById("cap-spin-style")) {
  const s = document.createElement("style");
  s.id = "cap-spin-style";
  s.textContent = "@keyframes spin { to { transform: rotate(360deg); } }";
  document.head.appendChild(s);
}
