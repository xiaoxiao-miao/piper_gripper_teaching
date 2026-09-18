import React, { useState, useEffect, useRef } from "react";
import { getSocket } from "./socket";
import type {
  AppMode, ModeChangePayload, RobotState, CameraState,
  LoggerState, EventLog, SaveStep, SaveProgressPayload,
} from "./types";
import TopBar from "./components/TopBar";
import WorkflowNav from "./components/WorkflowNav";
import StatusPanel from "./components/StatusPanel";
import CapturePage from "./pages/CapturePage";
import DatasetPage from "./pages/DatasetPage";
import DiagnosticsPage from "./pages/DiagnosticsPage";
import SetupPage from "./pages/SetupPage";
import ScreenshotsGallery from "./pages/dataset/ScreenshotsGallery";
import SafetyBanner from "./components/SafetyBanner";

export type Page = "capture" | "dataset" | "screenshots" | "diagnostics" | "setup";

export default function App() {
  const [mode, setMode] = useState<AppMode>("IDLE");
  const [nextAction, setNextAction] = useState("");
  const [availableActions, setAvailableActions] = useState<string[]>([]);
  const [robot, setRobot] = useState<RobotState>({ connected: false });
  const [camera, setCamera] = useState<CameraState>({ available: false });
  const [logger, setLogger] = useState<LoggerState>({});
  const [events, setEvents] = useState<EventLog[]>([]);
  const [page, setPage] = useState<Page>("capture");
  const [devMode, setDevMode] = useState(false);

  const SAVE_STEP_LABELS: Record<string, string> = {
    stop_cameras:  "카메라 녹화 중단",
    flush_csv:     "CSV 저장",
    align_frames:  "프레임 정렬",
    video_color:   "컬러 비디오 인코딩",
    video_depth:   "뎁스 비디오 인코딩",
    video_webcam_0: "웹캠 0 비디오 인코딩",
    video_webcam_1: "웹캠 1 비디오 인코딩",
    video_cam0: "cam0 비디오 인코딩",
    video_cam1: "cam1 비디오 인코딩",
    video_realsense: "RealSense 비디오 인코딩",
  };

  const initSaveSteps = (): SaveStep[] =>
    Object.entries(SAVE_STEP_LABELS).map(([key, label]) => ({
      key, label, status: "waiting", detail: "",
    }));

  const [saveSteps, setSaveSteps] = useState<SaveStep[]>(initSaveSteps());

  const socketRef = useRef(getSocket());

  useEffect(() => {
    const s = socketRef.current;

    s.on("mode_change", (data: ModeChangePayload) => {
      setMode(prev => {
        // SAVED/DISCARDED → RETURN_HOME → TEACH_READY 흐름 감지: capture 페이지로 이동
        if (data.mode === "TEACH_READY" && (prev === "SAVED" || prev === "DISCARDED" || prev === "RETURN_HOME")) {
          setPage("capture");
        }
        return data.mode;
      });
      setNextAction(data.next_action);
      setAvailableActions(data.available_actions);
      if (data.mode === "PROCESSING") {
        setSaveSteps(initSaveSteps());
      }
    });

    s.on("save_progress", (data: SaveProgressPayload) => {
      setSaveSteps(prev => {
        if (!prev.some(s => s.key === data.step)) {
          return [...prev, {
            key: data.step,
            label: SAVE_STEP_LABELS[data.step] ?? data.step,
            status: data.status,
            detail: data.detail ?? "",
          }];
        }
        return prev.map(s =>
          s.key === data.step
            ? { ...s, status: data.status, detail: data.detail ?? "" }
            : s
        );
      });
    });

    s.on("robot_state", (data: RobotState) => setRobot(data));
    s.on("camera_state", (data: CameraState) => setCamera(data));
    s.on("logger_state", (data: LoggerState) => setLogger(data));
    s.on("event_log", (entry: EventLog) => {
      setEvents(prev => [...prev.slice(-99), entry]);
    });

    return () => {
      s.off("mode_change");
      s.off("save_progress");
      s.off("robot_state");
      s.off("camera_state");
      s.off("logger_state");
      s.off("event_log");
    };
  }, []);

  const renderMain = () => {
    if (page === "dataset") return <DatasetPage />;
    if (page === "screenshots") return <ScreenshotsGallery />;
    if (page === "diagnostics") return <DiagnosticsPage devMode={devMode} robot={robot} camera={camera} />;
    if (page === "setup") return <SetupPage robot={robot} />;
    return (
      <CapturePage
        mode={mode}
        availableActions={availableActions}
        robot={robot}
        camera={camera}
        events={events}
        episodeId={logger.episode_id}
        saveSteps={saveSteps}
      />
    );
  };

  return (
    <div style={styles.root}>
      <TopBar
        mode={mode}
        robot={robot}
        camera={camera}
        logger={logger}
        devMode={devMode}
        onToggleDev={() => setDevMode(v => !v)}
      />
      <SafetyBanner robot={robot} />
      <div style={styles.body}>
        <WorkflowNav
          page={page}
          onNavigate={setPage}
        />
        <main style={styles.main}>
          {renderMain()}
        </main>
        <StatusPanel
          page={page}
          mode={mode}
          robot={robot}
          camera={camera}
          logger={logger}
          events={events}
          availableActions={availableActions}
          nextAction={nextAction}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    overflow: "hidden",
    background: "#f0f2f5",
  },
  body: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
  },
  main: {
    flex: 1,
    overflow: "auto",
    padding: "16px",
  },
};
