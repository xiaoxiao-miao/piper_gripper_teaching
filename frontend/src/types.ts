// 상태 기계 상태값
export type AppMode =
  | "IDLE"
  | "CONNECTING"
  | "READY"
  | "CALIBRATING"
  | "TEACH_READY"
  | "TEACH_STARTING"
  | "TEACH_RECORDING"
  | "TEACH_STOPPING"
  | "TRAJECTORY_CHECK"
  | "RETURN_HOME"
  | "REPLAY_READY"
  | "REPLAY_RECORDING"
  | "PROCESSING"
  | "REVIEW"
  | "SAVED"
  | "DISCARDED"
  | "DEBUG_RETURN"
  | "DEBUG_REPLAY";

export type SaveStepStatus = "waiting" | "running" | "ok" | "failed";

export interface SaveStep {
  key: string;
  label: string;
  status: SaveStepStatus;
  detail: string;
}

export interface SaveProgressPayload {
  step: string;
  status: "running" | "ok" | "failed";
  detail: string;
}

// SocketIO 서버→클라이언트 이벤트
export interface ModeChangePayload {
  mode: AppMode;
  next_action: string;
  available_actions: string[];
  trajectory_summary?: TrajectoryCheck;
}

export interface RobotState {
  connected: boolean;
  position?: number[];     // q1..q6 (rad)
  gripper?: number;
  gripper_responsive?: boolean | null;  // false=마지막 명령이 피드백에 반영 안 됨(케이블 단선 등)
  velocity?: number[];
  is_moving?: boolean;
  mode?: string;
  hz?: number;
  replay_progress?: number;  // 0.0~1.0 재생 진행률 (REPLAY_RECORDING 중)
  joint_limits?: { min: number[]; max: number[] };  // ROM 캘리브레이션 결과
  feedback_age_s?: number | null;
  motor_enable_status?: boolean[];
  motor_fault_codes?: number[];
  motor_faults?: string[][];
  motor_status_age_s?: number | null;
  transition_state?: string;
  transition_error?: string | null;
  calibration_span?: number[];
  calibration_valid?: boolean[];
  calibration_min_span?: number;
  safety_latched?: boolean;
  safety_reason?: string | null;
  can_healthy?: boolean;
  abort_class?: "MOTION_ABORTED" | "SAFETY_LOCKED" | "CAN_LOST" | null;
  last_motion_abort?: { reason?: string; abort_class?: string; [key: string]: unknown };
  recovery?: Record<string, unknown>;
}

export interface SingleCameraState {
  id?: string;
  name?: string;
  label?: string;
  role?: string;
  type?: string;
  legacy_id?: string;
  streams?: string[];
  enabled?: boolean;
  aliases?: string[];
  device?: string;
  available: boolean;
  streaming?: boolean;
  recording?: boolean;
  fps?: number;
  resolution?: string;
  captured_frames?: number;
  written_frames?: number;
  dropped_frames?: number;
  queue_len?: number;
  // Spinnaker(polarizer) 전용
  pixel_format?: string;
  exposure_us?: number;
  exposure_us_min?: number;
  exposure_us_max_hw?: number;
  exposure_us_max_hint_for_target_fps?: number;
  exposure_auto?: string; // "Off" | "Continuous" | ...
}

export interface CameraState {
  // multi-camera (new)
  connected?: boolean;
  cameras?: Record<string, SingleCameraState>;
  // legacy single-camera fields (backward compat)
  available?: boolean;
  streaming?: boolean;
  recording?: boolean;
  fps?: number;
  captured_frames?: number;
  written_frames?: number;
  dropped_frames?: number;
  queue_len?: number;
}

export interface LoggerState {
  episode_id?: string;
  disk_free_gb?: number;
  teach_samples?: number;
  exec_samples?: number;
  cam_frames?: number;
}

export interface EventLog {
  t: number;
  message: string;
}

export interface TrajectoryCheck {
  ok: boolean;
  duration_s?: number;
  samples?: number;
  max_joint_velocity_rad_s?: number;
  velocity_limit_ok?: boolean;
  max_waypoint_delta_rad?: number;
  waypoint_delta_ok?: boolean;
  gripper_events?: number;
  reason?: string;
  speed_scale?: number;
}

export interface EpisodeTake {
  take: string;          // "take_001", "take_002", ...
  has_video: boolean;
  has_teach: boolean;
  has_webcam_0?: boolean;
  has_webcam_1?: boolean;
  cameras?: Array<{
    id: string;
    label: string;
    role?: string;
    legacy_id?: string;
    video: string;
    streams?: Array<{
      id: string;
      label: string;
      kind: string;
      video: string;
      url: string;
      maskable: boolean;
    }>;
  }>;
  size_mb?: number;
  review_issues?: ReviewIssue[];
  review_issue_count?: number;
}

export interface ReviewIssue {
  key: string;
  label: string;
  severity?: "info" | "warning" | "danger";
  source?: string;
  lerobot_index?: number;
  task_key?: string;
  status?: "open" | "auto_fixed" | "needs_review" | "resolved";
  note?: string;
  suggested_actions?: Array<{
    type: "mark_failure" | "trim" | "set_mask" | string;
    params?: Record<string, unknown>;
  }>;
  created_at?: string;
  updated_at?: string;
}

export interface CameraManifestEntry {
  id: string;
  label: string;
  type: string;
  role: string;
  enabled: boolean;
  device: string;
  streams: string[];
  legacy_id: string;
  export_presets: string[];
}

export interface CameraManifest {
  version: number;
  cameras: CameraManifestEntry[];
}

export interface Episode {
  episode_id: string;
  task: string;
  created_at: string;
  success: boolean | null;
  takes: EpisodeTake[];
  takes_count: number;
  size_mb?: number;
  duration_s?: number | null;
  preview_video_url?: string | null;
  completeness?: Record<string, boolean>;
  has_postprocess?: boolean;  // 마지막 take에 활성화된 postprocess 편집이 있는지
  review_issues?: ReviewIssue[];
  review_issue_count?: number;
  // 구버전 호환
  stats?: Record<string, number>;
  checklist?: Record<string, boolean>;
}

export interface JointSample {
  t: number;      // seconds from start of recording
  q: number[];    // [q1..q6] in radians
  gripper: number;
}

export interface TakeJointsData {
  teach: JointSample[];
  executed: JointSample[];
  command: JointSample[];
}

export interface EESample {
  t: number;                           // seconds from start
  gripper: number;
  ee: [number, number, number];        // [x, y, z] in meters (base_link frame)
  links: [number, number, number][];   // [6] link1..link6 positions in meters
}

export interface TakeTrajectoryData {
  teach: EESample[];
  executed: EESample[];
  command: EESample[];
}

export interface TrimMeta {
  enabled: boolean;
  cut_t: number | null;   // gripper open 시각 (null = 자동 감지)
  margin: number;          // cut_t 기준 offset(초). 음수면 open signal 이전에 자름
}

export interface MaskMeta {
  enabled: boolean;
  camera_id?: string;  // target RGB camera id/alias. Defaults to manifest role=overview
  polygon: [number, number][];  // normalized 0-1 좌표 (keep 영역 폴리곤)
  fill: "black" | "frame_capture";
  capture_t?: number;  // frame_capture 타입: 캡쳐 기준 시각(초)
}

export interface EditMeta {
  trim: TrimMeta;
  mask: MaskMeta;
  timing?: {
    mode: "teaching_normalized" | "physical_replay";
  };
}

export interface MaskLibraryEntry {
  id: string;
  name: string;
  polygon: [number, number][];
  created_at: string;
  source_episode?: string;
  source_take?: string;
}
