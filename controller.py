"""
Controller
----------
전체 상태 기계(State Machine) + 오케스트레이션.

상태 전이:
  IDLE → CONNECTING → READY
       → TEACH_READY → TEACH_RECORDING → TRAJECTORY_CHECK
       → RETURN_HOME → REPLAY_READY → REPLAY_RECORDING
       → REVIEW → SAVED | DISCARDED

핵심 설계:
  - 프론트엔드는 controller.get_status() 를 보고 available_actions 만 표시
  - controller가 모든 상태 전이 guard를 가짐
  - SocketIO emit은 on_emit 콜백으로 주입받음 (테스트 가능)
"""

import csv
import json
import os
import threading
import time
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Optional

import rclpy
import numpy as np
from rclpy.executors import MultiThreadedExecutor

from nodes.piper_node import PiperTeachReplayNode
from nodes.camera_manager import CameraManager
from storage.episode_manager import EpisodeManager
from config import (
    REPLAY_DEFAULT_SPEED,
    HOME_APPROACH_SPEED,
    DATASET_PATH,
    GRIPPER_OPEN_RAD,
    GRIPPER_TEACH_START_PCT,
    JOINT_LIMITS_FILE,
    MAX_JOINT_VELOCITY_RAD_S,
)

CALIBRATION_POSE_FILE = os.path.join(DATASET_PATH, "calibration_pose.json")


class State(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    READY = "READY"
    CALIBRATING = "CALIBRATING"
    TEACH_READY = "TEACH_READY"
    TEACH_STARTING = "TEACH_STARTING"
    TEACH_RECORDING = "TEACH_RECORDING"
    TEACH_STOPPING = "TEACH_STOPPING"
    TRAJECTORY_CHECK = "TRAJECTORY_CHECK"
    RETURN_HOME = "RETURN_HOME"
    REPLAY_READY = "REPLAY_READY"
    REPLAY_RECORDING = "REPLAY_RECORDING"
    PROCESSING = "PROCESSING"   # replay 완료 후 CSV flush + 비디오 인코딩 중
    REVIEW = "REVIEW"
    SAVED = "SAVED"
    DISCARDED = "DISCARDED"
    DEBUG_RETURN = "DEBUG_RETURN"
    DEBUG_REPLAY = "DEBUG_REPLAY"


_NEXT_ACTIONS = {
    State.IDLE:             "Connect the robot and camera to begin.",
    State.CONNECTING:       "Waiting for robot and camera to connect...",
    State.READY:            "System ready. Press 'Confirm Ready Pose' to start a teaching session.",
    State.CALIBRATING:      "Move arm freely through its full range of motion, then press 'Finish Calibration'.",
    State.TEACH_READY:      "Robot is in home position. Press 'Start Teaching' when ready.",
    State.TEACH_STARTING:   "Preparing freedrive and waiting for six-axis motor feedback...",
    State.TEACH_RECORDING:  "Guide the robot arm by hand. Press 'Stop Teaching' when done.",
    State.TEACH_STOPPING:   "Stopping teaching, re-arming the robot, and preserving the trajectory...",
    State.TRAJECTORY_CHECK: "Trajectory recorded. Review the summary and return home before replay.",
    State.RETURN_HOME:      "Robot is returning to home position. Please wait.",
    State.REPLAY_READY:     "Robot is at home. Set replay speed and press 'Start Replay Recording'.",
    State.REPLAY_RECORDING: "Replay in progress. Camera and joint data are being recorded.",
    State.PROCESSING:       "Replay done. Encoding videos and saving data...",
    State.REVIEW:           "Review the episode. Label as success/failure and save or discard.",
    State.SAVED:            "Episode saved. Start a new episode or add another take.",
    State.DISCARDED:        "Episode discarded. You can start a new episode.",
    State.DEBUG_RETURN:     "Dataset debug: returning to the selected trajectory start.",
    State.DEBUG_REPLAY:     "Dataset debug replay is running without camera recording.",
}

_AVAILABLE_ACTIONS: dict[State, list[str]] = {
    State.IDLE:             ["connect"],
    State.CONNECTING:       [],
    State.READY:            ["confirm_ready_pose", "calibrate", "open_gripper", "close_gripper"],
    State.CALIBRATING:      ["stop_calibration", "hold_position", "resume_calibration"],
    State.TEACH_READY:      ["start_teach", "calibrate", "open_gripper", "close_gripper"],
    State.TEACH_STARTING:   [],
    State.TEACH_RECORDING:  ["stop_teach", "open_gripper", "close_gripper", "hold_position"],
    State.TEACH_STOPPING:   [],
    State.TRAJECTORY_CHECK: ["return_home", "return_home_direct", "discard_episode"],
    State.RETURN_HOME:      ["hold_position"],
    State.REPLAY_READY:     ["start_replay", "discard_episode", "open_gripper", "close_gripper"],
    State.REPLAY_RECORDING: ["stop_replay", "hold_position", "open_gripper", "close_gripper", "gripper_set"],
    State.PROCESSING:       [],
    State.REVIEW:           ["save_episode", "discard_episode", "retake_replay", "retake_teach"],
    State.SAVED:            ["new_episode", "add_take"],
    State.DISCARDED:        ["new_episode"],
    State.DEBUG_RETURN:     ["hold_position"],
    State.DEBUG_REPLAY:     ["stop_debug_replay", "hold_position"],
}


class Controller:

    def __init__(self, on_emit: Optional[Callable] = None):
        """
        on_emit(event, data): SocketIO emit 콜백.
                              None이면 emit 없이 동작 (테스트용).
        """
        self._on_emit = on_emit or (lambda e, d: None)
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._teach_stop_lock = threading.Lock()
        self._replay_start_lock = threading.Lock()
        self._replay_partial = False
        self._replay_manual_stop_requested = False
        self._debug_lock = threading.RLock()
        self._debug_trajectory: list[dict] = []
        self._debug_source: dict = {}
        self._debug_check: dict = {}
        self._debug_start_ready = False
        self._debug_operation: Optional[str] = None
        self._debug_previous_state: Optional[State] = None
        self._debug_run_dir: Optional[str] = None
        self._debug_started_at: Optional[str] = None
        self._debug_speed: Optional[float] = None
        self._debug_stop_requested = False
        self._debug_last_error = ""
        self._debug_last_diagnostics: dict = {}
        self._debug_sequence: Optional[dict] = None

        # ── 서브시스템 ──────────────────────────────────────────────
        self._piper: Optional[PiperTeachReplayNode] = None
        self._cameras: Optional[CameraManager] = None
        self._episode: EpisodeManager = EpisodeManager(DATASET_PATH)
        self._ros_executor: Optional[MultiThreadedExecutor] = None
        self._ros_thread: Optional[threading.Thread] = None

        # ── calibration ────────────────────────────────────────────────
        self._calib_prev_state: State = State.READY

        # ── trajectory cache (teach → check → replay 사이에 보관) ───
        self._teach_buffer: list[dict] = []
        self._smoothed_trajectory: list[dict] = []
        self._trajectory_check: dict = {}
        self._replay_speed: float = REPLAY_DEFAULT_SPEED

        # ── 상태 폴링 스레드 ───────────────────────────────────────
        self._poll_stop = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # ── 이벤트 로그 ────────────────────────────────────────────
        self._event_log: list[dict] = []

        print("[Controller] Initialized")

    # ─────────────────────────────────────────────────────────────────
    # State helpers
    # ─────────────────────────────────────────────────────────────────

    def _set_state(self, new_state: State, extra: dict = None):
        with self._lock:
            self._state = new_state
        payload = self.get_status()
        if extra:
            payload.update(extra)
        self._on_emit("mode_change", payload)
        self._add_event_log(f"State → {new_state.value}")

    def get_state(self) -> State:
        with self._lock:
            return self._state

    def get_status(self) -> dict:
        state = self.get_state()
        status = {
            "mode": state.value,
            "next_action": _NEXT_ACTIONS.get(state, ""),
            "available_actions": _AVAILABLE_ACTIONS.get(state, []),
        }
        if self._trajectory_check:
            status["trajectory_summary"] = self._trajectory_check
        status["replay_partial"] = bool(getattr(self, "_replay_partial", False))
        return status

    def _add_event_log(self, message: str):
        entry = {"t": time.time(), "message": message}
        self._event_log.append(entry)
        if len(self._event_log) > 200:
            self._event_log = self._event_log[-200:]
        self._on_emit("event_log", entry)

    def get_event_log(self) -> list[dict]:
        return self._event_log[-50:]

    # ─────────────────────────────────────────────────────────────────
    # Polling loop (10Hz → robot_state, camera_state, logger_state emit)
    # ─────────────────────────────────────────────────────────────────

    def _poll_loop(self):
        while not self._poll_stop.is_set():
            self._emit_robot_state()
            self._emit_camera_state()
            self._emit_logger_state()
            time.sleep(0.1)  # 10 Hz

    def _emit_robot_state(self):
        if self._piper is None:
            self._on_emit("robot_state", {"connected": False})
            return
        state = self._piper.get_state()
        self._on_emit("robot_state", {
            "connected": True,
            "position": [round(v, 4) for v in state["position"]],
            "gripper": round(state["gripper"], 4),
            "gripper_responsive": state.get("gripper_responsive"),
            "velocity": [round(v, 4) for v in state["velocity"]],
            "is_moving": state["is_moving"],
            "mode": state["mode"],
            "hz": state["hz"],
            "replay_progress": state.get("replay_progress", 0.0),
            "feedback_age_s": state.get("feedback_age_s"),
            "motor_enable_status": state.get("motor_enable_status", []),
            "motor_fault_codes": state.get("motor_fault_codes", []),
            "motor_faults": state.get("motor_faults", []),
            "motor_status_age_s": state.get("motor_status_age_s"),
            "transition_state": state.get("transition_state"),
            "transition_error": state.get("transition_error"),
            "calibration_span": state.get("calibration_span", []),
            "calibration_valid": state.get("calibration_valid", []),
            "calibration_min_span": state.get("calibration_min_span"),
            "safety_latched": state.get("safety_latched", False),
            "safety_reason": state.get("safety_reason"),
            "can_healthy": state.get("can_healthy", True),
            "abort_class": state.get("abort_class"),
            "last_motion_abort": state.get("last_motion_abort", {}),
            "recovery": state.get("recovery", {}),
        })

    def _emit_camera_state(self):
        if self._cameras is None:
            self._on_emit("camera_state", {"connected": False, "cameras": {}})
            return
        self._on_emit("camera_state", self._cameras.get_status())

    def _emit_logger_state(self):
        import shutil as _shutil
        free = _shutil.disk_usage(DATASET_PATH).free // (1024 ** 3)
        piper_samples = len(self._piper.get_teach_buffer()) if self._piper else 0
        self._on_emit("logger_state", {
            "episode_id": self._episode.current_episode_id,
            "disk_free_gb": free,
            "teach_samples": piper_samples,
            "exec_samples": len(self._piper.get_executed_buffer()) if self._piper else 0,
            "cam_frames": self._get_total_cam_frames(),
        })

    def _get_total_cam_frames(self) -> int:
        """모든 카메라의 captured_frames 합계."""
        if self._cameras is None:
            return 0
        status = self._cameras.get_status()
        total = 0
        for cam_info in status.get("cameras", {}).values():
            total += cam_info.get("captured_frames", 0)
        return total

    # ─────────────────────────────────────────────────────────────────
    # Actions (guard → execute → transition)
    # ─────────────────────────────────────────────────────────────────

    def connect(self) -> dict:
        # 레이스 컨디션 방지: 상태 확인과 변경을 같은 lock 안에서 처리
        with self._lock:
            if self._state not in (State.IDLE, State.DISCARDED, State.SAVED):
                return {"ok": False, "reason": f"Already in state {self._state.value}"}
            self._state = State.CONNECTING

        # lock 밖에서 emit (deadlock 방지)
        self._on_emit("mode_change", self.get_status())
        self._add_event_log("State → CONNECTING")
        threading.Thread(target=self._connect_async, daemon=True).start()
        return {"ok": True}

    def _connect_async(self):
        try:
            # ROS2 초기화
            if not rclpy.ok():
                rclpy.init()

            self._piper = PiperTeachReplayNode(on_state_update=None)
            self._ros_executor = MultiThreadedExecutor()
            self._ros_executor.add_node(self._piper)
            self._ros_thread = threading.Thread(target=self._ros_executor.spin, daemon=True)
            self._ros_thread.start()

            # 카메라 초기화 (RealSense + USB 웹캠, 1대라도 연결되면 진행)
            self._cameras = CameraManager()
            cam_ok = self._cameras.connect()
            if not cam_ok:
                print("[Controller] Warning: no cameras available — continuing without camera")

            time.sleep(1.0)  # 안정화

            # 저장된 joint limits 로드 (있으면 적용)
            self._load_joint_limits()

            self._set_state(State.READY)
            self._add_event_log("Robot and camera connected")

        except Exception as e:
            self._add_event_log(f"Connection failed: {e}")
            self._set_state(State.IDLE)

    def confirm_ready_pose(self) -> dict:
        if self.get_state() != State.READY:
            return {"ok": False, "reason": "Not in READY state"}

        # gripper close 시 기록된 암 위치를 캘리브레이션 포즈로 저장
        if self._piper:
            pose = self._piper.get_gripper_close_arm_pos()
            if pose:
                try:
                    os.makedirs(os.path.dirname(CALIBRATION_POSE_FILE), exist_ok=True)
                    with open(CALIBRATION_POSE_FILE, "w") as f:
                        json.dump(pose, f)
                    self._add_event_log(f"Calibration pose saved: {[round(v,3) for v in pose]}")
                except Exception as e:
                    self._add_event_log(f"Failed to save calibration pose: {e}")

        self._set_state(State.TEACH_READY)
        return {"ok": True}

    def start_teach(self, task: str = "unspecified", operator: str = "unknown") -> dict:
        if self.get_state() != State.TEACH_READY:
            return {"ok": False, "reason": "Must be in TEACH_READY state"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}

        # 활성 episode가 없을 때만 새 episode 생성 (retake/add_take 후에는 이미 take가 준비되어 있음)
        if self._episode.current_episode_id is None:
            self._episode.create_episode(task=task, operator=operator)

        # Page load/connect must never move hardware.  Apply the slightly-open
        # preset only when the operator explicitly starts a Teach run.
        self._piper.gripper_set(GRIPPER_TEACH_START_PCT)
        self._piper.start_teach(
            disable_delay=5.0,
            failure_callback=self._on_teach_release_failed,
            started_callback=self._on_teach_recording_started,
        )
        self._set_state(State.TEACH_STARTING)
        take = self._episode.get_current_take_name()
        self._add_event_log(
            f"Teaching started: {task} ({take}), gripper={GRIPPER_TEACH_START_PCT:.0f}%"
        )
        return {
            "ok": True,
            "episode_id": self._episode.current_episode_id,
            "take": take,
            "gripper_pct": GRIPPER_TEACH_START_PCT,
        }

    def _on_teach_release_failed(self, reason: str):
        if self.get_state() == State.TEACH_STARTING:
            self._add_event_log(f"Teaching cancelled: freedrive handshake failed: {reason}")
            self._set_state(State.TEACH_READY)

    def _on_teach_recording_started(self):
        """Publish TEACH_RECORDING only after all six axes report disabled."""
        if self.get_state() == State.TEACH_STARTING:
            self._set_state(State.TEACH_RECORDING)

    def stop_teach(self) -> dict:
        # Re-arm/trajectory processing takes long enough for a user (or two UIs)
        # to submit the button repeatedly while the state still says RECORDING.
        # Only the first request may perform the motor handshake and endpoint seed.
        if not self._teach_stop_lock.acquire(blocking=False):
            return {"ok": False, "reason": "Teach stop is already in progress"}
        try:
            return self._stop_teach_once()
        finally:
            self._teach_stop_lock.release()

    def _stop_teach_once(self) -> dict:
        if self.get_state() != State.TEACH_RECORDING:
            return {"ok": False, "reason": "Not in TEACH_RECORDING state"}

        # Change state before the blocking re-arm handshake.  This makes a
        # second UI/client request harmless even when it arrives from another
        # device while the first request is still running.
        self._set_state(State.TEACH_STOPPING)
        teach_buffer = self._piper.stop_teach()
        if teach_buffer is None:
            reason = self._piper._last_transition_error or "failed to re-arm robot"
            self._add_event_log(f"Teaching stop failed: {reason}")
            self._set_state(State.TEACH_RECORDING)
            return {"ok": False, "reason": reason}
        self._teach_buffer = teach_buffer

        if len(self._teach_buffer) < 10:
            self._add_event_log("Teaching too short, discarding")
            self._set_state(State.TEACH_READY)
            return {"ok": False, "reason": "Teaching too short (< 10 samples)"}

        # Trajectory smoothing & check
        smoothed, check = PiperTeachReplayNode.smooth_trajectory(self._teach_buffer, self._replay_speed)
        self._smoothed_trajectory = smoothed
        self._trajectory_check = check
        self._trajectory_check["gripper_events"] = sum(
            1 for e in self._piper.get_events() if "gripper" in e.get("event", "")
        )

        # Flush teach CSV
        self._episode.flush_teach_joint(self._teach_buffer)
        self._episode.flush_events(self._piper.get_events())

        self._set_state(State.TRAJECTORY_CHECK)
        self._add_event_log(f"Teach done: {len(self._teach_buffer)} samples, check={'OK' if check['ok'] else 'WARN'}")
        return {"ok": True, "trajectory_check": check}

    def return_home(self) -> dict:
        if self.get_state() != State.TRAJECTORY_CHECK:
            return {"ok": False, "reason": "Must be in TRAJECTORY_CHECK state"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        if not self._smoothed_trajectory:
            self._add_event_log("return_home blocked: no trajectory — going to TEACH_READY")
            self._set_state(State.TEACH_READY)
            return {"ok": False, "reason": "No trajectory available"}

        self._set_state(State.RETURN_HOME)
        self._add_event_log("Returning to teach_start via reverse trajectory")

        ok = self._piper.safe_return_to_start(
            trajectory=self._smoothed_trajectory,
            done_callback=self._on_home_reached,
            abort_callback=self._on_home_abort,
        )
        if not ok:
            self._add_event_log("safe_return_to_start refused — reverting to TRAJECTORY_CHECK")
            self._set_state(State.TRAJECTORY_CHECK)
            reason = self._piper._last_transition_error or "safe_return_to_start refused"
            return {"ok": False, "reason": reason}

        return {"ok": True}

    def return_home_direct(self) -> dict:
        """역방향 궤적 없이 SAFE_RETURN_WAYPOINTS 경유로 바로 홈 복귀."""
        with self._lock:
            if self._state != State.TRAJECTORY_CHECK:
                return {"ok": False, "reason": "Must be in TRAJECTORY_CHECK state"}
            self._state = State.RETURN_HOME
        if self._piper is None:
            self._set_state(State.TRAJECTORY_CHECK)
            return {"ok": False, "reason": "Robot not connected"}

        self._on_emit("mode_change", self.get_status())
        self._add_event_log(f"State → {State.RETURN_HOME.value}")
        self._add_event_log("Returning home directly via safe waypoints (no backtrace)")

        teach_start = self._get_teach_start_position()
        if teach_start is None:
            self._set_state(State.TRAJECTORY_CHECK)
            return {"ok": False, "reason": "No teach start position available"}
        ok = self._piper.go_to_safe_ready(
            done_callback=self._on_home_reached,
            abort_callback=self._on_home_abort,
            final_position=teach_start,
        )
        if not ok:
            self._add_event_log("go_to_safe_ready refused — reverting to TRAJECTORY_CHECK")
            self._set_state(State.TRAJECTORY_CHECK)
            reason = self._piper._last_transition_error or (
                "SAFE_RETURN_WAYPOINTS not configured. Set them in the Setup page first."
            )
            return {"ok": False, "reason": reason}

        return {"ok": True}

    def _get_teach_start_position(self) -> Optional[list]:
        """smoothed trajectory의 첫 번째 waypoint (teaching 시작 위치). 없으면 None."""
        if self._smoothed_trajectory:
            first = self._smoothed_trajectory[0]
            return [first.get(f"q{i+1}", 0.0) for i in range(6)]
        if self._teach_buffer:
            first = self._teach_buffer[0]
            return [first.get(f"q{i+1}", 0.0) for i in range(6)]
        return None  # [0]*6 zero-pose fallback 금지

    _START_POSE_TOL = 0.08   # rad: 이 이상이면 replay 차단

    def _on_home_reached(self):
        """홈 복귀 완료 — start pose 오차 확인 후 REPLAY_READY 또는 TRAJECTORY_CHECK."""
        if self._piper and self._smoothed_trajectory:
            target = [self._smoothed_trajectory[0].get(f"q{i+1}", 0.0) for i in range(6)]
            cur = self._piper.get_state().get("position", [])
            if len(cur) >= 6:
                err = max(abs(cur[i] - target[i]) for i in range(6))
                if err > self._START_POSE_TOL:
                    self._add_event_log(
                        f"Start pose error {err:.3f} rad > {self._START_POSE_TOL} — "
                        f"replay blocked. Return home again or retake."
                    )
                    self._set_state(State.TRAJECTORY_CHECK)
                    return
                self._add_event_log(f"Start pose OK: err={err:.3f} rad")

        self._add_event_log("Robot reached start position")
        self._set_state(State.REPLAY_READY)

    def _on_home_abort(self):
        """홈 복귀 실패(waypoint timeout) — TRAJECTORY_CHECK로 복귀해 재시도 허용."""
        self._add_event_log("Return home failed (timeout) — reverted to TRAJECTORY_CHECK. Retry or discard.")
        self._set_state(State.TRAJECTORY_CHECK)

    def _is_at_trajectory_start(self, tol: float = 0.06) -> tuple:
        """현재 위치가 trajectory 시작점에 충분히 가까운지 확인."""
        if not self._smoothed_trajectory or not self._piper:
            return False, 999.0
        traj0 = [self._smoothed_trajectory[0].get(f"q{i+1}", 0.0) for i in range(6)]
        cur = self._piper.get_state().get("position", [])
        if len(cur) < 6:
            return False, 999.0
        err = max(abs(cur[i] - traj0[i]) for i in range(6))
        return err < tol, round(err, 4)

    def start_replay(self, speed_scale: Optional[float] = None) -> dict:
        replay_lock = getattr(self, "_replay_start_lock", None)
        if replay_lock is None:
            replay_lock = threading.Lock()
            self._replay_start_lock = replay_lock
        if not replay_lock.acquire(blocking=False):
            return {"ok": False, "reason": "Replay start is already in progress"}
        try:
            return self._start_replay_once(speed_scale)
        finally:
            replay_lock.release()

    def _start_replay_once(self, speed_scale: Optional[float] = None) -> dict:
        def reject(reason: str, **details) -> dict:
            self._add_event_log(f"Replay start rejected: {reason}")
            return {"ok": False, "reason": reason, **details}

        if self.get_state() != State.REPLAY_READY:
            return reject("Must be in REPLAY_READY state")
        if not self._smoothed_trajectory:
            return reject("No trajectory available")
        if self._piper is None:
            return reject("Robot not connected")

        at_start, err = self._is_at_trajectory_start(tol=0.06)
        if not at_start:
            return reject(
                f"Robot not at trajectory start (err={err} rad > 0.06). Return home first."
            )

        speed = speed_scale if speed_scale is not None else self._replay_speed
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            return reject("Invalid replay speed")
        base_max_velocity = float(
            self._trajectory_check.get("max_joint_velocity_rad_s", 0.0) or 0.0
        )
        effective_max_velocity = base_max_velocity * speed
        speed_adjusted = False
        requested_speed = speed
        if effective_max_velocity > MAX_JOINT_VELOCITY_RAD_S:
            safe_speed = (
                MAX_JOINT_VELOCITY_RAD_S / base_max_velocity
                if base_max_velocity > 0 else 0.0
            )
            # A recorded max can put a nominal 50% setting a few thousandths
            # above the limit.  Clamp only this near-boundary case; materially
            # unsafe requests (for example 70% here) remain explicit failures.
            if 0.0 < speed - safe_speed <= 0.02:
                speed = max(0.05, safe_speed * 0.99)
                effective_max_velocity = base_max_velocity * speed
                speed_adjusted = True
                self._add_event_log(
                    f"Replay speed safety-clamped: {requested_speed:.1%} → {speed:.1%}"
                )
            else:
                recommended = max(0.0, int(safe_speed * 1000) / 1000.0)
                return reject(
                    f"Replay speed {speed:.0%} requires {effective_max_velocity:.3f}rad/s, "
                    f"above the {MAX_JOINT_VELOCITY_RAD_S:.3f}rad/s limit. "
                    f"Use {recommended:.1%} or slower.",
                    requested_speed=speed,
                    recommended_max_speed=recommended,
                    effective_max_velocity_rad_s=round(effective_max_velocity, 4),
                )
        self._replay_speed = speed   # cleanup 시 동일한 속도로 비디오 인코딩
        self._replay_partial = False
        self._replay_manual_stop_requested = False
        self._set_state(State.REPLAY_RECORDING)

        # 카메라 녹화 시작 (연결된 모든 카메라)
        episode_dir = self._episode.get_episode_dir()
        if self._cameras:
            self._cameras.start_all_recording(episode_dir)
        self._add_event_log(f"Replay started (speed={speed})")

        # 로봇 replay
        started = self._piper.start_replay(
            self._smoothed_trajectory,
            speed_scale=speed,
            done_callback=self._on_replay_done,
        )
        if not started:
            if self._cameras:
                self._cameras.stop_all_recording()
            reason = self._piper._last_transition_error or "failed to prepare joint control"
            self._add_event_log(f"Replay start failed: {reason}")
            self._set_state(State.REPLAY_READY)
            return reject(reason)
        return {
            "ok": True,
            "speed_scale_applied": round(speed, 4),
            "speed_adjusted": speed_adjusted,
        }

    def _on_replay_done(self):
        worker = self._piper.get_replay_worker_status().get("result", {}) if self._piper else {}
        stopped = bool(worker.get("stopped"))
        if worker and not worker.get("ok") and not stopped:
            reason = worker.get("error") or "replay worker failed"
            self._add_event_log(f"Replay aborted: {reason}")
            self._stop_replay_cleanup(
                partial=True,
                abort_reason=reason,
                abort_class=worker.get("abort_class") or "MOTION_ABORTED",
                worker_diagnostics=worker,
            )
            return
        if stopped:
            self._add_event_log("Replay stopped by operator — saving partial recording")
        self._stop_replay_cleanup(partial=stopped)

    def stop_replay(self) -> dict:
        """사용자가 수동으로 멈춤"""
        if self.get_state() != State.REPLAY_RECORDING:
            return {"ok": False, "reason": "Not in REPLAY_RECORDING state"}
        self._replay_manual_stop_requested = True
        self._replay_partial = True
        # The subprocess owns command timing and flushes its CSV only on exit.
        # Wait for the monitor to collect it before starting episode processing.
        if self._piper:
            self._piper.stop_replay()
        self._stop_replay_cleanup(partial=True)
        return {"ok": True, "partial": True, "next_state": "PROCESSING"}

    def _stop_replay_cleanup(
        self,
        partial: bool = False,
        abort_reason: str = "",
        abort_class: str | None = None,
        worker_diagnostics: Optional[dict] = None,
    ):
        # 이중 호출 방지: 상태를 원자적으로 확인 후 변경
        # (자연 완료 시 replay thread 내부에서, 수동 중단 시 main thread에서 모두 호출될 수 있음)
        with self._lock:
            if self._state != State.REPLAY_RECORDING:
                return  # 이미 cleanup 완료
            self._replay_partial = bool(partial or self._replay_partial)
            self._state = State.PROCESSING  # ← replay 완료, 후처리 시작

        # PROCESSING 상태 즉시 알림 (프론트엔드가 진행 표시 시작)
        self._on_emit("mode_change", self.get_status())

        def _progress(step: str, status: str, detail: str = ""):
            self._on_emit("save_progress", {"step": step, "status": status, "detail": detail})

        # ── 카메라 녹화 중단 ───────────────────────────────────────────
        _progress("stop_cameras", "running")
        cam_buffers: dict = {}
        if self._cameras:
            cam_buffers = self._cameras.stop_all_recording()
        total_cam = sum(len(v) for v in cam_buffers.values())
        _progress("stop_cameras", "ok", f"{total_cam} frames")

        # ── 로봇 버퍼 + CSV flush ─────────────────────────────────────
        _progress("flush_csv", "running")
        exec_buf = self._piper.get_executed_buffer() if self._piper else []
        cmd_buf = self._piper.get_command_buffer() if self._piper else []
        self._episode.flush_executed_joint(exec_buf)
        self._episode.flush_replay_command(cmd_buf)
        self._episode.flush_camera_frames_multi(cam_buffers)
        self._episode.flush_events(self._piper.get_events() if self._piper else [])
        self._episode.mark_replay_result(
            partial=self._replay_partial,
            command_samples=len(cmd_buf),
            executed_samples=len(exec_buf),
            reason=abort_reason or ("operator_stop" if self._replay_partial else "completed"),
            abort_class=abort_class,
            export_excluded_default=bool(abort_class),
            diagnostics=worker_diagnostics,
        )
        _progress("flush_csv", "ok", f"exec={len(exec_buf)} cmd={len(cmd_buf)}")

        # ── 프레임 정렬 ───────────────────────────────────────────────
        _progress("align_frames", "running")
        self._episode.generate_aligned_frames()
        _progress("align_frames", "ok")

        # ── 비디오 인코딩 ─────────────────────────────────────────────
        self._episode.generate_preview_videos(
            replay_speed=self._replay_speed,
            on_step=_progress,
        )

        # ── REVIEW 전환 ───────────────────────────────────────────────
        with self._lock:
            self._state = State.REVIEW
        self._on_emit("mode_change", self.get_status())
        completion = "partial" if self._replay_partial else "complete"
        self._add_event_log(
            f"Replay {completion} → REVIEW (cam={total_cam}, exec={len(exec_buf)} samples)"
        )

    def save_episode(self, success: bool, reason: str = "") -> dict:
        if self.get_state() != State.REVIEW:
            return {"ok": False, "reason": "Not in REVIEW state"}
        result = self._episode.finalize_episode(success=success, reason=reason)
        quality = self._episode.compute_tracking_quality()
        self._set_state(State.SAVED)
        self._add_event_log(f"Episode saved: success={success}")
        return {"ok": True, **result, "quality": quality}

    def discard_episode(self) -> dict:
        state = self.get_state()
        if state not in (State.TRAJECTORY_CHECK, State.REPLAY_READY, State.REVIEW):
            return {"ok": False, "reason": f"Cannot discard in state {state.value}"}
        self._episode.discard_episode()
        self._set_state(State.DISCARDED)
        self._add_event_log("Episode discarded")
        return {"ok": True}

    def new_episode(self) -> dict:
        state = self.get_state()
        if state not in (State.SAVED, State.DISCARDED):
            return {"ok": False, "reason": "Episode not finalized yet"}
        # 버퍼 초기화 + episode 참조 해제 (start_teach에서 새 episode 생성하도록)
        self._teach_buffer = []
        self._smoothed_trajectory = []
        self._trajectory_check = {}
        self._episode.current_episode_id = None
        self._episode.current_episode_dir = None
        self._episode.current_take_dir = None
        self._episode._take_num = 0
        if self._piper:
            self._piper.clear_events()
            self._piper.clear_gripper_close_arm_pos()

        # 로봇이 연결되어 있으면 safe_ready 위치로 복귀 후 TEACH_READY
        def _on_home():
            self._add_event_log("Robot at safe ready pose — ready for new episode")
            self._set_state(State.TEACH_READY)

        def _on_home_abort():
            self._add_event_log("Return home failed — reverted to TEACH_READY")
            self._set_state(State.TEACH_READY)

        if self._piper:
            self._set_state(State.RETURN_HOME)
            ok = self._piper.go_to_safe_ready(done_callback=_on_home, abort_callback=_on_home_abort)
            if not ok:
                self._add_event_log("SAFE_RETURN_WAYPOINTS not configured — skipping home move")
                self._set_state(State.TEACH_READY)
        else:
            self._set_state(State.TEACH_READY)

        return {"ok": True}

    def retake_replay(self) -> dict:
        """REVIEW → 동일 trajectory로 재촬영 (속도 변경 가능).
        같은 episode에 새 take를 추가하고 홈 복귀 후 REPLAY_READY."""
        if self.get_state() != State.REVIEW:
            return {"ok": False, "reason": "Not in REVIEW state"}
        if not self._smoothed_trajectory:
            return {"ok": False, "reason": "No trajectory to replay"}
        # 같은 episode에 새 take 추가 (episode 폐기 없음)
        self._episode.add_take()
        if self._piper:
            self._piper.clear_events()
        self._set_state(State.RETURN_HOME)
        self._add_event_log(f"Retake replay (take {self._episode.get_current_take_name()}) — reverse return to teach_start")
        if self._piper:
            ok = self._piper.safe_return_to_start(
                trajectory=self._smoothed_trajectory,
                done_callback=self._on_home_reached,
            )
            if not ok:
                self._add_event_log("safe_return_to_start refused — reverting to TRAJECTORY_CHECK")
                self._set_state(State.TRAJECTORY_CHECK)
        return {"ok": True}

    def retake_teach(self) -> dict:
        """REVIEW → 새 Teaching. 같은 episode에 새 take 추가 후 홈 복귀 → TEACH_READY."""
        if self.get_state() != State.REVIEW:
            return {"ok": False, "reason": "Not in REVIEW state"}
        # 같은 episode에 새 take 추가
        self._episode.add_take()
        self._teach_buffer = []
        self._smoothed_trajectory = []
        self._trajectory_check = {}
        if self._piper:
            self._piper.clear_events()
            self._piper.clear_gripper_close_arm_pos()
        self._add_event_log(f"Retake teach (take {self._episode.get_current_take_name()}) — going to safe ready pose")

        def _on_home_for_teach():
            self._add_event_log("Robot at safe ready pose — ready to re-teach")
            self._set_state(State.TEACH_READY)

        def _on_home_for_teach_abort():
            self._add_event_log("Return home failed — reverted to TEACH_READY. Manually move robot before teaching.")
            self._set_state(State.TEACH_READY)

        if self._piper:
            self._set_state(State.RETURN_HOME)
            ok = self._piper.go_to_safe_ready(done_callback=_on_home_for_teach, abort_callback=_on_home_for_teach_abort)
            if not ok:
                self._add_event_log(
                    "SAFE_RETURN_WAYPOINTS not configured — manual reset required before re-teaching"
                )
                self._set_state(State.TEACH_READY)
        else:
            self._set_state(State.TEACH_READY)
        return {"ok": True}

    def add_take_to_episode(self) -> dict:
        """SAVED → 같은 episode에 새 take 추가 후 홈 복귀 → TEACH_READY."""
        if self.get_state() != State.SAVED:
            return {"ok": False, "reason": "Not in SAVED state"}
        self._episode.add_take()
        self._teach_buffer = []
        self._smoothed_trajectory = []
        self._trajectory_check = {}
        if self._piper:
            self._piper.clear_events()
        self._add_event_log(f"Adding take {self._episode.get_current_take_name()} to episode — going to safe ready pose")

        def _on_home_for_teach():
            self._add_event_log("Robot at safe ready pose — ready for new take")
            self._set_state(State.TEACH_READY)

        def _on_home_for_new_take_abort():
            self._add_event_log("Return home failed — reverted to TEACH_READY. Manually move robot before new take.")
            self._set_state(State.TEACH_READY)

        if self._piper:
            self._set_state(State.RETURN_HOME)
            ok = self._piper.go_to_safe_ready(done_callback=_on_home_for_teach, abort_callback=_on_home_for_new_take_abort)
            if not ok:
                self._add_event_log(
                    "SAFE_RETURN_WAYPOINTS not configured — manual reset required before new take"
                )
                self._set_state(State.TEACH_READY)
        else:
            self._set_state(State.TEACH_READY)
        return {"ok": True, "episode_id": self._episode.current_episode_id}

    # ─────────────────────────────────────────────────────────────────
    # Dataset debug replay (no camera recording, isolated output)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _debug_idle_states() -> set[State]:
        return {
            State.IDLE,
            State.READY,
            State.TEACH_READY,
            State.REVIEW,
            State.SAVED,
            State.DISCARDED,
        }

    def get_debug_replay_status(self) -> dict:
        with self._debug_lock:
            source = {
                key: value for key, value in self._debug_source.items()
                if key != "take_dir"
            }
            loaded = bool(self._debug_trajectory)
            operation = self._debug_operation
            start_ready = self._debug_start_ready
            error = self._debug_last_error
            run_dir = self._debug_run_dir
            diagnostics = dict(self._debug_last_diagnostics)
            current_sequence = getattr(self, "_debug_sequence", None)
            sequence = dict(current_sequence) if current_sequence else None
            if sequence:
                sequence.pop("entries", None)
                sequence.pop("root", None)
                sequence.pop("previous_state", None)
                sequence.pop("results", None)

        start_error = None
        if loaded and self._piper is not None:
            q0 = [self._debug_trajectory[0][f"q{i}"] for i in range(1, 7)]
            current = self._piper.get_state().get("position", [])
            if len(current) >= 6:
                start_error = max(abs(float(current[i]) - q0[i]) for i in range(6))
                if start_error > 0.06:
                    start_ready = False

        state = self.get_state()
        idle = state in self._debug_idle_states()
        return {
            "ok": True,
            "loaded": loaded,
            "source": source,
            "trajectory_check": self._debug_check,
            "operation": operation,
            "start_ready": bool(start_ready),
            "start_error_rad": round(start_error, 4) if start_error is not None else None,
            "robot_connected": self._piper is not None,
            "controller_state": state.value,
            "can_return": bool(loaded and self._piper is not None and idle and not operation),
            "can_start": bool(
                loaded and self._piper is not None and idle and not operation
                and start_ready and start_error is not None and start_error <= 0.06
            ),
            "last_error": error,
            "last_run_dir": os.path.relpath(run_dir, DATASET_PATH) if run_dir else None,
            "last_diagnostics": diagnostics,
            "worker": self._piper.get_replay_worker_status() if self._piper else {},
            "sequence": sequence,
        }

    def load_debug_trajectory(
        self,
        take_dir: str,
        episode_id: str,
        take: str,
        source: str = "teach",
    ) -> dict:
        if source not in {"teach", "command"}:
            return {"ok": False, "reason": "source must be teach or command"}
        if self.get_state() not in self._debug_idle_states():
            return {"ok": False, "reason": f"Robot is busy in {self.get_state().value}"}
        with self._debug_lock:
            if self._debug_operation:
                return {"ok": False, "reason": "A debug motion is already running"}

        filename = "teach_joint.csv" if source == "teach" else "replay_command.csv"
        path = os.path.join(take_dir, filename)
        if not os.path.isfile(path):
            return {"ok": False, "reason": f"Missing {filename}"}

        rows: list[dict] = []
        try:
            with open(path, newline="") as file:
                for raw in csv.DictReader(file):
                    q = [float(raw[f"q{i}"]) for i in range(1, 7)]
                    if not all(np.isfinite(value) for value in q):
                        continue
                    row = {f"q{i+1}": q[i] for i in range(6)}
                    row["t_host_ns"] = int(float(raw.get("t_host_ns", 0)))
                    gripper = raw.get("gripper")
                    row["gripper"] = (
                        float(gripper)
                        if gripper not in (None, "") and np.isfinite(float(gripper))
                        else None
                    )
                    rows.append(row)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "reason": f"Failed to read trajectory: {exc}"}
        if len(rows) < 10:
            return {"ok": False, "reason": f"Trajectory too short ({len(rows)} samples)"}

        if source == "teach":
            trajectory, check = PiperTeachReplayNode.smooth_trajectory(rows, REPLAY_DEFAULT_SPEED)
        else:
            trajectory = rows
            q = np.array([[row[f"q{i}"] for i in range(1, 7)] for row in rows])
            max_velocity = float(np.max(np.abs(np.diff(q, axis=0))) * 50.0)
            check = {
                "ok": True,
                "samples": len(rows),
                "duration_s": round((len(rows) - 1) / 50.0, 2),
                "max_joint_velocity_rad_s": round(max_velocity, 4),
                "velocity_limit_ok": max_velocity <= MAX_JOINT_VELOCITY_RAD_S,
                "source": "replay_command.csv",
            }

        if not trajectory or not all(
            np.isfinite(row.get(f"q{i}", float("nan")))
            for row in trajectory for i in range(1, 7)
        ):
            return {"ok": False, "reason": "Trajectory contains invalid joint values"}

        with self._debug_lock:
            self._debug_trajectory = trajectory
            self._debug_source = {
                "episode_id": episode_id,
                "take": take,
                "source": source,
                "filename": filename,
                "samples": len(trajectory),
                "take_dir": take_dir,
            }
            self._debug_check = check
            self._debug_start_ready = False
            self._debug_last_error = ""
            self._debug_last_diagnostics = {}
        self._add_event_log(
            f"Debug trajectory loaded: {episode_id}/{take}/{filename} ({len(trajectory)} samples)"
        )
        return self.get_debug_replay_status()

    def _restore_after_debug_operation(self, expected: str) -> None:
        with self._debug_lock:
            if self._debug_operation != expected:
                return
            previous = self._debug_previous_state or State.READY
            self._debug_operation = None
            self._debug_previous_state = None
        self._set_state(previous)

    def return_debug_replay_to_start(self) -> dict:
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        state = self.get_state()
        if state not in self._debug_idle_states() or state == State.IDLE:
            return {"ok": False, "reason": f"Robot is not ready in {state.value}"}
        with self._debug_lock:
            if not self._debug_trajectory:
                return {"ok": False, "reason": "Load a debug trajectory first"}
            if self._debug_operation:
                return {"ok": False, "reason": "A debug motion is already running"}
            target = [self._debug_trajectory[0][f"q{i}"] for i in range(1, 7)]
            self._debug_operation = "return"
            self._debug_previous_state = state
            self._debug_start_ready = False
            self._debug_last_error = ""

        self._set_state(State.DEBUG_RETURN)

        def done():
            with self._debug_lock:
                self._debug_start_ready = True
            self._add_event_log("Debug replay start pose reached")
            self._restore_after_debug_operation("return")

        def abort():
            with self._debug_lock:
                self._debug_last_error = self._piper._last_transition_error or "Return to start failed"
            self._restore_after_debug_operation("return")

        started = self._piper.go_to_safe_ready(
            final_position=target,
            done_callback=done,
            abort_callback=abort,
        )
        if not started:
            with self._debug_lock:
                self._debug_last_error = self._piper._last_transition_error or "Return to start refused"
            self._restore_after_debug_operation("return")
            return {"ok": False, "reason": self._debug_last_error}
        return {"ok": True, "operation": "return"}

    def _allocate_debug_run_dir(self) -> str:
        take_dir = str(self._debug_source["take_dir"])
        root = os.path.join(take_dir, "debug_replays")
        os.makedirs(root, exist_ok=True)
        numbers = []
        for name in os.listdir(root):
            if name.startswith("run_") and name[4:].isdigit():
                numbers.append(int(name[4:]))
        path = os.path.join(root, f"run_{max(numbers, default=0) + 1:03d}")
        os.makedirs(path, exist_ok=False)
        return path

    @staticmethod
    def _write_debug_csv(path: str, rows: list[dict], fields: list[str]) -> None:
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _debug_diagnostics(
        command: list[dict], executed: list[dict], worker: Optional[dict] = None
    ) -> dict:
        result: dict = {
            "command_samples": len(command),
            "executed_samples": len(executed),
        }
        if worker:
            started = worker.get("started_at")
            finished = worker.get("finished_at")
            if started is not None and finished is not None:
                result["duration_s"] = round(float(finished) - float(started), 3)
            result["gripper_timing"] = [
                {
                    "source_idx": event.get("source_idx"),
                    "hold_to_converged_ms": round(
                        (int(event.get("converged_t_host_ns", 0))
                         - int(event.get("hold_started_t_host_ns", 0))) * 1e-6, 3
                    ),
                    "gripper_motion_ms": round(
                        (int(event.get("gripper_finished_t_host_ns", 0))
                         - int(event.get("converged_t_host_ns", 0))) * 1e-6, 3
                    ),
                    "resume_ramp_ms": round(
                        (int(event.get("resume_finished_t_host_ns", 0))
                         - int(event.get("resume_started_t_host_ns", 0))) * 1e-6, 3
                    ),
                }
                for event in worker.get("gripper_events", [])
            ]
        if command:
            result["max_lateness_ms"] = round(max(float(row.get("lateness_ms", 0.0)) for row in command), 3)
            indices = [float(row.get("source_idx", 0.0)) for row in command]
            result["max_source_step"] = round(
                max((indices[i+1] - indices[i] for i in range(len(indices)-1)), default=0.0), 4
            )
        if len(command) > 1:
            command_times = np.array([int(row["t_host_ns"]) for row in command], dtype=np.int64)
            command_gaps = np.diff(command_times) * 1e-6
            pause_ranges = [
                (
                    int(event.get("hold_started_t_host_ns", 0)),
                    int(
                        event.get("resume_started_t_host_ns")
                        or event.get("resume_finished_t_host_ns", 0)
                    ),
                )
                for event in (worker or {}).get("gripper_events", [])
            ]
            motion_gaps = [
                float(gap) for index, gap in enumerate(command_gaps)
                if not any(
                    int(command_times[index]) <= start
                    and int(command_times[index + 1]) >= finish
                    for start, finish in pause_ranges if start and finish
                )
            ]
            result["command_gap_max_ms"] = round(max(motion_gaps, default=0.0), 3)
            result["command_gap_p95_ms"] = round(
                float(np.percentile(motion_gaps, 95)) if motion_gaps else 0.0, 3
            )
            result["command_gaps_over_50ms"] = sum(gap > 50.0 for gap in motion_gaps)
            result["gripper_pause_count"] = len(pause_ranges)
        if len(executed) > 1:
            times = np.array([int(row["t_host_ns"]) for row in executed], dtype=np.int64)
            gaps = np.diff(times) * 1e-6
            result["feedback_gap_max_ms"] = round(float(np.max(gaps)), 3)
            result["feedback_gaps_over_50ms"] = int(np.sum(gaps > 50.0))
        if len(command) > 1 and executed:
            cmd_t = np.array([int(row["t_host_ns"]) for row in command], dtype=np.float64)
            exec_rows = [row for row in executed if cmd_t[0] <= int(row["t_host_ns"]) <= cmd_t[-1]]
            if exec_rows:
                exec_t = np.array([int(row["t_host_ns"]) for row in exec_rows], dtype=np.float64)
                errors = []
                for joint in range(1, 7):
                    cmd_q = np.array([float(row[f"q{joint}"]) for row in command])
                    exec_q = np.array([float(row[f"q{joint}"]) for row in exec_rows])
                    errors.append(np.abs(exec_q - np.interp(exec_t, cmd_t, cmd_q)))
                err = np.stack(errors, axis=1)
                result["tracking_error_mean_rad"] = np.mean(err, axis=0).round(5).tolist()
                result["tracking_error_max_rad"] = np.max(err, axis=0).round(5).tolist()
        return result

    def _finalize_debug_replay(self) -> None:
        with self._debug_lock:
            if self._debug_operation != "replay":
                return
            # Atomically claim finalization so Stop and the worker callback
            # cannot both flush/restore the same debug run.
            self._debug_operation = "replay_finalizing"
            run_dir = self._debug_run_dir
            stopped = self._debug_stop_requested
            source = {key: value for key, value in self._debug_source.items() if key != "take_dir"}
            started_at = self._debug_started_at
            speed = self._debug_speed
        command = self._piper.get_command_buffer() if self._piper else []
        executed = self._piper.get_executed_buffer() if self._piper else []
        worker_result = (
            self._piper.get_replay_worker_status().get("result", {})
            if self._piper else {}
        )
        diagnostics = self._debug_diagnostics(command, executed, worker_result)
        last_source_idx = float(command[-1].get("source_idx", -1.0)) if command else -1.0
        expected_last_idx = max(0, len(self._debug_trajectory) - 1)
        completed = bool(
            not stopped
            and worker_result.get("ok", command and last_source_idx >= expected_last_idx - 1e-3)
        )
        diagnostics.update({
            "completed": completed,
            "stopped_by_user": stopped,
            "speed_scale": speed,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "source": source,
            "trajectory_check": self._debug_check,
            "worker": worker_result,
        })
        try:
            if run_dir:
                joint_fields = ["t_host_ns"] + [f"q{i}" for i in range(1, 7)] + [f"dq{i}" for i in range(1, 7)] + ["gripper", "mode"]
                command_fields = ["t_host_ns", "scheduled_t_host_ns", "waypoint_idx", "source_idx", "lateness_ms", "velocity"] + [f"q{i}" for i in range(1, 7)] + ["gripper", "gripper_effort"]
                self._write_debug_csv(os.path.join(run_dir, "executed_joint.csv"), executed, joint_fields)
                self._write_debug_csv(os.path.join(run_dir, "replay_command.csv"), command, command_fields)
                with open(os.path.join(run_dir, "diagnostics.json"), "w") as file:
                    json.dump(diagnostics, file, ensure_ascii=False, indent=2)
        except OSError as exc:
            diagnostics["save_error"] = str(exc)
            with self._debug_lock:
                self._debug_last_error = f"Debug result save failed: {exc}"
        with self._debug_lock:
            self._debug_last_diagnostics = diagnostics
            self._debug_start_ready = False
        self._add_event_log(
            f"Debug replay {'stopped' if stopped else 'completed'}: "
            f"cmd={len(command)} exec={len(executed)}"
        )
        self._restore_after_debug_operation("replay_finalizing")

    def start_debug_replay(self, speed_scale: float = 0.5) -> dict:
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        state = self.get_state()
        if state not in self._debug_idle_states() or state == State.IDLE:
            return {"ok": False, "reason": f"Robot is not ready in {state.value}"}
        try:
            speed = float(speed_scale)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "Invalid speed"}
        if speed not in {0.3, 0.5, 0.7}:
            return {"ok": False, "reason": "Debug speed must be 0.3, 0.5, or 0.7"}

        with self._debug_lock:
            if not self._debug_trajectory:
                return {"ok": False, "reason": "Load a debug trajectory first"}
            if self._debug_operation:
                return {"ok": False, "reason": "A debug motion is already running"}
            q0 = [self._debug_trajectory[0][f"q{i}"] for i in range(1, 7)]
            current = self._piper.get_state().get("position", [])
            error = max(abs(float(current[i]) - q0[i]) for i in range(6)) if len(current) >= 6 else float("inf")
            if not self._debug_start_ready or error > 0.06:
                self._debug_start_ready = False
                return {"ok": False, "reason": f"Return to start first (err={error:.4f}rad)"}
            base_velocity = float(self._debug_check.get("max_joint_velocity_rad_s", 0.0) or 0.0)
            if base_velocity * speed > MAX_JOINT_VELOCITY_RAD_S:
                return {"ok": False, "reason": "Selected speed exceeds the joint velocity limit"}
            try:
                run_dir = self._allocate_debug_run_dir()
            except OSError as exc:
                return {"ok": False, "reason": f"Could not create debug run: {exc}"}
            self._debug_operation = "replay"
            self._debug_previous_state = state
            self._debug_run_dir = run_dir
            self._debug_started_at = datetime.now().isoformat()
            self._debug_speed = speed
            self._debug_stop_requested = False
            self._debug_last_error = ""
            trajectory = list(self._debug_trajectory)

        self._set_state(State.DEBUG_REPLAY)
        started = self._piper.start_replay(
            trajectory,
            speed_scale=speed,
            done_callback=self._finalize_debug_replay,
        )
        if not started:
            with self._debug_lock:
                self._debug_last_error = self._piper._last_transition_error or "Debug replay start failed"
            self._restore_after_debug_operation("replay")
            return {"ok": False, "reason": self._debug_last_error}
        self._add_event_log(f"Debug replay started without recording (speed={speed:.0%})")
        return {"ok": True, "run": os.path.basename(run_dir)}

    def stop_debug_replay(self) -> dict:
        with self._debug_lock:
            sequence_active = bool(self._debug_sequence and self._debug_sequence.get("active"))
            if sequence_active:
                self._debug_sequence["cancelled"] = True
            elif self._debug_operation != "replay":
                return {"ok": False, "reason": "No debug replay is running"}
            self._debug_stop_requested = True
        if sequence_active:
            if self._piper:
                self._piper.hold_position()
            self._finish_debug_sequence("stopped by user")
            return {"ok": True, "stopping": True, "sequence": True}
        if self._piper:
            # The worker owns the command CSV and completion signal.  Join it
            # here (with PiperNode's terminate/kill fallback) so a subsequent
            # Return request cannot remain blocked by stale "replay" state.
            self._piper.stop_replay()
        self._finalize_debug_replay()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._debug_lock:
                if self._debug_operation is None:
                    break
            time.sleep(0.01)
        with self._debug_lock:
            if self._debug_operation is not None:
                return {
                    "ok": False,
                    "reason": "Replay stopped, but debug state cleanup timed out",
                }
        return {"ok": True, "stopping": True}

    def start_debug_sequence(self, items: list[dict], cycles=1, speed_scale=0.3) -> dict:
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        state = self.get_state()
        if state not in self._debug_idle_states() or state == State.IDLE:
            return {"ok": False, "reason": f"Robot is not ready in {state.value}"}
        try:
            cycle_count = int(cycles)
            speed = float(speed_scale)
        except (TypeError, ValueError):
            return {"ok": False, "reason": "Invalid cycles or speed"}
        if not 1 <= cycle_count <= 5:
            return {"ok": False, "reason": "cycles must be between 1 and 5"}
        if speed not in {0.3, 0.5, 0.7}:
            return {"ok": False, "reason": "speed_scale must be 0.3, 0.5, or 0.7"}
        if len(items) != 2:
            return {"ok": False, "reason": "A/B sequence requires exactly two items"}
        with self._debug_lock:
            if self._debug_operation or (
                self._debug_sequence and self._debug_sequence.get("active")
            ):
                return {"ok": False, "reason": "A debug motion is already running"}

        saved = (
            list(self._debug_trajectory), dict(self._debug_source), dict(self._debug_check)
        )
        entries = []
        for label, item in zip(("A", "B"), items):
            result = self.load_debug_trajectory(
                take_dir=item["take_dir"],
                episode_id=item["episode_id"],
                take=item["take"],
                source=item.get("source", "teach"),
            )
            if not result.get("ok"):
                with self._debug_lock:
                    self._debug_trajectory, self._debug_source, self._debug_check = saved
                return {"ok": False, "reason": f"{label}: {result.get('reason', 'load failed')}"}
            with self._debug_lock:
                entries.append({
                    "label": label,
                    "trajectory": list(self._debug_trajectory),
                    "source": dict(self._debug_source),
                    "check": dict(self._debug_check),
                })

        sequence_id = datetime.now().strftime("sequence_%Y%m%d_%H%M%S")
        root = os.path.join(DATASET_PATH, "debug_sequences", sequence_id)
        try:
            os.makedirs(root, exist_ok=False)
            manifest = {
                "sequence_id": sequence_id,
                "cycles": cycle_count,
                "speed_scale": speed,
                "items": [
                    {key: value for key, value in entry["source"].items() if key != "take_dir"}
                    for entry in entries
                ],
                "started_at": datetime.now().isoformat(),
            }
            with open(os.path.join(root, "manifest.json"), "w") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)
        except OSError as exc:
            return {"ok": False, "reason": f"Could not create sequence output: {exc}"}

        with self._debug_lock:
            self._debug_sequence = {
                "active": True,
                "cancelled": False,
                "sequence_id": sequence_id,
                "root": root,
                "cycles": cycle_count,
                "cycle": 1,
                "item_index": 0,
                "label": "A",
                "phase": "RETURN",
                "speed_scale": speed,
                "completed_replays": 0,
                "total_replays": cycle_count * 2,
                "results": [],
                "entries": entries,
                "previous_state": state,
                "started_at": datetime.now().isoformat(),
            }
            self._debug_last_error = ""
            self._debug_operation = "sequence_return"
        self._add_event_log(
            f"Debug A/B sequence started: {sequence_id}, cycles={cycle_count}, speed={speed:.0%}"
        )
        self._begin_debug_sequence_item()
        with self._debug_lock:
            if not self._debug_sequence or not self._debug_sequence.get("active"):
                return {
                    "ok": False,
                    "reason": self._debug_last_error or "Sequence start failed",
                }
        return {"ok": True, "sequence_id": sequence_id, "total_replays": cycle_count * 2}

    def _begin_debug_sequence_item(self):
        with self._debug_lock:
            sequence = self._debug_sequence
            if not sequence or not sequence.get("active") or sequence.get("cancelled"):
                return
            entry = sequence["entries"][sequence["item_index"]]
            sequence["label"] = entry["label"]
            sequence["phase"] = "RETURN"
            self._debug_operation = "sequence_return"
            self._debug_trajectory = list(entry["trajectory"])
            self._debug_source = dict(entry["source"])
            self._debug_check = dict(entry["check"])
            self._debug_start_ready = False
            target = [self._debug_trajectory[0][f"q{i}"] for i in range(1, 7)]
        self._set_state(State.DEBUG_RETURN)
        started = self._piper.go_to_safe_ready(
            final_position=target,
            done_callback=self._debug_sequence_return_done,
            abort_callback=self._debug_sequence_return_abort,
        )
        if not started:
            self._finish_debug_sequence(
                self._piper._last_transition_error or "sequence return refused"
            )

    def _debug_sequence_return_done(self):
        with self._debug_lock:
            sequence = self._debug_sequence
            if not sequence or not sequence.get("active") or sequence.get("cancelled"):
                return
            sequence["phase"] = "REPLAY"
            self._debug_operation = "sequence_replay"
            trajectory = list(self._debug_trajectory)
            speed = float(sequence["speed_scale"])
        self._set_state(State.DEBUG_REPLAY)
        if not self._piper.start_replay(
            trajectory, speed_scale=speed, done_callback=self._debug_sequence_replay_done
        ):
            self._finish_debug_sequence(
                self._piper._last_transition_error or "sequence replay start failed"
            )

    def _debug_sequence_return_abort(self):
        self._finish_debug_sequence(
            self._piper._last_transition_error or "sequence return failed"
        )

    def _debug_sequence_replay_done(self):
        with self._debug_lock:
            sequence = self._debug_sequence
            if not sequence or not sequence.get("active"):
                return
            cycle = int(sequence["cycle"])
            label = str(sequence["label"])
            source = {
                key: value for key, value in self._debug_source.items() if key != "take_dir"
            }
            check = dict(self._debug_check)
            output = os.path.join(sequence["root"], f"cycle_{cycle:03d}", label)
        command = self._piper.get_command_buffer()
        executed = self._piper.get_executed_buffer()
        worker_result = self._piper.get_replay_worker_status().get("result", {})
        diagnostics = self._debug_diagnostics(command, executed, worker_result)
        diagnostics.update({
            "source": source,
            "trajectory_check": check,
            "worker": worker_result,
            "cycle": cycle,
            "label": label,
            "finished_at": datetime.now().isoformat(),
            "completed": bool(worker_result.get("ok")),
        })
        try:
            os.makedirs(output, exist_ok=False)
            joint_fields = ["t_host_ns"] + [f"q{i}" for i in range(1, 7)] + [f"dq{i}" for i in range(1, 7)] + ["gripper", "mode"]
            command_fields = ["t_host_ns", "scheduled_t_host_ns", "waypoint_idx", "source_idx", "lateness_ms", "velocity"] + [f"q{i}" for i in range(1, 7)] + ["gripper", "gripper_effort"]
            self._write_debug_csv(os.path.join(output, "executed_joint.csv"), executed, joint_fields)
            self._write_debug_csv(os.path.join(output, "replay_command.csv"), command, command_fields)
            with open(os.path.join(output, "diagnostics.json"), "w") as file:
                json.dump(diagnostics, file, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._finish_debug_sequence(f"sequence result save failed: {exc}")
            return

        with self._debug_lock:
            sequence = self._debug_sequence
            if not sequence or not sequence.get("active"):
                return
            sequence["results"].append(diagnostics)
            sequence["completed_replays"] += 1
            failed = not worker_result.get("ok")
            if not failed:
                if sequence["item_index"] == 0:
                    sequence["item_index"] = 1
                else:
                    sequence["item_index"] = 0
                    sequence["cycle"] += 1
                done = sequence["cycle"] > sequence["cycles"]
            else:
                done = False
        if failed:
            self._finish_debug_sequence(worker_result.get("error") or f"{label} replay failed")
        elif done:
            self._finish_debug_sequence("")
        else:
            self._begin_debug_sequence_item()

    def _finish_debug_sequence(self, error: str):
        with self._debug_lock:
            sequence = self._debug_sequence
            if not sequence or not sequence.get("active"):
                return
            sequence["active"] = False
            sequence["phase"] = "FAILED" if error else "COMPLETED"
            sequence["finished_at"] = datetime.now().isoformat()
            sequence["error"] = error or None
            summary = {
                key: value for key, value in sequence.items()
                if key not in {"entries", "root", "previous_state"}
            }
            by_label = {}
            for label in ("A", "B"):
                rows = [item for item in sequence["results"] if item.get("label") == label]
                if not rows:
                    continue
                by_label[label] = {
                    "runs": len(rows),
                    "duration_s_mean": round(float(np.mean([
                        float(item.get("worker", {}).get("finished_at", 0) or 0)
                        - float(item.get("worker", {}).get("started_at", 0) or 0)
                        for item in rows
                    ])), 3),
                    "command_gap_p95_ms_max": max(
                        float(item.get("command_gap_p95_ms", 0.0)) for item in rows
                    ),
                    "command_gap_max_ms": max(
                        float(item.get("command_gap_max_ms", 0.0)) for item in rows
                    ),
                    "feedback_gap_max_ms": max(
                        float(item.get("feedback_gap_max_ms", 0.0)) for item in rows
                    ),
                    "tracking_error_max_rad": [
                        max(
                            float(item.get("tracking_error_max_rad", [0.0] * 6)[joint])
                            for item in rows
                        ) for joint in range(6)
                    ],
                    "gripper_pause_count": sum(
                        int(item.get("gripper_pause_count", 0)) for item in rows
                    ),
                    "gripper_timing": [
                        timing
                        for item in rows
                        for timing in item.get("gripper_timing", [])
                    ],
                }
            summary["episodes"] = by_label
            if "A" in by_label and "B" in by_label:
                summary["ab_difference"] = {
                    "duration_s_mean_B_minus_A": round(
                        by_label["B"]["duration_s_mean"]
                        - by_label["A"]["duration_s_mean"], 3
                    ),
                    "command_gap_p95_ms_B_minus_A": round(
                        by_label["B"]["command_gap_p95_ms_max"]
                        - by_label["A"]["command_gap_p95_ms_max"], 3
                    ),
                    "feedback_gap_max_ms_B_minus_A": round(
                        by_label["B"]["feedback_gap_max_ms"]
                        - by_label["A"]["feedback_gap_max_ms"], 3
                    ),
                    "tracking_error_max_rad_B_minus_A": [
                        round(
                            by_label["B"]["tracking_error_max_rad"][joint]
                            - by_label["A"]["tracking_error_max_rad"][joint], 5
                        ) for joint in range(6)
                    ],
                }
            root = sequence["root"]
            previous = sequence["previous_state"]
            self._debug_operation = None
            self._debug_previous_state = None
            self._debug_last_error = error
            self._debug_last_diagnostics = (
                sequence["results"][-1] if sequence["results"] else {}
            )
        try:
            with open(os.path.join(root, "summary.json"), "w") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2, default=str)
        except OSError as exc:
            with self._debug_lock:
                self._debug_last_error = error or f"sequence summary save failed: {exc}"
        self._set_state(previous)
        self._add_event_log(
            f"Debug A/B sequence {'failed: ' + error if error else 'completed'}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Safety / gripper
    # ─────────────────────────────────────────────────────────────────

    def hold_position(self) -> dict:
        """Soft stop: 현재 position hold"""
        if self._piper:
            if not self._piper.hold_position():
                return {"ok": False, "reason": self._piper._last_transition_error}
            self._add_event_log("HOLD POSITION (soft stop)")
        with self._debug_lock:
            debug_operation = self._debug_operation
            sequence_active = bool(self._debug_sequence and self._debug_sequence.get("active"))
            if debug_operation == "return":
                self._debug_start_ready = False
                self._debug_last_error = "Debug return stopped by Hold"
            elif debug_operation == "replay":
                self._debug_stop_requested = True
            elif sequence_active:
                self._debug_sequence["cancelled"] = True
        if debug_operation == "return":
            self._restore_after_debug_operation("return")
        elif sequence_active:
            self._finish_debug_sequence("stopped by Hold")
        return {"ok": True}

    def motors_enable(self) -> dict:
        """모터 활성화 (Advanced joint control 용)"""
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        self._piper._cancel_pending_mode_timers()
        if not self._piper.enable_motors():
            return {"ok": False, "reason": self._piper._last_transition_error}
        self._add_event_log("Motors enabled (manual)")
        return {"ok": True}

    def motors_recover(self, confirmed: bool = False) -> dict:
        """Operator-confirmed recovery for a latched motor safety fault."""
        if not confirmed:
            return {
                "ok": False,
                "reason": "Operator confirmation is required before motor recovery",
            }
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        result = self._piper.recover_motors()
        if result.get("ok"):
            self._add_event_log("Motor safety fault recovered at current pose")
        else:
            self._add_event_log(
                f"Motor safety recovery failed: {result.get('reason', 'unknown')}"
            )
        return result

    def motors_disable(self) -> dict:
        """모터 비활성화 / freedrive (Advanced joint control 용)"""
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        self._piper._cancel_pending_mode_timers()
        if not self._piper.disable_motors():
            return {"ok": False, "reason": self._piper._last_transition_error}
        self._add_event_log("Motors disabled / freedrive (manual)")
        return {"ok": True}

    def gripper_set(self, pct: float) -> dict:
        """그리퍼를 0~100% 위치로 이동. 0=닫힘, 100=완전 열림."""
        if self._piper:
            self._piper.gripper_set(pct)
        return {"ok": True, "pct": pct}

    def gripper_open(self) -> dict:
        if self._piper:
            self._piper.gripper_open()
        return {"ok": True}

    def gripper_close(self) -> dict:
        if self._piper:
            self._piper.gripper_close()
        return {"ok": True}

    def go_back_to_teach(self) -> dict:
        """TRAJECTORY_CHECK / REPLAY_READY → TEACH_READY로 복귀.
        현재 take가 첫 번째이면 episode 전체 폐기, 아니면 현재 take만 폐기."""
        state = self.get_state()
        if state not in (State.TRAJECTORY_CHECK, State.REPLAY_READY):
            return {"ok": False, "reason": f"Cannot go back from state {state.value}"}
        take_num = self._episode._take_num
        if take_num <= 1:
            self._episode.discard_episode()
            self._add_event_log("Went back to TEACH_READY (episode discarded)")
        else:
            self._episode.discard_current_take()
            self._add_event_log(f"Went back to TEACH_READY (take {take_num} discarded, episode kept)")
        self._teach_buffer = []
        self._smoothed_trajectory = []
        self._trajectory_check = {}
        if self._piper:
            self._piper.clear_events()
        self._set_state(State.TEACH_READY)
        return {"ok": True}

    def set_replay_speed(self, speed: float) -> dict:
        clamped = max(0.1, min(2.0, speed))
        self._replay_speed = clamped
        return {"ok": True, "speed_scale": clamped}

    # ─────────────────────────────────────────────────────────────────
    # ROM Calibration
    # ─────────────────────────────────────────────────────────────────

    def _load_joint_limits(self):
        """서버 시작 시 저장된 joint_limits.json 로드 → piper node에 적용."""
        try:
            if not os.path.exists(JOINT_LIMITS_FILE):
                return
            with open(JOINT_LIMITS_FILE) as f:
                limits = json.load(f)
            if self._piper and self._piper.set_pm_joint_limits(limits):
                self._add_event_log(f"Joint limits loaded from {JOINT_LIMITS_FILE}")
            else:
                self._add_event_log(
                    f"Rejected invalid joint limits from {JOINT_LIMITS_FILE}; defaults preserved"
                )
        except Exception as e:
            self._add_event_log(f"Failed to load joint limits: {e}")

    def start_calibration(self) -> dict:
        """READY / TEACH_READY → CALIBRATING. 모터 OFF, ROM 측정 시작."""
        state = self.get_state()
        if state not in (State.READY, State.TEACH_READY):
            return {"ok": False, "reason": f"Cannot calibrate from state {state.value}"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        self._calib_prev_state = state
        if not self._piper.start_calib():
            return {"ok": False, "reason": self._piper._last_transition_error}
        self._set_state(State.CALIBRATING)
        self._add_event_log("Calibration started — move arm through full ROM")
        return {"ok": True}

    def stop_calibration(self) -> dict:
        """CALIBRATING → 이전 상태(READY / TEACH_READY). limits 저장 + 적용."""
        if self.get_state() != State.CALIBRATING:
            return {"ok": False, "reason": "Not in CALIBRATING state"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}

        result = self._piper.stop_calib()
        if not result.get("ok"):
            self._set_state(self._calib_prev_state)
            self._add_event_log(f"Calibration rejected: {result.get('reason', 'invalid range')}")
            return result
        limits = result["limits"]

        # piper node에 즉시 적용
        if not self._piper.set_pm_joint_limits(limits):
            self._set_state(self._calib_prev_state)
            return {"ok": False, "reason": "Validated calibration could not be applied"}

        # JSON 저장
        try:
            os.makedirs(os.path.dirname(JOINT_LIMITS_FILE), exist_ok=True)
            with open(JOINT_LIMITS_FILE, "w") as f:
                json.dump(limits, f, indent=2)
            self._add_event_log(f"Joint limits saved → {JOINT_LIMITS_FILE}")
        except Exception as e:
            self._add_event_log(f"Failed to save joint limits: {e}")

        self._set_state(self._calib_prev_state)
        measured = [k for k, v in limits.items() if v.get("measured")]
        self._add_event_log(f"Calibration done — measured: {measured}")
        return {"ok": True, "limits": limits}

    def resume_calibration(self) -> dict:
        """CALIBRATING 중 HOLD를 해제하고 기존 min/max 추적을 계속한다."""
        if self.get_state() != State.CALIBRATING:
            return {"ok": False, "reason": "Not in CALIBRATING state"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        if not self._piper.resume_calib():
            return {"ok": False, "reason": self._piper._last_transition_error}
        self._add_event_log("Calibration resumed — motors released, measured range preserved")
        return {"ok": True}

    def enable_phone_mouse(self) -> dict:
        """
        Phone Mouse 모드 진입 시 호출.
        - freedrive 5초 타이머 취소 (모터 OFF 방지)
        - 즉시 teach recording 시작 (motors ON 유지)
        TEACH_RECORDING 상태에서만 유효.
        """
        if self.get_state() not in (State.TEACH_STARTING, State.TEACH_RECORDING):
            return {"ok": False, "reason": "Not in a teach start/recording state"}
        if self._piper is None:
            return {"ok": False, "reason": "Robot not connected"}
        if not self._piper.begin_teach_recording():
            return {"ok": False, "reason": self._piper._last_transition_error}
        self._set_state(State.TEACH_RECORDING)
        self._add_event_log("Phone Mouse enabled — recording started (motors ON)")
        return {"ok": True}

    # ─────────────────────────────────────────────────────────────────
    # Phone Mouse (gyro-based joint delta streaming)
    # ─────────────────────────────────────────────────────────────────

    _PM_MAX_DELTA   = 0.05   # rad/step 최대 관절 이동량
    _PM_TIMEOUT_S   = 0.20   # watchdog 타임아웃 (초)

    def _ensure_pm_state(self):
        """phone mouse 관련 state 필드가 없으면 초기화."""
        if not hasattr(self, "_pm_lock"):
            self._pm_lock = threading.Lock()
            self._pm_last_cmd_time: float = 0.0
            self._pm_active: bool = False

    def handle_phone_mouse_cmd(self, data: dict):
        """
        Socket.IO 'phone_mouse_cmd' 이벤트 핸들러.
        data: { j1, j2, j4 (rad delta), seq }
        TEACH_RECORDING 상태에서만 동작.
        """
        if self._piper is None:
            return
        with self._lock:
            if self._state != State.TEACH_RECORDING:
                return

        self._ensure_pm_state()

        mx = self._PM_MAX_DELTA
        j1 = max(-mx, min(mx, float(data.get("j1", 0.0))))
        j2 = max(-mx, min(mx, float(data.get("j2", 0.0))))
        j4 = max(-mx, min(mx, float(data.get("j4", 0.0))))

        with self._pm_lock:
            self._pm_last_cmd_time = time.time()
            if not self._pm_active:
                self._pm_active = True
                threading.Thread(target=self._pm_watchdog, daemon=True).start()

        self._piper.apply_joint_delta(j1=j1, j2=j2, j4=j4)

    def _pm_watchdog(self):
        """200ms 이내에 새 명령이 없으면 pm_active = False 후 종료."""
        while True:
            time.sleep(0.05)
            with self._pm_lock:
                if time.time() - self._pm_last_cmd_time > self._PM_TIMEOUT_S:
                    self._pm_active = False
                    return

    def handle_ee_twist_cmd(self, data: dict):
        """
        Socket.IO 'ee_twist_cmd' 이벤트 핸들러 — Cartesian jog.
        data: { vx, vy, vz, wx, wy, wz (m/s or rad/s), dt (s) }
        TEACH_RECORDING 상태에서만 동작.
        """
        if self._piper is None:
            return
        with self._lock:
            if self._state != State.TEACH_RECORDING:
                return

        self._ensure_pm_state()
        with self._pm_lock:
            self._pm_last_cmd_time = time.time()
            if not self._pm_active:
                self._pm_active = True
                threading.Thread(target=self._pm_watchdog, daemon=True).start()

        self._piper.apply_ee_twist(
            vx=float(data.get("vx", 0.0)),
            vy=float(data.get("vy", 0.0)),
            vz=float(data.get("vz", 0.0)),
            wx=float(data.get("wx", 0.0)),
            wy=float(data.get("wy", 0.0)),
            wz=float(data.get("wz", 0.0)),
            dt=float(data.get("dt", 0.05)),
        )

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        robot = self._piper.get_state() if self._piper else {}
        return {
            "state": self.get_state().value,
            "piper_connected": self._piper is not None,
            "cameras_connected": self._cameras.any_available if self._cameras else False,
            "ros_ok": rclpy.ok() if rclpy else False,
            "teach_buffer_len": len(self._teach_buffer),
            "trajectory_check": self._trajectory_check,
            "episode_id": self._episode.current_episode_id,
            "event_log_len": len(self._event_log),
            "gripper_responsive": robot.get("gripper_responsive"),
            "motor_enable_status": robot.get("motor_enable_status", []),
            "motor_fault_codes": robot.get("motor_fault_codes", []),
            "motor_faults": robot.get("motor_faults", []),
            "arm_status": robot.get("arm_status", {}),
            "motor_status_age_s": robot.get("motor_status_age_s"),
            "joint_feedback_age_s": robot.get("feedback_age_s"),
            "transition_state": robot.get("transition_state"),
            "transition_error": robot.get("transition_error"),
            "safety_latched": robot.get("safety_latched", False),
            "safety_reason": robot.get("safety_reason"),
            "can_healthy": robot.get("can_healthy", True),
            "abort_class": robot.get("abort_class"),
            "last_motion_abort": robot.get("last_motion_abort", {}),
            "recovery": robot.get("recovery", {}),
            "calibration_span": robot.get("calibration_span", []),
            "calibration_valid": robot.get("calibration_valid", []),
            "debug_replay": self.get_debug_replay_status(),
        }

    # ─────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────

    def shutdown(self):
        self._poll_stop.set()
        if self._cameras:
            try:
                self._cameras.disconnect()
            except Exception:
                pass
        if self._ros_executor:
            try:
                self._ros_executor.shutdown()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
        print("[Controller] Shutdown complete")
