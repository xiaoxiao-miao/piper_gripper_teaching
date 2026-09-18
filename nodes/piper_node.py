"""
PiperTeachReplayNode
---------------------
두 가지 모드:
  TEACH  : /joint_states_single 구독 → teach_buffer 에 (t_host_ns, q1..q6, gripper) 기록
  REPLAY : smoothed trajectory → /joint_ctrl_single 발행 + executed_buffer 기록

gripper는 별도 이벤트(gripper_open / gripper_close)로 처리.
실제 E-stop은 CAN/물리 수준이므로, 여기서는 "현재 position hold" soft-stop만 제공.
"""

import time
import threading
import csv
import json
import shutil
import subprocess
import tempfile
from collections import deque
from typing import Callable, Optional

import numpy as np
from scipy.signal import savgol_filter

try:
    import pinocchio as pin
    _PIN_AVAILABLE = True
except ImportError:
    _PIN_AVAILABLE = False

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, UInt8MultiArray
from piper_msgs.msg import PiperStatusMsg
from piper_msgs.srv import Enable
from std_srvs.srv import Trigger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    ROS_PIPER_JOINT_STATE_TOPIC,
    ROS_PIPER_JOINT_CTRL_TOPIC,
    TEACH_LOG_HZ,
    REPLAY_COMMAND_HZ,
    REPLAY_DEFAULT_SPEED,
    REPLAY_SMOOTHING_WINDOW,
    REPLAY_SMOOTHING_POLYORDER,
    MAX_JOINT_VELOCITY_RAD_S,
    MAX_WAYPOINT_DELTA_RAD,
    HOME_VELOCITY,
    HOME_WAIT_S,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSE_RAD,
    GRIPPER_VELOCITY,
    GRIPPER_STEPS,
    GRIPPER_STEP_DELAY,
    GRIPPER_RESPONSE_MIN_FRAC,
    GRIPPER_RESPONSE_MIN_ABS_RAD,
    MOTOR_HANDSHAKE_TIMEOUT_S,
    JOINT_FEEDBACK_MAX_AGE_S,
    JOINT_FEEDBACK_ABORT_AGE_S,
    MOTION_START_TIMEOUT_S,
    MOTION_START_MIN_DELTA_RAD,
    CALIBRATION_MIN_SPAN_RAD,
    CALIBRATION_LIMIT_TOLERANCE_RAD,
    PIPER_HARDWARE_JOINT_LIMITS,
)

DOF = 6  # Piper 6-DOF arm (그리퍼 제외)


class PiperTeachReplayNode(Node):

    def __init__(self, on_state_update: Optional[Callable] = None):
        """
        on_state_update: 10Hz 폴링 대신 joint callback마다 호출할 콜백 (optional).
                         Controller 가 소켓 emit에 쓸 수 있음.
        """
        super().__init__("piper_teach_replay")

        self.on_state_update = on_state_update

        # ── publishers / subscribers ──────────────────────────────────
        self.pub = self.create_publisher(JointState, ROS_PIPER_JOINT_CTRL_TOPIC, 10)
        self._enable_client = self.create_client(Enable, 'enable_srv')
        self._recover_client = self.create_client(Trigger, 'recover_srv')
        self.motor_status_sub = self.create_subscription(
            UInt8MultiArray, 'motor_enable_status', self._motor_status_callback, 10
        )
        self.motor_fault_sub = self.create_subscription(
            UInt8MultiArray, 'motor_fault_status', self._motor_fault_callback, 10
        )
        self.motor_safety_sub = self.create_subscription(
            Bool, 'motor_safety_latched', self._motor_safety_callback, 10
        )
        self.can_health_sub = self.create_subscription(
            Bool, 'can_health', self._can_health_callback, 10
        )
        self.motor_limits_sub = self.create_subscription(
            Float64MultiArray, 'motor_joint_limits', self._motor_limits_callback, 10
        )
        self.arm_status_sub = self.create_subscription(
            PiperStatusMsg, 'arm_status', self._arm_status_callback, 10
        )
        self.sub = self.create_subscription(
            JointState, ROS_PIPER_JOINT_STATE_TOPIC, self._joint_callback, 10
        )

        # ── shared state ──────────────────────────────────────────────
        self._lock = threading.Lock()
        self._mode = "IDLE"           # IDLE | TEACH | REPLAY | HOLD

        self._latest_position = [0.0] * (DOF + 1)  # q1..q6 + gripper
        self._latest_velocity = [0.0] * (DOF + 1)
        self._latest_t_ns = 0

        # 그리퍼 응답성 — 명령 보낸 뒤 실제 피드백이 안 움직이면(물리 단선 등) False.
        # None = 이번 세션에서 아직 그리퍼를 움직여본 적 없음(판단 불가).
        self._gripper_responsive: Optional[bool] = None

        self._joint_hz_counter = 0
        self._joint_hz = 0.0
        self._hz_window_start = time.time()

        # ── teach buffer ──────────────────────────────────────────────
        self._teach_buffer: list[dict] = []
        self._teach_start_t: Optional[int] = None
        self._last_teach_log_t = 0.0   # host timestamp for throttling
        self._teach_disable_timer: Optional[threading.Timer] = None
        self._mode_generation = 0
        self._motion_generation = 0
        self._motion_active = False
        self._transition_lock = threading.RLock()
        # Never hold the transition metadata lock while waiting for a ROS
        # service future.  Motor feedback is dispatched by the ROS executor
        # and must remain able to run while a request thread waits for it.
        self._motion_state_lock = threading.Lock()
        self._motor_request_lock = threading.Lock()
        self._motor_request_generation = 0

        # ── motor enable state ────────────────────────────────────────
        self._motors_enabled = True
        self._motor_enable_status = [False] * DOF
        self._motor_fault_codes = [0] * DOF
        self._motor_status_t_ns = 0
        self._arm_status = {}
        self._transition_state = "READY"
        self._last_transition_error = ""
        self._safety_latched = False
        self._safety_reason = ""
        self._can_healthy = True
        self._abort_class = None
        self._last_motion_abort = {}
        self._last_recovery_result = {}

        # ── executed buffer (replay 중) ───────────────────────────────
        self._executed_buffer: list[dict] = []

        # ── replay ───────────────────────────────────────────────────
        self._replay_thread: Optional[threading.Thread] = None
        self._replay_stop_event = threading.Event()
        self._replay_command_buffer: list[dict] = []
        self._replay_progress: float = 0.0   # 0.0~1.0, 재생 진행률
        self._replay_worker_enabled = True
        self._replay_process: Optional[subprocess.Popen] = None
        self._replay_worker_dir: Optional[str] = None
        self._replay_worker_status: dict = {"phase": "IDLE"}
        self._last_replay_worker_result: dict = {}

        # ── events log (gripper 등) ───────────────────────────────────
        self._events: list[dict] = []

        # ── Phone Mouse 관절 한계 (calibration 전까지 기본값) ─────────
        # (min_rad, max_rad) per joint — set_pm_joint_limits()로 업데이트
        self._pm_joint_limits: list[tuple] = [
            (-2.618,  2.618),   # j1: base rotation
            (-0.5,    1.8),     # j2: shoulder — 기본 보수값 (calibration 후 갱신)
            (-1.8,    2.618),   # j3: elbow
            (-2.618,  2.618),   # j4: wrist pitch
            (-2.618,  2.618),   # j5: wrist roll
            (-2.618,  2.618),   # j6: wrist yaw
        ]
        self._joint_limits_validated = False
        self._hardware_joint_limits: list[tuple] = []
        self._hardware_joint_max_speed = [0.0] * DOF

        # ── Calibration (ROM 측정) ────────────────────────────────────
        self._calib_active = False
        self._calib_min = [float("inf")]  * DOF
        self._calib_max = [float("-inf")] * DOF

        # ── gripper state (command 기준) ──────────────────────────────
        self._gripper_cmd = 0.0  # 0 = open, 1 = close (normalized)
        # 그리퍼를 닫은 순간의 arm 위치 (return_home 시 해당 위치에서 gripper open)
        self._gripper_close_arm_pos: Optional[list] = None

        # ── Pinocchio (Cartesian jog IK) ─────────────────────────────
        self._pin_model = None
        self._pin_data = None
        self._pin_ee_id = 6  # joint6 = last arm joint (joints 1-6 = arm, 7-8 = gripper)
        if _PIN_AVAILABLE:
            self._load_urdf()

        self.get_logger().info("PiperTeachReplayNode initialized")

    # ─────────────────────────────────────────────────────────────────
    # ROS2 callback
    # ─────────────────────────────────────────────────────────────────

    def _joint_callback(self, msg: JointState):
        t_host_ns = time.time_ns()

        with self._lock:
            if msg.position:
                self._latest_position = list(msg.position[:DOF + 1])
            if msg.velocity:
                self._latest_velocity = list(msg.velocity[:DOF + 1])
            self._latest_t_ns = t_host_ns
            mode = self._mode

            # Hz 계산
            self._joint_hz_counter += 1
            now = time.time()
            elapsed = now - self._hz_window_start
            if elapsed >= 1.0:
                self._joint_hz = self._joint_hz_counter / elapsed
                self._joint_hz_counter = 0
                self._hz_window_start = now

        # Calibration: 관절 min/max 추적
        if self._calib_active:
            with self._lock:
                pos = self._latest_position[:DOF]
            for i in range(DOF):
                self._calib_min[i] = min(self._calib_min[i], pos[i])
                self._calib_max[i] = max(self._calib_max[i], pos[i])

        # TEACH 기록 (throttle to TEACH_LOG_HZ)
        if mode == "TEACH":
            interval = 1.0 / TEACH_LOG_HZ
            now_f = time.time()
            if now_f - self._last_teach_log_t >= interval:
                self._last_teach_log_t = now_f
                row = self._make_joint_row(t_host_ns, tag="teach")
                with self._lock:
                    self._teach_buffer.append(row)

        # REPLAY 중 executed 기록
        elif mode == "REPLAY":
            row = self._make_joint_row(t_host_ns, tag="replay")
            with self._lock:
                self._executed_buffer.append(row)

        if self.on_state_update:
            self.on_state_update(self.get_state())

    def _motor_status_callback(self, msg: UInt8MultiArray):
        status = [bool(value) for value in list(msg.data)[:DOF]]
        if len(status) != DOF:
            return
        with self._lock:
            previous_all = all(self._motor_enable_status)
            self._motor_enable_status = status
            self._motor_status_t_ns = time.time_ns()
            self._motors_enabled = all(status)
        with self._motion_state_lock:
            motion_active = self._motion_active
        if previous_all and motion_active and not all(status):
            self._enter_safety_lock(
                'ENABLE_DROPPED',
                {'motor_enable_status': [int(value) for value in status]},
            )

    def _motor_fault_callback(self, msg: UInt8MultiArray):
        codes = [int(value) & 0xFF for value in list(msg.data)[:DOF]]
        if len(codes) == DOF:
            with self._lock:
                self._motor_fault_codes = codes
            faults = [code & ~(1 << 6) for code in codes]
            if any(faults):
                self._enter_safety_lock('FOC_FAULT', {'fault_codes': faults})

    def _motor_safety_callback(self, msg: Bool):
        if bool(msg.data):
            self._enter_safety_lock(self._safety_reason or 'DRIVER_SAFETY_LATCH')
        elif (
            self._safety_latched
            and self._can_healthy
            and not any(code & ~(1 << 6) for code in self._motor_fault_codes)
        ):
            # Driver is authoritative for its latch.  CAN-link recovery can
            # safely clear the latch while all motors remain disabled; it does
            # not require a prior Cowork-side successful result.
            self._safety_latched = False
            self._safety_reason = ''
            self._abort_class = None

    def _can_health_callback(self, msg: Bool):
        self._can_healthy = bool(msg.data)
        if not self._can_healthy:
            self._enter_safety_lock('CAN_LOST', {'abort_class': 'CAN_LOST'})

    def _motor_limits_callback(self, msg: Float64MultiArray):
        values = [float(value) for value in msg.data]
        if len(values) != DOF * 3 or not all(np.isfinite(value) for value in values):
            return
        limits = []
        speeds = []
        for index in range(DOF):
            lo, hi, speed = values[index * 3:index * 3 + 3]
            if lo >= hi or speed <= 0.0:
                return
            limits.append((lo, hi))
            speeds.append(speed)
        self._hardware_joint_limits = limits
        self._hardware_joint_max_speed = speeds

    def _effective_joint_limits(self) -> list[tuple]:
        hardware = list(getattr(self, "_hardware_joint_limits", []))
        calibrated_valid = bool(getattr(self, "_joint_limits_validated", False))
        if hardware and calibrated_valid:
            return [
                (
                    max(hw[0], calibrated[0] - CALIBRATION_LIMIT_TOLERANCE_RAD),
                    min(hw[1], calibrated[1] + CALIBRATION_LIMIT_TOLERANCE_RAD),
                )
                for hw, calibrated in zip(hardware, self._pm_joint_limits)
            ]
        if hardware:
            return hardware
        if calibrated_valid:
            return [
                (
                    calibrated[0] - CALIBRATION_LIMIT_TOLERANCE_RAD,
                    calibrated[1] + CALIBRATION_LIMIT_TOLERANCE_RAD,
                )
                for calibrated in self._pm_joint_limits
            ]
        return []

    def _replay_joint_limits(self) -> list[tuple]:
        """Hard limits for a physically taught trajectory.

        Calibration extrema are soft workspace limits for jog controls.  A
        teach trajectory is sampled from the real arm and can legitimately
        extend past an incompletely swept calibration range.  Replay therefore
        uses the SDK motor limits when available, falling back to calibrated
        limits only until the driver has published its limits.
        """
        hardware = list(getattr(self, "_hardware_joint_limits", []))
        if len(hardware) == DOF:
            return hardware
        return list(PIPER_HARDWARE_JOINT_LIMITS)

    def _enter_safety_lock(self, reason: str, details: Optional[dict] = None):
        first = not self._safety_latched
        if not first:
            self.request_replay_stop()
            self._cancel_motion_operation()
            return
        self._safety_latched = True
        self._safety_reason = str(reason)
        self._abort_class = 'CAN_LOST' if reason == 'CAN_LOST' else 'SAFETY_LOCKED'
        self._last_motion_abort = {
            'abort_class': self._abort_class,
            'reason': self._safety_reason,
            't_host_ns': time.time_ns(),
            **(details or {}),
        }
        self._cancel_motion_operation()
        self.request_replay_stop()
        self.get_logger().error(
            f"{self._abort_class}: {self._safety_reason}"
        )

    def _record_motion_abort(self, reason: str, details: Optional[dict] = None):
        self._abort_class = 'MOTION_ABORTED'
        self._last_transition_error = str(reason)
        self._last_motion_abort = {
            'abort_class': 'MOTION_ABORTED',
            'reason': str(reason),
            't_host_ns': time.time_ns(),
            **(details or {}),
        }

    def _arm_status_callback(self, msg: PiperStatusMsg):
        with self._lock:
            self._arm_status = {
                "ctrl_mode": int(msg.ctrl_mode),
                "arm_status": int(msg.arm_status),
                "mode_feedback": int(msg.mode_feedback),
                "teach_status": int(msg.teach_status),
                "motion_status": int(msg.motion_status),
                "err_code": int(msg.err_code),
            }

    def _make_joint_row(self, t_host_ns: int, tag: str) -> dict:
        with self._lock:
            pos = self._latest_position.copy()
            vel = self._latest_velocity.copy()
            gripper_cmd = self._gripper_cmd

        row: dict = {"t_host_ns": t_host_ns, "mode": tag}
        for i in range(DOF):
            row[f"q{i+1}"] = pos[i] if i < len(pos) else 0.0
        row["gripper"] = pos[DOF] if DOF < len(pos) else 0.0
        # Preserve the operator's discrete 0/50/100% request separately from
        # the measured ramp position.  Replay can then reproduce 50% without
        # interpreting it as a generic OPEN command.
        row["gripper_target"] = GRIPPER_CLOSE_RAD + (
            GRIPPER_OPEN_RAD - GRIPPER_CLOSE_RAD
        ) * (1.0 - gripper_cmd)
        for i in range(DOF):
            row[f"dq{i+1}"] = vel[i] if i < len(vel) else 0.0
        return row

    # ─────────────────────────────────────────────────────────────────
    # State queries
    # ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._lock:
            now_ns = time.time_ns()
            feedback_age_s = (
                (now_ns - self._latest_t_ns) * 1e-9
                if self._latest_t_ns else None
            )
            motor_status_age_s = (
                (now_ns - self._motor_status_t_ns) * 1e-9
                if self._motor_status_t_ns else None
            )
            calib_span = [
                self._calib_max[i] - self._calib_min[i]
                if np.isfinite(self._calib_min[i]) and np.isfinite(self._calib_max[i])
                else 0.0
                for i in range(DOF)
            ]
            return {
                "position": self._latest_position[:DOF],
                "gripper": self._latest_position[DOF] if len(self._latest_position) > DOF else 0.0,
                "gripper_responsive": self._gripper_responsive,
                "velocity": self._latest_velocity[:DOF],
                "is_moving": any(abs(v) > 0.05 for v in self._latest_velocity[:DOF]),
                "mode": self._mode,
                "hz": round(self._joint_hz, 1),
                "t_ns": self._latest_t_ns,
                "replay_progress": round(self._replay_progress, 3),
                "replay_worker": self.get_replay_worker_status(),
                "feedback_age_s": round(feedback_age_s, 4) if feedback_age_s is not None else None,
                "motor_enable_status": list(self._motor_enable_status),
                "motor_fault_codes": list(self._motor_fault_codes),
                "motor_faults": [
                    [name for bit, name in (
                        (0, "low_voltage"), (1, "motor_hot"),
                        (2, "overcurrent"), (3, "driver_hot"),
                        (4, "collision"), (5, "driver_error"), (7, "stall"),
                    ) if code & (1 << bit)]
                    for code in self._motor_fault_codes
                ],
                "arm_status": dict(self._arm_status),
                "motor_status_age_s": round(motor_status_age_s, 4) if motor_status_age_s is not None else None,
                "transition_state": self._transition_state,
                "transition_error": self._last_transition_error or None,
                "safety_latched": self._safety_latched,
                "safety_reason": self._safety_reason or None,
                "can_healthy": self._can_healthy,
                "abort_class": self._abort_class,
                "last_motion_abort": dict(self._last_motion_abort),
                "recovery": dict(self._last_recovery_result),
                "calibration_span": [round(value, 4) for value in calib_span],
                "calibration_valid": [value >= CALIBRATION_MIN_SPAN_RAD for value in calib_span],
                "calibration_min_span": CALIBRATION_MIN_SPAN_RAD,
                "hardware_joint_limits": [
                    list(limit) for limit in getattr(self, "_hardware_joint_limits", [])
                ],
                "effective_joint_limits": [list(limit) for limit in self._effective_joint_limits()],
            }

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def is_moving(self) -> bool:
        with self._lock:
            return any(abs(v) > 0.05 for v in self._latest_velocity[:DOF])

    # ─────────────────────────────────────────────────────────────────
    # Teach
    # ─────────────────────────────────────────────────────────────────

    def start_teach(
        self,
        disable_delay: float = 5.0,
        failure_callback: Optional[Callable[[str], None]] = None,
        started_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Teaching 시작.
        disable_delay 초 후에 모터를 비활성화해 손으로 자유롭게 움직일 수 있게 함.
        _mode 는 모터 비활성화 시점까지 IDLE 유지 → 카운트다운 중 joint callback 미기록.
        """
        with self._lock:
            self._mode = "IDLE"  # 아직 recording 시작 안 함
            self._teach_buffer.clear()
            self._teach_start_t = None
            self._last_teach_log_t = 0.0

        generation = self._cancel_pending_mode_timers()

        def _on_disable():
            with self._transition_lock:
                if generation != self._mode_generation:
                    return
            if not self.disable_motors():
                reason = self._last_transition_error or "failed to enter freedrive"
                self._log_event(f"teach_freedrive_failed: {reason}")
                if failure_callback:
                    failure_callback(reason)
                return
            with self._transition_lock:
                if generation != self._mode_generation:
                    return
            with self._lock:
                self._mode = "TEACH"  # 모터 비활성화 후 recording 시작
                self._teach_start_t = time.time_ns()
                self._last_teach_log_t = 0.0
            self._log_event("teach_recording_started")
            self.get_logger().info("Teach recording started (motors disabled)")
            if started_callback:
                started_callback()

        # disable_delay 초 후 모터 비활성화 + recording 시작
        self._teach_disable_timer = threading.Timer(disable_delay, _on_disable)
        self._teach_disable_timer.daemon = True
        self._teach_disable_timer.start()

        self._log_event(f"teach_start (motors_off_in={disable_delay:.0f}s)")
        self.get_logger().info(f"Teach started — motors will disable in {disable_delay}s")

    def begin_teach_recording(self) -> bool:
        """
        모터를 건드리지 않고 즉시 recording 시작.
        Phone Mouse 모드에서 "Enable Phone Mouse" 클릭 시 호출.
        - 5초 freedrive 타이머가 걸려 있으면 취소 (모터 OFF 방지)
        - _mode = "TEACH" 로 전환 → _joint_callback 이 teach_buffer 기록 시작
        """
        # freedrive 타이머 취소 (아직 안 끝났으면)
        self._cancel_pending_mode_timers()

        with self._lock:
            motors_enabled = all(self._motor_enable_status)
        if not motors_enabled and not self.enable_motors():
            self._log_event(f"phone_mouse_enable_failed: {self._last_transition_error}")
            return False

        with self._lock:
            if self._mode == "TEACH":
                return True  # 이미 recording 중
            self._mode = "TEACH"
            self._teach_buffer.clear()
            self._teach_start_t = time.time_ns()
            self._last_teach_log_t = 0.0

        self._log_event("teach_recording_started (phone_mouse, motors_on)")
        self.get_logger().info("Phone Mouse teach recording started (motors ON)")
        return True

    def stop_teach(self) -> Optional[list[dict]]:
        self._cancel_pending_mode_timers()

        # 모터 재활성화 및 CAN joint control 복구
        if not self._prepare_joint_motion(hardware_limits_only=True):
            self._log_event(f"teach_stop_rearm_failed: {self._last_transition_error}")
            return None

        with self._lock:
            self._mode = "IDLE"
            buf = self._teach_buffer.copy()
        self._log_event("teach_stop")
        self.get_logger().info(f"Teach recording stopped: {len(buf)} samples")
        return buf

    # ─────────────────────────────────────────────────────────────────
    # Trajectory smoothing & validation
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def smooth_trajectory(teach_buffer: list[dict], speed_scale: float = REPLAY_DEFAULT_SPEED) -> tuple[list[dict], dict]:
        """
        teach_buffer를 실제 timestamp 기반 resampling + Savitzky-Golay smoothing.
        Returns (smoothed_trajectory, check_result)
        """
        from scipy.interpolate import interp1d as _interp1d

        if len(teach_buffer) < REPLAY_SMOOTHING_WINDOW + 2:
            return teach_buffer, {"ok": False, "reason": "Too few samples to smooth"}

        keys = [f"q{i+1}" for i in range(DOF)]
        arr = np.array([[r[k] for k in keys] for r in teach_buffer])  # (N, 6)

        # ── 실제 teach timestamp 추출 (단조 보장) ────────────────────────
        t_raw = np.array(
            [(r["t_host_ns"] - teach_buffer[0]["t_host_ns"]) * 1e-9 for r in teach_buffer],
            dtype=float,
        )
        for i in range(1, len(t_raw)):
            if t_raw[i] <= t_raw[i - 1]:
                t_raw[i] = t_raw[i - 1] + 1e-3

        # ── 실제 duration 기반 uniform 50Hz grid ─────────────────────────
        dt = 1.0 / TEACH_LOG_HZ
        t_uni = np.arange(0, t_raw[-1] + 0.5 * dt, dt)  # +0.5*dt: 마지막 pose 잘림 방지

        # ── Joint: linear interpolation → resample to uniform grid ───────
        interp_fn = _interp1d(
            t_raw, arr, axis=0, kind="linear",
            bounds_error=False, fill_value=(arr[0], arr[-1]),
        )
        arr_resampled = interp_fn(t_uni)

        # ── Gripper: nearest-neighbor (state 전환 보존) ───────────────────
        g_raw = np.array([
            r.get("gripper_target", r.get("gripper", 0.0)) for r in teach_buffer
        ])
        g_fn = _interp1d(
            t_raw, g_raw, kind="nearest",
            bounds_error=False, fill_value=(g_raw[0], g_raw[-1]),
        )
        gripper_resampled = g_fn(t_uni)

        # ── Savitzky-Golay smoothing ──────────────────────────────────────
        window = REPLAY_SMOOTHING_WINDOW
        if window > len(arr_resampled):
            window = len(arr_resampled) if len(arr_resampled) % 2 == 1 else len(arr_resampled) - 1

        smoothed = savgol_filter(arr_resampled, window_length=window, polyorder=REPLAY_SMOOTHING_POLYORDER, axis=0)

        # Velocity check (finite differences at uniform 50Hz)
        velocity = np.diff(smoothed, axis=0) / dt
        max_vel = float(np.max(np.abs(velocity)))

        actual_duration = float(t_raw[-1])
        check = {
            "ok": True,
            "duration_s": round(actual_duration, 2),
            "samples": len(smoothed),
            "max_joint_velocity_rad_s": round(max_vel, 4),
            "velocity_limit_ok": max_vel <= MAX_JOINT_VELOCITY_RAD_S,
            "speed_scale": speed_scale,
        }

        if not check["velocity_limit_ok"]:
            check["ok"] = False
            check["reason"] = f"Max velocity {max_vel:.3f} rad/s exceeds limit {MAX_JOINT_VELOCITY_RAD_S}"

        # Waypoint delta check
        deltas = np.abs(np.diff(smoothed, axis=0))
        max_delta = float(np.max(deltas))
        check["max_waypoint_delta_rad"] = round(max_delta, 4)
        check["waypoint_delta_ok"] = max_delta <= MAX_WAYPOINT_DELTA_RAD

        # ── Build result rows (실제 timestamp 기반) ───────────────────────
        t0_ns = teach_buffer[0]["t_host_ns"]
        result = []
        for i, (row_s, grip) in enumerate(zip(smoothed, gripper_resampled)):
            r = {f"q{j+1}": float(row_s[j]) for j in range(DOF)}
            r["t_host_ns"] = t0_ns + int(t_uni[i] * 1e9)
            r["gripper"] = float(grip)
            r["gripper_target"] = float(grip)
            result.append(r)

        return result, check

    # ─────────────────────────────────────────────────────────────────
    # Home
    # ─────────────────────────────────────────────────────────────────

    def _send_until_close(
        self,
        wp: list,
        velocity: float,
        gripper: Optional[float],
        tol: float = 0.05,
        vel_tol: float = 0.4,
        timeout: float = 15.0,
        resend_hz: float = 5.0,
        motion_generation: Optional[int] = None,
    ) -> bool:
        """
        wp에 도달할 때까지 resend_hz 주기로 명령을 반복 전송.
        ctrl_node의 hold 모드 race condition을 방지하기 위해 단발 전송 대신 사용.
        """
        t0 = time.time()
        n = min(len(wp), DOF)
        pos_err = float("inf")
        max_vel = float("inf")
        resend_interval = 1.0 / resend_hz
        last_send = 0.0
        with self._lock:
            start_position = list(self._latest_position[:DOF])
        target_requires_motion = (
            len(start_position) >= n and
            max(abs(start_position[i] - wp[i]) for i in range(n)) >= tol
        )
        best_error = max(
            (abs(start_position[i] - wp[i]) for i in range(n)), default=0.0
        )
        last_progress_t = t0

        while time.time() - t0 < timeout:
            now = time.time()
            if motion_generation is not None and not self._motion_is_current(motion_generation):
                self._last_transition_error = "motion cancelled"
                return False
            feedback_age = self._feedback_age_s()
            if feedback_age > JOINT_FEEDBACK_ABORT_AGE_S:
                self._last_transition_error = f"stale joint feedback: age={feedback_age:.3f}s"
                self.get_logger().error(self._last_transition_error)
                self._log_event("motion_abort_stale_feedback")
                return False
            if feedback_age > JOINT_FEEDBACK_MAX_AGE_S:
                # Do not resend or decide arrival from old data.  The Piper driver
                # occasionally pauses joint publication briefly while entering HOLD.
                time.sleep(0.02)
                continue
            if now - last_send >= resend_interval:
                if self._publish_joint_cmd(
                    wp[:DOF], velocity=velocity, gripper=gripper
                ) is False:
                    return False
                last_send = now

            with self._lock:
                cur = list(self._latest_position[:DOF])
                vel = list(self._latest_velocity[:DOF])
            if len(cur) < n:
                time.sleep(0.02)
                continue

            pos_err = max(abs(cur[i] - wp[i]) for i in range(n))
            max_vel = max(abs(v) for v in vel[:n]) if vel else 0.0
            if best_error - pos_err >= 0.01:
                best_error = pos_err
                last_progress_t = now
            if (
                pos_err > 0.20
                and now - t0 >= MOTION_START_TIMEOUT_S + 0.10
                and now - last_progress_t >= 0.5
            ):
                reason = (
                    f"motion stalled: pos_err={pos_err:.3f}rad, "
                    f"no improvement for {now - last_progress_t:.2f}s"
                )
                self._record_motion_abort(reason, {"target": list(wp[:DOF])})
                self.get_logger().error(reason)
                return False
            if pos_err < tol and max_vel < vel_tol:
                return True
            if target_requires_motion and now - t0 >= MOTION_START_TIMEOUT_S:
                moved = max(abs(cur[i] - start_position[i]) for i in range(n))
                if moved < MOTION_START_MIN_DELTA_RAD:
                    self._last_transition_error = (
                        f"no physical motion after {MOTION_START_TIMEOUT_S:.2f}s: "
                        f"delta={moved:.4f}rad, motor_feedback={self._motor_enable_status}"
                        f", motor_fault_codes={getattr(self, '_motor_fault_codes', [])}"
                        f", arm_status={getattr(self, '_arm_status', {})}"
                    )
                    self.get_logger().error(self._last_transition_error)
                    self._log_event("motion_start_watchdog_abort")
                    return False
            time.sleep(0.02)

        self.get_logger().warn(
            f"_wait_until_close timeout: pos_err={pos_err:.3f}rad vel={max_vel:.3f}rad/s"
        )
        self._last_transition_error = (
            f"waypoint timeout: pos_err={pos_err:.3f}rad, vel={max_vel:.3f}rad/s"
        )
        return False

    def _wait_until_close(
        self,
        target: list,
        tol: float = 0.05,
        vel_tol: float = 0.12,
        timeout: float = 8.0,
        motion_generation: Optional[int] = None,
    ) -> bool:
        """
        피드백 기반 도착 확인.
        pos_err < tol AND max_vel < vel_tol → True 반환.
        timeout 초 초과 → False (경고 로그 출력).
        """
        t0 = time.time()
        n = min(len(target), DOF)
        pos_err = float("inf")
        max_vel = float("inf")
        while time.time() - t0 < timeout:
            if motion_generation is not None and not self._motion_is_current(motion_generation):
                self._last_transition_error = "motion cancelled"
                return False
            feedback_age = self._feedback_age_s()
            if feedback_age > JOINT_FEEDBACK_ABORT_AGE_S:
                self._last_transition_error = f"stale joint feedback: age={feedback_age:.3f}s"
                self.get_logger().error(self._last_transition_error)
                self._log_event("motion_abort_stale_feedback")
                return False
            if feedback_age > JOINT_FEEDBACK_MAX_AGE_S:
                time.sleep(0.03)
                continue
            with self._lock:
                cur = list(self._latest_position[:DOF])
                vel = list(self._latest_velocity[:DOF])
            if len(cur) < n:
                time.sleep(0.03)
                continue
            pos_err = max(abs(cur[i] - target[i]) for i in range(n))
            max_vel = max(abs(v) for v in vel[:n]) if vel else 0.0
            if pos_err < tol and max_vel < vel_tol:
                return True
            time.sleep(0.03)
        self.get_logger().warn(
            f"_wait_until_close timeout: pos_err={pos_err:.3f}rad vel={max_vel:.3f}rad/s"
        )
        return False

    def _follow_home_waypoints(
        self,
        waypoints: list,
        done_callback: Optional[Callable] = None,
        abort_callback: Optional[Callable] = None,
        gripper_during: Optional[float] = None,
        gripper_after: Optional[float] = None,
    ) -> bool:
        """
        검증된 waypoint sequence를 순서대로 실행하는 안전 복귀 primitive.
        각 waypoint 도착 실패(timeout) 시 그 자리에서 abort — 다음 waypoint로 진행하지 않는다.
        done_callback은 모든 waypoint 도착 성공 시에만 호출된다.
        abort_callback은 waypoint timeout 실패 시에만 호출된다.
        """
        if not waypoints:
            self.get_logger().error("_follow_home_waypoints: empty waypoints — refused.")
            return False

        # Validate the complete route before moving.  Previously a later
        # waypoint could be rejected after the arm had already reached the
        # first waypoint, and that immediate rejection was reported as a
        # misleading arrival timeout.
        effective_limits = self._effective_joint_limits()
        for waypoint_index, waypoint in enumerate(waypoints):
            if len(waypoint) < DOF or not all(
                np.isfinite(float(value)) for value in waypoint[:DOF]
            ):
                self._last_transition_error = (
                    f"safe_home waypoint {waypoint_index} is incomplete or non-finite"
                )
                self.get_logger().error(self._last_transition_error)
                return False
            if effective_limits:
                violations = [
                    (
                        joint_index + 1,
                        float(waypoint[joint_index]),
                        effective_limits[joint_index][0],
                        effective_limits[joint_index][1],
                    )
                    for joint_index in range(DOF)
                    if not effective_limits[joint_index][0]
                    <= float(waypoint[joint_index])
                    <= effective_limits[joint_index][1]
                ]
                if violations:
                    detail = ", ".join(
                        f"J{joint}={value:.4f} outside [{lo:.4f}, {hi:.4f}]"
                        for joint, value, lo, hi in violations
                    )
                    self._last_transition_error = (
                        f"safe_home waypoint {waypoint_index} rejected: {detail}"
                    )
                    self._record_motion_abort(
                        self._last_transition_error,
                        {"waypoint_index": waypoint_index, "target": list(waypoint[:DOF])},
                    )
                    self.get_logger().error(self._last_transition_error)
                    return False

        motion_generation = self._begin_motion_operation()
        if motion_generation is None:
            self._last_transition_error = "another automatic motion is already running"
            self.get_logger().warn(self._last_transition_error)
            return False

        if not self._prepare_joint_motion():
            self.get_logger().error(
                f"_follow_home_waypoints: re-arm failed: {self._last_transition_error}"
            )
            if abort_callback:
                abort_callback()
            self._end_motion_operation(motion_generation)
            return False

        def _loop():
            try:
                with self._lock:
                    gripper_now = (
                        self._latest_position[DOF]
                        if len(self._latest_position) > DOF
                        else 0.0
                    )
                grip = gripper_during if gripper_during is not None else gripper_now

                for idx, wp in enumerate(waypoints):
                    if not self._motion_is_current(motion_generation):
                        return
                    with self._lock:
                        cur = list(self._latest_position[:DOF])

                    max_dist = max(abs(cur[i] - wp[i]) for i in range(min(DOF, len(wp))))
                    timeout = max(6.0, max_dist / 0.15 + 3.0)

                    self.get_logger().info(
                        f"safe_home waypoint {idx}/{len(waypoints)-1}: "
                        f"{[round(x, 3) for x in wp]}, dist={max_dist:.3f}rad, timeout={timeout:.1f}s"
                    )

                    # 단발 명령으로는 ctrl_node hold 모드 재진입 시 무시됨 → 도달까지 반복 송신
                    reached = self._send_until_close(
                        wp, velocity=HOME_VELOCITY, gripper=grip,
                        tol=0.05, vel_tol=0.4, timeout=timeout,
                        motion_generation=motion_generation,
                    )

                    if not reached:
                        if not self._motion_is_current(motion_generation):
                            return
                        reason = self._last_transition_error or (
                            f"not reached in {timeout:.1f}s"
                        )
                        self.get_logger().error(
                            f"safe_home waypoint {idx} aborted: {reason}. "
                            f"Robot stopped at current position."
                        )
                        self._log_event(f"safe_home_abort_waypoint_{idx}")
                        if abort_callback:
                            abort_callback()
                        return  # abort — done_callback 호출 금지

                if gripper_after is not None:
                    self._gripper_move(gripper_after)

                self._log_event("safe_home_done")
                self.get_logger().info("safe_home: all waypoints reached")
                if done_callback:
                    done_callback()
            finally:
                self._end_motion_operation(motion_generation)

        threading.Thread(target=_loop, daemon=True).start()
        return True

    def safe_return_to_start(
        self,
        trajectory: list,
        done_callback: Optional[Callable] = None,
        abort_callback: Optional[Callable] = None,
    ) -> bool:
        """
        recorded trajectory를 역방향으로 따라 teach_start까지 안전 복귀.

        - gripper OPEN↔CLOSE 전환점은 downsample 시에도 반드시 포함
        - 각 waypoint에서 _wait_until_close() 도착 확인
        - 도착 실패 시 해당 위치에서 abort (다음 waypoint 진행 금지, done_callback 호출 금지)
        - done_callback은 teach_start 도달 성공 시에만 호출
        - gripper 처리: trajectory gripper 필드 역방향 상태 전환 감지 → OPEN/CLOSE 자동 처리
        """
        if not trajectory:
            self.get_logger().error("safe_return_to_start: empty trajectory — refused.")
            return False

        _GRIPPER_MID = (GRIPPER_OPEN_RAD + GRIPPER_CLOSE_RAD) / 2.0
        n = len(trajectory)

        # ── 그리퍼 전환 인덱스 추출 (forward trajectory) ──────────────
        # 이 인덱스들은 downsample 후에도 반드시 포함
        transition_fwd: set = set()
        _prev_g_open = None
        for i, wp in enumerate(trajectory):
            g = wp.get("gripper")
            if g is None:
                continue
            g_open = g > _GRIPPER_MID
            if g_open != _prev_g_open and _prev_g_open is not None:
                transition_fwd.update([max(0, i - 1), i])
            _prev_g_open = g_open

        # ── 현재 로봇 위치에서 가장 가까운 trajectory 시작점 탐색 ─────
        # replay 후 tracking error로 인해 trajectory[-1]과 실제 위치가 다를 수 있음.
        # 후반부 50%에서만 탐색 (앞부분은 실제로 진행된 구간이 아닐 가능성 높음).
        with self._lock:
            cur_pos = list(self._latest_position[:DOF])

        start_fwd_idx = n - 1  # 기본값: 마지막 waypoint
        if len(cur_pos) >= DOF:
            best_err = float("inf")
            search_start = max(0, n // 2)
            for i in range(search_start, n):
                q = [trajectory[i].get(f"q{j+1}", 0.0) for j in range(DOF)]
                err = max(abs(cur_pos[k] - q[k]) for k in range(DOF))
                if err < best_err:
                    best_err = err
                    start_fwd_idx = i
            self.get_logger().info(
                f"safe_return_to_start: nearest fwd_idx={start_fwd_idx}/{n-1} "
                f"(pos_err={best_err:.3f}rad)"
            )

        # ── reverse + downsample (~50 waypoints) ─────────────────────
        relevant_transitions = {i for i in transition_fwd if i <= start_fwd_idx}
        stride = max(1, start_fwd_idx // 50) if start_fwd_idx > 0 else 1
        must_include = {0, start_fwd_idx} | relevant_transitions
        fwd_indices = sorted(
            set(range(start_fwd_idx, -1, -stride)) | must_include,
            reverse=True,   # end → start 순서 (역방향)
        )
        sampled = [trajectory[i] for i in fwd_indices]

        self.get_logger().info(
            f"safe_return_to_start: {n} waypoints → {len(sampled)} sampled "
            f"(stride={stride}, gripper transitions={len(relevant_transitions)//2})"
        )
        return self._execute_reverse_path(sampled, done_callback=done_callback, abort_callback=abort_callback)

    def _execute_reverse_path(
        self,
        waypoints: list,
        done_callback: Optional[Callable] = None,
        abort_callback: Optional[Callable] = None,
    ) -> bool:
        """
        역방향 waypoint sequence를 순차 실행.
        각 waypoint 도착 실패 시 abort. done_callback은 전체 성공 시에만 호출.
        abort_callback은 waypoint timeout 실패 시 호출.
        """
        if not waypoints:
            self.get_logger().error("_execute_reverse_path: empty waypoints.")
            return False

        if not self._prepare_joint_motion(hardware_limits_only=True):
            self.get_logger().error(
                f"_execute_reverse_path: re-arm failed: {self._last_transition_error}"
            )
            if abort_callback:
                abort_callback()
            return False

        _GRIPPER_MID = (GRIPPER_OPEN_RAD + GRIPPER_CLOSE_RAD) / 2.0
        RETURN_VELOCITY = 20.0   # MotionCtrl_2 % — replay보다 느리게

        def _loop():
            prev_gripper_is_open: Optional[bool] = None
            total = len(waypoints)

            for idx, wp in enumerate(waypoints):
                q = [wp.get(f"q{j+1}", 0.0) for j in range(DOF)]
                gripper_raw = wp.get("gripper")

                # 그리퍼 상태 전환 감지 (_replay_loop과 동일 로직)
                gripper_cmd: Optional[float] = None
                if gripper_raw is not None:
                    g_open = gripper_raw > _GRIPPER_MID
                    if g_open != prev_gripper_is_open:
                        if prev_gripper_is_open is not None:
                            # 실제 전환 — 명령 발행
                            gripper_cmd = GRIPPER_OPEN_RAD if g_open else GRIPPER_CLOSE_RAD
                            self.get_logger().info(
                                f"[ReverseReturn] Gripper {'OPEN' if g_open else 'CLOSE'} "
                                f"@ waypoint {idx}/{total-1}"
                            )
                        prev_gripper_is_open = g_open

                with self._lock:
                    cur = list(self._latest_position[:DOF])

                max_dist = max(abs(cur[i] - q[i]) for i in range(DOF)) if cur else 0.0
                # 보수적 timeout: 20% velocity ≈ 0.05 rad/s 기준 (실측치)
                timeout = max(10.0, max_dist / 0.05 + 2.0)

                if not self._publish_joint_cmd(
                    q, velocity=RETURN_VELOCITY, gripper=gripper_cmd,
                    hardware_limits_only=True,
                ):
                    reason = self._last_transition_error or "joint command rejected"
                    self.get_logger().error(
                        f"[ReverseReturn] Waypoint {idx}/{total-1} command rejected: "
                        f"{reason}"
                    )
                    self._log_event(f"reverse_return_rejected_{idx}")
                    if abort_callback:
                        abort_callback()
                    return

                reached = self._wait_until_close(q, tol=0.06, vel_tol=0.20, timeout=timeout)
                if not reached:
                    self.get_logger().error(
                        f"[ReverseReturn] Waypoint {idx}/{total-1} not reached "
                        f"(max_dist={max_dist:.3f}rad, timeout={timeout:.1f}s) — aborting."
                    )
                    self._log_event(f"reverse_return_abort_{idx}")
                    if abort_callback:
                        abort_callback()
                    return  # ★ abort — done_callback 호출 금지

                # 그리퍼 명령 후 actuation 대기
                if gripper_cmd is not None:
                    time.sleep(GRIPPER_STEPS * GRIPPER_STEP_DELAY + 0.2)

            self._log_event("reverse_return_done")
            self.get_logger().info("[ReverseReturn] Reached teach_start successfully.")
            if done_callback:
                done_callback()

        threading.Thread(target=_loop, daemon=True).start()
        return True

    def go_to_safe_ready(
        self,
        done_callback: Optional[Callable] = None,
        abort_callback: Optional[Callable] = None,
        gripper_after: Optional[float] = None,
        final_position: Optional[list] = None,
    ) -> bool:
        """
        저장된 safe return waypoints 순서대로 이동.
        우선순위: dataset/safe_return_waypoints.json (UI로 기록) → config.SAFE_RETURN_WAYPOINTS
        둘 다 없으면 이동을 거부하고 False를 반환한다.
        done_callback은 모든 waypoint 도착 성공 시에만 호출된다.
        """
        import json as _json
        import os as _os
        from config import SAFE_RETURN_WAYPOINTS, DATASET_PATH

        waypoints = None

        # 1순위: UI로 저장한 JSON 파일
        wp_file = _os.path.join(DATASET_PATH, "safe_return_waypoints.json")
        if _os.path.exists(wp_file):
            try:
                with open(wp_file) as f:
                    data = _json.load(f)
                if data and isinstance(data, list):
                    waypoints = data
            except Exception as e:
                self.get_logger().warn(f"go_to_safe_ready: failed to read {wp_file}: {e}")

        # 2순위: config.py 정적 설정
        if not waypoints and SAFE_RETURN_WAYPOINTS:
            waypoints = SAFE_RETURN_WAYPOINTS

        if not waypoints:
            self._last_transition_error = "no safe return waypoints configured"
            self.get_logger().error(
                "go_to_safe_ready(): no safe waypoints defined — movement refused. "
                "Record waypoints in the Setup page."
            )
            self._log_event("safe_home_refused_no_waypoints")
            return False

        # Replay 복귀는 teach 시작점을 명시적인 최종 목표로 사용한다.
        if final_position is not None:
            if len(final_position) != DOF or not all(np.isfinite(v) for v in final_position):
                self._last_transition_error = "invalid final return position"
                return False
            waypoints = list(waypoints) + [list(final_position)]
            self.get_logger().info(
                f"go_to_safe_ready: +teach_start as final waypoint "
                f"{[round(v, 3) for v in final_position]}"
            )

        # 일반 ready 복귀는 캘리브레이션 포즈를 최종 waypoint로 사용한다.
        calib_file = _os.path.join(DATASET_PATH, "calibration_pose.json")
        if final_position is None and _os.path.exists(calib_file):
            try:
                with open(calib_file) as f:
                    calib_pose = _json.load(f)
                if isinstance(calib_pose, list) and len(calib_pose) == DOF:
                    waypoints = list(waypoints) + [calib_pose]
                    self.get_logger().info(
                        f"go_to_safe_ready: +calibration_pose as final waypoint "
                        f"{[round(v, 3) for v in calib_pose]}"
                    )
            except Exception as e:
                self.get_logger().warn(f"go_to_safe_ready: failed to read calibration_pose.json: {e}")

        self.get_logger().info(
            f"go_to_safe_ready: {len(waypoints)} waypoints (source={'file' if _os.path.exists(wp_file) else 'config'})"
        )
        return self._follow_home_waypoints(
            waypoints=waypoints,
            done_callback=done_callback,
            abort_callback=abort_callback,
            gripper_after=gripper_after,
        )

    def go_home(
        self,
        home_position: Optional[list] = None,
        done_callback: Optional[Callable] = None,
        gripper_during: Optional[float] = None,   # 이동 중 유지할 그리퍼 위치
        gripper_after: Optional[float] = None,    # 도착 후 열 그리퍼 위치
    ) -> bool:
        """
        단일 target pose로 이동 (return_home / go_home_with_release 내부용).
        home_position: q1..q6 (rad). None이면 이동을 거부하고 False를 반환한다.
        도착 실패(timeout) 시에도 done_callback을 호출한다 (복귀 흐름 유지).
        ★ 직접 호출 금지 — 안전 복귀에는 go_to_safe_ready() / _follow_home_waypoints() 사용.
        """
        if home_position is None:
            self.get_logger().error(
                "go_home() requires explicit home_position — movement refused."
            )
            self._log_event("go_home_refused_no_target")
            return False

        if not self._prepare_joint_motion():
            self.get_logger().error(f"go_home: re-arm failed: {self._last_transition_error}")
            return False

        pos = list(home_position)
        self.get_logger().info(f"go_home target: {[round(p, 3) for p in pos]}")

        def _sequential():
            with self._lock:
                current = list(self._latest_position[:DOF])
                gripper_now = (
                    self._latest_position[DOF]
                    if len(self._latest_position) > DOF
                    else 0.0
                )

            grip = gripper_during if gripper_during is not None else gripper_now
            self.get_logger().info(f"  current:  {[round(p, 3) for p in current]}")
            self.get_logger().info(f"  gripper:  {grip:.3f}")

            # Phase 1: J4, J5, J6 먼저 — 손목을 접어 엔드이펙터 반경 최소화
            working = list(current)
            group1 = [3, 4, 5]
            max_dist_1 = max(abs(current[i] - pos[i]) for i in group1)
            if max_dist_1 >= 0.01:
                for i in group1:
                    working[i] = pos[i]
                timeout1 = max(4.0, max_dist_1 / 0.25 + 1.0)
                self.get_logger().info(
                    f"  J4-6: dist={max_dist_1:.3f} rad, timeout={timeout1:.1f}s"
                )
                self._publish_joint_cmd(working, velocity=HOME_VELOCITY, gripper=grip)
                if not self._wait_until_close(working[:DOF], tol=0.05, timeout=timeout1):
                    self.get_logger().warn(
                        f"go_home: J4-6 did not converge in {timeout1:.1f}s — continuing"
                    )

            # Phase 2: J1, J2, J3 — 어깨/팔꿈치 이동
            group2 = [0, 1, 2]
            max_dist_2 = max(abs(current[i] - pos[i]) for i in group2)
            if max_dist_2 >= 0.01:
                for i in group2:
                    working[i] = pos[i]
                timeout2 = max(4.0, max_dist_2 / 0.25 + 1.0)
                self.get_logger().info(
                    f"  J1-3: dist={max_dist_2:.3f} rad, timeout={timeout2:.1f}s"
                )
                self._publish_joint_cmd(working, velocity=HOME_VELOCITY, gripper=grip)
                if not self._wait_until_close(working[:DOF], tol=0.05, timeout=timeout2):
                    self.get_logger().warn(
                        f"go_home: J1-3 did not converge in {timeout2:.1f}s — continuing"
                    )

            if max_dist_1 < 0.01 and max_dist_2 < 0.01:
                self.get_logger().info("go_home: already at target (no movement needed)")

            if gripper_after is not None:
                self._gripper_move(gripper_after)
            if done_callback:
                done_callback()

        threading.Thread(target=_sequential, daemon=True).start()
        return True

    def go_home_with_release(
        self,
        teach_start: list,
        gripper_close_pos: list,
        done_callback: Optional[Callable] = None,
    ):
        """
        [DEPRECATED] — use safe_return_to_start() instead.
        safe_return_to_start()는 trajectory gripper 필드를 자동 감지해
        grasp 위치에서 gripper를 열고 역방향 경로로 복귀한다.

        2-phase return home with object release at pickup location.
        Phase 1: arm → gripper_close_pos  (gripper stays CLOSED — holding object)
        Phase 2: open gripper (release object) → arm → teach_start
        """
        self.get_logger().info(
            f"go_home_with_release: phase1={[round(p, 3) for p in gripper_close_pos]}, "
            f"phase2={[round(p, 3) for p in teach_start]}"
        )

        def _phase2():
            # Release object at the exact position where it was picked up
            self._gripper_move(GRIPPER_OPEN_RAD)
            time.sleep(GRIPPER_STEPS * GRIPPER_STEP_DELAY + 0.3)
            # Now move arm back to teaching start
            self.go_home(home_position=teach_start, done_callback=done_callback)

        # Phase 1: travel to pickup position keeping gripper closed
        self.go_home(
            home_position=gripper_close_pos,
            done_callback=_phase2,
            gripper_during=GRIPPER_CLOSE_RAD,
        )

    # ─────────────────────────────────────────────────────────────────
    # Replay
    # ─────────────────────────────────────────────────────────────────

    def _smooth_replay_trajectory(self, trajectory, speed_scale):
        """Smooth arm joints only; preserve source timing, gripper and endpoints."""
        source = [dict(row) for row in trajectory]
        total = len(source)
        if total <= 4:
            return source
        speed = max(0.05, float(speed_scale))
        window = max(3, round(0.8 / speed)) if speed < 0.8 else 3
        half = window // 2
        smoothed = []
        for index, original in enumerate(source):
            lo = max(0, index - half)
            hi = min(total, index + half + 1)
            row = dict(original)
            for joint in range(1, DOF + 1):
                row[f"q{joint}"] = sum(
                    float(source[item].get(f"q{joint}", 0.0))
                    for item in range(lo, hi)
                ) / (hi - lo)
            row["gripper"] = original.get("gripper")
            smoothed.append(row)
        for joint in range(1, DOF + 1):
            smoothed[0][f"q{joint}"] = float(source[0].get(f"q{joint}", 0.0))
            smoothed[-1][f"q{joint}"] = float(source[-1].get(f"q{joint}", 0.0))
        self.get_logger().info(
            f"[Replay] smoothing window={window}, stream={REPLAY_COMMAND_HZ}Hz"
        )
        return smoothed

    def start_replay(
        self,
        trajectory: list[dict],
        speed_scale: float = REPLAY_DEFAULT_SPEED,
        done_callback: Optional[Callable] = None,
    ):
        if self._replay_thread and self._replay_thread.is_alive():
            self.get_logger().warn("Replay already running")
            return False

        prepared = self._smooth_replay_trajectory(trajectory, speed_scale)
        replay_limits = self._replay_joint_limits()
        if replay_limits:
            for row_idx, row in enumerate(prepared):
                for joint_idx, (lo, hi) in enumerate(replay_limits, start=1):
                    value = float(row.get(f"q{joint_idx}", float("nan")))
                    if not np.isfinite(value) or not lo <= value <= hi:
                        self._last_transition_error = (
                            f"replay target outside safety limits at row {row_idx}, "
                            f"J{joint_idx}={value:.4f} not in [{lo:.4f}, {hi:.4f}]"
                        )
                        self._record_motion_abort(self._last_transition_error)
                        return False

        if not self._prepare_joint_motion(hardware_limits_only=True):
            self.get_logger().error(
                f"start_replay: re-arm failed: {self._last_transition_error}"
            )
            return False

        self._replay_stop_event.clear()
        with self._lock:
            self._mode = "REPLAY"
            self._executed_buffer.clear()
            self._replay_command_buffer.clear()

        self._log_event("replay_start")
        if getattr(self, "_replay_worker_enabled", False):
            return self._start_replay_worker(prepared, speed_scale, done_callback)
        self._replay_thread = threading.Thread(
            target=self._replay_loop,
            args=(trajectory, speed_scale, done_callback),
            daemon=True,
        )
        self._replay_thread.start()
        return True

    def _start_replay_worker(self, trajectory, speed_scale, done_callback):
        runtime_root = os.getenv("XDG_RUNTIME_DIR") or tempfile.gettempdir()
        work_dir = tempfile.mkdtemp(prefix="piper_replay_", dir=runtime_root)
        request_path = os.path.join(work_dir, "request.json")
        result_path = os.path.join(work_dir, "result.json")
        command_path = os.path.join(work_dir, "commands.csv")
        request = {
            "trajectory": trajectory,
            "speed_scale": float(speed_scale),
            "command_hz": REPLAY_COMMAND_HZ,
            "teach_hz": TEACH_LOG_HZ,
            "joint_ctrl_topic": ROS_PIPER_JOINT_CTRL_TOPIC,
            "joint_state_topic": ROS_PIPER_JOINT_STATE_TOPIC,
            "motor_status_topic": "motor_enable_status",
            "motor_fault_topic": "motor_fault_status",
            "motor_safety_topic": "motor_safety_latched",
            "can_health_topic": "can_health",
            "gripper_open_rad": GRIPPER_OPEN_RAD,
            "gripper_close_rad": GRIPPER_CLOSE_RAD,
            "gripper_steps": GRIPPER_STEPS,
            "gripper_step_delay": GRIPPER_STEP_DELAY,
            "resume_ramp_s": 0.15,
            "feedback_max_age_s": JOINT_FEEDBACK_MAX_AGE_S,
            "motor_status_max_age_s": 0.50,
        }
        with open(request_path, "w") as file:
            json.dump(request, file)
        worker_path = os.path.join(os.path.dirname(__file__), "replay_streamer.py")
        try:
            process = subprocess.Popen(
                [
                    sys.executable, worker_path,
                    "--request", request_path,
                    "--result", result_path,
                    "--commands", command_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            shutil.rmtree(work_dir, ignore_errors=True)
            self._last_transition_error = f"replay worker start failed: {exc}"
            with self._lock:
                self._mode = "IDLE"
            return False

        self._replay_process = process
        self._replay_worker_dir = work_dir
        self._replay_worker_status = {
            "phase": "STARTING", "pid": process.pid, "progress": 0.0,
        }
        self._last_replay_worker_result = {}
        self.get_logger().info(f"Replay worker started: pid={process.pid}")

        def _monitor():
            try:
                if process.stdout:
                    for raw_line in process.stdout:
                        line = raw_line.rstrip()
                        if line.startswith("REPLAY_STATUS "):
                            try:
                                status = json.loads(line[len("REPLAY_STATUS "):])
                                status["pid"] = process.pid
                                self._replay_worker_status = status
                                if "progress" in status:
                                    self._replay_progress = float(status["progress"])
                            except (TypeError, ValueError, json.JSONDecodeError):
                                self.get_logger().warn(f"Invalid replay worker status: {line}")
                        elif line:
                            self.get_logger().info(f"[ReplayWorker] {line}")
                return_code = process.wait()
                result = {}
                if os.path.isfile(result_path):
                    with open(result_path) as file:
                        result = json.load(file)
                if os.path.isfile(command_path):
                    with open(command_path, newline="") as file:
                        rows = list(csv.DictReader(file))
                    numeric = {
                        "t_host_ns", "scheduled_t_host_ns", "waypoint_idx", "source_idx",
                        "lateness_ms", "velocity", "gripper", "gripper_effort",
                        *[f"q{i}" for i in range(1, DOF + 1)],
                    }
                    for row in rows:
                        for key in numeric:
                            if key in row and row[key] not in (None, ""):
                                row[key] = int(row[key]) if key in {
                                    "t_host_ns", "scheduled_t_host_ns", "waypoint_idx"
                                } else float(row[key])
                    with self._lock:
                        self._replay_command_buffer = rows
                if return_code != 0:
                    result["error"] = result.get("error") or (
                        f"worker exited with code {return_code}"
                    )
                    result["ok"] = False
                else:
                    result.setdefault("ok", bool(result.get("completed", False)))
                result["return_code"] = return_code
                result["pid"] = process.pid
                self._last_replay_worker_result = result
                if not result.get("ok"):
                    self._last_transition_error = result.get("error") or "replay worker failed"
                    abort_class = result.get("abort_class") or "MOTION_ABORTED"
                    if abort_class in {"SAFETY_LOCKED", "CAN_LOST"}:
                        self._enter_safety_lock(
                            result.get("error") or abort_class,
                            {"abort_class": abort_class, "worker": dict(result)},
                        )
                    else:
                        self._record_motion_abort(
                            self._last_transition_error,
                            {"worker": dict(result)},
                        )
                    with self._lock:
                        current = list(self._latest_position[:DOF])
                    if not self._safety_latched and self._can_healthy:
                        self._publish_joint_cmd(
                            current, velocity=5.0, streaming=False,
                            hardware_limits_only=True,
                        )
            except Exception as exc:
                self._last_transition_error = f"replay worker monitor failed: {exc}"
                self._last_replay_worker_result = {"ok": False, "error": str(exc)}
            finally:
                self._replay_progress = 0.0
                with self._lock:
                    self._mode = "IDLE"
                self._replay_worker_status = {
                    **self._replay_worker_status,
                    "phase": "DONE",
                    "result": dict(self._last_replay_worker_result),
                }
                self._replay_process = None
                self._replay_worker_dir = None
                shutil.rmtree(work_dir, ignore_errors=True)
                self._log_event("replay_end")
                self.get_logger().info(
                    f"Replay worker finished: {self._last_replay_worker_result}"
                )
                if done_callback:
                    # Do not run camera encoding or the next Safe Return inside
                    # the process-monitor thread.  It must be able to terminate
                    # so a concurrent Stop can join it and safely collect CSVs.
                    threading.Thread(target=done_callback, daemon=True).start()

        self._replay_thread = threading.Thread(target=_monitor, daemon=True)
        self._replay_thread.start()
        return True

    def request_replay_stop(self):
        self._replay_stop_event.set()
        process = getattr(self, "_replay_process", None)
        if process and process.poll() is None:
            process.terminate()
            return True
        return False

    def _force_stop_replay_worker(self):
        process = getattr(self, "_replay_process", None)
        if process and process.poll() is None:
            process.kill()
            self._last_transition_error = "replay worker termination timeout"
            return True
        return False

    def get_replay_worker_status(self):
        status = dict(getattr(self, "_replay_worker_status", {"phase": "IDLE"}))
        status["result"] = dict(getattr(self, "_last_replay_worker_result", {}))
        return status

    def _replay_loop(self, trajectory: list[dict], speed_scale: float, done_callback: Optional[Callable]):
        """Replay on a fixed-rate monotonic clock using interpolated stream targets."""
        total = len(trajectory)
        if total == 0:
            return
        speed_scale = max(0.05, float(speed_scale))
        command_period = 1.0 / REPLAY_COMMAND_HZ
        self.get_logger().info(
            f"Replay loop started: {total} waypoints, speed={speed_scale}, "
            f"stream={REPLAY_COMMAND_HZ}Hz"
        )

        trajectory = self._smooth_replay_trajectory(trajectory, speed_scale)

        gripper_mid = 0.02219
        first_gripper = next(
            (wp.get("gripper") for wp in trajectory if wp.get("gripper") is not None),
            None,
        )
        if first_gripper is None:
            gripper_is_open: Optional[bool] = None
            gripper_cmd: Optional[float] = None
        else:
            gripper_is_open = first_gripper > gripper_mid
            gripper_cmd = GRIPPER_OPEN_RAD if gripper_is_open else GRIPPER_CLOSE_RAD
            self._publish_gripper_cmd(gripper_cmd, effort=1.0 if gripper_is_open else 3.0)
        gripper_effort = 1.0 if gripper_is_open is not False else 3.0

        cmd_velocity = min(100.0, max(10.0, 100.0 * speed_scale))
        playback_start = time.monotonic()
        next_deadline = playback_start
        # Advance the trajectory by a fixed amount per command. Deriving the
        # source index from wall time made a delayed Python/ROS tick catch up by
        # skipping many waypoints at once (25+ observed in a real replay), which
        # reached the arm as a visible position step. A delayed replay may now
        # finish slightly later, but it never sacrifices motion continuity.
        source_step = TEACH_LOG_HZ * speed_scale / REPLAY_COMMAND_HZ
        source_idx = 0.0
        missed_deadlines = 0
        max_lateness = 0.0
        max_source_step = 0.0
        completed = False
        last_pos = [trajectory[0].get(f"q{joint}", 0.0) for joint in range(1, DOF + 1)]

        try:
            while not self._replay_stop_event.is_set():
                wait_s = next_deadline - time.monotonic()
                if wait_s > 0 and self._replay_stop_event.wait(wait_s):
                    break

                tick_time = time.monotonic()
                lateness = max(0.0, tick_time - next_deadline)
                max_lateness = max(max_lateness, lateness)
                source_idx = min(max(0.0, source_idx), total - 1.0)
                lo = int(source_idx)
                hi = min(total - 1, lo + 1)
                alpha = source_idx - lo
                pos = [
                    (1.0 - alpha) * trajectory[lo].get(f"q{joint}", 0.0)
                    + alpha * trajectory[hi].get(f"q{joint}", 0.0)
                    for joint in range(1, DOF + 1)
                ]
                raw_gripper = trajectory[lo].get("gripper")
                desired_open = (
                    raw_gripper > gripper_mid if raw_gripper is not None else gripper_is_open
                )
                gripper_transition = (
                    desired_open is not None and gripper_is_open is not None
                    and desired_open != gripper_is_open
                )
                if desired_open is not None and gripper_is_open is None:
                    gripper_is_open = desired_open
                if desired_open is not None:
                    gripper_cmd = GRIPPER_OPEN_RAD if desired_open else GRIPPER_CLOSE_RAD
                    gripper_effort = 1.0 if desired_open else 3.0

                t_cmd = time.time_ns()
                self._publish_joint_cmd(
                    pos, velocity=cmd_velocity, streaming=True,
                    hardware_limits_only=True,
                )
                last_pos = pos
                row = {f"q{joint+1}": pos[joint] for joint in range(DOF)}
                row.update({
                    "gripper": gripper_cmd,
                    "gripper_effort": gripper_effort,
                    "t_host_ns": t_cmd,
                    "scheduled_t_host_ns": t_cmd - int(lateness * 1e9),
                    "waypoint_idx": int(round(source_idx)),
                    "source_idx": round(source_idx, 4),
                    "lateness_ms": round(lateness * 1000.0, 3),
                    "velocity": cmd_velocity,
                })
                with self._lock:
                    self._replay_command_buffer.append(row)
                self._replay_progress = source_idx / max(1.0, total - 1.0)

                if gripper_transition:
                    block_start = time.monotonic()
                    if not self._wait_until_close(pos, tol=0.04, vel_tol=0.1, timeout=3.0):
                        self.get_logger().warn(
                            f"[Replay] gripper keyframe {source_idx:.1f}: not converged"
                        )
                    with self._lock:
                        current_gripper = (
                            self._latest_position[DOF]
                            if len(self._latest_position) > DOF else 0.0
                        )
                    for step in range(1, GRIPPER_STEPS + 1):
                        intermediate = current_gripper + (
                            gripper_cmd - current_gripper
                        ) * step / GRIPPER_STEPS
                        self._publish_gripper_cmd(intermediate, effort=gripper_effort)
                        if self._replay_stop_event.wait(GRIPPER_STEP_DELAY):
                            break
                    self._replay_stop_event.wait(0.15)
                    gripper_is_open = desired_open
                    blocked = time.monotonic() - block_start
                    next_deadline += blocked
                    self.get_logger().info(
                        f"[Replay] gripper {'OPEN' if desired_open else 'CLOSE'}; "
                        f"trajectory paused {blocked:.2f}s"
                    )

                if source_idx >= total - 1.0:
                    completed = True
                    break

                previous_source_idx = source_idx
                source_idx = min(total - 1.0, source_idx + source_step)
                max_source_step = max(max_source_step, source_idx - previous_source_idx)
                next_deadline += command_period
                now = time.monotonic()
                if next_deadline < now:
                    skipped = int((now - next_deadline) / command_period) + 1
                    missed_deadlines += skipped
                    # Drop overdue wall-clock deadlines, not trajectory points.
                    # The next command continues from source_idx above.
                    next_deadline = now + command_period
        finally:
            if not completed:
                with self._lock:
                    last_pos = list(self._latest_position[:DOF])
            endpoint_velocity = cmd_velocity if completed else 5.0
            self._publish_joint_cmd(
                last_pos, velocity=endpoint_velocity, streaming=False,
                hardware_limits_only=True,
            )
            if completed and not self._wait_until_close(
                last_pos, tol=0.04, vel_tol=0.1, timeout=3.0
            ):
                self.get_logger().warn(
                    "[Replay] final endpoint did not fully converge within 3.0s"
                )
            self._replay_progress = 0.0
            with self._lock:
                self._mode = "IDLE"
            self._log_event(
                f"replay_timing missed={missed_deadlines} "
                f"max_lateness_ms={max_lateness * 1000.0:.1f} "
                f"max_source_step={max_source_step:.3f}"
            )
            self._log_event("replay_end")
            self.get_logger().info(
                f"Replay loop finished: completed={completed}, missed={missed_deadlines}, "
                f"max_lateness={max_lateness * 1000.0:.1f}ms, "
                f"max_source_step={max_source_step:.3f}"
            )
            if done_callback:
                done_callback()

    def _replay_loop_legacy(self, trajectory: list[dict], speed_scale: float, done_callback: Optional[Callable]):
        self.get_logger().info(f"Replay loop started: {len(trajectory)} waypoints, speed={speed_scale}")
        dt = 1.0 / TEACH_LOG_HZ / speed_scale  # 실제 간격 (속도 배율 적용)

        # ── 저속 재생 smoothing (이동평균) ────────────────────────────────
        # 저속 시 teach 궤적의 고주파 미세진동이 눈에 띄게 증폭됨.
        # waypoint 수·dt 변경 없이 관절값만 이동평균으로 평탄화 → 총 시간 유지.
        # 그리퍼 값은 전환점 타이밍 보존을 위해 원본 그대로 사용.
        _SMOOTH_THRESHOLD = 0.8
        if speed_scale < _SMOOTH_THRESHOLD and len(trajectory) > 4:
            window = max(2, round(0.8 / speed_scale))  # 30% → window≈3, 50% → window≈2
            half = window // 2
            n_traj = len(trajectory)
            smoothed = []
            for _i in range(n_traj):
                lo = max(0, _i - half)
                hi = min(n_traj, _i + half + 1)
                avg: dict = {f"q{k}": sum(trajectory[j].get(f"q{k}", 0.0) for j in range(lo, hi)) / (hi - lo)
                             for k in range(1, 7)}
                avg["gripper"] = trajectory[_i].get("gripper")  # gripper는 원본 유지
                smoothed.append(avg)
            self.get_logger().info(
                f"[Replay] Smoothing: speed={speed_scale:.0%} window={window} ({n_traj} waypoints, dt={dt:.3f}s)"
            )
            trajectory = smoothed

        # ── 그리퍼 재생 전략 ──────────────────────────────────────────────
        # 고정 임계값 0.027 rad: 이 이상이면 OPEN, 미만이면 CLOSED.
        # - 닫힘(하드웨어) ≈ 0.008 rad → < 0.027 → CLOSED ✓
        # - 50% 열림 ≈ 0.034 rad  → > 0.02219 → OPEN  ✓
        # - 100% 열림 ≈ 0.070 rad → > 0.02219 → OPEN  ✓
        _GRIPPER_MID = 0.02219
        _g_all = [wp["gripper"] for wp in trajectory if wp.get("gripper") is not None]
        self.get_logger().info(
            f"[Replay] Gripper threshold: MID={_GRIPPER_MID:.4f} (fixed) "
            f"teach range=[{min(_g_all) if _g_all else '?':.4f}, {max(_g_all) if _g_all else '?':.4f}]"
        )

        # ── 그리퍼 전환 직전 keyframe 사전 계산 ──────────────────────────
        # 각 OPEN↔CLOSE 전환점 바로 직전 waypoint에서 도착을 확인한 뒤 gripper cmd를 보냄.
        # 전환점마다 대표 1개만 (여러 개 wait은 replay 지연 유발).
        gripper_keyframes: set = set()
        _prev_g_open = None
        for _ki, _wp in enumerate(trajectory):
            _g = _wp.get("gripper")
            if _g is None:
                continue
            _g_open = _g > _GRIPPER_MID
            if _g_open != _prev_g_open and _prev_g_open is not None:
                gripper_keyframes.add(max(0, _ki - 1))
            _prev_g_open = _g_open

        # ── 초기 그리퍼 상태 설정 ──────────────────────────────────────────
        # trajectory 첫 번째 gripper 값으로 initial state 초기화.
        # prev_gripper_is_open = None 상태로 루프 진입하면 첫 waypoint에서
        # "None → CLOSED" 가짜 전환이 감지되어 replay 시작 즉시 그리퍼가 닫힘.
        # 해결: prev_gripper_is_open을 첫 waypoint 기반으로 미리 설정.
        # 루프 자체가 첫 waypoint에서 gripper_cmd_val를 publish하므로 별도 이동 불필요.
        first_gripper = next(
            (wp.get("gripper") for wp in trajectory if wp.get("gripper") is not None), None
        )
        if first_gripper is not None:
            initial_is_open = first_gripper > _GRIPPER_MID
            prev_gripper_is_open: Optional[bool] = initial_is_open
            # 초기 상태도 전환 snap 로직과 동일하게 적용:
            # - OPEN → GRIPPER_OPEN_RAD (100%)
            # - CLOSED → GRIPPER_CLOSE_RAD (0%) : 임계값 미만은 항상 완전 닫힘으로 snap
            if initial_is_open:
                gripper_cmd_val: Optional[float] = GRIPPER_OPEN_RAD
            else:
                gripper_cmd_val: Optional[float] = GRIPPER_CLOSE_RAD
            self.get_logger().info(
                f"[Replay] Initial gripper preset: {'OPEN' if initial_is_open else 'CLOSE'} "
                f"(raw={first_gripper:.4f} → cmd={gripper_cmd_val:.4f})"
            )
        else:
            prev_gripper_is_open: Optional[bool] = None
            gripper_cmd_val: Optional[float] = None
        gripper_cmd_effort: float = 1.0            # 발행할 effort (N/m)

        cmd_velocity = 100.0 * speed_scale  # MotionCtrl_2 속도% (드라이버 velocity[0] 기준)
        total = len(trajectory)
        try:
            for i, wp in enumerate(trajectory):
                self._replay_progress = i / total if total > 0 else 0.0
                if self._replay_stop_event.is_set():
                    self.get_logger().info("Replay stopped by user")
                    break

                pos = [wp.get(f"q{j+1}", 0.0) for j in range(DOF)]
                gripper = wp.get("gripper")
                t_cmd = time.time_ns()

                # 그리퍼 OPEN/CLOSE 상태 전환 감지
                if gripper is not None:
                    cur_is_open = (gripper > _GRIPPER_MID)
                    if cur_is_open != prev_gripper_is_open:
                        prev_gripper_is_open = cur_is_open
                        if cur_is_open:
                            # 열 때: 항상 100% 완전 개방 → 물건을 확실히 놓음
                            gripper_cmd_val = GRIPPER_OPEN_RAD
                            snap_note = f" → snap to OPEN({GRIPPER_OPEN_RAD})"
                        else:
                            # 닫을 때: 항상 0% 완전 닫힘 → 물건을 확실히 집음
                            gripper_cmd_val = GRIPPER_CLOSE_RAD
                            snap_note = f" → snap to CLOSE({GRIPPER_CLOSE_RAD})"
                        # 닫을 때: 최대 힘(3 N/m) → 물체를 확실히 파지
                        # 열 때: 기본 힘(1 N/m)
                        gripper_cmd_effort = 1.0 if cur_is_open else 3.0
                        self.get_logger().info(
                            f"[Replay] Gripper {'OPEN' if cur_is_open else 'CLOSE'} "
                            f"@ waypoint {i} (teach={gripper:.4f}{snap_note}, effort={gripper_cmd_effort}N/m)"
                        )
                        # ── 전환 시 multi-step 그리퍼 작동 (블로킹) ──────────────────
                        # 단일 큰 step → 드라이버 미반응 문제 해결.
                        # arm을 현재 waypoint pos에 hold하면서 그리퍼를 GRIPPER_STEPS 단계로 이동.
                        with self._lock:
                            _g_cur = (self._latest_position[DOF]
                                      if len(self._latest_position) > DOF else 0.0)
                        for _gs in range(1, GRIPPER_STEPS + 1):
                            _g_inter = _g_cur + (gripper_cmd_val - _g_cur) * _gs / GRIPPER_STEPS
                            self._publish_joint_cmd(
                                pos, velocity=GRIPPER_VELOCITY,
                                gripper=_g_inter, gripper_effort=gripper_cmd_effort,
                                hardware_limits_only=True,
                            )
                            time.sleep(GRIPPER_STEP_DELAY)
                        time.sleep(0.15)  # 그리퍼 settle 후 다음 waypoint 진행

                self._publish_joint_cmd(pos, velocity=cmd_velocity,
                                        gripper=gripper_cmd_val,
                                        gripper_effort=gripper_cmd_effort,
                                        hardware_limits_only=True)

                cmd_row = {f"q{j+1}": pos[j] for j in range(DOF)}
                cmd_row["gripper"] = gripper_cmd_val
                cmd_row["gripper_effort"] = gripper_cmd_effort
                cmd_row["t_host_ns"] = t_cmd
                cmd_row["waypoint_idx"] = i
                with self._lock:
                    self._replay_command_buffer.append(cmd_row)

                # keyframe: arm 도착 확인 (gripper 전환은 위에서 이미 처리됨)
                if i in gripper_keyframes:
                    reached = self._wait_until_close(pos, tol=0.04, vel_tol=0.1, timeout=3.0)
                    if not reached:
                        self.get_logger().warn(
                            f"[Replay] Keyframe {i}: not converged"
                        )
                    time.sleep(0.1)
                else:
                    time.sleep(dt)

        finally:
            self._replay_progress = 0.0
            with self._lock:
                self._mode = "IDLE"
            self._log_event("replay_end")
            self.get_logger().info("Replay loop finished")
            if done_callback:
                done_callback()

    def stop_replay(self) -> tuple[list[dict], list[dict]]:
        """replay 중단 + (executed_buffer, command_buffer) 반환"""
        self.request_replay_stop()
        if self._replay_thread:
            self._replay_thread.join(timeout=2.0)
        if self._replay_thread and self._replay_thread.is_alive():
            self._force_stop_replay_worker()
            self._replay_thread.join(timeout=1.0)
            with self._lock:
                current = list(self._latest_position[:DOF])
            self._publish_joint_cmd(
                current, velocity=5.0, streaming=False,
                hardware_limits_only=True,
            )
        with self._lock:
            self._mode = "IDLE"
            exec_buf = self._executed_buffer.copy()
            cmd_buf = self._replay_command_buffer.copy()
        self.get_logger().info(f"Replay stopped: {len(exec_buf)} executed samples")
        return exec_buf, cmd_buf

    # ─────────────────────────────────────────────────────────────────
    # Motor enable / disable
    # ─────────────────────────────────────────────────────────────────

    def _cancel_pending_mode_timers(self):
        """Invalidate callbacks created by an older mode transition."""
        with self._transition_lock:
            self._mode_generation += 1
            if self._teach_disable_timer is not None:
                self._teach_disable_timer.cancel()
                self._teach_disable_timer = None
            return self._mode_generation

    def _begin_motion_operation(self) -> Optional[int]:
        """Reserve the single automatic-motion slot and return its generation."""
        with self._motion_state_lock:
            if self._motion_active:
                return None
            self._motion_generation += 1
            self._motion_active = True
            return self._motion_generation

    def _motion_is_current(self, generation: int) -> bool:
        with self._motion_state_lock:
            return self._motion_active and generation == self._motion_generation

    def _end_motion_operation(self, generation: int) -> None:
        with self._motion_state_lock:
            if generation == self._motion_generation:
                self._motion_active = False

    def _cancel_motion_operation(self) -> None:
        with self._motion_state_lock:
            self._motion_generation += 1
            self._motion_active = False

    def _request_motor_state(self, enabled: bool) -> bool:
        desired = [bool(enabled)] * DOF
        state_name = "ENABLING" if enabled else "DISABLING"
        # Serialize physical enable/disable requests, but deliberately do not
        # use _transition_lock for the blocking wait below.  The executor's
        # /motor_enable_status callback previously blocked on that lock before
        # it could complete the service future, producing a deterministic
        # timeout even though the driver had returned success.
        with self._motor_request_lock:
            with self._transition_lock:
                self._motor_request_generation += 1
                generation = self._motor_request_generation
                self._transition_state = state_name
                self._last_transition_error = ""

            def fail(reason: str) -> bool:
                with self._transition_lock:
                    if generation == self._motor_request_generation:
                        self._last_transition_error = reason
                        self._transition_state = "ERROR"
                return False

            if not self._enable_client.wait_for_service(timeout_sec=0.75):
                return fail("enable_srv unavailable")

            request = Enable.Request()
            request.enable_request = bool(enabled)
            future = self._enable_client.call_async(request)
            deadline = time.monotonic() + MOTOR_HANDSHAKE_TIMEOUT_S
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)

            if not future.done():
                return fail(f"{state_name.lower()} service timeout")

            try:
                service_ok = bool(future.result().enable_response)
            except Exception as exc:
                return fail(f"{state_name.lower()} service failed: {exc}")

            feedback_deadline = time.monotonic() + 0.5
            while time.monotonic() < feedback_deadline:
                with self._lock:
                    actual = list(self._motor_enable_status)
                    status_age = (
                        (time.time_ns() - self._motor_status_t_ns) * 1e-9
                        if self._motor_status_t_ns else float("inf")
                    )
                if service_ok and status_age <= 0.25 and actual == desired:
                    with self._transition_lock:
                        if generation != self._motor_request_generation:
                            return False
                        self._motors_enabled = enabled
                        self._transition_state = "ENABLED" if enabled else "FREEDRIVE"
                    self._log_event(
                        f"motors_{'enabled' if enabled else 'disabled'} "
                        f"feedback={[int(value) for value in actual]}"
                    )
                    self.get_logger().info(
                        f"Motor enable feedback: {[int(value) for value in actual]}"
                    )
                    return True
                time.sleep(0.01)

            with self._lock:
                actual = list(self._motor_enable_status)
            reason = (
                f"motor feedback mismatch: requested={desired}, actual={actual}, "
                f"service_ok={service_ok}"
            )
            self.get_logger().error(reason)
            return fail(reason)

    def enable_motors(self) -> bool:
        """Enable all six motors and wait for actual driver feedback."""
        if self._safety_latched:
            self._last_transition_error = (
                f"SAFETY_LOCKED: {self._safety_reason}; use motor recovery"
            )
            return False
        self._abort_class = None
        self._last_motion_abort = {}
        return self._request_motor_state(True)

    def disable_motors(self) -> bool:
        """Disable all six arm motors and wait until freedrive is confirmed."""
        return self._request_motor_state(False)

    def _feedback_is_fresh(self, max_age_s: float = JOINT_FEEDBACK_MAX_AGE_S) -> bool:
        age_s = self._feedback_age_s()
        if age_s > max_age_s:
            self._last_transition_error = f"stale joint feedback: age={age_s:.3f}s"
            self.get_logger().error(self._last_transition_error)
            return False
        return True

    def _feedback_age_s(self) -> float:
        with self._lock:
            return (
                (time.time_ns() - self._latest_t_ns) * 1e-9
                if self._latest_t_ns else float("inf")
            )

    def _prepare_joint_motion(self, hardware_limits_only: bool = False) -> bool:
        """Re-arm CAN joint control and seed it with the current physical pose."""
        self._cancel_pending_mode_timers()
        self.request_replay_stop()
        with self._lock:
            self._mode = "IDLE"

        if not self._can_healthy:
            self._last_transition_error = "CAN_LOST: feedback is not healthy"
            return False
        if self._safety_latched:
            self._last_transition_error = (
                f"SAFETY_LOCKED: {self._safety_reason}; recover motors first"
            )
            return False
        self._abort_class = None
        self._last_motion_abort = {}

        if not self._feedback_is_fresh():
            return False
        if not self.enable_motors():
            return False
        if not self._feedback_is_fresh():
            return False

        with self._lock:
            current = list(self._latest_position[:DOF])
        for _ in range(15):
            if not self._publish_joint_cmd(
                current, velocity=5.0,
                hardware_limits_only=hardware_limits_only,
            ):
                return False
            time.sleep(0.02)
        self._transition_state = "JOINT_CONTROL_READY"
        self._log_event("joint_control_rearmed")
        return True

    def recover_motors(self) -> dict:
        """Call the driver's bounded reset/re-enable sequence."""
        if not self._safety_latched:
            return {"ok": False, "reason": "No latched safety fault"}
        if not self._can_healthy:
            return {"ok": False, "reason": "CAN_LOST: feedback is not healthy"}
        self.request_replay_stop()
        self._cancel_motion_operation()
        if not self._recover_client.wait_for_service(timeout_sec=0.75):
            return {"ok": False, "reason": "recover_srv unavailable"}
        future = self._recover_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 4.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            return {"ok": False, "reason": "motor recovery service timeout"}
        try:
            response = future.result()
            result = json.loads(response.message) if response.message else {}
            result.setdefault("ok", bool(response.success))
        except Exception as exc:
            result = {"ok": False, "reason": f"motor recovery failed: {exc}"}
        self._last_recovery_result = dict(result)
        if result.get("ok"):
            self._safety_latched = False
            self._safety_reason = ""
            self._abort_class = None
            self._transition_state = (
                "JOINT_CONTROL_READY"
                if result.get("motors_enabled")
                else "READY_MOTORS_DISABLED"
            )
            self._last_transition_error = ""
            self._log_event("motor_safety_recovered")
        else:
            self._transition_state = "SAFETY_LOCKED"
            self._last_transition_error = str(result.get("reason", "recovery failed"))
        return result

    # ─────────────────────────────────────────────────────────────────
    # Soft stop (Hold position)
    # ─────────────────────────────────────────────────────────────────

    def hold_position(self) -> bool:
        """
        현재 position을 그대로 유지 명령 (soft stop). 물리적 E-stop 아님.
        모터가 비활성화 상태면 먼저 활성화한 뒤 hold.
        """
        self.request_replay_stop()
        worker_thread = getattr(self, "_replay_thread", None)
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=0.5)
        if worker_thread and worker_thread.is_alive():
            self._force_stop_replay_worker()
            worker_thread.join(timeout=0.5)
        self._cancel_motion_operation()
        # 모터가 꺼진 상태(teaching 중)라면 먼저 활성화
        if not self._motors_enabled:
            if not self.enable_motors():
                return False
            time.sleep(0.5)  # 활성화 안정화 대기
        with self._lock:
            pos = self._latest_position[:DOF]
            self._mode = "HOLD"
        self._publish_joint_cmd(pos, velocity=0.0)
        self._log_event("hold_position")
        self.get_logger().warn("HOLD POSITION commanded (soft stop)")
        return True

    # ─────────────────────────────────────────────────────────────────
    # Phone Mouse — joint delta
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # Phone Mouse — joint delta + limit calibration
    # ─────────────────────────────────────────────────────────────────

    def set_pm_joint_limits(self, limits: dict) -> bool:
        """
        calibration 결과를 반영.
        limits: {"j1": {"min": float, "max": float}, ...}
        """
        validated = []
        for i in range(DOF):
            key = f"j{i+1}"
            if key not in limits:
                self.get_logger().error(f"Joint limits rejected: missing {key}")
                return False
            try:
                lo = float(limits[key]["min"])
                hi = float(limits[key]["max"])
            except (KeyError, TypeError, ValueError):
                self.get_logger().error(f"Joint limits rejected: malformed {key}")
                return False
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                self.get_logger().error(
                    f"Joint limits rejected: {key} min={lo}, max={hi}"
                )
                return False
            validated.append((lo, hi))
        self._pm_joint_limits = validated
        self._joint_limits_validated = True
        self.get_logger().info(f"PM joint limits updated: {validated}")
        return True

    def start_calib(self) -> bool:
        """ROM calibration 시작 — 모터 OFF, min/max 추적 시작."""
        self._cancel_pending_mode_timers()
        self._calib_min = [float("inf")]  * DOF
        self._calib_max = [float("-inf")] * DOF
        if not self.disable_motors():
            self._calib_active = False
            self._log_event(f"calib_start_failed: {self._last_transition_error}")
            return False
        self._calib_active = True
        self._log_event("calib_start")
        self.get_logger().info("Calibration started — move arm through full ROM")
        return True

    def stop_calib(self) -> dict:
        """ROM calibration 종료 — 모터 ON, 실측 extrema를 limits로 반환."""
        self._calib_active = False
        if not self._prepare_joint_motion():
            return {
                "ok": False,
                "reason": self._last_transition_error or "failed to re-enable motors",
            }
        self._log_event("calib_stop")

        limits = {}
        invalid = []
        for i in range(DOF):
            lo = self._calib_min[i]
            hi = self._calib_max[i]
            span = hi - lo
            valid = (
                np.isfinite(lo) and np.isfinite(hi) and lo < hi and
                span >= CALIBRATION_MIN_SPAN_RAD
            )
            if valid:
                limits[f"j{i+1}"] = {
                    "min": round(lo, 4),
                    "max": round(hi, 4),
                    "measured": True,
                    "span": round(span, 4),
                }
            else:
                old_lo, old_hi = self._pm_joint_limits[i]
                limits[f"j{i+1}"] = {
                    "min": round(old_lo, 4),
                    "max": round(old_hi, 4),
                    "measured": False,
                    "span": round(span, 4) if np.isfinite(span) else 0.0,
                }
                invalid.append(f"j{i+1}")

        if invalid:
            reason = (
                f"Insufficient calibration span for {', '.join(invalid)}; "
                f"minimum={CALIBRATION_MIN_SPAN_RAD:.2f} rad. Existing limits preserved."
            )
            self.get_logger().error(reason)
            return {"ok": False, "reason": reason, "limits": limits, "invalid_joints": invalid}

        self.get_logger().info(f"Calibration result: {limits}")
        return {"ok": True, "limits": limits}

    def resume_calib(self) -> bool:
        """Release a calibration HOLD without resetting collected extrema."""
        if not self._calib_active:
            self._last_transition_error = "calibration is not active"
            return False
        if not self.disable_motors():
            return False
        with self._lock:
            self._mode = "IDLE"
        self._log_event("calib_resumed_freedrive")
        self.get_logger().info("Calibration resumed — arm motors disabled")
        return True

    def apply_joint_delta(self, j1=0.0, j2=0.0, j3=0.0, j4=0.0, j5=0.0, j6=0.0):
        """
        현재 위치에서 delta(rad)만큼 각 관절을 이동.
        그리퍼는 건드리지 않음. Phone Mouse 모드에서 호출.
        _pm_joint_limits 내로 클램프.
        """
        with self._lock:
            pos = list(self._latest_position[:DOF])
            gripper = self._latest_position[DOF] if len(self._latest_position) > DOF else None

        deltas = [j1, j2, j3, j4, j5, j6]
        for i, d in enumerate(deltas):
            lo, hi = self._pm_joint_limits[i]
            pos[i] = max(lo, min(hi, pos[i] + d))

        self._publish_joint_cmd(pos, velocity=10.0, gripper=gripper)

    # ─────────────────────────────────────────────────────────────────
    # Cartesian Jog — Pinocchio IK (ee_twist_cmd)
    # ─────────────────────────────────────────────────────────────────

    _URDF_PATH = (
        "/home/cglab/robotarm/piper_ws/src/Piper_ros/src/piper_description"
        "/urdf/piper_description.urdf"
    )
    _EE_LAMBDA   = 0.05    # DLS damping coefficient
    _EE_MAX_STEP = 0.03    # rad/step clamp per joint
    _EE_Z_MIN    = 0.05    # workspace floor limit (m)
    _EE_COND_WARN = 100    # condition number: scale-down threshold
    _EE_COND_MAX  = 300    # condition number: reject threshold

    def _load_urdf(self):
        """Pinocchio 모델을 URDF에서 로드. 실패하면 _pin_model=None 유지."""
        try:
            model = pin.buildModelFromUrdf(self._URDF_PATH)
            data = model.createData()
            self._pin_model = model
            self._pin_data = data
            self.get_logger().info(
                f"Pinocchio loaded: {model.njoints} joints, nq={model.nq}, ee_id={self._pin_ee_id}"
            )
        except Exception as exc:
            self.get_logger().error(f"Pinocchio URDF load failed: {exc}")

    def _compute_jacobian(self, q6: list) -> Optional[np.ndarray]:
        """
        arm q (6 values, rad) → 6×6 Jacobian (LOCAL_WORLD_ALIGNED frame).
        Pinocchio 없으면 None.
        """
        if self._pin_model is None:
            return None
        q = pin.neutral(self._pin_model)   # nq=8 (6 arm + 2 gripper)
        q[:6] = np.array(q6, dtype=float)
        pin.computeJointJacobians(self._pin_model, self._pin_data, q)
        J_full = pin.getJointJacobian(
            self._pin_model, self._pin_data,
            self._pin_ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )  # shape (6, 8)
        return J_full[:, :6]  # (6, 6) — arm columns only

    def _get_ee_position(self, q6: list) -> Optional[np.ndarray]:
        """FK → end-effector world position (3,). None if Pinocchio unavailable."""
        if self._pin_model is None:
            return None
        q = pin.neutral(self._pin_model)
        q[:6] = np.array(q6, dtype=float)
        pin.forwardKinematics(self._pin_model, self._pin_data, q)
        return self._pin_data.oMi[self._pin_ee_id].translation.copy()

    def apply_ee_twist(
        self, vx=0.0, vy=0.0, vz=0.0,
        wx=0.0, wy=0.0, wz=0.0, dt=0.05
    ):
        """
        End-effector twist → Damped Least Squares IK → joint delta.
        Pinocchio 없으면 joint-space fallback (vx→j1, vy→j2).
        TEACH_RECORDING 상태에서 controller가 호출.
        """
        with self._lock:
            q6 = list(self._latest_position[:DOF])
            gripper = (
                self._latest_position[DOF]
                if len(self._latest_position) > DOF
                else None
            )

        # z_min 안전: 엔드이펙터가 바닥에 너무 가까우면 하강 차단
        ee_pos = self._get_ee_position(q6)
        if ee_pos is not None and ee_pos[2] < self._EE_Z_MIN and vz < 0:
            vz = 0.0

        J = self._compute_jacobian(q6)
        if J is None:
            # Pinocchio 없음 → 단순 joint jog fallback
            self.apply_joint_delta(j1=vx * dt * 5, j2=vy * dt * 5)
            return

        twist = np.array([vx, vy, vz, wx, wy, wz], dtype=float)
        lam2  = self._EE_LAMBDA ** 2
        A     = J @ J.T + lam2 * np.eye(6)

        cond = np.linalg.cond(J)
        if cond > self._EE_COND_MAX:
            self.get_logger().warn(f"Cartesian jog: singularity cond={cond:.0f} → rejected")
            return
        scale = 1.0 if cond <= self._EE_COND_WARN else self._EE_COND_WARN / cond

        dq = scale * (J.T @ np.linalg.solve(A, twist)) * dt
        dq = np.clip(dq, -self._EE_MAX_STEP, self._EE_MAX_STEP)

        new_q = [
            max(lo, min(hi, q6[i] + float(dq[i])))
            for i, (lo, hi) in enumerate(self._pm_joint_limits)
        ]
        self._publish_joint_cmd(new_q, velocity=10.0, gripper=gripper)

    # ─────────────────────────────────────────────────────────────────
    # Gripper events
    # ─────────────────────────────────────────────────────────────────

    def gripper_set(self, pct: float):
        """그리퍼를 0~100% 범위로 이동.
        pct=0 → CLOSE (0.0 rad), pct=100 → OPEN (GRIPPER_OPEN_RAD).
        """
        pct = max(0.0, min(100.0, float(pct)))
        target_rad = GRIPPER_CLOSE_RAD + (GRIPPER_OPEN_RAD - GRIPPER_CLOSE_RAD) * pct / 100.0
        if pct == 0.0:
            # close와 동일하게 arm 위치 기록
            with self._lock:
                self._gripper_close_arm_pos = self._latest_position[:DOF].copy()
            self._gripper_cmd = 1.0
            self._log_event("gripper_close")
        else:
            self._gripper_cmd = 1.0 - pct / 100.0
            self._log_event(f"gripper_set_{int(pct)}pct")
        self.get_logger().info(f"Gripper SET {pct:.0f}% → {target_rad:.4f} rad")
        threading.Thread(target=self._gripper_move, args=(target_rad,), daemon=True).start()

    def gripper_open(self):
        self._gripper_cmd = 0.0
        self._log_event("gripper_open")
        self.get_logger().info(f"Gripper OPEN → {GRIPPER_OPEN_RAD} rad")
        threading.Thread(
            target=self._gripper_move,
            args=(GRIPPER_OPEN_RAD,),
            daemon=True,
        ).start()

    def gripper_close(self):
        self._gripper_cmd = 1.0
        # 그리퍼를 닫는 순간의 arm 위치 기록 (return_home 시 여기서 물건 내려놓기용)
        with self._lock:
            self._gripper_close_arm_pos = self._latest_position[:DOF].copy()
        self._log_event("gripper_close")
        self.get_logger().info(f"Gripper CLOSE → {GRIPPER_CLOSE_RAD} rad (arm pos recorded)")
        threading.Thread(
            target=self._gripper_move,
            args=(GRIPPER_CLOSE_RAD,),
            daemon=True,
        ).start()

    def get_gripper_close_arm_pos(self) -> Optional[list]:
        """그리퍼를 닫았던 시점의 arm joint 위치 (없으면 None)."""
        return self._gripper_close_arm_pos.copy() if self._gripper_close_arm_pos else None

    def clear_gripper_close_arm_pos(self):
        self._gripper_close_arm_pos = None

    def _gripper_move(self, target_rad: float):
        """그리퍼를 target_rad까지 GRIPPER_STEPS 단계로 나눠 천천히 이동.

        arm target이나 motor enable 상태를 변경하지 않는 gripper-only 명령을 사용한다.
        """
        with self._lock:
            current = self._latest_position[DOF] if len(self._latest_position) > DOF else 0.0
            mode = self._mode

        if mode == "TEACH":
            for i in range(1, GRIPPER_STEPS + 1):
                intermediate = current + (target_rad - current) * i / GRIPPER_STEPS
                self._publish_gripper_cmd(intermediate)
                time.sleep(GRIPPER_STEP_DELAY)
            self.get_logger().info(
                f"[Gripper/TEACH] → {target_rad:.4f} rad "
                f"(arm feedback={[int(value) for value in self._motor_enable_status]})"
            )
            self._check_gripper_responded(current, target_rad)
            return

        delta = target_rad - current
        for i in range(1, GRIPPER_STEPS + 1):
            intermediate = current + delta * i / GRIPPER_STEPS
            self._publish_gripper_cmd(intermediate)
            time.sleep(GRIPPER_STEP_DELAY)
        self._check_gripper_responded(current, target_rad)

    def _check_gripper_responded(self, start_rad: float, target_rad: float):
        """명령 완료 후 실제 피드백이 target 방향으로 충분히 움직였는지 확인.
        물리 케이블 단선 등으로 명령이 하드웨어에 안 먹힐 때(로그/HTTP 200은 정상으로
        찍히지만 실제로는 무시됨) UI에 경고를 띄우기 위한 신호."""
        time.sleep(GRIPPER_STEP_DELAY * 2)  # 마지막 스텝 이후 피드백 정착 시간
        with self._lock:
            final_pos = self._latest_position[DOF] if len(self._latest_position) > DOF else 0.0
        intended = abs(target_rad - start_rad)
        moved = abs(final_pos - start_rad)
        responded = moved >= max(GRIPPER_RESPONSE_MIN_ABS_RAD, GRIPPER_RESPONSE_MIN_FRAC * intended)
        with self._lock:
            was_responsive = self._gripper_responsive
            self._gripper_responsive = responded
        if not responded and was_responsive is not False:  # 상태 바뀔 때만 로그 (스팸 방지)
            self.get_logger().warning(
                f"Gripper may not be responding — target={target_rad:.4f} start={start_rad:.4f} "
                f"feedback={final_pos:.4f} rad (moved {moved:.4f}, expected ≥"
                f"{max(GRIPPER_RESPONSE_MIN_ABS_RAD, GRIPPER_RESPONSE_MIN_FRAC * intended):.4f}). "
                f"Check physical gripper cable."
            )

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _publish_joint_cmd(
        self,
        positions: list[float],
        velocity: float = 30.0,
        gripper: Optional[float] = None,
        gripper_effort: float = 1.0,
        streaming: bool = False,
        hardware_limits_only: bool = False,
    ) -> bool:
        """
        positions: q1..q6 (6 joints).
        gripper: joint7 위치(rad). None이면 7번째 joint 미포함.
        gripper_effort: 그리퍼 힘 (N 단위). driver가 *1000 하여 GripperCtrl로 전달.
                        driver 범위: 0.5~3 N → 500~3000. 기본 1.0 → 1000.
        """
        arm = [float(p) for p in positions[:DOF]]
        if len(arm) != DOF or not all(np.isfinite(value) for value in arm):
            self._last_transition_error = "joint command rejected: incomplete or non-finite target"
            self._record_motion_abort(self._last_transition_error)
            return False
        effective_limits = (
            self._replay_joint_limits()
            if hardware_limits_only
            else self._effective_joint_limits()
        )
        if effective_limits:
            outside = [
                idx + 1 for idx, value in enumerate(arm)
                if not effective_limits[idx][0] <= value <= effective_limits[idx][1]
            ]
            if outside:
                self._last_transition_error = (
                    "joint command outside validated safety limits: "
                    + ", ".join(f"J{idx}" for idx in outside)
                )
                self._record_motion_abort(self._last_transition_error, {"target": arm})
                return False
        if getattr(self, "_safety_latched", False) or not getattr(self, "_can_healthy", True):
            self._last_transition_error = getattr(self, "_safety_reason", "") or "CAN_LOST"
            return False
        msg = JointState()
        msg.header.frame_id = "piper_stream" if streaming else "piper_endpoint"
        n = DOF + (1 if gripper is not None else 0)
        msg.name = [f"joint{i+1}" for i in range(n)]
        msg.position = arm
        if gripper is not None:
            msg.position.append(float(gripper))
        msg.velocity = [float(velocity)] * n
        # effort[6] = gripper force. arm joints effort = 0 (unused by driver)
        msg.effort = [0.0] * DOF
        if gripper is not None:
            msg.effort.append(float(gripper_effort))
        self.pub.publish(msg)
        return True

    def _publish_gripper_cmd(self, position: float, effort: float = 1.0):
        """Publish a gripper-only command; the driver must not touch arm targets."""
        msg = JointState()
        msg.name = ["gripper"]
        msg.position = [float(position)]
        msg.velocity = [float(GRIPPER_VELOCITY)]
        msg.effort = [float(effort)]
        self.pub.publish(msg)

    def _log_event(self, event: str):
        self._events.append({"t_host_ns": time.time_ns(), "event": event})

    def get_teach_buffer(self) -> list[dict]:
        with self._lock:
            return self._teach_buffer.copy()

    def get_executed_buffer(self) -> list[dict]:
        with self._lock:
            return self._executed_buffer.copy()

    def get_command_buffer(self) -> list[dict]:
        with self._lock:
            return self._replay_command_buffer.copy()

    def get_events(self) -> list[dict]:
        return self._events.copy()

    def clear_events(self):
        self._events.clear()
