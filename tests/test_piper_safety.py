import csv
import importlib.util
import os
import sys
import tempfile
import threading
import time
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nodes import piper_node


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, **_kwargs):
        self.messages.append(("info", str(message)))

    def warn(self, message, **_kwargs):
        self.messages.append(("warn", str(message)))

    def error(self, message, **_kwargs):
        self.messages.append(("error", str(message)))


def _bare_cowork_node():
    node = piper_node.PiperTeachReplayNode.__new__(piper_node.PiperTeachReplayNode)
    node._lock = threading.Lock()
    node._transition_lock = threading.RLock()
    node._motion_state_lock = threading.Lock()
    node._motor_request_lock = threading.Lock()
    node._motor_request_generation = 0
    node._logger = _Logger()
    node.get_logger = lambda: node._logger
    node._events = []
    node._log_event = lambda event: node._events.append(event)
    node._last_transition_error = ""
    node._motor_enable_status = [False] * 6
    node._latest_position = [0.0] * 7
    node._latest_velocity = [0.0] * 7
    node._latest_t_ns = 0
    node._mode = "IDLE"
    node._motion_generation = 0
    node._motion_active = False
    node._motor_status_t_ns = 0
    node._motors_enabled = False
    node._transition_state = "READY"
    node._pm_joint_limits = [(-1.0, 1.0)] * 6
    node._replay_stop_event = threading.Event()
    node._replay_command_buffer = []
    node._replay_progress = 0.0
    return node


class CoworkSafetyTests(unittest.TestCase):
    def test_motor_feedback_callback_cannot_deadlock_service_handshake(self):
        node = _bare_cowork_node()

        class Future:
            complete = False

            def done(self):
                return self.complete

            @staticmethod
            def result():
                return types.SimpleNamespace(enable_response=True)

        future = Future()
        node._enable_client = types.SimpleNamespace(
            wait_for_service=lambda timeout_sec: True,
            call_async=lambda _request: future,
        )
        result = []
        request_thread = threading.Thread(
            target=lambda: result.append(node._request_motor_state(True)), daemon=True
        )
        request_thread.start()
        time.sleep(0.02)

        # This callback runs on the ROS executor.  It must be able to publish
        # feedback before that executor marks the service future complete.
        def executor_dispatch():
            node._motor_status_callback(types.SimpleNamespace(data=[1] * 6))
            future.complete = True

        callback_thread = threading.Thread(target=executor_dispatch, daemon=True)
        callback_thread.start()
        callback_thread.join(timeout=0.5)
        request_thread.join(timeout=1.0)

        self.assertFalse(callback_thread.is_alive())
        self.assertFalse(request_thread.is_alive())
        self.assertEqual(result, [True])

    def test_inverted_saved_limits_are_rejected_atomically(self):
        node = _bare_cowork_node()
        limits = {
            f"j{i}": {"min": -0.5, "max": 0.5}
            for i in range(1, 7)
        }
        limits["j5"] = {"min": 0.2, "max": 0.05}

        self.assertFalse(node.set_pm_joint_limits(limits))
        self.assertEqual(node._pm_joint_limits, [(-1.0, 1.0)] * 6)

    def test_short_calibration_is_rejected_and_preserves_limits(self):
        node = _bare_cowork_node()
        node._calib_active = True
        node._calib_min = [0.0] * 6
        node._calib_max = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1]
        node._prepare_joint_motion = lambda: True

        result = node.stop_calib()

        self.assertFalse(result["ok"])
        self.assertEqual(result["invalid_joints"], ["j5", "j6"])
        self.assertEqual(result["limits"]["j5"]["min"], -1.0)
        self.assertEqual(result["limits"]["j5"]["max"], 1.0)

    def test_calibration_uses_measured_extrema_without_angular_margin(self):
        node = _bare_cowork_node()
        node._calib_active = True
        node._calib_min = [-0.2] * 6
        node._calib_max = [0.3] * 6
        node._prepare_joint_motion = lambda: True

        result = node.stop_calib()

        self.assertTrue(result["ok"])
        self.assertEqual(result["limits"]["j1"]["min"], -0.2)
        self.assertEqual(result["limits"]["j1"]["max"], 0.3)

    def test_hardware_limits_are_not_shrunk(self):
        node = _bare_cowork_node()
        node._hardware_joint_limits = []
        node._hardware_joint_max_speed = []
        values = []
        for _ in range(6):
            values.extend([-1.0, 1.0, 1.5])

        node._motor_limits_callback(types.SimpleNamespace(data=values))

        self.assertEqual(node._hardware_joint_limits, [(-1.0, 1.0)] * 6)

    def test_calibration_tolerance_never_expands_hardware_limits(self):
        node = _bare_cowork_node()
        node._joint_limits_validated = True
        node._pm_joint_limits = [(-0.8, 0.8)] * 6
        node._hardware_joint_limits = [(-0.802, 0.802)] * 6

        effective = node._effective_joint_limits()

        self.assertEqual(effective, [(-0.802, 0.802)] * 6)

    def test_calibration_tolerance_accepts_small_replay_rounding_error(self):
        node = _bare_cowork_node()
        node._joint_limits_validated = True
        node._pm_joint_limits = [(-1.0, 0.6887)] * 6
        node._hardware_joint_limits = [(-2.0, 2.0)] * 6

        effective = node._effective_joint_limits()

        self.assertLessEqual(0.6891, effective[5][1])
        self.assertAlmostEqual(effective[5][1], 0.7387)

    def test_taught_replay_uses_hardware_not_calibration_extrema(self):
        node = _bare_cowork_node()
        node._joint_limits_validated = True
        node._pm_joint_limits = [(-0.5, 0.5)] * 6
        node._hardware_joint_limits = [(-2.0, 2.0)] * 6

        self.assertEqual(node._replay_joint_limits(), [(-2.0, 2.0)] * 6)

    def test_taught_replay_falls_back_to_vendor_limits_without_driver_snapshot(self):
        node = _bare_cowork_node()
        node._joint_limits_validated = True
        node._pm_joint_limits = [(-0.5, 0.5)] * 6
        node._hardware_joint_limits = []

        limits = node._replay_joint_limits()

        self.assertEqual(limits, piper_node.PIPER_HARDWARE_JOINT_LIMITS)

    def test_teach_gripper_never_changes_arm_enable(self):
        node = _bare_cowork_node()
        node._mode = "TEACH"
        node._latest_position[6] = 0.02
        published = []
        node._publish_gripper_cmd = lambda position, effort=1.0: published.append(position)
        node.enable_motors = lambda: self.fail("arm must not be enabled")
        node.disable_motors = lambda: self.fail("arm must not be disabled")
        old_delay = piper_node.GRIPPER_STEP_DELAY
        piper_node.GRIPPER_STEP_DELAY = 0.0
        try:
            node._gripper_move(0.07)
        finally:
            piper_node.GRIPPER_STEP_DELAY = old_delay

        self.assertEqual(len(published), piper_node.GRIPPER_STEPS)
        self.assertAlmostEqual(published[-1], 0.07)

    def test_resume_calibration_releases_arm_without_resetting_extrema(self):
        node = _bare_cowork_node()
        node._calib_active = True
        node._calib_min = [-0.4] * 6
        node._calib_max = [0.4] * 6
        node.disable_motors = lambda: True

        self.assertTrue(node.resume_calib())
        self.assertEqual(node._calib_min, [-0.4] * 6)
        self.assertEqual(node._calib_max, [0.4] * 6)
        self.assertEqual(node._mode, "IDLE")

    def test_motion_start_watchdog_aborts_static_feedback(self):
        node = _bare_cowork_node()
        node._latest_position = [0.0] * 7
        node._feedback_age_s = lambda: 0.0
        node._publish_joint_cmd = lambda *_args, **_kwargs: None

        reached = node._send_until_close(
            [0.2] * 6, velocity=10.0, gripper=None, timeout=0.7, resend_hz=20.0
        )

        self.assertFalse(reached)
        self.assertIn("no physical motion", node._last_transition_error)

    def test_brief_feedback_gap_during_motion_does_not_abort(self):
        node = _bare_cowork_node()
        node._latest_position = [0.0] * 7
        node._latest_t_ns = time.time_ns() - int(0.15 * 1e9)
        node._publish_joint_cmd = lambda *_args, **_kwargs: None

        def refresh_feedback():
            time.sleep(0.08)
            with node._lock:
                node._latest_position[:6] = [0.2] * 6
                node._latest_velocity[:6] = [0.0] * 6
                node._latest_t_ns = time.time_ns()

        threading.Thread(target=refresh_feedback, daemon=True).start()
        reached = node._send_until_close(
            [0.2] * 6, velocity=10.0, gripper=None, timeout=0.6, resend_hz=20.0
        )
        self.assertTrue(reached)

    def test_only_one_home_motion_can_run(self):
        node = _bare_cowork_node()
        first = node._begin_motion_operation()
        self.assertIsNotNone(first)
        self.assertIsNone(node._begin_motion_operation())
        node._end_motion_operation(first)
        self.assertIsNotNone(node._begin_motion_operation())

    def test_replay_uses_fixed_rate_stream_and_explicit_endpoint(self):
        node = _bare_cowork_node()
        published = []
        node._publish_joint_cmd = lambda positions, **kwargs: published.append(
            (time.monotonic(), list(positions), kwargs)
        )
        node._publish_gripper_cmd = lambda *_args, **_kwargs: None
        trajectory = [
            {**{f"q{joint}": index * 0.01 for joint in range(1, 7)}, "gripper": None}
            for index in range(6)
        ]
        old_command_hz = piper_node.REPLAY_COMMAND_HZ
        old_teach_hz = piper_node.TEACH_LOG_HZ
        piper_node.REPLAY_COMMAND_HZ = 100
        piper_node.TEACH_LOG_HZ = 100
        try:
            node._replay_loop(trajectory, speed_scale=1.0, done_callback=None)
        finally:
            piper_node.REPLAY_COMMAND_HZ = old_command_hz
            piper_node.TEACH_LOG_HZ = old_teach_hz

        stream = [entry for entry in published if entry[2].get("streaming")]
        self.assertGreaterEqual(len(stream), 4)
        self.assertFalse(published[-1][2]["streaming"])
        self.assertIn("lateness_ms", node._replay_command_buffer[0])
        intervals = [stream[i + 1][0] - stream[i][0] for i in range(len(stream) - 1)]
        self.assertLess(max(intervals), 0.03)

    def test_replay_delay_does_not_skip_trajectory_targets(self):
        node = _bare_cowork_node()
        published = []

        def publish(positions, **kwargs):
            published.append((list(positions), kwargs))
            if kwargs.get("streaming") and len(published) == 3:
                # Simulate a camera/ROS scheduling stall longer than several ticks.
                time.sleep(0.08)

        node._publish_joint_cmd = publish
        node._publish_gripper_cmd = lambda *_args, **_kwargs: None
        node._wait_until_close = lambda *_args, **_kwargs: True
        trajectory = [
            {**{f"q{joint}": index * 0.01 for joint in range(1, 7)}, "gripper": None}
            for index in range(30)
        ]
        old_command_hz = piper_node.REPLAY_COMMAND_HZ
        old_teach_hz = piper_node.TEACH_LOG_HZ
        piper_node.REPLAY_COMMAND_HZ = 100
        piper_node.TEACH_LOG_HZ = 100
        try:
            node._replay_loop(trajectory, speed_scale=1.0, done_callback=None)
        finally:
            piper_node.REPLAY_COMMAND_HZ = old_command_hz
            piper_node.TEACH_LOG_HZ = old_teach_hz

        source_indices = [row["source_idx"] for row in node._replay_command_buffer]
        source_steps = [
            source_indices[index + 1] - source_indices[index]
            for index in range(len(source_indices) - 1)
        ]
        self.assertLessEqual(max(source_steps), 1.0001)
        self.assertAlmostEqual(source_indices[-1], len(trajectory) - 1)
        self.assertTrue(any("max_source_step=1.000" in event for event in node._events))

    def test_replay_high_speed_smoothing_preserves_endpoints(self):
        node = _bare_cowork_node()
        published = []
        node._publish_joint_cmd = lambda positions, **kwargs: published.append(
            (list(positions), kwargs)
        )
        node._publish_gripper_cmd = lambda *_args, **_kwargs: None
        node._wait_until_close = lambda *_args, **_kwargs: True
        values = [0.0, 0.0, 0.9, 0.0, 0.0, 0.3]
        trajectory = [
            {**{f"q{joint}": value for joint in range(1, 7)}, "gripper": None}
            for value in values
        ]
        old_command_hz = piper_node.REPLAY_COMMAND_HZ
        old_teach_hz = piper_node.TEACH_LOG_HZ
        piper_node.REPLAY_COMMAND_HZ = 500
        piper_node.TEACH_LOG_HZ = 500
        try:
            node._replay_loop(trajectory, speed_scale=1.0, done_callback=None)
        finally:
            piper_node.REPLAY_COMMAND_HZ = old_command_hz
            piper_node.TEACH_LOG_HZ = old_teach_hz

        stream = [positions for positions, kwargs in published if kwargs.get("streaming")]
        self.assertAlmostEqual(stream[0][0], values[0])
        self.assertAlmostEqual(stream[-1][0], values[-1])
        self.assertAlmostEqual(stream[2][0], 0.3)
        self.assertEqual(published[-1][0], [values[-1]] * 6)

    def test_duplicate_teach_stop_is_rejected_while_first_is_running(self):
        from controller import Controller

        controller = Controller.__new__(Controller)
        controller._teach_stop_lock = threading.Lock()
        controller._teach_stop_lock.acquire()
        try:
            result = controller.stop_teach()
        finally:
            controller._teach_stop_lock.release()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "Teach stop is already in progress")

    def test_dataset_debug_loads_command_without_touching_episode_manager(self):
        from controller import Controller, State

        controller = Controller.__new__(Controller)
        controller._state = State.READY
        controller._lock = threading.Lock()
        controller._debug_lock = threading.RLock()
        controller._debug_trajectory = []
        controller._debug_source = {}
        controller._debug_check = {}
        controller._debug_start_ready = False
        controller._debug_operation = None
        controller._debug_previous_state = None
        controller._debug_run_dir = None
        controller._debug_started_at = None
        controller._debug_stop_requested = False
        controller._debug_last_error = ""
        controller._debug_last_diagnostics = {}
        controller._piper = None
        controller._event_log = []
        controller._on_emit = lambda *_args: None

        with tempfile.TemporaryDirectory() as take_dir:
            path = os.path.join(take_dir, "replay_command.csv")
            fields = ["t_host_ns"] + [f"q{i}" for i in range(1, 7)] + ["gripper"]
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for index in range(12):
                    writer.writerow({
                        "t_host_ns": 1_000_000_000 + index * 20_000_000,
                        **{f"q{i}": index * 0.001 * i for i in range(1, 7)},
                        "gripper": 0.07,
                    })

            result = controller.load_debug_trajectory(
                take_dir, "episode_test", "take_001", source="command"
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["loaded"])
        self.assertEqual(result["source"]["filename"], "replay_command.csv")
        self.assertEqual(result["source"]["samples"], 12)
        self.assertFalse(result["start_ready"])

    def test_debug_diagnostics_reports_source_steps_and_feedback_gaps(self):
        from controller import Controller

        command = []
        executed = []
        for index in range(4):
            command.append({
                "t_host_ns": 1_000_000_000 + index * 20_000_000,
                "source_idx": index * 0.7,
                "lateness_ms": index,
                **{f"q{i}": index * 0.01 for i in range(1, 7)},
            })
            executed.append({
                "t_host_ns": 1_000_000_000 + index * 20_000_000,
                **{f"q{i}": index * 0.01 for i in range(1, 7)},
            })
        executed.append({
            "t_host_ns": 1_200_000_000,
            **{f"q{i}": 0.03 for i in range(1, 7)},
        })

        diagnostics = Controller._debug_diagnostics(command, executed)

        self.assertAlmostEqual(diagnostics["max_source_step"], 0.7)
        self.assertEqual(diagnostics["max_lateness_ms"], 3.0)
        self.assertEqual(diagnostics["feedback_gaps_over_50ms"], 1)
        self.assertEqual(diagnostics["tracking_error_max_rad"], [0.0] * 6)


def _load_driver_module():
    path = "/home/cglab/robotarm/piper_ws/src/Piper_ros/src/piper/piper/piper_ctrl_single_node.py"
    spec = importlib.util.spec_from_file_location("piper_ctrl_single_node_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakePiper:
    def __init__(self, status):
        self.status = list(status)
        self.enable_calls = []
        self.disable_calls = []
        self.motion1_calls = []
        self.motion2_calls = []
        self.gripper_calls = []
        self.joint_calls = []
        self.disconnect_calls = []
        self.connect_calls = []

    def GetArmLowSpdInfoMsgs(self):
        result = types.SimpleNamespace()
        for index, enabled in enumerate(self.status, start=1):
            motor = types.SimpleNamespace(
                foc_status=types.SimpleNamespace(driver_enable_status=enabled)
            )
            setattr(result, f"motor_{index}", motor)
        return result

    @staticmethod
    def isOk():
        return True

    def EnableArm(self, motor):
        self.enable_calls.append(motor)
        if motor == 7:
            self.status = [True] * 6
        else:
            self.status[motor - 1] = True

    def DisableArm(self, motor):
        self.disable_calls.append(motor)
        if motor == 7:
            # Simulate the observed partial-disable failure on J5/J6.
            self.status[:4] = [False] * 4
        else:
            self.status[motor - 1] = False

    def MotionCtrl_1(self, *args):
        self.motion1_calls.append(args)

    def MotionCtrl_2(self, *args):
        self.motion2_calls.append(args)

    def GripperCtrl(self, *args):
        self.gripper_calls.append(args)

    def JointCtrl(self, *args):
        self.joint_calls.append(args)

    def DisconnectPort(self, **kwargs):
        self.disconnect_calls.append(kwargs)

    def ConnectPort(self, **kwargs):
        self.connect_calls.append(kwargs)


class DriverHandshakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver_module()

    def _node(self, status):
        node = self.driver.PiperRosNode.__new__(self.driver.PiperRosNode)
        node.piper = _FakePiper(status)
        node.gripper_exist = True
        node.joint_states = types.SimpleNamespace(position=[0.0] * 7)
        node._last_joint_feedback_monotonic = time.monotonic()
        node.motor_enable_status_pub = _Publisher()
        node._last_motor_status_pub = 0.0
        node._mode_transition_lock = threading.Lock()
        node._stream_lock = threading.RLock()
        node._joint_ctrl_send_lock = threading.Lock()
        node.stream_joint_hz = 100.0
        node.stream_input_period_max_s = 0.10
        node.stream_velocity_limit_rad_s = 1.5
        node._stream_generation = 0
        node._stream_segment_start = None
        node._stream_segment_target = None
        node._stream_segment_started_at = 0.0
        node._stream_segment_duration = 0.01
        node._stream_last_receive = 0.0
        node._stream_last_sent = None
        node._stream_last_send_t = 0.0
        node._stream_next_send_t = 0.0
        node._stream_first_receive = 0.0
        node._stream_receive_count = 0
        node._stream_send_count = 0
        node._stream_max_input_gap = 0.0
        node._stream_max_output_gap = 0.0
        node._stream_max_joint_step = 0.0
        node._logger = _Logger()
        node.get_logger = lambda: node._logger
        node.target_positions = [0.1] * 6
        node.last_command_positions = [0.1] * 6
        node.is_holding = True
        node.arrival_time = 1.0
        node._streaming_control = False
        node._last_motion_velocity = None
        node._PiperRosNode__enable_flag = all(status)
        node._PiperRosNode__gripper_enabled = False
        node._arm_expected_enabled = all(status)
        node._safety_latched = False
        node._safety_reason = ""
        node._safety_snapshot = {}
        node._recovering = False
        node._recovery_attempts = 0
        node._recovery_cooldown_until = 0.0
        node._can_healthy = True
        node._can_reconnect_lock = threading.Lock()
        node._can_reconnect_attempts = 0
        node._can_last_reconnect_t = 0.0
        node._can_last_reconnect_error = ""
        node._can_sdk_initialized = False
        node.can_reconnect_interval_s = 3.0
        node.can_port = "can1"
        node._motor_enable_status = list(status)
        node._motor_fault_codes = [0] * 6
        return node

    def test_latched_fault_rejects_plain_enable(self):
        node = self._node([True, False, True, True, True, True])
        node._safety_latched = True
        node._safety_reason = "FOC_FAULT"

        success, status = node._set_arm_enabled(True, timeout=0.1)

        self.assertFalse(success)
        self.assertEqual(status, [True, False, True, True, True, True])
        self.assertEqual(node.piper.enable_calls, [])

    def test_can_reconnect_recreates_sdk_socket_without_enabling_motion(self):
        node = self._node([False] * 6)
        node._PiperRosNode__enable_flag = True

        attempted = node._attempt_can_reconnect(force=True)

        self.assertTrue(attempted)
        self.assertEqual(node.piper.disconnect_calls, [{"thread_timeout": 0.5}])
        self.assertEqual(node.piper.connect_calls, [{
            "can_init": True,
            "piper_init": False,
            "start_thread": True,
        }])
        self.assertEqual(node.piper.enable_calls, [])
        self.assertFalse(node.GetEnableFlag())

    def test_recovered_can_clears_latch_without_motor_reset_or_enable(self):
        node = self._node([False] * 6)
        node._safety_latched = True
        node._safety_reason = "CAN_LOST"
        node._last_joint_feedback_monotonic = time.monotonic()

        result = node._recover_safety_fault()

        self.assertTrue(result["ok"])
        self.assertFalse(result["motors_enabled"])
        self.assertFalse(node._safety_latched)
        self.assertEqual(node.piper.enable_calls, [])
        self.assertEqual(node.piper.motion1_calls, [])
        self.assertEqual(node.piper.motion2_calls, [])

    def test_partial_disable_retries_j5_and_j6_individually(self):
        node = self._node([True] * 6)

        success, status = node._set_arm_enabled(False, timeout=0.8)

        self.assertTrue(success)
        self.assertEqual(status, [False] * 6)
        self.assertIn(5, node.piper.disable_calls)
        self.assertIn(6, node.piper.disable_calls)
        self.assertIsNone(node.target_positions)
        self.assertFalse(node.is_holding)

    def test_enable_restores_can_joint_mode(self):
        node = self._node([False] * 6)

        success, status = node._set_arm_enabled(True, timeout=0.4)

        self.assertTrue(success)
        self.assertEqual(status, [True] * 6)
        self.assertIn((0x00, 0x06, 0x02), node.piper.motion1_calls)
        self.assertIn((0x00, 0x04, 0x00), node.piper.motion1_calls)
        self.assertIn((0x01, 0x01, 30, 0x00), node.piper.motion2_calls)

    def test_enable_timeout_reports_failure(self):
        node = self._node([False] * 6)
        node.piper.EnableArm = lambda motor: node.piper.enable_calls.append(motor)

        success, status = node._set_arm_enabled(True, timeout=0.15)

        self.assertFalse(success)
        self.assertEqual(status, [False] * 6)
        # CAN mode is prepared while the arm is still disabled, but only the
        # current physical pose may be seeded; no stale target is restored.
        self.assertEqual(node.piper.motion2_calls, [(0x01, 0x01, 30, 0x00)])
        self.assertEqual(node.piper.joint_calls, [(0, 0, 0, 0, 0, 0)])
        self.assertIsNone(node.target_positions)

    def test_enable_seeds_physical_pose_before_first_enable_command(self):
        node = self._node([False] * 6)
        node.joint_states.position[:6] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
        ordering = []
        original_joint = node.piper.JointCtrl
        original_enable = node.piper.EnableArm
        node.piper.JointCtrl = lambda *args: (ordering.append("seed"), original_joint(*args))[-1]
        node.piper.EnableArm = lambda motor: (ordering.append("enable"), original_enable(motor))[-1]

        success, _ = node._set_arm_enabled(True, timeout=0.4)

        self.assertTrue(success)
        self.assertLess(ordering.index("seed"), ordering.index("enable"))
        factor = 57324.840764
        self.assertEqual(
            node.piper.joint_calls[0],
            tuple(round(value * factor) for value in node.joint_states.position[:6]),
        )

    def test_gripper_only_message_does_not_send_arm_control(self):
        node = self._node([False] * 6)
        node.gripper_val_mutiple = 1
        node.position_tolerance = 0.01
        node._PiperRosNode__gripper_enabled = True
        message = self.driver.JointState()
        message.name = ["gripper"]
        message.position = [0.04]
        message.velocity = [5.0]
        message.effort = [1.5]

        node.joint_callback(message)

        self.assertEqual(node.piper.enable_calls, [])
        self.assertEqual(node.piper.disable_calls, [])
        self.assertEqual(node.piper.gripper_calls[-1][0], 40000)

    def test_streaming_bypasses_hold_and_caches_motion_mode(self):
        node = self._node([True] * 6)
        node.gripper_val_mutiple = 1
        node.position_tolerance = 0.01

        message = self.driver.JointState()
        message.header.frame_id = "piper_stream"
        message.name = [f"joint{i}" for i in range(1, 7)]
        message.position = [0.1] * 6
        message.velocity = [50.0] * 6
        message.effort = [0.0] * 6
        node.joint_callback(message)
        message.position = [0.101] * 6
        node.joint_callback(message)

        self.assertTrue(node._streaming_control)
        self.assertFalse(node.is_holding)
        self.assertEqual(len(node.piper.joint_calls), 0)
        self.assertEqual(node.piper.motion2_calls, [(0x01, 0x01, 50)])

        self.assertTrue(node._publish_streaming_joint_ctrl())
        self.assertEqual(len(node.piper.joint_calls), 1)

        message.header.frame_id = "piper_endpoint"
        node.joint_callback(message)
        self.assertFalse(node._streaming_control)
        self.assertEqual(len(node.piper.motion2_calls), 2)
        self.assertEqual(len(node.piper.joint_calls), 2)

    def test_stream_interpolation_is_monotonic_bounded_and_rate_limited(self):
        node = self._node([True] * 6)
        node.is_holding = False
        node._update_stream_target([0.0] * 6, now_t=10.0)
        self.assertTrue(node._publish_streaming_joint_ctrl(now_t=10.0))
        node._update_stream_target([0.3] * 6, now_t=10.1)

        self.assertTrue(node._publish_streaming_joint_ctrl(now_t=10.15))
        self.assertFalse(node._publish_streaming_joint_ctrl(now_t=10.151))
        self.assertTrue(node._publish_streaming_joint_ctrl(now_t=10.20))
        self.assertTrue(node._publish_streaming_joint_ctrl(now_t=10.30))

        factor = 57324.840764
        positions = [call[0] / factor for call in node.piper.joint_calls]
        self.assertEqual(positions, sorted(positions))
        self.assertLessEqual(max(positions), 0.30001)
        self.assertAlmostEqual(positions[1], 0.075, places=3)
        self.assertAlmostEqual(positions[-1], 0.3, places=3)
        self.assertAlmostEqual(node._stream_segment_duration, 0.2)

    def test_stream_stale_input_holds_latest_target_without_overshoot(self):
        node = self._node([True] * 6)
        node.is_holding = False
        node._update_stream_target([0.0] * 6, now_t=20.0)
        node._publish_streaming_joint_ctrl(now_t=20.0)
        node._update_stream_target([0.1] * 6, now_t=20.05)
        node._publish_streaming_joint_ctrl(now_t=20.10)
        node._publish_streaming_joint_ctrl(now_t=21.0)

        factor = 57324.840764
        self.assertAlmostEqual(node.piper.joint_calls[-1][0] / factor, 0.1, places=4)
        self.assertLessEqual(
            max(call[0] / factor for call in node.piper.joint_calls), 0.10001
        )

    def test_stream_invalid_input_interval_uses_safe_default(self):
        node = self._node([True] * 6)
        node.is_holding = False
        node._update_stream_target([0.0] * 6, now_t=40.0)
        node._update_stream_target([0.015] * 6, now_t=40.0)

        self.assertAlmostEqual(node._stream_segment_duration, 0.02)
        self.assertTrue(node._publish_streaming_joint_ctrl(now_t=40.01))
        factor = 57324.840764
        self.assertAlmostEqual(node.piper.joint_calls[-1][0] / factor, 0.0075, places=4)

    def test_motion_reset_invalidates_pending_stream(self):
        node = self._node([True] * 6)
        node.is_holding = False
        node._update_stream_target([0.2] * 6, now_t=50.0)
        generation = node._stream_generation

        node._reset_motion_state()

        self.assertFalse(node._streaming_control)
        self.assertGreater(node._stream_generation, generation)
        self.assertFalse(node._publish_streaming_joint_ctrl(now_t=51.0))
        self.assertEqual(node.piper.joint_calls, [])

    def test_endpoint_cancels_pending_stream_before_immediate_send(self):
        node = self._node([True] * 6)
        node.gripper_val_mutiple = 1
        node.position_tolerance = 0.01
        node.is_holding = False
        node._update_stream_target([0.2] * 6, now_t=30.0)

        message = self.driver.JointState()
        message.header.frame_id = "piper_endpoint"
        message.name = [f"joint{i}" for i in range(1, 7)]
        message.position = [0.05] * 6
        message.velocity = [10.0] * 6
        message.effort = [0.0] * 6
        node.joint_callback(message)

        self.assertFalse(node._streaming_control)
        self.assertFalse(node._publish_streaming_joint_ctrl(now_t=31.0))
        factor = 57324.840764
        self.assertAlmostEqual(node.piper.joint_calls[-1][0] / factor, 0.05, places=4)

    def test_non_finite_stream_target_is_rejected(self):
        node = self._node([True] * 6)
        node.gripper_val_mutiple = 1
        message = self.driver.JointState()
        message.header.frame_id = "piper_stream"
        message.name = [f"joint{i}" for i in range(1, 7)]
        message.position = [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
        message.velocity = [30.0] * 6
        message.effort = [0.0] * 6

        node.joint_callback(message)

        self.assertFalse(node._streaming_control)
        self.assertEqual(node.piper.joint_calls, [])
        self.assertEqual(node.piper.motion2_calls, [])


if __name__ == "__main__":
    unittest.main()
