import os
import json
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nodes import replay_streamer
from controller import Controller, State


def _bare_streamer():
    worker = replay_streamer.ReplayStreamer.__new__(replay_streamer.ReplayStreamer)
    worker.stop_requested = False
    worker.phase = "RUNNING"
    worker.motor_status = [True] * 6
    worker.motor_status_t = 10.0
    worker.latest_feedback_t = 10.0
    worker.latest_position = [0.0] * 6
    worker.latest_velocity = [0.0] * 6
    worker.feedback_history = []
    worker.motor_fault_codes = [0] * 6
    worker.safety_latched = False
    worker.can_healthy = True
    worker.feedback_max_age_s = 0.10
    worker.motor_status_max_age_s = 0.50
    worker.trajectory = [
        {**{f"q{joint}": index * 0.1 for joint in range(1, 7)}, "gripper": 0.07}
        for index in range(5)
    ]
    worker.source_idx = 0.0
    worker.source_step = 0.3
    worker.period = 0.02
    worker.next_deadline = 10.0
    worker.max_lateness = 0.0
    worker.missed = 0
    worker.last_position = [0.0] * 6
    worker.gripper_open = 0.07
    worker.gripper_close = 0.0
    worker.gripper_is_open = True
    worker.gripper_position = 0.07
    worker.pending_gripper_position = 0.07
    worker.has_gripper_target = False
    worker.last_status_t = 10.0
    worker.resume_ramp_s = 0.15
    worker.phase_started = 10.0
    worker.ramp_start = [0.0] * 6
    worker.ramp_target = [0.1] * 6
    worker.ramp_source_idx = 0.3
    worker.current_gripper_event = None
    worker.gripper_events = []
    worker.convergence = {}
    worker.convergence_stable_since = None
    worker.convergence_best_error = 1.0
    worker.convergence_last_progress_t = 5.0
    worker.convergence_deadline = 9.0
    worker.convergence_hard_deadline = 10.0
    worker.convergence_extended = False
    worker.endpoint_next_publish_t = 0.0
    worker.published = []
    worker.finished = []
    worker._publish_arm = lambda position, streaming, source_idx=None, lateness=0.0: (
        worker.published.append((list(position), streaming, source_idx, lateness))
    )
    worker._emit = lambda _payload: None
    worker._finish = lambda completed=False, error="", **_kwargs: worker.finished.append(
        (completed, error)
    )
    return worker


class ReplayStreamerStateTests(unittest.TestCase):
    def test_position_drift_accepts_arrival_when_sdk_velocity_spikes(self):
        worker = _bare_streamer()
        worker.ramp_target = [0.001] * 6
        worker.latest_position = [0.001] * 6
        worker.latest_velocity = [0.501, 0.0, 0.0, 0.478, 0.0, 0.0]
        worker.feedback_history = [
            (9.82, [0.0] * 6, list(worker.latest_velocity)),
            (10.0, [0.001] * 6, list(worker.latest_velocity)),
        ]
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            status = worker._convergence_status()
        self.assertTrue(status["stationary"])
        self.assertEqual(status["stationary_source"], "position_drift")
        self.assertLessEqual(status["position_drift_200ms_rad"], 0.003)

    def test_motor_fault_is_classified_as_safety_locked(self):
        worker = _bare_streamer()
        worker.motor_fault_codes[1] = 180
        finished = []
        worker._finish = lambda completed=False, error="", **kwargs: finished.append(
            (completed, error, kwargs.get("abort_class"))
        )
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            worker._tick()
        self.assertEqual(finished[0][2], "SAFETY_LOCKED")
        self.assertIn("motor safety fault", finished[0][1])

    def test_discrete_half_open_target_is_preserved(self):
        worker = _bare_streamer()
        worker.has_gripper_target = True
        worker.gripper_position = 0.0
        worker.trajectory[0]["gripper_target"] = 0.035
        self.assertAlmostEqual(worker._desired_gripper_position(), 0.035)

    def test_legacy_measured_half_open_value_keeps_binary_behavior(self):
        worker = _bare_streamer()
        worker.has_gripper_target = False
        worker.gripper_position = 0.0
        worker.trajectory[0]["gripper"] = 0.035
        worker.gripper_open = 0.07
        worker.gripper_close = 0.0
        self.assertAlmostEqual(worker._desired_gripper_position(), 0.07)

    def test_50hz_ticks_advance_source_without_skipping(self):
        worker = _bare_streamer()
        for now in (10.0, 10.02, 10.04, 10.06):
            worker.latest_feedback_t = now
            worker.motor_status_t = now
            with mock.patch.object(replay_streamer.time, "monotonic", return_value=now):
                worker._tick()

        self.assertEqual(len(worker.published), 4)
        self.assertEqual(
            [round(item[2], 2) for item in worker.published],
            [0.0, 0.3, 0.6, 0.9],
        )
        self.assertLessEqual(max(worker.published[-1][0]), 0.09)

    def test_stale_feedback_aborts_before_publishing(self):
        worker = _bare_streamer()
        worker.latest_feedback_t = 9.0
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            worker._tick()
        self.assertEqual(worker.published, [])
        self.assertIn("stale feedback", worker.finished[0][1])

    def test_motor_disable_aborts_before_publishing(self):
        worker = _bare_streamer()
        worker.motor_status[-1] = False
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            worker._tick()
        self.assertEqual(worker.published, [])
        self.assertIn("motor status invalid", worker.finished[0][1])

    def test_resume_ramp_uses_150ms_and_reaches_next_source(self):
        worker = _bare_streamer()
        worker.phase = "RESUME_RAMP"
        worker.ramp_target = [0.3] * 6
        for now in (10.075, 10.15):
            worker.latest_feedback_t = now
            worker.motor_status_t = now
            with mock.patch.object(replay_streamer.time, "monotonic", return_value=now):
                worker._tick()

        self.assertAlmostEqual(worker.published[0][0][0], 0.15, places=5)
        self.assertAlmostEqual(worker.published[1][0][0], 0.30, places=5)
        self.assertEqual(worker.phase, "RUNNING")
        self.assertAlmostEqual(worker.source_idx, 0.3)

    def test_gripper_hold_timeout_aborts_instead_of_continuing(self):
        worker = _bare_streamer()
        worker.phase = "GRIPPER_HOLD"
        worker.phase_started = 5.0
        worker.ramp_target = [1.0] * 6
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            worker._tick()
        self.assertIn("gripper keyframe arm", worker.finished[0][1])

    def test_stop_finishes_as_partial_capture_without_another_command(self):
        worker = _bare_streamer()
        worker.stop_requested = True
        with mock.patch.object(replay_streamer.time, "monotonic", return_value=10.0):
            worker._tick()
        self.assertEqual(worker.published, [])
        self.assertEqual(worker.finished, [(False, "")])

    def test_stopped_worker_result_is_valid_partial_capture(self):
        worker = replay_streamer.ReplayStreamer.__new__(replay_streamer.ReplayStreamer)
        worker.completed = False
        worker.stop_requested = True
        worker.error = ""
        worker.commands = []
        worker.max_gap = 0.0
        worker.max_lateness = 0.0
        worker.missed = 0
        worker.gripper_events = []
        worker.started_at = 1.0
        worker.finished_at = 2.0
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker.command_path = os.path.join(tmp_dir, "commands.csv")
            worker.result_path = os.path.join(tmp_dir, "result.json")
            worker._write_outputs()
            with open(worker.result_path) as file:
                result = json.load(file)
        self.assertTrue(result["ok"])
        self.assertTrue(result["stopped"])
        self.assertFalse(result["completed"])

    def test_controller_worker_stop_routes_to_partial_cleanup(self):
        class StoppedPiper:
            @staticmethod
            def get_replay_worker_status():
                return {"result": {"ok": True, "stopped": True}}

        controller = Controller.__new__(Controller)
        controller._piper = StoppedPiper()
        controller._event_log = []
        controller._on_emit = lambda *_args: None
        calls = []
        controller._stop_replay_cleanup = lambda partial=False: calls.append(partial)

        controller._on_replay_done()

        self.assertEqual(calls, [True])

    def test_finish_does_not_shutdown_rclpy_inside_timer_callback(self):
        worker = replay_streamer.ReplayStreamer.__new__(replay_streamer.ReplayStreamer)
        worker.phase = "RUNNING"
        worker.latest_position = [0.0] * 6
        worker.last_position = [0.1] * 6
        worker.commands = []
        worker.max_gap = 0.0
        worker.max_lateness = 0.0
        worker.stop_requested = False
        worker.gripper_events = []
        worker.missed = 0
        worker.started_at = 1.0
        worker.published = []
        worker._publish_arm = lambda position, streaming: worker.published.append(
            (list(position), streaming)
        )
        worker._emit = lambda _payload: None
        worker._write_outputs = lambda: None
        worker.timer = mock.Mock()

        with mock.patch.object(replay_streamer.rclpy, "shutdown") as shutdown:
            worker._finish(completed=True)

        shutdown.assert_not_called()
        worker.timer.cancel.assert_called_once()
        self.assertEqual(worker.phase, "DONE")


class _SequencePiper:
    def __init__(self, result=None):
        self.result = result or {
            "ok": True,
            "started_at": 1.0,
            "finished_at": 2.0,
            "gripper_events": [],
        }
        self.return_targets = []
        self._last_transition_error = ""

    def go_to_safe_ready(self, final_position, done_callback, abort_callback):
        self.return_targets.append(list(final_position))
        self.done_callback = done_callback
        self.abort_callback = abort_callback
        return True

    def get_command_buffer(self):
        return []

    def get_executed_buffer(self):
        return []

    def get_replay_worker_status(self):
        return {"result": dict(self.result)}


class TeachStartPresetTests(unittest.TestCase):
    def test_teach_start_applies_configured_preset_before_freedrive(self):
        calls = []

        class FakePiper:
            started_callback = None

            def gripper_set(self, pct):
                calls.append(("gripper", pct))

            def start_teach(self, **kwargs):
                calls.append(("teach", kwargs["disable_delay"]))
                self.started_callback = kwargs["started_callback"]

        class FakeEpisode:
            current_episode_id = "episode_test"

            @staticmethod
            def get_current_take_name():
                return "take_001"

        controller = Controller.__new__(Controller)
        controller._lock = threading.Lock()
        controller._state = State.TEACH_READY
        controller._piper = FakePiper()
        controller._episode = FakeEpisode()
        controller._trajectory_check = {}
        controller._event_log = []
        controller._on_emit = lambda *_args: None

        result = controller.start_teach(task="test", operator="tester")

        self.assertTrue(result["ok"])
        self.assertEqual(result["gripper_pct"], 70.0)
        self.assertEqual(calls[:2], [("gripper", 70.0), ("teach", 5.0)])
        self.assertEqual(controller.get_state(), State.TEACH_STARTING)

        controller._piper.started_callback()
        self.assertEqual(controller.get_state(), State.TEACH_RECORDING)


class DebugSequenceTests(unittest.TestCase):
    def test_diagnostics_excludes_intentional_gripper_pause_gap(self):
        command = [
            {"t_host_ns": 1_000_000_000, "source_idx": 0.0, **{f"q{i}": 0.0 for i in range(1, 7)}},
            {"t_host_ns": 1_020_000_000, "source_idx": 0.3, **{f"q{i}": 0.1 for i in range(1, 7)}},
            {"t_host_ns": 2_000_000_000, "source_idx": 0.3, **{f"q{i}": 0.1 for i in range(1, 7)}},
            {"t_host_ns": 2_020_000_000, "source_idx": 0.6, **{f"q{i}": 0.2 for i in range(1, 7)}},
        ]
        worker = {"gripper_events": [{
            "hold_started_t_host_ns": 1_030_000_000,
            "resume_started_t_host_ns": 1_990_000_000,
        }]}
        diagnostics = Controller._debug_diagnostics(command, [], worker)
        self.assertEqual(diagnostics["command_gap_max_ms"], 20.0)
        self.assertEqual(diagnostics["gripper_pause_count"], 1)

    def _controller(self, root, result=None):
        controller = Controller.__new__(Controller)
        controller._debug_lock = threading.RLock()
        controller._lock = threading.Lock()
        controller._state = State.READY
        controller._piper = _SequencePiper(result)
        controller._debug_operation = "sequence_return"
        controller._debug_trajectory = []
        controller._debug_source = {}
        controller._debug_check = {}
        controller._debug_start_ready = False
        controller._debug_previous_state = None
        controller._debug_last_error = ""
        controller._debug_last_diagnostics = {}
        controller._event_log = []
        controller._on_emit = lambda *_args: None
        controller._set_state = lambda state: setattr(controller, "_state", state)
        rows_a = [{**{f"q{i}": 0.1 for i in range(1, 7)}, "gripper": 0.07}]
        rows_b = [{**{f"q{i}": -0.2 for i in range(1, 7)}, "gripper": 0.0}]
        controller._debug_sequence = {
            "active": True,
            "cancelled": False,
            "sequence_id": "test",
            "root": root,
            "cycles": 2,
            "cycle": 1,
            "item_index": 0,
            "label": "A",
            "phase": "RETURN",
            "speed_scale": 0.3,
            "completed_replays": 0,
            "total_replays": 4,
            "results": [],
            "entries": [
                {"label": "A", "trajectory": rows_a, "source": {"episode_id": "A"}, "check": {}},
                {"label": "B", "trajectory": rows_b, "source": {"episode_id": "B"}, "check": {}},
            ],
            "previous_state": State.READY,
            "started_at": "now",
        }
        return controller

    def test_each_item_begins_with_safe_return(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            controller._begin_debug_sequence_item()
            self.assertEqual(controller._piper.return_targets, [[0.1] * 6])
            self.assertEqual(controller._debug_sequence["phase"], "RETURN")

    def test_success_advances_a_to_b_without_direct_join(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root)
            controller._debug_trajectory = controller._debug_sequence["entries"][0]["trajectory"]
            controller._debug_source = {"episode_id": "A"}
            controller._debug_check = {}
            controller._debug_sequence_replay_done()

            self.assertEqual(controller._debug_sequence["item_index"], 1)
            self.assertEqual(controller._piper.return_targets, [[-0.2] * 6])
            self.assertTrue(os.path.isfile(os.path.join(root, "cycle_001", "A", "diagnostics.json")))

    def test_worker_failure_cancels_remaining_queue(self):
        with tempfile.TemporaryDirectory() as root:
            controller = self._controller(root, {"ok": False, "error": "worker crashed"})
            controller._debug_trajectory = controller._debug_sequence["entries"][0]["trajectory"]
            controller._debug_source = {"episode_id": "A"}
            controller._debug_check = {}
            controller._debug_sequence_replay_done()

            self.assertFalse(controller._debug_sequence["active"])
            self.assertEqual(controller._debug_sequence["phase"], "FAILED")
            self.assertEqual(controller._piper.return_targets, [])
            self.assertTrue(os.path.isfile(os.path.join(root, "summary.json")))


class _ReplayStartPiper:
    def __init__(self):
        self.started_speed = None
        self._last_transition_error = ""

    def get_state(self):
        return {"position": [0.0] * 6}

    def start_replay(self, _trajectory, speed_scale, done_callback):
        self.started_speed = speed_scale
        self.done_callback = done_callback
        return True


class ReplayStartAdmissionTests(unittest.TestCase):
    def _controller(self):
        controller = Controller.__new__(Controller)
        controller._state = State.REPLAY_READY
        controller._lock = threading.Lock()
        controller._replay_start_lock = threading.Lock()
        controller._smoothed_trajectory = [
            {**{f"q{i}": 0.0 for i in range(1, 7)}, "gripper": 0.07}
        ]
        controller._trajectory_check = {"max_joint_velocity_rad_s": 3.0319}
        controller._replay_speed = 0.5
        controller._piper = _ReplayStartPiper()
        controller._cameras = None
        controller._episode = mock.Mock()
        controller._episode.get_episode_dir.return_value = "/tmp"
        controller._event_log = []
        controller._on_emit = lambda *_args: None
        controller._set_state = lambda state: setattr(controller, "_state", state)
        return controller

    def test_nominal_50_percent_is_safely_clamped_at_near_boundary(self):
        controller = self._controller()
        result = controller.start_replay(0.5)
        self.assertTrue(result["ok"])
        self.assertTrue(result["speed_adjusted"])
        self.assertLess(controller._piper.started_speed, 0.495)
        self.assertEqual(controller._state, State.REPLAY_RECORDING)

    def test_materially_unsafe_speed_is_rejected_with_recommendation(self):
        controller = self._controller()
        result = controller.start_replay(0.7)
        self.assertFalse(result["ok"])
        self.assertEqual(result["recommended_max_speed"], 0.494)
        self.assertIsNone(controller._piper.started_speed)
        self.assertEqual(controller._state, State.REPLAY_READY)


if __name__ == "__main__":
    unittest.main()
