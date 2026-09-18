#!/usr/bin/env python3
"""Isolated fixed-rate Piper replay publisher.

The Flask process writes a request JSON and launches this module in a fresh
Python process.  Keeping the ROS timer outside Flask/camera threads prevents
MJPEG and SocketIO GIL stalls from starving the arm command stream.
"""

import argparse
import csv
import json
import math
import signal
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt8MultiArray


DOF = 6


class ReplayStreamer(Node):
    def __init__(self, request: dict, result_path: str, command_path: str):
        super().__init__("piper_replay_streamer")
        self.request = request
        self.result_path = result_path
        self.command_path = command_path
        self.trajectory = list(request["trajectory"])
        self.speed = float(request["speed_scale"])
        self.command_hz = float(request.get("command_hz", 50.0))
        self.teach_hz = float(request.get("teach_hz", 50.0))
        self.source_step = self.teach_hz * self.speed / self.command_hz
        self.period = 1.0 / self.command_hz
        self.gripper_open = float(request.get("gripper_open_rad", 0.07))
        self.gripper_close = float(request.get("gripper_close_rad", 0.0))
        self.gripper_steps = int(request.get("gripper_steps", 8))
        self.gripper_step_delay = float(request.get("gripper_step_delay", 0.08))
        self.resume_ramp_s = float(request.get("resume_ramp_s", 0.15))
        self.feedback_max_age_s = float(request.get("feedback_max_age_s", 0.10))
        self.motor_status_max_age_s = float(
            request.get("motor_status_max_age_s", 0.50)
        )

        self.arm_pub = self.create_publisher(JointState, request["joint_ctrl_topic"], 10)
        self.create_subscription(
            JointState, request["joint_state_topic"], self._feedback_callback, 10
        )
        self.create_subscription(
            UInt8MultiArray, request["motor_status_topic"], self._motor_callback, 10
        )
        self.create_subscription(
            UInt8MultiArray, request.get("motor_fault_topic", "motor_fault_status"),
            self._motor_fault_callback, 10,
        )
        self.create_subscription(
            Bool, request.get("motor_safety_topic", "motor_safety_latched"),
            self._motor_safety_callback, 10,
        )
        self.create_subscription(
            Bool, request.get("can_health_topic", "can_health"),
            self._can_health_callback, 10,
        )

        self.latest_position = []
        self.latest_velocity = []
        self.latest_feedback_t = 0.0
        self.feedback_history = deque(maxlen=100)
        self.motor_status = [False] * DOF
        self.motor_status_t = 0.0
        self.motor_fault_codes = [0] * DOF
        self.safety_latched = False
        self.can_healthy = True
        self.phase = "WAIT_READY"
        self.ready_deadline = time.monotonic() + 2.0
        self.source_idx = 0.0
        self.last_position = [self.trajectory[0][f"q{i}"] for i in range(1, DOF + 1)]
        self.has_gripper_target = any(
            row.get("gripper_target") is not None for row in self.trajectory
        )
        gripper_key = "gripper_target" if self.has_gripper_target else "gripper"
        self.last_gripper = next(
            (row.get(gripper_key) for row in self.trajectory if row.get(gripper_key) is not None),
            None,
        )
        try:
            initial_gripper = float(self.last_gripper)
        except (TypeError, ValueError):
            initial_gripper = self.gripper_close
        if not math.isfinite(initial_gripper):
            initial_gripper = self.gripper_close
        self.gripper_position = min(
            self.gripper_open, max(self.gripper_close, initial_gripper)
        )
        self.gripper_is_open = (
            self.gripper_position > 0.02219
        )
        self.pending_gripper_open = self.gripper_is_open
        self.pending_gripper_position = self.gripper_position
        self.phase_started = time.monotonic()
        self.gripper_step = 0
        self.gripper_next_t = 0.0
        self.ramp_start = list(self.last_position)
        self.ramp_target = list(self.last_position)
        self.ramp_source_idx = 0.0
        self.commands = []
        self.gripper_events = []
        self.current_gripper_event = None
        self.started_at = None
        self.finished_at = None
        self.next_deadline = 0.0
        self.max_lateness = 0.0
        self.max_gap = 0.0
        self.last_command_t = 0.0
        self.missed = 0
        self.stop_requested = False
        self.completed = False
        self.error = ""
        self.abort_class = None
        self.convergence = {}
        self.convergence_stable_since = None
        self.convergence_best_error = float("inf")
        self.convergence_last_progress_t = 0.0
        self.convergence_deadline = 0.0
        self.convergence_hard_deadline = 0.0
        self.convergence_extended = False
        self.endpoint_next_publish_t = 0.0
        self.last_status_t = 0.0
        self.timer = self.create_timer(self.period, self._tick)

    def _feedback_callback(self, msg):
        if len(msg.position) >= DOF:
            self.latest_position = [float(v) for v in msg.position[:DOF]]
            self.latest_velocity = [float(v) for v in msg.velocity[:DOF]]
            self.latest_feedback_t = time.monotonic()
            self.feedback_history.append((
                self.latest_feedback_t,
                list(self.latest_position),
                list(self.latest_velocity),
            ))

    def _motor_callback(self, msg):
        values = list(msg.data)[:DOF]
        if len(values) == DOF:
            self.motor_status = [bool(value) for value in values]
            self.motor_status_t = time.monotonic()

    def _motor_fault_callback(self, msg):
        values = [int(value) & 0xFF for value in list(msg.data)[:DOF]]
        if len(values) == DOF:
            self.motor_fault_codes = values

    def _motor_safety_callback(self, msg):
        self.safety_latched = bool(msg.data)

    def _can_health_callback(self, msg):
        self.can_healthy = bool(msg.data)

    @staticmethod
    def _emit(payload):
        print("REPLAY_STATUS " + json.dumps(payload, separators=(",", ":")), flush=True)

    def request_stop(self):
        self.stop_requested = True

    def _fresh_feedback(self, now):
        return (
            len(self.latest_position) >= DOF
            and now - self.latest_feedback_t <= self.feedback_max_age_s
        )

    def _fresh_motor_status(self, now):
        return self.motor_status_t > 0.0 and (
            now - self.motor_status_t <= self.motor_status_max_age_s
        )

    def _publish_arm(self, position, streaming, source_idx=None, lateness=0.0):
        msg = JointState()
        msg.header.frame_id = "piper_stream" if streaming else "piper_endpoint"
        msg.name = [f"joint{i}" for i in range(1, DOF + 1)]
        msg.position = [float(value) for value in position]
        msg.velocity = [min(100.0, max(10.0, self.speed * 100.0))] * DOF
        msg.effort = [0.0] * DOF
        self.arm_pub.publish(msg)
        if not streaming:
            return
        now_ns = time.time_ns()
        now = time.monotonic()
        if self.last_command_t:
            self.max_gap = max(self.max_gap, now - self.last_command_t)
        self.last_command_t = now
        row = {
            "t_host_ns": now_ns,
            "scheduled_t_host_ns": now_ns - int(lateness * 1e9),
            "waypoint_idx": int(round(source_idx or 0.0)),
            "source_idx": round(float(source_idx or 0.0), 4),
            "lateness_ms": round(lateness * 1000.0, 3),
            "velocity": msg.velocity[0],
            "gripper": self.gripper_position,
            "gripper_effort": 1.0 if self.gripper_is_open else 3.0,
        }
        row.update({f"q{i + 1}": float(position[i]) for i in range(DOF)})
        self.commands.append(row)

    def _publish_gripper(self, position, effort):
        msg = JointState()
        msg.name = ["gripper"]
        msg.position = [float(position)]
        msg.velocity = [5.0]
        msg.effort = [float(effort)]
        self.arm_pub.publish(msg)

    def _trajectory_position(self, index):
        index = min(max(0.0, index), len(self.trajectory) - 1.0)
        lo = int(index)
        hi = min(len(self.trajectory) - 1, lo + 1)
        alpha = index - lo
        return [
            (1.0 - alpha) * float(self.trajectory[lo][f"q{joint}"])
            + alpha * float(self.trajectory[hi][f"q{joint}"])
            for joint in range(1, DOF + 1)
        ]

    def _desired_gripper_position(self):
        row = self.trajectory[int(min(self.source_idx, len(self.trajectory) - 1))]
        value = row.get("gripper_target") if self.has_gripper_target else row.get("gripper")
        if value is None:
            return self.gripper_position
        try:
            value = float(value)
        except (TypeError, ValueError):
            return self.gripper_position
        if not math.isfinite(value):
            return self.gripper_position
        if self.has_gripper_target:
            return min(self.gripper_open, max(self.gripper_close, value))
        # Legacy recordings only contain measured ramp feedback. Keep their
        # former binary behavior so a physical ramp does not become many stops.
        return self.gripper_open if value > 0.02219 else self.gripper_close

    def _begin_gripper_hold(self, now, position, desired_position):
        self._publish_arm(position, streaming=False)
        # The intentional gripper pause is not a command-stream timing gap.
        self.last_command_t = 0.0
        self.phase = "GRIPPER_HOLD"
        self.phase_started = now
        self.pending_gripper_position = desired_position
        self.pending_gripper_open = desired_position > 0.02219
        self.ramp_target = list(position)
        initial = self._convergence_status()
        distance = initial["max_error_rad"]
        expected_rate = max(0.10, self.speed * 0.50)
        timeout = min(8.0, max(4.0, distance / expected_rate + 1.5))
        self.convergence = initial
        self.convergence_stable_since = None
        self.convergence_best_error = distance
        self.convergence_last_progress_t = now
        self.convergence_deadline = now + timeout
        self.convergence_hard_deadline = now + min(8.0, timeout + 2.0)
        self.convergence_extended = False
        self.endpoint_next_publish_t = now + 0.20
        self.current_gripper_event = {
            "source_idx": round(float(self.source_idx), 4),
            "from": "open" if self.gripper_is_open else "closed",
            "to": round(desired_position, 5),
            "hold_started_t_host_ns": time.time_ns(),
        }
        self._emit({"event": "phase", "phase": self.phase, "source_idx": self.source_idx})

    def _convergence_status(self):
        if len(self.latest_position) < DOF:
            return {
                "target": list(self.ramp_target), "feedback": [], "error_rad": [],
                "velocity_rad_s": [], "max_error_rad": float("inf"),
                "max_velocity_rad_s": float("inf"), "max_error_joint": None,
            }
        errors = [abs(self.latest_position[i] - self.ramp_target[i]) for i in range(DOF)]
        velocities = [abs(v) for v in self.latest_velocity[:DOF]]
        error = max(errors)
        velocity = max(abs(v) for v in self.latest_velocity) if self.latest_velocity else 0.0
        now = time.monotonic()
        history = [
            sample for sample in getattr(self, "feedback_history", [])
            if now - sample[0] <= 0.20
        ]
        if len(history) >= 2 and history[-1][0] - history[0][0] >= 0.10:
            position_drift = max(
                max(sample[1][joint] for sample in history)
                - min(sample[1][joint] for sample in history)
                for joint in range(DOF)
            )
        else:
            position_drift = float("inf")
        stationary = velocity <= 0.12 or position_drift <= 0.003
        return {
            "target": [round(v, 6) for v in self.ramp_target],
            "feedback": [round(v, 6) for v in self.latest_position[:DOF]],
            "error_rad": [round(v, 6) for v in errors],
            "velocity_rad_s": [round(v, 6) for v in velocities],
            "max_error_rad": round(error, 6),
            "max_velocity_rad_s": round(velocity, 6),
            "position_drift_200ms_rad": (
                round(position_drift, 6) if math.isfinite(position_drift) else None
            ),
            "stationary": stationary,
            "stationary_source": (
                "velocity" if velocity <= 0.12 else "position_drift"
                if position_drift <= 0.003 else "none"
            ),
            "max_error_joint": errors.index(error) + 1,
        }

    def _feedback_converged(self):
        status = self._convergence_status()
        return status["max_error_rad"] <= 0.04 and bool(status.get("stationary"))

    def _finish(self, completed=False, error="", abort_class=None):
        if self.phase == "DONE":
            return
        position = (
            self.last_position if completed
            else self.latest_position if len(self.latest_position) >= DOF
            else self.last_position
        )
        self._publish_arm(position, streaming=False)
        self.completed = bool(completed)
        self.error = str(error)
        self.abort_class = abort_class or (None if completed or self.stop_requested else "MOTION_ABORTED")
        self.finished_at = time.time()
        self.phase = "DONE"
        self._emit({
            "event": "done", "completed": self.completed,
            "stopped": self.stop_requested, "error": self.error,
            "abort_class": self.abort_class,
            "commands": len(self.commands), "max_gap_ms": round(self.max_gap * 1000.0, 3),
            "max_lateness_ms": round(self.max_lateness * 1000.0, 3),
        })
        self._write_outputs()
        self.timer.cancel()

    def _write_outputs(self):
        fields = [
            "t_host_ns", "scheduled_t_host_ns", "waypoint_idx", "source_idx",
            "lateness_ms", "velocity", *[f"q{i}" for i in range(1, 7)],
            "gripper", "gripper_effort",
        ]
        with open(self.command_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.commands)
        result = {
            # An operator Stop is a valid partial capture, not a worker error.
            "ok": (self.completed or self.stop_requested) and not self.error,
            "completed": self.completed,
            "stopped": self.stop_requested,
            "error": self.error,
            "command_samples": len(self.commands),
            "max_gap_ms": round(self.max_gap * 1000.0, 3),
            "max_lateness_ms": round(self.max_lateness * 1000.0, 3),
            "missed_deadlines": self.missed,
            "gripper_events": self.gripper_events,
            "abort_class": getattr(self, "abort_class", None),
            "convergence": getattr(self, "convergence", {}),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        with open(self.result_path, "w") as file:
            json.dump(result, file, indent=2)

    def _tick(self):
        now = time.monotonic()
        if self.stop_requested:
            self._finish()
            return

        if self.phase == "WAIT_READY":
            fault_codes = [
                code & ~(1 << 6) for code in getattr(self, "motor_fault_codes", [0] * DOF)
            ]
            if getattr(self, "safety_latched", False) or any(fault_codes):
                self._finish(
                    error=f"motor safety fault: {fault_codes}",
                    abort_class="SAFETY_LOCKED",
                )
                return
            if not getattr(self, "can_healthy", True):
                self._finish(error="CAN feedback lost", abort_class="CAN_LOST")
                return
            if (
                self._fresh_feedback(now)
                and self._fresh_motor_status(now)
                and all(self.motor_status)
            ):
                start_error = max(
                    abs(self.latest_position[i] - self.last_position[i]) for i in range(DOF)
                )
                if start_error > 0.06:
                    self._finish(error=f"start pose error {start_error:.4f}rad exceeds 0.06rad")
                    return
                self.started_at = time.time()
                self.phase = "RUNNING"
                self.phase_started = now
                self.next_deadline = now
                self._publish_gripper(
                    self.gripper_position,
                    1.0 if self.gripper_is_open else 3.0,
                )
                self._emit({"event": "ready", "phase": self.phase})
            elif now >= self.ready_deadline:
                if not self._fresh_feedback(now):
                    reason = "stale feedback"
                elif not self._fresh_motor_status(now):
                    reason = "stale motor enable status"
                else:
                    reason = f"motors not enabled: {[int(v) for v in self.motor_status]}"
                self._finish(error=reason)
            return

        fault_codes = [
            code & ~(1 << 6) for code in getattr(self, "motor_fault_codes", [0] * DOF)
        ]
        if getattr(self, "safety_latched", False) or any(fault_codes):
            self._finish(
                error=f"motor safety fault: {fault_codes}",
                abort_class="SAFETY_LOCKED",
            )
            return
        if not getattr(self, "can_healthy", True):
            self._finish(error="CAN feedback lost", abort_class="CAN_LOST")
            return
        if not self._fresh_motor_status(now) or not all(self.motor_status):
            self._finish(
                error=f"motor status invalid: {[int(v) for v in self.motor_status]}",
                abort_class="SAFETY_LOCKED",
            )
            return
        if not self._fresh_feedback(now):
            self._finish(error=f"stale feedback age={now - self.latest_feedback_t:.3f}s")
            return

        if self.phase == "RUNNING":
            lateness = max(0.0, now - self.next_deadline)
            self.max_lateness = max(self.max_lateness, lateness)
            if lateness > self.period:
                self.missed += int(lateness / self.period)
            position = self._trajectory_position(self.source_idx)
            desired_gripper = self._desired_gripper_position()
            if abs(desired_gripper - self.gripper_position) > 0.002:
                self._begin_gripper_hold(now, position, desired_gripper)
                return
            self._publish_arm(position, True, self.source_idx, lateness)
            self.last_position = position
            if self.source_idx >= len(self.trajectory) - 1.0:
                self._finish(completed=True)
                return
            self.source_idx = min(
                len(self.trajectory) - 1.0, self.source_idx + self.source_step
            )
            self.next_deadline = now + self.period

        elif self.phase == "GRIPPER_HOLD":
            if now + 1e-9 >= self.endpoint_next_publish_t:
                self._publish_arm(self.ramp_target, streaming=False)
                self.endpoint_next_publish_t = now + 0.20

            status = self._convergence_status()
            status.update({
                "waited_s": round(now - self.phase_started, 3),
                "extended": self.convergence_extended,
            })
            self.convergence = status
            error = status["max_error_rad"]
            velocity = status["max_velocity_rad_s"]
            if self.convergence_best_error - error >= 0.01:
                self.convergence_best_error = error
                self.convergence_last_progress_t = now

            if error <= 0.04 and bool(status.get("stationary")):
                self.convergence_stable_since = self.convergence_stable_since or now
                if now - self.convergence_stable_since >= 0.10:
                    self.phase = "GRIPPER_RAMP"
                    self.phase_started = now
                    self.gripper_step = 0
                    self.gripper_next_t = now
                    if self.current_gripper_event is not None:
                        self.current_gripper_event["converged_t_host_ns"] = time.time_ns()
                        self.current_gripper_event["convergence"] = dict(self.convergence)
                    return
            else:
                self.convergence_stable_since = None

            if (
                error > 0.04
                and now - self.phase_started >= 0.5
                and now - self.convergence_last_progress_t >= 0.5
            ):
                self._finish(
                    error=(
                        "gripper keyframe arm stalled: "
                        f"J{status['max_error_joint']} error={error:.4f}rad"
                    ),
                    abort_class="MOTION_ABORTED",
                )
                return
            if now >= self.convergence_deadline:
                recently_progressing = now - self.convergence_last_progress_t < 0.5
                if (
                    not self.convergence_extended
                    and recently_progressing
                    and now < self.convergence_hard_deadline
                ):
                    self.convergence_extended = True
                    self.convergence_deadline = self.convergence_hard_deadline
                else:
                    self._finish(
                        error=(
                            "gripper keyframe arm convergence timeout: "
                            f"J{status['max_error_joint']} error={error:.4f}rad "
                            f"velocity={velocity:.4f}rad/s"
                        ),
                        abort_class="MOTION_ABORTED",
                    )
                    return

        elif self.phase == "GRIPPER_RAMP":
            if now + 1e-9 < self.gripper_next_t:
                return
            self.gripper_step += 1
            start = self.gripper_position
            target = self.pending_gripper_position
            alpha = min(1.0, self.gripper_step / max(1, self.gripper_steps))
            effort = 1.0 if self.pending_gripper_open else 3.0
            self._publish_gripper(start + (target - start) * alpha, effort)
            self.gripper_next_t = now + self.gripper_step_delay
            if self.gripper_step >= self.gripper_steps:
                self.gripper_is_open = self.pending_gripper_open
                self.gripper_position = target
                self.phase = "GRIPPER_SETTLE"
                self.phase_started = now
                if self.current_gripper_event is not None:
                    self.current_gripper_event["gripper_finished_t_host_ns"] = time.time_ns()

        elif self.phase == "GRIPPER_SETTLE":
            if now - self.phase_started >= 0.15:
                self.phase = "RESUME_RAMP"
                self.phase_started = now
                if self.current_gripper_event is not None:
                    self.current_gripper_event["resume_started_t_host_ns"] = time.time_ns()
                self.ramp_start = (
                    list(self.latest_position) if len(self.latest_position) >= DOF
                    else list(self.last_position)
                )
                self.ramp_source_idx = min(
                    len(self.trajectory) - 1.0, self.source_idx + self.source_step
                )
                self.ramp_target = self._trajectory_position(self.ramp_source_idx)

        elif self.phase == "RESUME_RAMP":
            alpha = min(1.0, (now - self.phase_started) / max(0.01, self.resume_ramp_s))
            position = [
                self.ramp_start[i] + (self.ramp_target[i] - self.ramp_start[i]) * alpha
                for i in range(DOF)
            ]
            self._publish_arm(position, True, self.source_idx, 0.0)
            self.last_position = position
            if alpha >= 1.0:
                if self.current_gripper_event is not None:
                    self.current_gripper_event["resume_finished_t_host_ns"] = time.time_ns()
                    self.gripper_events.append(self.current_gripper_event)
                    self.current_gripper_event = None
                self.source_idx = self.ramp_source_idx
                self.phase = "RUNNING"
                self.next_deadline = now + self.period

        if now - self.last_status_t >= 0.5 and self.phase != "DONE":
            self.last_status_t = now
            self._emit({
                "event": "progress", "phase": self.phase,
                "progress": self.source_idx / max(1.0, len(self.trajectory) - 1.0),
                "source_idx": round(self.source_idx, 3),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--commands", required=True)
    args = parser.parse_args()
    with open(args.request) as file:
        request = json.load(file)
    if len(request.get("trajectory", [])) < 2:
        raise SystemExit("trajectory must contain at least two samples")

    rclpy.init()
    node = ReplayStreamer(request, args.result, args.commands)
    signal.signal(signal.SIGTERM, lambda *_args: node.request_stop())
    signal.signal(signal.SIGINT, lambda *_args: node.request_stop())
    try:
        # Never call rclpy.shutdown() from inside a timer callback.  Humble can
        # wait on the currently executing callback and deadlock, leaving the
        # parent forever in REPLAY even though the final endpoint was sent.
        while rclpy.ok() and node.phase != "DONE":
            rclpy.spin_once(node, timeout_sec=0.10)
    except KeyboardInterrupt:
        node.request_stop()
        while rclpy.ok() and node.phase != "DONE":
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node.phase != "DONE":
            node._finish(error=node.error or "worker terminated")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
