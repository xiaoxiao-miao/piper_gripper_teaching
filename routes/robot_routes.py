"""
로봇 직접 제어 API (고급 기능)
  POST /api/robot/gripper/open
  POST /api/robot/gripper/close
  POST /api/robot/preset  body: {"index": 0}
  GET  /api/robot/state
  GET  /api/robot/safe_waypoints
  POST /api/robot/safe_waypoints/record
  POST /api/robot/safe_waypoints/save   body: {"waypoints": [[q1..q6], ...]}
"""

import json
import os

from flask import Blueprint, request, jsonify, current_app

bp = Blueprint("robot", __name__, url_prefix="/api/robot")


def _waypoints_file() -> str:
    from config import DATASET_PATH
    return os.path.join(DATASET_PATH, "safe_return_waypoints.json")


def _ctrl():
    return current_app.config["CONTROLLER"]


@bp.get("/state")
def robot_state():
    ctrl = _ctrl()
    if ctrl._piper is None:
        return jsonify({"connected": False})
    return jsonify({"connected": True, **ctrl._piper.get_state()})


@bp.post("/gripper/set")
def gripper_set():
    """body: {"pct": 25}  → 0~100% 위치로 이동"""
    data = request.get_json(silent=True) or {}
    pct = float(data.get("pct", 100))
    return jsonify(_ctrl().gripper_set(pct))


@bp.post("/gripper/open")
def gripper_open():
    return jsonify(_ctrl().gripper_open())


@bp.post("/gripper/close")
def gripper_close():
    return jsonify(_ctrl().gripper_close())


@bp.post("/hold")
def hold():
    return jsonify(_ctrl().hold_position())


@bp.post("/motors/enable")
def motors_enable():
    return jsonify(_ctrl().motors_enable())


@bp.post("/motors/disable")
def motors_disable():
    return jsonify(_ctrl().motors_disable())


@bp.post("/motors/recover")
def motors_recover():
    data = request.get_json(silent=True) or {}
    return jsonify(_ctrl().motors_recover(confirmed=data.get("confirmed") is True))


@bp.post("/phone_mouse/enable")
def phone_mouse_enable():
    """Phone Mouse 모드 진입: freedrive 타이머 취소 + 즉시 teach recording 시작"""
    return jsonify(_ctrl().enable_phone_mouse())


@bp.post("/calibrate/start")
def calibrate_start():
    """ROM calibration 시작: 모터 OFF, min/max 추적"""
    return jsonify(_ctrl().start_calibration())


@bp.post("/calibrate/stop")
def calibrate_stop():
    """ROM calibration 종료: limits 저장 후 이전 상태 복귀"""
    return jsonify(_ctrl().stop_calibration())


@bp.post("/calibrate/resume")
def calibrate_resume():
    """Hold 이후 기존 측정 범위를 유지한 채 freedrive 재개."""
    return jsonify(_ctrl().resume_calibration())


@bp.get("/calibrate/limits")
def calibrate_limits():
    """현재 적용 중인 joint limits 조회"""
    ctrl = _ctrl()
    if ctrl._piper is None:
        return jsonify({"ok": False, "reason": "Robot not connected"})
    limits = {}
    for i, (lo, hi) in enumerate(ctrl._piper._pm_joint_limits):
        limits[f"j{i+1}"] = {"min": round(lo, 4), "max": round(hi, 4)}
    return jsonify({"ok": True, "limits": limits})


@bp.post("/preset")
def move_preset():
    """safe ready pose로 이동 (SAFE_RETURN_WAYPOINTS 사용)"""
    ctrl = _ctrl()
    if ctrl._piper is None:
        return jsonify({"ok": False, "reason": "Robot not connected"})

    data = request.get_json(silent=True) or {}
    idx = int(data.get("index", 0))

    if idx == 0:
        ok = ctrl._piper.go_to_safe_ready()
        if not ok:
            return jsonify({"ok": False, "reason": "No safe waypoints defined. Record them in Setup page."})
        return jsonify({"ok": True, "preset": "safe_ready"})

    return jsonify({"ok": False, "reason": "Only preset 0 (safe ready) supported"})


# ── Safe Return Waypoints ────────────────────────────────────────────────────

@bp.get("/safe_waypoints")
def get_safe_waypoints():
    """저장된 safe return waypoints 반환."""
    f = _waypoints_file()
    if os.path.exists(f):
        try:
            with open(f) as fp:
                data = json.load(fp)
            return jsonify({"ok": True, "waypoints": data})
        except Exception as e:
            return jsonify({"ok": False, "reason": str(e), "waypoints": []})
    return jsonify({"ok": True, "waypoints": []})


@bp.post("/safe_waypoints/record")
def record_safe_waypoint():
    """현재 로봇 joint 위치를 캡처해 반환 (저장은 클라이언트→save 호출)."""
    ctrl = _ctrl()
    if ctrl._piper is None:
        return jsonify({"ok": False, "reason": "Robot not connected"})
    pos = ctrl._piper.get_state().get("position", [])
    if len(pos) < 6:
        return jsonify({"ok": False, "reason": "Joint data not available yet"})
    return jsonify({"ok": True, "position": [round(v, 5) for v in pos[:6]]})


@bp.post("/safe_waypoints/save")
def save_safe_waypoints():
    """waypoints 목록을 JSON 파일로 영구 저장."""
    data = request.get_json(silent=True) or {}
    waypoints = data.get("waypoints", [])
    if not isinstance(waypoints, list):
        return jsonify({"ok": False, "reason": "waypoints must be a list"})

    f = _waypoints_file()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as fp:
        json.dump(waypoints, fp, indent=2)
    return jsonify({"ok": True, "count": len(waypoints)})
