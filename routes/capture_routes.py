"""
캡처 워크플로 관련 API
  POST /api/connect
  POST /api/ready
  POST /api/teach/start
  POST /api/teach/stop
  POST /api/home
  POST /api/replay/start
  POST /api/replay/stop
  POST /api/hold
  GET  /api/status
  GET  /api/diagnostics
"""

from flask import Blueprint, request, jsonify, current_app

bp = Blueprint("capture", __name__, url_prefix="/api")


def _ctrl():
    return current_app.config["CONTROLLER"]


@bp.get("/status")
def status():
    return jsonify(_ctrl().get_status())


@bp.get("/diagnostics")
def diagnostics():
    return jsonify(_ctrl().get_diagnostics())


@bp.get("/events")
def events():
    return jsonify(_ctrl().get_event_log())


@bp.post("/connect")
def connect():
    return jsonify(_ctrl().connect())


@bp.post("/ready")
def ready():
    return jsonify(_ctrl().confirm_ready_pose())


@bp.post("/teach/start")
def teach_start():
    data = request.get_json(silent=True) or {}
    return jsonify(_ctrl().start_teach(
        task=data.get("task", "unspecified"),
        operator=data.get("operator", "unknown"),
    ))


@bp.post("/teach/stop")
def teach_stop():
    return jsonify(_ctrl().stop_teach())


@bp.post("/home")
def return_home():
    return jsonify(_ctrl().return_home())


@bp.post("/home/direct")
def return_home_direct():
    """역방향 궤적 없이 safe waypoints로 바로 홈 복귀."""
    return jsonify(_ctrl().return_home_direct())


@bp.post("/replay/start")
def replay_start():
    data = request.get_json(silent=True) or {}
    speed = data.get("speed_scale")
    result = _ctrl().start_replay(speed_scale=speed)
    return jsonify(result), (200 if result.get("ok") else 409)


@bp.post("/replay/stop")
def replay_stop():
    return jsonify(_ctrl().stop_replay())


@bp.post("/hold")
def hold():
    return jsonify(_ctrl().hold_position())


@bp.post("/go_back")
def go_back():
    """이전 teaching 단계로 복귀 (episode 폐기)"""
    return jsonify(_ctrl().go_back_to_teach())


@bp.post("/replay/speed")
def set_speed():
    data = request.get_json(silent=True) or {}
    speed = float(data.get("speed_scale", 0.3))
    return jsonify(_ctrl().set_replay_speed(speed))
