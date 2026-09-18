from flask import Blueprint, abort, current_app, jsonify, request

from camera_manifest import load_manifest, update_manifest_edits


bp = Blueprint("cameras", __name__, url_prefix="/api/cameras")


def _ctrl():
    return current_app.config["CONTROLLER"]


@bp.get("/manifest")
def get_manifest():
    return jsonify(load_manifest())


@bp.put("/manifest")
def put_manifest():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(update_manifest_edits(data))
    except (OSError, ValueError):
        return abort(400)


@bp.get("/<cam_id>/controls")
def get_camera_controls(cam_id: str):
    mgr = _ctrl()._cameras
    if mgr is None:
        return jsonify({"error": "camera manager not ready"}), 503
    return jsonify(mgr.get_camera_controls(cam_id))


@bp.put("/<cam_id>/controls")
def put_camera_controls(cam_id: str):
    mgr = _ctrl()._cameras
    if mgr is None:
        return jsonify({"error": "camera manager not ready"}), 503
    data = request.get_json(silent=True) or {}
    result = mgr.set_camera_controls(
        cam_id,
        exposure_us=data.get("exposure_us"),
        gain_db=data.get("gain_db"),
        exposure_auto=data.get("exposure_auto"),
        gain_auto=data.get("gain_auto"),
    )
    return jsonify(result), (400 if "error" in result else 200)
