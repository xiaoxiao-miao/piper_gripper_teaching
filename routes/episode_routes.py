"""
Episode 관리 API
  GET    /api/episodes
  GET    /api/episodes/<episode_id>
  POST   /api/episodes/save     body: {"success": true, "reason": ""}
  POST   /api/episodes/discard
  POST   /api/episodes/retake_replay
  POST   /api/episodes/retake_teach
  POST   /api/episodes/add_take   (SAVED 상태에서 이어서 촬영)
  POST   /api/episodes/new
  DELETE /api/episodes/<episode_id>
  DELETE /api/episodes/<episode_id>/takes/<take>
  GET    /api/episodes/<episode_id>/video              ← 최신 take의 video.mp4
  GET    /api/episodes/<episode_id>/video_depth        ← 최신 take의 video_depth.mp4
  GET    /api/episodes/<episode_id>/takes/<take>/video
  GET    /api/episodes/<episode_id>/takes/<take>/video_depth
  GET    /api/episodes/<episode_id>/takes/<take>/joints      ← joint CSV data (teach/executed/command)
  GET    /api/episodes/<episode_id>/takes/<take>/trajectory  ← FK trajectory (ee + link positions)
  GET    /api/episodes/<episode_id>/frame/<frame_idx>        ← 최신 take의 frame
  POST   /api/episodes/<episode_id>/takes/<take>/regenerate_videos  ← 카메라 영상 재생성
"""

import csv as csv_module
import json
import os
import uuid
from datetime import datetime
from typing import Optional

from flask import Blueprint, current_app, jsonify, request, send_file, abort
from camera_manifest import camera_by_id, default_mask_camera_id, overview_camera, video_file_name, camera_csv_name, frames_dir_name

bp = Blueprint("episodes", __name__, url_prefix="/api/episodes")


def _ctrl():
    return current_app.config["CONTROLLER"]


def _mgr():
    return _ctrl()._episode


def _take_dir(episode_id: str, take: Optional[str] = None) -> Optional[str]:
    """take 디렉터리 반환. take=None이면 최신 take."""
    mgr = _mgr()
    if take:
        from config import DATASET_PATH
        dataset_root = os.path.realpath(DATASET_PATH)
        d = os.path.realpath(os.path.join(dataset_root, episode_id, "takes", take))
        if os.path.commonpath([dataset_root, d]) != dataset_root:
            return None
        return d if os.path.isdir(d) else None
    return mgr.get_latest_take_dir(episode_id)


def _latest_take_name(episode_id: str) -> str | None:
    td = _take_dir(episode_id)
    return os.path.basename(td) if td else None


def _review_issues_path(episode_id: str, take: str) -> str | None:
    td = _take_dir(episode_id, take)
    if not td:
        return None
    return os.path.join(td, "review_issues.json")


def _load_review_issues_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    issues = data.get("issues", []) if isinstance(data, dict) else data
    if not isinstance(issues, list):
        return []
    return [i for i in issues if isinstance(i, dict)]


def _save_review_issues_file(path: str, issues: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)


def _normalize_review_issue(raw: dict) -> dict:
    key = str(raw.get("key") or raw.get("issue_id") or f"issue_{uuid.uuid4().hex[:12]}")
    now = datetime.now().isoformat()
    issue = {
        "key": key,
        "label": str(raw.get("label") or key.replace("_", " ")),
        "severity": str(raw.get("severity") or "warning"),
        "source": str(raw.get("source") or "manual"),
        "status": str(raw.get("status") or "open"),
        "note": str(raw.get("note") or ""),
        "created_at": str(raw.get("created_at") or now),
        "updated_at": now,
    }
    for optional in ("task_key", "lerobot_index", "suggested_actions"):
        if optional in raw:
            issue[optional] = raw[optional]
    return issue


def _episode_label_path(episode_id: str) -> str:
    from config import DATASET_PATH
    return os.path.join(DATASET_PATH, episode_id, "label.json")


def _set_episode_success(episode_id: str, success: bool, reason: str = "") -> bool:
    from config import DATASET_PATH
    ep_dir = os.path.join(DATASET_PATH, episode_id)
    if not os.path.isdir(ep_dir):
        return False
    takes_dir = os.path.join(ep_dir, "takes")
    takes = sorted(d for d in os.listdir(takes_dir) if os.path.isdir(os.path.join(takes_dir, d))) if os.path.isdir(takes_dir) else []
    label = {
        "success": bool(success),
        "failure_reason": "" if success else reason,
        "labeled_at": datetime.now().isoformat(),
        "takes": takes,
    }
    with open(os.path.join(ep_dir, "label.json"), "w") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    meta_path = os.path.join(ep_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        if not isinstance(meta.get("task"), dict):
            meta["task"] = {}
        meta["task"]["success"] = bool(success)
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


def _episode_success(episode_id: str) -> bool | None:
    path = _episode_label_path(episode_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            label = json.load(f)
        return label.get("success")
    except (OSError, json.JSONDecodeError):
        return None


def _has_open_review_issues(episode_id: str) -> bool:
    take = _latest_take_name(episode_id)
    if not take:
        return False
    path = _review_issues_path(episode_id, take)
    if not path:
        return False
    return any(i.get("status", "open") in {"open", "needs_review"} for i in _load_review_issues_file(path))


def _video_duration_s(path: str) -> float | None:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and frames and frames > 0:
            return round(float(frames / fps), 2)
    except Exception:
        return None
    return None


def _preview_video_path(take_dir: str) -> str | None:
    candidates: list[str] = []
    for cam in [overview_camera(), camera_by_id("cam0"), camera_by_id("realsense")]:
        if cam:
            candidates.append(video_file_name(cam, "color"))
    candidates.append("video.mp4")
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(take_dir, name)
        if os.path.exists(path):
            return path
    return None


def _take_completeness(take_dir: str, ep_success: bool | None) -> dict:
    return {
        "rgb": os.path.exists(os.path.join(take_dir, "video.mp4")),
        "depth": os.path.exists(os.path.join(take_dir, "video_depth.mp4")),
        "cam0": os.path.exists(os.path.join(take_dir, "video_webcam_0.mp4")),
        "cam1": os.path.exists(os.path.join(take_dir, "video_webcam_1.mp4")),
        "joints": os.path.exists(os.path.join(take_dir, "teach_joint.csv")) and os.path.exists(os.path.join(take_dir, "executed_joint.csv")),
        "traj": os.path.exists(os.path.join(take_dir, "teach_joint.csv")),
        "label": ep_success is not None,
    }


@bp.get("")
def list_episodes():
    from config import DATASET_PATH
    paged = "page" in request.args or "page_size" in request.args
    page = max(1, request.args.get("page", default=1, type=int) or 1)
    page_size = min(100, max(1, request.args.get("page_size", default=24, type=int) or 24))
    if paged:
        total = _mgr().count_episodes()
        episodes = _mgr().list_episodes(
            offset=(page - 1) * page_size,
            limit=page_size,
            lightweight=True,
        )
    else:
        # Backward compatibility for scripts that still expect a bare array.
        total = None
        episodes = _mgr().list_episodes()
    # 각 에피소드의 마지막 take edit_meta.json 확인 → has_postprocess 플래그
    for ep in episodes:
        ep_id = ep.get("episode_id", "")
        takes_dir = os.path.join(DATASET_PATH, ep_id, "takes")
        has_pp = False
        latest_take = None
        if os.path.isdir(takes_dir):
            take_names = sorted(d for d in os.listdir(takes_dir)
                                if os.path.isdir(os.path.join(takes_dir, d)))
            if take_names:
                latest_take = take_names[-1]
                em_path = os.path.join(takes_dir, take_names[-1], "edit_meta.json")
                if os.path.exists(em_path):
                    try:
                        with open(em_path) as f:
                            em = json.load(f)
                        has_pp = (
                            bool(em.get("trim", {}).get("enabled"))
                            or bool(em.get("mask", {}).get("enabled"))
                            or em.get("timing", {}).get("mode") == "physical_replay"
                        )
                    except (OSError, json.JSONDecodeError):
                        pass
        ep["has_postprocess"] = has_pp
        if latest_take:
            td = os.path.join(takes_dir, latest_take)
            preview_path = _preview_video_path(td)
            ep["preview_video_url"] = f"/api/episodes/{ep_id}/preview_video" if preview_path else None
            # Opening every video with OpenCV dominated large-dataset list
            # latency.  Paged clients load duration only from joint/video detail.
            ep["duration_s"] = (
                None if paged else _video_duration_s(preview_path) if preview_path else None
            )
            ep["completeness"] = _take_completeness(td, ep.get("success"))
            latest_take_info = next((t for t in ep.get("takes", []) if t.get("take") == latest_take), None)
            ep["review_issues"] = (latest_take_info or {}).get("review_issues", [])
            ep["review_issue_count"] = len(ep["review_issues"])
        else:
            ep["preview_video_url"] = None
            ep["duration_s"] = None
            ep["completeness"] = {
                "rgb": False, "depth": False, "cam0": False, "cam1": False,
                "joints": False, "traj": False, "label": ep.get("success") is not None,
            }
            ep["review_issues"] = []
            ep["review_issue_count"] = 0
    if not paged:
        return jsonify(episodes)
    return jsonify({
        "items": episodes,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": page * page_size < total,
    })


@bp.get("/<episode_id>")
def get_episode(episode_id: str):
    meta = _mgr().get_episode_meta(episode_id)
    if meta is None:
        return abort(404)
    return jsonify(meta)


@bp.post("/save")
def save_episode():
    data = request.get_json(silent=True) or {}
    success = bool(data.get("success", True))
    reason = str(data.get("reason", ""))
    return jsonify(_ctrl().save_episode(success=success, reason=reason))


@bp.post("/discard")
def discard_episode():
    return jsonify(_ctrl().discard_episode())


@bp.post("/retake_replay")
def retake_replay():
    return jsonify(_ctrl().retake_replay())


@bp.post("/retake_teach")
def retake_teach():
    return jsonify(_ctrl().retake_teach())


@bp.post("/add_take")
def add_take():
    """SAVED 상태에서 같은 episode에 새 take 추가 (이어서 촬영)."""
    return jsonify(_ctrl().add_take_to_episode())


@bp.post("/new")
def new_episode():
    return jsonify(_ctrl().new_episode())


@bp.patch("/<episode_id>")
def update_episode(episode_id: str):
    data = request.get_json(silent=True) or {}
    ok = _mgr().update_episode_task(episode_id, str(data.get("task", "")))
    if not ok:
        return abort(404)
    return jsonify({"ok": True})


@bp.delete("/<episode_id>")
def delete_episode(episode_id: str):
    ok = _mgr().delete_episode(episode_id)
    if not ok:
        return abort(404)
    return jsonify({"ok": True})


@bp.delete("/<episode_id>/takes/<take>")
def delete_take(episode_id: str, take: str):
    ok = _mgr().delete_take(episode_id, take)
    if not ok:
        return abort(404)
    return jsonify({"ok": True})


# ── Dataset robot debug replay (no camera recording) ─────────────────

@bp.get("/debug_replay/status")
def debug_replay_status():
    return jsonify(_ctrl().get_debug_replay_status())


@bp.post("/<episode_id>/takes/<take>/debug_replay/load")
def debug_replay_load(episode_id: str, take: str):
    take_dir = _take_dir(episode_id, take)
    if not take_dir:
        return jsonify({"ok": False, "reason": "Episode take not found"}), 404
    data = request.get_json(silent=True) or {}
    result = _ctrl().load_debug_trajectory(
        take_dir=take_dir,
        episode_id=episode_id,
        take=take,
        source=str(data.get("source", "teach")),
    )
    return jsonify(result), (200 if result.get("ok") else 409)


@bp.post("/debug_replay/return")
def debug_replay_return():
    result = _ctrl().return_debug_replay_to_start()
    return jsonify(result), (200 if result.get("ok") else 409)


@bp.post("/debug_replay/start")
def debug_replay_start():
    data = request.get_json(silent=True) or {}
    result = _ctrl().start_debug_replay(data.get("speed_scale", 0.5))
    return jsonify(result), (200 if result.get("ok") else 409)


@bp.post("/debug_replay/stop")
def debug_replay_stop():
    result = _ctrl().stop_debug_replay()
    return jsonify(result), (200 if result.get("ok") else 409)


@bp.post("/debug_replay/sequence/start")
def debug_replay_sequence_start():
    data = request.get_json(silent=True) or {}
    raw_items = data.get("items") or []
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            return jsonify({"ok": False, "reason": "Each sequence item must be an object"}), 400
        episode_id = str(raw.get("episode_id", ""))
        take = str(raw.get("take", ""))
        take_dir = _take_dir(episode_id, take)
        if not take_dir:
            return jsonify({
                "ok": False,
                "reason": f"Episode take not found: {episode_id}/{take}",
            }), 404
        items.append({
            "episode_id": episode_id,
            "take": take,
            "take_dir": take_dir,
            "source": str(raw.get("source", "teach")),
        })
    result = _ctrl().start_debug_sequence(
        items=items,
        cycles=data.get("cycles", 1),
        speed_scale=data.get("speed_scale", 0.3),
    )
    return jsonify(result), (200 if result.get("ok") else 409)


# ── Dataset QA / review issues ────────────────────────────────────────

@bp.get("/<episode_id>/takes/<take>/review_issues")
def get_review_issues(episode_id: str, take: str):
    path = _review_issues_path(episode_id, take)
    if not path:
        return abort(404)
    return jsonify({"issues": _load_review_issues_file(path)})


@bp.post("/<episode_id>/takes/<take>/review_issues")
def add_review_issues(episode_id: str, take: str):
    path = _review_issues_path(episode_id, take)
    if not path:
        return abort(404)
    data = request.get_json(silent=True) or {}
    raw_issues = data.get("issues", data)
    if isinstance(raw_issues, dict):
        raw_issues = [raw_issues]
    if not isinstance(raw_issues, list):
        return abort(400)
    current = _load_review_issues_file(path)
    by_key = {i.get("key"): i for i in current}
    for raw in raw_issues:
        if not isinstance(raw, dict):
            continue
        issue = _normalize_review_issue(raw)
        by_key[issue["key"]] = {**by_key.get(issue["key"], {}), **issue}
    issues = list(by_key.values())
    _save_review_issues_file(path, issues)
    return jsonify({"issues": issues})


@bp.patch("/<episode_id>/takes/<take>/review_issues/<issue_key>")
def update_review_issue(episode_id: str, take: str, issue_key: str):
    path = _review_issues_path(episode_id, take)
    if not path:
        return abort(404)
    data = request.get_json(silent=True) or {}
    issues = _load_review_issues_file(path)
    found = False
    for issue in issues:
        if issue.get("key") == issue_key:
            for field in ("label", "severity", "source", "status", "note", "suggested_actions"):
                if field in data:
                    issue[field] = data[field]
            issue["updated_at"] = datetime.now().isoformat()
            found = True
            break
    if not found:
        return abort(404)
    _save_review_issues_file(path, issues)
    return jsonify({"issues": issues})


def _load_edit_meta(episode_id: str, take: str) -> tuple[str, dict] | tuple[None, None]:
    td = _take_dir(episode_id, take)
    if not td:
        return None, None
    path = os.path.join(td, "edit_meta.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
    else:
        meta = {}
    meta.setdefault("trim", {"enabled": False, "cut_t": None, "margin": 1.0})
    meta.setdefault("mask", {"enabled": False, "camera_id": default_mask_camera_id() or "", "polygon": [], "fill": "black"})
    meta.setdefault("timing", {"mode": "teaching_normalized"})
    return path, meta


def _apply_review_action(episode_id: str, take: str, issue: dict, action: dict) -> str:
    action_type = action.get("type")
    params = action.get("params", {}) if isinstance(action.get("params"), dict) else {}
    if action_type == "mark_failure":
        reason = params.get("reason") or issue.get("note") or issue.get("label") or "review issue marked failure"
        if not _set_episode_success(episode_id, False, str(reason)):
            return "failed"
        return "applied"
    if action_type == "trim":
        path, meta = _load_edit_meta(episode_id, take)
        if not path or meta is None:
            return "failed"
        trim = meta.setdefault("trim", {})
        trim["enabled"] = True
        if "cut_t" in params:
            trim["cut_t"] = params["cut_t"]
        if "margin" in params:
            trim["margin"] = params["margin"]
        with open(path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return "applied"
    if action_type == "set_mask":
        path, meta = _load_edit_meta(episode_id, take)
        if not path or meta is None:
            return "failed"
        mask = meta.setdefault("mask", {})
        mask.update(params)
        with open(path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return "applied"
    return "unsupported"


@bp.post("/review_issues/apply")
def apply_review_issues():
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    if not isinstance(items, list):
        return abort(400)
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        episode_id = str(item.get("episode_id", ""))
        take = str(item.get("take") or _latest_take_name(episode_id) or "")
        issue_key = item.get("issue_key")
        path = _review_issues_path(episode_id, take) if episode_id and take else None
        if not path:
            results.append({"episode_id": episode_id, "take": take, "status": "missing_take"})
            continue
        issues = _load_review_issues_file(path)
        selected = [i for i in issues if not issue_key or i.get("key") == issue_key]
        for issue in selected:
            actions = issue.get("suggested_actions", [])
            if isinstance(actions, dict):
                actions = [actions]
            applied = [_apply_review_action(episode_id, take, issue, a) for a in actions if isinstance(a, dict)]
            if applied and all(s == "applied" for s in applied):
                issue["status"] = "auto_fixed"
            elif applied:
                issue["status"] = "needs_review"
            issue["updated_at"] = datetime.now().isoformat()
            results.append({"episode_id": episode_id, "take": take, "issue_key": issue.get("key"), "actions": applied})
        _save_review_issues_file(path, issues)
    return jsonify({"results": results})


# ── Video serving ──────────────────────────────────────────────────────

@bp.get("/<episode_id>/video")
def get_video(episode_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = os.path.join(td, "video.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/video_depth")
def get_video_depth(episode_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_depth.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/preview_video")
def get_preview_video(episode_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = _preview_video_path(td)
    if not path:
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/video_webcam_0")
def get_video_webcam0(episode_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_webcam_0.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/video_webcam_1")
def get_video_webcam1(episode_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_webcam_1.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/video_camera/<cam_id>")
def get_video_camera(episode_id: str, cam_id: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    cam = camera_by_id(cam_id)
    if not cam:
        return abort(404)
    path = os.path.join(td, video_file_name(cam, "color"))
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/video_camera/<cam_id>/<stream>")
def get_video_camera_stream(episode_id: str, cam_id: str, stream: str):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    cam = camera_by_id(cam_id)
    if not cam or stream not in cam.get("streams", ["color"]):
        return abort(404)
    path = os.path.join(td, video_file_name(cam, stream))
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video")
def get_take_video(episode_id: str, take: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    path = os.path.join(td, "video.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video_depth")
def get_take_video_depth(episode_id: str, take: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_depth.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/preview_video")
def get_take_preview_video(episode_id: str, take: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    path = _preview_video_path(td)
    if not path:
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video_webcam_0")
def get_take_video_webcam0(episode_id: str, take: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_webcam_0.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video_webcam_1")
def get_take_video_webcam1(episode_id: str, take: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    path = os.path.join(td, "video_webcam_1.mp4")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video_camera/<cam_id>")
def get_take_video_camera(episode_id: str, take: str, cam_id: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    cam = camera_by_id(cam_id)
    if not cam:
        return abort(404)
    path = os.path.join(td, video_file_name(cam, "color"))
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.get("/<episode_id>/takes/<take>/video_camera/<cam_id>/<stream>")
def get_take_video_camera_stream(episode_id: str, take: str, cam_id: str, stream: str):
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    cam = camera_by_id(cam_id)
    if not cam or stream not in cam.get("streams", ["color"]):
        return abort(404)
    path = os.path.join(td, video_file_name(cam, stream))
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="video/mp4")


@bp.post("/<episode_id>/takes/<take>/regenerate_videos")
def regenerate_take_videos(episode_id: str, take: str):
    """이미 완료된 take의 카메라 영상을 다시 생성 (예: 옛날 편광 카메라 녹화에
    AoLP/DoLP/2x2 grid 뷰를 새로 채워넣을 때 — 호환 재구성)."""
    td = _take_dir(episode_id, take)
    if not td:
        return abort(404)
    result = _mgr().regenerate_camera_videos(episode_id, take)
    return jsonify(result), (200 if result.get("ok") else 400)


@bp.get("/<episode_id>/frame/<int:frame_idx>")
def get_frame(episode_id: str, frame_idx: int):
    td = _take_dir(episode_id)
    if not td:
        return abort(404)
    path = os.path.join(td, "frames", f"color_{frame_idx:06d}.png")
    if not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype="image/png")


@bp.get("/<episode_id>/takes/<take>/joints")
def get_take_joints(episode_id: str, take: str):
    from config import DATASET_PATH
    td = os.path.join(DATASET_PATH, episode_id, "takes", take)
    if not os.path.isdir(td):
        return abort(404)

    def read_joint_csv(fname: str) -> list[dict]:
        path = os.path.join(td, fname)
        if not os.path.exists(path):
            return []
        samples = []
        t_offset = None
        with open(path) as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                try:
                    t_ns = int(float(row["t_host_ns"]))
                    if t_offset is None:
                        t_offset = t_ns
                    samples.append({
                        "t": (t_ns - t_offset) / 1e9,
                        "q": [float(row[f"q{i}"]) for i in range(1, 7)],
                        "gripper": float(row.get("gripper", 0)),
                    })
                except (KeyError, ValueError):
                    continue
        return samples

    return jsonify({
        "teach": read_joint_csv("teach_joint.csv"),
        "executed": read_joint_csv("executed_joint.csv"),
        "command": read_joint_csv("replay_command.csv"),
    })


@bp.post("/export")
def export_episodes():
    from config import DATASET_PATH
    import io, zipfile

    data = request.get_json(silent=True) or {}
    episode_ids: list[str] = data.get("episode_ids", [])
    include_frames: bool = bool(data.get("include_frames", False))

    if not episode_ids:
        return abort(400)

    from datetime import datetime, timezone

    readme = _build_readme(episode_ids, include_frames)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.writestr("README.txt", readme)
        for ep_id in episode_ids:
            ep_dir = os.path.join(DATASET_PATH, ep_id)
            if not os.path.isdir(ep_dir):
                continue
            for root, dirs, files in os.walk(ep_dir):
                if not include_frames:
                    # Skip frame image directories (frames/, frames_webcam_0/, etc.)
                    dirs[:] = [d for d in dirs if not d.startswith("frames")]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, os.path.dirname(ep_dir))
                    try:
                        zf.write(fpath, arcname)
                    except OSError:
                        pass

    buf.seek(0)
    name = f"piper_dataset_{len(episode_ids)}ep.zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=name)


@bp.post("/export_lerobot")
def export_lerobot():
    """선택 에피소드를 LeRobot v2.0 형식 ZIP으로 변환·제공."""
    import io, zipfile, tempfile
    from converters.lerobot_converter import convert_episodes
    from config import DATASET_PATH, POSTPROCESS_DEFAULT_MASK_FILL_COLOR

    data = request.get_json(silent=True) or {}
    episode_ids: list[str] = data.get("episode_ids", [])
    if not episode_ids:
        return abort(400)
    if data.get("success_only"):
        episode_ids = [ep_id for ep_id in episode_ids if _episode_success(ep_id) is True]
    if data.get("issue_policy") == "exclude_open":
        episode_ids = [ep_id for ep_id in episode_ids if not _has_open_review_issues(ep_id)]
    if not episode_ids:
        return abort(400)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "lerobot")
        preset = data.get("preset", data.get("export_preset", "default"))
        mask_mode = data.get("mask_mode", "edit")
        mask_fill_color = data.get("mask_fill_color", POSTPROCESS_DEFAULT_MASK_FILL_COLOR)
        preserve_source_indices = bool(data.get("preserve_source_indices", False))
        include_default_excluded = bool(data.get("include_default_excluded", False))
        if preserve_source_indices and not include_default_excluded:
            return jsonify({
                "ok": False,
                "error": "preserve_source_indices requires include_default_excluded",
            }), 400
        convert_episodes(
            episode_ids,
            DATASET_PATH,
            out_dir,
            export_preset=preset,
            mask_mode=mask_mode,
            mask_fill_color=mask_fill_color,
            preserve_source_indices=preserve_source_indices,
            include_default_excluded=include_default_excluded,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, tmp)
                    try:
                        zf.write(fpath, arcname)
                    except OSError:
                        pass
        buf.seek(0)
        name = f"piper_lerobot_{len(episode_ids)}ep.zip"
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=name)


def _build_readme(episode_ids: list[str], include_frames: bool) -> str:
    from datetime import datetime, timezone
    from config import DATASET_PATH

    ep_lines = []
    for ep_id in episode_ids:
        meta_path = os.path.join(DATASET_PATH, ep_id, "meta.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            task = meta.get("task", {}).get("instruction") or meta.get("task") or "—"
            if isinstance(task, dict):
                task = task.get("instruction", "—")
            success = meta.get("success")
            label = "success" if success is True else "failure" if success is False else "unlabeled"
            takes = len(meta.get("takes", []))
            ep_lines.append(f"  {ep_id}  [{label}]  takes={takes}  task={task}")
        except (OSError, json.JSONDecodeError):
            ep_lines.append(f"  {ep_id}")

    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    frame_note = (
        "Included: mp4 video, depth video, CSV joint logs, webcam video, raw PNG frames"
        if include_frames else
        "Included: mp4 video, depth video, CSV joint logs, webcam video  (raw frames excluded)"
    )

    return f"""\
Piper Robot Arm — Demonstration Dataset
========================================
Exported : {exported_at}
Episodes : {len(episode_ids)}
{frame_note}

EPISODES
--------
{chr(10).join(ep_lines)}

DIRECTORY STRUCTURE
-------------------
<episode_id>/
  meta.json                  Episode metadata (task, success label, timestamps)
  takes/
    take_001/
      teach_joint.csv        Joint angles recorded during kinesthetic teaching
      executed_joint.csv     Joint angles executed during replay
      replay_command.csv     Joint commands sent to the robot during replay
      video.mp4              RGB video from primary depth camera (color stream)
      video_depth.mp4        Depth-colorised video from primary camera
      video_webcam_0.mp4     RGB video from external webcam 0  (if available)
      video_webcam_1.mp4     RGB video from external webcam 1  (if available)
      frames/                Raw PNG frames — color_XXXXXX.png  (if included)
      frames_webcam_0/       Raw PNG frames from webcam 0       (if included)
    take_002/ ...

CSV FORMAT  (teach_joint.csv / executed_joint.csv / replay_command.csv)
-----------------------------------------------------------------------
Column      Type     Description
----------  -------  ---------------------------------------------------
t_host_ns   int64    Host-side timestamp, nanoseconds since Unix epoch
q1..q6      float    Joint angles in radians (joint 1 = base, 6 = wrist)
gripper     float    Gripper opening in metres (0 = closed, ~0.07 = open)

ROBOT
-----
Platform : AgileX Piper 6-DOF collaborative arm
Kinematics: DH parameters (dh_is_offset=0x01)
  Link  a (mm)    alpha (rad)  d (mm)
   1      0.00    0.000        123.00
   2      0.00   -π/2            0.00
   3    285.03    0.000           0.00
   4    -21.98   +π/2          250.75
   5      0.00   -π/2            0.00
   6      0.00   +π/2           91.00
"""


def _get_teach_duration(take_dir: str) -> float | None:
    """teach_joint.csv의 첫/마지막 t_host_ns 차이(초) 반환."""
    path = os.path.join(take_dir, "teach_joint.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv_module.DictReader(f))
    if len(rows) < 2:
        return None
    t0 = int(float(rows[0]["t_host_ns"]))
    t1 = int(float(rows[-1]["t_host_ns"]))
    dur = (t1 - t0) / 1e9
    return dur if dur > 0 else None


def _get_video_duration(video_path: str) -> float | None:
    """OpenCV metadata로 mp4 duration(초) 반환."""
    if not os.path.exists(video_path):
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps and fps > 0 and frames and frames > 0:
        return float(frames / fps)
    return None


@bp.get("/<episode_id>/takes/<take>/frame_webcam1_at")
def get_take_frame_webcam1_at(episode_id: str, take: str):
    """
    t_sec에 가장 가까운 webcam_1 프레임 반환.

    ref=video (기본): t_sec는 비디오 플레이어 시각.
        video duration으로 정규화한 비율로 카메라 프레임을 찾음.
        replay 비디오가 speed-adjust되어 있어도 올바른 프레임을 반환.
    ref=camera: 카메라 실시간 절대 시각 기준 (직접 timestamp 매핑).
    """
    from config import DATASET_PATH
    t_sec = float(request.args.get("t", 0))
    ref = request.args.get("ref", "video")  # "video" | "camera"

    cam_id = default_mask_camera_id()
    if not cam_id:
        return abort(404)
    return _send_camera_frame_at(episode_id, take, cam_id, t_sec, ref)


@bp.get("/<episode_id>/takes/<take>/frame_camera_at/<cam_id>")
def get_take_frame_camera_at(episode_id: str, take: str, cam_id: str):
    t_sec = float(request.args.get("t", 0))
    ref = request.args.get("ref", "video")
    return _send_camera_frame_at(episode_id, take, cam_id, t_sec, ref)


def _send_camera_frame_at(episode_id: str, take: str, cam_id: str, t_sec: float, ref: str):
    from config import DATASET_PATH
    td = os.path.join(DATASET_PATH, episode_id, "takes", take)
    cam = camera_by_id(cam_id)
    if not cam:
        return abort(404)
    frames_dir = os.path.join(td, frames_dir_name(cam))
    if not os.path.isdir(frames_dir):
        return abort(404)

    cam_csv_path = os.path.join(td, camera_csv_name(cam))
    frame_idx = 0
    if os.path.exists(cam_csv_path):
        with open(cam_csv_path) as f:
            rows = list(csv_module.DictReader(f))
        if rows:
            if ref == "video":
                # 비디오 시간을 카메라 프레임 비율로 변환.
                video_duration = _get_video_duration(os.path.join(td, video_file_name(cam, "color")))
                if video_duration and video_duration > 0:
                    frac = max(0.0, min(1.0, t_sec / video_duration))
                    row_idx = int(round(frac * (len(rows) - 1)))
                else:
                    row_idx = 0
                frame_idx = int(rows[row_idx]["frame_idx"])
            else:
                # ref == "camera": 카메라 실시간 절대 시각 기준
                cam_t0_ns = int(float(rows[0]["t_host_ns"]))
                t_target_ns = cam_t0_ns + int(t_sec * 1e9)
                best = min(rows, key=lambda r: abs(int(float(r["t_host_ns"])) - t_target_ns))
                frame_idx = int(best["frame_idx"])

    for ext in ("jpg", "png"):
        path = os.path.join(frames_dir, f"color_{frame_idx:06d}.{ext}")
        if os.path.exists(path):
            return send_file(path, mimetype=f"image/{'jpeg' if ext == 'jpg' else ext}")
    return abort(404)


@bp.get("/<episode_id>/takes/<take>/edit_meta")
def get_edit_meta(episode_id: str, take: str):
    from config import DATASET_PATH
    td = os.path.join(DATASET_PATH, episode_id, "takes", take)
    if not os.path.isdir(td):
        return abort(404)
    path = os.path.join(td, "edit_meta.json")
    if not os.path.exists(path):
        return jsonify({"trim": {"enabled": False, "cut_t": None, "margin": 1.0},
                        "mask": {"enabled": False, "camera_id": default_mask_camera_id() or "", "polygon": [], "fill": "black"},
                        "timing": {"mode": "teaching_normalized"}})
    with open(path) as f:
        meta = json.load(f)
    meta.setdefault("timing", {"mode": "teaching_normalized"})
    return jsonify(meta)


@bp.put("/<episode_id>/takes/<take>/edit_meta")
def put_edit_meta(episode_id: str, take: str):
    from config import DATASET_PATH
    td = os.path.join(DATASET_PATH, episode_id, "takes", take)
    if not os.path.isdir(td):
        return abort(404)
    data = request.get_json(silent=True)
    if data is None:
        return abort(400)
    timing_mode = data.get("timing", {}).get("mode", "teaching_normalized")
    if timing_mode not in ("teaching_normalized", "physical_replay"):
        return jsonify({"ok": False, "error": "invalid timing mode"}), 400
    data.setdefault("timing", {})["mode"] = timing_mode
    path = os.path.join(td, "edit_meta.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"ok": True})


@bp.get("/<episode_id>/takes/<take>/trajectory")
def get_take_trajectory(episode_id: str, take: str):
    from config import DATASET_PATH
    from kinematics.fk import compute_trajectory

    td = os.path.join(DATASET_PATH, episode_id, "takes", take)
    if not os.path.isdir(td):
        return abort(404)

    cache_path = os.path.join(td, "trajectory_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return jsonify(json.load(f))

    def read_csv(fname: str) -> list[dict]:
        path = os.path.join(td, fname)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return list(csv_module.DictReader(f))

    result = {
        "teach":    compute_trajectory(read_csv("teach_joint.csv")),
        "executed": compute_trajectory(read_csv("executed_joint.csv")),
        "command":  compute_trajectory(read_csv("replay_command.csv")),
    }

    with open(cache_path, "w") as f:
        json.dump(result, f)

    return jsonify(result)
