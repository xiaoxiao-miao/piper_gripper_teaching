"""
스크린샷 API
  POST   /api/screenshots               body: {"cam_id","mode","quality"} → 현재 프레임 저장
  GET    /api/screenshots               → 저장된 스크린샷 목록
  GET    /api/screenshots/<name>        → 이미지 파일
  DELETE /api/screenshots/<name>        → 삭제
  POST   /api/screenshots/download_raw12  body: {"filenames":[...]} → Stokes(S0/S1/S2×RGB=9채널) raw + AoLP/DoLP zip 다운로드
"""

import io
import json
import os
import time
import zipfile
from datetime import datetime

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, jsonify, request, send_file

from config import DATASET_PATH

bp = Blueprint("screenshots", __name__, url_prefix="/api/screenshots")

SCREENSHOTS_DIR = os.path.join(DATASET_PATH, "screenshots")


def _ctrl():
    return current_app.config["CONTROLLER"]


def _ensure_dir():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def _safe_path(filename: str) -> str | None:
    root = os.path.realpath(SCREENSHOTS_DIR)
    target = os.path.realpath(os.path.join(SCREENSHOTS_DIR, filename))
    if os.path.commonpath([root, target]) != root:
        return None
    return target


@bp.post("")
def take_screenshot():
    data = request.get_json(silent=True) or {}
    cam_id = str(data.get("cam_id", "primary"))
    mode = str(data.get("mode", "rgb"))
    try:
        quality = max(1, min(100, int(data.get("quality", 90))))
    except (TypeError, ValueError):
        quality = 90

    mgr = _ctrl()._cameras
    if mgr is None:
        return jsonify({"error": "camera manager not ready"}), 503
    jpg = mgr.get_jpeg_bytes(cam_id=cam_id, quality=quality, mode=mode)
    if jpg is None:
        return jsonify({"error": f"no frame available for camera '{cam_id}'"}), 404

    _ensure_dir()
    ts = time.time()
    stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_cam = "".join(c for c in cam_id if c.isalnum() or c in "-_") or "cam"
    safe_mode = "".join(c for c in mode if c.isalnum() or c in "-_") or "rgb"
    filename = f"{stamp}_{safe_cam}_{safe_mode}.jpg"
    path = os.path.join(SCREENSHOTS_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpg)

    meta = {"filename": filename, "cam_id": cam_id, "mode": mode, "ts": ts}

    # 편광 카메라면 Stokes/AoLP/DoLP raw + 4가지 컬러 모드 JPEG을 모두 같은
    # 스탬프로 한 묶음으로 저장 — 갤러리 뷰어에서 찍은 모드와 상관없이 다 비교 가능하게.
    raw_bundle = mgr.get_raw_bundle_pngs(cam_id)
    if raw_bundle:
        color_files = {mode: filename}
        for vm in ("rgb", "aolp", "dolp", "grid", "stokes"):
            if vm in color_files:
                continue
            vm_jpg = mgr.get_jpeg_bytes(cam_id=cam_id, quality=quality, mode=vm)
            if not vm_jpg:
                continue
            vm_fname = f"{stamp}_{safe_cam}_{vm}.jpg"
            with open(os.path.join(SCREENSHOTS_DIR, vm_fname), "wb") as f:
                f.write(vm_jpg)
            color_files[vm] = vm_fname
        meta["color_files"] = color_files

        raw_files = {}
        for key, png_bytes in raw_bundle.items():
            raw_fname = f"{stamp}_{safe_cam}_{key}.png"
            with open(os.path.join(SCREENSHOTS_DIR, raw_fname), "wb") as f:
                f.write(png_bytes)
            raw_files[key] = raw_fname
        meta["raw_files"] = raw_files

    with open(path + ".json", "w") as f:
        json.dump(meta, f)

    return jsonify({**meta, "url": f"/api/screenshots/{filename}"})


@bp.get("")
def list_screenshots():
    _ensure_dir()
    items = []
    for fname in os.listdir(SCREENSHOTS_DIR):
        if not fname.endswith(".jpg"):
            continue
        meta_path = os.path.join(SCREENSHOTS_DIR, fname + ".json")
        if not os.path.exists(meta_path):
            # color_files의 companion 모드 JPEG(자기 메타데이터 없음) — 대표 항목에서만 노출.
            continue
        meta = {"filename": fname, "cam_id": "", "mode": "", "ts": 0}
        try:
            with open(meta_path) as f:
                meta.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
        meta["url"] = f"/api/screenshots/{fname}"
        items.append(meta)
    items.sort(key=lambda m: m.get("ts", 0), reverse=True)
    return jsonify(items)


@bp.get("/<path:filename>")
def get_screenshot(filename: str):
    _ensure_dir()
    path = _safe_path(filename)
    if path is None:
        return abort(400)
    if not os.path.isfile(path):
        return abort(404)
    mimetype = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    return send_file(path, mimetype=mimetype)


@bp.delete("/<path:filename>")
def delete_screenshot(filename: str):
    path = _safe_path(filename)
    if path is None:
        return abort(400)

    meta_path = path + ".json"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            for fname in {**meta.get("color_files", {}), **meta.get("raw_files", {})}.values():
                companion = _safe_path(fname)
                if companion and os.path.isfile(companion):
                    os.remove(companion)
        except (OSError, json.JSONDecodeError):
            pass
        os.remove(meta_path)

    if os.path.isfile(path):
        os.remove(path)
    return jsonify({"ok": True})


@bp.post("/download_raw12")
def download_raw12():
    """
    선택한 스크린샷들의 Stokes 큐브(S0/S1/S2×RGB=9채널) + AoLP/DoLP/Stokes raw를
    zip으로 묶어서 반환. 저장된 raw 모자이크(mosaic.png)로부터 그 자리에서 재구성하므로
    촬영 당시 카메라가 켜져있지 않아도, 시간이 지난 뒤에도 동작함.
    body: {"filenames": ["<screenshot .jpg 파일명>", ...]}
    """
    from nodes.spinnaker_node import extract_12ch_cube_from_mosaic

    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames")
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"error": "filenames (non-empty list) required"}), 400

    buf = io.BytesIO()
    added = 0
    skipped = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            if not isinstance(fname, str):
                continue
            path = _safe_path(fname)
            meta_path = path + ".json" if path else None
            if path is None or meta_path is None or not os.path.isfile(meta_path):
                skipped.append(fname)
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped.append(fname)
                continue

            raw_files = meta.get("raw_files", {})
            stamp = os.path.splitext(fname)[0]
            wrote_any = False

            mosaic_fname = raw_files.get("mosaic")
            mosaic_path = _safe_path(mosaic_fname) if mosaic_fname else None
            if mosaic_path and os.path.isfile(mosaic_path):
                mosaic = cv2.imread(mosaic_path, cv2.IMREAD_UNCHANGED)
                result = extract_12ch_cube_from_mosaic(mosaic) if mosaic is not None else None
                if result is not None:
                    cube, labels = result
                    npz_buf = io.BytesIO()
                    np.savez_compressed(npz_buf, cube=cube, channel_labels=np.array(labels))
                    zf.writestr(f"{stamp}_stokes9ch.npz", npz_buf.getvalue())
                    wrote_any = True

            for key in ("aolp", "dolp"):
                sub_fname = raw_files.get(key)
                sub_path = _safe_path(sub_fname) if sub_fname else None
                if sub_path and os.path.isfile(sub_path):
                    zf.write(sub_path, arcname=f"{stamp}_{key}.png")
                    wrote_any = True

            if wrote_any:
                added += 1
            else:
                skipped.append(fname)

    if added == 0:
        return jsonify({
            "error": "no downloadable raw data found (편광 카메라 스크린샷만 지원)",
            "skipped": skipped,
        }), 404

    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip", as_attachment=True,
        download_name="polarization_raw_bundle.zip",
    )
