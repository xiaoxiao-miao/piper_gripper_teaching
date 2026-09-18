import copy
import json
import os
from typing import Any


ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(ROOT, "cameras.json")

DEFAULT_MANIFEST: dict[str, Any] = {
    "version": 1,
    "cameras": [
        {
            "id": "polarizer",
            "label": "FLIR Polarization Cam",
            "type": "spinnaker",
            "role": "polarization",
            "enabled": True,
            "device": "auto",
            "fps": 15,
            "streams": ["color", "aolp", "dolp", "grid", "stokes"],
            "legacy_id": "polarizer",
            "aliases": [],
            "export_presets": ["default", "all", "debug"],
        },
        {
            "id": "cam0",
            "label": "Cam0 Ego",
            "type": "opencv",
            "role": "ego",
            "enabled": True,
            "device": "auto:C270:0",
            "fps": 15,
            "streams": ["color"],
            "legacy_id": "webcam_0",
            "aliases": ["webcam_0"],
            "export_presets": ["default", "all", "debug"],
        },
        {
            "id": "cam1",
            "label": "Cam1 Overview",
            "type": "opencv",
            "role": "overview",
            "enabled": True,
            "device": "auto:C270:1",
            "fps": 15,
            "streams": ["color"],
            "legacy_id": "webcam_1",
            "aliases": ["webcam_1"],
            "export_presets": ["default", "all", "debug"],
        },
        {
            "id": "realsense",
            "label": "RealSense Depth Sensor",
            "type": "realsense",
            "role": "depth_sensor",
            "enabled": True,
            "device": "auto",
            "streams": ["color", "depth"],
            "legacy_id": "realsense",
            "aliases": ["realsense_depth"],
            "export_presets": ["all", "debug"],
        },
    ],
}

IMMUTABLE_FIELDS = {"id", "role", "type", "legacy_id"}
EDITABLE_FIELDS = {"label", "enabled", "device", "export_presets"}


def _normalize_camera(cam: dict[str, Any]) -> dict[str, Any]:
    out = dict(cam)
    out["id"] = str(out.get("id", "")).strip()
    out["label"] = str(out.get("label") or out["id"])
    out["type"] = str(out.get("type", "opencv")).strip()
    out["role"] = str(out.get("role", out["id"])).strip()
    out["enabled"] = bool(out.get("enabled", True))
    out["device"] = str(out.get("device", "auto"))
    try:
        out["fps"] = max(1, int(out.get("fps", 30)))
    except (TypeError, ValueError):
        out["fps"] = 30
    streams = out.get("streams", ["color"])
    out["streams"] = [str(s) for s in streams] if isinstance(streams, list) else ["color"]
    legacy_id = out.get("legacy_id")
    out["legacy_id"] = str(legacy_id) if legacy_id else out["id"]
    aliases = out.get("aliases", [])
    out["aliases"] = [str(a) for a in aliases] if isinstance(aliases, list) else []
    if "depth" in out["streams"]:
        depth_alias = f"{out['id']}_depth"
        if depth_alias not in out["aliases"]:
            out["aliases"].append(depth_alias)
    presets = out.get("export_presets", [])
    out["export_presets"] = [str(p) for p in presets] if isinstance(presets, list) else []
    return out


def normalize_manifest(data: dict[str, Any] | None) -> dict[str, Any]:
    manifest = copy.deepcopy(DEFAULT_MANIFEST)
    if isinstance(data, dict):
        manifest.update({k: v for k, v in data.items() if k != "cameras"})
        if isinstance(data.get("cameras"), list):
            manifest["cameras"] = data["cameras"]
    manifest["version"] = int(manifest.get("version", 1))
    seen: set[str] = set()
    cameras: list[dict[str, Any]] = []
    for raw in manifest.get("cameras", []):
        if not isinstance(raw, dict):
            continue
        cam = _normalize_camera(raw)
        if not cam["id"] or cam["id"] in seen:
            continue
        seen.add(cam["id"])
        cameras.append(cam)
    manifest["cameras"] = cameras
    return manifest


def load_manifest(path: str = MANIFEST_PATH) -> dict[str, Any]:
    if not os.path.exists(path):
        save_manifest(DEFAULT_MANIFEST, path)
        return copy.deepcopy(DEFAULT_MANIFEST)
    try:
        with open(path) as f:
            return normalize_manifest(json.load(f))
    except (OSError, json.JSONDecodeError, ValueError):
        return copy.deepcopy(DEFAULT_MANIFEST)


def save_manifest(manifest: dict[str, Any], path: str = MANIFEST_PATH) -> dict[str, Any]:
    normalized = normalize_manifest(manifest)
    with open(path, "w") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def update_manifest_edits(edits: dict[str, Any], path: str = MANIFEST_PATH) -> dict[str, Any]:
    current = load_manifest(path)
    by_id = {cam["id"]: cam for cam in current["cameras"]}
    for raw in edits.get("cameras", []) if isinstance(edits, dict) else []:
        if not isinstance(raw, dict):
            continue
        cam_id = raw.get("id")
        if cam_id not in by_id:
            continue
        target = by_id[cam_id]
        for key in EDITABLE_FIELDS:
            if key in raw:
                target[key] = raw[key]
    return save_manifest(current, path)


def camera_aliases(cam: dict[str, Any]) -> set[str]:
    aliases = {cam["id"]}
    legacy_id = cam.get("legacy_id")
    if legacy_id:
        aliases.add(str(legacy_id))
    aliases.update(str(alias) for alias in cam.get("aliases", []) if alias)
    return aliases


def resolve_camera_id(cam_id: str, manifest: dict[str, Any] | None = None) -> str | None:
    manifest = manifest or load_manifest()
    for cam in manifest["cameras"]:
        if cam_id in camera_aliases(cam):
            return cam["id"]
    return None


def camera_by_id(cam_id: str, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    manifest = manifest or load_manifest()
    resolved = resolve_camera_id(cam_id, manifest)
    if not resolved:
        return None
    return next((cam for cam in manifest["cameras"] if cam["id"] == resolved), None)


def legacy_id_for(cam: dict[str, Any]) -> str:
    return str(cam.get("legacy_id") or cam["id"])


def frames_dir_name(cam: dict[str, Any]) -> str:
    legacy = legacy_id_for(cam)
    if cam["type"] == "realsense" or legacy == "realsense":
        return "frames"
    return f"frames_{legacy}"


def camera_csv_name(cam: dict[str, Any]) -> str:
    legacy = legacy_id_for(cam)
    return f"camera_frames_{legacy}.csv"


def video_file_name(cam: dict[str, Any], stream: str = "color") -> str:
    legacy = legacy_id_for(cam)
    if cam["type"] == "realsense" or legacy == "realsense":
        return "video_depth.mp4" if stream == "depth" else "video.mp4"
    if stream == "color":
        return f"video_{legacy}.mp4"
    return f"video_{legacy}_{stream}.mp4"


def export_cameras(preset: str = "default", manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    manifest = manifest or load_manifest()
    return [
        cam for cam in manifest["cameras"]
        if cam.get("enabled", True)
        and "color" in cam.get("streams", [])
        and preset in cam.get("export_presets", [])
    ]


def overview_camera(manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    manifest = manifest or load_manifest()
    for cam in manifest["cameras"]:
        if cam.get("role") == "overview":
            return cam
    return next((cam for cam in manifest["cameras"] if "color" in cam.get("streams", [])), None)


def default_mask_camera(manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return overview_camera(manifest)


def default_mask_camera_id(manifest: dict[str, Any] | None = None) -> str | None:
    cam = default_mask_camera(manifest)
    return cam.get("id") if cam else None
