"""
CameraManager
--------------
manifest에 정의된 RealSense / USB 웹캠을 통합 관리.

핵심 설계:
  - 카메라 3대 중 1대라도 연결되면 시스템 동작
  - 각 카메라 연결 실패는 graceful — 나머지 카메라는 정상 동작
  - 동일한 인터페이스: get_status(), get_jpeg_bytes(cam_id), start/stop_recording()
  - stream_ids: manifest camera id + legacy alias

사용:
    mgr = CameraManager()
    ok = mgr.connect()          # 1대라도 연결되면 True
    mgr.start_all_recording(ep_dir)
    buffers = mgr.stop_all_recording()  # dict: cam_id → [frame_dict, ...]
    mgr.disconnect()
"""

import threading
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nodes.realsense_node import RealsenseCapture
from nodes.webcam_node import WebcamCapture, discover_webcams
from nodes.spinnaker_node import SpinnakerCapture
from camera_manifest import camera_aliases, legacy_id_for, load_manifest, resolve_camera_id


class CameraManager:
    """
    모든 카메라의 라이프사이클을 관리.
    """

    def __init__(self, manifest: Optional[dict] = None):
        self.manifest = manifest or load_manifest()
        self.realsense: Optional[RealsenseCapture] = None
        self.webcams: list[WebcamCapture] = []
        self.spinnakers: list[SpinnakerCapture] = []
        self._camera_entries: dict[str, dict] = {}
        self._captures: dict[str, object] = {}
        self._aliases: dict[str, str] = {}
        self._connected = False
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        모든 카메라 연결 시도. 1대라도 성공하면 True 반환.
        실패한 카메라는 조용히 skip.
        """
        any_ok = False

        auto_cache: dict[str | None, list[str]] = {}

        for entry in self.manifest.get("cameras", []):
            cam_id = entry["id"]
            self._camera_entries[cam_id] = entry
            for alias in camera_aliases(entry):
                self._aliases[alias] = cam_id
            if not entry.get("enabled", True):
                continue

            if entry.get("type") == "realsense":
                rs = RealsenseCapture()
                ok = rs.start_stream()
                if ok:
                    self.realsense = rs
                    self._captures[cam_id] = rs
                    any_ok = True
                    print(f"[CameraManager] {cam_id}: RealSense connected")
                else:
                    rs.stop_stream()
                    print(f"[CameraManager] {cam_id}: RealSense not available")
                continue

            if entry.get("type") == "opencv":
                dev_path = self._resolve_opencv_device(entry, auto_cache)
                if not dev_path:
                    print(f"[CameraManager] {cam_id}: no OpenCV device resolved")
                    continue
                try:
                    target_fps = max(1, int(entry.get("fps", 30)))
                except (TypeError, ValueError):
                    target_fps = 30
                wc = WebcamCapture(
                    device_path=dev_path,
                    name=legacy_id_for(entry),
                    target_fps=target_fps,
                )
                ok = wc.start_stream()
                if ok:
                    self.webcams.append(wc)
                    self._captures[cam_id] = wc
                    any_ok = True
                    print(f"[CameraManager] {cam_id}: webcam connected ({dev_path})")
                else:
                    print(f"[CameraManager] {cam_id} ({dev_path}): failed, skipping")
                continue

            if entry.get("type") == "spinnaker":
                serial = self._resolve_spinnaker_serial(entry)
                try:
                    target_fps = max(1, int(entry.get("fps", 15)))
                except (TypeError, ValueError):
                    target_fps = 15
                sc = SpinnakerCapture(
                    serial=serial,
                    name=legacy_id_for(entry),
                    target_fps=target_fps,
                )
                ok = sc.start_stream()
                if ok:
                    self.spinnakers.append(sc)
                    self._captures[cam_id] = sc
                    any_ok = True
                    print(f"[CameraManager] {cam_id}: Spinnaker camera connected (serial={serial or 'auto'})")
                else:
                    print(f"[CameraManager] {cam_id}: Spinnaker camera not available")

        self._connected = any_ok
        return any_ok

    @staticmethod
    def _resolve_spinnaker_serial(entry: dict) -> Optional[str]:
        device = str(entry.get("device", "auto"))
        if device.startswith("serial:"):
            return device[len("serial:"):]
        return None

    def _resolve_opencv_device(self, entry: dict, auto_cache: dict) -> Optional[str]:
        device = str(entry.get("device", "auto"))
        if device.startswith("/dev/") and "*" in device:
            import glob
            # PCI-prefix glob: e.g. /dev/v4l/by-path/pci-0000:65:00.3-usb-*-video-index0
            matches = sorted(glob.glob(device))
            if not matches:
                print(f"[CameraManager] glob '{device}' matched nothing")
                return None
            resolved = matches[0]
            print(f"[CameraManager] glob '{device}' → {resolved}")
            return resolved
        if device.startswith("/dev/"):
            return device
        if not device.startswith("auto"):
            return device

        parts = device.split(":")
        name_filter = parts[1] if len(parts) >= 2 and parts[1] else None
        index = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
        if name_filter not in auto_cache:
            auto_cache[name_filter] = discover_webcams(name_filter=name_filter, max_cameras=max(8, index + 1))
        devices = auto_cache[name_filter]
        return devices[index] if index < len(devices) else None

    def disconnect(self):
        if self.realsense:
            self.realsense.stop_stream()
            self.realsense = None
        for wc in self.webcams:
            wc.stop_stream()
        self.webcams = []
        for sc in self.spinnakers:
            sc.stop_stream()
        self.spinnakers = []
        self._captures = {}
        self._connected = False

    # ─────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────

    def start_all_recording(self, episode_dir: str):
        """모든 연결된 카메라 동시 녹화 시작."""
        if self.realsense:
            self.realsense.start_recording(episode_dir)
        for wc in self.webcams:
            wc.start_recording(episode_dir)
        for sc in self.spinnakers:
            sc.start_recording(episode_dir)

    def stop_all_recording(self) -> dict:
        """
        모든 카메라 녹화 종료.
        Returns: {'realsense': [...], 'webcam_0': [...], ...}
        각 값은 camera_frames buffer (list[dict]).
        """
        buffers: dict = {}
        for cam_id, capture in self._captures.items():
            entry = self._camera_entries.get(cam_id, {"id": cam_id})
            key = legacy_id_for(entry)
            buffers[key] = capture.stop_recording()
        return buffers

    # ─────────────────────────────────────────────────────────────────
    # Live MJPEG (per camera)
    # ─────────────────────────────────────────────────────────────────

    def get_jpeg_bytes(self, cam_id: str = "primary", quality: int = 70, mode: str = "rgb") -> Optional[bytes]:
        """
        cam_id: 'primary' | manifest id | legacy alias | '<camera_id>_depth'
        'primary': 첫 번째 available color camera
        mode: SpinnakerCapture 전용 view mode ('rgb'|'aolp'|'dolp'|'grid'|'stokes') — 다른 캡처 타입은 무시.
        """
        if cam_id == "primary":
            for entry in self.manifest.get("cameras", []):
                capture = self._captures.get(entry["id"])
                if capture and "color" in entry.get("streams", []):
                    return self._get_jpeg_from_capture(capture, quality, mode)
            return None

        if cam_id.endswith("_depth"):
            base_id = cam_id[:-6]
            resolved = self._aliases.get(base_id) or resolve_camera_id(base_id, self.manifest)
            capture = self._captures.get(resolved) if resolved else None
            if capture and hasattr(capture, "get_depth_jpeg_bytes"):
                return capture.get_depth_jpeg_bytes(quality)
            return None

        resolved = self._aliases.get(cam_id) or resolve_camera_id(cam_id, self.manifest)
        capture = self._captures.get(resolved) if resolved else None
        return self._get_jpeg_from_capture(capture, quality, mode) if capture else None

    @staticmethod
    def _get_jpeg_from_capture(capture, quality: int, mode: str) -> Optional[bytes]:
        if isinstance(capture, SpinnakerCapture):
            return capture.get_jpeg_bytes(quality, mode)
        return capture.get_jpeg_bytes(quality)

    # ─────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # Controls (exposure / gain 등 — 지원하는 캡처 타입만)
    # ─────────────────────────────────────────────────────────────────

    def get_camera_controls(self, cam_id: str) -> dict:
        resolved = self._aliases.get(cam_id) or resolve_camera_id(cam_id, self.manifest)
        capture = self._captures.get(resolved) if resolved else None
        if capture and hasattr(capture, "get_controls"):
            return capture.get_controls()
        return {}

    def set_camera_controls(self, cam_id: str, **kwargs) -> dict:
        resolved = self._aliases.get(cam_id) or resolve_camera_id(cam_id, self.manifest)
        capture = self._captures.get(resolved) if resolved else None
        if capture and hasattr(capture, "set_controls"):
            return capture.set_controls(**kwargs)
        return {"error": f"camera '{cam_id}' not found or does not support controls"}

    def get_raw_bundle_pngs(self, cam_id: str) -> Optional[dict]:
        """편광 카메라 전용: Stokes/AoLP/DoLP raw PNG 묶음. 지원 안 하면 None."""
        resolved = self._aliases.get(cam_id) or resolve_camera_id(cam_id, self.manifest)
        capture = self._captures.get(resolved) if resolved else None
        if capture and hasattr(capture, "get_raw_bundle_pngs"):
            return capture.get_raw_bundle_pngs()
        return None

    def get_status(self) -> dict:
        """전체 카메라 상태 dict. 프론트엔드 camera_state 이벤트용."""
        cameras: dict = {}

        for entry in self.manifest.get("cameras", []):
            cam_id = entry["id"]
            capture = self._captures.get(cam_id)
            if capture:
                info = capture.get_status()
            else:
                info = {"available": False, "name": cam_id, "device": entry.get("device", "—")}
            info.update({
                "id": cam_id,
                "label": entry.get("label", cam_id),
                "role": entry.get("role", ""),
                "type": entry.get("type", ""),
                "legacy_id": entry.get("legacy_id", cam_id),
                "streams": entry.get("streams", ["color"]),
                "enabled": entry.get("enabled", True),
                "aliases": sorted(camera_aliases(entry)),
            })
            cameras[cam_id] = info

        # 편의: 하나라도 연결되어 있으면 connected=True
        any_connected = any(v.get("available", False) for v in cameras.values())
        return {
            "connected": any_connected,
            "cameras": cameras,
        }

    @property
    def any_available(self) -> bool:
        """녹화/스트리밍 가능한 카메라가 1대 이상인지."""
        if self.realsense and self.realsense._available:
            return True
        if any(wc._available for wc in self.webcams):
            return True
        return any(sc._available for sc in self.spinnakers)

    # ─────────────────────────────────────────────────────────────────
    # Backward-compat shim (controller가 단일 RealsenseCapture 처럼 쓰는 곳)
    # ─────────────────────────────────────────────────────────────────

    def start_stream(self):
        """connect()의 별칭 — 기존 코드 호환."""
        return self.connect()

    def start_recording(self, episode_dir: str):
        self.start_all_recording(episode_dir)

    def stop_recording(self) -> dict:
        return self.stop_all_recording()
