"""
WebcamCapture
--------------
USB 웹캠 (Logitech C270 등) 캡처 모듈.
RealsenseCapture와 동일한 인터페이스 — CameraManager에서 교환 가능하게 사용.

특징:
  - OpenCV VideoCapture 기반
  - 스트림 스레드 + 비동기 writer 스레드 (RealsenseCapture와 동일 패턴)
  - 컬러 PNG 저장 (depth 없음)
  - 자동 해상도 협상 (640×480 → 960×720 순으로 시도)
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
try:
    from config import COLOR_SAVE_JPEG_QUALITY
except ImportError:
    COLOR_SAVE_JPEG_QUALITY = 85

WRITER_QUEUE_MAX = 150   # ~5초 @30fps
TARGET_RESOLUTIONS = [(480, 360), (640, 480)]  # 우선순위 순


@dataclass
class WebcamFrame:
    rgb: np.ndarray       # (H, W, 3) BGR
    t_host_ns: int        # 호스트 타임스탬프


class WebcamCapture:
    """
    단일 USB 웹캠 캡처.

    Args:
        device_path: '/dev/video4' 등 V4L2 device path
        name: 식별용 이름 ('webcam_0', 'webcam_1' 등)
        target_fps: 목표 FPS (드라이버가 최대한 맞춤)
    """

    def __init__(self, device_path: str, name: str = "webcam", target_fps: int = 30):
        self.device_path = device_path
        self.name = name
        self._target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._streaming = False
        self._available = False

        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()

        # live frame (MJPEG 스트리밍용)
        self._latest_frame: Optional[WebcamFrame] = None
        self._frame_lock = threading.Lock()
        self._jpeg_lock = threading.Lock()
        self._jpeg_cache_t_ns = 0
        self._jpeg_cache_quality = -1
        self._jpeg_cache: Optional[bytes] = None
        self._live_fps = 0.0
        self._live_count = 0
        self._live_hz_t = time.time()

        # 녹화
        self._recording = False
        self._frames_dir: Optional[str] = None
        self._write_queue: queue.Queue = queue.Queue(maxsize=WRITER_QUEUE_MAX)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop = threading.Event()

        # 통계
        self._captured_count = 0
        self._written_count = 0
        self._dropped_count = 0
        self._camera_frames_buffer: list[dict] = []

        # 해상도 (start_stream 후 확정)
        self._width = 640
        self._height = 480

    # ─────────────────────────────────────────────────────────────────
    # Stream lifecycle
    # ─────────────────────────────────────────────────────────────────

    def start_stream(self) -> bool:
        """스트림 시작. 성공하면 True, 실패하면 False (graceful)."""
        if self._streaming:
            return True

        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[{self.name}] Cannot open {self.device_path}")
            return False

        # MJPEG으로 전환 — USB 허브 대역폭 절약 (YUYV 대비 10~20배 압축)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        # 해상도 협상
        opened_res = None
        for w, h in TARGET_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w == w and actual_h == h:
                opened_res = (w, h)
                break

        if opened_res is None:
            # 드라이버가 어느 해상도도 지원 안 함 → 현재 해상도 그대로 사용
            opened_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        # USB 재연결/전원 복귀 직후 첫 프레임이 비는 장치가 있으므로 짧게 warm-up.
        # 한 번의 read 실패만으로 연결된 카메라를 영구 skip하지 않는다.
        ret = False
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.05)
        if not ret:
            print(f"[{self.name}] Test frame read failed after warm-up — device unusable")
            cap.release()
            return False

        self._cap = cap
        self._width, self._height = opened_res
        self._available = True
        self._streaming = True
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name=f"{self.name}_stream"
        )
        self._stream_thread.start()
        print(f"[{self.name}] Started: {self._width}×{self._height} @ {self._target_fps}fps  ({self.device_path})")
        return True

    def stop_stream(self):
        self._stream_stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._streaming = False
        self._available = False

    def _stream_loop(self):
        while not self._stream_stop.is_set():
            if self._cap is None:
                break
            ret, frame_bgr = self._cap.read()
            if not ret:
                print(f"[{self.name}] Frame read failed — stopping stream")
                break

            t_ns = time.time_ns()
            wf = WebcamFrame(rgb=frame_bgr, t_host_ns=t_ns)
            self._process_frame(wf)

        self._streaming = False
        self._available = False

    def _process_frame(self, frame: WebcamFrame):
        with self._frame_lock:
            self._latest_frame = frame

        # FPS 계산
        self._live_count += 1
        now = time.time()
        elapsed = now - self._live_hz_t
        if elapsed >= 1.0:
            self._live_fps = self._live_count / elapsed
            self._live_count = 0
            self._live_hz_t = now

        if self._recording:
            self._captured_count += 1
            try:
                self._write_queue.put_nowait((self._captured_count - 1, frame))
            except queue.Full:
                self._dropped_count += 1

    # ─────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────

    def start_recording(self, episode_dir: str):
        frames_dir = os.path.join(episode_dir, f"frames_{self.name}")
        os.makedirs(frames_dir, exist_ok=True)
        self._frames_dir = frames_dir

        # 큐 비우기
        while not self._write_queue.empty():
            try:
                self._write_queue.get_nowait()
            except queue.Empty:
                break

        self._captured_count = 0
        self._written_count = 0
        self._dropped_count = 0
        self._camera_frames_buffer = []

        self._writer_stop.clear()
        self._recording = True

        self._writer_thread = threading.Thread(
            target=self._write_loop, daemon=True, name=f"{self.name}_writer"
        )
        self._writer_thread.start()
        print(f"[{self.name}] Recording started → {frames_dir}")

    def stop_recording(self) -> list[dict]:
        self._recording = False
        self._writer_stop.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=15.0)
        print(f"[{self.name}] Recording stopped: {self._written_count} frames written, "
              f"{self._dropped_count} dropped")
        return self._camera_frames_buffer.copy()

    def _write_loop(self):
        while not self._writer_stop.is_set() or not self._write_queue.empty():
            try:
                frame_idx, frame = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            color_fname = f"color_{frame_idx:06d}.jpg"
            color_path = os.path.join(self._frames_dir, color_fname)
            cv2.imwrite(color_path, frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])

            self._camera_frames_buffer.append({
                "frame_idx": frame_idx,
                "t_host_ns": frame.t_host_ns,
                "color_path": f"frames_{self.name}/{color_fname}",
            })
            self._written_count += 1

    # ─────────────────────────────────────────────────────────────────
    # Live MJPEG
    # ─────────────────────────────────────────────────────────────────

    def get_jpeg_bytes(self, quality: int = 70) -> Optional[bytes]:
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        # Multiple MJPEG clients often request the same 15fps source frame at 30fps.
        # Encode it once instead of competing with the replay scheduler for every client.
        with self._jpeg_lock:
            if (
                self._jpeg_cache is not None
                and self._jpeg_cache_t_ns == frame.t_host_ns
                and self._jpeg_cache_quality == quality
            ):
                return self._jpeg_cache
            ok, buf = cv2.imencode(
                ".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not ok:
                return None
            self._jpeg_cache = bytes(buf)
            self._jpeg_cache_t_ns = frame.t_host_ns
            self._jpeg_cache_quality = quality
            return self._jpeg_cache

    # ─────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "name":            self.name,
            "device":          self.device_path,
            "available":       self._available,
            "streaming":       self._streaming,
            "recording":       self._recording,
            "fps":             round(self._live_fps, 1),
            "resolution":      f"{self._width}×{self._height}",
            "captured_frames": self._captured_count,
            "written_frames":  self._written_count,
            "dropped_frames":  self._dropped_count,
            "queue_len":       self._write_queue.qsize(),
        }


# ─────────────────────────────────────────────────────────────────────
# Device auto-discovery
# ─────────────────────────────────────────────────────────────────────

def discover_webcams(
    name_filter: Optional[str] = None,
    max_cameras: int = 2,
) -> list[str]:
    """
    v4l2-ctl --list-devices 결과를 파싱해 웹캠 device path 목록 반환.

    Args:
        name_filter: None이면 모든 카메라, 문자열이면 그 이름 포함된 카메라만
                     (예: "C270" → Logitech C270만)
        max_cameras: 최대 반환 수

    Returns:
        ['/dev/video4', ...] — 각 물리 카메라의 primary video node (index0)
    """
    import subprocess
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout
    except Exception as e:
        print(f"[discover_webcams] v4l2-ctl failed: {e}")
        return []

    devices: list[str] = []
    current_name: Optional[str] = None
    added_current_device = False  # 현재 device block에서 이미 추가했는지

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            # 빈 줄 = device block 구분자
            current_name = None
            added_current_device = False
            continue

        if stripped.startswith("/dev/"):
            if current_name is None:
                continue
            if name_filter and name_filter.lower() not in current_name.lower():
                continue
            # 각 물리 카메라의 첫 번째 /dev/videoN 만 사용
            # (같은 device block의 두 번째 노드 /dev/video11 등은 skip)
            if stripped.startswith("/dev/video") and not added_current_device:
                devices.append(stripped)
                added_current_device = True
                if len(devices) >= max_cameras:
                    return devices
        else:
            # 장치 이름 줄 (예: "C270 HD WEBCAM (usb-...):")
            current_name = stripped.rstrip(":")
            added_current_device = False  # 새 device block 시작

    return devices
