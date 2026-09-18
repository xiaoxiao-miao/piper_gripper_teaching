"""
SpinnakerCapture
-----------------
FLIR/Teledyne 편광(polarization) 카메라 (Blackfly S BFS-U3-51S5PC 등) 캡처 모듈.
WebcamCapture / RealsenseCapture와 동일한 인터페이스 — CameraManager에서 교환 가능하게 사용.

특징:
  - PySpin (Spinnaker SDK) 기반, USB3 Vision
  - 센서는 편광 모자이크(BayerRGPolarized8: 2x2 Bayer × 2x2 편광각) 원본을 낸다.
    프레임마다 4가지 뷰를 미리 계산해 캐시:
      rgb   : ImageUtilityPolarization.CreateStokesS0() — 무편광 총광량 컬러
      aolp  : CreateAolp() → 360° 레인보우(hue) 컬러맵
      dolp  : CreateDolp() → red-black 컬러맵
      grid  : 위 rgb/aolp/dolp + (H=AoLP,S=DoLP,V=S0) 조합 뷰를 2x2로 합성
    AoLP는 라디안 [0,pi) 주기, DoLP는 [0,1] 비율로 가정 — 실카메라에서 실측 검증 필요.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from config import COLOR_SAVE_JPEG_QUALITY
except ImportError:
    COLOR_SAVE_JPEG_QUALITY = 85

try:
    import PySpin
    _SPIN_AVAILABLE = True
except ImportError:
    PySpin = None
    _SPIN_AVAILABLE = False

WRITER_QUEUE_MAX = 150      # ~5초 @30fps
GET_NEXT_IMAGE_TIMEOUT_MS = 1000
# 2448x2048 BayerRGPolarized8 기준 실측: exposure=1000us일 때 AcquisitionFrameRate.GetMax()=75.79fps
# → 프레임 주기 - 노출시간 ≈ 12194us(readout 등 고정 오버헤드). UI 힌트용 추정치일 뿐,
# 실제 상한은 ExposureTime.SetValue()의 성공/실패로 판단해야 함.
FRAME_OVERHEAD_US_HINT = 12200
# ROI 축소 비율 (width/height 각각). 0.5 = 전체 해상도의 1/4 픽셀 수 → 처리량 감소로 15fps 확보 목적.
# (ROI 자체는 문제 없었음이 확인됨 — 실제 원인은 image_result.GetNDArray()를
# CreateStokesS0 호출 전에 먼저 불러서 생긴 부작용이었음. _stream_loop에서 순서 수정함.)
ROI_SCALE = 0.5
# DoLP 표시용 대비 게인. 실내 확산광 장면은 실제 DoLP가 낮아(5~20%) 그대로면 거의 안 보임.
DOLP_DISPLAY_GAIN = 3.0
# S0의 물리 최댓값(이 카메라 4탭 합의 이론적 상한, 8bit×4). S1/S2 역산·bwr 시각화의 고정
# 스케일로 공용 사용 — CreateStokesS1/S2 자체의 내장 정규화 범위는 SDK가 노출하지 않아
# 신뢰할 수 없다고 확인됐음(|S1|,|S2|<=S0 물리 부등식이 대부분 픽셀에서 깨졌었음).
STOKES_S0_MAX = 1020.0


@dataclass
class SpinnakerFrame:
    rgb: np.ndarray         # (H, W, 3) BGR8 — S0(무편광) 디모자이킹 결과 (표시/녹화용)
    aolp: np.ndarray        # (H, W, 3) BGR8 — AoLP, 360° 레인보우(hue) 컬러맵 (표시용)
    dolp: np.ndarray        # (H, W, 3) BGR8 — DoLP, red-black 컬러맵 (표시용)
    grid: np.ndarray        # (H, W, 3) BGR8 — RGB/AoLP/DoLP/Combined 2x2 합성 (표시용)
    stokes: np.ndarray      # (H, W, 3) BGR8 — S1/S2 × R/G/B 6타일(2행3열) bwr 합성 (표시용, 스케일링 없음)
    aolp_raw: np.ndarray     # (H, W) uint8, 0=0rad ~ 255=pi — 컬러라이즈 이전 raw 값 (저장용)
    dolp_raw: np.ndarray     # (H, W) uint8, 0=0 ~ 255=1.0 비율 — 컬러라이즈 이전 raw 값 (저장용)
    # 센서 원본 BayerRGPolarized8 모자이크 (표준 8bit라 GetNDArray가 안정적으로 읽힘).
    # S0/S1/S2/AoLP/DoLP 등 정밀 재구성이 필요하면 저장해둔 이걸로 나중에 오프라인
    # 재처리한다 — RGB16s/RGB32f를 매 라이브 프레임 GetNDArray로 읽으려던 시도는
    # (a) 그 포맷 자체를 GetNDArray가 못 읽어서 -1003 에러가 나고 (b) 프레임당 PySpin
    # 호출이 너무 늘어나 fps가 1 미만으로 떨어져서 라이브 루프에서 뺐다.
    raw_mosaic: np.ndarray   # (H, W) uint8
    t_host_ns: int


VIEW_MODES = ("rgb", "aolp", "dolp", "grid", "stokes")


def _colorize_aolp(aolp_u8: np.ndarray) -> np.ndarray:
    """정규화된 AoLP(uint8 0~255, ABSOLUTE_DATA_RANGE로 [0,pi)를 매핑) → 360° 레인보우 hue BGR."""
    val = aolp_u8.mean(axis=2) if aolp_u8.ndim == 3 else aolp_u8
    hue = (val.astype(np.float32) / 255.0 * 179.0).astype(np.uint8)
    sat = np.full(hue.shape, 255, dtype=np.uint8)
    value = np.full(hue.shape, 255, dtype=np.uint8)
    return cv2.cvtColor(cv2.merge([hue, sat, value]), cv2.COLOR_HSV2BGR)


def _colorize_dolp(dolp_u8: np.ndarray) -> np.ndarray:
    """정규화된 DoLP(uint8 0~255, ABSOLUTE_DATA_RANGE로 [0,1]을 매핑) → red-black BGR (0=검정, 255=순수 빨강).
    실내 확산광 장면은 실제 DoLP가 보통 5~20%라 그대로 매핑하면 거의 안 보여서 표시용 게인을 곱함."""
    val = dolp_u8.mean(axis=2) if dolp_u8.ndim == 3 else dolp_u8
    r = np.clip(val.astype(np.float32) * DOLP_DISPLAY_GAIN, 0, 255).astype(np.uint8)
    zeros = np.zeros_like(r)
    return cv2.merge([zeros, zeros, r])  # BGR: R채널만 사용


def _combined_hsv(aolp_u8: np.ndarray, dolp_u8: np.ndarray, s0_gray: np.ndarray) -> np.ndarray:
    """H=AoLP, S=DoLP, V=S0 밝기 — 편광각+편광도+무편광광량을 한 장에."""
    aolp_v = aolp_u8.mean(axis=2) if aolp_u8.ndim == 3 else aolp_u8
    dolp_v = dolp_u8.mean(axis=2) if dolp_u8.ndim == 3 else dolp_u8
    hue = (aolp_v.astype(np.float32) / 255.0 * 179.0).astype(np.uint8)
    sat = np.clip(dolp_v.astype(np.float32) * DOLP_DISPLAY_GAIN, 0, 255).astype(np.uint8)
    val = s0_gray.astype(np.uint8)
    return cv2.cvtColor(cv2.merge([hue, sat, val]), cv2.COLOR_HSV2BGR)


def _make_grid(rgb: np.ndarray, aolp_bgr: np.ndarray, dolp_bgr: np.ndarray, combined_bgr: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    tile_w, tile_h = w // 2, h // 2
    resize = lambda im: cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    top = np.hstack([resize(rgb), resize(aolp_bgr)])
    bottom = np.hstack([resize(dolp_bgr), resize(combined_bgr)])
    grid = np.vstack([top, bottom])
    for text, (x, y) in [
        ("RGB", (10, 25)), ("AoLP", (tile_w + 10, 25)),
        ("DoLP", (10, tile_h + 25)), ("Combined", (tile_w + 10, tile_h + 25)),
    ]:
        cv2.putText(grid, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return grid


def _derive_s1_s2(frame_bgr: np.ndarray, aolp_arr: np.ndarray, dolp_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """검증된 S0(frame_bgr)·AoLP·DoLP로부터 S1,S2를 물리 단위로 역산 — (H,W,3), RGB 채널 순.
    CreateStokesS1/S2를 직접 디코딩하지 않는 이유: 그 내장 정규화 범위를 SDK가 노출하지
    않아 신뢰 불가로 확인됨(|S1|,|S2|<=S0 물리 부등식이 대부분 픽셀에서 깨졌었음). 대신 이미
    ABSOLUTE_DATA_RANGE로 검증된 AoLP(0~pi)/DoLP(0~1)로부터 S1=S0·DoLP·cos(2·AoLP),
    S2=S0·DoLP·sin(2·AoLP)로 구하면 그 부등식이 항상 자동으로 성립한다."""
    s0 = frame_bgr[:, :, ::-1].astype(np.float32) * (STOKES_S0_MAX / 255.0)  # BGR8 표시값 → RGB 물리값(0~1020)
    aolp = aolp_arr.astype(np.float32) * (np.pi / 255.0)
    dolp = dolp_arr.astype(np.float32) / 255.0
    s1 = s0 * dolp * np.cos(2.0 * aolp)
    s2 = s0 * dolp * np.sin(2.0 * aolp)
    return s1, s2


def _colorize_bwr(signed: np.ndarray, vmax: float = STOKES_S0_MAX) -> np.ndarray:
    """부호 있는 값(H,W) → blue-white-red 다이버징 컬러맵 BGR8 (matplotlib 'bwr'과 동일한 보간).
    채널/프레임마다 자기 자신의 min~max로 다시 정규화하지 않고 고정된 vmax(대칭)를 그대로
    쓴다 — 실제 크기가 작으면 흰색에 가깝게 나오는 게 맞다(과장해서 스케일링하지 않음)."""
    t = np.clip(signed.astype(np.float32) / vmax, -1.0, 1.0)
    r = np.where(t >= 0, 1.0, t + 1.0)
    g = np.where(t >= 0, 1.0 - t, t + 1.0)
    b = np.where(t >= 0, 1.0 - t, 1.0)
    bgr = np.clip(np.stack([b, g, r], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    return bgr


def _make_stokes_grid(s1: np.ndarray, s2: np.ndarray, vmax: float = STOKES_S0_MAX) -> np.ndarray:
    """S1/S2 (H,W,3, RGB 채널 순, 물리 단위) → R/G/B 각각 bwr로 시각화한 2행(S1,S2)×3열(R,G,B) 그리드.
    6타일 전부 동일한 고정 vmax를 써서 채널/타일 간 크기 비교가 왜곡되지 않게 한다."""
    h, w = s1.shape[:2]
    tile_w, tile_h = w // 2, h // 2
    resize = lambda im: cv2.resize(im, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    row1 = np.hstack([resize(_colorize_bwr(s1[:, :, i], vmax)) for i in range(3)])
    row2 = np.hstack([resize(_colorize_bwr(s2[:, :, i], vmax)) for i in range(3)])
    grid = np.vstack([row1, row2])
    for i, label in enumerate(("S1_R", "S1_G", "S1_B", "S2_R", "S2_G", "S2_B")):
        x, y = (i % 3) * tile_w + 8, (i // 3) * tile_h + 22
        cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return grid


class SpinnakerCapture:
    """
    단일 FLIR 편광 카메라 캡처.

    Args:
        serial: 카메라 시리얼 넘버 (None이면 연결된 첫 카메라 사용)
        name: 식별용 이름 ('polarizer' 등)
        target_fps: 목표 FPS
    """

    def __init__(self, serial: Optional[str] = None, name: str = "polarizer", target_fps: int = 15):
        self.serial = serial
        self.name = name
        self._target_fps = target_fps

        self._system = None
        self._cam_list = None
        self._cam = None
        self._cam_lock = threading.Lock()   # ExposureTime/Gain 등 노드 읽기/쓰기 보호
        self._image_processor = None
        self._streaming = False
        self._available = False

        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()

        # live frame (MJPEG 스트리밍용)
        self._latest_frame: Optional[SpinnakerFrame] = None
        self._frame_lock = threading.Lock()
        self._jpeg_lock = threading.Lock()
        self._jpeg_cache: dict[tuple[str, int], tuple[int, bytes]] = {}  # (mode,quality) -> (t_host_ns, jpeg)
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

        self._width = 0
        self._height = 0
        self._quad_diag_logged = False

    # ─────────────────────────────────────────────────────────────────
    # Stream lifecycle
    # ─────────────────────────────────────────────────────────────────

    def start_stream(self) -> bool:
        """스트림 시작. 성공하면 True, 실패하면 False (graceful)."""
        if not _SPIN_AVAILABLE:
            print(f"[{self.name}] PySpin not installed — skipping")
            return False
        if self._streaming:
            return True

        try:
            system = PySpin.System.GetInstance()
            cam_list = system.GetCameras()

            cam = None
            for i in range(cam_list.GetSize()):
                candidate = cam_list.GetByIndex(i)
                try:
                    cand_serial = candidate.TLDevice.DeviceSerialNumber.GetValue()
                except PySpin.SpinnakerException:
                    del candidate
                    continue
                if self.serial is None or cand_serial == self.serial:
                    cam = candidate
                    break
                del candidate

            if cam is None:
                print(f"[{self.name}] No matching Spinnaker camera found (serial={self.serial})")
                cam_list.Clear()
                system.ReleaseInstance()
                return False

            cam.Init()
            self._image_processor = PySpin.ImageProcessor()

            # 편광 원본 모자이크 (2x2 Bayer × 2x2 편광각)
            cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRGPolarized8)

            # ROI로 해상도를 줄여 15fps를 안정적으로 확보 (binning은 쓰지 않음 —
            # binning은 서로 다른 편광각 픽셀을 섞어버려 편광 데이터를 망가뜨림).
            # 4x4 매크로픽셀(2x2 Bayer × 2x2 편광각) 정렬 유지를 위해 GetInc() 배수로 스냅, 중앙 크롭.
            try:
                # OffsetX/Y를 먼저 0으로 되돌려야 Width/Height.GetMax()가 "진짜" 센서 전체
                # 해상도를 돌려준다 — 이전 세션에서 offset이 남아있으면 GetMax()가 그만큼
                # 줄어든 값을 보고해 재시작할 때마다 해상도가 계속 작아지는 버그가 생김.
                cam.OffsetX.SetValue(0)
                cam.OffsetY.SetValue(0)
                full_w, full_h = cam.Width.GetMax(), cam.Height.GetMax()
                inc_w, inc_h = cam.Width.GetInc(), cam.Height.GetInc()
                target_w = max(inc_w, ((int(full_w * ROI_SCALE)) // inc_w) * inc_w)
                target_h = max(inc_h, ((int(full_h * ROI_SCALE)) // inc_h) * inc_h)
                off_x = ((full_w - target_w) // 2 // inc_w) * inc_w
                off_y = ((full_h - target_h) // 2 // inc_h) * inc_h
                cam.Width.SetValue(target_w)
                cam.Height.SetValue(target_h)
                cam.OffsetX.SetValue(off_x)
                cam.OffsetY.SetValue(off_y)
            except PySpin.SpinnakerException as e:
                print(f"[{self.name}] Could not set ROI (using full resolution): {e}")

            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

            try:
                cam.AcquisitionFrameRateEnable.SetValue(True)
                max_fps = cam.AcquisitionFrameRate.GetMax()
                cam.AcquisitionFrameRate.SetValue(min(float(self._target_fps), max_fps))
            except PySpin.SpinnakerException as e:
                print(f"[{self.name}] Could not set frame rate: {e}")

            # AcquisitionFrameRateEnable=True(고정 fps)면 이 카메라는 ExposureAuto 자체를
            # 못 쓰게 잠금 — "not writable" 에러가 나던 이유. 자동노출/고정프레임레이트는
            # 서로 배타적이라 Auto로 되돌릴 수 없음. 대신 매번 짧은 안전한 수동 노출값으로
            # 명시적으로 리셋 — 이전 세션에서 남은 긴 노출값(수백ms)이 그대로 이어져
            # fps가 1 미만까지 떨어지는 문제를 막음. 밝기는 API/슬라이더로 다시 조정 가능.
            SAFE_DEFAULT_EXPOSURE_US = 8000.0
            try:
                cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                lo = cam.ExposureTime.GetMin()
                hi = cam.ExposureTime.GetMax()
                cam.ExposureTime.SetValue(max(lo, min(SAFE_DEFAULT_EXPOSURE_US, hi)))
            except PySpin.SpinnakerException as e:
                print(f"[{self.name}] Could not reset exposure to safe default: {e}")
            try:
                cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                cam.Gain.SetValue(cam.Gain.GetMin())
            except PySpin.SpinnakerException as e:
                print(f"[{self.name}] Could not reset gain: {e}")

            self._width = cam.Width.GetValue()
            self._height = cam.Height.GetValue()

            cam.BeginAcquisition()

            # Warm-up — 재연결 직후 첫 프레임이 비는 경우가 있어 짧게 재시도.
            warm_ok = False
            for _ in range(10):
                try:
                    test_image = cam.GetNextImage(GET_NEXT_IMAGE_TIMEOUT_MS)
                    if not test_image.IsIncomplete():
                        warm_ok = True
                        test_image.Release()
                        break
                    test_image.Release()
                except PySpin.SpinnakerException:
                    pass
                time.sleep(0.05)

            if not warm_ok:
                print(f"[{self.name}] Test frame read failed after warm-up — device unusable")
                cam.EndAcquisition()
                cam.DeInit()
                del cam
                cam_list.Clear()
                system.ReleaseInstance()
                return False

            self._system = system
            self._cam_list = cam_list
            self._cam = cam
            self._quad_diag_logged = False
            self._available = True
            self._streaming = True
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_loop, daemon=True, name=f"{self.name}_stream"
            )
            self._stream_thread.start()
            print(f"[{self.name}] Started: {self._width}×{self._height} @ {self._target_fps}fps (serial={self.serial or 'auto'})")
            return True

        except PySpin.SpinnakerException as e:
            print(f"[{self.name}] Spinnaker init failed: {e}")
            return False

    def stop_stream(self):
        self._stream_stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=3.0)

        if self._cam is not None:
            try:
                self._cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            try:
                self._cam.DeInit()
            except PySpin.SpinnakerException:
                pass
            del self._cam
            self._cam = None
        self._image_processor = None
        if self._cam_list is not None:
            self._cam_list.Clear()
            self._cam_list = None
        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None

        self._streaming = False
        self._available = False

    def _stream_loop(self):
        consecutive_timeouts = 0
        while not self._stream_stop.is_set():
            try:
                image_result = self._cam.GetNextImage(GET_NEXT_IMAGE_TIMEOUT_MS)
                consecutive_timeouts = 0
            except PySpin.SpinnakerException as e:
                # GetNextImage()가 정상적으로 1초를 기다리지 않고 즉시 예외를 던지는 경우
                # (카메라/USB가 실제로 멈춘 상태) 여기서 sleep 없이 continue만 하면 로그도
                # 없이 CPU만 태우는 조용한 무한루프가 됨 — 연속 타임아웃을 감지해서 로그 +
                # 약간의 sleep으로 스로틀링.
                consecutive_timeouts += 1
                if consecutive_timeouts in (20, 100) or consecutive_timeouts % 500 == 0:
                    print(
                        f"[{self.name}] GetNextImage failing repeatedly "
                        f"({consecutive_timeouts}x) — camera/USB may be stuck: {e}"
                    )
                if consecutive_timeouts > 5:
                    time.sleep(0.1)
                continue

            try:
                if image_result.IsIncomplete():
                    image_result.Release()
                    continue

                # 원래부터 안정적으로 동작 확인된 경로: S0(RGB16s) → Normalize(RGB8) →
                # Convert(BGR8) → GetNDArray. GetNDArray는 8bit 포맷에서만 안정적으로
                # 동작하고 RGB16s/RGB32f를 직접 못 읽는다(반복 확인됨) — 그래서 항상
                # CreateNormalized로 8bit로 바꾼 뒤에만 GetNDArray를 부른다.
                s0_image = PySpin.ImageUtilityPolarization.CreateStokesS0(
                    PySpin.Image.Create(image_result), PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR
                )
                rgb8_image = PySpin.ImageUtility.CreateNormalized(s0_image, PySpin.PixelFormat_RGB8)
                bgr_image = self._image_processor.Convert(rgb8_image, PySpin.PixelFormat_BGR8)
                frame_bgr = np.array(bgr_image.GetNDArray(), copy=True)
            except PySpin.SpinnakerException as e:
                print(f"[{self.name}] Frame conversion failed: {e}")
                image_result.Release()
                continue

            # AoLP/DoLP — CreateAolp/CreateDolp는 BayerRGPolarized8을 공식 지원(안정적으로
            # 동작 확인됨). RGB32f로 나오는데 GetNDArray가 float 포맷을 직접 못 읽어서
            # CreateNormalized(ABSOLUTE_DATA_RANGE)로 물리적 고정 범위(AoLP 0~pi, DoLP 0~1)를
            # RGB8로 눌러서 읽는다 — 프레임마다 값이 흔들리는 동적 정규화가 아님.
            # (S1/S2를 받아 직접 계산하는 방식은 라이브 프레임마다 PySpin 호출이 너무 늘어나
            # fps가 1 미만으로 떨어지고, RGB16s를 GetNDArray로 못 읽는 문제까지 겹쳐 폐기함 —
            # raw Stokes/모자이크 저장은 아래에서 원본만 캡처해 acquisition 루프 밖에서 처리.)
            try:
                aolp_image = PySpin.ImageUtilityPolarization.CreateAolp(PySpin.Image.Create(image_result))
                dolp_image = PySpin.ImageUtilityPolarization.CreateDolp(PySpin.Image.Create(image_result))
                aolp_norm = PySpin.ImageUtility.CreateNormalized(
                    aolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
                )
                dolp_norm = PySpin.ImageUtility.CreateNormalized(
                    dolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
                )
                aolp_arr = np.array(aolp_norm.GetNDArray(), copy=True)  # uint8 0~255
                dolp_arr = np.array(dolp_norm.GetNDArray(), copy=True)  # uint8 0~255
                aolp_raw = aolp_arr.mean(axis=2).astype(np.uint8) if aolp_arr.ndim == 3 else aolp_arr
                dolp_raw = dolp_arr.mean(axis=2).astype(np.uint8) if dolp_arr.ndim == 3 else dolp_arr
            except PySpin.SpinnakerException as e:
                if not self._quad_diag_logged:  # 최초 1회만 로그 — 매 프레임 스팸 방지
                    print(f"[{self.name}] AoLP/DoLP computation failed: {e}")
                    self._quad_diag_logged = True
                aolp_raw = dolp_raw = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                aolp_arr = dolp_arr = None

            # raw_mosaic: 센서 원본 그대로. GetNDArray()는 BayerRGPolarized8 자체에 대해서도
            # "pixel format not supported" [-1003]로 실패함(실측 확인 — RGB16s/RGB32f와 같은
            # 문제가 8bit 원본 포맷에도 있음). 그동안 이 예외가 조용히 잡혀서 frame_bgr의 R
            # 채널(이미 디모자이킹되어 해상도가 절반으로 줄어든 값)로 대체되고 있었는데, 이걸
            # 진짜 raw mosaic인 것처럼 다시 CreateStokesS0/CreateAolp/CreateDolp에 먹이면
            # 실제 편광 모자이크가 아닌 걸 그런 척 재-디모자이킹하는 셈이라 그럴듯하지만 틀린
            # 색/해상도가 나옴 — 이게 "복원한 RGB가 원본과 미세하게 다르다"는 문제의 원인.
            # GetData()는 포맷과 무관하게 원시 바이트를 그대로 주므로 이걸로 우회한다.
            try:
                w_raw, h_raw = image_result.GetWidth(), image_result.GetHeight()
                raw_mosaic = np.frombuffer(image_result.GetData(), dtype=np.uint8, count=w_raw * h_raw).reshape(
                    h_raw, w_raw
                ).copy()
            except PySpin.SpinnakerException:
                raw_mosaic = frame_bgr[:, :, 0]  # 폴백 — raw 다운로드 정확도만 떨어짐

            image_result.Release()

            aolp_bgr = _colorize_aolp(aolp_raw)
            dolp_bgr = _colorize_dolp(dolp_raw)
            s0_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            combined_bgr = _combined_hsv(aolp_raw, dolp_raw, s0_gray)
            grid_bgr = _make_grid(frame_bgr, aolp_bgr, dolp_bgr, combined_bgr)

            # S1/S2 채널별(R/G/B) bwr 시각화 — 이미 계산해둔 S0(frame_bgr)/AoLP/DoLP를 그대로
            # 재사용(추가 PySpin 호출 없음). AoLP/DoLP 계산이 실패한 프레임은 빈 화면으로 대체.
            if aolp_arr is not None and dolp_arr is not None and aolp_arr.ndim == 3 and dolp_arr.ndim == 3:
                s1_phys, s2_phys = _derive_s1_s2(frame_bgr, aolp_arr, dolp_arr)
                stokes_bgr = _make_stokes_grid(s1_phys, s2_phys)
            else:
                stokes_bgr = np.zeros_like(grid_bgr)

            t_ns = time.time_ns()
            frame = SpinnakerFrame(
                rgb=frame_bgr, aolp=aolp_bgr, dolp=dolp_bgr, grid=grid_bgr, stokes=stokes_bgr,
                aolp_raw=aolp_raw, dolp_raw=dolp_raw, raw_mosaic=raw_mosaic, t_host_ns=t_ns,
            )
            self._process_frame(frame)

        self._streaming = False
        self._available = False

    def _process_frame(self, frame: SpinnakerFrame):
        with self._frame_lock:
            self._latest_frame = frame

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
        # 녹화 중엔 센서 원본 모자이크만 저장 (무손실 PNG, 유일한 진짜 raw 소스 —
        # 여기서 color/AoLP/DoLP/grid 전부 재구성 가능하므로 나머지를 매 프레임 같이
        # 쓰면 중복이고 15fps 쓰기 I/O만 늘어남). color/aolpview/dolpview/gridview는
        # postprocess 시점에 reconstruct_polarizer_frame_views()로 한 번에 재구성한다.
        while not self._writer_stop.is_set() or not self._write_queue.empty():
            try:
                frame_idx, frame = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            mosaic_fname = f"mosaic_{frame_idx:06d}.png"
            cv2.imwrite(os.path.join(self._frames_dir, mosaic_fname), frame.raw_mosaic)

            self._camera_frames_buffer.append({
                "frame_idx": frame_idx,
                "t_host_ns": frame.t_host_ns,
                "mosaic_path": f"frames_{self.name}/{mosaic_fname}",
            })
            self._written_count += 1

    # ─────────────────────────────────────────────────────────────────
    # Live MJPEG
    # ─────────────────────────────────────────────────────────────────

    def get_jpeg_bytes(self, quality: int = 70, mode: str = "rgb") -> Optional[bytes]:
        """mode: 'rgb' | 'aolp' | 'dolp' | 'grid' | 'stokes'"""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        img = getattr(frame, mode, None)
        if img is None:
            img = frame.rgb
            mode = "rgb"
        cache_key = (mode, quality)
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(cache_key)
            if cached is not None and cached[0] == frame.t_host_ns:
                return cached[1]
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            jpeg_bytes = bytes(buf)
            self._jpeg_cache[cache_key] = (frame.t_host_ns, jpeg_bytes)
            return jpeg_bytes

    def get_raw_bundle_pngs(self) -> Optional[dict[str, bytes]]:
        """
        스크린샷용 raw 묶음: 센서 원본 모자이크(S0/S1/S2 등 오프라인 재구성용) +
        AoLP/DoLP raw(컬러라이즈 이전 8bit). AoLP: 0=0rad~255=pi, DoLP: 0=0~255=1.0 비율.
        """
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        out = {}
        for key, img in (
            ("mosaic", frame.raw_mosaic), ("aolp", frame.aolp_raw), ("dolp", frame.dolp_raw),
        ):
            ok, buf = cv2.imencode(".png", img)
            if ok:
                out[key] = bytes(buf)
        return out or None

    # ─────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        status = {
            "name":            self.name,
            "device":          f"serial:{self.serial}" if self.serial else "auto",
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
        if self._cam is not None:
            with self._cam_lock:
                try:
                    status["pixel_format"] = self._cam.PixelFormat.GetCurrentEntry().GetSymbolic()
                    status["exposure_us"] = self._cam.ExposureTime.GetValue()
                    status["exposure_us_min"] = self._cam.ExposureTime.GetMin()
                    status["exposure_auto"] = self._cam.ExposureAuto.GetCurrentEntry().GetSymbolic()
                    status["exposure_us_max_hw"] = self._cam.ExposureTime.GetMax()
                    frame_period_us = 1_000_000.0 / max(1, self._target_fps)
                    status["exposure_us_max_hint_for_target_fps"] = round(
                        max(status["exposure_us_min"], frame_period_us - FRAME_OVERHEAD_US_HINT), 1
                    )
                except PySpin.SpinnakerException:
                    pass
        return status

    # ─────────────────────────────────────────────────────────────────
    # Controls (exposure / gain)
    # ─────────────────────────────────────────────────────────────────

    def get_controls(self) -> dict:
        """현재 노출/게인 값과 허용 범위. 카메라 미연결이면 빈 dict."""
        if self._cam is None:
            return {}
        with self._cam_lock:
            try:
                frame_period_us = 1_000_000.0 / max(1, self._target_fps)
                # 실측 기반 프레임당 고정 오버헤드(센서 readout 등) 추정치.
                # 실제 상한은 ExposureTime.SetValue()가 거부하는지로 판단하는 게 정확함 — 이건 UI 힌트용.
                exposure_hint_max = max(
                    self._cam.ExposureTime.GetMin(),
                    frame_period_us - FRAME_OVERHEAD_US_HINT,
                )
                return {
                    "pixel_format": self._cam.PixelFormat.GetCurrentEntry().GetSymbolic(),
                    "exposure_us": self._cam.ExposureTime.GetValue(),
                    "exposure_us_min": self._cam.ExposureTime.GetMin(),
                    "exposure_us_max_hw": self._cam.ExposureTime.GetMax(),
                    "exposure_us_max_hint_for_target_fps": round(exposure_hint_max, 1),
                    "exposure_auto": self._cam.ExposureAuto.GetCurrentEntry().GetSymbolic(),
                    "gain_db": self._cam.Gain.GetValue(),
                    "gain_db_min": self._cam.Gain.GetMin(),
                    "gain_db_max": self._cam.Gain.GetMax(),
                    "gain_auto": self._cam.GainAuto.GetCurrentEntry().GetSymbolic(),
                    "target_fps": self._target_fps,
                    "achieved_fps": round(self._live_fps, 2),
                }
            except PySpin.SpinnakerException as e:
                return {"error": str(e)}

    def set_controls(
        self,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        exposure_auto: Optional[bool] = None,
        gain_auto: Optional[bool] = None,
    ) -> dict:
        """
        노출 시간(µs)/게인(dB) 설정. None인 필드는 건드리지 않음.
        exposure_us/gain_db를 주면 해당 Auto는 자동으로 Off.
        하드웨어가 현재 프레임레이트와 충돌해 거부하면 SpinnakerException 메시지를 그대로 반환.
        """
        if self._cam is None:
            return {"error": "camera not connected"}
        with self._cam_lock:
            try:
                if exposure_auto is not None:
                    self._cam.ExposureAuto.SetValue(
                        PySpin.ExposureAuto_Continuous if exposure_auto else PySpin.ExposureAuto_Off
                    )
                if exposure_us is not None:
                    if self._cam.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
                        self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                    lo, hi = self._cam.ExposureTime.GetMin(), self._cam.ExposureTime.GetMax()
                    self._cam.ExposureTime.SetValue(max(lo, min(float(exposure_us), hi)))

                if gain_auto is not None:
                    self._cam.GainAuto.SetValue(
                        PySpin.GainAuto_Continuous if gain_auto else PySpin.GainAuto_Off
                    )
                if gain_db is not None:
                    if self._cam.GainAuto.GetValue() != PySpin.GainAuto_Off:
                        self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                    lo, hi = self._cam.Gain.GetMin(), self._cam.Gain.GetMax()
                    self._cam.Gain.SetValue(max(lo, min(float(gain_db), hi)))
            except PySpin.SpinnakerException as e:
                result = self.get_controls()
                result["error"] = str(e)
                return result
        return self.get_controls()


def reconstruct_polarizer_frame_views(frames_dir: str) -> int:
    """
    frames_dir 안의 편광 카메라 raw 파일들로부터 color/AoLP/DoLP/2x2 grid/Stokes(S1,S2 bwr)
    뷰(재생용 JPG)를 재구성해서 같은 디렉터리에 저장한다: color_*.jpg, aolpview_*.jpg,
    dolpview_*.jpg, gridview_*.jpg, stokesview_*.jpg. 이미 있는 파일은 건너뜀(파일 단위 idempotent).

    세대 호환 — 세션 중 저장 포맷이 여러 번 바뀌어서 두 경로를 다 지원한다:
      (a) mosaic_*.png만 있는 신형 녹화 → PySpin으로 처음부터 전부 재구성.
      (b) color_*.jpg + aolp_*.png/dolp_*.png(raw, 컬러라이즈 이전)만 있는 구형 녹화
          → mosaic 없이도 이미 있는 raw 값을 그대로 컬러라이즈만 하면 됨(PySpin 불필요, 빠름).
    stokesview(S1/S2 채널별 bwr)는 R/G/B 채널별 AoLP/DoLP가 필요한데 구형 녹화의
    aolp_*.png/dolp_*.png는 채널 평균만 저장돼 있어 복원 불가 — mosaic이 있는 프레임만 생성됨.
    라이브 카메라 연결 없이도 동작.

    Returns: 새로 만든(또는 채워넣은) 프레임 수.
    """
    if not os.path.isdir(frames_dir):
        return 0

    # mosaic_/color_/aolp_/dolp_ 중 뭐라도 있는 프레임 인덱스를 전부 수집.
    # (aolpview_/dolpview_/gridview_/stokesview_는 "aolp_"/"dolp_"로 시작하지 않으므로 안 겹침.)
    indices: set[str] = set()
    for fname in os.listdir(frames_dir):
        for prefix in ("mosaic_", "color_", "aolp_", "dolp_"):
            if fname.startswith(prefix):
                ext = os.path.splitext(fname)[1]
                indices.add(fname[len(prefix):-len(ext)])
                break
    if not indices:
        return 0

    image_processor = None  # PySpin 경로가 실제로 필요할 때만 생성
    processed = 0
    for idx_str in sorted(indices):
        color_path = os.path.join(frames_dir, f"color_{idx_str}.jpg")
        aolpview_path = os.path.join(frames_dir, f"aolpview_{idx_str}.jpg")
        dolpview_path = os.path.join(frames_dir, f"dolpview_{idx_str}.jpg")
        gridview_path = os.path.join(frames_dir, f"gridview_{idx_str}.jpg")
        stokesview_path = os.path.join(frames_dir, f"stokesview_{idx_str}.jpg")
        need_color = not os.path.exists(color_path)
        need_aolpview = not os.path.exists(aolpview_path)
        need_dolpview = not os.path.exists(dolpview_path)
        need_gridview = not os.path.exists(gridview_path)
        need_stokesview = not os.path.exists(stokesview_path)
        if not (need_color or need_aolpview or need_dolpview or need_gridview or need_stokesview):
            continue  # 이미 다 있음

        frame_bgr = cv2.imread(color_path) if os.path.exists(color_path) else None
        aolp_raw = cv2.imread(os.path.join(frames_dir, f"aolp_{idx_str}.png"), cv2.IMREAD_UNCHANGED)
        dolp_raw = cv2.imread(os.path.join(frames_dir, f"dolp_{idx_str}.png"), cv2.IMREAD_UNCHANGED)
        aolp_arr_full = dolp_arr_full = None  # (H,W,3) — stokesview 전용, mosaic 경로에서만 채워짐

        mosaic_path = os.path.join(frames_dir, f"mosaic_{idx_str}.png")
        need_mosaic = frame_bgr is None or aolp_raw is None or dolp_raw is None or need_stokesview
        if need_mosaic and os.path.exists(mosaic_path):
            if not _SPIN_AVAILABLE:
                continue
            if image_processor is None:
                image_processor = PySpin.ImageProcessor()
            mosaic = cv2.imread(mosaic_path, cv2.IMREAD_UNCHANGED)
            if mosaic is None:
                continue
            h, w = mosaic.shape[:2]
            mosaic = np.ascontiguousarray(mosaic, dtype=np.uint8)
            src_img = PySpin.Image.Create(w, h, 0, 0, PySpin.PixelFormat_BayerRGPolarized8, mosaic)

            if frame_bgr is None:
                try:
                    # RGB(S0) — 라이브 스트림과 동일한 검증된 경로.
                    s0_image = PySpin.ImageUtilityPolarization.CreateStokesS0(
                        PySpin.Image.Create(src_img), PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR
                    )
                    rgb8_image = PySpin.ImageUtility.CreateNormalized(s0_image, PySpin.PixelFormat_RGB8)
                    bgr_image = image_processor.Convert(rgb8_image, PySpin.PixelFormat_BGR8)
                    frame_bgr = np.array(bgr_image.GetNDArray(), copy=True)
                except PySpin.SpinnakerException as e:
                    print(f"[reconstruct_polarizer_frame_views] {idx_str}: RGB reconstruction failed: {e}")

            if aolp_raw is None or dolp_raw is None or need_stokesview:
                try:
                    # AoLP/DoLP — 라이브 스트림과 동일한 검증된 경로 (ABSOLUTE_DATA_RANGE 고정 범위).
                    # stokesview는 채널 평균(aolp_raw/dolp_raw)이 아니라 R/G/B별 원본(aolp_arr/dolp_arr)이
                    # 필요해서 이미 있던 값과 무관하게 여기서 다시 계산해둔다.
                    aolp_image = PySpin.ImageUtilityPolarization.CreateAolp(PySpin.Image.Create(src_img))
                    dolp_image = PySpin.ImageUtilityPolarization.CreateDolp(PySpin.Image.Create(src_img))
                    aolp_norm = PySpin.ImageUtility.CreateNormalized(
                        aolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
                    )
                    dolp_norm = PySpin.ImageUtility.CreateNormalized(
                        dolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
                    )
                    aolp_arr = np.array(aolp_norm.GetNDArray(), copy=True)
                    dolp_arr = np.array(dolp_norm.GetNDArray(), copy=True)
                    aolp_arr_full, dolp_arr_full = aolp_arr, dolp_arr
                    if aolp_raw is None:
                        aolp_raw = aolp_arr.mean(axis=2).astype(np.uint8) if aolp_arr.ndim == 3 else aolp_arr
                    if dolp_raw is None:
                        dolp_raw = dolp_arr.mean(axis=2).astype(np.uint8) if dolp_arr.ndim == 3 else dolp_arr
                except PySpin.SpinnakerException as e:
                    print(f"[reconstruct_polarizer_frame_views] {idx_str}: AoLP/DoLP failed: {e}")

        if frame_bgr is None or aolp_raw is None or dolp_raw is None:
            continue  # 소스 부족(mosaic도 raw도 없음) — 이 프레임은 재구성 불가

        if need_color:
            cv2.imwrite(color_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])
        aolp_bgr = _colorize_aolp(aolp_raw)
        dolp_bgr = _colorize_dolp(dolp_raw)
        if need_aolpview:
            cv2.imwrite(aolpview_path, aolp_bgr, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])
        if need_dolpview:
            cv2.imwrite(dolpview_path, dolp_bgr, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])
        if need_gridview:
            s0_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            combined_bgr = _combined_hsv(aolp_raw, dolp_raw, s0_gray)
            grid_bgr = _make_grid(frame_bgr, aolp_bgr, dolp_bgr, combined_bgr)
            cv2.imwrite(gridview_path, grid_bgr, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])
        if need_stokesview and aolp_arr_full is not None and dolp_arr_full is not None:
            s1_phys, s2_phys = _derive_s1_s2(frame_bgr, aolp_arr_full, dolp_arr_full)
            stokes_bgr = _make_stokes_grid(s1_phys, s2_phys)
            cv2.imwrite(stokesview_path, stokes_bgr, [cv2.IMWRITE_JPEG_QUALITY, COLOR_SAVE_JPEG_QUALITY])
        processed += 1

    return processed


def extract_12ch_cube_from_mosaic(mosaic: np.ndarray) -> Optional[tuple]:
    """
    저장된 raw BayerRGPolarized8 모자이크로부터 Stokes S0/S1/S2 × RGB = 9채널 큐브를 재구성.
    라이브 카메라 연결 없이도 동작하는 순수 이미지 유틸리티라 스크린샷 다운로드 시점에
    호출해도 됨 (촬영 당시 실행 불필요).

    원래 4개 편광각(I0/I45/I90/I135) × RGB = 12채널로 설계했으나, 그 추출에 쓰이는
    ExtractPolarQuadrant가 이 SDK에서 컬러 편광 소스(BayerRGPolarized8)를 지원하지 않는
    것으로 확인됨(colorProcessingAlg 파라미터가 없어 데모자이킹 자체를 못 함). 대신
    컬러 지원이 확인된 CreateStokesS0/S1/S2로 9채널을 재구성 — 선형 편광에 한해 정보량은
    동일함(I0/I45/I90/I135 ↔ S0/S1/S2는 선형 변환 관계).

    Returns: (cube, channel_labels) — cube: (H, W, 9) int16, 실패 시 None.
    """
    if not _SPIN_AVAILABLE:
        return None
    try:
        mosaic = np.ascontiguousarray(mosaic, dtype=np.uint8)
        h, w = mosaic.shape[:2]
        src_img = PySpin.Image.Create(w, h, 0, 0, PySpin.PixelFormat_BayerRGPolarized8, mosaic)

        # S0: CreateStokesS0 결과(RGB16s)는 GetNDArray가 직접 못 읽어서 CreateNormalized로
        # RGB8로 눌러 읽은 뒤 물리 스케일(0~1020, 이 카메라 4탭 합의 이론적 최댓값)로 되돌린다.
        s0_image = PySpin.ImageUtilityPolarization.CreateStokesS0(
            PySpin.Image.Create(src_img), PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR
        )
        s0_norm = PySpin.ImageUtility.CreateNormalized(s0_image, PySpin.PixelFormat_RGB8)
        s0 = np.array(s0_norm.GetNDArray(), copy=True).astype(np.float32) / 255.0 * STOKES_S0_MAX  # (H,W,3)

        # S1/S2: CreateStokesS1/S2(RGB16s)도 같은 방식(CreateNormalized 2-인자, 내장 고정
        # 범위 자동 적용)으로 읽을 수는 있으나, S0처럼 그 내장 범위가 정확히 몇인지 SDK가
        # 노출해주지 않는다. 예전엔 S0과 같은 ±1020으로 가정했는데, 실측해보니(라이브 카메라로
        # DoLP=sqrt(S1²+S2²)/S0을 역산해 SDK의 공식 CreateDolp 결과와 비교) 60~80배 이상
        # 어긋났고 — |S1|,|S2|<=S0이어야 하는 편광 물리 부등식이 거의 모든 픽셀(99.9%)에서
        # 깨졌다. 즉 그 가정이 틀렸었다. CreateStokesS1/S2를 직접 디코딩하는 대신, 이미
        # ABSOLUTE_DATA_RANGE로 검증된 CreateAolp/CreateDolp(공식 물리 범위: AoLP 0~π,
        # DoLP 0~1, 라이브 스트림에서 매 프레임 검증되어 사용 중)로부터
        # S1 = S0·DoLP·cos(2·AoLP), S2 = S0·DoLP·sin(2·AoLP) 로 역산한다.
        # 이러면 |S1|,|S2|<=S0가 항상 자동으로 성립하고, S0의 절대 스케일(1020)이
        # 정확한지 여부와 무관하게 S0/S1/S2가 서로 일관된 단위를 갖는다.
        aolp_image = PySpin.ImageUtilityPolarization.CreateAolp(PySpin.Image.Create(src_img))
        dolp_image = PySpin.ImageUtilityPolarization.CreateDolp(PySpin.Image.Create(src_img))
        aolp_norm = PySpin.ImageUtility.CreateNormalized(
            aolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
        )
        dolp_norm = PySpin.ImageUtility.CreateNormalized(
            dolp_image, PySpin.PixelFormat_RGB8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE
        )
        aolp = np.array(aolp_norm.GetNDArray(), copy=True).astype(np.float32) / 255.0 * np.pi  # (H,W,3) 0~pi
        dolp = np.array(dolp_norm.GetNDArray(), copy=True).astype(np.float32) / 255.0  # (H,W,3) 0~1

        s1 = s0 * dolp * np.cos(2.0 * aolp)
        s2 = s0 * dolp * np.sin(2.0 * aolp)

        planes = [
            np.clip(s0, -32768, 32767).astype(np.int16),
            np.clip(s1, -32768, 32767).astype(np.int16),
            np.clip(s2, -32768, 32767).astype(np.int16),
        ]
        labels = []
        for name in ("S0", "S1", "S2"):
            labels.extend([f"{name}_R", f"{name}_G", f"{name}_B"])
        cube = np.concatenate(planes, axis=2)  # (H,W,9) int16
        return cube, labels
    except PySpin.SpinnakerException as e:
        print(f"[extract_12ch_cube_from_mosaic] failed: {e}")
        return None


def discover_spinnaker_serials() -> list[str]:
    """연결된 모든 Spinnaker 카메라의 시리얼 넘버 목록."""
    if not _SPIN_AVAILABLE:
        return []
    serials: list[str] = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            try:
                serials.append(cam.TLDevice.DeviceSerialNumber.GetValue())
            except PySpin.SpinnakerException:
                pass
            del cam
        cam_list.Clear()
    finally:
        system.ReleaseInstance()
    return serials
