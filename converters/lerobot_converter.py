"""
Piper 데이터셋 → LeRobot v2.0 형식 변환기
- executed_joint.csv → observation.state, action (7-dim: q1-q6, gripper)
- webcam_0 / webcam_1 영상 → OpenCV frame decode/resize → 224×224 @ 15fps
- 15fps 그리드에 맞게 joint 데이터 선형 보간
- 트리밍 규칙: 10초 이후 첫 gripper open 전환 감지 후 +1초에서 끊음
"""

import csv
import json
import math
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from camera_manifest import (
    camera_by_id,
    camera_csv_name,
    default_mask_camera_id,
    export_cameras,
    frames_dir_name,
    legacy_id_for,
    load_manifest,
    overview_camera,
    video_file_name,
)

TARGET_FPS = 15
IMG_SIZE = 224
CODEBASE_VERSION = "v2.0"
ROBOT_TYPE = "piper"

# gripper open/closed 판정 임계값 (piper_node.py 와 동일)
GRIPPER_MID = 0.02219
# 이 시간(초) 이후에 발생한 gripper open 전환만 인식
RELEASE_SEARCH_START = 10.0
# gripper open 전환 이후 추가로 포함할 시간(초)
RELEASE_TAIL = 1.0


def _get_ffmpeg():
    from imageio_ffmpeg import get_ffmpeg_exe
    return get_ffmpeg_exe()


def _get_csv_duration(csv_path: str) -> float | None:
    """CSV 첫/마지막 t_host_ns 차이(초) 반환."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None
    t0 = int(float(rows[0]["t_host_ns"]))
    t1 = int(float(rows[-1]["t_host_ns"]))
    duration = (t1 - t0) / 1e9
    return duration if duration > 0 else None


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


def _get_take_video_duration(take_dir: str, cameras: list[dict] | None = None) -> float | None:
    """export 기준 비디오 duration. export camera를 우선 사용."""
    cam_files = [video_file_name(cam, "color") for cam in cameras or []]
    cam_files += ["video_webcam_1.mp4", "video_webcam_0.mp4", "video.mp4"]
    for cam_file in cam_files:
        duration = _get_video_duration(os.path.join(take_dir, cam_file))
        if duration and duration > 0:
            return duration
    return None


def _resample_joints(
    csv_path: str,
    target_fps: int = TARGET_FPS,
    output_duration: float | None = None,
) -> list[dict]:
    """
    joint CSV를 target_fps 그리드로 선형 보간.

    output_duration이 주어지면 원본 joint 시간축 전체를 output_duration 안으로
    정규화한다. replay 비디오가 executed_joint 실제 시간보다 짧게 speed-adjust된
    경우, parquet timestamp가 비디오 timestamp와 같은 시간축을 쓰도록 맞춘다.
    """
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    t0 = int(float(rows[0]["t_host_ns"]))
    t_arr = np.array([(int(float(r["t_host_ns"])) - t0) / 1e9 for r in rows])
    q_arr = np.array(
        [[float(r[f"q{i}"]) for i in range(1, 7)] + [float(r["gripper"])] for r in rows]
    )

    t_end = t_arr[-1]
    dst_end = float(output_duration) if output_duration and output_duration > 0 else float(t_end)
    t_grid = np.arange(0, dst_end + 1 / target_fps / 2, 1 / target_fps)
    t_grid = t_grid[t_grid <= dst_end + 1e-6]
    if output_duration and output_duration > 0:
        src_t_grid = np.clip(t_grid / max(dst_end, 1e-9) * t_end, 0.0, t_end)
    else:
        src_t_grid = t_grid

    q_grid = np.stack(
        [np.interp(src_t_grid, t_arr, q_arr[:, i]) for i in range(7)], axis=1
    )
    return [{"t": float(t), "q": q.tolist()} for t, q in zip(t_grid, q_grid)]


def _find_trim_gripper_open_t(samples: list[dict]) -> float | None:
    """10초 이후 첫 gripper closed→open 전환 시각만 반환 (마진 미포함)."""
    prev_open: bool | None = None
    for s in samples:
        t = s["t"]
        is_open = s["q"][6] > GRIPPER_MID
        if prev_open is not None and t > RELEASE_SEARCH_START:
            if not prev_open and is_open:
                return t
        prev_open = is_open
    return None


def _find_trim_time(samples: list[dict]) -> float | None:
    """10초 이후 첫 gripper OPEN 전환 시각 + RELEASE_TAIL(1초) 반환."""
    t = _find_trim_gripper_open_t(samples)
    return t + RELEASE_TAIL if t is not None else None


def _trim_samples(samples: list[dict], cut_t: float | None) -> list[dict]:
    """cut_t(초) 이후 프레임을 제거. cut_t=None이면 원본 반환."""
    if cut_t is None:
        return samples
    return [s for s in samples if s["t"] <= cut_t + 1e-6]


def _find_mask_frame(take_dir: str, t_sec: float, video_duration: float | None = None, camera_id: str | None = None) -> str | None:
    """비디오 t_sec에 가장 가까운 mask target camera 원본 프레임 파일 경로 반환."""
    cam = camera_by_id(camera_id) if camera_id else overview_camera()
    cam = cam or overview_camera()
    if not cam:
        return None
    csv_path = os.path.join(take_dir, camera_csv_name(cam))
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if video_duration and video_duration > 0:
        row_idx = int(round(np.clip(t_sec / video_duration, 0.0, 1.0) * (len(rows) - 1)))
        best = rows[row_idx]
    else:
        t0 = int(float(rows[0]["t_host_ns"]))
        t_target = t0 + int(t_sec * 1e9)
        best = min(rows, key=lambda r: abs(int(float(r["t_host_ns"])) - t_target))
    frame_idx = int(best["frame_idx"])
    frames_dir = os.path.join(take_dir, frames_dir_name(cam))
    for ext in ("jpg", "png"):
        path = os.path.join(frames_dir, f"color_{frame_idx:06d}.{ext}")
        if os.path.exists(path):
            return path
    return None


def _parse_hex_color(value: str | None, default: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    """Parse #RRGGBB. Export profiles may override the neutral default."""
    if not value:
        return default
    s = value.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return default
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return default


def _make_mask_overlay(polygon_norm: list, size: int = IMG_SIZE, fill_color: str | None = None) -> str | None:
    """
    polygon_norm: [[x_norm, y_norm], ...] normalized 0-1 좌표 (keep 영역)
    keep 영역 안은 투명, 밖은 fill_color 불투명인 PNG 파일 경로를 반환.
    polygon이 비어있으면 None 반환.
    """
    if not polygon_norm or len(polygon_norm) < 3:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    import tempfile
    r, g, b = _parse_hex_color(fill_color)
    img = Image.new("RGBA", (size, size), (r, g, b, 255))
    draw = ImageDraw.Draw(img)
    # keep 영역만 투명으로
    pts = [(int(x * size), int(y * size)) for x, y in polygon_norm]
    draw.polygon(pts, fill=(0, 0, 0, 0))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def _make_frame_capture_overlay(polygon_norm: list, frame_path: str, size: int = IMG_SIZE) -> str | None:
    """
    프레임 캡쳐 마스크: keep 영역(polygon 내부) = 투명, 외부 = 캡쳐 프레임 픽셀(불투명)
    ffmpeg overlay 시 keep 영역은 원본 비디오가, 외부는 고정 프레임이 보임.
    """
    if not polygon_norm or len(polygon_norm) < 3:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    import tempfile
    img = Image.open(frame_path).convert("RGBA").resize((size, size), Image.LANCZOS)
    # inside polygon → transparent (alpha=0), outside keeps frame pixels (alpha=255)
    alpha = Image.new("L", (size, size), 255)  # start fully opaque
    draw = ImageDraw.Draw(alpha)
    pts = [(int(x * size), int(y * size)) for x, y in polygon_norm]
    draw.polygon(pts, fill=0)  # inside = transparent
    img.putalpha(alpha)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def _encode_video(
    src_path: str,
    dst_path: str,
    target_fps: int = TARGET_FPS,
    size: int = IMG_SIZE,
    max_duration: float | None = None,
    mask_overlay: str | None = None,
    ss: float = 0.0,
    time_scale: float = 1.0,
):
    """Decode source frames, resize them, and encode a clean H.264 MP4.

    The source file is deliberately never passed to ffmpeg. OpenCV turns it into
    raw pixels first, so rotation tags, camera tags, chapters, audio, and all other
    source-container metadata cannot leak into the LeRobot export. ffmpeg is used
    only as a raw-frame H.264 encoder for broad LeRobot/player compatibility.

    ``ss`` is the source start offset in seconds. ``max_duration`` limits the
    exported *output* duration after that offset. ``time_scale`` changes only the
    presentation duration: 2.0 emits every decoded source frame for twice as
    long, preserving the physical replay speed when the preview was normalized
    to teaching speed. An optional RGBA mask is composited after resizing.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for LeRobot video export") from exc

    if target_fps <= 0 or size <= 0:
        raise ValueError("target_fps and size must be positive")
    time_scale = float(time_scale)
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be a positive finite number")
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV could not open source video: {src_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(src_fps) or src_fps <= 0:
        src_fps = float(target_fps)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(max(0.0, float(ss)) * src_fps)))
    available_frames = max(0, src_frame_count - start_frame)
    available_duration = (
        available_frames / src_fps * time_scale if available_frames else None
    )

    requested_duration = None
    if max_duration is not None:
        requested_duration = max(0.0, float(max_duration))
    output_duration = requested_duration
    if available_duration is not None:
        output_duration = (
            available_duration
            if output_duration is None
            else min(output_duration, available_duration)
        )
    output_frame_count = (
        max(0, int(math.ceil(output_duration * target_fps - 1e-9)))
        if output_duration is not None
        else None
    )

    overlay_bgra = None
    if mask_overlay:
        overlay_bgra = cv2.imread(mask_overlay, cv2.IMREAD_UNCHANGED)
        if overlay_bgra is None:
            cap.release()
            raise RuntimeError(f"OpenCV could not open mask overlay: {mask_overlay}")
        overlay_bgra = cv2.resize(overlay_bgra, (size, size), interpolation=cv2.INTER_AREA)
        if overlay_bgra.ndim != 3 or overlay_bgra.shape[2] not in (3, 4):
            cap.release()
            raise RuntimeError("Mask overlay must be an RGB or RGBA image")

    dst_dir = os.path.dirname(dst_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=".lerobot-video-", suffix=".mp4", dir=dst_dir, delete=False
    )
    tmp_path = tmp_file.name
    tmp_file.close()

    ffmpeg = _get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{size}x{size}", "-r", str(target_fps), "-i", "pipe:0",
        "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
        tmp_path,
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    encoded_frames = 0
    current_source_index = start_frame - 1
    current_frame = None

    try:
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while output_frame_count is None or encoded_frames < output_frame_count:
            desired_source_index = start_frame + int(
                round(encoded_frames * src_fps / (target_fps * time_scale))
            )
            while current_source_index < desired_source_index:
                ok, decoded = cap.read()
                if not ok:
                    current_frame = None
                    break
                current_source_index += 1
                current_frame = decoded
            if current_frame is None:
                break

            interpolation = (
                cv2.INTER_AREA
                if current_frame.shape[1] > size or current_frame.shape[0] > size
                else cv2.INTER_LINEAR
            )
            frame = cv2.resize(current_frame, (size, size), interpolation=interpolation)

            if overlay_bgra is not None:
                if overlay_bgra.shape[2] == 4:
                    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
                    frame = np.clip(
                        frame.astype(np.float32) * (1.0 - alpha)
                        + overlay_bgra[:, :, :3].astype(np.float32) * alpha,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    frame = overlay_bgra.copy()

            if process.stdin is None:
                raise RuntimeError("ffmpeg raw-frame input pipe is unavailable")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            encoded_frames += 1

        if encoded_frames == 0:
            raise RuntimeError(f"No frames decoded from source video: {src_path}")

        if process.stdin is not None:
            process.stdin.close()
            process.stdin = None
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg raw-frame encode failed:\n{stderr[-1000:]}")
        os.replace(tmp_path, dst_path)
    except (BrokenPipeError, OSError) as exc:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"ffmpeg raw-frame encode failed: {exc}\n{stderr[-1000:]}") from exc
    finally:
        cap.release()
        if process.poll() is None:
            process.terminate()
            process.wait()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.stderr is not None:
            process.stderr.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _get_latest_take_dir(ep_dir: str) -> str | None:
    """Return the newest take containing usable joint data.

    An interrupted retry can leave a newer empty take directory.  Falling back
    prevents that placeholder from silently removing an otherwise valid episode
    from an export (the 2026-08-27 batch had exactly one such case).
    """
    takes_dir = os.path.join(ep_dir, "takes")
    if not os.path.isdir(takes_dir):
        return None
    takes = sorted(
        [d for d in os.listdir(takes_dir) if os.path.isdir(os.path.join(takes_dir, d))]
    )
    for take in reversed(takes):
        take_dir = os.path.join(takes_dir, take)
        if any(os.path.isfile(os.path.join(take_dir, name)) for name in ("executed_joint.csv", "teach_joint.csv")):
            return take_dir
    return None


def _cam_feature_key(cam: dict) -> str:
    return f"observation.images.{cam['id']}"


def convert_episodes(
    episode_ids: list[str],
    dataset_path: str,
    out_dir: str,
    export_preset: str = "default",
    mask_mode: str = "edit",
    mask_fill_color: str = "#000000",
    preserve_source_indices: bool = False,
    include_default_excluded: bool = False,
) -> None:
    """
    선택된 episode_ids를 LeRobot v2.0 형식으로 out_dir에 변환 저장.
    각 episode의 마지막 take를 LeRobot 에피소드 1개로 매핑.
    """
    if preserve_source_indices and not include_default_excluded:
        raise ValueError(
            "preserve_source_indices requires include_default_excluded so the "
            "LeRobot index range remains contiguous"
        )
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest()
    export_cam_list = export_cameras(export_preset, manifest)
    if not export_cam_list:
        export_cam_list = export_cameras("default", manifest)

    # ── 1. 태스크 목록 수집 ──────────────────────────────────────────────
    task_to_idx: dict[str, int] = {}
    ep_tasks: list[str] = []

    for ep_id in episode_ids:
        meta_path = os.path.join(dataset_path, ep_id, "meta.json")
        task = "unknown"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            raw = meta.get("task", {})
            if isinstance(raw, dict):
                task = raw.get("instruction") or raw.get("text") or "unknown"
            elif isinstance(raw, str):
                task = raw or "unknown"
        except (OSError, json.JSONDecodeError):
            pass
        ep_tasks.append(task)
        if task not in task_to_idx:
            task_to_idx[task] = len(task_to_idx)

    # ── 2. 에피소드별 변환 ───────────────────────────────────────────────
    global_frame_idx = 0
    episodes_meta: list[dict] = []
    all_states: list[list[float]] = []
    all_actions: list[list[float]] = []

    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)

    source_map: list[dict] = []
    used_episode_indices: set[int] = set()

    for _source_ep_idx, (ep_id, task) in enumerate(zip(episode_ids, ep_tasks)):
        ep_dir = os.path.join(dataset_path, ep_id)
        take_dir = _get_latest_take_dir(ep_dir)
        if take_dir is None:
            continue
        take_name = os.path.basename(take_dir)
        replay_result_path = os.path.join(take_dir, "replay_result.json")
        replay_result: dict = {}
        if os.path.isfile(replay_result_path):
            try:
                with open(replay_result_path) as file:
                    replay_result = json.load(file)
            except (OSError, json.JSONDecodeError):
                replay_result = {}
        default_excluded = bool(replay_result.get("export_excluded_default"))
        if default_excluded and not include_default_excluded and export_preset != "debug":
            continue

        # ── edit_meta 및 timing policy 로드 ──────────────────────────────
        edit_meta_path = os.path.join(take_dir, "edit_meta.json")
        edit_meta: dict = {}
        if os.path.exists(edit_meta_path):
            try:
                with open(edit_meta_path) as f:
                    edit_meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                edit_meta = {}

        timing_cfg = edit_meta.get("timing", {})
        timing_mode = timing_cfg.get("mode", "teaching_normalized")
        if timing_mode not in ("teaching_normalized", "physical_replay"):
            timing_mode = "teaching_normalized"

        source_video_duration = _get_take_video_duration(take_dir, export_cam_list)
        joint_csv = os.path.join(take_dir, "executed_joint.csv")
        if not os.path.exists(joint_csv):
            joint_csv = os.path.join(take_dir, "teach_joint.csv")

        joint_duration = _get_csv_duration(joint_csv)
        video_time_scale = 1.0
        full_video_duration = source_video_duration
        if (
            timing_mode == "physical_replay"
            and source_video_duration is not None
            and source_video_duration > 0
            and joint_duration is not None
            and joint_duration > 0
        ):
            # Preview MP4 is normally shortened back to teaching speed. Stretch
            # its decoded pixels to the executed trajectory duration for selected
            # episodes without reintroducing source-container metadata.
            full_video_duration = joint_duration
            video_time_scale = joint_duration / source_video_duration

        raw_samples = _resample_joints(joint_csv, output_duration=full_video_duration)
        if not raw_samples:
            continue
        # Training exports compact excluded inputs. Review exports can include
        # every input and keep the archive's original episode file/index.
        original_index_raw = edit_meta.get("export", {}).get("original_episode_index", _source_ep_idx)
        try:
            original_ep_idx = int(original_index_raw)
        except (TypeError, ValueError):
            original_ep_idx = _source_ep_idx
        ep_idx = original_ep_idx if preserve_source_indices else len(episodes_meta)
        if ep_idx < 0 or ep_idx in used_episode_indices:
            raise ValueError(f"invalid or duplicate preserved episode index: {ep_idx}")
        used_episode_indices.add(ep_idx)

        # ── 트리밍 ────────────────────────────────────────────────────────
        trim_cfg = edit_meta.get("trim", {})
        if trim_cfg.get("enabled"):
            # 사용자가 설정한 cut_t (없으면 자동 감지)
            manual_cut_t = trim_cfg.get("cut_t")
            if manual_cut_t is not None:
                base_t = float(manual_cut_t)
            else:
                base_t = _find_trim_gripper_open_t(raw_samples) or raw_samples[-1]["t"]
            margin = float(trim_cfg.get("margin", 1.0))
            video_duration: float | None = base_t + margin
            if full_video_duration is not None:
                video_duration = min(video_duration, full_video_duration)
        else:
            # 기본 자동 트리밍 (converter 상수 기반)
            auto_cut = _find_trim_time(raw_samples)
            video_duration = auto_cut

        samples = _trim_samples(raw_samples, video_duration)

        # ── 마스크 ────────────────────────────────────────────────────────
        mask_cfg = edit_meta.get("mask", {})
        fallback_mask_camera_id = default_mask_camera_id(manifest)
        mask_overlay_path: str | None = None
        mask_enabled = mask_mode != "off" and bool(mask_cfg.get("enabled") and mask_cfg.get("polygon"))
        if mask_enabled:
            fill = mask_cfg.get("fill", "black")
            if fill == "frame_capture":
                capture_t = float(mask_cfg.get("capture_t", 0))
                mask_camera_id = str(mask_cfg.get("camera_id") or fallback_mask_camera_id or "")
                frame_path = _find_mask_frame(take_dir, capture_t, full_video_duration, mask_camera_id)
                if frame_path:
                    mask_overlay_path = _make_frame_capture_overlay(mask_cfg["polygon"], frame_path)
            else:  # "black" or default
                mask_overlay_path = _make_mask_overlay(mask_cfg["polygon"], fill_color=mask_fill_color)

        n_frames = len(samples)
        task_idx = task_to_idx[task]

        # Parquet 행 구성
        rows = []
        for fi, s in enumerate(samples):
            q = s["q"]
            rows.append({
                "index":             global_frame_idx + fi,
                "episode_index":     ep_idx,
                "frame_index":       fi,
                "timestamp":         float(s["t"]),
                "observation.state": q,
                "action":            q,
                "next.done":         fi == n_frames - 1,
                "task_index":        task_idx,
            })
            all_states.append(q)
            all_actions.append(q)

        df = pd.DataFrame(rows)
        parquet_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
        df.to_parquet(parquet_path, index=False)

        # 비디오 변환 — 트림 + 마스크 적용
        for cam in export_cam_list:
            cam_key = _cam_feature_key(cam)
            cam_file = video_file_name(cam, "color")
            src = os.path.join(take_dir, cam_file)
            if not os.path.exists(src):
                continue
            dst = os.path.join(
                out_dir, "videos", "chunk-000", cam_key,
                f"episode_{ep_idx:06d}.mp4",
            )
            mask_camera_id = str(mask_cfg.get("camera_id") or fallback_mask_camera_id or "")
            mask_cam = camera_by_id(mask_camera_id) or overview_camera()
            mask_target_id = mask_cam.get("id") if mask_cam else None
            apply_mask = mask_overlay_path if cam.get("id") == mask_target_id else None
            _encode_video(
                src,
                dst,
                max_duration=video_duration,
                mask_overlay=apply_mask,
                ss=0.0,
                time_scale=video_time_scale,
            )

        # 임시 마스크 PNG 정리
        if mask_overlay_path and os.path.exists(mask_overlay_path):
            os.unlink(mask_overlay_path)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         [task],
            "length":        n_frames,
            "source_episode_id": ep_id,
            "source_take": take_name,
            "task": task,
            "task_index": task_idx,
            "source_order": _source_ep_idx,
            "original_episode_index": original_ep_idx,
            "review_excluded": default_excluded,
        })
        source_entry = {
            "episode_index": ep_idx,
            "original_episode_index": original_ep_idx,
            "lerobot_episode_file": f"episode_{ep_idx:06d}",
            "source_episode_id": ep_id,
            "source_take": take_name,
            "task": task,
            "task_index": task_idx,
            "length": n_frames,
            "review_excluded": default_excluded,
            "review_exclusion_reason": replay_result.get("export_exclusion_reason"),
            "video_duration_s": video_duration,
            "full_video_duration_s": full_video_duration,
            "source_video_duration_s": source_video_duration,
            "joint_duration_s": joint_duration,
            "timing": {
                "mode": timing_mode,
                "video_time_scale": video_time_scale,
            },
            "trim": {
                "enabled": bool(trim_cfg.get("enabled")),
                "cut_t": trim_cfg.get("cut_t"),
                "margin": trim_cfg.get("margin"),
                "applied_duration_s": video_duration,
            },
            "mask": {
                "mode": mask_mode,
                "enabled": mask_enabled,
                "camera_id": str(mask_cfg.get("camera_id") or fallback_mask_camera_id or ""),
                "fill": mask_cfg.get("fill", "black"),
                "fill_color": mask_fill_color if mask_enabled and mask_cfg.get("fill", "black") != "frame_capture" else None,
            },
        }
        source_map.append(source_entry)
        global_frame_idx += n_frames

    # ── 3. 통계 계산 ─────────────────────────────────────────────────────
    def _stats(arr: list[list[float]]) -> dict:
        a = np.array(arr, dtype=np.float32)
        return {
            "mean": a.mean(axis=0).tolist(),
            "std":  a.std(axis=0).tolist(),
            "min":  a.min(axis=0).tolist(),
            "max":  a.max(axis=0).tolist(),
        }

    joint_names = ["q1", "q2", "q3", "q4", "q5", "q6", "gripper"]
    stats_data: dict[str, dict] = {}
    if all_states:
        stats_data["observation.state"] = _stats(all_states)
        stats_data["action"] = _stats(all_actions)

    # ── 4. 메타 파일 작성 ────────────────────────────────────────────────
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    total_eps = len(episodes_meta)
    total_frames = global_frame_idx
    if preserve_source_indices and sorted(used_episode_indices) != list(range(total_eps)):
        raise ValueError(
            "preserved review indices must form a contiguous 0..N-1 range; "
            f"got {sorted(used_episode_indices)}"
        )

    cam_feature = {
        "dtype": "video",
        "shape": [IMG_SIZE, IMG_SIZE, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": TARGET_FPS,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "fps": TARGET_FPS,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": joint_names},
            "action":            {"dtype": "float32", "shape": [7], "names": joint_names},
            **{_cam_feature_key(cam): cam_feature for cam in export_cam_list},
        },
        "splits":          {"train": f"0:{total_eps}"},
        "total_episodes":  total_eps,
        "total_frames":    total_frames,
        "total_tasks":     len(task_to_idx),
        "total_chunks":    1,
        "chunks_size":     1000,
        "data_path":       "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path":      "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "camera_manifest": {
            "export_preset": export_preset,
            "exported_cameras": [
                {
                    "id": cam["id"],
                    "role": cam.get("role"),
                    "label": cam.get("label"),
                    "legacy_id": legacy_id_for(cam),
                    "feature_key": _cam_feature_key(cam),
                }
                for cam in export_cam_list
            ],
            "legacy_feature_note": "Previous exports used observation.images.cam_webcam_0/1; this export uses role ids cam0/cam1.",
        },
        "export_options": {
            "mask_mode": mask_mode,
            "mask_fill_color": mask_fill_color,
            "video_processing": "opencv_raw_frames",
            "video_resolution": [IMG_SIZE, IMG_SIZE],
            "source_metadata_copied": False,
            "per_episode_timing_modes": ["teaching_normalized", "physical_replay"],
            "preserve_source_indices": preserve_source_indices,
            "include_default_excluded": include_default_excluded,
        },
    }

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats_data, f, indent=2)

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task, idx in sorted(task_to_idx.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep_meta in episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")

    with open(os.path.join(meta_dir, "source_episodes.jsonl"), "w") as f:
        for row in source_map:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(meta_dir, "source_episodes.csv"), "w", newline="") as f:
        fieldnames = [
            "episode_index",
            "original_episode_index",
            "lerobot_episode_file",
            "source_episode_id",
            "source_take",
            "task",
            "task_index",
            "length",
            "review_excluded",
            "review_exclusion_reason",
            "video_duration_s",
            "full_video_duration_s",
            "source_video_duration_s",
            "joint_duration_s",
            "timing.mode",
            "timing.video_time_scale",
            "mask.mode",
            "mask.enabled",
            "mask.camera_id",
            "mask.fill",
            "mask.fill_color",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_map:
            writer.writerow({
                "episode_index": row["episode_index"],
                "original_episode_index": row["original_episode_index"],
                "lerobot_episode_file": row["lerobot_episode_file"],
                "source_episode_id": row["source_episode_id"],
                "source_take": row["source_take"],
                "task": row["task"],
                "task_index": row["task_index"],
                "length": row["length"],
                "review_excluded": row["review_excluded"],
                "review_exclusion_reason": row["review_exclusion_reason"],
                "video_duration_s": row["video_duration_s"],
                "full_video_duration_s": row["full_video_duration_s"],
                "source_video_duration_s": row["source_video_duration_s"],
                "joint_duration_s": row["joint_duration_s"],
                "timing.mode": row["timing"]["mode"],
                "timing.video_time_scale": row["timing"]["video_time_scale"],
                "mask.mode": row["mask"]["mode"],
                "mask.enabled": row["mask"]["enabled"],
                "mask.camera_id": row["mask"]["camera_id"],
                "mask.fill": row["mask"]["fill"],
                "mask.fill_color": row["mask"]["fill_color"],
            })
