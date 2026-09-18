"""
EpisodeManager
--------------
episode = 하나의 작업 (예: "pick red cup").
take    = episode 내의 개별 녹화 시도 (teach→replay 1회).

디렉터리 구조:
  dataset/
    episode_YYYYMMDD_HHMMSS/
      meta.json        ← episode 메타 (task, takes 목록 등)
      label.json       ← 최종 라벨 (save 시 작성)
      takes/
        take_001/
          teach_joint.csv, executed_joint.csv, camera_frames.csv,
          aligned_frames.csv, replay_command.csv, events.json
          frames/ (color_*.png, depth_*.png)
          video.mp4, video_depth.mp4
        take_002/
          ...
"""

import csv
import json
import os
import shutil
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASET_PATH, REALSENSE_FPS
from camera_manifest import (
    camera_csv_name,
    frames_dir_name,
    legacy_id_for,
    load_manifest,
    video_file_name,
)

DOF = 6


def _episode_id() -> str:
    return datetime.now().strftime("episode_%Y%m%d_%H%M%S")


class EpisodeManager:

    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        os.makedirs(dataset_path, exist_ok=True)

        self.current_episode_id: Optional[str] = None
        self.current_episode_dir: Optional[str] = None
        self.current_take_dir: Optional[str] = None
        self._take_num: int = 0
        self._meta: dict = {}

    # ─────────────────────────────────────────────────────────────────
    # Episode / take lifecycle
    # ─────────────────────────────────────────────────────────────────

    def create_episode(self, task: str = "unspecified", operator: str = "unknown") -> str:
        """새 episode + take_001 생성."""
        episode_id = _episode_id()
        episode_dir = os.path.join(self.dataset_path, episode_id)
        os.makedirs(episode_dir, exist_ok=True)

        self.current_episode_id = episode_id
        self.current_episode_dir = episode_dir
        self._take_num = 0

        self._meta = {
            "episode_id": episode_id,
            "version": "piper-realsense-v0.2",
            "created_at": datetime.now().isoformat(),
            "task": {"instruction": task, "success": None},
            "operator": operator,
            "robot": {
                "model": "AgileX Piper",
                "dof": DOF,
                "gripper": "1-DOF gripper",
                "control_mode": "kinesthetic_teach_replay",
            },
            "camera": {
                "model": "Intel RealSense",
                "mount": "external tripod",
                "stream": ["rgb", "depth"],
                "fps": REALSENSE_FPS,
                "resolution": "1280x720",
            },
            "takes": [],
        }
        self._write_episode_json("meta.json", self._meta)
        # 첫 번째 take 생성
        self.add_take()
        print(f"[EpisodeManager] Created episode: {episode_id}")
        return episode_id

    def add_take(self) -> str:
        """현재 episode에 새 take 추가. 이후 flush 메서드는 이 take 디렉터리에 씁니다."""
        if not self.current_episode_dir:
            raise RuntimeError("No active episode. Call create_episode() first.")

        self._take_num += 1
        take_name = f"take_{self._take_num:03d}"
        take_dir = os.path.join(self.current_episode_dir, "takes", take_name)
        os.makedirs(take_dir, exist_ok=True)
        os.makedirs(os.path.join(take_dir, "frames"), exist_ok=True)

        self.current_take_dir = take_dir

        # meta.json takes 목록 갱신
        if take_name not in self._meta.get("takes", []):
            self._meta.setdefault("takes", []).append(take_name)
            self._write_episode_json("meta.json", self._meta)

        print(f"[EpisodeManager] Added take: {take_name} in {self.current_episode_id}")
        return take_dir

    def get_episode_dir(self) -> Optional[str]:
        """현재 take 디렉터리 반환 (카메라 녹화 경로로 사용)."""
        return self.current_take_dir

    def get_takes_count(self) -> int:
        """현재 episode의 take 수 반환."""
        if not self.current_episode_dir:
            return 0
        takes_dir = os.path.join(self.current_episode_dir, "takes")
        if not os.path.isdir(takes_dir):
            return 0
        return len([d for d in os.listdir(takes_dir) if d.startswith("take_")])

    def get_current_take_name(self) -> Optional[str]:
        if self._take_num == 0:
            return None
        return f"take_{self._take_num:03d}"

    def discard_current_take(self):
        """현재 take 데이터만 삭제. episode 폴더는 유지."""
        if self.current_take_dir and os.path.exists(self.current_take_dir):
            shutil.rmtree(self.current_take_dir)
            self._take_num = max(0, self._take_num - 1)
            # meta에서 제거
            take_name = f"take_{self._take_num + 1:03d}"
            if take_name in self._meta.get("takes", []):
                self._meta["takes"].remove(take_name)
                self._write_episode_json("meta.json", self._meta)
            print(f"[EpisodeManager] Take discarded: {take_name}")
        self.current_take_dir = None

    def discard_episode(self):
        """전체 episode 삭제."""
        if self.current_episode_dir and os.path.exists(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
            print(f"[EpisodeManager] Episode discarded: {self.current_episode_id}")
        self.current_episode_id = None
        self.current_episode_dir = None
        self.current_take_dir = None
        self._take_num = 0

    # ─────────────────────────────────────────────────────────────────
    # CSV writers  (모두 current_take_dir 사용)
    # ─────────────────────────────────────────────────────────────────

    def flush_teach_joint(self, buffer: list[dict]) -> int:
        if not buffer or not self.current_take_dir:
            return 0
        fields = ["t_host_ns"] + [f"q{i+1}" for i in range(DOF)] + [f"dq{i+1}" for i in range(DOF)] + ["gripper", "mode"]
        path = os.path.join(self.current_take_dir, "teach_joint.csv")
        n = self._write_csv(path, fields, buffer)
        self._meta["recording"] = self._meta.get("recording", {})
        self._meta["recording"]["teach_recorded"] = True
        self._write_episode_json("meta.json", self._meta)
        print(f"[EpisodeManager] teach_joint.csv: {n} rows")
        return n

    def flush_executed_joint(self, buffer: list[dict]) -> int:
        if not buffer or not self.current_take_dir:
            return 0
        fields = ["t_host_ns"] + [f"q{i+1}" for i in range(DOF)] + [f"dq{i+1}" for i in range(DOF)] + ["gripper", "mode"]
        path = os.path.join(self.current_take_dir, "executed_joint.csv")
        n = self._write_csv(path, fields, buffer)
        print(f"[EpisodeManager] executed_joint.csv: {n} rows")
        return n

    def flush_replay_command(self, buffer: list[dict]) -> int:
        if not buffer or not self.current_take_dir:
            return 0
        fields = [
            "t_host_ns", "scheduled_t_host_ns", "waypoint_idx", "source_idx",
            "lateness_ms", "velocity",
        ] + [f"q{i+1}" for i in range(DOF)] + ["gripper", "gripper_effort"]
        path = os.path.join(self.current_take_dir, "replay_command.csv")
        n = self._write_csv(path, fields, buffer)
        print(f"[EpisodeManager] replay_command.csv: {n} rows")
        return n

    def flush_camera_frames(self, buffer: list[dict]) -> int:
        """단일 카메라 buffer (realsense 등, 하위 호환)."""
        if not buffer or not self.current_take_dir:
            return 0
        fields = ["frame_idx", "t_host_ns", "t_sensor_ms", "color_path", "depth_path"]
        path = os.path.join(self.current_take_dir, "camera_frames.csv")
        n = self._write_csv(path, fields, buffer)
        print(f"[EpisodeManager] camera_frames.csv: {n} rows")
        return n

    def flush_camera_frames_multi(self, buffers: dict) -> int:
        """
        multi-camera buffer flush.
        buffers: {'realsense': [...], 'webcam_0': [...], ...}
        각 카메라별로 camera_frames_{cam_id}.csv 저장.
        """
        if not self.current_take_dir:
            return 0
        total = 0
        for cam_id, buffer in buffers.items():
            if not buffer:
                continue
            # 공통 필드 + camera_id 컬럼 추가
            all_keys = set()
            for row in buffer:
                all_keys.update(row.keys())
            # 표준 필드 순서
            ordered = ["frame_idx", "t_host_ns"]
            for k in ["t_sensor_ms", "color_path", "depth_path"]:
                if k in all_keys:
                    ordered.append(k)
            # 나머지 필드
            for k in sorted(all_keys - set(ordered)):
                ordered.append(k)

            path = os.path.join(self.current_take_dir, f"camera_frames_{cam_id}.csv")
            n = self._write_csv(path, ordered, buffer)
            print(f"[EpisodeManager] camera_frames_{cam_id}.csv: {n} rows")
            total += n
        return total

    def flush_events(self, events: list[dict]):
        if not self.current_take_dir:
            return
        path = os.path.join(self.current_take_dir, "events.json")
        with open(path, "w") as f:
            json.dump(events, f, indent=2)
        print(f"[EpisodeManager] events.json: {len(events)} events")

    def mark_replay_result(
        self,
        *,
        partial: bool,
        command_samples: int,
        executed_samples: int,
        reason: str,
        abort_class: str | None = None,
        export_excluded_default: bool = False,
        diagnostics: dict | None = None,
    ):
        """Persist whether this take completed or was intentionally stopped."""
        if not self.current_take_dir:
            return
        result = {
            "completed": not partial,
            "partial": bool(partial),
            "reason": str(reason),
            "abort_class": abort_class,
            "export_excluded_default": bool(export_excluded_default),
            "diagnostics": dict(diagnostics or {}),
            "command_samples": int(command_samples),
            "executed_samples": int(executed_samples),
            "finished_at": datetime.now().isoformat(),
        }
        with open(os.path.join(self.current_take_dir, "replay_result.json"), "w") as file:
            json.dump(result, file, indent=2)
        self._meta.setdefault("recording", {})["last_replay"] = {
            **result,
            "take": self.get_current_take_name(),
        }
        self._write_episode_json("meta.json", self._meta)

    # ─────────────────────────────────────────────────────────────────
    # Alignment
    # ─────────────────────────────────────────────────────────────────

    def generate_aligned_frames(self) -> int:
        if not self.current_take_dir:
            return 0

        # 멀티카메라: realsense 또는 primary webcam의 frame CSV 사용
        joint_path = os.path.join(self.current_take_dir, "executed_joint.csv")
        cam_path = None
        for candidate in ["camera_frames_realsense.csv", "camera_frames_webcam_0.csv", "camera_frames.csv"]:
            p = os.path.join(self.current_take_dir, candidate)
            if os.path.exists(p):
                cam_path = p
                break

        if cam_path is None or not os.path.exists(joint_path):
            print("[EpisodeManager] Missing camera_frames or executed_joint CSV")
            return 0

        cam_rows = self._read_csv(cam_path)
        joint_rows = self._read_csv(joint_path)
        if not cam_rows or not joint_rows:
            return 0

        joint_ts = np.array([int(r["t_host_ns"]) for r in joint_rows], dtype=np.int64)
        joint_vals = {
            f"q{i+1}": np.array([float(r.get(f"q{i+1}", 0)) for r in joint_rows])
            for i in range(DOF)
        }
        joint_vals["gripper"] = np.array([float(r.get("gripper", 0)) for r in joint_rows])

        aligned_fields = (
            ["frame_idx", "t_host_ns", "color_path", "depth_path"]
            + [f"q{i+1}" for i in range(DOF)]
            + ["gripper", "episode_id", "take"]
        )

        aligned_rows = []
        for row in cam_rows:
            t_cam = int(row["t_host_ns"])
            idx = np.searchsorted(joint_ts, t_cam)
            idx = int(np.clip(idx, 0, len(joint_ts) - 1))

            if idx == 0 or idx >= len(joint_ts) - 1:
                interp_vals = {k: float(v[idx]) for k, v in joint_vals.items()}
            else:
                t0, t1 = joint_ts[idx - 1], joint_ts[idx]
                alpha = (t_cam - t0) / max(t1 - t0, 1)
                interp_vals = {
                    k: float(v[idx - 1] + alpha * (v[idx] - v[idx - 1]))
                    for k, v in joint_vals.items()
                }

            aligned_rows.append({
                "frame_idx": row["frame_idx"],
                "t_host_ns": row["t_host_ns"],
                "color_path": row["color_path"],
                "depth_path": row.get("depth_path", ""),
                **interp_vals,
                "episode_id": self.current_episode_id,
                "take": self.get_current_take_name(),
            })

        out_path = os.path.join(self.current_take_dir, "aligned_frames.csv")
        n = self._write_csv(out_path, aligned_fields, aligned_rows)
        print(f"[EpisodeManager] aligned_frames.csv: {n} rows")
        return n

    # ─────────────────────────────────────────────────────────────────
    # Preview video
    # ─────────────────────────────────────────────────────────────────

    def generate_preview_videos(
        self,
        fps: int = REALSENSE_FPS,
        replay_speed: float = 1.0,
        on_step=None,          # callable(step: str, status: str, detail: str)
    ) -> dict:
        """
        모든 카메라 비디오 생성.

        핵심 원칙:
          - executed_joint.csv 를 replay 실제 시간의 ground truth로 사용
          - target_duration = replay_duration_s * replay_speed  (Teaching 속도 환산)
          - 각 카메라: output_fps = n_frames / target_duration
          - frame drop이 있어도 모든 카메라 영상이 같은 길이로 출력됨
        """
        replay_speed = max(0.05, float(replay_speed))

        def _step(step: str, status: str, detail: str = ""):
            if on_step:
                on_step(step, status, detail)

        # ── replay 실제 시간 측정 (executed_joint.csv 기준) ──────────
        replay_duration_s = self._get_replay_duration()
        if replay_duration_s is None:
            # fallback: 단순 fps/speed 비율
            print("[EpisodeManager] executed_joint.csv 없음 — fps 비율 방식으로 fallback")
            result = {}
            rs_fps = REALSENSE_FPS / replay_speed
            for step_key, fn in [
                ("video_color",    lambda: self.generate_preview_video(fps=rs_fps, stream="color")),
                ("video_depth",    lambda: self.generate_preview_video(fps=rs_fps, stream="depth")),
                ("video_webcam_0", lambda: self._generate_webcam_video_simple("webcam_0", replay_speed)),
                ("video_webcam_1", lambda: self._generate_webcam_video_simple("webcam_1", replay_speed)),
            ]:
                _step(step_key, "running")
                r = fn()
                _step(step_key, "ok" if r else "failed")
                result[step_key.replace("video_", "")] = r
            return result

        target_duration_s = replay_duration_s * replay_speed
        print(
            f"[EpisodeManager] replay_duration={replay_duration_s:.2f}s, "
            f"replay_speed={replay_speed:.2f}, target_duration={target_duration_s:.2f}s"
        )

        result = {}
        jobs = []
        manifest = load_manifest()
        for cam in manifest.get("cameras", []):
            if not cam.get("enabled", True):
                continue
            legacy = legacy_id_for(cam)
            fdir = frames_dir_name(cam)
            csv_name = camera_csv_name(cam)

            if cam.get("type") == "spinnaker":
                # 녹화 중엔 센서 원본 모자이크만 저장했으므로, 영상 만들기 전에 여기서
                # color/AoLP/DoLP/grid 뷰 프레임을 먼저 재구성해야 아래 job들이 소스를 찾음.
                from nodes.spinnaker_node import reconstruct_polarizer_frame_views
                n = reconstruct_polarizer_frame_views(os.path.join(self.current_take_dir, fdir))
                if n:
                    print(f"[EpisodeManager] {cam['id']}: reconstructed {n} frame views from mosaic")

            if "color" in cam.get("streams", []):
                step_key = "video_color" if legacy == "realsense" else f"video_{cam['id']}"
                fallback_csv = "camera_frames.csv" if legacy == "realsense" else None
                jobs.append((step_key, fdir, "color_", video_file_name(cam, "color"), csv_name, fallback_csv))
            if "depth" in cam.get("streams", []):
                jobs.append(("video_depth", fdir, "depth_", video_file_name(cam, "depth"), csv_name, "camera_frames.csv"))
            # 편광 카메라: 컬러라이즈된 AoLP/DoLP/2x2 grid 시각화 영상도 생성 (prefix가 raw
            # aolp_*/dolp_* PNG와 안 겹치게 aolpview_/dolpview_/gridview_ 사용).
            if "aolp" in cam.get("streams", []):
                jobs.append((f"video_{cam['id']}_aolp", fdir, "aolpview_", video_file_name(cam, "aolp"), csv_name, None))
            if "dolp" in cam.get("streams", []):
                jobs.append((f"video_{cam['id']}_dolp", fdir, "dolpview_", video_file_name(cam, "dolp"), csv_name, None))
            if "grid" in cam.get("streams", []):
                jobs.append((f"video_{cam['id']}_grid", fdir, "gridview_", video_file_name(cam, "grid"), csv_name, None))
            if "stokes" in cam.get("streams", []):
                jobs.append((f"video_{cam['id']}_stokes", fdir, "stokesview_", video_file_name(cam, "stokes"), csv_name, None))

        # Legacy fallback if cameras.json is missing old webcams.
        if not jobs:
            jobs = [
                ("video_color",    "frames",          "color_",  "video.mp4",        "camera_frames_realsense.csv", "camera_frames.csv"),
                ("video_depth",    "frames",          "depth_",  "video_depth.mp4",  "camera_frames_realsense.csv", "camera_frames.csv"),
                ("video_webcam_0", "frames_webcam_0", "color_",  "video_webcam_0.mp4", "camera_frames_webcam_0.csv", None),
                ("video_webcam_1", "frames_webcam_1", "color_",  "video_webcam_1.mp4", "camera_frames_webcam_1.csv", None),
            ]
        for step_key, fdir, prefix, out_name, cam_csv_name, fallback_csv_name in jobs:
            _step(step_key, "running")
            r = self._generate_video_concat(
                frames_dir=os.path.join(self.current_take_dir, fdir),
                prefix=prefix, out_name=out_name,
                cam_csv=os.path.join(self.current_take_dir, cam_csv_name) if cam_csv_name else None,
                fallback_csv=os.path.join(self.current_take_dir, fallback_csv_name) if fallback_csv_name else None,
                replay_duration_s=replay_duration_s, target_duration_s=target_duration_s,
            )
            result[step_key.replace("video_", "")] = r
            _step(step_key, "ok" if r else "failed",
                  f"{target_duration_s:.1f}s" if r else "skipped (no frames)")
        return result

    def regenerate_camera_videos(self, episode_id: str, take: str, on_step=None) -> dict:
        """
        이미 완료된 기존 episode/take에 대해 카메라 영상을 다시 생성.
        처리 로직이 나중에 추가/수정됐을 때(예: 편광 카메라 AoLP/DoLP/grid 뷰 호환 재구성)
        옛날 take에도 적용하는 용도 — current_take_dir을 일시적으로 바꿔서
        generate_preview_videos()를 재사용한다.
        """
        take_dir = os.path.join(self.dataset_path, episode_id, "takes", take)
        if not os.path.isdir(take_dir):
            return {"ok": False, "error": "take not found"}

        prev_episode_id, prev_take_dir = self.current_episode_id, self.current_take_dir
        self.current_episode_id = episode_id
        self.current_take_dir = take_dir
        try:
            result = self.generate_preview_videos(on_step=on_step)
        finally:
            self.current_episode_id, self.current_take_dir = prev_episode_id, prev_take_dir
        return {"ok": True, "result": result}

    def _get_replay_duration(self) -> Optional[float]:
        """executed_joint.csv 첫/마지막 t_host_ns 로 실제 replay 시간(초) 측정."""
        if not self.current_take_dir:
            return None
        path = os.path.join(self.current_take_dir, "executed_joint.csv")
        try:
            rows = self._read_csv(path)
            if len(rows) < 2:
                return None
            t0 = int(rows[0]["t_host_ns"])
            t1 = int(rows[-1]["t_host_ns"])
            d = (t1 - t0) / 1e9
            return d if d > 0.5 else None
        except Exception:
            return None

    def _generate_video_concat(
        self,
        frames_dir: str,
        prefix: str,
        out_name: str,
        cam_csv: Optional[str],
        fallback_csv: Optional[str],
        replay_duration_s: float,
        target_duration_s: float,
    ) -> Optional[str]:
        """
        output_fps = n_frames / target_duration_s 로 비디오 생성.

        - target_duration_s = replay_duration_s × replay_speed (Teaching 속도 환산)
        - frame drop이 있어도 모든 카메라 영상이 같은 길이로 출력됨
        - -framerate output_fps 방식 사용: 각 프레임이 1/output_fps 초로 처리되어
          총 duration = n / output_fps = target_duration_s 보장
        """
        if not self.current_take_dir or not os.path.isdir(frames_dir):
            return None

        all_files = sorted(os.listdir(frames_dir))
        frames = sorted([
            f for f in all_files
            if f.startswith(prefix) and (f.endswith(".jpg") or f.endswith(".png"))
        ])
        if not frames:
            return None

        # 확장자 자동 감지 (.jpg 우선 — 신규 기록, .png fallback — 구형 기록)
        ext = os.path.splitext(frames[0])[1]   # ".jpg" or ".png"
        out_path = os.path.join(self.current_take_dir, out_name)
        n = len(frames)

        # output_fps: n 프레임을 target_duration_s 안에 담는 fps
        output_fps = max(1.0, n / target_duration_s)

        print(
            f"[EpisodeManager] {out_name}: n={n} ({ext}), replay={replay_duration_s:.2f}s, "
            f"target={target_duration_s:.2f}s, output_fps={output_fps:.1f}"
        )

        # -framerate output_fps -i pattern: 각 프레임을 1/output_fps 초로 처리
        # → 총 duration = n / output_fps = target_duration_s (정확히 일치)
        result = self._run_ffmpeg_sequential(
            frames_dir, f"{prefix}%06d{ext}", out_path, output_fps
        )
        if result:
            print(f"[EpisodeManager] {out_name} (H.264): {n} frames → {target_duration_s:.1f}s")
            return result

        print(f"[EpisodeManager] ffmpeg failed for {out_name}, using cv2 fallback")
        return self._generate_preview_video_fallback(
            frames_dir, frames, out_path, int(round(output_fps))
        )

    def _generate_webcam_video_simple(self, cam_name: str, replay_speed: float = 1.0) -> Optional[str]:
        """fallback: CSV timestamp 없을 때 단순 fps 비율 방식."""
        if not self.current_take_dir:
            return None
        frames_dir = os.path.join(self.current_take_dir, f"frames_{cam_name}")
        if not os.path.isdir(frames_dir):
            return None
        pngs = sorted([f for f in os.listdir(frames_dir) if f.startswith("color_") and f.endswith(".png")])
        if not pngs:
            return None
        out_path = os.path.join(self.current_take_dir, f"video_{cam_name}.mp4")
        # 실제 fps 측정 불가 시 30fps 가정
        output_fps = max(1.0, 30.0 / replay_speed)
        return self._run_ffmpeg_sequential(frames_dir, "color_%06d.png", out_path, output_fps) or \
               self._generate_preview_video_fallback(frames_dir, pngs, out_path, int(round(output_fps)))

    def _run_ffmpeg_sequential(self, frames_dir: str, pattern: str, out_path: str, fps: float) -> Optional[str]:
        """순차 PNG → H.264 (concat 없이 단순 인코딩)."""
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            import subprocess as _sp
            ffmpeg_bin = get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin, "-y",
                "-framerate", f"{fps:.4f}",
                "-i", os.path.join(frames_dir, pattern),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "23",
                out_path,
            ]
            r = _sp.run(cmd, capture_output=True, text=True, timeout=120)
            return out_path if r.returncode == 0 else None
        except Exception:
            return None

    def generate_preview_video(self, fps: float = REALSENSE_FPS, stream: str = "color") -> Optional[str]:
        """
        단독 호출용 (generate_preview_videos 미사용 시).
        fps: 출력 fps (float 허용).
        """
        if not self.current_take_dir:
            return None
        frames_dir = os.path.join(self.current_take_dir, "frames")
        if not os.path.isdir(frames_dir):
            return None

        prefix = "color_" if stream == "color" else "depth_"
        out_name = "video.mp4" if stream == "color" else "video_depth.mp4"
        pngs = sorted([f for f in os.listdir(frames_dir) if f.startswith(prefix) and f.endswith(".png")])
        if not pngs:
            print(f"[EpisodeManager] generate_preview_video: no {stream} frames found")
            return None

        out_path = os.path.join(self.current_take_dir, out_name)
        fps_str = f"{fps:.4f}"
        print(f"[EpisodeManager] {out_name}: output_fps={fps:.1f} ({len(pngs)} frames)")

        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            import subprocess as _sp

            ffmpeg_bin = get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin, "-y",
                "-framerate", fps_str,
                "-i", os.path.join(frames_dir, f"{prefix}%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                out_path,
            ]
            result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"[EpisodeManager] ffmpeg failed:\n{result.stderr[-1000:]}")
                return self._generate_preview_video_fallback(frames_dir, pngs, out_path, int(round(fps)))
            print(f"[EpisodeManager] {out_name} (H.264) generated: {len(pngs)} frames")
            return out_path

        except Exception as e:
            print(f"[EpisodeManager] H.264 encoding error: {e}, falling back to mp4v")
            return self._generate_preview_video_fallback(frames_dir, pngs, out_path, int(round(fps)))

    def _generate_preview_video_fallback(self, frames_dir, pngs, out_path, fps) -> Optional[str]:
        sample = cv2.imread(os.path.join(frames_dir, pngs[0]))
        if sample is None:
            return None
        h, w = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for fname in pngs:
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is not None:
                writer.write(img)
        writer.release()
        print(f"[EpisodeManager] video.mp4 (mp4v fallback): {len(pngs)} frames")
        return out_path

    # ─────────────────────────────────────────────────────────────────
    # Finalize
    # ─────────────────────────────────────────────────────────────────

    def finalize_episode(self, success: bool, reason: str = "") -> dict:
        label = {
            "success": success,
            "failure_reason": reason if not success else "",
            "labeled_at": datetime.now().isoformat(),
            "takes": self._meta.get("takes", []),
        }
        if self.current_episode_dir:
            self._write_episode_json("label.json", label)
            self._meta["task"]["success"] = success
            self._write_episode_json("meta.json", self._meta)

        checklist = self._build_checklist()
        print(f"[EpisodeManager] Episode finalized: success={success}, takes={self._take_num}")
        return {"label": label, "checklist": checklist, "takes": self._take_num}

    def _build_checklist(self) -> dict:
        if not self.current_take_dir:
            return {}
        manifest = load_manifest()
        cam_frame_files = ["camera_frames.csv"] + [
            camera_csv_name(cam) for cam in manifest.get("cameras", [])
        ]
        files = {
            "teach_joint.csv": "teach_recorded",
            "executed_joint.csv": "executed_recorded",
            "aligned_frames.csv": "aligned_generated",
            "events.json": "events_logged",
            "video.mp4": "preview_video",
        }
        result = {}
        for fname, key in files.items():
            result[key] = os.path.exists(os.path.join(self.current_take_dir, fname))
        # 카메라 프레임: 하나라도 있으면 True
        result["camera_frames_recorded"] = any(
            os.path.exists(os.path.join(self.current_take_dir, f)) for f in cam_frame_files
        )
        for cam in manifest.get("cameras", []):
            legacy = legacy_id_for(cam)
            recorded = os.path.exists(os.path.join(self.current_take_dir, camera_csv_name(cam)))
            result[f"{cam['id']}_recorded"] = recorded
            if legacy != cam["id"]:
                result[f"{legacy}_recorded"] = recorded
        result["labeled"] = os.path.exists(os.path.join(self.current_episode_dir, "label.json"))
        return result

    # ─────────────────────────────────────────────────────────────────
    # Dataset index
    # ─────────────────────────────────────────────────────────────────

    def _episode_names(self) -> list[str]:
        names = []
        for name in sorted(os.listdir(self.dataset_path), reverse=True):
            ep_dir = os.path.join(self.dataset_path, name)
            if os.path.isdir(ep_dir) and os.path.isfile(os.path.join(ep_dir, "meta.json")):
                names.append(name)
        return names

    def list_episodes(
        self,
        offset: int = 0,
        limit: Optional[int] = None,
        lightweight: bool = False,
    ) -> list[dict]:
        """List a slice of episodes.

        ``lightweight`` avoids recursive directory-size walks.  Dataset list
        pagination uses it so initial rendering cost depends on page size, not
        the total number of episodes or extracted frame files.
        """
        episodes = []
        names = self._episode_names()
        start = max(0, int(offset))
        selected_names = names[start:] if limit is None else names[start:start + max(0, int(limit))]
        for name in selected_names:
            ep_dir = os.path.join(self.dataset_path, name)
            meta_path = os.path.join(ep_dir, "meta.json")
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            label_path = os.path.join(ep_dir, "label.json")
            label = {}
            if os.path.exists(label_path):
                try:
                    with open(label_path) as f:
                        label = json.load(f)
                except (OSError, json.JSONDecodeError):
                    label = {}

            # takes 목록 조회
            takes = self._list_takes(ep_dir, include_size=not lightweight)

            episode = {
                "episode_id": name,
                "task": meta.get("task", {}).get("instruction", ""),
                "created_at": meta.get("created_at", ""),
                "success": label.get("success"),
                "takes": takes,
                "takes_count": len(takes),
            }
            if not lightweight:
                episode["size_mb"] = self._dir_size_mb(ep_dir)
            episodes.append(episode)
        return episodes

    def count_episodes(self) -> int:
        return len(self._episode_names())

    def _list_takes(self, ep_dir: str, include_size: bool = True) -> list[dict]:
        takes_dir = os.path.join(ep_dir, "takes")
        if not os.path.isdir(takes_dir):
            return []
        takes = []
        for take_name in sorted(os.listdir(takes_dir)):
            take_dir = os.path.join(takes_dir, take_name)
            if not os.path.isdir(take_dir):
                continue
            has_video = os.path.exists(os.path.join(take_dir, "video.mp4"))
            has_teach = os.path.exists(os.path.join(take_dir, "teach_joint.csv"))
            manifest = load_manifest()
            episode_id = os.path.basename(ep_dir)
            camera_videos = {}
            for cam in manifest.get("cameras", []):
                streams = []
                for stream in cam.get("streams", ["color"]):
                    video_file = video_file_name(cam, stream)
                    if not os.path.exists(os.path.join(take_dir, video_file)):
                        continue
                    stream_labels = {
                        "color": "Color", "depth": "Depth",
                        "aolp": "AoLP", "dolp": "DoLP", "grid": "2×2 Grid", "stokes": "S1/S2 (bwr)",
                    }
                    streams.append({
                        "id": stream,
                        "label": stream_labels.get(stream, stream.capitalize()),
                        "kind": stream,
                        "video": video_file,
                        "url": f"/api/episodes/{episode_id}/takes/{take_name}/video_camera/{cam['id']}/{stream}",
                        "maskable": stream == "color",
                    })
                if streams:
                    color_stream = next((s for s in streams if s["kind"] == "color"), streams[0])
                    camera_videos[cam["id"]] = {
                        "id": cam["id"],
                        "label": cam.get("label", cam["id"]),
                        "role": cam.get("role", ""),
                        "legacy_id": legacy_id_for(cam),
                        "video": color_stream["video"],
                        "streams": streams,
                    }
            take_info = {
                "take":        take_name,
                "has_video":   has_video,
                "has_teach":   has_teach,
                "has_webcam_0": os.path.exists(os.path.join(take_dir, "video_webcam_0.mp4")),
                "has_webcam_1": os.path.exists(os.path.join(take_dir, "video_webcam_1.mp4")),
                "cameras":      list(camera_videos.values()),
                "review_issues": self._load_review_issues(take_dir),
            }
            if include_size:
                take_info["size_mb"] = self._dir_size_mb(take_dir)
            take_info["review_issue_count"] = len(take_info["review_issues"])
            takes.append(take_info)
        return takes

    @staticmethod
    def _load_review_issues(take_dir: str) -> list[dict]:
        path = os.path.join(take_dir, "review_issues.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(data, dict):
            issues = data.get("issues", [])
        else:
            issues = data
        if not isinstance(issues, list):
            return []
        return [i for i in issues if isinstance(i, dict)]

    def get_episode_meta(self, episode_id: str) -> Optional[dict]:
        meta_path = os.path.join(self.dataset_path, episode_id, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path) as f:
            return json.load(f)

    def get_latest_take_dir(self, episode_id: str) -> Optional[str]:
        """특정 episode의 최신 take 디렉터리 반환."""
        takes_dir = os.path.join(self.dataset_path, episode_id, "takes")
        if not os.path.isdir(takes_dir):
            return None
        takes = sorted([d for d in os.listdir(takes_dir) if d.startswith("take_")])
        if not takes:
            return None
        return os.path.join(takes_dir, takes[-1])

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _dir_size_mb(path: str) -> float:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return round(total / (1024 * 1024), 1)

    def update_episode_task(self, episode_id: str, task: str) -> bool:
        ep_dir = os.path.join(self.dataset_path, episode_id)
        meta_path = os.path.join(ep_dir, "meta.json")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            meta = json.load(f)
        if "task" not in meta or not isinstance(meta["task"], dict):
            meta["task"] = {}
        meta["task"]["instruction"] = task
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return True

    def delete_episode(self, episode_id: str) -> bool:
        ep_dir = os.path.join(self.dataset_path, episode_id)
        if not os.path.isdir(ep_dir):
            return False
        shutil.rmtree(ep_dir)
        print(f"[EpisodeManager] Deleted episode: {episode_id}")
        return True

    def delete_take(self, episode_id: str, take: str) -> bool:
        take_dir = os.path.join(self.dataset_path, episode_id, "takes", take)
        if not os.path.isdir(take_dir):
            return False
        shutil.rmtree(take_dir)
        print(f"[EpisodeManager] Deleted take: {episode_id}/{take}")
        return True

    def _write_episode_json(self, filename: str, data: dict):
        """episode 루트 디렉터리에 JSON 파일 쓰기."""
        path = os.path.join(self.current_episode_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _write_csv(path: str, fields: list[str], rows: list[dict]) -> int:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def compute_tracking_quality(self) -> dict:
        """
        replay_command.csv vs executed_joint.csv를 timestamp 기준으로 정렬해
        관절 추종 오차를 계산하고 quality.json으로 저장한다.
        save_episode() 직후 호출.
        """
        if not self.current_take_dir:
            return {}
        cmd_path = os.path.join(self.current_take_dir, "replay_command.csv")
        exe_path = os.path.join(self.current_take_dir, "executed_joint.csv")
        if not (os.path.exists(cmd_path) and os.path.exists(exe_path)):
            return {"ok": False, "reason": "missing CSVs"}

        cmd_rows = self._read_csv(cmd_path)
        exe_rows = self._read_csv(exe_path)
        if not cmd_rows or not exe_rows:
            return {"ok": False, "reason": "empty CSV"}

        try:
            exe_ts = np.array([int(r["t_host_ns"]) for r in exe_rows], dtype=np.int64)
        except (KeyError, ValueError):
            return {"ok": False, "reason": "malformed t_host_ns in executed_joint.csv"}

        errors = []
        for row in cmd_rows:
            try:
                t = int(row["t_host_ns"])
            except (KeyError, ValueError):
                continue
            idx = int(np.searchsorted(exe_ts, t))
            # 앞뒤 중 더 가까운 샘플 선택
            candidates = [c for c in [idx - 1, idx] if 0 <= c < len(exe_rows)]
            if not candidates:
                continue
            idx = min(candidates, key=lambda k: abs(exe_ts[k] - t))
            try:
                err = max(
                    abs(float(row.get(f"q{i+1}", 0)) - float(exe_rows[idx].get(f"q{i+1}", 0)))
                    for i in range(DOF)
                )
            except (ValueError, TypeError):
                continue
            errors.append(err)

        if not errors:
            return {"ok": False, "reason": "no aligned rows"}

        THRESHOLD = 0.08  # rad (~4.6°)
        max_err = float(np.max(errors))
        mean_err = float(np.mean(errors))
        try:
            cmd_ts = np.array(
                [int(row["t_host_ns"]) for row in cmd_rows], dtype=np.int64
            )
            command_intervals_ms = np.diff(cmd_ts) * 1e-6
            interval_p95_ms = float(np.percentile(command_intervals_ms, 95))
            interval_max_ms = float(np.max(command_intervals_ms))
        except (KeyError, ValueError, TypeError):
            interval_p95_ms = float("nan")
            interval_max_ms = float("nan")
        lateness_values = []
        for row in cmd_rows:
            try:
                lateness_values.append(float(row.get("lateness_ms", "")))
            except (TypeError, ValueError):
                continue
        lateness_p95_ms = (
            float(np.percentile(lateness_values, 95)) if lateness_values else None
        )
        lateness_max_ms = float(np.max(lateness_values)) if lateness_values else None
        timing_ok = (
            np.isfinite(interval_p95_ms) and interval_p95_ms <= 30.0
            and np.isfinite(interval_max_ms) and interval_max_ms <= 100.0
        )
        quality = {
            "max_tracking_error_rad":  round(max_err, 4),
            "mean_tracking_error_rad": round(mean_err, 4),
            "command_interval_p95_ms": round(interval_p95_ms, 2),
            "command_interval_max_ms": round(interval_max_ms, 2),
            "command_lateness_p95_ms": (
                round(lateness_p95_ms, 2) if lateness_p95_ms is not None else None
            ),
            "command_lateness_max_ms": (
                round(lateness_max_ms, 2) if lateness_max_ms is not None else None
            ),
            "command_timing_ok": bool(timing_ok),
            "tracking_ok":       max_err < THRESHOLD,
            "save_for_training": max_err < THRESHOLD and timing_ok,
        }
        path = os.path.join(self.current_take_dir, "quality.json")
        with open(path, "w") as f:
            json.dump(quality, f, indent=2)
        print(
            f"[EpisodeManager] quality.json: "
            f"max={max_err:.4f}rad mean={mean_err:.4f}rad "
            f"ok={quality['tracking_ok']}"
        )
        return quality

    @staticmethod
    def _read_csv(path: str) -> list[dict]:
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
