#!/usr/bin/env python3
"""Audit the 2026-08-27 LeRobot batch and optionally apply review decisions.

The archive is the authority for LeRobot episode-index -> source mapping.  The
known skipped source episode is added explicitly so the report always covers
the requested 106 recordings.  Applying decisions is non-destructive: timing
and trim choices go to edit_meta.json, while unusable takes are marked as
default-export-excluded in replay_result.json.  Raw recordings are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


GRIPPER_MID = 0.02219
REVIEWED_FAST = {
    1, 2, 3, 6, 11, 13, 14, 16, 18, 20, 21, 31, 36, 40, 44, 47,
    56, 71, 80, 87, 88, 89, 91, 92, 94, 95, 96, 97, 99, 100, 101, 103,
}
REVIEWED_EXCLUDE = {
    26: "reviewed failure",
    31: "camera framing cropped",
    65: "task-unrelated motion included",
    77: "task-unrelated motion included",
}
EXPLICIT_TRIM = {1: 0.5, 3: 0.5}
KNOWN_MISSING_EPISODE = "episode_20260827_152724"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error):
        return []


def _duration(rows: list[dict[str, str]]) -> float | None:
    if len(rows) < 2:
        return None
    try:
        value = (float(rows[-1]["t_host_ns"]) - float(rows[0]["t_host_ns"])) / 1e9
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _percentile(values: list[float], percentile: float) -> float | None:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    index = (len(finite) - 1) * percentile
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return finite[lo]
    return finite[lo] * (hi - index) + finite[hi] * (index - lo)


def _joint_speed(rows: list[dict[str, str]]) -> tuple[float | None, float | None]:
    speeds: list[float] = []
    for before, after in zip(rows, rows[1:]):
        try:
            dt = (float(after["t_host_ns"]) - float(before["t_host_ns"])) / 1e9
            if dt <= 0:
                continue
            speeds.append(max(abs(float(after[f"q{i}"]) - float(before[f"q{i}"])) / dt for i in range(1, 7)))
        except (KeyError, TypeError, ValueError):
            continue
    return _percentile(speeds, 0.95), max(speeds) if speeds else None


def _gripper_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {"open_t_s": None, "tail_s": None, "transitions": 0, "arm_motion_after_open_rad": None}
    try:
        t0 = float(rows[0]["t_host_ns"])
    except (KeyError, TypeError, ValueError):
        return {"open_t_s": None, "tail_s": None, "transitions": 0, "arm_motion_after_open_rad": None}
    previous: bool | None = None
    transitions = 0
    open_index: int | None = None
    for index, row in enumerate(rows):
        try:
            opened = float(row["gripper"]) > GRIPPER_MID
            t = (float(row["t_host_ns"]) - t0) / 1e9
        except (KeyError, TypeError, ValueError):
            continue
        if previous is not None and opened != previous:
            transitions += 1
        if previous is False and opened and t > 10.0:
            open_index = index
            break
        previous = opened
    if open_index is None:
        return {"open_t_s": None, "tail_s": None, "transitions": transitions, "arm_motion_after_open_rad": None}
    open_row = rows[open_index]
    open_t = (float(open_row["t_host_ns"]) - t0) / 1e9
    end_t = (float(rows[-1]["t_host_ns"]) - t0) / 1e9
    motion = 0.0
    try:
        start_q = [float(open_row[f"q{i}"]) for i in range(1, 7)]
        for row in rows[open_index:]:
            motion = max(motion, max(abs(float(row[f"q{i}"]) - start_q[i - 1]) for i in range(1, 7)))
    except (KeyError, TypeError, ValueError):
        motion = math.nan
    return {
        "open_t_s": round(open_t, 6),
        "tail_s": round(max(0.0, end_t - open_t), 6),
        "transitions": transitions,
        "arm_motion_after_open_rad": round(motion, 6) if math.isfinite(motion) else None,
    }


def _pick_take(ep_dir: Path, preferred: str | None) -> tuple[str | None, Path | None]:
    takes = ep_dir / "takes"
    candidates = []
    if preferred:
        candidates.append(preferred)
    if takes.is_dir():
        candidates.extend(sorted((p.name for p in takes.iterdir() if p.is_dir()), reverse=True))
    for name in dict.fromkeys(candidates):
        take_dir = takes / name
        if (take_dir / "executed_joint.csv").exists() or (take_dir / "teach_joint.csv").exists():
            return name, take_dir
    return None, None


def _load_source_map(archive: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(archive) as zipped:
        lines = zipped.read("lerobot/meta/source_episodes.jsonl").decode().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _apply(row: dict[str, Any], take_dir: Path) -> None:
    index = row.get("archive_episode_index")
    edit_path = take_dir / "edit_meta.json"
    edit = _read_json(edit_path)
    edit.setdefault("trim", {"enabled": False, "cut_t": None, "margin": 1.0})
    edit.setdefault("mask", {"enabled": False, "camera_id": "cam1", "polygon": [], "fill": "black"})
    edit.setdefault("timing", {"mode": "teaching_normalized"})
    edit.setdefault("export", {})["original_episode_index"] = (
        index if index is not None else 105
    )
    if index in REVIEWED_FAST and index not in REVIEWED_EXCLUDE:
        edit["timing"]["mode"] = "physical_replay"
    if index in EXPLICIT_TRIM and row.get("gripper_open_t_s") is not None:
        edit["trim"] = {
            "enabled": True,
            "cut_t": row["gripper_open_t_s"],
            "margin": EXPLICIT_TRIM[index],
        }
    _write_json(edit_path, edit)

    if index in REVIEWED_EXCLUDE:
        result_path = take_dir / "replay_result.json"
        result = _read_json(result_path)
        result["export_excluded_default"] = True
        result["export_exclusion_reason"] = REVIEWED_EXCLUDE[index]
        result["export_reviewed_at"] = datetime.now().isoformat()
        _write_json(result_path, result)


def audit(dataset: Path, archive: Path, apply: bool) -> list[dict[str, Any]]:
    source_map = _load_source_map(archive)
    specs = [
        (entry.get("episode_index"), entry["source_episode_id"], entry.get("source_take"), entry)
        for entry in source_map
    ]
    mapped = {spec[1] for spec in specs}
    if KNOWN_MISSING_EPISODE not in mapped:
        specs.append((None, KNOWN_MISSING_EPISODE, "take_001", {}))

    report: list[dict[str, Any]] = []
    for index, episode_id, preferred_take, source in specs:
        take, take_dir = _pick_take(dataset / episode_id, preferred_take)
        row: dict[str, Any] = {
            "archive_episode_index": index,
            "episode_id": episode_id,
            "take": take,
            "in_archive": index is not None,
            "reviewed_fast": index in REVIEWED_FAST,
            "reviewed_excluded": index in REVIEWED_EXCLUDE,
            "exclusion_reason": REVIEWED_EXCLUDE.get(index),
        }
        if take_dir is None:
            row["hazards"] = ["missing_joint_data"]
            report.append(row)
            continue
        executed = _rows(take_dir / "executed_joint.csv")
        teach = _rows(take_dir / "teach_joint.csv")
        command = _rows(take_dir / "replay_command.csv")
        source_video_duration = source.get("full_video_duration_s")
        physical_duration = _duration(executed) or _duration(teach)
        teach_duration = _duration(teach)
        # replay_command is the clean 50 Hz target stream.  executed_joint is
        # higher-rate feedback and differentiating its timestamp jitter greatly
        # exaggerates speed, so only use it as a fallback.
        p95, maximum = _joint_speed(command or executed or teach)
        grip = _gripper_metrics(executed or teach)
        ratio = (
            physical_duration / float(source_video_duration)
            if physical_duration and source_video_duration and float(source_video_duration) > 0
            else None
        )
        hazards: list[str] = []
        if index in REVIEWED_FAST:
            hazards.append("reviewed_too_fast_or_completion_differs")
        if index in REVIEWED_EXCLUDE:
            hazards.append("exclude_from_curated_export")
        if grip["open_t_s"] is None:
            hazards.append("no_release_transition_after_10s")
        elif grip["tail_s"] is not None and grip["tail_s"] > 2.0:
            hazards.append("long_tail_after_release")
        if ratio is not None and ratio > 1.5:
            hazards.append("speed_normalization_applied_gt_1_5x")
        apparent_p95 = p95 * ratio if p95 is not None and ratio is not None else None
        row.update({
            "source_video_duration_s": source_video_duration,
            "physical_duration_s": round(physical_duration, 6) if physical_duration else None,
            "teach_duration_s": round(teach_duration, 6) if teach_duration else None,
            "command_duration_s": round(_duration(command), 6) if _duration(command) else None,
            "physical_to_export_ratio": round(ratio, 6) if ratio else None,
            "export_to_teach_duration_ratio": round(float(source_video_duration) / teach_duration, 6) if source_video_duration and teach_duration else None,
            "physical_joint_speed_p95_rad_s": round(p95, 6) if p95 else None,
            "physical_joint_speed_max_rad_s": round(maximum, 6) if maximum else None,
            "normalized_export_speed_p95_rad_s": round(apparent_p95, 6) if apparent_p95 else None,
            "gripper_open_t_s": grip["open_t_s"],
            "tail_after_open_s": grip["tail_s"],
            "gripper_transitions_before_release": grip["transitions"],
            "arm_motion_after_open_rad": grip["arm_motion_after_open_rad"],
            "recommended_timing": "physical_replay" if index in REVIEWED_FAST and index not in REVIEWED_EXCLUDE else "teaching_normalized",
            "hazards": hazards,
        })
        report.append(row)
        if apply:
            _apply(row, take_dir)
    speed_values = sorted(
        row["normalized_export_speed_p95_rad_s"]
        for row in report
        if row.get("normalized_export_speed_p95_rad_s") is not None
        and not row.get("reviewed_excluded")
    )
    speed_threshold = _percentile(speed_values, 0.95)
    for row in report:
        value = row.get("normalized_export_speed_p95_rad_s")
        row["speed_outlier_threshold_p95_rad_s"] = round(speed_threshold, 6) if speed_threshold else None
        row["objective_speed_outlier"] = bool(
            value is not None and speed_threshold is not None and value >= speed_threshold
        )
        if row["objective_speed_outlier"]:
            row.setdefault("hazards", []).append("objective_speed_outlier_top_5pct")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", type=Path, default=root / "dataset")
    parser.add_argument("--archive", type=Path, default=root / "piper_lerobot_106ep.zip")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset" / "reviews")
    parser.add_argument("--apply", action="store_true", help="write non-destructive export decisions")
    args = parser.parse_args()

    report = audit(args.dataset, args.archive, args.apply)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "export_20260827_106ep_audit"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "archive": str(args.archive),
        "archive_count": sum(bool(row["in_archive"]) for row in report),
        "audited_count": len(report),
        "applied": args.apply,
        "reviewed_fast_indices": sorted(REVIEWED_FAST),
        "reviewed_excluded_indices": REVIEWED_EXCLUDE,
        "episodes": report,
    }
    _write_json(json_path, payload)
    fields = sorted({key for row in report for key in row})
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in report:
            writer.writerow({**row, "hazards": ";".join(row.get("hazards", []))})
    print(json.dumps({
        "ok": True,
        "archive_count": payload["archive_count"],
        "audited_count": payload["audited_count"],
        "applied": args.apply,
        "json": str(json_path),
        "csv": str(csv_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
