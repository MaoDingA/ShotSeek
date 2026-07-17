"""FFmpeg scene detection and complete half-open shot-grid generation."""

from __future__ import annotations

import re
import subprocess
from fractions import Fraction
from pathlib import Path

from shotseek.m0 import ensure_within_project
from shotseek.media.schema import Shot, ShotBoundary, VideoContract
from shotseek.media.timebase import frame_to_ms, pts_to_frame

FRAME_RE = re.compile(r"frame:\s*\d+\s+pts:\s*(\d+)\s+pts_time:[^\s]+")
SCORE_RE = re.compile(r"lavfi\.scene_score=([0-9.]+)")


def detect_shot_boundaries(
    project_root: Path,
    video_path: Path,
    video: VideoContract,
    *,
    threshold: float = 0.30,
    min_gap_frames: int = 6,
) -> list[ShotBoundary]:
    if not 0.0 < threshold < 1.0 or min_gap_frames < 1:
        raise ValueError("invalid shot detector parameters")
    root = project_root.resolve()
    resolved = ensure_within_project(root, video_path)
    filter_value = f"select='gt(scene,{threshold:.6f})',metadata=print"
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(resolved),
            "-an", "-vf", filter_value, "-f", "null", "-",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    fps = Fraction(video.fps_num, video.fps_den)
    time_base = Fraction(video.time_base_num, video.time_base_den)
    pending_pts: int | None = None
    candidates: list[tuple[int, int, float]] = []
    for line in completed.stderr.splitlines():
        frame_match = FRAME_RE.search(line)
        if frame_match:
            pending_pts = int(frame_match.group(1))
            continue
        score_match = SCORE_RE.search(line)
        if score_match and pending_pts is not None:
            frame = pts_to_frame(pending_pts, time_base, fps)
            candidates.append((frame, pending_pts, float(score_match.group(1))))
            pending_pts = None

    filtered: list[tuple[int, int, float]] = []
    for frame, pts, score in candidates:
        if frame <= 0 or frame >= video.frame_count:
            continue
        if filtered and frame - filtered[-1][0] < min_gap_frames:
            if score > filtered[-1][2]:
                filtered[-1] = (frame, pts, score)
            continue
        filtered.append((frame, pts, score))
    return [
        ShotBoundary(
            boundary_id=f"boundary_{index:04d}",
            frame=frame,
            timestamp_ms=frame_to_ms(frame, fps),
            pts=pts,
            scene_score=score,
            detector="ffmpeg_scene",
        )
        for index, (frame, pts, score) in enumerate(filtered, start=1)
    ]


def build_shot_grid(
    video: VideoContract, boundaries: list[ShotBoundary]
) -> list[Shot]:
    fps = Fraction(video.fps_num, video.fps_den)
    frames = [0, *(item.frame for item in boundaries), video.frame_count]
    if frames != sorted(set(frames)):
        raise ValueError("shot boundaries must be unique and sorted")
    shots = [
        Shot(
            shot_id=f"shot_{index:04d}",
            start_frame=start,
            end_frame=end,
            start_ms=frame_to_ms(start, fps),
            end_ms=frame_to_ms(end, fps),
            duration_frames=end - start,
        )
        for index, (start, end) in enumerate(zip(frames, frames[1:]), start=1)
    ]
    validate_shot_grid(shots, video.frame_count)
    return shots


def validate_shot_grid(shots: list[Shot], frame_count: int) -> None:
    errors: list[str] = []
    if not shots:
        errors.append("shot grid is empty")
    elif shots[0].start_frame != 0 or shots[-1].end_frame != frame_count:
        errors.append("shot grid does not cover the full video")
    for previous, current in zip(shots, shots[1:]):
        if previous.end_frame != current.start_frame:
            errors.append(f"gap or overlap between {previous.shot_id} and {current.shot_id}")
    if any(item.duration_frames <= 0 for item in shots):
        errors.append("shot grid contains a zero-frame shot")
    if errors:
        raise ValueError("; ".join(errors))
