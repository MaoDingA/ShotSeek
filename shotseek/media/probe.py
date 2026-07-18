"""FFprobe-backed video contract with explicit CFR rejection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from shotseek.media.schema import VideoContract
from shotseek.media.timebase import frame_to_ms, parse_ratio
from shotseek.m0 import ensure_within_project


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video_contract(project_root: Path, video_path: Path) -> VideoContract:
    root = project_root.resolve()
    resolved = ensure_within_project(root, video_path)
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,duration,duration_ts,nb_frames",
            "-of", "json", str(resolved),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    avg_fps = parse_ratio(stream["avg_frame_rate"])
    nominal_fps = parse_ratio(stream["r_frame_rate"])
    if avg_fps != nominal_fps:
        raise ValueError(
            f"M1 supports CFR video only: avg={avg_fps}, nominal={nominal_fps}"
        )
    frame_count_raw = stream.get("nb_frames")
    if frame_count_raw in {None, "", "N/A"}:
        raise ValueError("M1 requires a known CFR frame count")
    frame_count = int(frame_count_raw)
    expected_duration_ms = frame_to_ms(frame_count, avg_fps)
    stream_duration = stream.get("duration")
    if stream_duration not in {None, "", "N/A"}:
        measured_duration_ms = int(round(float(stream_duration) * 1000))
    elif stream.get("duration_ts") not in {None, "", "N/A"}:
        measured_duration_ms = frame_to_ms(
            int(stream["duration_ts"]),
            1 / parse_ratio(stream["time_base"]),
        )
    else:
        # A longer audio stream or codec padding may extend the container.
        # CFR frame count and rate define the video timeline exactly.
        measured_duration_ms = expected_duration_ms
    frame_duration_ms = float(1000 / avg_fps)
    if abs(measured_duration_ms - expected_duration_ms) > frame_duration_ms:
        raise ValueError("video stream duration is inconsistent with CFR frame count")
    time_base = parse_ratio(stream["time_base"])
    return VideoContract(
        path=str(resolved.relative_to(root)),
        sha256=_sha256(resolved),
        bytes=int(payload["format"].get("size") or resolved.stat().st_size),
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_ms=expected_duration_ms,
        frame_count=frame_count,
        fps_num=avg_fps.numerator,
        fps_den=avg_fps.denominator,
        time_base_num=time_base.numerator,
        time_base_den=time_base.denominator,
        video_codec=str(stream["codec_name"]),
        cfr=True,
    )
