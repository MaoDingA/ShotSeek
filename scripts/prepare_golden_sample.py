#!/usr/bin/env python3
"""Download and prepare the openly licensed M0 Tears of Steel sample."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
SOURCE_URL = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
SOURCE_PATH = SAMPLES_DIR / "tears_of_steel_720p.mov"
VIDEO_PATH = SAMPLES_DIR / "golden.mp4"
AUDIO_PATH = SAMPLES_DIR / "golden.mp3"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [
            "curl",
            "-L",
            "--fail",
            "--show-error",
            "--retry",
            "12",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(SOURCE_PATH),
            SOURCE_URL,
        ]
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            "00:04:52",
            "-i",
            str(SOURCE_PATH),
            "-t",
            "75",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-vf",
            "scale=-2:720",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            "-y",
            str(VIDEO_PATH),
        ]
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(VIDEO_PATH),
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-ac",
            "1",
            "-y",
            str(AUDIO_PATH),
        ]
    )
    print(VIDEO_PATH.relative_to(PROJECT_ROOT))
    print(AUDIO_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
