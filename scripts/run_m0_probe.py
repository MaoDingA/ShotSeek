#!/usr/bin/env python3
"""Command-line entry point for the ShotSeek M0 contract probe."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shotseek.m0 import run_probe
from shotseek.providers.stepfun import (
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_VISION_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ShotSeek M0 contract probe")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Call the live StepFun APIs")
    mode.add_argument(
        "--fixture", action="store_true", help="Run deterministically without network or API key"
    )
    parser.add_argument("--video", type=Path, required=True, help="MP4 path inside this repository")
    parser.add_argument("--audio-url", help="Public audio URL required by StepAudio file ASR")
    return parser.parse_args()


def load_project_env() -> None:
    """Load simple KEY=VALUE entries from the ignored project .env file."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"\"", "'"}:
            cleaned = cleaned[1:-1]
        os.environ.setdefault(key.strip(), cleaned)


def main() -> int:
    args = parse_args()
    mode = "live" if args.live else "fixture"
    video = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video

    if mode == "fixture":
        run_dir = run_probe(
            project_root=PROJECT_ROOT,
            video_path=video,
            mode="fixture",
        )
    else:
        load_project_env()
        api_key = os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")
        audio_url = args.audio_url or os.environ.get("GOLDEN_AUDIO_URL")
        run_dir = run_probe(
            project_root=PROJECT_ROOT,
            video_path=video,
            mode="live",
            api_key=api_key,
            audio_url=audio_url,
            base_url=os.environ.get("STEPFUN_BASE_URL", DEFAULT_BASE_URL),
            vision_model=os.environ.get("STEPFUN_VISION_MODEL", DEFAULT_VISION_MODEL),
            asr_model=os.environ.get("STEPFUN_ASR_MODEL", DEFAULT_ASR_MODEL),
        )

    print(run_dir.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
