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
