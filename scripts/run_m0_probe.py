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
    DEFAULT_ASR_BASE_URL,
    DEFAULT_ASR_MODEL,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_FILES_BASE_URL,
    DEFAULT_SSE_ASR_BASE_URL,
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
    parser.add_argument(
        "--video-chunks",
        type=Path,
        help="Optional project-local JSON manifest of contiguous <=10s public video URLs",
    )
    parser.add_argument(
        "--vision-cache-run",
        type=Path,
        help="Optional project-local live run whose visual evidence matches --video",
    )
    parser.add_argument(
        "--asr-transport",
        choices=("async_file", "sse"),
        default="async_file",
        help="Use timestamped Step Plan SSE or standard async file ASR",
    )
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
    video_chunks = (
        args.video_chunks
        if args.video_chunks is None or args.video_chunks.is_absolute()
        else PROJECT_ROOT / args.video_chunks
    )
    vision_cache_run = (
        args.vision_cache_run
        if args.vision_cache_run is None or args.vision_cache_run.is_absolute()
        else PROJECT_ROOT / args.vision_cache_run
    )

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
        legacy_base_url = os.environ.get("STEPFUN_BASE_URL")
        run_dir = run_probe(
            project_root=PROJECT_ROOT,
            video_path=video,
            mode="live",
            api_key=api_key,
            audio_url=audio_url,
            video_chunks_path=video_chunks,
            vision_cache_run=vision_cache_run,
            asr_transport=args.asr_transport,
            files_base_url=os.environ.get(
                "STEPFUN_FILES_BASE_URL", DEFAULT_FILES_BASE_URL
            ),
            chat_base_url=os.environ.get(
                "STEPFUN_CHAT_BASE_URL", legacy_base_url or DEFAULT_CHAT_BASE_URL
            ),
            asr_base_url=os.environ.get(
                "STEPFUN_ASR_BASE_URL", DEFAULT_ASR_BASE_URL
            ),
            sse_asr_base_url=os.environ.get(
                "STEPFUN_SSE_ASR_BASE_URL", DEFAULT_SSE_ASR_BASE_URL
            ),
            vision_model=os.environ.get("STEPFUN_VISION_MODEL", DEFAULT_VISION_MODEL),
            asr_model=os.environ.get("STEPFUN_ASR_MODEL", DEFAULT_ASR_MODEL),
        )

    print(run_dir.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
