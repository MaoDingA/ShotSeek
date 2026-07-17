"""Command-line entry point for the ShotSeek Production Runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from shotseek.runtime.api import create_runtime_app
from shotseek.runtime.pipeline import PipelineSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ShotSeek Production Runtime")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("live", "fixture"),
        default=os.environ.get("SHOTSEEK_RUNTIME_MODE", "fixture"),
        help="fixture is deterministic and visibly cached; live calls StepFun",
    )
    parser.add_argument(
        "--chunk-duration-seconds",
        type=int,
        choices=range(1, 61),
        default=int(os.environ.get("SHOTSEEK_CHUNK_DURATION_SECONDS", "30")),
        metavar="1-60",
    )
    parser.add_argument(
        "--vision-workers",
        type=int,
        choices=range(1, 5),
        default=int(os.environ.get("SHOTSEEK_VISION_WORKERS", "3")),
        metavar="1-4",
    )
    parser.add_argument(
        "--proxy-passthrough",
        action="store_true",
        default=os.environ.get("SHOTSEEK_PROXY_PASSTHROUGH") == "1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--allow-network-query", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    if not (project_root / "pyproject.toml").is_file():
        raise SystemExit(f"not a ShotSeek project root: {project_root}")
    api_key = os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")
    settings = PipelineSettings(
        mode=args.mode,
        api_key=api_key,
        chunk_duration_ms=args.chunk_duration_seconds * 1_000,
        vision_workers=args.vision_workers,
        proxy_passthrough=args.proxy_passthrough,
    )
    app = create_runtime_app(
        project_root=project_root,
        runtime_root=args.runtime_root,
        pipeline_settings=settings,
        search_allow_network=args.allow_network_query,
        search_api_key=api_key,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
