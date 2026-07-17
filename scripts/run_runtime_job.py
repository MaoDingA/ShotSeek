#!/usr/bin/env python3
"""Run one reproducible Production Runtime job and emit a safe report."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.runtime import JobState, RuntimePaths, RuntimeRegistry, RuntimeWorker, store_upload
from shotseek.runtime.pipeline import PipelineSettings, ProductionPipeline


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stage_durations(events: list[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    index = 0
    while index < len(events) - 1:
        current = events[index]
        following_index = index + 1
        while (
            following_index < len(events)
            and events[following_index].state == current.state
        ):
            following_index += 1
        if following_index >= len(events):
            break
        following = events[following_index]
        try:
            started = datetime.fromisoformat(current.created_at)
            finished = datetime.fromisoformat(following.created_at)
        except ValueError:
            index = following_index
            continue
        seconds = max(0.0, (finished - started).total_seconds())
        result[current.state.value] = result.get(current.state.value, 0.0) + seconds
        index = following_index
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("live", "fixture"), default="fixture")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--chunk-duration-seconds",
        type=int,
        default=30,
        choices=range(1, 61),
        metavar="1-60",
    )
    parser.add_argument(
        "--vision-workers",
        type=int,
        default=3,
        choices=range(1, 5),
        metavar="1-4",
    )
    parser.add_argument("--proxy-passthrough", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source = ensure_within_project(project_root, args.video)
    runtime_root = ensure_within_project(project_root, args.runtime_root)
    if not source.is_file():
        raise SystemExit(f"video not found: {source}")
    api_key = os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")
    settings = PipelineSettings(
        mode=args.mode,
        api_key=api_key,
        chunk_duration_ms=args.chunk_duration_seconds * 1_000,
        vision_workers=args.vision_workers,
        proxy_passthrough=args.proxy_passthrough,
    )
    paths = RuntimePaths(project_root, runtime_root)
    paths.ensure()
    registry = RuntimeRegistry(paths.registry)

    with source.open("rb") as handle:
        stored = store_upload(paths, handle, source.name)
    video, _ = registry.register_video(
        sha256=stored.sha256,
        original_filename=stored.original_filename,
        source_path=str(stored.path.relative_to(project_root)),
        bytes=stored.bytes,
    )
    active = registry.active_job_for_video(video.video_id)
    latest = registry.latest_job_for_video(video.video_id)
    if active is not None:
        job = active
    elif latest is not None and latest.state == JobState.READY:
        job = latest
    else:
        job = registry.create_job(video.video_id)
        registry.update_video(video.video_id, status="PROCESSING")
        job = registry.transition(
            job.job_id,
            JobState.QUEUED,
            message="命令行运行等待 Worker",
        )

    started = time.perf_counter()
    if job.state not in {JobState.READY, JobState.PARTIAL}:
        pipeline = ProductionPipeline(
            paths=paths,
            registry=registry,
            settings=settings,
        )
        completed = RuntimeWorker(
            registry,
            pipeline,
            max_retries=args.max_retries,
        ).run_once()
        if completed is not None:
            job = completed
    elapsed = time.perf_counter() - started

    video = registry.get_video(video.video_id)
    events = registry.events(job.job_id)
    artifacts = registry.list_artifacts(job.video_id)
    report = {
        "schema_version": "m3-runtime-run-report-v1",
        "mode": args.mode,
        "status": job.state.value,
        "job_id": job.job_id,
        "video_id": job.video_id,
        "pipeline": {
            "chunk_duration_ms": settings.chunk_duration_ms,
            "vision_workers": settings.vision_workers,
            "proxy_passthrough": settings.proxy_passthrough,
        },
        "video": {
            "sha256": video.sha256 if video else stored.sha256,
            "duration_ms": video.duration_ms if video else None,
            "scene_count": video.scene_count if video else 0,
            "status": video.status if video else None,
        },
        "elapsed_s": round(elapsed, 3),
        "retry_count": job.retry_count,
        "error_code": job.error_code,
        "message": job.message,
        "stage_durations_s": stage_durations(events),
        "event_count": len(events),
        "artifacts": [
            {
                "kind": item.kind,
                "status": item.status,
                "provider": item.provider,
                "model": item.model,
            }
            for item in artifacts
        ],
        "registry": registry.diagnostics(),
    }
    report_path = runtime_root / "run_report.json"
    dump_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if job.state not in {JobState.READY, JobState.PARTIAL}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
