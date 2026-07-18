"""Production Runtime HTTP API for uploads, jobs, events and artifacts."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from shotseek.agent import ShotSeekAgent
from shotseek.export.delivery import ExportFormat, render_export
from shotseek.m0 import ensure_within_project
from shotseek.runtime.paths import RuntimePaths, store_upload_stream
from shotseek.runtime.pipeline import PipelineSettings, ProductionPipeline
from shotseek.runtime.registry import RuntimeRegistry
from shotseek.runtime.schema import JobState, TERMINAL_STATES
from shotseek.runtime.worker import RuntimeWorker, StageExecutor


class RuntimeAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeSearchRequest(RuntimeAPIModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)
    planner_mode: Literal["auto", "rule", "stepfun", "cache"] = "auto"
    verifier_mode: Literal["auto", "rule", "stepfun", "cache"] = "rule"


def _database_scenes(database_path: Path) -> list[dict]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT scene_json, dialogue FROM scene ORDER BY start_ms, scene_id"
        ).fetchall()
    items: list[dict] = []
    for row in rows:
        scene = json.loads(row["scene_json"])
        scene["dialogue"] = row["dialogue"]
        items.append(scene)
    return items


def _load_optional_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if not value:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("unsupported byte range")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise ValueError("invalid byte range")
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("invalid suffix range")
        return max(0, size - length), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("range is outside the file")
    return start, min(end, size - 1)


def _file_chunks(path: Path, start: int, end: int):
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                return
            remaining -= len(block)
            yield block


def _job_payload(registry: RuntimeRegistry, job_id: str) -> dict:
    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    video = registry.get_video(job.video_id)
    return {
        "job": job.model_dump(mode="json"),
        "video": video.model_dump(mode="json") if video else None,
    }


def create_runtime_app(
    *,
    project_root: Path,
    runtime_root: Path | None = None,
    executor: StageExecutor | None = None,
    pipeline_settings: PipelineSettings | None = None,
    start_worker: bool = True,
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024,
    search_allow_network: bool = False,
    search_api_key: str | None = None,
) -> FastAPI:
    paths = RuntimePaths(project_root, runtime_root)
    paths.ensure()
    registry = RuntimeRegistry(paths.registry)
    if executor is None and pipeline_settings is not None:
        executor = ProductionPipeline(
            paths=paths,
            registry=registry,
            settings=pipeline_settings,
        )
    worker = (
        RuntimeWorker(registry, executor) if executor is not None and start_worker else None
    )
    agent_cache: dict[str, ShotSeekAgent] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start(recover=True)
        else:
            registry.recover_incomplete_jobs()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    app = FastAPI(
        title="ShotSeek Production Runtime",
        version="0.3.0",
        description="Project-root locked long-video processing runtime",
        lifespan=lifespan,
    )
    app.state.runtime_paths = paths
    app.state.runtime_registry = registry
    app.state.runtime_worker = worker

    @app.get("/health")
    def health() -> dict:
        diagnostics = registry.diagnostics()
        return {
            "status": "ok" if diagnostics["integrity_check"] == "ok" else "failed",
            "service": "shotseek-runtime",
            "schema_version": "m3-runtime-api-v1",
            "worker_enabled": worker is not None,
            "worker_error": worker.last_error if worker is not None else None,
            "project_root": str(paths.project_root),
            "registry": diagnostics,
        }

    @app.get("/api/v1/videos")
    def list_videos() -> dict:
        items = [item.model_dump(mode="json") for item in registry.list_videos()]
        return {"items": items, "count": len(items)}

    def ready_database(video_id: str) -> tuple[object, Path]:
        video = registry.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")
        if video.status not in {"READY", "PARTIAL"} or not video.search_db_path:
            raise HTTPException(status_code=409, detail=f"video is {video.status}")
        database = ensure_within_project(
            paths.project_root,
            paths.project_root / video.search_db_path,
        )
        if not database.is_file():
            raise HTTPException(status_code=409, detail="search index is unavailable")
        return video, database

    @app.get("/api/v1/videos/{video_id}")
    def get_video(video_id: str) -> dict:
        video = registry.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")
        return video.model_dump(mode="json")

    @app.get("/api/v1/videos/{video_id}/media")
    def get_video_media(
        video_id: str,
        request: Request,
        kind: Literal["proxy", "source"] = Query(default="proxy"),
    ) -> StreamingResponse:
        video = registry.get_video(video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")
        relative = video.proxy_path if kind == "proxy" else video.source_path
        if not relative:
            raise HTTPException(status_code=409, detail=f"{kind} media is unavailable")
        media_path = ensure_within_project(
            paths.project_root,
            paths.project_root / relative,
        )
        if not media_path.is_file():
            raise HTTPException(status_code=404, detail="media file not found")
        size = media_path.stat().st_size
        try:
            requested = _byte_range(request.headers.get("range"), size)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=416,
                detail="range not satisfiable",
                headers={"Content-Range": f"bytes */{size}"},
            )
        start, end = requested or (0, size - 1)
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Cache-Control": "private, max-age=3600",
        }
        status_code = 200
        if requested is not None:
            status_code = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        media_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        return StreamingResponse(
            _file_chunks(media_path, start, end),
            status_code=status_code,
            media_type=media_type,
            headers=headers,
        )

    @app.get("/api/v1/videos/{video_id}/export")
    def export_scenes(
        video_id: str,
        format: ExportFormat = Query(),
        scene_id: list[str] = Query(default=[]),
    ) -> Response:
        video, database = ready_database(video_id)
        scenes = _database_scenes(database)
        if scene_id:
            requested = set(scene_id)
            selected = [scene for scene in scenes if scene["scene_id"] in requested]
            found = {scene["scene_id"] for scene in selected}
            missing = sorted(requested - found)
            if missing:
                raise HTTPException(
                    status_code=404,
                    detail={"message": "scene not found", "scene_ids": missing},
                )
            scenes = selected
        if video.fps is None:
            raise HTTPException(status_code=409, detail="video frame rate is unavailable")
        try:
            document = render_export(
                format,
                scenes,
                video_id=video.video_id,
                source_name=video.original_filename,
                fps=video.fps,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        filename = f"shotseek-{video.video_id}.{document.extension}"
        return Response(
            document.content,
            media_type=document.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-ShotSeek-Scene-Count": str(len(scenes)),
            },
        )

    @app.get("/api/v1/videos/{video_id}/scenes")
    def list_scenes(video_id: str) -> dict:
        _, database = ready_database(video_id)
        items = _database_scenes(database)
        return {"items": items, "count": len(items)}

    @app.get("/api/v1/videos/{video_id}/scenes/{scene_id}")
    def get_scene(video_id: str, scene_id: str) -> dict:
        _, database = ready_database(video_id)
        item = next(
            (scene for scene in _database_scenes(database) if scene["scene_id"] == scene_id),
            None,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="scene not found")
        return item

    @app.get("/api/v1/videos/{video_id}/scenes/{scene_id}/preview")
    def get_scene_preview(video_id: str, scene_id: str) -> FileResponse:
        _, database = ready_database(video_id)
        exists = any(
            scene["scene_id"] == scene_id for scene in _database_scenes(database)
        )
        if not exists:
            raise HTTPException(status_code=404, detail="scene not found")
        preview = ensure_within_project(
            paths.project_root,
            paths.video_root(video_id) / "previews" / f"{scene_id}.jpg",
        )
        if not preview.is_file():
            raise HTTPException(status_code=404, detail="scene preview not found")
        return FileResponse(
            preview,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @app.get("/api/v1/videos/{video_id}/scenes/{scene_id}/evidence")
    def get_scene_evidence(video_id: str, scene_id: str) -> dict:
        _, database = ready_database(video_id)
        scene = next(
            (item for item in _database_scenes(database) if item["scene_id"] == scene_id),
            None,
        )
        if scene is None:
            raise HTTPException(status_code=404, detail="scene not found")
        root = paths.video_root(video_id)
        aligned_path = root / "timeline" / "aligned_visual_events.json"
        utterances_path = root / "timeline" / "contextualized_utterances.json"
        aligned = _load_optional_json(aligned_path, [])
        utterances = _load_optional_json(utterances_path, [])
        visual = next(
            (
                item
                for item in aligned
                if item.get("event_id") == scene.get("visual_event_id")
            ),
            None,
        )
        dialogue_ids = set(scene.get("utterance_ids") or [])
        dialogue = [
            item for item in utterances if item.get("utterance_id") in dialogue_ids
        ]
        return {
            "schema_version": "m3-evidence-drawer-v1",
            "scene_id": scene_id,
            "visual": visual,
            "dialogue": dialogue,
            "boundary": {
                "strategy": visual.get("boundary_strategy") if visual else None,
                "raw_start_ms": visual.get("raw_global_start_ms") if visual else None,
                "raw_end_ms": visual.get("raw_global_end_ms") if visual else None,
                "final_start_ms": scene["start_ms"],
                "final_end_ms": scene["end_ms"],
                "start_delta_frames": visual.get("start_delta_frames") if visual else None,
                "end_delta_frames": visual.get("end_delta_frames") if visual else None,
            },
            "evidence_refs": scene.get("evidence_refs", []),
        }

    @app.post("/api/v1/videos/{video_id}/search")
    def search(video_id: str, request: RuntimeSearchRequest) -> dict:
        _, database = ready_database(video_id)
        agent = agent_cache.get(video_id)
        if agent is None:
            root = paths.video_root(video_id)
            agent = ShotSeekAgent(
                database_path=database,
                planner_cache_dir=paths.root / "cache" / "planner",
                verifier_cache_dir=paths.root / "cache" / "verifier",
                trace_dir=root / "traces",
            )
            agent_cache[video_id] = agent
        try:
            result = agent.search(
                request.query,
                top_k=request.top_k,
                planner_mode=request.planner_mode,
                verifier_mode=request.verifier_mode,
                allow_network=search_allow_network,
                api_key=search_api_key,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result.model_dump(mode="json")

    @app.get("/api/v1/jobs")
    def list_jobs() -> dict:
        items = [item.model_dump(mode="json") for item in registry.list_jobs()]
        return {"items": items, "count": len(items)}

    @app.post("/api/v1/jobs", status_code=202)
    async def create_job(
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
    ) -> dict:
        try:
            stored = await store_upload_stream(
                paths,
                request.stream(),
                filename,
                max_bytes=max_upload_bytes,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        relative_path = str(stored.path.relative_to(paths.project_root))
        video, created = registry.register_video(
            sha256=stored.sha256,
            original_filename=stored.original_filename,
            source_path=relative_path,
            bytes=stored.bytes,
        )
        active = registry.active_job_for_video(video.video_id)
        latest = registry.latest_job_for_video(video.video_id)
        reused = active is not None or (
            not created
            and video.status == "READY"
            and latest is not None
            and latest.state == JobState.READY
        )
        if active is not None:
            job = active
        elif reused and latest is not None:
            job = latest
        else:
            job = registry.create_job(video.video_id)
            registry.update_video(video.video_id, status="PROCESSING")
            job = registry.transition(
                job.job_id,
                JobState.QUEUED,
                message="等待媒体 Worker",
            )
        return {
            "job": job.model_dump(mode="json"),
            "video": registry.get_video(video.video_id).model_dump(mode="json"),
            "upload_created": stored.created,
            "job_reused": reused,
        }

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        return _job_payload(registry, job_id)

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        registry.request_cancel(job_id)
        registry.cancel_if_requested(job_id)
        return _job_payload(registry, job_id)

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        after: int = Query(default=0, ge=0),
        once: bool = Query(default=False),
    ) -> StreamingResponse:
        if registry.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def stream() -> AsyncIterator[str]:
            cursor = after
            while True:
                events = registry.events(job_id, after=cursor)
                for event in events:
                    cursor = event.event_id
                    payload = json.dumps(
                        event.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield f"id: {event.event_id}\nevent: job\ndata: {payload}\n\n"
                job = registry.get_job(job_id)
                if once or job is None or job.state in TERMINAL_STATES:
                    yield "event: end\ndata: {}\n\n"
                    return
                if not events:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict:
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.state not in {JobState.READY, JobState.PARTIAL}:
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}")
        video = registry.get_video(job.video_id)
        artifacts = registry.list_artifacts(job.video_id)
        return {
            "job": job.model_dump(mode="json"),
            "video": video.model_dump(mode="json") if video else None,
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
        }

    static_root = ensure_within_project(
        paths.project_root,
        paths.project_root / "shotseek" / "runtime" / "static",
    )
    static_index = static_root / "index.html"
    static_assets = static_root / "assets"
    static_favicon = static_root / "favicon.svg"
    if static_index.is_file():
        if static_assets.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=static_assets),
                name="shotseek-assets",
            )

        if static_favicon.is_file():

            @app.get("/favicon.svg", include_in_schema=False)
            def favicon() -> FileResponse:
                return FileResponse(static_favicon, media_type="image/svg+xml")

        @app.get("/", include_in_schema=False)
        def workbench() -> FileResponse:
            return FileResponse(static_index, media_type="text/html")

    return app
