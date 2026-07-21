"""Minimal read-only FastAPI surface for M2 search and evidence inspection."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from shotseek.agent import ShotSeekAgent
from shotseek.traces.store import TraceStore


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchRequest(APIModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20)
    planner_mode: Literal["auto", "rule", "stepfun", "cache"] = "auto"
    verifier_mode: Literal["auto", "rule", "stepfun", "cache"] = "auto"


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.search_count = 0
        self.status_counts = {
            "LIVE": 0,
            "CACHED": 0,
            "FALLBACK": 0,
            "RULE": 0,
        }
        self.latencies: list[float] = []

    def record(self, status: str, latency_ms: float) -> None:
        with self._lock:
            self.search_count += 1
            self.status_counts[status] += 1
            self.latencies.append(latency_ms)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self.search_count
            return {
                "schema_version": "m2-api-metrics-v1",
                "search_count": total,
                "status_counts": dict(self.status_counts),
                "cache_rate": (
                    self.status_counts["CACHED"] / total if total else 0.0
                ),
                "fallback_rate": (
                    self.status_counts["FALLBACK"] / total if total else 0.0
                ),
                "query_p50_ms": self._percentile(self.latencies, 0.50),
                "query_p95_ms": self._percentile(self.latencies, 0.95),
            }


def _database_scene_rows(database_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT scene_json, dialogue FROM scene ORDER BY start_ms, scene_id"
        )
        result = []
        for row in rows:
            scene = json.loads(row["scene_json"])
            scene["dialogue"] = row["dialogue"]
            result.append(scene)
        return result


def _video_descriptor(database_path: Path, manifest_path: Path | None) -> dict[str, Any]:
    scenes = _database_scene_rows(database_path)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path is not None and manifest_path.is_file()
        else {}
    )
    digest = str(manifest.get("input_video_sha256") or "unknown")
    return {
        "video_id": f"video_{digest[:16]}",
        "source_sha256": digest,
        "scene_count": len(scenes),
        "duration_ms": max((item["end_ms"] for item in scenes), default=0),
        "timeline_schema_version": "scene-v1",
        "search_state": "READY",
    }


def create_app(
    *,
    database_path: Path,
    manifest_path: Path | None = None,
    trace_dir: Path | None = None,
    planner_cache_dir: Path | None = None,
    verifier_cache_dir: Path | None = None,
    allow_network: bool = False,
    api_key: str | None = None,
) -> FastAPI:
    if not database_path.is_file():
        raise FileNotFoundError(f"search database not found: {database_path}")
    video = _video_descriptor(database_path, manifest_path)
    trace_store = TraceStore(trace_dir) if trace_dir is not None else None
    agent = ShotSeekAgent(
        database_path=database_path,
        planner_cache_dir=planner_cache_dir,
        verifier_cache_dir=verifier_cache_dir,
        trace_dir=trace_dir,
    )
    metrics = RuntimeMetrics()
    app = FastAPI(
        title="ShotSeek API",
        version="0.2.0",
        description="Read-only evidence-aligned scene retrieval API",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "shotseek",
            "database_readable": database_path.is_file(),
            "network_enabled": allow_network,
        }

    @app.get("/videos")
    def list_videos() -> dict[str, Any]:
        return {"items": [video], "count": 1}

    @app.get("/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        if video_id != video["video_id"]:
            raise HTTPException(status_code=404, detail="video not found")
        return video

    @app.get("/videos/{video_id}/scenes")
    def list_video_scenes(video_id: str) -> dict[str, Any]:
        if video_id != video["video_id"]:
            raise HTTPException(status_code=404, detail="video not found")
        scenes = _database_scene_rows(database_path)
        return {"items": scenes, "count": len(scenes)}

    @app.get("/scenes/{scene_id}")
    def get_scene(scene_id: str) -> dict[str, Any]:
        scenes = _database_scene_rows(database_path)
        scene = next(
            (item for item in scenes if item["scene_id"] == scene_id),
            None,
        )
        if scene is None:
            raise HTTPException(status_code=404, detail="scene not found")
        return scene

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        trace = trace_store.get(trace_id) if trace_store is not None else None
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace.model_dump(mode="json")

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        return metrics.snapshot()

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        started = perf_counter()
        response = agent.search(
            request.query,
            top_k=request.top_k,
            planner_mode=request.planner_mode,
            verifier_mode=request.verifier_mode,
            api_key=api_key,
            allow_network=allow_network,
        )
        elapsed_ms = (perf_counter() - started) * 1000
        metrics.record(response.trace.status, elapsed_ms)
        return response.model_dump(mode="json")

    return app
