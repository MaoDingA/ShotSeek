"""M0 contract probe orchestration for live and deterministic fixture modes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from shotseek.providers.stepfun import (
    DEFAULT_ASR_BASE_URL,
    DEFAULT_ASR_MODEL,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_FILES_BASE_URL,
    DEFAULT_VISION_MODEL,
)
from shotseek.providers.stepfun.asr import normalize_asr_response, run_asr
from shotseek.providers.stepfun.files import upload_video
from shotseek.providers.stepfun.vision import (
    VISION_PROMPT_VERSION,
    VISION_SCHEMA_VERSION,
    VisionResponseError,
    analyze_video,
    normalize_vision_response,
)
from shotseek.schemas import RunManifest, RunReport, VideoChunkInput, VideoInfo
from shotseek.timeline.normalize import normalize_timeline
from shotseek.timeline.validate import validate_evidence_timeline

M0_SCHEMA_VERSION = "m0-schema-v2"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_rate(value: str | None) -> float:
    if not value or value == "0/0":
        raise ValueError("video frame rate is missing")
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def ensure_within_project(project_root: Path, candidate: Path) -> Path:
    root = project_root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay inside project root: {resolved}") from exc
    return resolved


def probe_video(project_root: Path, video_path: Path) -> VideoInfo:
    resolved = ensure_within_project(project_root, video_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,channels",
        "-of",
        "json",
        str(resolved),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise ValueError("input file has no video stream")
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"), None
    )
    duration_ms = int(round(float(payload["format"]["duration"]) * 1000))
    frame_count_raw = video_stream.get("nb_frames")
    frame_count = None
    if frame_count_raw not in {None, "N/A", ""}:
        frame_count = int(frame_count_raw)
    relative_path = str(resolved.relative_to(project_root.resolve()))
    return VideoInfo(
        path=relative_path,
        sha256=_sha256(resolved),
        bytes=int(payload["format"].get("size") or resolved.stat().st_size),
        duration_ms=duration_ms,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=_parse_rate(video_stream.get("r_frame_rate")),
        frame_count=frame_count,
        video_codec=str(video_stream["codec_name"]),
        audio_codec=(str(audio_stream["codec_name"]) if audio_stream else None),
        audio_channels=(int(audio_stream["channels"]) if audio_stream else None),
    )


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def load_video_chunks(
    project_root: Path,
    manifest_path: Path,
    video_duration_ms: int,
) -> list[VideoChunkInput]:
    resolved = ensure_within_project(project_root, manifest_path)
    payload = _json_load(resolved)
    items = payload.get("chunks") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("video chunk manifest must contain a non-empty chunks array")

    chunks = [VideoChunkInput.model_validate(item) for item in items]
    seen_ids: set[str] = set()
    expected_start_ms = 0
    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"duplicate video chunk_id: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        if chunk.source_start_ms != expected_start_ms:
            raise ValueError(
                "video chunks must be ordered and contiguous: "
                f"expected {expected_start_ms}, got {chunk.source_start_ms}"
            )
        expected_start_ms = chunk.source_end_ms

    if expected_start_ms != video_duration_ms:
        raise ValueError(
            "video chunks must cover the full source duration: "
            f"covered {expected_start_ms}, video is {video_duration_ms}"
        )
    return chunks


def run_probe(
    *,
    project_root: Path,
    video_path: Path,
    mode: str,
    api_key: str | None = None,
    audio_url: str | None = None,
    video_chunks_path: Path | None = None,
    files_base_url: str = DEFAULT_FILES_BASE_URL,
    chat_base_url: str = DEFAULT_CHAT_BASE_URL,
    asr_base_url: str = DEFAULT_ASR_BASE_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
    asr_model: str = DEFAULT_ASR_MODEL,
) -> Path:
    root = project_root.resolve()
    if mode not in {"live", "fixture"}:
        raise ValueError("mode must be live or fixture")
    video = probe_video(root, video_path)
    if mode == "fixture" and video_chunks_path is not None:
        raise ValueError("fixture mode does not accept a video chunk manifest")
    video_chunks = (
        load_video_chunks(root, video_chunks_path, video.duration_ms)
        if video_chunks_path is not None
        else None
    )
    run_id = _new_run_id()
    run_dir = ensure_within_project(root, root / "runs" / "m0" / run_id)
    raw_dir = run_dir / "raw"
    normalized_dir = run_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=False)
    normalized_dir.mkdir(parents=True, exist_ok=False)

    models = {"vision": vision_model, "asr": asr_model}
    versions = {
        "prompt": VISION_PROMPT_VERSION,
        "vision_schema": VISION_SCHEMA_VERSION,
        "m0_schema": M0_SCHEMA_VERSION,
    }
    manifest = RunManifest(
        run_id=run_id,
        mode=mode,
        created_at=datetime.now(UTC).isoformat(),
        video=video,
        models=models,
        versions=versions,
        inputs={
            "video": video.path,
            "audio_url": "<provided>" if mode == "live" else None,
            "video_delivery": (
                "fixture"
                if mode == "fixture"
                else (
                    "direct_url_chunks" if video_chunks is not None else "files_api"
                )
            ),
            "video_chunks": (
                str(
                    ensure_within_project(root, video_chunks_path).relative_to(root)
                )
                if video_chunks_path is not None
                else None
            ),
        },
    )
    _json_dump(run_dir / "manifest.json", manifest.model_dump(mode="json"))

    metrics: dict[str, int | float | bool] = {
        "cache_hit": mode == "fixture",
        "upload_ms": 0,
        "vision_ms": 0,
        "asr_ms": 0,
        "normalize_ms": 0,
        "completed_stage_count": 0,
    }
    errors: list[str] = []
    completed_stages: list[str] = []

    def mark_stage(stage: str) -> None:
        completed_stages.append(stage)
        metrics["completed_stage_count"] = len(completed_stages)

    try:
        if mode == "fixture":
            fixture_dir = root / "tests" / "fixtures" / "stepfun"
            stepfun_file_raw = _json_load(fixture_dir / "stepfun_file.sample.json")
            vision_raw = _json_load(fixture_dir / "vision_response.sample.json")
            asr_raw = _json_load(fixture_dir / "asr_response.sample.json")
            visual_events = normalize_vision_response(vision_raw, model=vision_model)
            utterances = normalize_asr_response(asr_raw)
            _json_dump(raw_dir / "stepfun_file.json", stepfun_file_raw)
            _json_dump(raw_dir / "vision_response.json", vision_raw)
            _json_dump(raw_dir / "asr_response.json", asr_raw)
            _json_dump(
                normalized_dir / "visual_events.json",
                [event.model_dump(mode="json") for event in visual_events],
            )
            _json_dump(
                normalized_dir / "utterances.json",
                [utterance.model_dump(mode="json") for utterance in utterances],
            )
            mark_stage("fixture_loaded")
        else:
            if not api_key:
                raise ValueError("STEPFUN_API_KEY is required in live mode")
            if not audio_url:
                raise ValueError("--audio-url or GOLDEN_AUDIO_URL is required in live mode")

            if video_chunks is None:
                started = time.perf_counter()
                try:
                    uploaded, stepfun_file_raw = upload_video(
                        root / video.path,
                        api_key=api_key,
                        base_url=files_base_url,
                    )
                finally:
                    metrics["upload_ms"] = round(
                        (time.perf_counter() - started) * 1000
                    )
                _json_dump(raw_dir / "stepfun_file.json", stepfun_file_raw)
                mark_stage("upload")
            else:
                stepfun_file_raw = {
                    "mode": "direct_url_chunks",
                    "files_api_used": False,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_start_ms": chunk.source_start_ms,
                            "source_end_ms": chunk.source_end_ms,
                            "url": "<provided>",
                        }
                        for chunk in video_chunks
                    ],
                }
                _json_dump(raw_dir / "stepfun_file.json", stepfun_file_raw)
                metrics["vision_chunk_count"] = len(video_chunks)
                metrics["vision_completed_chunk_count"] = 0
                mark_stage("vision_input")

            started = time.perf_counter()
            try:
                if video_chunks is None:
                    visual_events, vision_raw = analyze_video(
                        uploaded.file_uri,
                        api_key=api_key,
                        model=vision_model,
                        base_url=chat_base_url,
                    )
                else:
                    visual_events = []
                    vision_raw_chunks: list[dict[str, Any]] = []
                    for chunk in video_chunks:
                        try:
                            chunk_events, chunk_raw = analyze_video(
                                chunk.url,
                                api_key=api_key,
                                model=vision_model,
                                base_url=chat_base_url,
                                chunk_id_override=chunk.chunk_id,
                                source_start_ms=chunk.source_start_ms,
                            )
                        except VisionResponseError as exc:
                            vision_raw_chunks.append(
                                {
                                    "chunk_id": chunk.chunk_id,
                                    "source_start_ms": chunk.source_start_ms,
                                    "source_end_ms": chunk.source_end_ms,
                                    "error": str(exc),
                                    "response": {
                                        "mode": "normalization_failed",
                                        "attempts": exc.attempts,
                                    },
                                }
                            )
                            _json_dump(
                                raw_dir / "vision_response.json",
                                {"mode": "direct_url_chunks", "chunks": vision_raw_chunks},
                            )
                            raise
                        visual_events.extend(chunk_events)
                        vision_raw_chunks.append(
                            {
                                "chunk_id": chunk.chunk_id,
                                "source_start_ms": chunk.source_start_ms,
                                "source_end_ms": chunk.source_end_ms,
                                "response": chunk_raw,
                            }
                        )
                        metrics["vision_completed_chunk_count"] = len(
                            vision_raw_chunks
                        )
                        vision_raw = {
                            "mode": "direct_url_chunks",
                            "chunks": vision_raw_chunks,
                        }
                        _json_dump(raw_dir / "vision_response.json", vision_raw)
                        _json_dump(
                            normalized_dir / "visual_events.json",
                            [
                                event.model_dump(mode="json")
                                for event in visual_events
                            ],
                        )
            finally:
                metrics["vision_ms"] = round((time.perf_counter() - started) * 1000)
            metrics["visual_event_count"] = len(visual_events)
            _json_dump(raw_dir / "vision_response.json", vision_raw)
            _json_dump(
                normalized_dir / "visual_events.json",
                [event.model_dump(mode="json") for event in visual_events],
            )
            mark_stage("vision")

            asr_partial: dict[str, Any] = {"submit": None, "result": None}

            def save_asr_submit(submit_raw: dict[str, Any]) -> None:
                asr_partial["submit"] = submit_raw
                _json_dump(raw_dir / "asr_response.json", asr_partial)

            def save_asr_result(result_raw: dict[str, Any]) -> None:
                asr_partial["result"] = result_raw
                _json_dump(raw_dir / "asr_response.json", asr_partial)

            started = time.perf_counter()
            try:
                utterances, asr_raw = run_asr(
                    audio_url,
                    api_key=api_key,
                    model=asr_model,
                    base_url=asr_base_url,
                    channel=video.audio_channels or 1,
                    on_submit=save_asr_submit,
                    on_result=save_asr_result,
                )
            except httpx.HTTPStatusError as exc:
                try:
                    response_body: Any = exc.response.json()
                except (json.JSONDecodeError, ValueError):
                    response_body = exc.response.text
                _json_dump(
                    raw_dir / "asr_response.json",
                    {
                        "status": "failed",
                        "http_status": exc.response.status_code,
                        "endpoint": str(exc.request.url.copy_with(query=None)),
                        "response": response_body,
                    },
                )
                raise
            finally:
                metrics["asr_ms"] = round((time.perf_counter() - started) * 1000)
            metrics["utterance_count"] = len(utterances)
            _json_dump(raw_dir / "asr_response.json", asr_raw)
            _json_dump(
                normalized_dir / "utterances.json",
                [utterance.model_dump(mode="json") for utterance in utterances],
            )
            mark_stage("asr")

        started = time.perf_counter()
        evidence = normalize_timeline(video.duration_ms, visual_events, utterances)
        validate_evidence_timeline(evidence, video.duration_ms)
        metrics["normalize_ms"] = round((time.perf_counter() - started) * 1000)
        metrics["visual_event_count"] = len(visual_events)
        metrics["utterance_count"] = len(utterances)
        metrics["evidence_count"] = len(evidence)
        _json_dump(
            normalized_dir / "evidence_timeline.json",
            [item.model_dump(mode="json") for item in evidence],
        )
        mark_stage("timeline")
        report = RunReport(
            run_id=run_id,
            mode=mode,
            status="pass",
            video={
                "sha256": video.sha256,
                "duration_ms": video.duration_ms,
                "bytes": video.bytes,
            },
            models=models,
            versions=versions,
            metrics=metrics,
            completed_stages=completed_stages,
            errors=[],
        )
        _json_dump(run_dir / "run_report.json", report.model_dump(mode="json"))
        return run_dir
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        report = RunReport(
            run_id=run_id,
            mode=mode,
            status="partial" if completed_stages else "failed",
            video={
                "sha256": video.sha256,
                "duration_ms": video.duration_ms,
                "bytes": video.bytes,
            },
            models=models,
            versions=versions,
            metrics=metrics,
            completed_stages=completed_stages,
            errors=errors,
        )
        _json_dump(run_dir / "run_report.json", report.model_dump(mode="json"))
        raise
