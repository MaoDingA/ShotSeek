"""M0 contract probe orchestration for live and deterministic fixture modes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shotseek.providers.stepfun import (
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_VISION_MODEL,
)
from shotseek.providers.stepfun.asr import normalize_asr_response, run_asr
from shotseek.providers.stepfun.files import upload_video
from shotseek.providers.stepfun.vision import (
    VISION_PROMPT_VERSION,
    VISION_SCHEMA_VERSION,
    analyze_video,
    normalize_vision_response,
)
from shotseek.schemas import RunManifest, RunReport, VideoInfo
from shotseek.timeline.normalize import normalize_timeline
from shotseek.timeline.validate import validate_evidence_timeline

M0_SCHEMA_VERSION = "m0-schema-v1"


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


def run_probe(
    *,
    project_root: Path,
    video_path: Path,
    mode: str,
    api_key: str | None = None,
    audio_url: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
    asr_model: str = DEFAULT_ASR_MODEL,
) -> Path:
    root = project_root.resolve()
    if mode not in {"live", "fixture"}:
        raise ValueError("mode must be live or fixture")
    video = probe_video(root, video_path)
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

            started = time.perf_counter()
            uploaded, stepfun_file_raw = upload_video(
                root / video.path,
                api_key=api_key,
                base_url=base_url,
            )
            metrics["upload_ms"] = round((time.perf_counter() - started) * 1000)
            _json_dump(raw_dir / "stepfun_file.json", stepfun_file_raw)
            mark_stage("upload")

            started = time.perf_counter()
            visual_events, vision_raw = analyze_video(
                uploaded.file_uri,
                api_key=api_key,
                model=vision_model,
                base_url=base_url,
            )
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
            utterances, asr_raw = run_asr(
                audio_url,
                api_key=api_key,
                model=asr_model,
                base_url=base_url,
                channel=video.audio_channels or 1,
                on_submit=save_asr_submit,
                on_result=save_asr_result,
            )
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
