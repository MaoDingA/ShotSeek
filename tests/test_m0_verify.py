from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from shotseek.fixtures import update_fixtures_from_live_run
from shotseek.m0_verify import verify_fixture_bundle, verify_live_run
from shotseek.schemas import EvidenceSpan, RunManifest, RunReport, Utterance, VideoInfo, VisualEvent
from shotseek.timeline.normalize import normalize_timeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_run(*, speaker_id: str | None) -> Path:
    profile = "complete" if speaker_id else "partial"
    run_id = f"m0-verify-{profile}-{os.getpid()}"
    run_dir = PROJECT_ROOT / ".tmp" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    visual = VisualEvent(
        event_id="visual_0001",
        approx_start_ms=100,
        approx_end_ms=500,
        summary="A person enters the room.",
        characters=["person"],
        actions=["enters"],
        confidence=0.9,
        model="step-3.7-flash",
    )
    utterance = Utterance(
        utterance_id="utterance_0001",
        start_ms=600,
        end_ms=900,
        text="Hello",
        speaker_id=speaker_id,
    )
    evidence = normalize_timeline(75_000, [visual], [utterance])
    video = VideoInfo(
        path="samples/golden.mp4",
        sha256="a" * 64,
        bytes=1_000_000,
        duration_ms=75_000,
        width=1280,
        height=720,
        fps=24.0,
        frame_count=1800,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=1,
    )
    manifest = RunManifest(
        run_id=run_id,
        mode="live",
        created_at="2026-07-16T00:00:00Z",
        video=video,
        models={"vision": "step-3.7-flash", "asr": "stepaudio-2.5-asr"},
        versions={"m0_schema": "test"},
        inputs={"video_delivery": "files_api", "asr_transport": "async_file"},
    )
    gates = {
        "golden_video_public_license": True,
        "video_under_128mb": True,
        "files_api_upload": True,
        "structured_visual_events": True,
        "timestamped_asr": True,
        "speaker_info": speaker_id is not None,
        "unified_timeline": True,
        "timeline_in_bounds": True,
        "raw_and_normalized_separated": True,
    }
    complete = all(gates.values())
    report = RunReport(
        run_id=run_id,
        mode="live",
        status="pass" if complete else "partial",
        video={"sha256": video.sha256, "duration_ms": 75_000, "bytes": video.bytes},
        models=manifest.models,
        versions=manifest.versions,
        metrics={
            "file_upload_ms": 10,
            "vision_request_ms": 20,
            "asr_submit_ms": 5,
            "asr_total_ms": 30,
            "normalization_ms": 1,
            "visual_event_count": 1,
            "utterance_count": 1,
            "evidence_count": 2,
        },
        cache={"file_hit": False, "vision_hit": False, "asr_hit": False},
        gates=gates,
        m0_complete=complete,
        completed_stages=["upload", "vision", "asr", "timeline"],
        errors=[] if complete else ["gate_failed: speaker_info"],
    )
    dump(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    dump(
        run_dir / "raw/stepfun_file.json",
        {"final": {"id": "file_fixture_redacted", "status": "processed"}},
    )
    dump(run_dir / "raw/vision_response.json", {"choices": []})
    dump(run_dir / "raw/asr_response.json", {"result": []})
    dump(run_dir / "normalized/visual_events.json", [visual.model_dump(mode="json")])
    dump(run_dir / "normalized/utterances.json", [utterance.model_dump(mode="json")])
    dump(
        run_dir / "normalized/evidence_timeline.json",
        [item.model_dump(mode="json") for item in evidence],
    )
    dump(run_dir / "run_report.json", report.model_dump(mode="json"))
    return run_dir


def test_verifier_accepts_a_fully_evidenced_live_run() -> None:
    run_dir = make_run(speaker_id="spk_1")
    try:
        result = verify_live_run(PROJECT_ROOT, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    assert result["runtime_complete"] is True
    assert result["failed_checks"] == []


def test_verifier_keeps_missing_speaker_as_a_hard_failure() -> None:
    run_dir = make_run(speaker_id=None)
    try:
        result = verify_live_run(PROJECT_ROOT, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    assert result["runtime_complete"] is False
    assert result["checks"]["speaker_info"] is False
    assert result["checks"]["report_gates_match_evidence"] is True


def test_current_fixture_bundle_is_sanitized_and_live_derived() -> None:
    result = verify_fixture_bundle(PROJECT_ROOT)
    assert result["fixture_sanitized"] is True
    assert result["fixture_live_derived"] is True


def test_fixture_updater_refuses_a_partial_live_run() -> None:
    run_dir = make_run(speaker_id=None)
    try:
        with pytest.raises(ValueError, match="speaker_info"):
            update_fixtures_from_live_run(PROJECT_ROOT, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
