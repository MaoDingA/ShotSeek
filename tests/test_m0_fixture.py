from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import httpx
import pytest

from shotseek.m0 import ensure_within_project, probe_video, run_probe
from shotseek.schemas import UploadedFile, Utterance, VisualEvent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_VIDEO = PROJECT_ROOT / "samples" / "golden.mp4"


def test_path_lock_rejects_outside_project() -> None:
    with pytest.raises(ValueError, match="inside project root"):
        ensure_within_project(PROJECT_ROOT, PROJECT_ROOT.parent / "outside.mp4")


def test_fixture_mode_does_not_construct_an_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the end-to-end fixture test")

    def fail_http_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixture mode attempted network access")

    monkeypatch.setattr(httpx, "Client", fail_http_client)
    run_dir = run_probe(
        project_root=PROJECT_ROOT,
        video_path=GOLDEN_VIDEO,
        mode="fixture",
        api_key=None,
        audio_url=None,
    )
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (run_dir / "normalized" / "evidence_timeline.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "pass"
    assert report["metrics"]["cache_hit"] is True
    assert manifest["inputs"]["video_delivery"] == "fixture"
    assert manifest["inputs"]["fixture_profile"] == "live_sse_plus_contract"
    assert report["metrics"]["visual_event_count"] == 3
    assert report["metrics"]["utterance_count"] == 10
    assert len(evidence) == 13


def test_live_failure_preserves_completed_upload_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the live failure test")

    def fake_upload(
        *args: object, **kwargs: object
    ) -> tuple[UploadedFile, dict[str, object]]:
        return (
            UploadedFile(
                file_id="file_partial_fixture",
                file_uri="stepfile://file_partial_fixture",
                filename="golden.mp4",
                bytes=GOLDEN_VIDEO.stat().st_size,
                sha256="9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8",
                status="processed",
            ),
            {"upload": {"id": "file_partial_fixture"}, "final": {"status": "processed"}},
        )

    def fail_vision(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated vision failure")

    monkeypatch.setattr("shotseek.m0.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fail_vision)
    run_id = f"test-vision-failure-{os.getpid()}"
    run_dir = PROJECT_ROOT / "runs" / "m0" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    monkeypatch.setattr("shotseek.m0._new_run_id", lambda: run_id)
    with pytest.raises(RuntimeError, match="simulated vision failure"):
        run_probe(
            project_root=PROJECT_ROOT,
            video_path=GOLDEN_VIDEO,
            mode="live",
            api_key="fixture-key",
            audio_url="https://example.invalid/golden.mp3",
        )
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "partial"
    assert report["completed_stages"] == ["upload"]
    assert (run_dir / "raw" / "stepfun_file.json").is_file()
    assert not (run_dir / "raw" / "vision_response.json").exists()


def test_live_probe_routes_each_provider_to_its_own_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the live routing test")

    seen: dict[str, str] = {}

    def fake_upload(
        *args: object, **kwargs: object
    ) -> tuple[UploadedFile, dict[str, object]]:
        seen["files"] = str(kwargs["base_url"])
        return (
            UploadedFile(
                file_id="file_routing_fixture",
                file_uri="stepfile://file_routing_fixture",
                filename="golden.mp4",
                bytes=GOLDEN_VIDEO.stat().st_size,
                sha256="9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8",
                status="processed",
            ),
            {
                "upload": {"id": "file_routing_fixture"},
                "final": {"id": "file_routing_fixture", "status": "processed"},
            },
        )

    def fake_vision(
        *args: object, **kwargs: object
    ) -> tuple[list[VisualEvent], dict[str, object]]:
        seen["chat"] = str(kwargs["base_url"])
        return (
            [
                VisualEvent(
                    event_id="visual_routing_fixture",
                    approx_start_ms=1000,
                    approx_end_ms=2000,
                    summary="A person enters a room.",
                    characters=["person"],
                    actions=["enters"],
                    confidence=0.8,
                    model="fixture-model",
                )
            ],
            {"choices": []},
        )

    def fake_asr(
        *args: object, **kwargs: object
    ) -> tuple[list[Utterance], dict[str, object]]:
        seen["asr"] = str(kwargs["base_url"])
        return (
            [
                Utterance(
                    utterance_id="utterance_routing_fixture",
                    start_ms=2500,
                    end_ms=3000,
                    text="Hello",
                    speaker_id="spk_1",
                )
            ],
            {"submit": {"task_id": "fixture"}, "result": {"result": []}},
        )

    monkeypatch.setattr("shotseek.m0.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fake_vision)
    monkeypatch.setattr("shotseek.m0.run_asr", fake_asr)
    run_dir = run_probe(
        project_root=PROJECT_ROOT,
        video_path=GOLDEN_VIDEO,
        mode="live",
        api_key="fixture-key",
        audio_url="https://example.invalid/golden.mp3",
        files_base_url="https://files.example.invalid/v1",
        chat_base_url="https://chat.example.invalid/step_plan/v1",
        asr_base_url="https://asr.example.invalid/v1",
    )

    assert seen == {
        "files": "https://files.example.invalid/v1",
        "chat": "https://chat.example.invalid/step_plan/v1",
        "asr": "https://asr.example.invalid/v1",
    }
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["m0_complete"] is True
    assert all(report["gates"].values())


def test_direct_video_chunks_skip_files_and_preserve_source_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the direct URL test")

    duration_ms = probe_video(PROJECT_ROOT, GOLDEN_VIDEO).duration_ms
    manifest_path = PROJECT_ROOT / ".tmp" / f"direct-video-chunks-test-{os.getpid()}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    start_ms = 0
    index = 0
    while start_ms < duration_ms:
        end_ms = min(start_ms + 10_000, duration_ms)
        chunks.append(
            {
                "chunk_id": f"chunk_{index:03d}",
                "source_start_ms": start_ms,
                "source_end_ms": end_ms,
                "url": f"https://example.invalid/chunk-{index:03d}.mp4",
            }
        )
        start_ms = end_ms
        index += 1
    manifest_path.write_text(
        json.dumps({"chunks": chunks}),
        encoding="utf-8",
    )

    seen_urls: list[str] = []

    def fail_upload(*args: object, **kwargs: object) -> None:
        raise AssertionError("direct URL mode must not call the Files API")

    def fake_vision(
        file_uri: str, *args: object, **kwargs: object
    ) -> tuple[list[VisualEvent], dict[str, object]]:
        seen_urls.append(file_uri)
        chunk_id = str(kwargs["chunk_id_override"])
        source_start_ms = int(kwargs["source_start_ms"])
        return (
            [
                VisualEvent(
                    event_id=f"{chunk_id}:visual_0001",
                    approx_start_ms=100,
                    approx_end_ms=500,
                    summary="A person crosses the frame.",
                    characters=["person"],
                    actions=["crosses"],
                    confidence=0.8,
                    model="fixture-model",
                    chunk_id=chunk_id,
                    source_start_ms=source_start_ms,
                )
            ],
            {"choices": [{"message": {"content": "fixture"}}]},
        )

    def fake_asr(
        *args: object, **kwargs: object
    ) -> tuple[list[Utterance], dict[str, object]]:
        return (
            [
                Utterance(
                    utterance_id="utterance_direct_fixture",
                    start_ms=500,
                    end_ms=900,
                    text="Hello",
                    speaker_id="spk_1",
                )
            ],
            {"submit": {"task_id": "fixture"}, "result": {"result": []}},
        )

    monkeypatch.setattr("shotseek.m0.upload_video", fail_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fake_vision)
    monkeypatch.setattr("shotseek.m0.run_asr", fake_asr)
    try:
        run_dir = run_probe(
            project_root=PROJECT_ROOT,
            video_path=GOLDEN_VIDEO,
            mode="live",
            api_key="fixture-key",
            audio_url="https://example.invalid/golden.mp3",
            video_chunks_path=manifest_path,
        )
    finally:
        manifest_path.unlink(missing_ok=True)

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    delivery = json.loads(
        (run_dir / "raw" / "stepfun_file.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (run_dir / "normalized" / "evidence_timeline.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "partial"
    assert report["m0_complete"] is False
    assert report["gates"]["files_api_upload"] is False
    assert report["gates"]["speaker_info"] is True
    assert report["completed_stages"] == ["vision_input", "vision", "asr", "timeline"]
    assert report["metrics"]["vision_chunk_count"] == len(chunks)
    assert report["metrics"]["vision_completed_chunk_count"] == len(chunks)
    assert manifest["inputs"]["video_delivery"] == "direct_url_chunks"
    assert delivery["files_api_used"] is False
    assert all(item["url"] == "<provided>" for item in delivery["chunks"])
    assert seen_urls == [str(item["url"]) for item in chunks]
    visual_starts = [item["start_ms"] for item in evidence if item["kind"] == "visual"]
    assert visual_starts[0] == 100
    assert visual_starts[-1] == int(chunks[-1]["source_start_ms"]) + 100


def test_cached_vision_can_require_a_fresh_files_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the cache upload test")

    def fake_cached_vision(*args: object, **kwargs: object):
        return (
            [VisualEvent(
                event_id="visual_cached_fixture",
                approx_start_ms=100,
                approx_end_ms=500,
                summary="A person crosses the frame.",
                confidence=0.8,
                model="step-3.7-flash",
            )],
            {"choices": [{"message": {"content": "fixture"}}]},
            {"mode": "direct_url_chunks", "files_api_used": False, "chunks": []},
        )

    def fake_upload(*args: object, **kwargs: object):
        return (
            UploadedFile(
                file_id="file_fresh_fixture",
                file_uri="stepfile://file_fresh_fixture",
                filename="golden.mp4",
                bytes=GOLDEN_VIDEO.stat().st_size,
                sha256="a" * 64,
                status="processed",
            ),
            {
                "upload": {"id": "file_fresh_fixture"},
                "final": {"id": "file_fresh_fixture", "status": "processed"},
            },
        )

    def fake_asr(*args: object, **kwargs: object):
        return (
            [Utterance(
                utterance_id="utterance_cached_fixture",
                start_ms=600,
                end_ms=900,
                text="Hello",
                speaker_id="spk_1",
            )],
            {"submit": {"task_id": "fixture"}, "result": {"result": []}},
        )

    def fail_vision(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached vision should not be analyzed again")

    monkeypatch.setattr("shotseek.m0.load_cached_vision", fake_cached_vision)
    monkeypatch.setattr("shotseek.m0.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fail_vision)
    monkeypatch.setattr("shotseek.m0.run_asr", fake_asr)

    run_dir = run_probe(
        project_root=PROJECT_ROOT,
        video_path=GOLDEN_VIDEO,
        mode="live",
        api_key="fixture-key",
        audio_url="https://example.invalid/golden.mp3",
        vision_cache_run=PROJECT_ROOT / ".tmp" / "cached-live-run",
        require_files_upload=True,
    )

    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    file_raw = json.loads(
        (run_dir / "raw" / "stepfun_file.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "pass"
    assert report["completed_stages"] == [
        "vision_cache", "upload", "vision", "asr", "timeline"
    ]
    assert report["cache"] == {
        "file_hit": False, "vision_hit": True, "asr_hit": False
    }
    assert manifest["inputs"]["video_delivery"] == "files_api_plus_vision_cache"
    assert manifest["inputs"]["files_upload_required"] is True
    assert file_raw["final"]["id"] == "file_fresh_fixture"
    assert all(report["gates"].values())


def test_live_asr_http_failure_is_preserved_as_a_raw_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the ASR failure test")

    def fake_upload(
        *args: object, **kwargs: object
    ) -> tuple[UploadedFile, dict[str, object]]:
        return (
            UploadedFile(
                file_id="file_asr_failure_fixture",
                file_uri="stepfile://file_asr_failure_fixture",
                filename="golden.mp4",
                bytes=GOLDEN_VIDEO.stat().st_size,
                sha256="9a11b716f750bd61f081c47f2195ca3fdacf8b098891d862c273bfd172c50aa8",
                status="processed",
            ),
            {"upload": {"id": "file_asr_failure_fixture"}, "final": {"status": "processed"}},
        )

    def fake_vision(
        *args: object, **kwargs: object
    ) -> tuple[list[VisualEvent], dict[str, object]]:
        return (
            [
                VisualEvent(
                    event_id="visual_asr_failure_fixture",
                    approx_start_ms=100,
                    approx_end_ms=500,
                    summary="A person crosses the frame.",
                    confidence=0.8,
                    model="fixture-model",
                )
            ],
            {"choices": []},
        )

    def fail_asr(*args: object, **kwargs: object) -> None:
        request = httpx.Request(
            "POST",
            "https://asr.example.invalid/v1/audio/asr/file/submit",
        )
        response = httpx.Response(
            402,
            request=request,
            json={"error": {"message": "quota exceeded"}},
        )
        raise httpx.HTTPStatusError("quota exceeded", request=request, response=response)

    monkeypatch.setattr("shotseek.m0.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fake_vision)
    monkeypatch.setattr("shotseek.m0.run_asr", fail_asr)
    run_id = f"test-asr-failure-{os.getpid()}"
    run_dir = PROJECT_ROOT / "runs" / "m0" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    monkeypatch.setattr("shotseek.m0._new_run_id", lambda: run_id)
    with pytest.raises(httpx.HTTPStatusError):
        run_probe(
            project_root=PROJECT_ROOT,
            video_path=GOLDEN_VIDEO,
            mode="live",
            api_key="fixture-key",
            audio_url="https://example.invalid/golden.mp3",
        )
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    raw_asr = json.loads(
        (run_dir / "raw" / "asr_response.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "partial"
    assert report["completed_stages"] == ["upload", "vision"]
    assert raw_asr == {
        "endpoint": "https://asr.example.invalid/v1/audio/asr/file/submit",
        "http_status": 402,
        "response": {"error": {"message": "quota exceeded"}},
        "status": "failed",
    }


def test_cached_vision_and_sse_asr_produce_timeline_but_keep_hard_gates_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not GOLDEN_VIDEO.exists():
        pytest.skip("run scripts/prepare_golden_sample.py to enable the cache test")

    def fake_cached_vision(
        *args: object, **kwargs: object
    ) -> tuple[list[VisualEvent], dict[str, object], dict[str, object]]:
        return (
            [
                VisualEvent(
                    event_id="chunk_000:visual_0001",
                    approx_start_ms=100,
                    approx_end_ms=500,
                    summary="A person crosses the frame.",
                    confidence=0.8,
                    model="step-3.7-flash",
                    chunk_id="chunk_000",
                )
            ],
            {"mode": "direct_url_chunks", "chunks": []},
            {
                "mode": "direct_url_chunks",
                "files_api_used": False,
                "chunks": [],
            },
        )

    def fake_sse_asr(
        *args: object, **kwargs: object
    ) -> tuple[list[Utterance], dict[str, object]]:
        return (
            [
                Utterance(
                    utterance_id="utterance_sse_fixture",
                    start_ms=600,
                    end_ms=900,
                    text="Hello",
                    speaker_id=None,
                    source="stepfun_asr_sse",
                )
            ],
            {"transport": "sse", "events": []},
        )

    def fail_unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("cached SSE probe called an unexpected provider")

    monkeypatch.setattr("shotseek.m0.load_cached_vision", fake_cached_vision)
    monkeypatch.setattr("shotseek.m0.run_sse_asr", fake_sse_asr)
    monkeypatch.setattr("shotseek.m0.upload_video", fail_unexpected)
    monkeypatch.setattr("shotseek.m0.analyze_video", fail_unexpected)
    monkeypatch.setattr("shotseek.m0.run_asr", fail_unexpected)

    run_dir = run_probe(
        project_root=PROJECT_ROOT,
        video_path=GOLDEN_VIDEO,
        mode="live",
        api_key="fixture-key",
        audio_url="https://example.invalid/golden.mp3",
        vision_cache_run=PROJECT_ROOT / ".tmp" / "cached-live-run",
        asr_transport="sse",
    )
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (run_dir / "normalized" / "evidence_timeline.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "partial"
    assert report["m0_complete"] is False
    assert report["completed_stages"] == ["vision_cache", "vision", "asr", "timeline"]
    assert report["cache"] == {
        "file_hit": True,
        "vision_hit": True,
        "asr_hit": False,
    }
    assert report["gates"]["timestamped_asr"] is True
    assert report["gates"]["unified_timeline"] is True
    assert report["gates"]["files_api_upload"] is False
    assert report["gates"]["speaker_info"] is False
    assert sorted(report["errors"]) == [
        "gate_failed: files_api_upload",
        "gate_failed: speaker_info",
    ]
    assert {item["kind"] for item in evidence} == {"visual", "dialogue"}
