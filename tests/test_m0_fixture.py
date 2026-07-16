from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from shotseek.m0 import ensure_within_project, run_probe
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
    evidence = json.loads(
        (run_dir / "normalized" / "evidence_timeline.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "pass"
    assert report["metrics"]["cache_hit"] is True
    assert report["metrics"]["visual_event_count"] == 4
    assert report["metrics"]["utterance_count"] == 6
    assert len(evidence) == 10


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
    before = set((PROJECT_ROOT / "runs" / "m0").iterdir())
    with pytest.raises(RuntimeError, match="simulated vision failure"):
        run_probe(
            project_root=PROJECT_ROOT,
            video_path=GOLDEN_VIDEO,
            mode="live",
            api_key="fixture-key",
            audio_url="https://example.invalid/golden.mp3",
        )
    created = set((PROJECT_ROOT / "runs" / "m0").iterdir()) - before
    assert len(created) == 1
    run_dir = created.pop()
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
            {"upload": {"id": "file_routing_fixture"}, "final": {"status": "processed"}},
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
                )
            ],
            {"submit": {"task_id": "fixture"}, "result": {"result": []}},
        )

    monkeypatch.setattr("shotseek.m0.upload_video", fake_upload)
    monkeypatch.setattr("shotseek.m0.analyze_video", fake_vision)
    monkeypatch.setattr("shotseek.m0.run_asr", fake_asr)
    run_probe(
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
