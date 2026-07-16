from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from shotseek.m0 import ensure_within_project, run_probe

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
