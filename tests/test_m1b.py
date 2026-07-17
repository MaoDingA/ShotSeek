import json
from pathlib import Path

import pytest

from shotseek.m1b import run_m1b, verify_m1b
from shotseek.scenes.builder import interval_distance_ms
from shotseek.scenes.schema import EvidenceRef, Scene


def test_interval_distance() -> None:
    assert interval_distance_ms(0, 1000, 500, 1500) == 0
    assert interval_distance_ms(0, 1000, 1200, 1300) == 200
    assert interval_distance_ms(1200, 1300, 0, 1000) == 200


def test_scene_rejects_mismatched_evidence_refs() -> None:
    with pytest.raises(ValueError, match="dialogue evidence"):
        Scene(
            scene_id="scene_0001",
            start_ms=0,
            end_ms=1000,
            start_frame=0,
            end_frame=24,
            shot_ids=["shot_0001"],
            summary="test",
            characters=[],
            actions=[],
            objects=[],
            location=None,
            visible_text=[],
            visual_event_id="visual_1",
            utterance_ids=["utterance_1"],
            evidence_refs=[EvidenceRef(kind="visual", evidence_id="visual_1")],
            confidence=0.9,
        )


def test_golden_m1b_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    m1a = root / "runs" / "m1a" / "20260717-m1a-v1"
    if not m1a.is_dir():
        return
    first = run_m1b(
        project_root=root,
        m1a_dir=m1a,
        output_dir=root / "runs" / "tests" / "m1b-first",
    )
    second = run_m1b(
        project_root=root,
        m1a_dir=m1a,
        output_dir=root / "runs" / "tests" / "m1b-second",
    )
    assert verify_m1b(first)["status"] == "pass"
    assert verify_m1b(second)["status"] == "pass"
    first_report = json.loads((first / "scene_build_report.json").read_text())
    second_report = json.loads((second / "scene_build_report.json").read_text())
    assert (
        first_report["deterministic_payload_sha256"]
        == second_report["deterministic_payload_sha256"]
    )
