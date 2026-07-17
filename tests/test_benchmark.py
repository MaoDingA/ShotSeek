import json
import shutil
from pathlib import Path

import pytest

from shotseek.evaluation.benchmark import (
    BenchmarkCase,
    BenchmarkThresholds,
    load_cases,
    run_benchmark,
)
from shotseek.retrieval.sqlite_index import build_index
from shotseek.scenes.schema import EvidenceRef, Scene


def _scene(
    *,
    scene_id: str,
    start_ms: int,
    end_ms: int,
    summary: str,
    characters: list[str],
    actions: list[str],
    location: str,
    visual_id: str,
) -> Scene:
    return Scene(
        scene_id=scene_id,
        start_ms=start_ms,
        end_ms=end_ms,
        start_frame=start_ms // 40,
        end_frame=end_ms // 40,
        shot_ids=[f"shot_{scene_id[-4:]}"],
        summary=summary,
        characters=characters,
        actions=actions,
        objects=[],
        location=location,
        visible_text=[],
        visual_event_id=visual_id,
        utterance_ids=[],
        evidence_refs=[
            EvidenceRef(kind="visual", evidence_id=visual_id),
        ],
        confidence=0.95,
    )


def test_benchmark_case_requires_complete_temporal_reference() -> None:
    with pytest.raises(ValueError, match="requires both start and end"):
        BenchmarkCase(
            query_id="bad",
            category="visual",
            text="woman standing room",
            acceptable_scene_ids=["scene_0001"],
            reference_start_ms=0,
        )


def test_benchmark_generates_reproducible_reports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / "runs" / "tests" / "benchmark" / tmp_path.name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    database = work / "search.sqlite3"
    build_index(
        database,
        [
            _scene(
                scene_id="scene_0001",
                start_ms=0,
                end_ms=1_000,
                summary="woman standing in room",
                characters=["woman"],
                actions=["standing"],
                location="room",
                visual_id="visual_0001",
            ),
            _scene(
                scene_id="scene_0002",
                start_ms=1_000,
                end_ms=2_000,
                summary="man moving on street",
                characters=["man"],
                actions=["moving"],
                location="street",
                visual_id="visual_0002",
            ),
        ],
        [],
    )
    queries = work / "queries.jsonl"
    cases = [
        {
            "query_id": "dev_001",
            "category": "visual_action",
            "text": "woman standing room",
            "acceptable_scene_ids": ["scene_0001"],
            "reference_start_ms": 0,
            "reference_end_ms": 1_000,
        },
        {
            "query_id": "dev_002",
            "category": "visual_action",
            "text": "man moving street",
            "acceptable_scene_ids": ["scene_0002"],
        },
        {
            "query_id": "dev_003",
            "category": "hard_negative",
            "text": "robot aiming rooftop",
            "acceptable_scene_ids": [],
        },
    ]
    queries.write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )

    loaded = load_cases([queries])
    assert [case.query_id for case in loaded] == [
        "dev_001",
        "dev_002",
        "dev_003",
    ]
    result = run_benchmark(
        project_root=root,
        database_path=database,
        query_paths=[queries],
        output_dir=work / "report",
        split="development-test",
        thresholds=BenchmarkThresholds(
            recall_at_1=1.0,
            recall_at_3=1.0,
            evidence_support_rate=1.0,
            direct_evidence_rate=1.0,
            p95_latency_ms=3_000,
            median_boundary_error_ms=0,
        ),
    )
    assert result.evaluation["pass"] is True
    assert result.evaluation["metrics"]["recall_at_1"] == 1.0
    assert result.evaluation["metrics"]["temporal"]["mean_iou"] == 1.0
    assert result.evaluation["provenance"]["network_calls"] == 0
    assert (result.output_dir / "evaluation.json").is_file()
    assert (result.output_dir / "results.json").is_file()
    assert "ShotSeek Benchmark" in (
        result.output_dir / "report.md"
    ).read_text(encoding="utf-8")
    assert "<!doctype html>" in (
        result.output_dir / "report.html"
    ).read_text(encoding="utf-8").lower()
