import json
from pathlib import Path

from shotseek.agent import ShotSeekAgent
from shotseek.retrieval.sqlite_index import build_index
from shotseek.scenes.schema import EvidenceRef, Scene


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runs" / "m1c" / "latest" / "search.sqlite3"


def test_agent_preserves_all_m1_expected_hits() -> None:
    if not DATABASE.is_file():
        return
    agent = ShotSeekAgent(database_path=DATABASE)
    for raw in (ROOT / "eval" / "m1_queries.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        case = json.loads(raw)
        response = agent.search(case["text"], planner_mode="rule")
        actual = [item.candidate.scene_id for item in response.hits]
        expected = case["acceptable_scene_ids"]
        if expected:
            assert actual
            assert actual[0] in expected, (case["query_id"], actual, expected)
        else:
            assert actual == [], (case["query_id"], actual)
        assert response.trace.verification["unsupported_claim_count"] == 0


def test_agent_applies_ordinal_after_evidence_verification() -> None:
    if not DATABASE.is_file():
        return
    response = ShotSeekAgent(database_path=DATABASE).search(
        "first young man looking to the right outdoors",
        planner_mode="rule",
    )
    assert response.trace.final_scene_ids == ["scene_0008"]
    assert response.trace.temporal["ordinal"]["input_candidate_count"] >= 1


def test_agent_persists_readable_trace() -> None:
    if not DATABASE.is_file():
        return
    trace_dir = ROOT / "runs" / "tests" / "m2-traces"
    agent = ShotSeekAgent(database_path=DATABASE, trace_dir=trace_dir)
    response = agent.search("mechanical ocular implant", planner_mode="rule")
    stored = agent.trace_store.get(response.trace.trace_id)
    assert stored == response.trace


def test_agent_applies_video_alias_and_preserves_user_query(tmp_path: Path) -> None:
    database = tmp_path / "search.sqlite3"
    scene = Scene(
        scene_id="scene_0001",
        start_ms=0,
        end_ms=1_000,
        start_frame=0,
        end_frame=25,
        shot_ids=["shot_0001"],
        summary="An older man with a cybernetic ocular implant holds a radio.",
        characters=["man with cybernetic eye implant"],
        actions=["holding radio"],
        objects=["radio", "cybernetic eye implant"],
        location="indoor room",
        visible_text=[],
        visual_event_id="visual_0001",
        utterance_ids=[],
        evidence_refs=[
            EvidenceRef(kind="visual", evidence_id="visual_0001")
        ],
        confidence=0.9,
    )
    build_index(database, [scene], [])

    response = ShotSeekAgent(
        database_path=database,
        query_aliases={"老爷爷": "man with cybernetic eye implant"},
    ).search("找到老爷爷", planner_mode="rule")

    assert response.trace.final_scene_ids == ["scene_0001"]
    assert response.trace.query == "找到老爷爷"
    assert response.trace.query_spec.raw_query == "找到老爷爷"
    assert response.trace.retrieval["query_alias_matches"] == ["老爷爷"]
    assert "video aliases: 老爷爷" in response.trace.planner.route_reason


def test_agent_resolves_translated_synonym_to_video_alias(tmp_path: Path) -> None:
    database = tmp_path / "search.sqlite3"
    scene = Scene(
        scene_id="scene_0001",
        start_ms=0,
        end_ms=1_000,
        start_frame=0,
        end_frame=25,
        shot_ids=["shot_0001"],
        summary="A man with a cybernetic ocular implant holds a radio.",
        characters=["man with cybernetic eye implant"],
        actions=["holding radio"],
        objects=["radio", "cybernetic eye implant"],
        location="indoor room",
        visible_text=[],
        visual_event_id="visual_0001",
        utterance_ids=[],
        evidence_refs=[
            EvidenceRef(kind="visual", evidence_id="visual_0001")
        ],
        confidence=0.9,
    )
    build_index(database, [scene], [])
    planner_fixture = {
        "choices": [
            {
                "message": {
                    "content": {
                        "entities": [
                            {"text": "old man", "role": "subject"}
                        ],
                        "evidence_preference": ["visual", "dialogue"],
                        "require_direct_evidence": True,
                    }
                }
            }
        ]
    }

    response = ShotSeekAgent(
        database_path=database,
        query_aliases={"老爷爷": "man with cybernetic eye implant"},
    ).search("找到大爷", planner_fixture=planner_fixture)

    assert response.trace.final_scene_ids == ["scene_0001"]
    assert response.trace.query_spec.raw_query == "找到大爷"
    assert [item.text for item in response.trace.query_spec.entities] == [
        "man with cybernetic eye implant"
    ]
    assert response.trace.retrieval["query_alias_matches"] == ["老爷爷"]

    fallback = ShotSeekAgent(
        database_path=database,
        query_aliases={"老爷爷": "man with cybernetic eye implant"},
    ).search("大爷", planner_mode="rule")

    assert fallback.trace.final_scene_ids == ["scene_0001"]
    assert [item.text for item in fallback.trace.query_spec.entities] == [
        "man with cybernetic eye implant"
    ]


def test_semantic_video_alias_does_not_promote_generic_man(tmp_path: Path) -> None:
    database = tmp_path / "search.sqlite3"
    scene = Scene(
        scene_id="scene_0001",
        start_ms=0,
        end_ms=1_000,
        start_frame=0,
        end_frame=25,
        shot_ids=["shot_0001"],
        summary="A man holds a radio.",
        characters=["man"],
        actions=["holding radio"],
        objects=["radio"],
        location="indoor room",
        visible_text=[],
        visual_event_id="visual_0001",
        utterance_ids=[],
        evidence_refs=[
            EvidenceRef(kind="visual", evidence_id="visual_0001")
        ],
        confidence=0.9,
    )
    build_index(database, [scene], [])

    response = ShotSeekAgent(
        database_path=database,
        query_aliases={"老爷爷": "man with cybernetic eye implant"},
    ).search("man", planner_mode="rule")

    assert response.trace.retrieval["query_alias_matches"] == []
    assert [item.text for item in response.trace.query_spec.entities] == ["man"]
