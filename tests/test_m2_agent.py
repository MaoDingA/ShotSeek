import json
from pathlib import Path

from shotseek.agent import ShotSeekAgent

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
