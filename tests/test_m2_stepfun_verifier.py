import json
from pathlib import Path

from shotseek.agent import ShotSeekAgent
from shotseek.planning.rules import build_rule_spec
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.verification.router import EvidenceVerifierRouter
from shotseek.verification.stepfun import normalize_verifier_response

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runs" / "m1c" / "latest" / "search.sqlite3"
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "stepfun"
    / "query_verifier_response.contract.sample.json"
)


def test_verifier_fixture_contract_is_strictly_parsed() -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = normalize_verifier_response(raw)
    assert result.verdict == "supported"
    assert result.direct_evidence is True


def test_stepfun_cannot_upgrade_rule_unsupported_candidate() -> None:
    if not DATABASE.is_file():
        return
    spec = build_rule_spec("large mechanical quadruped robot outdoors")
    candidates, _ = retrieve_candidates(DATABASE, spec)
    wrong = next(item for item in candidates if item.scene_id != "scene_0003")
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result, trace = EvidenceVerifierRouter().verify(
        spec,
        wrong,
        mode="stepfun",
        fixture_response=raw,
    )
    assert result.verdict == "unsupported"
    assert result.verifier == "cache"
    assert trace["status"] == "CACHED"


def test_stepfun_verifier_falls_back_without_network() -> None:
    if not DATABASE.is_file():
        return
    spec = build_rule_spec("mechanical ocular implant")
    candidate = retrieve_candidates(DATABASE, spec)[0][0]
    result, trace = EvidenceVerifierRouter().verify(
        spec,
        candidate,
        mode="stepfun",
    )
    assert result.verdict == "supported"
    assert result.verifier == "rule"
    assert trace["status"] == "FALLBACK"


def test_agent_trace_records_offline_model_verification() -> None:
    if not DATABASE.is_file():
        return
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    response = ShotSeekAgent(database_path=DATABASE).search(
        "mechanical ocular implant",
        planner_mode="rule",
        verifier_mode="stepfun",
        verifier_fixture=raw,
    )
    assert response.trace.final_scene_ids == ["scene_0016"]
    assert response.trace.status == "CACHED"
    assert response.trace.verification["model_candidate_limit"] == 5
    assert response.trace.verification["status_counts"]["CACHED"] > 0
