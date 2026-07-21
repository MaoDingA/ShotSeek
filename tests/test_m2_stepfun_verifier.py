import json
from pathlib import Path

from shotseek.agent import ShotSeekAgent
from shotseek.planning.rules import build_rule_spec
from shotseek.planning.schema import EntityConstraint, QuerySpecV2
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.verification.router import EvidenceVerifierRouter
from shotseek.verification.rules import RuleEvidenceVerifier
from shotseek.verification.schema import CandidateScene, ScoreComponents
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


def test_stepfun_can_confirm_cross_language_synonym_with_direct_evidence() -> None:
    spec = QuerySpecV2(
        raw_query="找到有人按下播放按钮",
        entities=[EntityConstraint(text="person", role="subject")],
        actions=["press"],
        objects=["play button"],
    )
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    candidate = CandidateScene(
        scene_id="scene_0001",
        start_ms=0,
        end_ms=1_000,
        start_frame=0,
        end_frame=25,
        summary="A person presses a playback button.",
        characters=["person"],
        actions=["pressing"],
        objects=["playback button"],
        location=None,
        visible_text=[],
        dialogue="",
        shot_ids=["shot_0001"],
        evidence_refs=[{"kind": "visual", "evidence_id": "visual_0001"}],
        retrieval_route="relaxed_or",
        retrieval_score=0.8,
        components=ScoreComponents(
            lexical_score=0.8,
            dialogue_score=0.0,
            visual_score=0.75,
            entity_score=1.0,
            temporal_score=1.0,
            evidence_coverage=0.0,
            boundary_quality=1.0,
            contradiction_penalty=0.0,
        ),
    )
    result, trace = EvidenceVerifierRouter().verify(
        spec,
        candidate,
        mode="auto",
        fixture_response=raw,
    )
    assert result.verdict == "supported"
    assert result.direct_evidence is True
    assert result.failed_constraints == []
    assert trace["status"] == "CACHED"


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


def _visual_candidate(
    *,
    summary: str,
    characters: list[str],
    actions: list[str],
    objects: list[str],
) -> CandidateScene:
    return CandidateScene(
        scene_id="scene_test",
        start_ms=0,
        end_ms=1_000,
        start_frame=0,
        end_frame=25,
        summary=summary,
        characters=characters,
        actions=actions,
        objects=objects,
        location="indoor room",
        visible_text=[],
        dialogue="",
        shot_ids=["shot_0001"],
        evidence_refs=[{"kind": "visual", "evidence_id": "visual_0001"}],
        retrieval_route="relaxed_or",
        retrieval_score=0.8,
        components=ScoreComponents(
            lexical_score=0.7,
            dialogue_score=0.0,
            visual_score=0.7,
            entity_score=0.7,
            temporal_score=1.0,
            evidence_coverage=0.0,
            boundary_quality=1.0,
            contradiction_penalty=0.0,
        ),
    )


def test_rule_verifier_supports_multi_person_synonyms() -> None:
    spec = QuerySpecV2(
        raw_query="金发的人和戴眼镜的人在一起",
        entities=[
            EntityConstraint(text="blonde person"),
            EntityConstraint(text="person wearing glasses"),
        ],
    )
    candidate = _visual_candidate(
        summary=(
            "A man with glasses works at holographic displays while a "
            "blonde person stands behind him."
        ),
        characters=["man with glasses", "blonde person behind him"],
        actions=["looking at holographic displays"],
        objects=["holographic displays"],
    )

    result = RuleEvidenceVerifier().verify(spec, candidate)

    assert result.verdict == "supported"
    assert result.direct_evidence is True


def test_rule_verifier_supports_relationship_and_object_synonyms() -> None:
    spec = QuerySpecV2(
        raw_query="机械手在两个人中间",
        entities=[EntityConstraint(text="people")],
        objects=["robot arm"],
        keywords=["between"],
    )
    candidate = _visual_candidate(
        summary=(
            "A young man faces a woman, with a robotic prosthetic hand "
            "positioned between them."
        ),
        characters=["young man", "woman"],
        actions=[],
        objects=["metallic robotic prosthetic hand"],
    )

    result = RuleEvidenceVerifier().verify(spec, candidate)

    assert result.verdict == "supported"
    assert result.direct_evidence is True


def test_stepfun_cannot_invent_missing_spatial_relationship() -> None:
    spec = QuerySpecV2(
        raw_query="机械手在两个人中间",
        entities=[EntityConstraint(text="people")],
        objects=["robot arm"],
        keywords=["between"],
    )
    candidate = _visual_candidate(
        summary=(
            "A blonde woman looks toward another person with a visible "
            "robotic arm."
        ),
        characters=["blonde woman", "person with robotic arm"],
        actions=["woman looks toward person"],
        objects=["robotic mechanical arm"],
    )
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))

    result, trace = EvidenceVerifierRouter().verify(
        spec,
        candidate,
        mode="auto",
        fixture_response=raw,
    )

    assert result.verdict == "unsupported"
    assert "keywords" in result.failed_constraints
    assert trace["status"] == "RULE"


def test_rule_verifier_supports_cross_field_visual_synonyms() -> None:
    spec = QuerySpecV2(
        raw_query="穿军装的女人站在戴机械眼男人后面",
        entities=[
            EntityConstraint(text="woman"),
            EntityConstraint(text="man"),
        ],
        actions=["stand"],
        objects=["military uniform", "mechanical eye"],
        locations=["behind"],
    )
    candidate = _visual_candidate(
        summary=(
            "A man with a cybernetic ocular implant sits at a desk while "
            "a woman in military-style attire stands behind him."
        ),
        characters=[
            "man with cybernetic eye implant",
            "woman in military attire",
        ],
        actions=["woman standing behind man"],
        objects=["radio"],
    )

    baseline = RuleEvidenceVerifier().verify(spec, candidate)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result, trace = EvidenceVerifierRouter().verify(
        spec,
        candidate,
        mode="auto",
        fixture_response=raw,
    )

    assert baseline.verdict == "unsupported"
    assert result.verdict == "supported"
    assert result.direct_evidence is True
    assert trace["status"] == "CACHED"


def test_rule_verifier_does_not_equate_smoking_with_emitting_smoke() -> None:
    spec = QuerySpecV2(
        raw_query="机器人抽烟",
        entities=[EntityConstraint(text="robot")],
        actions=["smoking"],
    )
    candidate = _visual_candidate(
        summary="Smoke emits from the robot's head.",
        characters=["large robot"],
        actions=["smoke emits from robot head"],
        objects=["robot head"],
    )

    result = RuleEvidenceVerifier().verify(spec, candidate)

    assert result.verdict == "unsupported"


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
