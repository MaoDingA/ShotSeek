import json
from pathlib import Path

from shotseek.planning.rules import build_rule_spec
from shotseek.retrieval.candidates import normalized_tokens, retrieve_candidates
from shotseek.retrieval.temporal import resolve_temporal_constraints
from shotseek.verification.rules import RuleEvidenceVerifier

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runs" / "m1c" / "latest" / "search.sqlite3"


def _database_available() -> bool:
    return DATABASE.is_file()


def test_normalized_tokens_cover_common_inflections_and_spatial_synonyms() -> None:
    assert normalized_tokens(
        "woman rides and mounted; dragon's wings flew towards a nearby fire"
    ) == [
        "woman",
        "ride",
        "and",
        "dragon",
        "wings",
        "fly",
        "toward",
        "near",
        "fire",
    ]
    assert normalized_tokens("pressing a glowing emblem beside stakes") == [
        "press", "glow", "emblem", "near", "stake"
    ]


def test_top20_candidate_recall_preserves_exact_dialogue() -> None:
    if not _database_available():
        return
    spec = build_rule_spec('"Memory override in progress"')
    candidates, trace = retrieve_candidates(DATABASE, spec, limit=20)
    assert candidates[0].scene_id == "scene_0002"
    assert trace["routes"] == ["exact_dialogue"]


def test_rule_verifier_requires_all_requested_evidence() -> None:
    if not _database_available():
        return
    spec = build_rule_spec("man with mechanical ocular implant speaking")
    candidates, _ = retrieve_candidates(DATABASE, spec, limit=20)
    results = [RuleEvidenceVerifier().verify(spec, item) for item in candidates]
    supported = [item.scene_id for item in results if item.verdict == "supported"]
    assert supported == ["scene_0016"]
    assert results[0].components.evidence_coverage == 1.0


def test_temporal_then_ordinal_is_deterministic() -> None:
    if not _database_available():
        return
    spec = build_rule_spec("mechanical ocular implant after scoped rifle")
    candidates, _ = retrieve_candidates(DATABASE, spec, limit=20)
    first, trace_one = resolve_temporal_constraints(DATABASE, spec, candidates)
    second, trace_two = resolve_temporal_constraints(DATABASE, spec, candidates)
    assert [item.scene_id for item in first] == ["scene_0016"]
    assert first == second
    assert trace_one == trace_two


def test_negative_constraint_creates_explicit_contradiction() -> None:
    if not _database_available():
        return
    spec = build_rule_spec("woman outdoors not distressed")
    candidates, _ = retrieve_candidates(DATABASE, spec, limit=20)
    result = RuleEvidenceVerifier().verify(spec, candidates[0])
    if "distressed" in json.dumps(candidates[0].model_dump()).lower():
        assert result.verdict == "unsupported"
        assert result.contradictions
