import json
from pathlib import Path

from shotseek.m1c import run_m1c, verify_m1c
from shotseek.retrieval.query_rules import plan_query
from shotseek.retrieval.sqlite_index import search


def test_rule_query_planner_extracts_dialogue_and_visual_terms() -> None:
    spec = plan_query('"You have your robotics" young man outdoors')
    assert spec.quoted_text == "you have your robotics"
    assert spec.terms == ["young", "man", "outdoors"]
    assert spec.temporal_relation is None


def test_rule_query_planner_extracts_after_relation() -> None:
    spec = plan_query("机械义眼 之后 瞄准步枪")
    assert spec.terms == ["mechanical", "ocular", "implant"]
    assert spec.anchor_terms == ["scoped", "rifle"]
    assert spec.temporal_relation == "after"


def test_golden_m1c_hits_all_targets() -> None:
    root = Path(__file__).resolve().parents[1]
    m1a = root / "runs" / "m1a" / "20260717-m1a-v1"
    m1b = root / "runs" / "m1b" / "20260717-m1b-v1"
    if not m1a.is_dir() or not m1b.is_dir():
        return
    output = run_m1c(
        project_root=root,
        m1a_dir=m1a,
        m1b_dir=m1b,
        output_dir=root / "runs" / "tests" / "m1c",
    )
    assert verify_m1c(output)["status"] == "pass"
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["metrics"]["recall_at_1"] == 1.0
    assert evaluation["metrics"]["recall_at_3"] == 1.0
    hits = search(output / "search.sqlite3", "mechanical ocular implant")
    assert hits[0].scene_id == "scene_0016"
