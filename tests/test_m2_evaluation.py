import json
from collections import Counter
from pathlib import Path

from shotseek.evaluation.m2 import run_m2_evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_m2_query_matrix_has_exact_requested_distribution() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "eval" / "m2_queries.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(cases) == 25
    assert Counter(item["category"] for item in cases) == {
        "chinese_synonym": 8,
        "english_synonym": 4,
        "complex_temporal": 5,
        "ordinal": 3,
        "negation": 2,
        "hard_negative": 3,
    }


def test_full_m2_evaluation_passes_offline() -> None:
    database = ROOT / "runs" / "m1c" / "latest" / "search.sqlite3"
    if not database.is_file():
        return
    output = run_m2_evaluation(
        project_root=ROOT,
        output_dir=ROOT / "runs" / "tests" / "m2-evaluation",
    )
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["pass"] is True
    assert evaluation["metrics"]["query_count"] == 40
    assert evaluation["metrics"]["network_call_count"] == 0
