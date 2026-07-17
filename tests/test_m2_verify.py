import json
from pathlib import Path

from shotseek.m2_verify import verify_m2_completion

ROOT = Path(__file__).resolve().parents[1]


def test_m2_completion_audit_passes_without_recursive_repository_checks() -> None:
    evaluation = ROOT / "runs" / "m2" / "evaluation-v1"
    required = [
        evaluation / "evaluation.json",
        ROOT / "runs" / "m2a" / "live-planner-v1" / "run_report.json",
        ROOT / "runs" / "m2b" / "live-verifier-v1" / "run_report.json",
    ]
    if not all(path.is_file() for path in required):
        return
    result = verify_m2_completion(
        project_root=ROOT,
        evaluation_dir=evaluation,
        run_repository_checks=False,
    )
    assert result["status"] == "pass"
    assert result["headline_metrics"]["query_count"] == 40
    assert all(result["gates"].values())
