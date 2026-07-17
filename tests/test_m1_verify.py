from pathlib import Path

from shotseek.m1_verify import verify_m1_completion


def test_golden_m1_completion_without_recursive_repository_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    m1a = root / "runs" / "m1a" / "20260717-m1a-v1"
    m1b = root / "runs" / "m1b" / "20260717-m1b-v1"
    m1c = root / "runs" / "m1c" / "20260717-m1c-v1"
    if not all(path.is_dir() for path in (m1a, m1b, m1c)):
        return
    result = verify_m1_completion(
        project_root=root,
        m1a_dir=m1a,
        m1b_dir=m1b,
        m1c_dir=m1c,
        run_repository_checks=False,
    )
    assert result["status"] == "pass"
    assert result["headline_metrics"]["recall_at_1"] == 1.0
