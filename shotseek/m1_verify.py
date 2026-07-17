"""One-command completion audit for the offline M1 searchable-scene core."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.m1a import verify_m1a
from shotseek.m1b import verify_m1b
from shotseek.m1c import verify_m1c


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_check(command: list[str], root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "pass": completed.returncode == 0,
        "returncode": completed.returncode,
        "output_tail": output[-2000:],
    }


def verify_m1_completion(
    *,
    project_root: Path,
    m1a_dir: Path,
    m1b_dir: Path,
    m1c_dir: Path,
    run_repository_checks: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    m1a = ensure_within_project(root, m1a_dir)
    m1b = ensure_within_project(root, m1b_dir)
    m1c = ensure_within_project(root, m1c_dir)
    stage_results = {
        "m1a": verify_m1a(m1a),
        "m1b": verify_m1b(m1b),
        "m1c": verify_m1c(m1c),
    }
    m1a_report = _load_json(m1a / "run_report.json")
    m1b_report = _load_json(m1b / "scene_build_report.json")
    m1c_report = _load_json(m1c / "run_report.json")
    repository_checks: dict[str, Any] = {}
    if run_repository_checks:
        repository_checks = {
            "pytest": _run_check(
                [str(root / ".venv" / "bin" / "python"), "-m", "pytest", "-q"],
                root,
            ),
            "hygiene": _run_check(
                [
                    str(root / ".venv" / "bin" / "python"),
                    "scripts/check_repository_hygiene.py",
                ],
                root,
            ),
            "diff_check": _run_check(["git", "diff", "--check"], root),
        }
    gates = {
        "m1a_pass": stage_results["m1a"]["status"] == "pass",
        "m1b_pass": stage_results["m1b"]["status"] == "pass",
        "m1c_pass": stage_results["m1c"]["status"] == "pass",
        "network_calls_zero": (
            m1a_report["network_calls"]
            == m1b_report["network_calls"]
            == m1c_report["network_calls"]
            == 0
        ),
        "manual_boundary_gate_pass": m1a_report["gates"][
            "median_shot_iou_at_least_0_70"
        ],
        "scene_reference_gate_pass": m1b_report["gates"][
            "strict_reference_audit_pass"
        ],
        "retrieval_quality_gates_pass": all(m1c_report["gates"].values()),
    }
    gates.update(
        {
            f"repository_{name}_pass": bool(result["pass"])
            for name, result in repository_checks.items()
        }
    )
    return {
        "schema_version": "m1-completion-audit-v1",
        "status": "pass" if all(gates.values()) else "failed",
        "stages": stage_results,
        "gates": gates,
        "repository_checks": repository_checks,
        "headline_metrics": {
            "shots": m1a_report["metrics"]["shot_count"],
            "visual_events": m1a_report["metrics"]["visual_event_count"],
            "utterances": m1a_report["metrics"]["utterance_count"],
            "manual_median_shot_iou": m1a_report["metrics"][
                "median_shot_iou"
            ],
            "scenes": m1b_report["metrics"]["scene_count"],
            "queries": m1c_report["metrics"]["query_count"],
            "recall_at_1": m1c_report["metrics"]["recall_at_1"],
            "recall_at_3": m1c_report["metrics"]["recall_at_3"],
            "query_p95_ms": m1c_report["metrics"]["query_p95_ms"],
        },
    }
