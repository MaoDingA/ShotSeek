"""One-command audit for the complete M2 Agentic Retrieval release."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from shotseek.api import create_app
from shotseek.m0 import ensure_within_project
from shotseek.m1_verify import verify_m1_completion


def _load(path: Path) -> Any:
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
        "output_tail": output[-3000:],
    }


def verify_m2_completion(
    *,
    project_root: Path,
    evaluation_dir: Path,
    run_repository_checks: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    evaluation_path = ensure_within_project(root, evaluation_dir)
    evaluation = _load(evaluation_path / "evaluation.json")
    ablation = _load(evaluation_path / "ablation.json")
    m2a_live = _load(
        root / "runs" / "m2a" / "live-planner-v1" / "run_report.json"
    )
    m2a_fixture = _load(
        root / "runs" / "m2a" / "fixture-planner-v1" / "run_report.json"
    )
    m2b_live = _load(
        root / "runs" / "m2b" / "live-verifier-v1" / "run_report.json"
    )
    m2b_fixture = _load(
        root / "runs" / "m2b" / "fixture-verifier-v1" / "run_report.json"
    )
    m1 = verify_m1_completion(
        project_root=root,
        m1a_dir=root / "runs" / "m1a" / "latest",
        m1b_dir=root / "runs" / "m1b" / "latest",
        m1c_dir=root / "runs" / "m1c" / "latest",
        run_repository_checks=False,
    )

    app = create_app(
        database_path=root / "runs" / "m1c" / "latest" / "search.sqlite3",
        manifest_path=root / "runs" / "m1a" / "latest" / "manifest.json",
        trace_dir=root / "runs" / "m2" / "completion-api-traces",
    )
    with TestClient(app) as client:
        health = client.get("/health")
        search = client.post(
            "/search",
            json={
                "query": "mechanical ocular implant",
                "planner_mode": "rule",
                "verifier_mode": "rule",
            },
        )
        trace_id = (
            search.json()["trace"]["trace_id"]
            if search.status_code == 200
            else "missing"
        )
        trace = client.get(f"/traces/{trace_id}")

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

    systems = ablation["systems"]
    gates = {
        "m1_compatibility_pass": m1["status"] == "pass",
        "m2a_live_planner_pass": (
            m2a_live["status"] == "pass"
            and m2a_live["planner_status"] == "LIVE"
            and m2a_live["network_calls"] == 1
        ),
        "m2a_fixture_offline_pass": (
            m2a_fixture["status"] == "pass"
            and m2a_fixture["planner_status"] == "CACHED"
            and m2a_fixture["network_calls"] == 0
        ),
        "m2b_live_verifier_pass": (
            m2b_live["status"] == "pass"
            and m2b_live["verifier_status"] == "LIVE"
            and m2b_live["network_calls"] == 1
            and m2b_live["direct_evidence"] is True
        ),
        "m2b_fixture_offline_pass": (
            m2b_fixture["status"] == "pass"
            and m2b_fixture["verifier_status"] == "CACHED"
            and m2b_fixture["network_calls"] == 0
        ),
        "m2_evaluation_all_gates_pass": (
            evaluation["pass"] is True
            and all(evaluation["gates"].values())
        ),
        "m2_ablation_complete": set(systems) == {
            "m1_fts_baseline",
            "m2_recall_without_verification",
            "m2_full_agent",
        },
        "api_health_pass": (
            health.status_code == 200
            and health.json()["status"] == "ok"
        ),
        "api_search_pass": (
            search.status_code == 200
            and search.json()["trace"]["final_scene_ids"]
            == ["scene_0016"]
        ),
        "api_trace_pass": trace.status_code == 200,
    }
    gates.update(
        {
            f"repository_{name}_pass": bool(result["pass"])
            for name, result in repository_checks.items()
        }
    )
    return {
        "schema_version": "m2-completion-audit-v1",
        "status": "pass" if all(gates.values()) else "failed",
        "gates": gates,
        "repository_checks": repository_checks,
        "headline_metrics": {
            **{
                key: evaluation["metrics"][key]
                for key in (
                    "query_count",
                    "recall_at_1",
                    "recall_at_3",
                    "mrr",
                    "candidate_recall_at_20",
                    "planner_accuracy",
                    "verifier_precision",
                    "evidence_support_rate",
                    "exact_dialogue_recall_at_1",
                    "negative_high_confidence_false_positive_count",
                    "fallback_success_rate",
                )
            },
            "query_p95_ms": evaluation["metrics"]["phase_latency_ms"][
                "total"
            ]["p95"],
            "m1_fts_recall_at_1": systems["m1_fts_baseline"][
                "recall_at_1"
            ],
            "recall_without_verification_at_1": systems[
                "m2_recall_without_verification"
            ]["recall_at_1"],
        },
    }
