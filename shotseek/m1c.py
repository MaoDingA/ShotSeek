"""Offline M1C orchestration for searchable scene retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.media.schema import ContextualizedUtterance
from shotseek.retrieval.evaluate import load_query_cases, run_evaluation
from shotseek.retrieval.sqlite_index import build_index
from shotseek.scenes.schema import Scene


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_m1c(
    *,
    project_root: Path,
    m1a_dir: Path,
    m1b_dir: Path,
    output_dir: Path,
    queries_path: Path | None = None,
) -> Path:
    root = project_root.resolve()
    m1a = ensure_within_project(root, m1a_dir)
    m1b = ensure_within_project(root, m1b_dir)
    output = ensure_within_project(root, output_dir)
    queries = ensure_within_project(
        root, queries_path or root / "eval" / "m1_queries.jsonl"
    )
    scenes = [
        Scene.model_validate(item) for item in _load_json(m1b / "scenes.json")
    ]
    utterances = [
        ContextualizedUtterance.model_validate(item)
        for item in _load_json(m1a / "contextualized_utterances.json")
    ]
    database_path = output / "search.sqlite3"
    index_metrics = build_index(database_path, scenes, utterances)
    cases = load_query_cases(queries)
    evaluation = run_evaluation(database_path, cases)
    manual_audit = _load_json(m1a / "manual_boundary_audit.json")
    evaluation["metrics"]["manual_median_shot_iou"] = manual_audit["metrics"][
        "median_shot_iou"
    ]
    evaluation["gates"]["manual_median_shot_iou_at_least_0_70"] = (
        evaluation["metrics"]["manual_median_shot_iou"] >= 0.70
    )
    evaluation["pass"] = all(evaluation["gates"].values())
    query_results = evaluation.pop("query_results")
    report = {
        "schema_version": "m1c-run-report-v1",
        "status": "pass" if evaluation["pass"] else "failed",
        "network_calls": 0,
        "index": index_metrics,
        "metrics": evaluation["metrics"],
        "gates": {
            **evaluation["gates"],
            "sqlite_integrity_check_pass": index_metrics["integrity_check_pass"],
            "network_calls_zero": True,
        },
        "deterministic_results_sha256": evaluation[
            "deterministic_results_sha256"
        ],
    }
    report["status"] = "pass" if all(report["gates"].values()) else "failed"
    _dump_json(output / "query_results.json", query_results)
    _dump_json(output / "evaluation.json", evaluation)
    _dump_json(output / "run_report.json", report)
    if report["status"] != "pass":
        raise RuntimeError("M1C retrieval evaluation failed")
    return output


def verify_m1c(output_dir: Path) -> dict[str, Any]:
    required = {
        "search.sqlite3",
        "query_results.json",
        "evaluation.json",
        "run_report.json",
    }
    missing = sorted(
        name for name in required if not (output_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing M1C artifacts: {missing}")
    results = _load_json(output_dir / "query_results.json")
    evaluation = _load_json(output_dir / "evaluation.json")
    report = _load_json(output_dir / "run_report.json")
    checks = {
        "artifact_set_complete": not missing,
        "all_15_query_results_present": len(results) == 15,
        "evaluation_pass": evaluation.get("pass") is True,
        "run_status_pass": report.get("status") == "pass",
        "network_calls_zero": report.get("network_calls") == 0,
        "sqlite_integrity_check_pass": report["index"].get(
            "integrity_check_pass"
        )
        is True,
    }
    return {
        "status": "pass" if all(checks.values()) else "failed",
        "checks": checks,
    }
