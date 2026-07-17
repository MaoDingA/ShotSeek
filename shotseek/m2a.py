"""Auditable M2A planner runs in rule, fixture, or live mode."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.planning.router import PlannerRouter
from shotseek.planning.schema import PlannerTrace, QuerySpecV2
from shotseek.planning.stepfun import (
    PLANNER_PROMPT_VERSION,
    PLANNER_SCHEMA_VERSION,
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def run_m2a(
    *,
    project_root: Path,
    query: str,
    mode: str,
    output_dir: Path | None = None,
    api_key: str | None = None,
    fixture_path: Path | None = None,
    top_k: int = 3,
) -> Path:
    root = project_root.resolve()
    if mode not in {"rule", "fixture", "live"}:
        raise ValueError("mode must be rule, fixture, or live")
    output = ensure_within_project(
        root, output_dir or root / "runs" / "m2a" / _run_id()
    )
    fixture_response = None
    if mode == "fixture":
        source = ensure_within_project(
            root,
            fixture_path
            or root
            / "tests"
            / "fixtures"
            / "stepfun"
            / "query_planner_response.sample.json",
        )
        fixture_response = _load(source)
    router = PlannerRouter(cache_dir=output / "cache")
    result = router.plan(
        query,
        mode=("rule" if mode == "rule" else "stepfun"),
        top_k=top_k,
        api_key=api_key,
        allow_network=mode == "live",
        fixture_response=fixture_response,
    )
    expected_status = {
        "rule": {"RULE"},
        "fixture": {"CACHED"},
        "live": {"LIVE"},
    }[mode]
    actual_live = result.trace.status in expected_status
    manifest = {
        "schema_version": "m2a-manifest-v1",
        "mode": mode,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "prompt_version": PLANNER_PROMPT_VERSION,
        "query_schema_version": PLANNER_SCHEMA_VERSION,
        "network_allowed": mode == "live",
        "network_calls": 1 if result.trace.status == "LIVE" else 0,
    }
    if result.raw_response is not None:
        _dump(output / "raw" / "planner_response.json", result.raw_response)
    _dump(
        output / "normalized" / "query_spec.json",
        result.query_spec.model_dump(mode="json"),
    )
    _dump(output / "planner_trace.json", result.trace.model_dump(mode="json"))
    _dump(output / "manifest.json", manifest)
    report = {
        "schema_version": "m2a-run-report-v1",
        "status": "pass" if actual_live else "failed",
        "mode": mode,
        "planner_status": result.trace.status,
        "planner": result.trace.planner,
        "fallback_reason": result.trace.fallback_reason,
        "latency_ms": result.trace.latency_ms,
        "query_spec_valid": True,
        "network_calls": manifest["network_calls"],
        "gates": {
            "expected_mode_succeeded": actual_live,
            "query_spec_v2_valid": True,
            "trace_valid": True,
            "fixture_network_calls_zero": mode != "fixture"
            or manifest["network_calls"] == 0,
        },
    }
    _dump(output / "run_report.json", report)
    if report["status"] != "pass":
        raise RuntimeError(f"M2A {mode} run did not use expected planner mode")
    return output


def verify_m2a(output_dir: Path) -> dict[str, Any]:
    required = {
        "manifest.json",
        "normalized/query_spec.json",
        "planner_trace.json",
        "run_report.json",
    }
    missing = sorted(
        name for name in required if not (output_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing M2A artifacts: {missing}")
    spec = QuerySpecV2.model_validate(
        _load(output_dir / "normalized" / "query_spec.json")
    )
    trace = PlannerTrace.model_validate(_load(output_dir / "planner_trace.json"))
    report = _load(output_dir / "run_report.json")
    manifest = _load(output_dir / "manifest.json")
    checks = {
        "query_spec_v2_valid": spec.schema_version == "query-v2",
        "planner_trace_valid": trace.trace_id.startswith("trace_"),
        "run_status_pass": report.get("status") == "pass",
        "fixture_network_calls_zero": manifest["mode"] != "fixture"
        or manifest["network_calls"] == 0,
    }
    return {"status": "pass" if all(checks.values()) else "failed", "checks": checks}
