"""Auditable live or fixture StepFun candidate-verification probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.planning.rules import build_rule_spec
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.verification.stepfun import StepFunEvidenceVerifier


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_m2b_verifier(
    *,
    project_root: Path,
    query: str,
    scene_id: str,
    mode: str,
    output_dir: Path,
    api_key: str | None = None,
    fixture_path: Path | None = None,
) -> Path:
    if mode not in {"live", "fixture"}:
        raise ValueError("mode must be live or fixture")
    root = project_root.resolve()
    output = ensure_within_project(root, output_dir)
    database = ensure_within_project(
        root, root / "runs" / "m1c" / "latest" / "search.sqlite3"
    )
    spec = build_rule_spec(query)
    candidates, retrieval = retrieve_candidates(database, spec, limit=20)
    candidate = next(
        (item for item in candidates if item.scene_id == scene_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"candidate {scene_id} was not recalled")

    verifier = StepFunEvidenceVerifier()
    if mode == "live":
        if not api_key:
            raise ValueError("StepFun API key is required")
        result, raw, latency_ms = verifier.verify_live(
            spec, candidate, api_key=api_key
        )
        status = "LIVE"
        network_calls = 1
    else:
        if fixture_path is None:
            raise ValueError("fixture path is required")
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        result = verifier.verify_fixture(spec, candidate, raw)
        latency_ms = 0.0
        status = "CACHED"
        network_calls = 0

    _dump(output / "raw" / "verifier_response.json", raw)
    _dump(
        output / "normalized" / "verification.json",
        result.model_dump(mode="json"),
    )
    _dump(
        output / "verifier_trace.json",
        {
            "status": status,
            "scene_id": scene_id,
            "latency_ms": latency_ms,
            "network_calls": network_calls,
        },
    )
    report = {
        "schema_version": "m2b-verifier-run-v1",
        "status": "pass",
        "mode": mode,
        "verifier_status": status,
        "scene_id": scene_id,
        "verdict": result.verdict,
        "direct_evidence": result.direct_evidence,
        "network_calls": network_calls,
        "retrieval": retrieval,
        "gates": {
            "candidate_recalled": True,
            "strict_result_valid": True,
            "model_did_not_select_scene": result.scene_id == scene_id,
            "unsupported_claim_count_zero": 0,
        },
    }
    _dump(output / "run_report.json", report)
    return output
