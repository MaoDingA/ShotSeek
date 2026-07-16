"""Independent M0 completion audit against the frozen teacher checklist."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from shotseek.schemas import EvidenceSpan, RunManifest, RunReport, Utterance, VisualEvent
from shotseek.timeline.validate import validate_evidence_timeline

REQUIRED_ARTIFACTS = (
    "manifest.json",
    "raw/stepfun_file.json",
    "raw/vision_response.json",
    "raw/asr_response.json",
    "normalized/visual_events.json",
    "normalized/utterances.json",
    "normalized/evidence_timeline.json",
    "run_report.json",
)

TEACHER_CHECKS = (
    "golden_video_public_license",
    "video_under_128mb",
    "files_api_upload",
    "structured_visual_events",
    "timestamped_asr",
    "speaker_info",
    "unified_timeline",
    "timeline_in_bounds",
    "raw_and_normalized_separated",
    "fixture_sanitized",
    "fixture_offline",
    "api_key_not_persisted",
    "run_report_actual_timings",
    "automated_tests_passed",
    "readme_updated",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path must stay inside project root: {resolved}") from exc
    return resolved


def _files_api_succeeded(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    final = raw.get("final")
    return (
        isinstance(final, dict)
        and bool(str(final.get("id", "")).strip())
        and str(final.get("status", "")).lower() in {"success", "processed"}
    )


def verify_live_run(
    project_root: Path,
    run_dir: Path,
    api_key: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    run = _inside(root, run_dir)
    checks: dict[str, bool] = {}
    diagnostics: list[str] = []

    missing = [name for name in REQUIRED_ARTIFACTS if not (run / name).is_file()]
    checks["required_artifacts"] = not missing
    if missing:
        diagnostics.append("missing artifacts: " + ", ".join(missing))
        return {
            "run_id": run.name,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "diagnostics": diagnostics,
            "runtime_complete": False,
        }

    try:
        manifest = RunManifest.model_validate(_load_json(run / "manifest.json"))
        report = RunReport.model_validate(_load_json(run / "run_report.json"))
        visual_events = [
            VisualEvent.model_validate(item)
            for item in _load_json(run / "normalized/visual_events.json")
        ]
        utterances = [
            Utterance.model_validate(item)
            for item in _load_json(run / "normalized/utterances.json")
        ]
        evidence = [
            EvidenceSpan.model_validate(item)
            for item in _load_json(run / "normalized/evidence_timeline.json")
        ]
        file_raw = _load_json(run / "raw/stepfun_file.json")
        raw_text = "\n".join(
            (run / name).read_text(encoding="utf-8")
            for name in REQUIRED_ARTIFACTS
            if name.startswith("raw/")
        )
    except Exception as exc:
        checks["schemas_parse"] = False
        diagnostics.append(f"schema parse failed: {type(exc).__name__}: {exc}")
        return {
            "run_id": run.name,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "diagnostics": diagnostics,
            "runtime_complete": False,
        }

    checks["schemas_parse"] = True
    checks["run_identity_matches"] = (
        manifest.mode == "live"
        and report.mode == "live"
        and manifest.run_id == report.run_id == run.name
    )
    checks["golden_duration_60_90s"] = 60_000 <= manifest.video.duration_ms <= 90_000
    license_text = (root / "samples/README.md").read_text(encoding="utf-8")
    checks["golden_video_public_license"] = (
        manifest.video.path.startswith("samples/")
        and "Creative Commons Attribution 3.0" in license_text
    )
    checks["video_under_128mb"] = manifest.video.bytes < 128 * 1024 * 1024
    checks["files_api_upload"] = _files_api_succeeded(file_raw)
    checks["structured_visual_events"] = bool(visual_events)
    checks["timestamped_asr"] = bool(utterances) and all(
        utterance.start_ms >= 0 and utterance.end_ms > utterance.start_ms
        for utterance in utterances
    )
    checks["speaker_info"] = bool(utterances) and all(
        bool(utterance.speaker_id) for utterance in utterances
    )

    try:
        validate_evidence_timeline(evidence, manifest.video.duration_ms)
        timeline_valid = True
    except ValueError as exc:
        timeline_valid = False
        diagnostics.append(str(exc))
    visual_ids = {item.event_id for item in visual_events}
    utterance_ids = {item.utterance_id for item in utterances}
    source_refs_resolve = all(
        (item.kind.value == "visual" and item.source_ref in visual_ids)
        or (item.kind.value == "dialogue" and item.source_ref in utterance_ids)
        for item in evidence
    )
    normalized_inputs_in_bounds = all(
        0
        <= item.source_start_ms + item.approx_start_ms
        < item.source_start_ms + item.approx_end_ms
        <= manifest.video.duration_ms
        for item in visual_events
    ) and all(
        0 <= item.start_ms < item.end_ms <= manifest.video.duration_ms
        for item in utterances
    )
    checks["unified_timeline"] = (
        bool(visual_events)
        and bool(utterances)
        and len(evidence) == len(visual_events) + len(utterances)
        and {item.kind.value for item in evidence} == {"visual", "dialogue"}
        and source_refs_resolve
    )
    checks["timeline_in_bounds"] = timeline_valid and normalized_inputs_in_bounds
    checks["raw_and_normalized_separated"] = all(
        (run / name).is_file()
        for name in REQUIRED_ARTIFACTS
        if name.startswith(("raw/", "normalized/"))
    )
    checks["raw_response_sanitized"] = (
        "Authorization" not in raw_text
        and "Bearer " not in raw_text
        and "/home/" not in raw_text
        and "/Users/" not in raw_text
        and (not api_key or api_key not in raw_text)
    )

    expected_counts = {
        "visual_event_count": len(visual_events),
        "utterance_count": len(utterances),
        "evidence_count": len(evidence),
    }
    checks["report_counts_match"] = all(
        int(report.metrics.get(name, -1)) == count
        for name, count in expected_counts.items()
    )
    timing_names = (
        "file_upload_ms",
        "vision_request_ms",
        "asr_submit_ms",
        "asr_total_ms",
        "normalization_ms",
    )
    timing_values = [report.metrics.get(name) for name in timing_names]
    checks["run_report_actual_timings"] = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        for value in timing_values
    ) and any(float(value) > 0 for value in timing_values)

    declared_runtime = {
        name: checks[name]
        for name in (
            "golden_video_public_license",
            "video_under_128mb",
            "files_api_upload",
            "structured_visual_events",
            "timestamped_asr",
            "speaker_info",
            "unified_timeline",
            "timeline_in_bounds",
            "raw_and_normalized_separated",
        )
    }
    checks["report_gates_match_evidence"] = report.gates == declared_runtime
    runtime_complete = all(declared_runtime.values())
    checks["report_status_matches_evidence"] = (
        report.m0_complete is runtime_complete
        and report.status == ("pass" if runtime_complete else "partial")
    )

    return {
        "run_id": run.name,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "diagnostics": diagnostics,
        "runtime_complete": runtime_complete,
    }


def verify_fixture_bundle(project_root: Path, api_key: str | None = None) -> dict[str, bool]:
    root = project_root.resolve()
    fixture_dir = root / "tests/fixtures/stepfun"
    required = (
        "vision_response.sample.json",
        "asr_sse_response.sample.json",
        "fixture_provenance.sample.json",
    )
    files_exist = all((fixture_dir / name).is_file() for name in required)
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(fixture_dir.glob("*.json"))
    )
    forbidden = (
        r"/home/",
        r"/Users/",
        r"X-Amz-(?:Credential|Signature)",
        r"https?://",
        r"Authorization",
        r"Bearer\s+[A-Za-z0-9._-]+",
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    )
    sanitized = files_exist and all(
        re.search(pattern, combined, re.IGNORECASE) is None for pattern in forbidden
    ) and (not api_key or api_key not in combined)

    live_derived = False
    if files_exist:
        provenance = _load_json(fixture_dir / "fixture_provenance.sample.json")
        live_derived = all(
            (fixture_dir / name).is_file()
            and isinstance(provenance.get(name), dict)
            and provenance[name].get("live_derived") is True
            for name in (
                "stepfun_file.sample.json",
                "vision_response.sample.json",
                "asr_response.sample.json",
            )
        )
    return {
        "fixture_sanitized": sanitized,
        "fixture_live_derived": live_derived,
    }


def verify_git_secret_absence(project_root: Path, api_key: str | None) -> bool:
    root = project_root.resolve()
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if listed.returncode != 0:
        return False
    suspicious = re.compile(
        rb"(?:Bearer\s+|STEPFUN_API_KEY\s*=\s*)[A-Za-z0-9._-]{40,}"
    )
    for raw_name in listed.stdout.split(b"\0"):
        if not raw_name:
            continue
        candidate = root / raw_name.decode("utf-8")
        if not candidate.is_file():
            continue
        data = candidate.read_bytes()
        if suspicious.search(data) or (api_key and api_key.encode("utf-8") in data):
            return False
    if not api_key:
        return True
    history = subprocess.run(
        ["git", "log", "--all", f"-S{api_key}", "--format=%H", "--"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return history.returncode == 0 and not history.stdout.strip()


def readme_matches_status(project_root: Path, runtime_complete: bool) -> bool:
    text = (project_root.resolve() / "README.md").read_text(encoding="utf-8")
    status_line = next(
        (line for line in text.splitlines() if "M0 Live 硬门槛" in line), ""
    )
    if runtime_complete:
        return bool(status_line) and "BLOCKED" not in status_line and "通过" in status_line
    return "BLOCKED" in status_line
