#!/usr/bin/env python3
"""Run the frozen M0 checklist and exit nonzero until every hard gate passes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shotseek.m0_verify import (
    TEACHER_CHECKS,
    readme_matches_status,
    verify_fixture_bundle,
    verify_git_secret_absence,
    verify_live_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify every ShotSeek M0 hard gate")
    parser.add_argument("--run", type=Path, required=True, help="Live runs/m0/<run_id> directory")
    return parser.parse_args()


def load_project_key() -> str | None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"STEPFUN_API_KEY", "STEP_API_KEY"}:
                return value.strip().strip("\"'") or None
    return os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")


def main() -> int:
    args = parse_args()
    run_dir = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    api_key = load_project_key()

    runtime = verify_live_run(PROJECT_ROOT, run_dir, api_key=api_key)
    fixture = verify_fixture_bundle(PROJECT_ROOT, api_key=api_key)
    tests = subprocess.run(
        [str(PROJECT_ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    test_output = "\n".join(
        line for line in (tests.stdout + tests.stderr).splitlines() if line.strip()
    )
    test_summary = test_output.splitlines()[-1] if test_output else "no pytest output"
    fixture_test_source = (PROJECT_ROOT / "tests/test_m0_fixture.py").read_text(
        encoding="utf-8"
    )
    fixture_offline = (
        tests.returncode == 0
        and "test_fixture_mode_does_not_construct_an_http_client" in fixture_test_source
        and "fixture mode attempted network access" in fixture_test_source
    )

    run_checks = runtime["checks"]
    checks = {
        "golden_video_public_license": run_checks.get("golden_video_public_license", False),
        "video_under_128mb": run_checks.get("video_under_128mb", False),
        "files_api_upload": run_checks.get("files_api_upload", False),
        "structured_visual_events": run_checks.get("structured_visual_events", False),
        "timestamped_asr": run_checks.get("timestamped_asr", False),
        "speaker_info": run_checks.get("speaker_info", False),
        "unified_timeline": run_checks.get("unified_timeline", False),
        "timeline_in_bounds": run_checks.get("timeline_in_bounds", False),
        "raw_and_normalized_separated": run_checks.get(
            "raw_and_normalized_separated", False
        ),
        "fixture_sanitized": fixture["fixture_sanitized"],
        "fixture_offline": fixture_offline,
        "api_key_not_persisted": verify_git_secret_absence(PROJECT_ROOT, api_key),
        "run_report_actual_timings": run_checks.get("run_report_actual_timings", False),
        "automated_tests_passed": tests.returncode == 0,
        "readme_updated": readme_matches_status(
            PROJECT_ROOT, runtime["runtime_complete"]
        ),
    }
    assert tuple(checks) == TEACHER_CHECKS

    integrity_checks = {
        "required_artifacts": run_checks.get("required_artifacts", False),
        "schemas_parse": run_checks.get("schemas_parse", False),
        "run_identity_matches": run_checks.get("run_identity_matches", False),
        "golden_duration_60_90s": run_checks.get("golden_duration_60_90s", False),
        "raw_response_sanitized": run_checks.get("raw_response_sanitized", False),
        "report_counts_match": run_checks.get("report_counts_match", False),
        "report_gates_match_evidence": run_checks.get(
            "report_gates_match_evidence", False
        ),
        "report_status_matches_evidence": run_checks.get(
            "report_status_matches_evidence", False
        ),
        "fixture_live_derived": fixture["fixture_live_derived"],
    }
    complete = all(checks.values()) and all(integrity_checks.values())
    payload = {
        "run_id": runtime["run_id"],
        "complete": complete,
        "teacher_checks": checks,
        "integrity_checks": integrity_checks,
        "failed_teacher_checks": [name for name, passed in checks.items() if not passed],
        "failed_integrity_checks": [
            name for name, passed in integrity_checks.items() if not passed
        ],
        "test_summary": test_summary,
        "diagnostics": runtime["diagnostics"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
