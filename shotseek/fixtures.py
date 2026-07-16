"""Sanitize successful live M0 responses into deterministic development fixtures."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shotseek.m0_verify import verify_live_run


def _sanitize_value(value: Any, *, api_key: str | None = None, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            lowered = key.lower()
            child_path = (*path, lowered)
            if lowered in {"authorization", "api_key", "apikey"}:
                result[key] = "<redacted>"
            elif lowered in {"url", "uri"}:
                result[key] = "<redacted_url>"
            elif lowered in {"task_id", "request_id", "trace_id", "session_id"}:
                result[key] = f"{lowered}_fixture_redacted"
            elif lowered in {"created", "created_at"} and isinstance(child, (int, float)):
                result[key] = 0
            elif lowered == "timestamp" and "meta" in child_path:
                result[key] = 0
            else:
                result[key] = _sanitize_value(
                    child, api_key=api_key, path=child_path
                )
        return result
    if isinstance(value, list):
        return [
            _sanitize_value(item, api_key=api_key, path=path) for item in value
        ]
    if isinstance(value, str):
        if api_key and api_key in value:
            value = value.replace(api_key, "<redacted>")
        if value.startswith(("/home/", "/Users/")):
            return "<redacted_path>"
        return value
    return value


def sanitize_stepfun_file(raw: Any, *, api_key: str | None = None) -> Any:
    result = _sanitize_value(copy.deepcopy(raw), api_key=api_key)
    if isinstance(result, dict):
        for section in ("upload", "final"):
            item = result.get(section)
            if isinstance(item, dict) and "id" in item:
                item["id"] = "file_fixture_redacted"
    return result


def sanitize_vision_response(raw: Any, *, api_key: str | None = None) -> Any:
    result = _sanitize_value(copy.deepcopy(raw), api_key=api_key)
    if isinstance(result, dict) and "id" in result:
        result["id"] = "chatcmpl_fixture_redacted"
    return result


def sanitize_asr_response(raw: Any, *, api_key: str | None = None) -> Any:
    return _sanitize_value(copy.deepcopy(raw), api_key=api_key)


def update_fixtures_from_live_run(
    project_root: Path,
    run_dir: Path,
    *,
    api_key: str | None = None,
) -> list[Path]:
    root = project_root.resolve()
    run = run_dir.resolve()
    verification = verify_live_run(root, run, api_key=api_key)
    if not verification["runtime_complete"] or verification["failed_checks"]:
        failed = ", ".join(verification["failed_checks"])
        raise ValueError(f"live run is not M0-complete: {failed}")

    raw_dir = run / "raw"
    fixture_dir = root / "tests/fixtures/stepfun"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "stepfun_file.sample.json": sanitize_stepfun_file(
            json.loads((raw_dir / "stepfun_file.json").read_text(encoding="utf-8")),
            api_key=api_key,
        ),
        "vision_response.sample.json": sanitize_vision_response(
            json.loads((raw_dir / "vision_response.json").read_text(encoding="utf-8")),
            api_key=api_key,
        ),
        "asr_response.sample.json": sanitize_asr_response(
            json.loads((raw_dir / "asr_response.json").read_text(encoding="utf-8")),
            api_key=api_key,
        ),
    }
    written: list[Path] = []
    for filename, payload in outputs.items():
        destination = fixture_dir / filename
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)

    existing_provenance_path = fixture_dir / "fixture_provenance.sample.json"
    provenance = (
        json.loads(existing_provenance_path.read_text(encoding="utf-8"))
        if existing_provenance_path.is_file()
        else {}
    )
    generated_at = datetime.now(UTC).isoformat()
    for filename in outputs:
        provenance[filename] = {
            "kind": "sanitized_live_response",
            "live_derived": True,
            "provider": "stepfun",
            "source_run_id": verification["run_id"],
            "generated_at": generated_at,
        }
    existing_provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(existing_provenance_path)
    return written
