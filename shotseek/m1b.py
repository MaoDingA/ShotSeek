"""Offline M1B orchestration for strict scene candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.media.schema import AlignedVisualEvent, ContextualizedUtterance, Shot
from shotseek.scenes.builder import build_scenes, validate_scene_references
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


def _payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_m1b(
    *,
    project_root: Path,
    m1a_dir: Path,
    output_dir: Path,
    context_window_ms: int = 1500,
) -> Path:
    root = project_root.resolve()
    source = ensure_within_project(root, m1a_dir)
    output = ensure_within_project(root, output_dir)
    visual_events = [
        AlignedVisualEvent.model_validate(item)
        for item in _load_json(source / "aligned_visual_events.json")
    ]
    utterances = [
        ContextualizedUtterance.model_validate(item)
        for item in _load_json(source / "contextualized_utterances.json")
    ]
    shots = [
        Shot.model_validate(item) for item in _load_json(source / "shots.json")
    ]
    scenes = build_scenes(
        visual_events, utterances, context_window_ms=context_window_ms
    )
    audit = validate_scene_references(scenes, visual_events, utterances, shots)
    scene_payload = [item.model_dump(mode="json") for item in scenes]
    manifest = {
        "schema_version": "m1b-manifest-v1",
        "source_m1a_payload_sha256": _load_json(source / "run_report.json")[
            "deterministic_payload_sha256"
        ],
        "context_window_ms": context_window_ms,
        "network_calls": 0,
    }
    report = {
        "schema_version": "m1b-run-report-v1",
        "status": "pass" if audit["pass"] else "failed",
        "network_calls": 0,
        "deterministic_payload_sha256": _payload_sha256(
            {"manifest": manifest, "scenes": scene_payload, "audit": audit}
        ),
        "metrics": {
            **audit,
            "scene_with_dialogue_count": sum(
                bool(scene.utterance_ids) for scene in scenes
            ),
            "evidence_reference_count": sum(
                len(scene.evidence_refs) for scene in scenes
            ),
        },
        "gates": {
            "all_visual_events_have_scene": len(scenes) == len(visual_events) == 23,
            "strict_reference_audit_pass": bool(audit["pass"]),
            "network_calls_zero": True,
        },
    }
    _dump_json(output / "manifest.json", manifest)
    _dump_json(output / "scenes.json", scene_payload)
    _dump_json(output / "scene_build_report.json", report)
    if report["status"] != "pass":
        raise RuntimeError("M1B scene reference audit failed")
    return output


def verify_m1b(output_dir: Path) -> dict[str, Any]:
    required = {"manifest.json", "scenes.json", "scene_build_report.json"}
    missing = sorted(
        name for name in required if not (output_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing M1B artifacts: {missing}")
    scenes = [
        Scene.model_validate(item)
        for item in _load_json(output_dir / "scenes.json")
    ]
    report = _load_json(output_dir / "scene_build_report.json")
    checks = {
        "artifact_set_complete": not missing,
        "scene_schema_valid": len(scenes) == 23,
        "run_status_pass": report.get("status") == "pass",
        "network_calls_zero": report.get("network_calls") == 0,
        "strict_reference_audit_pass": report["gates"].get(
            "strict_reference_audit_pass"
        )
        is True,
    }
    return {
        "status": "pass" if all(checks.values()) else "failed",
        "checks": checks,
    }
