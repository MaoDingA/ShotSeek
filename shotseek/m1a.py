"""Offline M1A orchestration: media truth, shot grid and evidence alignment."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project
from shotseek.media.probe import probe_video_contract
from shotseek.media.schema import M1AManifest
from shotseek.media.shots import build_shot_grid, detect_shot_boundaries
from shotseek.providers.stepfun.asr import normalize_asr_response
from shotseek.providers.stepfun.vision import normalize_vision_bundle
from shotseek.timeline.alignment import align_visual_events, contextualize_utterances
from shotseek.timeline.boundary_audit import (
    build_boundary_audit,
    build_manual_boundary_audit,
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payloads: dict[str, Any]) -> str:
    encoded = json.dumps(
        payloads, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_m1a(
    *,
    project_root: Path,
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.30,
    min_gap_frames: int = 6,
) -> Path:
    root = project_root.resolve()
    output = ensure_within_project(root, output_dir)
    video = probe_video_contract(root, video_path)
    fixture_dir = root / "tests" / "fixtures" / "stepfun"
    vision_path = fixture_dir / "vision_response.sample.json"
    asr_path = fixture_dir / "asr_response.sample.json"
    manifest = M1AManifest(
        input_video_sha256=video.sha256,
        source_fixture_sha256={
            "vision": _sha256(vision_path),
            "asr": _sha256(asr_path),
        },
        detector={
            "name": "ffmpeg_scene",
            "threshold": threshold,
            "min_gap_frames": min_gap_frames,
            "version": "m1a-detector-v1",
        },
    )
    boundaries = detect_shot_boundaries(
        root,
        video_path,
        video,
        threshold=threshold,
        min_gap_frames=min_gap_frames,
    )
    shots = build_shot_grid(video, boundaries)
    visual_events = normalize_vision_bundle(
        _load_json(vision_path), model="step-3.7-flash"
    )
    asr_raw = _load_json(asr_path)
    utterances = normalize_asr_response(asr_raw["result"])
    fps = Fraction(video.fps_num, video.fps_den)
    aligned = align_visual_events(
        visual_events, shots, fps=fps, frame_count=video.frame_count
    )
    contextualized = contextualize_utterances(
        utterances, shots, fps=fps, frame_count=video.frame_count
    )
    audit = build_boundary_audit(
        shots, aligned, contextualized, frame_count=video.frame_count
    )
    annotations = _load_json(root / "eval" / "m1a_boundary_annotations.json")
    manual_audit = build_manual_boundary_audit(aligned, annotations)
    payloads = {
        "manifest.json": manifest.model_dump(mode="json"),
        "video_info.json": video.model_dump(mode="json"),
        "shot_boundaries.json": [
            item.model_dump(mode="json") for item in boundaries
        ],
        "shots.json": [item.model_dump(mode="json") for item in shots],
        "aligned_visual_events.json": [
            item.model_dump(mode="json") for item in aligned
        ],
        "contextualized_utterances.json": [
            item.model_dump(mode="json") for item in contextualized
        ],
        "boundary_audit.json": audit,
        "manual_boundary_audit.json": manual_audit,
    }
    all_pass = bool(audit["pass"]) and bool(manual_audit["pass"])
    report = {
        "schema_version": "m1a-run-report-v1",
        "status": "pass" if all_pass else "failed",
        "network_calls": 0,
        "deterministic_payload_sha256": _payload_sha256(payloads),
        "artifacts": sorted(payloads),
        "metrics": {**audit["metrics"], **manual_audit["metrics"]},
        "gates": {**audit["gates"], **manual_audit["gates"]},
    }
    payloads["run_report.json"] = report
    for filename, payload in payloads.items():
        _dump_json(output / filename, payload)
    if not all_pass:
        raise RuntimeError("M1A boundary audit failed")
    return output


def verify_m1a(output_dir: Path) -> dict[str, Any]:
    required = {
        "manifest.json",
        "video_info.json",
        "shot_boundaries.json",
        "shots.json",
        "aligned_visual_events.json",
        "contextualized_utterances.json",
        "boundary_audit.json",
        "manual_boundary_audit.json",
        "run_report.json",
    }
    missing = sorted(
        name for name in required if not (output_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing M1A artifacts: {missing}")
    report = _load_json(output_dir / "run_report.json")
    audit = _load_json(output_dir / "boundary_audit.json")
    manual_audit = _load_json(output_dir / "manual_boundary_audit.json")
    checks = {
        "artifact_set_complete": not missing,
        "run_status_pass": report.get("status") == "pass",
        "network_calls_zero": report.get("network_calls") == 0,
        "boundary_audit_pass": audit.get("pass") is True,
        "manual_boundary_audit_pass": manual_audit.get("pass") is True,
        "manual_review_count_at_least_8": (
            manual_audit["metrics"].get("reviewed_event_count", 0) >= 8
        ),
        "visual_count_23": audit["metrics"].get("visual_event_count") == 23,
        "utterance_count_7": audit["metrics"].get("utterance_count") == 7,
    }
    return {
        "status": "pass" if all(checks.values()) else "failed",
        "checks": checks,
    }
