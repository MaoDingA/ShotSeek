"""Deterministic JSON, SRT, XML and CMX3600 EDL delivery exports."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Literal, Sequence

ExportFormat = Literal["json", "srt", "xml", "edl"]


@dataclass(frozen=True)
class ExportDocument:
    content: bytes
    media_type: str
    extension: str


def _srt_timecode(milliseconds: int) -> str:
    hours, remainder = divmod(max(0, milliseconds), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _frames(milliseconds: int, fps: int) -> int:
    return max(0, round(milliseconds * fps / 1_000))


def _edl_timecode(frame: int, fps: int) -> str:
    hours, remainder = divmod(max(0, frame), fps * 3_600)
    minutes, remainder = divmod(remainder, fps * 60)
    seconds, frames = divmod(remainder, fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _plain_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def _ordered_scenes(scenes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(scenes, key=lambda item: (int(item["start_ms"]), item["scene_id"]))
    identifiers = [item["scene_id"] for item in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scene identifiers must be unique")
    for scene in ordered:
        if int(scene["end_ms"]) <= int(scene["start_ms"]):
            raise ValueError(f"invalid scene range: {scene['scene_id']}")
    return ordered


def _json_export(
    scenes: list[dict[str, Any]], *, video_id: str, source_name: str, fps: float
) -> ExportDocument:
    payload = {
        "schema_version": "shotseek-delivery-v1",
        "video_id": video_id,
        "source_name": source_name,
        "fps": fps,
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    return ExportDocument(content, "application/json; charset=utf-8", "json")


def _srt_export(scenes: list[dict[str, Any]]) -> ExportDocument:
    blocks: list[str] = []
    for index, scene in enumerate(scenes, start=1):
        dialogue = _plain_text(scene.get("dialogue"))
        summary = _plain_text(scene.get("summary"))
        label = dialogue or summary or scene["scene_id"]
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_srt_timecode(int(scene['start_ms']))} --> "
                    f"{_srt_timecode(int(scene['end_ms']))}",
                    label,
                ]
            )
        )
    content = ("\n\n".join(blocks) + ("\n" if blocks else "")).encode()
    return ExportDocument(content, "application/x-subrip; charset=utf-8", "srt")


def _xml_export(
    scenes: list[dict[str, Any]], *, video_id: str, source_name: str, fps: float
) -> ExportDocument:
    root = ET.Element(
        "shotseek-delivery",
        {"version": "1", "video-id": video_id, "source": source_name, "fps": str(fps)},
    )
    for scene in scenes:
        node = ET.SubElement(
            root,
            "scene",
            {
                "id": str(scene["scene_id"]),
                "start-ms": str(scene["start_ms"]),
                "end-ms": str(scene["end_ms"]),
                "start-frame": str(scene.get("start_frame", "")),
                "end-frame": str(scene.get("end_frame", "")),
                "confidence": str(scene.get("confidence", "")),
            },
        )
        ET.SubElement(node, "summary").text = _plain_text(scene.get("summary"))
        ET.SubElement(node, "dialogue").text = _plain_text(scene.get("dialogue"))
        evidence = ET.SubElement(node, "evidence")
        for reference in scene.get("evidence_refs", []):
            ET.SubElement(
                evidence,
                "ref",
                {"kind": str(reference["kind"]), "id": str(reference["evidence_id"])},
            )
    ET.indent(root, space="  ")
    content = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    return ExportDocument(content, "application/xml; charset=utf-8", "xml")


def _edl_export(
    scenes: list[dict[str, Any]], *, source_name: str, fps: float
) -> ExportDocument:
    integer_fps = round(fps)
    if integer_fps not in {24, 25, 30} or not math.isclose(fps, integer_fps, abs_tol=0.01):
        raise ValueError("CMX3600 export currently requires 24, 25 or 30 fps CFR media")
    reel = "SHOTSEEK"
    record_cursor = integer_fps * 3_600
    lines = [
        f"TITLE: {_plain_text(source_name)[:60] or 'SHOTSEEK'}",
        "FCM: NON-DROP FRAME",
        "",
    ]
    for event, scene in enumerate(scenes, start=1):
        source_in = _frames(int(scene["start_ms"]), integer_fps)
        source_out = max(source_in + 1, _frames(int(scene["end_ms"]), integer_fps))
        duration = source_out - source_in
        record_out = record_cursor + duration
        lines.append(
            f"{event:03d}  {reel:<8} V     C        "
            f"{_edl_timecode(source_in, integer_fps)} "
            f"{_edl_timecode(source_out, integer_fps)} "
            f"{_edl_timecode(record_cursor, integer_fps)} "
            f"{_edl_timecode(record_out, integer_fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {_plain_text(source_name)}")
        lines.append(
            f"* SHOTSEEK {scene['scene_id']}: {_plain_text(scene.get('summary'))[:160]}"
        )
        lines.append("")
        record_cursor = record_out
    content = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    return ExportDocument(content, "text/plain; charset=utf-8", "edl")


def render_export(
    format: ExportFormat,
    scenes: Sequence[dict[str, Any]],
    *,
    video_id: str,
    source_name: str,
    fps: float,
) -> ExportDocument:
    """Render selected scenes without mutating the source index."""
    ordered = _ordered_scenes(scenes)
    if not ordered:
        raise ValueError("at least one scene is required for export")
    if format == "json":
        return _json_export(ordered, video_id=video_id, source_name=source_name, fps=fps)
    if format == "srt":
        return _srt_export(ordered)
    if format == "xml":
        return _xml_export(ordered, video_id=video_id, source_name=source_name, fps=fps)
    if format == "edl":
        return _edl_export(ordered, source_name=source_name, fps=fps)
    raise ValueError(f"unsupported export format: {format}")
