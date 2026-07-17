"""Machine-verifiable M1A boundary and traceability audit."""

from __future__ import annotations

from statistics import median

from shotseek.media.schema import AlignedVisualEvent, ContextualizedUtterance, Shot


def build_manual_boundary_audit(
    visual_events: list[AlignedVisualEvent],
    annotations: dict[str, object],
) -> dict[str, object]:
    predicted = {item.event_id: item for item in visual_events}
    results: list[dict[str, object]] = []
    raw_items = annotations.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("manual boundary annotations must contain an items array")
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError("manual boundary annotation must be an object")
        event_id = str(raw["event_id"])
        if event_id not in predicted:
            raise ValueError(f"manual annotation references unknown event: {event_id}")
        human_start = int(raw["human_start_frame"])
        human_end = int(raw["human_end_frame"])
        if human_start < 0 or human_end <= human_start:
            raise ValueError(f"invalid manual boundary: {event_id}")
        item = predicted[event_id]
        intersection = max(
            0,
            min(item.end_frame, human_end) - max(item.start_frame, human_start),
        )
        union = max(item.end_frame, human_end) - min(item.start_frame, human_start)
        results.append(
            {
                "event_id": event_id,
                "predicted_start_frame": item.start_frame,
                "predicted_end_frame": item.end_frame,
                "human_start_frame": human_start,
                "human_end_frame": human_end,
                "start_boundary_match": item.start_frame == human_start,
                "end_boundary_match": item.end_frame == human_end,
                "start_delta_frames": item.start_frame - human_start,
                "end_delta_frames": item.end_frame - human_end,
                "shot_iou": intersection / union,
                "note": str(raw.get("note", "")),
            }
        )
    median_iou = (
        median([float(item["shot_iou"]) for item in results]) if results else 0.0
    )
    gates = {
        "at_least_8_events": len(results) >= 8,
        "median_shot_iou_at_least_0_70": median_iou >= 0.70,
    }
    return {
        "schema_version": "m1a-manual-boundary-audit-v1",
        "review_method": annotations.get("review_method"),
        "reviewer": annotations.get("reviewer"),
        "metrics": {
            "reviewed_event_count": len(results),
            "start_boundary_match_count": sum(
                bool(item["start_boundary_match"]) for item in results
            ),
            "end_boundary_match_count": sum(
                bool(item["end_boundary_match"]) for item in results
            ),
            "median_shot_iou": median_iou,
        },
        "items": results,
        "gates": gates,
        "pass": all(gates.values()),
    }


def build_boundary_audit(
    shots: list[Shot],
    visual_events: list[AlignedVisualEvent],
    utterances: list[ContextualizedUtterance],
    *,
    frame_count: int,
) -> dict[str, object]:
    shot_starts = {shot.start_frame for shot in shots}
    shot_ends = {shot.end_frame for shot in shots}
    shot_ids = {shot.shot_id for shot in shots}
    gaps = sum(a.end_frame < b.start_frame for a, b in zip(shots, shots[1:]))
    overlaps = sum(a.end_frame > b.start_frame for a, b in zip(shots, shots[1:]))
    zero_frame = sum(shot.duration_frames <= 0 for shot in shots)
    off_grid = sum(
        item.start_frame not in shot_starts or item.end_frame not in shot_ends
        for item in visual_events
    )
    broken_visual_refs = sum(
        any(shot_id not in shot_ids for shot_id in item.shot_ids)
        for item in visual_events
    )
    broken_utterance_refs = sum(
        any(shot_id not in shot_ids for shot_id in item.shot_ids)
        for item in utterances
    )
    metrics: dict[str, int | float] = {
        "shot_count": len(shots),
        "shot_gap_count": gaps,
        "shot_overlap_count": overlaps,
        "zero_frame_shot_count": zero_frame,
        "covered_frames": sum(shot.duration_frames for shot in shots),
        "frame_count": frame_count,
        "visual_event_count": len(visual_events),
        "aligned_visual_event_count": len(visual_events) - broken_visual_refs,
        "utterance_count": len(utterances),
        "contextualized_utterance_count": len(utterances) - broken_utterance_refs,
        "off_grid_visual_boundary_count": off_grid,
        "broken_visual_reference_count": broken_visual_refs,
        "broken_utterance_reference_count": broken_utterance_refs,
        "median_abs_start_delta_frames": median(
            [abs(item.start_delta_frames) for item in visual_events]
        ),
        "median_abs_end_delta_frames": median(
            [abs(item.end_delta_frames) for item in visual_events]
        ),
    }
    gates = {
        "full_frame_coverage": metrics["covered_frames"] == frame_count,
        "no_shot_gaps": gaps == 0,
        "no_shot_overlaps": overlaps == 0,
        "no_zero_frame_shots": zero_frame == 0,
        "all_visual_events_aligned": len(visual_events) == 23 and broken_visual_refs == 0,
        "all_utterances_contextualized": len(utterances) == 7 and broken_utterance_refs == 0,
        "all_visual_boundaries_on_grid": off_grid == 0,
        "all_references_resolve": broken_visual_refs == 0 and broken_utterance_refs == 0,
    }
    return {
        "schema_version": "m1a-boundary-audit-v1",
        "metrics": metrics,
        "gates": gates,
        "pass": all(gates.values()),
    }
