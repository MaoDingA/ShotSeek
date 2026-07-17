"""Rules-only M1B scene construction and reference audit."""

from __future__ import annotations

from shotseek.media.schema import AlignedVisualEvent, ContextualizedUtterance, Shot
from shotseek.scenes.schema import EvidenceRef, Scene


def interval_distance_ms(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> int:
    if first_start < second_end and second_start < first_end:
        return 0
    if first_end <= second_start:
        return second_start - first_end
    return first_start - second_end


def build_scenes(
    visual_events: list[AlignedVisualEvent],
    utterances: list[ContextualizedUtterance],
    *,
    context_window_ms: int = 1500,
) -> list[Scene]:
    if context_window_ms < 0:
        raise ValueError("context_window_ms must be non-negative")
    scenes: list[Scene] = []
    for index, event in enumerate(
        sorted(visual_events, key=lambda item: (item.start_frame, item.end_frame, item.event_id)),
        start=1,
    ):
        dialogue = [
            item
            for item in utterances
            if interval_distance_ms(
                event.final_start_ms,
                event.final_end_ms,
                item.start_ms,
                item.end_ms,
            )
            <= context_window_ms
        ]
        dialogue.sort(key=lambda item: (item.start_ms, item.end_ms, item.utterance_id))
        utterance_ids = [item.utterance_id for item in dialogue]
        scenes.append(
            Scene(
                scene_id=f"scene_{index:04d}",
                start_ms=event.final_start_ms,
                end_ms=event.final_end_ms,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                shot_ids=event.shot_ids,
                summary=event.summary,
                characters=event.characters,
                actions=event.actions,
                objects=event.objects,
                location=event.location,
                visible_text=event.visible_text,
                visual_event_id=event.event_id,
                utterance_ids=utterance_ids,
                evidence_refs=[
                    EvidenceRef(kind="visual", evidence_id=event.event_id),
                    *[
                        EvidenceRef(kind="dialogue", evidence_id=utterance_id)
                        for utterance_id in utterance_ids
                    ],
                ],
                confidence=event.confidence,
            )
        )
    return scenes


def validate_scene_references(
    scenes: list[Scene],
    visual_events: list[AlignedVisualEvent],
    utterances: list[ContextualizedUtterance],
    shots: list[Shot],
) -> dict[str, int | bool]:
    visual_by_id = {item.event_id: item for item in visual_events}
    utterance_ids = {item.utterance_id for item in utterances}
    shot_by_id = {item.shot_id: item for item in shots}
    dangling_visual = 0
    dangling_dialogue = 0
    dangling_shots = 0
    non_contiguous_shots = 0
    boundary_mismatches = 0
    seen_scene_ids: set[str] = set()
    duplicate_scene_ids = 0
    for scene in scenes:
        if scene.scene_id in seen_scene_ids:
            duplicate_scene_ids += 1
        seen_scene_ids.add(scene.scene_id)
        if scene.visual_event_id not in visual_by_id:
            dangling_visual += 1
        dangling_dialogue += sum(
            utterance_id not in utterance_ids for utterance_id in scene.utterance_ids
        )
        resolved_shots = [
            shot_by_id[shot_id] for shot_id in scene.shot_ids if shot_id in shot_by_id
        ]
        dangling_shots += len(scene.shot_ids) - len(resolved_shots)
        if len(resolved_shots) == len(scene.shot_ids):
            if any(
                previous.end_frame != current.start_frame
                for previous, current in zip(resolved_shots, resolved_shots[1:])
            ):
                non_contiguous_shots += 1
            if (
                resolved_shots[0].start_frame != scene.start_frame
                or resolved_shots[-1].end_frame != scene.end_frame
                or resolved_shots[0].start_ms != scene.start_ms
                or resolved_shots[-1].end_ms != scene.end_ms
            ):
                boundary_mismatches += 1
    referenced_visual = {scene.visual_event_id for scene in scenes}
    unrepresented_visual = len(set(visual_by_id) - referenced_visual)
    metrics: dict[str, int | bool] = {
        "scene_count": len(scenes),
        "visual_event_count": len(visual_events),
        "dangling_visual_reference_count": dangling_visual,
        "dangling_dialogue_reference_count": dangling_dialogue,
        "dangling_shot_reference_count": dangling_shots,
        "non_contiguous_scene_count": non_contiguous_shots,
        "boundary_mismatch_count": boundary_mismatches,
        "duplicate_scene_id_count": duplicate_scene_ids,
        "unrepresented_visual_event_count": unrepresented_visual,
    }
    error_keys = {
        "dangling_visual_reference_count",
        "dangling_dialogue_reference_count",
        "dangling_shot_reference_count",
        "non_contiguous_scene_count",
        "boundary_mismatch_count",
        "duplicate_scene_id_count",
        "unrepresented_visual_event_count",
    }
    metrics["pass"] = (
        len(scenes) == len(visual_events)
        and all(metrics[key] == 0 for key in error_keys)
    )
    return metrics
