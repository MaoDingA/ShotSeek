"""Deterministic shot-first alignment for M0 evidence."""

from __future__ import annotations

from fractions import Fraction

from shotseek.media.schema import AlignedVisualEvent, ContextualizedUtterance, Shot
from shotseek.media.timebase import ms_to_ceil_frame, ms_to_floor_frame
from shotseek.schemas import Utterance, VisualEvent


def _overlapping_shots(start_frame: int, end_frame: int, shots: list[Shot]) -> list[Shot]:
    return [
        shot
        for shot in shots
        if shot.start_frame < end_frame and shot.end_frame > start_frame
    ]


def align_visual_events(
    events: list[VisualEvent], shots: list[Shot], *, fps: Fraction, frame_count: int
) -> list[AlignedVisualEvent]:
    aligned: list[AlignedVisualEvent] = []
    for event in events:
        raw_start_ms = event.source_start_ms + event.approx_start_ms
        raw_end_ms = event.source_start_ms + event.approx_end_ms
        raw_start_frame = min(ms_to_floor_frame(raw_start_ms, fps), frame_count - 1)
        raw_end_frame = min(
            max(ms_to_ceil_frame(raw_end_ms, fps), raw_start_frame + 1), frame_count
        )
        matched = _overlapping_shots(raw_start_frame, raw_end_frame, shots)
        if not matched:
            raise ValueError(f"visual event does not overlap a shot: {event.event_id}")
        start_frame = matched[0].start_frame
        end_frame = matched[-1].end_frame
        aligned.append(
            AlignedVisualEvent(
                event_id=event.event_id,
                chunk_id=event.chunk_id,
                source_start_ms=event.source_start_ms,
                raw_local_start_ms=event.approx_start_ms,
                raw_local_end_ms=event.approx_end_ms,
                raw_global_start_ms=raw_start_ms,
                raw_global_end_ms=raw_end_ms,
                final_start_ms=matched[0].start_ms,
                final_end_ms=matched[-1].end_ms,
                start_frame=start_frame,
                end_frame=end_frame,
                shot_ids=[shot.shot_id for shot in matched],
                summary=event.summary,
                characters=event.characters,
                actions=event.actions,
                objects=event.objects,
                location=event.location,
                visible_text=event.visible_text,
                confidence=event.confidence,
                source=event.source,
                model=event.model,
                start_delta_frames=start_frame - raw_start_frame,
                end_delta_frames=end_frame - raw_end_frame,
            )
        )
    return sorted(aligned, key=lambda item: (item.start_frame, item.end_frame, item.event_id))


def contextualize_utterances(
    utterances: list[Utterance], shots: list[Shot], *, fps: Fraction, frame_count: int
) -> list[ContextualizedUtterance]:
    result: list[ContextualizedUtterance] = []
    for utterance in utterances:
        start_frame = min(ms_to_floor_frame(utterance.start_ms, fps), frame_count - 1)
        end_frame = min(
            max(ms_to_ceil_frame(utterance.end_ms, fps), start_frame + 1), frame_count
        )
        matched = _overlapping_shots(start_frame, end_frame, shots)
        if not matched:
            raise ValueError(f"utterance does not overlap a shot: {utterance.utterance_id}")
        result.append(
            ContextualizedUtterance(
                utterance_id=utterance.utterance_id,
                start_ms=utterance.start_ms,
                end_ms=utterance.end_ms,
                text=utterance.text,
                speaker_id=utterance.speaker_id,
                words=[word.model_dump(mode="json") for word in utterance.words],
                source=utterance.source,
                shot_ids=[shot.shot_id for shot in matched],
            )
        )
    return result
