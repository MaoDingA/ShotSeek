"""Pure functions that map provider evidence onto the original video timeline."""

from __future__ import annotations

from collections.abc import Iterable

from shotseek.schemas import (
    BoundaryStatus,
    EvidenceKind,
    EvidenceSpan,
    Utterance,
    VisualEvent,
)


def apply_chunk_offset(
    local_start_ms: int,
    local_end_ms: int,
    source_start_ms: int,
) -> tuple[int, int]:
    if local_start_ms < 0 or source_start_ms < 0:
        raise ValueError("timestamps and source offset must be non-negative")
    if local_end_ms <= local_start_ms:
        raise ValueError("local_end_ms must be greater than local_start_ms")
    return source_start_ms + local_start_ms, source_start_ms + local_end_ms


def _distinct(values: Iterable[str | None]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _bounded_range(start_ms: int, end_ms: int, duration_ms: int) -> tuple[int, int] | None:
    if start_ms < 0:
        raise ValueError("start_ms must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("end_ms must be greater than start_ms")
    if duration_ms <= 0:
        raise ValueError("video_duration_ms must be positive")
    if start_ms >= duration_ms:
        return None
    bounded_end = min(end_ms, duration_ms)
    if bounded_end <= start_ms:
        return None
    return start_ms, bounded_end


def normalize_timeline(
    video_duration_ms: int,
    visual_events: Iterable[VisualEvent],
    utterances: Iterable[Utterance],
    *,
    visual_source_start_ms: int = 0,
) -> list[EvidenceSpan]:
    evidence: list[EvidenceSpan] = []

    for event in visual_events:
        global_start, global_end = apply_chunk_offset(
            event.approx_start_ms,
            event.approx_end_ms,
            visual_source_start_ms + event.source_start_ms,
        )
        bounded = _bounded_range(global_start, global_end, video_duration_ms)
        if bounded is None:
            continue
        start_ms, end_ms = bounded
        entities = _distinct(
            [*event.characters, *event.objects, event.location, *event.visible_text]
        )
        evidence.append(
            EvidenceSpan(
                evidence_id=f"visual:{event.event_id}",
                kind=EvidenceKind.VISUAL,
                start_ms=start_ms,
                end_ms=end_ms,
                text=event.summary,
                entities=entities,
                actions=_distinct(event.actions),
                confidence=event.confidence,
                source_ref=event.event_id,
                boundary_status=BoundaryStatus.APPROXIMATE,
            )
        )

    for utterance in utterances:
        bounded = _bounded_range(utterance.start_ms, utterance.end_ms, video_duration_ms)
        if bounded is None:
            continue
        start_ms, end_ms = bounded
        evidence.append(
            EvidenceSpan(
                evidence_id=f"dialogue:{utterance.utterance_id}",
                kind=EvidenceKind.DIALOGUE,
                start_ms=start_ms,
                end_ms=end_ms,
                text=utterance.text,
                entities=_distinct([utterance.speaker_id]),
                actions=[],
                confidence=1.0,
                source_ref=utterance.utterance_id,
                boundary_status=BoundaryStatus.ASR_TIMESTAMP,
            )
        )

    return sorted(evidence, key=lambda item: (item.start_ms, item.end_ms, item.evidence_id))
