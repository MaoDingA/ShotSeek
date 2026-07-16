from __future__ import annotations

from shotseek.schemas import (
    BoundaryStatus,
    EvidenceKind,
    Utterance,
    VisualEvent,
)
from shotseek.timeline.normalize import apply_chunk_offset, normalize_timeline
from shotseek.timeline.validate import validate_evidence_timeline


def make_visual(
    event_id: str = "visual_0001", start_ms: int = 2000, end_ms: int = 5000
) -> VisualEvent:
    return VisualEvent(
        event_id=event_id,
        approx_start_ms=start_ms,
        approx_end_ms=end_ms,
        summary="A woman lifts a red folder.",
        characters=["woman"],
        actions=["lifts"],
        objects=["red folder"],
        location="room",
        confidence=0.9,
        model="fixture-model",
    )


def make_utterance(
    utterance_id: str = "utterance_0001", start_ms: int = 1000, end_ms: int = 1800
) -> Utterance:
    return Utterance(
        utterance_id=utterance_id,
        start_ms=start_ms,
        end_ms=end_ms,
        text="This was not an accident.",
        speaker_id="spk_1",
    )


def test_chunk_offset_is_applied() -> None:
    assert apply_chunk_offset(2000, 5000, 120000) == (122000, 125000)


def test_evidence_timeline_is_sorted_and_traceable() -> None:
    evidence = normalize_timeline(10000, [make_visual()], [make_utterance()])
    assert [item.kind for item in evidence] == [EvidenceKind.DIALOGUE, EvidenceKind.VISUAL]
    assert all(item.source_ref for item in evidence)
    validate_evidence_timeline(evidence, 10000)


def test_evidence_is_clamped_to_video_duration() -> None:
    evidence = normalize_timeline(3000, [make_visual(start_ms=2000, end_ms=5000)], [])
    assert len(evidence) == 1
    assert evidence[0].start_ms == 2000
    assert evidence[0].end_ms == 3000


def test_evidence_starting_after_video_is_dropped() -> None:
    evidence = normalize_timeline(3000, [make_visual(start_ms=4000, end_ms=5000)], [])
    assert evidence == []


def test_visual_and_dialogue_boundary_statuses_are_explicit() -> None:
    evidence = normalize_timeline(10000, [make_visual()], [make_utterance()])
    by_kind = {item.kind: item for item in evidence}
    assert by_kind[EvidenceKind.VISUAL].boundary_status == BoundaryStatus.APPROXIMATE
    assert by_kind[EvidenceKind.DIALOGUE].boundary_status == BoundaryStatus.ASR_TIMESTAMP


def test_visual_chunk_source_offset_reaches_original_timeline() -> None:
    evidence = normalize_timeline(
        130000,
        [make_visual(start_ms=2000, end_ms=5000)],
        [],
        visual_source_start_ms=120000,
    )
    assert evidence[0].start_ms == 122000
    assert evidence[0].end_ms == 125000


def test_each_visual_event_can_carry_its_own_chunk_offset() -> None:
    event = make_visual(start_ms=500, end_ms=1500).model_copy(
        update={"source_start_ms": 20_000, "chunk_id": "chunk_002"}
    )
    evidence = normalize_timeline(30_000, [event], [])
    assert evidence[0].start_ms == 20_500
    assert evidence[0].end_ms == 21_500


def test_event_offset_and_legacy_call_offset_are_additive() -> None:
    event = make_visual(start_ms=500, end_ms=1500).model_copy(
        update={"source_start_ms": 20_000}
    )
    evidence = normalize_timeline(
        40_000, [event], [], visual_source_start_ms=10_000
    )
    assert evidence[0].start_ms == 30_500
