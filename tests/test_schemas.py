from __future__ import annotations

import pytest
from pydantic import ValidationError

from shotseek.schemas import Utterance, VideoChunkInput, VisualEvent, WordTimestamp


def visual_event(**overrides: object) -> VisualEvent:
    payload: dict[str, object] = {
        "event_id": "visual_0001",
        "approx_start_ms": 100,
        "approx_end_ms": 900,
        "summary": "A person lifts a folder.",
        "confidence": 0.8,
        "model": "fixture-model",
    }
    payload.update(overrides)
    return VisualEvent.model_validate(payload)


def test_negative_visual_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        visual_event(approx_start_ms=-1)


def test_reversed_visual_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        visual_event(approx_start_ms=900, approx_end_ms=100)


def test_visual_confidence_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValidationError):
        visual_event(confidence=1.1)


def test_utterance_preserves_word_timestamps() -> None:
    utterance = Utterance(
        utterance_id="utterance_0001",
        start_ms=1000,
        end_ms=2000,
        text="Hello world",
        words=[WordTimestamp(text="Hello", start_ms=1000, end_ms=1500)],
    )
    assert utterance.words[0].start_ms == 1000
    assert utterance.words[0].end_ms == 1500


def test_word_timestamp_outside_utterance_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Utterance(
            utterance_id="utterance_0001",
            start_ms=1000,
            end_ms=2000,
            text="Hello",
            words=[WordTimestamp(text="Hello", start_ms=900, end_ms=1500)],
        )


def test_video_chunk_rejects_more_than_ten_seconds() -> None:
    with pytest.raises(ValidationError, match="at most 10000 ms"):
        VideoChunkInput(
            chunk_id="chunk_000",
            source_start_ms=0,
            source_end_ms=10_001,
            url="https://example.invalid/chunk-000.mp4",
        )


def test_video_chunk_rejects_non_http_url() -> None:
    with pytest.raises(ValidationError, match="must use http"):
        VideoChunkInput(
            chunk_id="chunk_000",
            source_start_ms=0,
            source_end_ms=10_000,
            url="stepfile://file_fixture",
        )
