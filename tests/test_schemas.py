from __future__ import annotations

import pytest
from pydantic import ValidationError

from shotseek.schemas import Utterance, VisualEvent, WordTimestamp


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
