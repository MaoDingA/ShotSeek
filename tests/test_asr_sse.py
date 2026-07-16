from __future__ import annotations

import base64
import json

import httpx
import pytest

from shotseek.providers.stepfun.asr_sse import (
    normalize_sse_events,
    parse_sse_events,
    run_sse_asr,
)


def test_parse_and_normalize_sse_timestamp_events() -> None:
    raw = "\n".join(
        [
            'data: {"type":"transcript.text.delta","delta":"Hello",'
            '"start_time":100,"end_time":400}',
            'data: {"type":"transcript.text.delta","delta":"world",'
            '"start_time":450,"end_time":800}',
            'data: {"type":"transcript.text.delta","delta":"ignored",'
            '"start_time":900,"end_time":900}',
            'data: {"type":"transcript.text.done","text":"Hello world"}',
        ]
    )
    events = parse_sse_events(raw)
    utterances = normalize_sse_events(events)
    assert len(utterances) == 1
    assert utterances[0].start_ms == 100
    assert utterances[0].end_ms == 800
    assert utterances[0].text == "Hello world"
    assert [word.text for word in utterances[0].words] == ["Hello", "world"]
    assert utterances[0].speaker_id is None
    assert utterances[0].source == "stepfun_asr_sse"


def test_sse_asr_sends_enable_timestamp_without_persisting_audio_data() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=b"fixture-audio")
        payload = json.loads(request.read())
        assert payload["audio"]["data"] == base64.b64encode(
            b"fixture-audio"
        ).decode("ascii")
        assert payload["audio"]["input"]["transcription"] == {
            "model": "stepaudio-2.5-asr",
            "enable_itn": True,
            "enable_timestamp": True,
            "language": "en",
        }
        return httpx.Response(
            200,
            text=(
                'data: {"type":"transcript.text.delta","delta":"Hello",'
                '"start_time":100,"end_time":500}\n\n'
                'data: {"type":"transcript.text.done","text":"Hello"}\n\n'
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        utterances, raw = run_sse_asr(
            "https://example.invalid/golden.mp3",
            api_key="fixture-key",
            language="en",
            client=client,
        )

    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].url.path == "/step_plan/v1/audio/asr/sse"
    assert utterances[0].start_ms == 100
    assert raw["request"]["enable_timestamp"] is True
    assert raw["request"]["audio_bytes"] == len(b"fixture-audio")
    assert "data" not in raw["request"]


def test_sse_asr_rejects_response_without_valid_timestamps() -> None:
    events = [
        {
            "type": "transcript.text.delta",
            "delta": "Hello",
            "start_time": 0,
            "end_time": 0,
        }
    ]
    with pytest.raises(ValueError, match="usable timestamped deltas"):
        normalize_sse_events(events)


def test_sse_error_event_is_not_treated_as_transcript() -> None:
    with pytest.raises(RuntimeError, match="bad audio"):
        parse_sse_events('data: {"type":"error","message":"bad audio"}')
