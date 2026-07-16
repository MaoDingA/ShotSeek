from __future__ import annotations

from shotseek.fixtures import (
    sanitize_asr_response,
    sanitize_stepfun_file,
    sanitize_vision_response,
)


def test_file_fixture_sanitizer_redacts_provider_ids_and_urls() -> None:
    raw = {
        "upload": {"id": "file-live", "status": "processed", "url": "https://signed"},
        "final": {"id": "file-live", "status": "processed"},
    }
    sanitized = sanitize_stepfun_file(raw)
    assert sanitized["upload"]["id"] == "file_fixture_redacted"
    assert sanitized["final"]["id"] == "file_fixture_redacted"
    assert sanitized["upload"]["url"] == "<redacted_url>"


def test_vision_fixture_sanitizer_redacts_response_identity() -> None:
    sanitized = sanitize_vision_response(
        {"id": "chatcmpl-live", "created": 1234, "choices": []}
    )
    assert sanitized == {
        "id": "chatcmpl_fixture_redacted",
        "created": 0,
        "choices": [],
    }


def test_asr_fixture_sanitizer_keeps_speaker_and_audio_timestamps() -> None:
    raw = {
        "submit": {"task_id": "task-live"},
        "result": {
            "result": [
                {
                    "utterances": [
                        {
                            "start_time": 100,
                            "end_time": 500,
                            "text": "hello",
                            "speaker": {"id": "spk_1"},
                        }
                    ]
                }
            ],
            "meta": {"session_id": "session-live", "timestamp": 1234},
        },
    }
    sanitized = sanitize_asr_response(raw)
    utterance = sanitized["result"]["result"][0]["utterances"][0]
    assert sanitized["submit"]["task_id"] == "task_id_fixture_redacted"
    assert sanitized["result"]["meta"] == {
        "session_id": "session_id_fixture_redacted",
        "timestamp": 0,
    }
    assert utterance["speaker"]["id"] == "spk_1"
    assert (utterance["start_time"], utterance["end_time"]) == (100, 500)


def test_sanitizer_removes_exact_api_key_without_printing_it() -> None:
    secret = "A" * 64
    sanitized = sanitize_asr_response(
        {"debug": f"Bearer {secret}"}, api_key=secret
    )
    assert secret not in sanitized["debug"]
