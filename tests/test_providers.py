from __future__ import annotations

import json

from shotseek.providers.stepfun.asr import normalize_asr_response
from shotseek.providers.stepfun.vision import normalize_vision_response


def test_vision_response_accepts_fenced_json_and_local_time_aliases() -> None:
    content = {
        "chunk_id": "chunk_007",
        "events": [
            {
                "local_start_ms": 2000,
                "local_end_ms": 5000,
                "summary": "A person opens a door.",
                "characters": ["person"],
                "actions": ["opens"],
                "objects": ["door"],
                "location": "hallway",
                "visible_text": [],
                "confidence": 0.8,
            }
        ],
    }
    raw = {
        "choices": [
            {"message": {"content": f"```json\n{json.dumps(content)}\n```"}}
        ]
    }
    events = normalize_vision_response(raw, model="fixture-model")
    assert events[0].approx_start_ms == 2000
    assert events[0].approx_end_ms == 5000
    assert events[0].chunk_id == "chunk_007"
    assert events[0].event_id == "visual_0001"


def test_asr_response_preserves_speaker_and_words() -> None:
    raw = {
        "duration": 2.0,
        "result": [
            {
                "text": "Hello",
                "utterances": [
                    {
                        "text": "Hello",
                        "start_time": 100,
                        "end_time": 1000,
                        "speaker": {"id": "spk_1"},
                        "words": [
                            {"text": "Hello", "start_time": 100, "end_time": 900}
                        ],
                    }
                ],
            }
        ],
    }
    utterances = normalize_asr_response(raw)
    assert utterances[0].speaker_id == "spk_1"
    assert utterances[0].words[0].text == "Hello"
