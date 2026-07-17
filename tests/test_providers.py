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


def test_vision_response_repairs_common_scalar_and_list_drift() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "events": [
                                {
                                    "approx_start_ms": 0,
                                    "approx_end_ms": 1000,
                                    "summary": ["A person moves", "past a display"],
                                    "characters": "person",
                                    "actions": None,
                                    "objects": "display",
                                    "location": ["office", "hallway"],
                                    "visible_text": "PLAYBACK",
                                    "confidence": 0.75,
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }

    event = normalize_vision_response(raw, model="fixture-model")[0]
    assert event.summary == "A person moves / past a display"
    assert event.characters == ["person"]
    assert event.actions == []
    assert event.objects == ["display"]
    assert event.location == "office / hallway"
    assert event.visible_text == ["PLAYBACK"]


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
                            {"text": "Hello", "start_time": 100, "end_time": 900},
                            {"text": "zero", "start_time": 900, "end_time": 900},
                            {"text": "missing", "start_time": None, "end_time": None},
                            {"text": "outside", "start_time": 50, "end_time": 150},
                        ],
                    }
                ],
            }
        ],
    }
    utterances = normalize_asr_response(raw)
    assert utterances[0].speaker_id == "spk_1"
    assert utterances[0].words[0].text == "Hello"
    assert len(utterances[0].words) == 1
