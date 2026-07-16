from __future__ import annotations

import json
import re
from pathlib import Path

from shotseek.providers.stepfun.asr_sse import normalize_sse_events
from shotseek.providers.stepfun.vision import normalize_vision_response
from shotseek.timeline.normalize import normalize_timeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "stepfun"


def test_fixtures_contain_no_secrets_or_machine_paths() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FIXTURE_DIR.glob("*.json"))
    )
    forbidden = [
        r"sk-[A-Za-z0-9_-]{8,}",
        r"/home/",
        r"/Users/",
        r"X-Amz-(?:Credential|Signature)",
        r"https?://",
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    ]
    for pattern in forbidden:
        assert re.search(pattern, combined, re.IGNORECASE) is None, pattern


def test_same_fixture_produces_the_same_timeline() -> None:
    vision_raw = json.loads(
        (FIXTURE_DIR / "vision_response.sample.json").read_text(encoding="utf-8")
    )
    asr_raw = json.loads(
        (FIXTURE_DIR / "asr_sse_response.sample.json").read_text(encoding="utf-8")
    )

    def produce() -> list[dict[str, object]]:
        events = normalize_vision_response(vision_raw, model="step-3.7-flash")
        utterances = normalize_sse_events(asr_raw["events"])
        return [
            item.model_dump(mode="json")
            for item in normalize_timeline(75000, events, utterances)
        ]

    assert produce() == produce()


def test_fixture_provenance_distinguishes_live_responses_from_contract_samples() -> None:
    provenance = json.loads(
        (FIXTURE_DIR / "fixture_provenance.sample.json").read_text(encoding="utf-8")
    )
    assert provenance["vision_response.sample.json"]["live_derived"] is True
    assert provenance["asr_sse_response.sample.json"]["live_derived"] is True
    assert provenance["stepfun_file.contract.sample.json"]["live_derived"] is False
    assert provenance["asr_async_contract.sample.json"]["live_derived"] is False
