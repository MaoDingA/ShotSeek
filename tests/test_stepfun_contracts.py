from __future__ import annotations

import json
from pathlib import Path

import httpx

from shotseek.providers.stepfun.asr import submit_asr, wait_for_asr
from shotseek.providers.stepfun.files import upload_video
from shotseek.providers.stepfun.vision import analyze_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_files_api_uses_storage_multipart_contract() -> None:
    video_path = PROJECT_ROOT / ".tmp" / "contract-video.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"m0-fixture-video")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        body = request.read()
        seen["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "file_contract_fixture",
                "object": "file",
                "bytes": len(b"m0-fixture-video"),
                "filename": "contract-video.mp4",
                "purpose": "storage",
                "status": "processed",
            },
        )

    try:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            uploaded, raw = upload_video(
                video_path,
                api_key="fixture-key",
                base_url="https://api.example.invalid/v1",
                client=client,
            )
        assert seen["method"] == "POST"
        assert seen["path"] == "/v1/files"
        assert b"storage" in seen["body"]
        assert b"contract-video.mp4" in seen["body"]
        assert uploaded.file_uri == "stepfile://file_contract_fixture"
        assert raw["final"]["status"] == "processed"
    finally:
        video_path.unlink(missing_ok=True)


def test_vision_api_uses_video_url_and_json_mode() -> None:
    seen: dict[str, object] = {}
    content = {
        "chunk_id": "chunk_000",
        "events": [
            {
                "event_id": "visual_0001",
                "approx_start_ms": 100,
                "approx_end_ms": 900,
                "summary": "A person turns toward a monitor.",
                "characters": ["person"],
                "actions": ["turns"],
                "objects": ["monitor"],
                "location": "room",
                "visible_text": [],
                "confidence": 0.8,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.read())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(content)}}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        events, _ = analyze_video(
            "stepfile://file_contract_fixture",
            api_key="fixture-key",
            model="step-3.7-flash",
            base_url="https://api.example.invalid/v1",
            client=client,
        )
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["stream"] is False
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 4096
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["content"][0] == {
        "type": "video_url",
        "video_url": {"url": "stepfile://file_contract_fixture"},
    }
    assert events[0].approx_start_ms == 100


def test_vision_retries_empty_structured_output_with_larger_budget() -> None:
    payloads: list[dict[str, object]] = []
    valid_content = {
        "chunk_id": "chunk_006",
        "events": [
            {
                "approx_start_ms": 100,
                "approx_end_ms": 900,
                "summary": "A person turns toward a monitor.",
                "characters": ["person"],
                "actions": ["turns"],
                "objects": ["monitor"],
                "location": "room",
                "visible_text": [],
                "confidence": 0.8,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.read()))
        content = "" if len(payloads) == 1 else json.dumps(valid_content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        events, raw = analyze_video(
            "https://example.invalid/chunk-006.mp4",
            api_key="fixture-key",
            model="step-3.7-flash",
            base_url="https://api.example.invalid/step_plan/v1",
            client=client,
            chunk_id_override="chunk_006",
            source_start_ms=60_000,
        )

    assert [payload["max_tokens"] for payload in payloads] == [4096, 8192]
    assert raw["mode"] == "normalization_retry"
    assert len(raw["attempts"]) == 2
    assert events[0].event_id == "chunk_006:visual_0001"
    assert events[0].chunk_id == "chunk_006"
    assert events[0].source_start_ms == 60_000


def test_asr_submit_and_poll_contract() -> None:
    requests: list[dict[str, object]] = []
    query_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal query_count
        payload = json.loads(request.read())
        requests.append({"path": request.url.path, "payload": payload})
        if request.url.path.endswith("/submit"):
            return httpx.Response(200, json={"task_id": "task_contract_fixture"})
        query_count += 1
        if query_count == 1:
            return httpx.Response(200, json={"status": "PENDING"})
        if query_count == 2:
            return httpx.Response(200, json={"status": "RUNNING"})
        return httpx.Response(
            200,
            json={
                "duration": 1.0,
                "result": [
                    {
                        "text": "Hello",
                        "utterances": [
                            {"text": "Hello", "start_time": 0, "end_time": 900}
                        ],
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        task_id, _ = submit_asr(
            "https://example.invalid/golden.mp3",
            api_key="fixture-key",
            model="stepaudio-2.5-asr",
            base_url="https://api.example.invalid/v1",
            client=client,
        )
        final = wait_for_asr(
            task_id,
            api_key="fixture-key",
            base_url="https://api.example.invalid/v1",
            poll_interval_s=0,
            timeout_s=1,
            client=client,
        )

    submit_payload = requests[0]["payload"]
    assert submit_payload["audio"]["format"] == "mp3"
    assert submit_payload["request"]["show_utterances"] is True
    assert submit_payload["request"]["enable_speaker_info"] is True
    assert requests[1]["path"].endswith("/audio/asr/file/query")
    assert final["result"][0]["utterances"][0]["start_time"] == 0
