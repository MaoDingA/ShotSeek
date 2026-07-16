"""Step Plan HTTP+SSE ASR adapter with explicit timestamp support."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from shotseek.schemas import Utterance, WordTimestamp

from . import DEFAULT_ASR_MODEL, DEFAULT_SSE_ASR_BASE_URL
from .asr import infer_audio_format, validate_public_audio_url
from .http import request_with_retry

SSE_ASR_SCHEMA_VERSION = "stepfun-sse-asr-v1"


def _headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise ValueError("StepFun API key is required")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("SSE ASR event must be a JSON object")
        if parsed.get("type") == "error":
            raise RuntimeError(str(parsed.get("message") or "SSE ASR failed"))
        events.append(parsed)
    if not events:
        raise ValueError("SSE ASR response did not contain data events")
    return events


def normalize_sse_events(
    events: list[dict[str, Any]],
    *,
    merge_gap_ms: int = 800,
) -> list[Utterance]:
    if merge_gap_ms < 0:
        raise ValueError("merge_gap_ms must be non-negative")

    segments: list[WordTimestamp] = []
    for event in events:
        if event.get("type") != "transcript.text.delta":
            continue
        text = str(event.get("delta", "")).strip()
        start = event.get("start_time")
        end = event.get("end_time")
        if not text or not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start:
            continue
        segments.append(WordTimestamp(text=text, start_ms=start, end_ms=end))

    segments.sort(key=lambda item: (item.start_ms, item.end_ms, item.text))
    if not segments:
        raise ValueError("SSE ASR response did not contain usable timestamped deltas")

    groups: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []
    for segment in segments:
        if not current or segment.start_ms - current[-1].end_ms <= merge_gap_ms:
            current.append(segment)
        else:
            groups.append(current)
            current = [segment]
    if current:
        groups.append(current)

    utterances: list[Utterance] = []
    for index, words in enumerate(groups, start=1):
        utterances.append(
            Utterance(
                utterance_id=f"utterance_{index:04d}",
                start_ms=words[0].start_ms,
                end_ms=max(word.end_ms for word in words),
                text=" ".join(word.text for word in words),
                speaker_id=None,
                words=words,
                source="stepfun_asr_sse",
            )
        )
    return utterances


def run_sse_asr(
    audio_url: str,
    *,
    api_key: str,
    model: str = DEFAULT_ASR_MODEL,
    base_url: str = DEFAULT_SSE_ASR_BASE_URL,
    language: str | None = None,
    timeout_s: float = 180.0,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
) -> tuple[list[Utterance], dict[str, Any]]:
    """Download a public audio file, then submit it to Step Plan SSE ASR."""
    validate_public_audio_url(audio_url)
    audio_format = infer_audio_format(audio_url)
    owns_client = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=True,
    )
    try:
        audio_response = request_with_retry(
            lambda: http.get(audio_url),
            max_attempts=retry_attempts,
            base_delay_s=retry_base_delay_s,
        )
        transcription: dict[str, Any] = {
            "model": model,
            "enable_itn": True,
            "enable_timestamp": True,
        }
        if language:
            transcription["language"] = language
        format_payload: dict[str, Any] = {"type": audio_format}
        request_body = {
            "audio": {
                "data": base64.b64encode(audio_response.content).decode("ascii"),
                "input": {
                    "transcription": transcription,
                    "format": format_payload,
                },
            }
        }
        response = request_with_retry(
            lambda: http.post(
                f"{base_url.rstrip('/')}/audio/asr/sse",
                headers=_headers(api_key),
                json=request_body,
            ),
            max_attempts=retry_attempts,
            base_delay_s=retry_base_delay_s,
        )
        events = parse_sse_events(response.text)
        utterances = normalize_sse_events(events)
        raw = {
            "transport": "sse",
            "schema_version": SSE_ASR_SCHEMA_VERSION,
            "request": {
                "model": model,
                "audio_format": audio_format,
                "enable_timestamp": True,
                "language": language,
                "audio_bytes": len(audio_response.content),
            },
            "events": events,
        }
        return utterances, raw
    finally:
        if owns_client:
            http.close()
