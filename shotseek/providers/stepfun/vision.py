"""StepFun video event extraction with a strict M0 output contract."""

from __future__ import annotations

import json
from typing import Any

import httpx

from shotseek.schemas import VisualEvent

from . import DEFAULT_CHAT_BASE_URL, DEFAULT_VISION_MODEL
from .http import request_with_retry

VISION_PROMPT_VERSION = "m0-vision-v2"
VISION_SCHEMA_VERSION = "visual-event-v2"

VISION_PROMPT = """You are extracting directly observable visual evidence from one video clip.
Return one JSON object only, with this exact top-level shape:
{
  "chunk_id": "chunk_000",
  "events": [
    {
      "event_id": "visual_0001",
      "approx_start_ms": 0,
      "approx_end_ms": 1000,
      "summary": "A directly observable event",
      "characters": [],
      "actions": [],
      "objects": [],
      "location": null,
      "visible_text": [],
      "confidence": 0.0
    }
  ]
}
Rules:
- Use milliseconds relative to the beginning of this clip.
- Times are approximate visual observations, not final shot boundaries.
- Report only actions, objects, locations, people, and text directly visible in frames.
- When visually clear, include a broad apparent age group in each character
  description; omit it when uncertain and never claim an exact age.
- Do not infer real identities, hidden motives, backstory, relationships, or plot facts.
- Do not claim dialogue or speech content; audio evidence is handled by ASR separately.
- Use null or [] when uncertain. Do not invent missing facts.
- Every event must have end > start and confidence from 0 to 1.
- Keep events concise, non-overlapping where practical, and ordered by time.
"""


class VisionResponseError(ValueError):
    """Raised after preserving all responses from failed normalization attempts."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def _headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        raise ValueError("StepFun API key is required")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    first = candidate.find("{")
    last = candidate.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("vision response did not contain a JSON object")
    parsed = json.loads(candidate[first : last + 1])
    if not isinstance(parsed, dict):
        raise ValueError("vision response JSON must be an object")
    return parsed


def _string_list(value: Any) -> list[str]:
    """Coerce common model schema drift into a clean string list."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    """Coerce a scalar or list into one display-safe optional string."""
    values = _string_list(value)
    return " / ".join(values) if values else None


def _normalize_event_fields(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["summary"] = (
        _optional_string(normalized.get("summary")) or "Observed event"
    )
    for field in ("characters", "actions", "objects", "visible_text"):
        normalized[field] = _string_list(normalized.get(field))
    normalized["location"] = _optional_string(normalized.get("location"))
    return normalized


def normalize_vision_response(
    raw: dict[str, Any],
    *,
    model: str,
    chunk_id_override: str | None = None,
    source_start_ms: int = 0,
) -> list[VisualEvent]:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("vision response is missing choices[0].message.content") from exc
    payload = extract_json_object(str(content))
    chunk_id = chunk_id_override or str(payload.get("chunk_id") or "chunk_000")
    event_items = payload.get("events")
    if not isinstance(event_items, list):
        raise ValueError("vision response events must be an array")

    events: list[VisualEvent] = []
    for index, item in enumerate(event_items, start=1):
        if not isinstance(item, dict):
            raise ValueError("every visual event must be an object")
        normalized = _normalize_event_fields(item)
        if "approx_start_ms" not in normalized and "local_start_ms" in normalized:
            normalized["approx_start_ms"] = normalized.pop("local_start_ms")
        if "approx_end_ms" not in normalized and "local_end_ms" in normalized:
            normalized["approx_end_ms"] = normalized.pop("local_end_ms")
        event_id = str(normalized.get("event_id") or f"visual_{index:04d}")
        normalized["event_id"] = (
            f"{chunk_id}:{event_id}" if chunk_id_override else event_id
        )
        normalized["chunk_id"] = chunk_id
        normalized["source_start_ms"] = source_start_ms
        normalized["source"] = "stepfun_vision"
        normalized["model"] = model
        events.append(VisualEvent.model_validate(normalized))
    return events


def normalize_vision_bundle(
    raw: dict[str, Any],
    *,
    model: str,
) -> list[VisualEvent]:
    """Normalize either one response or the recorded multi-chunk response envelope."""
    if raw.get("mode") != "direct_url_chunks":
        return normalize_vision_response(raw, model=model)
    chunks = raw.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("multi-chunk vision response must include a non-empty chunks array")
    events: list[VisualEvent] = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("response"), dict):
            raise ValueError("every vision chunk must include a response object")
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("every vision chunk must include chunk_id")
        try:
            source_start_ms = int(chunk["source_start_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every vision chunk must include an integer source_start_ms"
            ) from exc
        events.extend(
            normalize_vision_response(
                chunk["response"],
                model=model,
                chunk_id_override=chunk_id,
                source_start_ms=source_start_ms,
            )
        )
    return events


def analyze_video(
    file_uri: str,
    *,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_CHAT_BASE_URL,
    timeout_s: float = 300.0,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
    reasoning_effort: str = "low",
    max_tokens: int = 4096,
    chunk_id_override: str | None = None,
    source_start_ms: int = 0,
) -> tuple[list[VisualEvent], dict[str, Any]]:
    """Call Chat Completions with one StepFun file-backed MP4."""
    if not file_uri.startswith(("stepfile://", "https://", "http://")):
        raise ValueError("video URI must be stepfile://, https://, or http://")
    request_body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": file_uri}},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        "stream": False,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(timeout_s))
    try:
        def send(body: dict[str, Any]) -> dict[str, Any]:
            response = request_with_retry(
                lambda: http.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=_headers(api_key),
                    json=body,
                ),
                max_attempts=retry_attempts,
                base_delay_s=retry_base_delay_s,
            )
            return response.json()

        raw = send(request_body)
        attempts = [{"max_tokens": max_tokens, "response": raw}]
        initial_error: str | None = None
        try:
            events = normalize_vision_response(
                raw,
                model=model,
                chunk_id_override=chunk_id_override,
                source_start_ms=source_start_ms,
            )
            return events, raw
        except ValueError as first_error:
            initial_error = str(first_error)
            if max_tokens >= 8192:
                raise VisionResponseError(initial_error, attempts) from first_error

        retry_tokens = 8192
        retry_body = {**request_body, "max_tokens": retry_tokens}
        retry_raw = send(retry_body)
        attempts.append({"max_tokens": retry_tokens, "response": retry_raw})
        try:
            events = normalize_vision_response(
                retry_raw,
                model=model,
                chunk_id_override=chunk_id_override,
                source_start_ms=source_start_ms,
            )
        except ValueError as retry_error:
            raise VisionResponseError(str(retry_error), attempts) from retry_error
        return (
            events,
            {
                "mode": "normalization_retry",
                "initial_error": initial_error,
                "attempts": attempts,
                "final_response": retry_raw,
            },
        )
    finally:
        if owns_client:
            http.close()
