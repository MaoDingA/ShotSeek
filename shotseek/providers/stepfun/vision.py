"""StepFun video event extraction with a strict M0 output contract."""

from __future__ import annotations

import json
from typing import Any

import httpx

from shotseek.schemas import VisualEvent

from . import DEFAULT_BASE_URL, DEFAULT_VISION_MODEL
from .http import request_with_retry

VISION_PROMPT_VERSION = "m0-vision-v1"
VISION_SCHEMA_VERSION = "visual-event-v1"

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
- Do not infer real identities, hidden motives, backstory, relationships, or plot facts.
- Do not claim dialogue or speech content; audio evidence is handled by ASR separately.
- Use null or [] when uncertain. Do not invent missing facts.
- Every event must have end > start and confidence from 0 to 1.
- Keep events concise, non-overlapping where practical, and ordered by time.
"""


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


def normalize_vision_response(raw: dict[str, Any], *, model: str) -> list[VisualEvent]:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("vision response is missing choices[0].message.content") from exc
    payload = extract_json_object(str(content))
    chunk_id = str(payload.get("chunk_id") or "chunk_000")
    event_items = payload.get("events")
    if not isinstance(event_items, list):
        raise ValueError("vision response events must be an array")

    events: list[VisualEvent] = []
    for index, item in enumerate(event_items, start=1):
        if not isinstance(item, dict):
            raise ValueError("every visual event must be an object")
        normalized = dict(item)
        if "approx_start_ms" not in normalized and "local_start_ms" in normalized:
            normalized["approx_start_ms"] = normalized.pop("local_start_ms")
        if "approx_end_ms" not in normalized and "local_end_ms" in normalized:
            normalized["approx_end_ms"] = normalized.pop("local_end_ms")
        normalized.setdefault("event_id", f"visual_{index:04d}")
        normalized["chunk_id"] = chunk_id
        normalized["source"] = "stepfun_vision"
        normalized["model"] = model
        events.append(VisualEvent.model_validate(normalized))
    return events


def analyze_video(
    file_uri: str,
    *,
    api_key: str,
    model: str = DEFAULT_VISION_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: float = 300.0,
    client: httpx.Client | None = None,
    retry_attempts: int = 3,
    retry_base_delay_s: float = 0.5,
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
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(timeout_s))
    try:
        response = request_with_retry(
            lambda: http.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=_headers(api_key),
                json=request_body,
            ),
            max_attempts=retry_attempts,
            base_delay_s=retry_base_delay_s,
        )
        raw = response.json()
        return normalize_vision_response(raw, model=model), raw
    finally:
        if owns_client:
            http.close()
