"""StepFun-backed QuerySpec v2 planner."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import httpx

from shotseek.planning.schema import PlannerResult, PlannerTrace, QuerySpecV2
from shotseek.providers.stepfun import DEFAULT_CHAT_BASE_URL, DEFAULT_VISION_MODEL
from shotseek.providers.stepfun.http import request_with_retry
from shotseek.providers.stepfun.vision import extract_json_object

PLANNER_PROMPT_VERSION = "m2-query-planner-v1"
PLANNER_SCHEMA_VERSION = "query-v2"
DEFAULT_PLANNER_MODEL = DEFAULT_VISION_MODEL

PLANNER_SYSTEM_PROMPT = """You are the query planning component of ShotSeek.
Convert the user's scene-search request into one strict JSON object.
You parse constraints only. Never select a scene, timestamp, frame, or shot.
Do not invent character names or identities.

Required shape:
{
  "schema_version": "query-v2",
  "intent": "find_scene",
  "raw_query": "copy the exact user query",
  "quoted_text": null,
  "entities": [{"text": "woman", "role": "subject"}],
  "actions": [],
  "objects": [],
  "locations": [],
  "keywords": [],
  "temporal_constraints": [
    {
      "relation": "after",
      "anchor": {
        "quoted_text": null,
        "entities": [],
        "actions": ["appear"],
        "objects": ["robot"],
        "locations": [],
        "keywords": []
      },
      "second_anchor": null
    }
  ],
  "ordinal": {"value": 1, "scope": "after_temporal_filter"},
  "negative_constraints": [],
  "evidence_preference": ["visual", "dialogue"],
  "require_direct_evidence": true,
  "top_k": 3
}

Rules:
- relation is before, after, during, or between.
- between requires second_anchor; other relations require it to be null.
- ordinal value is a positive integer or "last".
- negative field is entity, action, object, location, dialogue, or keyword.
- preserve quoted dialogue exactly without quotation marks.
- retain the exact input in raw_query.
- use empty arrays and null for absent constraints.
- output JSON only.
"""


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _entities(value: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    items = value if isinstance(value, list) else ([] if value is None else [value])
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            role = item.get("role")
            role = role if role in {"subject", "object", "other"} else "other"
        else:
            text = str(item).strip()
            role = "other"
        if text:
            result.append({"text": text, "role": role})
    return result


def _anchor(value: Any) -> dict[str, Any]:
    item = dict(value) if isinstance(value, dict) else {"keywords": _strings(value)}
    return {
        "quoted_text": item.get("quoted_text") or None,
        "entities": _entities(item.get("entities")),
        "actions": _strings(item.get("actions")),
        "objects": _strings(item.get("objects")),
        "locations": _strings(item.get("locations")),
        "keywords": _strings(item.get("keywords")),
    }


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["entities"] = _entities(result.get("entities"))
    for field in ("actions", "objects", "locations", "keywords"):
        result[field] = _strings(result.get(field))
    temporal = []
    for raw in result.get("temporal_constraints") or []:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["anchor"] = _anchor(item.get("anchor"))
        item["second_anchor"] = (
            _anchor(item.get("second_anchor"))
            if item.get("second_anchor") is not None
            else None
        )
        temporal.append(item)
    result["temporal_constraints"] = temporal
    ordinal = result.get("ordinal")
    if isinstance(ordinal, (int, str)):
        result["ordinal"] = {"value": ordinal, "scope": "matching_event"}
    negatives = []
    for raw in result.get("negative_constraints") or []:
        negatives.append(
            raw if isinstance(raw, dict) else {"field": "keyword", "text": str(raw)}
        )
    result["negative_constraints"] = negatives
    preferences = [
        item for item in _strings(result.get("evidence_preference"))
        if item in {"visual", "dialogue", "script"}
    ]
    result["evidence_preference"] = preferences or ["visual", "dialogue"]
    return result


def normalize_planner_response(raw: dict[str, Any], *, query: str, top_k: int) -> QuerySpecV2:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("planner response is missing choices[0].message.content") from exc
    payload = (
        dict(content)
        if isinstance(content, dict)
        else extract_json_object(str(content))
    )
    payload = _normalize_payload(payload)
    payload["schema_version"] = "query-v2"
    payload["intent"] = "find_scene"
    payload["raw_query"] = query
    payload["top_k"] = top_k
    return QuerySpecV2.model_validate(payload)


class StepFunPlanner:
    def __init__(
        self,
        *,
        model: str = DEFAULT_PLANNER_MODEL,
        base_url: str = DEFAULT_CHAT_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client

    def plan_fixture(
        self, query: str, raw: dict[str, Any], *, top_k: int = 3
    ) -> PlannerResult:
        started = perf_counter()
        spec = normalize_planner_response(raw, query=query, top_k=top_k)
        return PlannerResult(
            query_spec=spec,
            trace=PlannerTrace(
                trace_id="pending",
                status="CACHED",
                planner="stepfun",
                route_reason="offline sanitized StepFun fixture",
                cache_hit=True,
                latency_ms=(perf_counter() - started) * 1000,
                model=self.model,
                prompt_version=PLANNER_PROMPT_VERSION,
            ),
            raw_response=raw,
        )

    def plan_live(
        self,
        query: str,
        *,
        api_key: str,
        top_k: int = 3,
        retry_attempts: int = 3,
    ) -> PlannerResult:
        if not api_key.strip():
            raise ValueError("StepFun API key is required")
        started = perf_counter()
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "stream": False,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 4096,
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(120.0))
        try:
            response = request_with_retry(
                lambda: client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                ),
                max_attempts=retry_attempts,
            )
            raw = response.json()
            spec = normalize_planner_response(raw, query=query, top_k=top_k)
            return PlannerResult(
                query_spec=spec,
                trace=PlannerTrace(
                    trace_id="pending",
                    status="LIVE",
                    planner="stepfun",
                    route_reason="complex query requires structured planning",
                    cache_hit=False,
                    latency_ms=(perf_counter() - started) * 1000,
                    model=self.model,
                    prompt_version=PLANNER_PROMPT_VERSION,
                ),
                raw_response=raw,
            )
        finally:
            if owns_client:
                client.close()
