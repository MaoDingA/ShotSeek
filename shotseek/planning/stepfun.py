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

PLANNER_PROMPT_VERSION = "m2-query-planner-v4-bilingual"
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
- The evidence index is English. Every searchable value must be concise,
  lowercase English: quoted_text, entities.text, actions, objects, locations,
  keywords, temporal anchors, and negative constraint text.
- Translate Chinese and other non-English requests by meaning, never by
  transliteration. raw_query is the only field that may remain non-English.
- Remove request filler such as "find", "show me", "scene", and "shot" from
  searchable fields. Keep only observable people, actions, objects, places,
  dialogue, and temporal constraints.
- relation is before, after, during, or between.
- between requires second_anchor; other relations require it to be null.
- ordinal value is a positive integer or "last".
- ordinal must be null unless the request explicitly says first, second, last,
  a numbered occurrence, 第一次, 第二次, 最后一次, or another explicit ordinal.
- negative field is entity, action, object, location, dialogue, or keyword.
- preserve quoted dialogue exactly when it is already English; otherwise
  translate it to concise English transcript wording without quotation marks.
- retain the exact input in raw_query.
- entities is the only array whose items are objects. actions, objects,
  locations, keywords, and evidence_preference must contain strings only.
- use empty arrays and null for absent constraints.
- output JSON only.
"""


def _strings(value: Any) -> list[str]:
    """Normalize string arrays without caching Python dict representations."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in items:
        if isinstance(item, dict):
            raw = next(
                (
                    item.get(key)
                    for key in ("text", "value", "name")
                    if item.get(key) is not None
                ),
                "",
            )
        else:
            raw = item
        text = str(raw).strip()
        if text and text not in result:
            result.append(text)
    return result


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
    result: dict[str, Any] = {
        "quoted_text": (
            str(payload["quoted_text"]).strip()
            if payload.get("quoted_text")
            else None
        ),
        "entities": _entities(payload.get("entities")),
    }
    for field in ("actions", "objects", "locations", "keywords"):
        result[field] = _strings(payload.get(field))
    temporal = []
    for raw in payload.get("temporal_constraints") or []:
        if not isinstance(raw, dict):
            continue
        relation = str(raw.get("relation") or "").lower().strip()
        if relation not in {"before", "after", "during", "between"}:
            continue
        item = dict(raw)
        item["relation"] = relation
        item["anchor"] = _anchor(item.get("anchor"))
        item["second_anchor"] = (
            _anchor(item.get("second_anchor"))
            if item.get("second_anchor") is not None
            else None
        )
        if not any(item["anchor"].values()):
            continue
        if relation == "between" and (
            item["second_anchor"] is None
            or not any(item["second_anchor"].values())
        ):
            continue
        if relation != "between":
            item["second_anchor"] = None
        temporal.append(item)
    result["temporal_constraints"] = temporal
    ordinal = payload.get("ordinal")
    if isinstance(ordinal, (int, str)):
        value: Any = int(ordinal) if str(ordinal).isdigit() else ordinal
        result["ordinal"] = (
            {"value": value, "scope": "matching_event"}
            if value == "last" or isinstance(value, int) and value > 0
            else None
        )
    elif isinstance(ordinal, dict):
        value = ordinal.get("value")
        value = int(value) if str(value).isdigit() else value
        scope = ordinal.get("scope")
        result["ordinal"] = (
            {
                "value": value,
                "scope": (
                    scope
                    if scope in {"matching_event", "after_temporal_filter"}
                    else "matching_event"
                ),
            }
            if value == "last" or isinstance(value, int) and value > 0
            else None
        )
    else:
        result["ordinal"] = None
    negatives = []
    valid_negative_fields = {
        "entity", "action", "object", "location", "dialogue", "keyword"
    }
    for raw in payload.get("negative_constraints") or []:
        item = (
            dict(raw)
            if isinstance(raw, dict)
            else {"field": "keyword", "text": str(raw)}
        )
        field = str(item.get("field") or "keyword").lower().strip()
        text = str(item.get("text") or "").strip()
        if field in valid_negative_fields and text:
            negatives.append({"field": field, "text": text})
    result["negative_constraints"] = negatives
    preferences = [
        item for item in _strings(payload.get("evidence_preference"))
        if item in {"visual", "dialogue", "script"}
    ]
    result["evidence_preference"] = preferences or ["visual", "dialogue"]
    direct = payload.get("require_direct_evidence", True)
    result["require_direct_evidence"] = (
        direct.lower() not in {"false", "0", "no"}
        if isinstance(direct, str)
        else bool(direct)
    )
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
            "max_tokens": 1536,
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(45.0))

        def request(body: dict[str, Any]) -> dict[str, Any]:
            response = request_with_retry(
                lambda: client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                ),
                max_attempts=retry_attempts,
            )
            return response.json()

        repaired = False
        try:
            raw = request(request_body)
            try:
                spec = normalize_planner_response(
                    raw,
                    query=query,
                    top_k=top_k,
                )
            except (TypeError, ValueError) as error:
                invalid = raw.get("choices", [{}])[0].get("message", {}).get(
                    "content",
                    raw,
                )
                repair_body = {
                    **request_body,
                    "messages": [
                        *request_body["messages"],
                        {
                            "role": "assistant",
                            "content": (
                                invalid
                                if isinstance(invalid, str)
                                else json.dumps(invalid, ensure_ascii=False)
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Repair the previous output to the required "
                                "JSON schema. Arrays other than entities must "
                                "contain strings. Preserve the query meaning. "
                                f"Validation error: {type(error).__name__}"
                            ),
                        },
                    ],
                }
                raw = request(repair_body)
                spec = normalize_planner_response(
                    raw,
                    query=query,
                    top_k=top_k,
                )
                repaired = True
            route_reason = (
                "cross-language or complex query requires structured planning"
            )
            if repaired:
                route_reason += "; repaired invalid model output once"
            return PlannerResult(
                query_spec=spec,
                trace=PlannerTrace(
                    trace_id="pending",
                    status="LIVE",
                    planner="stepfun",
                    route_reason=route_reason,
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
