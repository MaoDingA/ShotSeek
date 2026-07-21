"""Planner routing, caching, fallback, and stable trace identity."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from shotseek.planning.cache import PlannerCache, cache_key
from shotseek.planning.rules import RulePlanner
from shotseek.planning.schema import PlannerResult, PlannerTrace
from shotseek.planning.stepfun import DEFAULT_PLANNER_MODEL, StepFunPlanner

PlannerMode = Literal["auto", "rule", "stepfun", "cache"]
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def query_requires_model(query: str) -> bool:
    # The evidence timeline is normalized to English. Route every Chinese
    # query through StepFun so arbitrary people, actions, objects, and places
    # are translated instead of depending on a small hand-written alias list.
    if HAN_RE.search(query):
        return True
    lowered = query.lower()
    complex_markers = (
        "before", "during", "between", "second", "last", "not ", "exclude",
        "之前", "期间", "当中", "之间", "第二次", "最后", "不是", "排除", "第3次",
    )
    if any(marker in lowered for marker in complex_markers):
        return True
    ordinal_match = re.search(r"第\s*[2-9]\d*\s*次", lowered)
    return ordinal_match is not None


def _trace_id(query: str, result: PlannerResult) -> str:
    payload = {
        "query": query,
        "spec": result.query_spec.model_dump(mode="json"),
        "planner": result.trace.planner,
        "status": result.trace.status,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return f"trace_{digest[:16]}"


def _with_trace(result: PlannerResult, **updates: Any) -> PlannerResult:
    trace = result.trace.model_copy(update=updates)
    provisional = result.model_copy(update={"trace": trace})
    return provisional.model_copy(
        update={"trace": trace.model_copy(update={"trace_id": _trace_id(result.query_spec.raw_query, provisional)})}
    )


class PlannerRouter:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        rule_planner: RulePlanner | None = None,
        stepfun_planner: StepFunPlanner | None = None,
    ) -> None:
        self.cache = PlannerCache(cache_dir) if cache_dir is not None else None
        self.rule = rule_planner or RulePlanner()
        self.stepfun = stepfun_planner or StepFunPlanner()

    def plan(
        self,
        query: str,
        *,
        mode: PlannerMode = "auto",
        top_k: int = 3,
        api_key: str | None = None,
        allow_network: bool = False,
        fixture_response: dict[str, Any] | None = None,
    ) -> PlannerResult:
        started = perf_counter()
        key = cache_key(query, top_k=top_k, model=self.stepfun.model)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                base = self.rule.plan(query, top_k=top_k)
                result = base.model_copy(
                    update={
                        "query_spec": cached,
                        "trace": PlannerTrace(
                            trace_id="pending",
                            status="CACHED",
                            planner="cache",
                            route_reason="content-addressed planner cache hit",
                            cache_hit=True,
                            latency_ms=(perf_counter() - started) * 1000,
                            model=self.stepfun.model,
                            prompt_version=base.trace.prompt_version,
                        ),
                    }
                )
                return _with_trace(result)
        if mode == "cache":
            fallback = self.rule.plan(
                query,
                top_k=top_k,
                status="FALLBACK",
                route_reason="cache miss fell back to deterministic planner",
            )
            return _with_trace(
                fallback,
                fallback_reason="cache_miss",
                latency_ms=(perf_counter() - started) * 1000,
            )

        use_model = mode == "stepfun" or (
            mode == "auto" and query_requires_model(query)
        )
        if mode == "rule" or not use_model:
            result = self.rule.plan(
                query,
                top_k=top_k,
                route_reason=(
                    "explicit rule mode"
                    if mode == "rule"
                    else "simple query routed to deterministic planner"
                ),
            )
            return _with_trace(result)

        try:
            if fixture_response is not None:
                result = self.stepfun.plan_fixture(
                    query, fixture_response, top_k=top_k
                )
            elif allow_network and api_key:
                result = self.stepfun.plan_live(
                    query, api_key=api_key, top_k=top_k
                )
            else:
                raise RuntimeError("StepFun planner unavailable without fixture or network")
        except Exception as exc:
            fallback = self.rule.plan(
                query,
                top_k=top_k,
                status="FALLBACK",
                route_reason="StepFun planner failed; deterministic fallback used",
            )
            return _with_trace(
                fallback,
                fallback_reason=type(exc).__name__,
                latency_ms=(perf_counter() - started) * 1000,
            )
        if self.cache is not None:
            self.cache.put(key, result.query_spec)
        return _with_trace(result)
