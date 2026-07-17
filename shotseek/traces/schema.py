"""Auditable M2 Agent Trace and search-response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from shotseek.planning.schema import PlannerTrace, QuerySpecV2
from shotseek.verification.schema import VerifiedHit


class TraceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentTrace(TraceModel):
    schema_version: Literal["m2-agent-trace-v1"] = "m2-agent-trace-v1"
    trace_id: str = Field(pattern=r"^agent_[a-f0-9]{16}$")
    status: Literal["LIVE", "CACHED", "FALLBACK", "RULE"]
    query: str = Field(min_length=1)
    query_spec: QuerySpecV2
    planner: PlannerTrace
    retrieval: dict[str, Any]
    temporal: dict[str, Any]
    verification: dict[str, Any]
    final_scene_ids: list[str]
    phase_latency_ms: dict[str, float]
    total_latency_ms: float = Field(ge=0.0)
    scoring_version: str


class AgentSearchResponse(TraceModel):
    hits: list[VerifiedHit]
    trace: AgentTrace
