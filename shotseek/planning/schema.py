"""Strict QuerySpec v2 and planner execution contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class EntityConstraint(PlanningModel):
    text: str = Field(min_length=1)
    role: Literal["subject", "object", "other"] = "other"


class AnchorSpec(PlanningModel):
    quoted_text: str | None = None
    entities: list[EntityConstraint] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_content(self) -> "AnchorSpec":
        if not any(
            (
                self.quoted_text,
                self.entities,
                self.actions,
                self.objects,
                self.locations,
                self.keywords,
            )
        ):
            raise ValueError("temporal anchor must contain at least one constraint")
        return self


class TemporalConstraint(PlanningModel):
    relation: Literal["before", "after", "during", "between"]
    anchor: AnchorSpec
    second_anchor: AnchorSpec | None = None

    @model_validator(mode="after")
    def validate_between(self) -> "TemporalConstraint":
        if self.relation == "between" and self.second_anchor is None:
            raise ValueError("between requires second_anchor")
        if self.relation != "between" and self.second_anchor is not None:
            raise ValueError("second_anchor is only valid for between")
        return self


class OrdinalConstraint(PlanningModel):
    value: int | Literal["last"]
    scope: Literal["matching_event", "after_temporal_filter"] = "matching_event"

    @model_validator(mode="after")
    def validate_value(self) -> "OrdinalConstraint":
        if isinstance(self.value, int) and self.value < 1:
            raise ValueError("ordinal value must be positive")
        return self


class NegativeConstraint(PlanningModel):
    field: Literal["entity", "action", "object", "location", "dialogue", "keyword"]
    text: str = Field(min_length=1)


class QuerySpecV2(PlanningModel):
    schema_version: Literal["query-v2"] = "query-v2"
    intent: Literal["find_scene"] = "find_scene"
    raw_query: str = Field(min_length=1)
    quoted_text: str | None = None
    entities: list[EntityConstraint] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)
    ordinal: OrdinalConstraint | None = None
    negative_constraints: list[NegativeConstraint] = Field(default_factory=list)
    evidence_preference: list[Literal["visual", "dialogue", "script"]] = Field(
        default_factory=lambda: ["visual", "dialogue"]
    )
    require_direct_evidence: bool = True
    top_k: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def reject_empty_plan(self) -> "QuerySpecV2":
        if not any(
            (
                self.quoted_text,
                self.entities,
                self.actions,
                self.objects,
                self.locations,
                self.keywords,
                self.temporal_constraints,
            )
        ):
            raise ValueError("query plan must contain at least one searchable constraint")
        return self


class PlannerTrace(PlanningModel):
    trace_id: str = Field(min_length=1)
    status: Literal["LIVE", "CACHED", "FALLBACK", "RULE"]
    planner: Literal["stepfun", "rule", "cache"]
    route_reason: str = Field(min_length=1)
    cache_hit: bool
    fallback_reason: str | None = None
    latency_ms: float = Field(ge=0.0)
    model: str | None = None
    prompt_version: str
    schema_version: Literal["query-v2"] = "query-v2"


class PlannerResult(PlanningModel):
    query_spec: QuerySpecV2
    trace: PlannerTrace
    raw_response: dict[str, Any] | None = None
