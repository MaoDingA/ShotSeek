"""Typed M1C query-plan and result contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QuerySpec(RetrievalModel):
    raw_query: str = Field(min_length=1)
    normalized_query: str = Field(min_length=1)
    quoted_text: str | None = None
    terms: list[str]
    temporal_relation: Literal["after"] | None = None
    anchor_terms: list[str] = Field(default_factory=list)
    ordinal: int | None = Field(default=None, ge=1)


class SearchHit(RetrievalModel):
    scene_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    summary: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    match_type: Literal["dialogue", "visual", "multimodal", "temporal"]
    evidence_refs: list[dict[str, str]]
