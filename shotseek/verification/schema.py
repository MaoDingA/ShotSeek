"""Typed candidate, verification, and final-hit contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VerificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreComponents(VerificationModel):
    lexical_score: float = Field(ge=0.0, le=1.0)
    dialogue_score: float = Field(ge=0.0, le=1.0)
    visual_score: float = Field(ge=0.0, le=1.0)
    entity_score: float = Field(ge=0.0, le=1.0)
    temporal_score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    boundary_quality: float = Field(ge=0.0, le=1.0)
    contradiction_penalty: float = Field(ge=0.0, le=1.0)


class CandidateScene(VerificationModel):
    scene_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    summary: str = Field(min_length=1)
    characters: list[str]
    actions: list[str]
    objects: list[str]
    location: str | None
    visible_text: list[str]
    dialogue: str
    shot_ids: list[str] = Field(min_length=1)
    evidence_refs: list[dict[str, str]]
    retrieval_route: Literal["exact_dialogue", "strict_and", "relaxed_or", "all_scenes"]
    retrieval_score: float = Field(ge=0.0, le=1.0)
    components: ScoreComponents


class VerificationResult(VerificationModel):
    scene_id: str
    verdict: Literal["supported", "unsupported", "uncertain"]
    direct_evidence: bool
    matched_constraints: list[str]
    failed_constraints: list[str]
    contradictions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    components: ScoreComponents
    verifier: Literal["rule", "stepfun", "cache"]


class VerifiedHit(VerificationModel):
    candidate: CandidateScene
    verification: VerificationResult
    final_score: float = Field(ge=0.0, le=1.0)
