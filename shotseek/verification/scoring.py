"""Versioned, query-aware scoring for verified M2 search hits."""

from __future__ import annotations

from typing import Literal

from shotseek.planning.schema import QuerySpecV2
from shotseek.verification.schema import ScoreComponents

SCORING_VERSION = "m2-evidence-score-v1"
ScoreProfile = Literal["dialogue", "visual", "multimodal", "temporal"]

WEIGHTS: dict[ScoreProfile, dict[str, float]] = {
    "dialogue": {
        "dialogue_score": 0.60,
        "entity_score": 0.15,
        "visual_score": 0.10,
        "evidence_coverage": 0.10,
        "boundary_quality": 0.05,
    },
    "visual": {
        "visual_score": 0.45,
        "lexical_score": 0.20,
        "entity_score": 0.15,
        "evidence_coverage": 0.15,
        "boundary_quality": 0.05,
    },
    "multimodal": {
        "dialogue_score": 0.30,
        "visual_score": 0.25,
        "lexical_score": 0.15,
        "entity_score": 0.10,
        "evidence_coverage": 0.15,
        "boundary_quality": 0.05,
    },
    "temporal": {
        "temporal_score": 0.30,
        "visual_score": 0.25,
        "lexical_score": 0.15,
        "entity_score": 0.10,
        "evidence_coverage": 0.15,
        "boundary_quality": 0.05,
    },
}


def select_profile(spec: QuerySpecV2) -> ScoreProfile:
    if spec.temporal_constraints or spec.ordinal is not None:
        return "temporal"
    has_visual = bool(
        spec.entities or spec.actions or spec.objects or spec.locations or spec.keywords
    )
    if spec.quoted_text and has_visual:
        return "multimodal"
    if spec.quoted_text:
        return "dialogue"
    return "visual"


def score_components(spec: QuerySpecV2, components: ScoreComponents) -> float:
    weights = WEIGHTS[select_profile(spec)]
    score = sum(getattr(components, name) * weight for name, weight in weights.items())
    score -= components.contradiction_penalty
    return max(0.0, min(1.0, score))
