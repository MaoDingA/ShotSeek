"""Deterministic temporal and ordinal resolution over recalled candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shotseek.planning.schema import AnchorSpec, QuerySpecV2
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.verification.schema import CandidateScene


def _anchor_query(anchor: AnchorSpec, raw_query: str) -> QuerySpecV2:
    return QuerySpecV2(
        raw_query=f"anchor:{raw_query}",
        quoted_text=anchor.quoted_text,
        entities=anchor.entities,
        actions=anchor.actions,
        objects=anchor.objects,
        locations=anchor.locations,
        keywords=anchor.keywords,
        evidence_preference=["visual", "dialogue"],
        top_k=20,
    )


def _mark_temporal(candidate: CandidateScene) -> CandidateScene:
    components = candidate.components.model_copy(update={"temporal_score": 1.0})
    return candidate.model_copy(update={"components": components})


def apply_ordinal_constraint(
    spec: QuerySpecV2,
    candidates: list[CandidateScene],
) -> tuple[list[CandidateScene], dict[str, Any] | None]:
    if spec.ordinal is None:
        return candidates, None
    ordered = sorted(candidates, key=lambda item: (item.start_ms, item.scene_id))
    value = spec.ordinal.value
    if value == "last":
        selected = ordered[-1:] if ordered else []
    else:
        selected = ordered[value - 1 : value] if len(ordered) >= value else []
    return selected, {
        "value": value,
        "scope": spec.ordinal.scope,
        "input_candidate_count": len(ordered),
        "selected_scene_ids": [item.scene_id for item in selected],
    }


def resolve_temporal_constraints(
    database_path: Path,
    spec: QuerySpecV2,
    candidates: list[CandidateScene],
    *,
    apply_ordinal: bool = True,
) -> tuple[list[CandidateScene], dict[str, Any]]:
    """Apply temporal anchors.

    The full M2 pipeline passes apply_ordinal=False, verifies direct evidence,
    then applies the ordinal over supported candidates only.
    """
    current = list(candidates)
    trace_constraints: list[dict[str, Any]] = []
    for constraint in spec.temporal_constraints:
        anchors, _ = retrieve_candidates(
            database_path,
            _anchor_query(constraint.anchor, spec.raw_query),
            limit=20,
        )
        if not anchors:
            return [], {
                "constraints": trace_constraints,
                "failure": "anchor_not_found",
                "ordinal": None,
                "valid_candidate_count": 0,
            }
        first = anchors[0]
        second = None
        before_count = len(current)
        if constraint.relation == "after":
            current = [item for item in current if item.start_ms >= first.end_ms]
        elif constraint.relation == "before":
            current = [item for item in current if item.end_ms <= first.start_ms]
        elif constraint.relation == "during":
            current = [
                item
                for item in current
                if item.start_ms >= first.start_ms and item.end_ms <= first.end_ms
            ]
        else:
            if constraint.second_anchor is None:
                raise ValueError("between requires a second anchor")
            second_candidates, _ = retrieve_candidates(
                database_path,
                _anchor_query(constraint.second_anchor, spec.raw_query),
                limit=20,
            )
            if not second_candidates:
                return [], {
                    "constraints": trace_constraints,
                    "failure": "second_anchor_not_found",
                    "ordinal": None,
                    "valid_candidate_count": 0,
                }
            second = second_candidates[0]
            left, right = sorted((first, second), key=lambda item: item.start_ms)
            current = [
                item
                for item in current
                if item.start_ms >= left.end_ms and item.end_ms <= right.start_ms
            ]
        current = [_mark_temporal(item) for item in current]
        trace_constraints.append(
            {
                "relation": constraint.relation,
                "anchor_candidate_count": len(anchors),
                "anchor_scene_id": first.scene_id,
                "second_anchor_scene_id": second.scene_id if second else None,
                "candidate_count_before": before_count,
                "candidate_count_after": len(current),
            }
        )
    ordinal_trace = None
    if apply_ordinal:
        current, ordinal_trace = apply_ordinal_constraint(spec, current)
    return current, {
        "constraints": trace_constraints,
        "failure": None,
        "ordinal": ordinal_trace if apply_ordinal else {"pending_verification": True}
        if spec.ordinal is not None
        else None,
        "valid_candidate_count": len(current),
    }
