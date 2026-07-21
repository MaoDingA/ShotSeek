"""Evidence-first verifier that never invents claims or model timestamps."""

from __future__ import annotations

from shotseek.planning.schema import NegativeConstraint, QuerySpecV2
from shotseek.retrieval.candidates import normalized_tokens
from shotseek.verification.schema import CandidateScene, VerificationResult


def _tokens(value: str | list[str]) -> set[str]:
    text = value if isinstance(value, str) else " ".join(value)
    return set(normalized_tokens(text))


def _coverage(needles: list[str], haystack: set[str]) -> float:
    requested = normalized_tokens(" ".join(needles))
    if not requested:
        return 1.0
    return sum(item in haystack for item in requested) / len(requested)


def _negative_haystack(
    candidate: CandidateScene, constraint: NegativeConstraint
) -> set[str]:
    fields = {
        "entity": candidate.characters + [candidate.summary],
        "action": candidate.actions + [candidate.summary],
        "object": candidate.objects + [candidate.summary],
        "location": [candidate.location or "", candidate.summary],
        "dialogue": [candidate.dialogue],
        "keyword": [
            candidate.summary,
            candidate.dialogue,
            *candidate.characters,
            *candidate.actions,
            *candidate.objects,
            *candidate.visible_text,
        ],
    }
    return _tokens(fields[constraint.field])


class RuleEvidenceVerifier:
    """Validate every requested field against structured scene evidence."""

    def verify(
        self, spec: QuerySpecV2, candidate: CandidateScene
    ) -> VerificationResult:
        matched: list[str] = []
        failed: list[str] = []
        group_scores: list[float] = []

        dialogue_score = candidate.components.dialogue_score
        if spec.quoted_text:
            dialogue_score = float(
                spec.quoted_text.lower() in candidate.dialogue.lower()
            )
            group_scores.append(dialogue_score)
            (matched if dialogue_score == 1.0 else failed).append("quoted_text")

        entity_evidence = _tokens(
            candidate.characters + [candidate.summary]
        )
        if candidate.characters:
            entity_evidence.add("person")
        if len(candidate.characters) >= 2:
            entity_evidence.add("people")
        structured = {
            "entities": entity_evidence,
            "actions": _tokens(candidate.actions + [candidate.summary]),
            "objects": _tokens(
                candidate.objects + candidate.characters + [candidate.summary]
            ),
            "locations": _tokens([candidate.location or "", candidate.summary]),
            "keywords": _tokens(
                [
                    candidate.summary,
                    candidate.dialogue,
                    *candidate.characters,
                    *candidate.actions,
                    *candidate.objects,
                    *candidate.visible_text,
                    candidate.location or "",
                ]
            ),
        }
        requested = {
            "entities": [item.text for item in spec.entities],
            "actions": spec.actions,
            "objects": spec.objects,
            "locations": spec.locations,
            "keywords": spec.keywords,
        }
        field_scores: dict[str, float] = {}
        for field, values in requested.items():
            if not values:
                continue
            coverage = _coverage(values, structured[field])
            field_scores[field] = coverage
            group_scores.append(coverage)
            (matched if coverage == 1.0 else failed).append(field)

        contradictions: list[str] = []
        for constraint in spec.negative_constraints:
            excluded = set(normalized_tokens(constraint.text))
            if excluded and excluded <= _negative_haystack(candidate, constraint):
                contradictions.append(f"{constraint.field}:{constraint.text}")

        evidence_coverage = (
            sum(group_scores) / len(group_scores) if group_scores else 0.0
        )
        has_visual_ref = any(
            ref.get("kind") == "visual" for ref in candidate.evidence_refs
        )
        has_dialogue_ref = any(
            ref.get("kind") == "dialogue" for ref in candidate.evidence_refs
        )
        direct_evidence = bool(
            (dialogue_score == 1.0 and has_dialogue_ref)
            or (
                any(value == 1.0 for value in field_scores.values())
                and has_visual_ref
            )
        )

        visual_fields = ("actions", "objects", "locations", "keywords")
        requested_visual = [name for name in visual_fields if requested[name]]
        visual_score = (
            sum(field_scores[name] for name in requested_visual)
            / len(requested_visual)
            if requested_visual
            else candidate.components.visual_score
        )
        components = candidate.components.model_copy(
            update={
                "dialogue_score": dialogue_score,
                "visual_score": visual_score,
                "entity_score": field_scores.get(
                    "entities", candidate.components.entity_score
                ),
                "evidence_coverage": evidence_coverage,
                "contradiction_penalty": 1.0 if contradictions else 0.0,
            }
        )
        all_constraints_match = bool(group_scores) and all(
            value == 1.0 for value in group_scores
        )
        if contradictions or failed:
            verdict = "unsupported"
        elif all_constraints_match and (
            direct_evidence or not spec.require_direct_evidence
        ):
            verdict = "supported"
        else:
            verdict = "uncertain"
        reason = {
            "supported": "all requested constraints have direct structured evidence",
            "unsupported": "one or more constraints conflict with candidate evidence",
            "uncertain": "candidate lacks sufficient direct evidence",
        }[verdict]
        confidence = (
            min(1.0, 0.70 + 0.30 * evidence_coverage)
            if verdict == "supported"
            else max(
                0.0,
                0.45 * evidence_coverage - components.contradiction_penalty,
            )
        )
        return VerificationResult(
            scene_id=candidate.scene_id,
            verdict=verdict,
            direct_evidence=direct_evidence,
            matched_constraints=matched,
            failed_constraints=failed,
            contradictions=contradictions,
            confidence=confidence,
            reason=reason,
            components=components,
            verifier="rule",
        )
