"""Cross-record validation for an M0 evidence timeline."""

from __future__ import annotations

from collections.abc import Sequence

from shotseek.schemas import BoundaryStatus, EvidenceKind, EvidenceSpan


def validate_evidence_timeline(
    evidence: Sequence[EvidenceSpan], video_duration_ms: int
) -> None:
    if video_duration_ms <= 0:
        raise ValueError("video_duration_ms must be positive")
    errors: list[str] = []
    previous_key: tuple[int, int, str] | None = None
    seen_ids: set[str] = set()

    for item in evidence:
        key = (item.start_ms, item.end_ms, item.evidence_id)
        if previous_key is not None and key < previous_key:
            errors.append(f"timeline is not sorted at {item.evidence_id}")
        previous_key = key
        if item.evidence_id in seen_ids:
            errors.append(f"duplicate evidence_id: {item.evidence_id}")
        seen_ids.add(item.evidence_id)
        if item.start_ms < 0 or item.end_ms <= item.start_ms:
            errors.append(f"invalid range: {item.evidence_id}")
        if item.end_ms > video_duration_ms:
            errors.append(f"evidence exceeds video duration: {item.evidence_id}")
        if not item.source_ref.strip():
            errors.append(f"missing source_ref: {item.evidence_id}")
        if item.kind == EvidenceKind.VISUAL and item.boundary_status != BoundaryStatus.APPROXIMATE:
            errors.append(f"visual boundary must be approximate: {item.evidence_id}")
        if item.kind == EvidenceKind.DIALOGUE and item.boundary_status != BoundaryStatus.ASR_TIMESTAMP:
            errors.append(f"dialogue boundary must be asr_timestamp: {item.evidence_id}")

    if errors:
        raise ValueError("; ".join(errors))
