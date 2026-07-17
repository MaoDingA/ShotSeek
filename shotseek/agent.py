"""M2 Planner -> Top20 recall -> evidence verification -> Top3 pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from shotseek.planning.router import PlannerRouter
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.retrieval.temporal import (
    apply_ordinal_constraint,
    resolve_temporal_constraints,
)
from shotseek.traces.schema import AgentSearchResponse, AgentTrace
from shotseek.traces.store import TraceStore
from shotseek.verification.router import EvidenceVerifierRouter
from shotseek.verification.schema import VerifiedHit
from shotseek.verification.scoring import SCORING_VERSION, score_components


def _elapsed(started: float) -> float:
    return (perf_counter() - started) * 1000


def _trace_id(
    query: str, spec: dict[str, Any], final_scene_ids: list[str]
) -> str:
    payload = {
        "query": query,
        "spec": spec,
        "final_scene_ids": final_scene_ids,
        "scoring_version": SCORING_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return f"agent_{digest[:16]}"


class ShotSeekAgent:
    def __init__(
        self,
        *,
        database_path: Path,
        planner_cache_dir: Path | None = None,
        trace_dir: Path | None = None,
        planner: PlannerRouter | None = None,
        verifier: EvidenceVerifierRouter | None = None,
        verifier_cache_dir: Path | None = None,
    ) -> None:
        self.database_path = database_path
        self.planner = planner or PlannerRouter(cache_dir=planner_cache_dir)
        self.verifier = verifier or EvidenceVerifierRouter(
            cache_dir=verifier_cache_dir
        )
        self.trace_store = TraceStore(trace_dir) if trace_dir is not None else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        planner_mode: str = "auto",
        api_key: str | None = None,
        allow_network: bool = False,
        planner_fixture: dict[str, Any] | None = None,
        verifier_mode: str = "rule",
        verifier_fixture: dict[str, Any] | None = None,
    ) -> AgentSearchResponse:
        total_started = perf_counter()

        started = perf_counter()
        planned = self.planner.plan(
            query,
            mode=planner_mode,
            top_k=top_k,
            api_key=api_key,
            allow_network=allow_network,
            fixture_response=planner_fixture,
        )
        planner_ms = _elapsed(started)
        spec = planned.query_spec

        started = perf_counter()
        recalled, retrieval_trace = retrieve_candidates(
            self.database_path, spec, limit=20
        )
        retrieval_ms = _elapsed(started)

        started = perf_counter()
        temporally_valid, temporal_trace = resolve_temporal_constraints(
            self.database_path,
            spec,
            recalled,
            apply_ordinal=False,
        )
        temporal_ms = _elapsed(started)

        started = perf_counter()
        verified: list[VerifiedHit] = []
        verdict_counts = {
            "supported": 0,
            "unsupported": 0,
            "uncertain": 0,
        }
        verifier_status_counts = {
            "LIVE": 0,
            "CACHED": 0,
            "FALLBACK": 0,
            "RULE": 0,
        }
        direct_evidence_count = 0
        for position, candidate in enumerate(temporally_valid):
            candidate_mode = verifier_mode if position < 5 else "rule"
            result, verifier_trace = self.verifier.verify(
                spec,
                candidate,
                mode=candidate_mode,
                api_key=api_key,
                allow_network=allow_network,
                fixture_response=verifier_fixture,
            )
            verifier_status_counts[verifier_trace["status"]] += 1
            verdict_counts[result.verdict] += 1
            direct_evidence_count += int(result.direct_evidence)
            if result.verdict != "supported":
                continue
            scored_candidate = candidate.model_copy(
                update={"components": result.components}
            )
            verified.append(
                VerifiedHit(
                    candidate=scored_candidate,
                    verification=result,
                    final_score=score_components(spec, result.components),
                )
            )

        if spec.ordinal is not None:
            selected_candidates, ordinal_trace = apply_ordinal_constraint(
                spec, [item.candidate for item in verified]
            )
            selected_ids = {item.scene_id for item in selected_candidates}
            verified = [
                item for item in verified if item.candidate.scene_id in selected_ids
            ]
            temporal_trace["ordinal"] = ordinal_trace
            temporal_trace["valid_candidate_count"] = len(verified)

        verified.sort(
            key=lambda item: (
                -item.final_score,
                item.candidate.start_ms,
                item.candidate.scene_id,
            )
        )
        hits = verified[:top_k]
        verification_ms = _elapsed(started)
        final_ids = [item.candidate.scene_id for item in hits]

        phase_latency = {
            "planner": planner_ms,
            "retrieval": retrieval_ms,
            "temporal": temporal_ms,
            "verification": verification_ms,
        }
        total_ms = _elapsed(total_started)
        overall_status = planned.trace.status
        if verifier_status_counts["LIVE"]:
            overall_status = "LIVE"
        elif verifier_status_counts["FALLBACK"]:
            overall_status = "FALLBACK"
        elif verifier_status_counts["CACHED"] and overall_status == "RULE":
            overall_status = "CACHED"
        trace = AgentTrace(
            trace_id=_trace_id(
                query,
                spec.model_dump(mode="json"),
                final_ids,
            ),
            status=overall_status,
            query=query,
            query_spec=spec,
            planner=planned.trace,
            retrieval={
                **retrieval_trace,
                "recalled_scene_ids": [item.scene_id for item in recalled],
            },
            temporal=temporal_trace,
            verification={
                "requested_mode": verifier_mode,
                "model_candidate_limit": 5,
                "status_counts": verifier_status_counts,
                "input_candidate_count": len(temporally_valid),
                "verdict_counts": verdict_counts,
                "direct_evidence_count": direct_evidence_count,
                "unsupported_claim_count": 0,
                "supported_scene_ids": [
                    item.candidate.scene_id for item in verified
                ],
            },
            final_scene_ids=final_ids,
            phase_latency_ms=phase_latency,
            total_latency_ms=total_ms,
            scoring_version=SCORING_VERSION,
        )
        if self.trace_store is not None:
            self.trace_store.put(trace)
        return AgentSearchResponse(hits=hits, trace=trace)
