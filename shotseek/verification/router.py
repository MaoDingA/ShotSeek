"""Route candidate verification through rule, cache, or StepFun."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shotseek.planning.schema import QuerySpecV2
from shotseek.verification.cache import VerifierCache, verifier_cache_key
from shotseek.verification.rules import RuleEvidenceVerifier
from shotseek.verification.schema import CandidateScene, VerificationResult
from shotseek.verification.stepfun import (
    StepFunEvidenceVerifier,
    requires_semantic_review,
)


class EvidenceVerifierRouter:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        rule_verifier: RuleEvidenceVerifier | None = None,
        stepfun_verifier: StepFunEvidenceVerifier | None = None,
    ) -> None:
        self.rule = rule_verifier or RuleEvidenceVerifier()
        self.stepfun = stepfun_verifier or StepFunEvidenceVerifier(
            rule_verifier=self.rule
        )
        self.cache = VerifierCache(cache_dir) if cache_dir is not None else None

    def verify(
        self,
        spec: QuerySpecV2,
        candidate: CandidateScene,
        *,
        mode: str = "rule",
        api_key: str | None = None,
        allow_network: bool = False,
        fixture_response: dict[str, Any] | None = None,
    ) -> tuple[VerificationResult, dict[str, Any]]:
        if mode not in {"auto", "rule", "stepfun", "cache"}:
            raise ValueError("invalid verifier mode")
        key = verifier_cache_key(spec, candidate, model=self.stepfun.model)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached, {
                    "status": "CACHED",
                    "cache_hit": True,
                    "fallback_reason": None,
                }
        baseline = self.rule.verify(spec, candidate)
        if mode == "rule" or (
            mode == "auto"
            and baseline.verdict != "uncertain"
            and not requires_semantic_review(spec, baseline)
        ):
            return baseline, {
                "status": "RULE",
                "cache_hit": False,
                "fallback_reason": None,
            }
        if mode == "cache":
            return baseline, {
                "status": "FALLBACK",
                "cache_hit": False,
                "fallback_reason": "cache_miss",
            }
        try:
            if fixture_response is not None:
                result = self.stepfun.verify_fixture(
                    spec, candidate, fixture_response
                )
                status = "CACHED"
                raw_response = fixture_response
                latency_ms = 0.0
            elif allow_network and api_key:
                result, raw_response, latency_ms = self.stepfun.verify_live(
                    spec, candidate, api_key=api_key
                )
                status = "LIVE"
            else:
                raise RuntimeError(
                    "StepFun verifier unavailable without fixture or network"
                )
        except Exception as exc:
            return baseline, {
                "status": "FALLBACK",
                "cache_hit": False,
                "fallback_reason": type(exc).__name__,
            }
        if self.cache is not None:
            self.cache.put(key, result)
        return result, {
            "status": status,
            "cache_hit": status == "CACHED",
            "fallback_reason": None,
            "latency_ms": latency_ms,
            "raw_response": raw_response,
        }
