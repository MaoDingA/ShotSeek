"""StepFun candidate verifier with deterministic evidence safety gates."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from shotseek.planning.schema import QuerySpecV2
from shotseek.providers.stepfun import DEFAULT_CHAT_BASE_URL, DEFAULT_VISION_MODEL
from shotseek.providers.stepfun.http import request_with_retry
from shotseek.providers.stepfun.vision import extract_json_object
from shotseek.retrieval.candidates import normalized_tokens
from shotseek.verification.rules import RuleEvidenceVerifier
from shotseek.verification.schema import CandidateScene, VerificationResult

VERIFIER_PROMPT_VERSION = "m2-evidence-verifier-v4-relationship-safe"
VERIFIER_SCHEMA_VERSION = "evidence-verdict-v1"
DEFAULT_VERIFIER_MODEL = DEFAULT_VISION_MODEL

VERIFIER_SYSTEM_PROMPT = """You are ShotSeek's candidate evidence verifier.
You receive one parsed QuerySpec and exactly one candidate scene with structured
evidence. Decide only whether that candidate is supported. Never select another
scene. Never create timestamps, frames, shot IDs, names, dialogue, or evidence.

Return one JSON object only:
{
  "verdict": "supported|unsupported|uncertain",
  "direct_evidence": true,
  "matched_constraints": ["quoted_text"],
  "failed_constraints": [],
  "contradictions": [],
  "confidence": 0.95,
  "reason": "short evidence-grounded reason"
}

Use supported only when every positive constraint is directly present and no
negative constraint is present. Treat unambiguous translation equivalents and
English synonyms as the same concept, but never use general world knowledge to
add evidence that the candidate does not state. Use uncertain when evidence is
incomplete.
"""

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SEMANTIC_REVIEW_MIN_COVERAGE = 0.60
RELATION_TOKENS = {
    "after",
    "before",
    "behind",
    "between",
    "during",
    "middle",
    "near",
    "together",
}


class ModelVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    verdict: str
    direct_evidence: bool
    matched_constraints: list[str] = Field(default_factory=list)
    failed_constraints: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


def normalize_verifier_response(raw: dict[str, Any]) -> ModelVerdict:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "verifier response is missing choices[0].message.content"
        ) from exc
    payload = (
        dict(content)
        if isinstance(content, dict)
        else extract_json_object(str(content))
    )
    verdict = str(payload.get("verdict", "")).lower().strip()
    if verdict not in {"supported", "unsupported", "uncertain"}:
        raise ValueError("invalid verifier verdict")
    normalized = {
        "verdict": verdict,
        "direct_evidence": bool(payload.get("direct_evidence", False)),
        "matched_constraints": [
            str(item) for item in payload.get("matched_constraints") or []
        ],
        "failed_constraints": [
            str(item) for item in payload.get("failed_constraints") or []
        ],
        "contradictions": [
            str(item) for item in payload.get("contradictions") or []
        ],
        "confidence": float(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason") or "model returned no reason"),
    }
    return ModelVerdict.model_validate(normalized)


def requires_semantic_review(
    spec: QuerySpecV2,
    baseline: VerificationResult,
) -> bool:
    """Allow synonym repair, but never override a missing relationship."""
    requested_relations = {
        "locations": set(normalized_tokens(" ".join(spec.locations))),
        "keywords": set(normalized_tokens(" ".join(spec.keywords))),
    }
    missing_relationship = any(
        field in baseline.failed_constraints
        and bool(tokens & RELATION_TOKENS)
        for field, tokens in requested_relations.items()
    )
    return bool(
        HAN_RE.search(spec.raw_query)
        and baseline.verdict == "unsupported"
        and baseline.direct_evidence
        and not baseline.contradictions
        and not missing_relationship
        and baseline.components.evidence_coverage >= SEMANTIC_REVIEW_MIN_COVERAGE
    )


class StepFunEvidenceVerifier:
    def __init__(
        self,
        *,
        model: str = DEFAULT_VERIFIER_MODEL,
        base_url: str = DEFAULT_CHAT_BASE_URL,
        client: httpx.Client | None = None,
        rule_verifier: RuleEvidenceVerifier | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.rule = rule_verifier or RuleEvidenceVerifier()

    def _apply_safety_gate(
        self,
        spec: QuerySpecV2,
        candidate: CandidateScene,
        model_result: ModelVerdict,
        *,
        verifier: str,
    ) -> VerificationResult:
        baseline = self.rule.verify(spec, candidate)
        precedence = {"unsupported": 0, "uncertain": 1, "supported": 2}
        semantic_upgrade = bool(
            requires_semantic_review(spec, baseline)
            and model_result.verdict == "supported"
            and model_result.direct_evidence
            and not model_result.failed_constraints
            and not model_result.contradictions
        )
        if semantic_upgrade:
            verdict = "supported"
            matched_constraints = (
                model_result.matched_constraints or baseline.matched_constraints
            )
            failed_constraints: list[str] = []
            contradictions: list[str] = []
            confidence = max(
                baseline.confidence,
                model_result.confidence
                * baseline.components.evidence_coverage,
            )
        else:
            verdict = min(
                (baseline.verdict, model_result.verdict),
                key=lambda item: precedence[item],
            )
            matched_constraints = baseline.matched_constraints
            failed_constraints = baseline.failed_constraints
            contradictions = baseline.contradictions
            confidence = min(baseline.confidence, model_result.confidence)
        direct = baseline.direct_evidence and model_result.direct_evidence
        if verdict == "supported" and not direct and spec.require_direct_evidence:
            verdict = "uncertain"
        return baseline.model_copy(
            update={
                "verdict": verdict,
                "direct_evidence": direct,
                "matched_constraints": matched_constraints,
                "failed_constraints": failed_constraints,
                "contradictions": contradictions,
                "confidence": confidence,
                "reason": model_result.reason,
                "verifier": verifier,
            }
        )

    def verify_fixture(
        self,
        spec: QuerySpecV2,
        candidate: CandidateScene,
        raw: dict[str, Any],
    ) -> VerificationResult:
        model_result = normalize_verifier_response(raw)
        return self._apply_safety_gate(
            spec, candidate, model_result, verifier="cache"
        )

    def verify_live(
        self,
        spec: QuerySpecV2,
        candidate: CandidateScene,
        *,
        api_key: str,
        retry_attempts: int = 3,
    ) -> tuple[VerificationResult, dict[str, Any], float]:
        if not api_key.strip():
            raise ValueError("StepFun API key is required")
        started = perf_counter()
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query_spec": spec.model_dump(mode="json"),
                            "candidate": candidate.model_dump(mode="json"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "stream": False,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 1024,
        }
        owns_client = self.client is None
        client = self.client or httpx.Client(timeout=httpx.Timeout(45.0))
        try:
            response = request_with_retry(
                lambda: client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                ),
                max_attempts=retry_attempts,
            )
            raw = response.json()
            model_result = normalize_verifier_response(raw)
            result = self._apply_safety_gate(
                spec, candidate, model_result, verifier="stepfun"
            )
            return result, raw, (perf_counter() - started) * 1000
        finally:
            if owns_client:
                client.close()
