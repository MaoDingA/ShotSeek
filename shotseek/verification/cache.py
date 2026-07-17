"""Content-addressed cache for candidate verification results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shotseek.planning.schema import QuerySpecV2
from shotseek.verification.schema import CandidateScene, VerificationResult
from shotseek.verification.stepfun import (
    DEFAULT_VERIFIER_MODEL,
    VERIFIER_PROMPT_VERSION,
    VERIFIER_SCHEMA_VERSION,
)


def verifier_cache_key(
    spec: QuerySpecV2,
    candidate: CandidateScene,
    *,
    model: str = DEFAULT_VERIFIER_MODEL,
) -> str:
    payload = {
        "spec": spec.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
        "model": model,
        "prompt_version": VERIFIER_PROMPT_VERSION,
        "schema_version": VERIFIER_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


class VerifierCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: str) -> VerificationResult | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        result = VerificationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        return result.model_copy(update={"verifier": "cache"})

    def put(self, key: str, result: VerificationResult) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
