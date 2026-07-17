"""Content-addressed QuerySpec cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shotseek.planning.schema import QuerySpecV2
from shotseek.planning.stepfun import PLANNER_PROMPT_VERSION, PLANNER_SCHEMA_VERSION


def cache_key(query: str, *, top_k: int, model: str) -> str:
    payload = {
        "query": query.strip(),
        "top_k": top_k,
        "model": model,
        "prompt_version": PLANNER_PROMPT_VERSION,
        "schema_version": PLANNER_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class PlannerCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> QuerySpecV2 | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return QuerySpecV2.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, spec: QuerySpecV2) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._path(key)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            spec.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
