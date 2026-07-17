"""Repository-local JSON trace persistence."""

from __future__ import annotations

import json
from pathlib import Path

from shotseek.traces.schema import AgentTrace


class TraceStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def put(self, trace: AgentTrace) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{trace.trace_id}.json"
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(
                trace.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def get(self, trace_id: str) -> AgentTrace | None:
        if not trace_id.startswith("agent_") or "/" in trace_id:
            return None
        path = self.directory / f"{trace_id}.json"
        if not path.is_file():
            return None
        return AgentTrace.model_validate_json(path.read_text(encoding="utf-8"))
