#!/usr/bin/env python3
"""Serve the read-only ShotSeek M2 API."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from shotseek.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.strip().partition("=")
        if separator:
            os.environ.setdefault(key.strip(), value.strip())


load_project_env()
app = create_app(
    database_path=ROOT / "runs" / "m1c" / "latest" / "search.sqlite3",
    manifest_path=ROOT / "runs" / "m1a" / "latest" / "manifest.json",
    trace_dir=ROOT / "runs" / "m2c" / "traces",
    planner_cache_dir=ROOT / "runs" / "m2c" / "planner-cache",
    verifier_cache_dir=ROOT / "runs" / "m2c" / "verifier-cache",
    allow_network=os.environ.get("SHOTSEEK_ALLOW_NETWORK") == "1",
    api_key=(
        os.environ.get("STEPFUN_API_KEY")
        or os.environ.get("STEP_API_KEY")
    ),
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("SHOTSEEK_HOST", "127.0.0.1"),
        port=int(os.environ.get("SHOTSEEK_PORT", "8000")),
        reload=False,
    )
