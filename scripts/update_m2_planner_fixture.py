#!/usr/bin/env python3
"""Create a sanitized planner fixture from a successful live M2A run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shotseek.fixtures import update_planner_fixture_from_live_run

ROOT = Path(__file__).resolve().parents[1]


def load_key() -> str | None:
    path = ROOT / ".env"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"STEPFUN_API_KEY", "STEP_API_KEY"}:
            return value.strip()
    return os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    outputs = update_planner_fixture_from_live_run(
        ROOT,
        ROOT / args.run,
        api_key=load_key(),
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
