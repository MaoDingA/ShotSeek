#!/usr/bin/env python3
"""Generate sanitized fixtures only from a fully successful live M0 run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shotseek.fixtures import update_fixtures_from_live_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize a complete M0 run into fixtures")
    parser.add_argument("--run", type=Path, required=True, help="Live runs/m0/<run_id> directory")
    return parser.parse_args()


def load_project_key() -> str | None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"STEPFUN_API_KEY", "STEP_API_KEY"}:
                return value.strip().strip("\"'") or None
    return os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")


def main() -> int:
    args = parse_args()
    run_dir = args.run if args.run.is_absolute() else PROJECT_ROOT / args.run
    try:
        written = update_fixtures_from_live_run(
            PROJECT_ROOT,
            run_dir,
            api_key=load_project_key(),
        )
    except ValueError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
