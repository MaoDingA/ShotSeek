#!/usr/bin/env python3
"""Run an auditable QuerySpec v2 planning request."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shotseek.m2a import run_m2a

ROOT = Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        cleaned = value.strip()
        os.environ.setdefault(key.strip(), cleaned)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--rule", action="store_true")
    modes.add_argument("--fixture", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fixture-path")
    args = parser.parse_args()
    mode = "live" if args.live else "fixture" if args.fixture else "rule"
    api_key = None
    if mode == "live":
        load_project_env()
        api_key = os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")
    output = run_m2a(
        project_root=ROOT,
        query=args.query,
        mode=mode,
        output_dir=ROOT / args.output if args.output else None,
        fixture_path=ROOT / args.fixture_path if args.fixture_path else None,
        api_key=api_key,
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
