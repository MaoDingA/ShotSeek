#!/usr/bin/env python3
"""Run a StepFun evidence-verifier contract probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shotseek.m2b import run_m2b_verifier

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
        if separator:
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--fixture", action="store_true")
    parser.add_argument("--query", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixture-path")
    args = parser.parse_args()
    mode = "live" if args.live else "fixture"
    load_project_env()
    output = run_m2b_verifier(
        project_root=ROOT,
        query=args.query,
        scene_id=args.scene_id,
        mode=mode,
        output_dir=ROOT / args.output,
        api_key=(
            os.environ.get("STEPFUN_API_KEY")
            or os.environ.get("STEP_API_KEY")
        ),
        fixture_path=(
            ROOT / args.fixture_path if args.fixture_path else None
        ),
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
