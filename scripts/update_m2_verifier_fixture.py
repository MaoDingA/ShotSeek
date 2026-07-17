#!/usr/bin/env python3
"""Promote a successful live verifier response to a sanitized fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shotseek.fixtures import update_verifier_fixture_from_live_run

ROOT = Path(__file__).resolve().parents[1]


def load_key() -> str | None:
    path = ROOT / ".env"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = raw.strip().partition("=")
            if separator and key in {"STEPFUN_API_KEY", "STEP_API_KEY"}:
                return value.strip()
    return os.environ.get("STEPFUN_API_KEY") or os.environ.get("STEP_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    written = update_verifier_fixture_from_live_run(
        ROOT,
        ROOT / args.run,
        api_key=load_key(),
    )
    for path in written:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
