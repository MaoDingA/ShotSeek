#!/usr/bin/env python3
"""Verify a generated M1B scene bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shotseek.m1b import verify_m1b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="runs/m1b/latest")
    args = parser.parse_args()
    result = verify_m1b(Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
