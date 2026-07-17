#!/usr/bin/env python3
"""Verify an M2A planner artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shotseek.m2a import verify_m2a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    result = verify_m2a(Path(args.run).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
