#!/usr/bin/env python3
"""Run and optionally save the complete M1 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shotseek.m1_verify import verify_m1_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1a", default="runs/m1a/latest")
    parser.add_argument("--m1b", default="runs/m1b/latest")
    parser.add_argument("--m1c", default="runs/m1c/latest")
    parser.add_argument("--output")
    parser.add_argument(
        "--skip-repository-checks",
        action="store_true",
        help="Skip pytest, hygiene and git diff checks.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = verify_m1_completion(
        project_root=root,
        m1a_dir=root / args.m1a,
        m1b_dir=root / args.m1b,
        m1c_dir=root / args.m1c,
        run_repository_checks=not args.skip_repository_checks,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
