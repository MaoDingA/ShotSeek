#!/usr/bin/env python3
"""Run and save the complete M2 release audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shotseek.m2_verify import verify_m2_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default="runs/m2/evaluation-v1")
    parser.add_argument(
        "--output", default="runs/m2/completion-report.json"
    )
    parser.add_argument("--skip-repository-checks", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = verify_m2_completion(
        project_root=root,
        evaluation_dir=root / args.evaluation,
        run_repository_checks=not args.skip_repository_checks,
    )
    encoded = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
