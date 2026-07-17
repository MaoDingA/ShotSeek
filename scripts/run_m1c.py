#!/usr/bin/env python3
"""Build and evaluate the M1C SQLite retrieval baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from shotseek.m1c import run_m1c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1a", default="runs/m1a/latest")
    parser.add_argument("--m1b", default="runs/m1b/latest")
    parser.add_argument("--output", default="runs/m1c/latest")
    parser.add_argument("--queries", default="eval/m1_queries.jsonl")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = run_m1c(
        project_root=root,
        m1a_dir=root / args.m1a,
        m1b_dir=root / args.m1b,
        output_dir=root / args.output,
        queries_path=root / args.queries,
    )
    print(output)


if __name__ == "__main__":
    main()
