#!/usr/bin/env python3
"""Build strict M1B scene candidates from an M1A bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from shotseek.m1b import run_m1b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1a", default="runs/m1a/latest")
    parser.add_argument("--output", default="runs/m1b/latest")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = run_m1b(
        project_root=root,
        m1a_dir=root / args.m1a,
        output_dir=root / args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
