#!/usr/bin/env python3
"""Run the deterministic M1A pipeline on the golden sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from shotseek.m1a import run_m1a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="samples/golden.mp4")
    parser.add_argument("--output", default="runs/m1a/latest")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = run_m1a(
        project_root=root,
        video_path=root / args.video,
        output_dir=root / args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
