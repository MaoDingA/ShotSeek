#!/usr/bin/env python3
"""Run the deterministic 40-query M2 evaluation."""

from pathlib import Path

from shotseek.evaluation.m2 import run_m2_evaluation

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    output = run_m2_evaluation(
        project_root=ROOT,
        output_dir=ROOT / "runs" / "m2" / "evaluation-v1",
    )
    print(output.relative_to(ROOT))
