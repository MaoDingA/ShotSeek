#!/usr/bin/env python3
"""Run a frozen ShotSeek benchmark split and emit versioned reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shotseek.evaluation.benchmark import (
    BenchmarkThresholds,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--queries",
        type=Path,
        action="append",
        required=True,
        help="Repeat to combine multiple frozen JSONL files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--planner-mode", default="rule")
    parser.add_argument("--verifier-mode", default="rule")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--min-recall-at-1", type=float, default=0.65)
    parser.add_argument("--min-recall-at-3", type=float, default=0.80)
    parser.add_argument("--min-evidence-support-rate", type=float, default=0.85)
    parser.add_argument("--min-direct-evidence-rate", type=float, default=0.85)
    parser.add_argument("--max-p95-ms", type=float, default=3_000.0)
    parser.add_argument(
        "--max-median-boundary-error-ms",
        type=float,
        default=1_500.0,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    result = run_benchmark(
        project_root=root,
        database_path=root / args.database,
        query_paths=[root / path for path in args.queries],
        output_dir=root / args.output,
        split=args.split,
        thresholds=BenchmarkThresholds(
            recall_at_1=args.min_recall_at_1,
            recall_at_3=args.min_recall_at_3,
            evidence_support_rate=args.min_evidence_support_rate,
            direct_evidence_rate=args.min_direct_evidence_rate,
            p95_latency_ms=args.max_p95_ms,
            median_boundary_error_ms=args.max_median_boundary_error_ms,
        ),
        planner_mode=args.planner_mode,
        verifier_mode=args.verifier_mode,
        top_k=args.top_k,
        deterministic_replay=not args.skip_replay,
    )
    summary = {
        "status": "pass" if result.evaluation["pass"] else "fail",
        "split": result.evaluation["split"],
        "output": str(result.output_dir.relative_to(root)),
        "metrics": result.evaluation["metrics"],
        "failed_gates": [
            name
            for name, passed in result.evaluation["gates"].items()
            if not passed
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_pass and not result.evaluation["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
