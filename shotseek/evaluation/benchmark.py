"""Reusable benchmark runner for regression, holdout and long-form datasets."""

from __future__ import annotations

import hashlib
import html
import json
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shotseek.agent import ShotSeekAgent
from shotseek.m0 import ensure_within_project


class BenchmarkCase(BaseModel):
    """One frozen natural-language retrieval judgement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)
    acceptable_scene_ids: list[str]
    reference_start_ms: int | None = Field(default=None, ge=0)
    reference_end_ms: int | None = Field(default=None, gt=0)
    expected_plan: dict[str, Any] | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "BenchmarkCase":
        if len(self.acceptable_scene_ids) != len(set(self.acceptable_scene_ids)):
            raise ValueError("acceptable_scene_ids must be unique")
        if (self.reference_start_ms is None) != (self.reference_end_ms is None):
            raise ValueError("temporal reference requires both start and end")
        if (
            self.reference_start_ms is not None
            and self.reference_end_ms is not None
            and self.reference_end_ms <= self.reference_start_ms
        ):
            raise ValueError("temporal reference range must be positive")
        return self


@dataclass(frozen=True)
class BenchmarkThresholds:
    recall_at_1: float = 0.65
    recall_at_3: float = 0.80
    evidence_support_rate: float = 0.85
    direct_evidence_rate: float = 0.85
    p95_latency_ms: float = 3_000.0
    median_boundary_error_ms: float = 1_500.0


@dataclass(frozen=True)
class BenchmarkRunResult:
    output_dir: Path
    evaluation: dict[str, Any]


def _load_jsonl(path: Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(BenchmarkCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid benchmark case") from exc
    return cases


def load_cases(paths: Iterable[Path]) -> list[BenchmarkCase]:
    cases = [case for path in paths for case in _load_jsonl(path)]
    if not cases:
        raise ValueError("benchmark dataset is empty")
    identifiers = [case.query_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
        )
        raise ValueError(f"duplicate query_id values: {duplicates}")
    return cases


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _dump_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dataset_sha256(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _temporal_iou(
    actual_start_ms: int,
    actual_end_ms: int,
    reference_start_ms: int,
    reference_end_ms: int,
) -> float:
    intersection = max(
        0,
        min(actual_end_ms, reference_end_ms)
        - max(actual_start_ms, reference_start_ms),
    )
    union = max(actual_end_ms, reference_end_ms) - min(
        actual_start_ms, reference_start_ms
    )
    return intersection / union if union else 0.0


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _database_integrity(database: Path) -> bool:
    with sqlite3.connect(database) as connection:
        return connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _result_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [item for item in results if item["positive"]]
    negatives = [item for item in results if not item["positive"]]
    positive_count = len(positives)
    returned_hits = [hit for item in results for hit in item["hits"]]
    latencies = [float(item["latency_ms"]) for item in results]
    timed = [item for item in results if item["temporal_reference"] is not None]
    matched_timed = [
        item for item in timed if item["temporal"]["start_error_ms"] is not None
    ]
    start_errors = [
        float(item["temporal"]["start_error_ms"]) for item in matched_timed
    ]
    end_errors = [
        float(item["temporal"]["end_error_ms"]) for item in matched_timed
    ]
    return {
        "query_count": len(results),
        "positive_query_count": positive_count,
        "negative_query_count": len(negatives),
        "recall_at_1": (
            sum(item["correct_at_1"] for item in positives) / positive_count
            if positive_count
            else 1.0
        ),
        "recall_at_3": (
            sum(item["correct_at_3"] for item in positives) / positive_count
            if positive_count
            else 1.0
        ),
        "mrr": (
            sum(item["reciprocal_rank"] for item in positives) / positive_count
            if positive_count
            else 1.0
        ),
        "negative_false_positive_query_count": sum(
            bool(item["scene_ids"]) for item in negatives
        ),
        "negative_false_positive_hit_count": sum(
            len(item["scene_ids"]) for item in negatives
        ),
        "returned_hit_count": len(returned_hits),
        "verifier_precision": (
            sum(hit["is_acceptable"] for hit in returned_hits) / len(returned_hits)
            if returned_hits
            else 1.0
        ),
        "evidence_support_rate": (
            sum(hit["verdict"] == "supported" for hit in returned_hits)
            / len(returned_hits)
            if returned_hits
            else 1.0
        ),
        "direct_evidence_rate": (
            sum(hit["direct_evidence"] for hit in returned_hits)
            / len(returned_hits)
            if returned_hits
            else 1.0
        ),
        "malformed_evidence_reference_count": sum(
            not ref.get("evidence_id") or ref.get("kind") not in {"visual", "dialogue"}
            for hit in returned_hits
            for ref in hit["evidence_refs"]
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "temporal": {
            "reference_query_count": len(timed),
            "matched_query_count": len(matched_timed),
            "mean_iou": (
                sum(item["temporal"]["iou"] for item in timed) / len(timed)
                if timed
                else None
            ),
            "median_start_error_ms": (
                _percentile(start_errors, 0.50) if start_errors else None
            ),
            "median_end_error_ms": (
                _percentile(end_errors, 0.50) if end_errors else None
            ),
        },
    }


def _category_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)
    return {
        category: {
            "query_count": metrics["query_count"],
            "positive_query_count": metrics["positive_query_count"],
            "recall_at_1": metrics["recall_at_1"],
            "recall_at_3": metrics["recall_at_3"],
            "mrr": metrics["mrr"],
            "negative_false_positive_query_count": metrics[
                "negative_false_positive_query_count"
            ],
        }
        for category, items in sorted(grouped.items())
        for metrics in [_result_metrics(items)]
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_report(evaluation: dict[str, Any], results: list[dict[str, Any]]) -> str:
    metrics = evaluation["metrics"]
    temporal = metrics["temporal"]
    lines = [
        f"# ShotSeek Benchmark — {evaluation['split']}",
        "",
        f"- 状态：**{'PASS' if evaluation['pass'] else 'FAIL'}**",
        (
            f"- 查询数：{metrics['query_count']}（正例 "
            f"{metrics['positive_query_count']} / 负例 "
            f"{metrics['negative_query_count']}）"
        ),
        (
            f"- Recall@1 / Recall@3 / MRR：{metrics['recall_at_1']:.3f} / "
            f"{metrics['recall_at_3']:.3f} / {metrics['mrr']:.3f}"
        ),
        (
            "- 证据支持率 / 直接证据率："
            f"{metrics['evidence_support_rate']:.3f} / "
            f"{metrics['direct_evidence_rate']:.3f}"
        ),
        (
            "- 查询延迟 P50 / P95："
            f"{metrics['latency_ms']['p50']:.1f} ms / "
            f"{metrics['latency_ms']['p95']:.1f} ms"
        ),
        f"- 时间 IoU：{_format_metric(temporal['mean_iou'])}",
        f"- 数据集 SHA-256：`{evaluation['provenance']['dataset_sha256']}`",
        f"- 数据库 SHA-256：`{evaluation['provenance']['database_sha256']}`",
        "",
        "## 验收门",
        "",
        "| Gate | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in evaluation["gates"].items()
    )
    lines.extend(
        [
            "",
            "## 分类指标",
            "",
            "| Category | Queries | Recall@1 | Recall@3 | MRR | Negative FP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for category, item in evaluation["category_metrics"].items():
        lines.append(
            f"| {category} | {item['query_count']} | {item['recall_at_1']:.3f} | "
            f"{item['recall_at_3']:.3f} | {item['mrr']:.3f} | "
            f"{item['negative_false_positive_query_count']} |"
        )
    lines.extend(
        [
            "",
            "## 逐条结果",
            "",
            "| Query | Category | Expected | Returned | R@1 | Latency |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for item in results:
        text = item["text"].replace("|", "\\|")
        expected = ", ".join(item["acceptable_scene_ids"]) or "NEGATIVE"
        returned = ", ".join(item["scene_ids"]) or "NONE"
        lines.append(
            f"| {item['query_id']}: {text} | {item['category']} | {expected} | "
            f"{returned} | {'✓' if item['correct_at_1'] else '—'} | "
            f"{item['latency_ms']:.1f} ms |"
        )
    return "\n".join(lines) + "\n"


def _html_report(evaluation: dict[str, Any], results: list[dict[str, Any]]) -> str:
    metrics = evaluation["metrics"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class=\"{'pass' if passed else 'fail'}\">"
        f"{'PASS' if passed else 'FAIL'}</td></tr>"
        for name, passed in evaluation["gates"].items()
    )
    result_rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['query_id'])}</td>"
        f"<td>{html.escape(item['category'])}</td>"
        f"<td>{html.escape(item['text'])}</td>"
        f"<td>{html.escape(', '.join(item['acceptable_scene_ids']) or 'NEGATIVE')}</td>"
        f"<td>{html.escape(', '.join(item['scene_ids']) or 'NONE')}</td>"
        f"<td>{item['latency_ms']:.1f} ms</td>"
        "</tr>"
        for item in results
    )
    status = "PASS" if evaluation["pass"] else "FAIL"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShotSeek Benchmark — {html.escape(evaluation['split'])}</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #090d14; color: #e8edf5; }}
main {{ max-width: 1180px; margin: auto; padding: 48px 28px 72px; }}
h1 {{ font-size: 34px; margin: 0 0 8px; }}
.subtle {{ color: #8f9bad; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 28px 0; }}
.card {{ background: #111824; border: 1px solid #243044; border-radius: 14px; padding: 18px; }}
.card strong {{ display: block; font-size: 27px; margin-top: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0 36px; }}
th, td {{ border-bottom: 1px solid #253044; padding: 11px 9px; text-align: left; vertical-align: top; }}
th {{ color: #9aa7b8; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
.pass {{ color: #63e6a6; font-weight: 700; }}
.fail {{ color: #ff7b87; font-weight: 700; }}
.status {{ display: inline-block; border-radius: 999px; padding: 5px 10px; background: #1b2635; }}
@media (max-width: 760px) {{ .cards {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body><main>
<span class="status {'pass' if evaluation['pass'] else 'fail'}">{status}</span>
<h1>ShotSeek Benchmark</h1>
<p class="subtle">Split: {html.escape(evaluation['split'])} · Generated: {html.escape(evaluation['generated_at'])}</p>
<section class="cards">
<div class="card">Recall@1<strong>{metrics['recall_at_1']:.3f}</strong></div>
<div class="card">Recall@3<strong>{metrics['recall_at_3']:.3f}</strong></div>
<div class="card">Direct Evidence<strong>{metrics['direct_evidence_rate']:.3f}</strong></div>
<div class="card">P95 Latency<strong>{metrics['latency_ms']['p95']:.1f} ms</strong></div>
</section>
<h2>Acceptance Gates</h2><table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Query Results</h2><table><thead><tr><th>ID</th><th>Category</th><th>Query</th><th>Expected</th><th>Returned</th><th>Latency</th></tr></thead><tbody>{result_rows}</tbody></table>
</main></body></html>
"""


def run_benchmark(
    *,
    project_root: Path,
    database_path: Path,
    query_paths: list[Path],
    output_dir: Path,
    split: str,
    thresholds: BenchmarkThresholds | None = None,
    planner_mode: str = "rule",
    verifier_mode: str = "rule",
    top_k: int = 3,
    deterministic_replay: bool = True,
) -> BenchmarkRunResult:
    root = project_root.resolve()
    database = ensure_within_project(root, database_path)
    paths = [ensure_within_project(root, path) for path in query_paths]
    output = ensure_within_project(root, output_dir)
    if not database.is_file():
        raise FileNotFoundError(database)
    if top_k < 3:
        raise ValueError("benchmark top_k must be at least 3")
    cases = load_cases(paths)
    output.mkdir(parents=True, exist_ok=True)
    agent = ShotSeekAgent(database_path=database, trace_dir=output / "traces")
    results: list[dict[str, Any]] = []
    replay_material: list[dict[str, Any]] = []

    for case in cases:
        started = perf_counter()
        response = agent.search(
            case.text,
            top_k=top_k,
            planner_mode=planner_mode,
            verifier_mode=verifier_mode,
            allow_network=False,
        )
        elapsed_ms = (perf_counter() - started) * 1_000
        hits = response.hits
        scene_ids = [hit.candidate.scene_id for hit in hits]
        reciprocal_rank = 0.0
        for rank, scene_id in enumerate(scene_ids, start=1):
            if scene_id in case.acceptable_scene_ids:
                reciprocal_rank = 1.0 / rank
                break

        temporal_reference = (
            {
                "start_ms": case.reference_start_ms,
                "end_ms": case.reference_end_ms,
            }
            if case.reference_start_ms is not None
            else None
        )
        matched_hit = next(
            (
                hit
                for hit in hits
                if hit.candidate.scene_id in case.acceptable_scene_ids
            ),
            None,
        )
        temporal = {
            "iou": None,
            "start_error_ms": None,
            "end_error_ms": None,
        }
        if temporal_reference is not None:
            if matched_hit is None:
                temporal["iou"] = 0.0
            else:
                candidate = matched_hit.candidate
                reference_start = int(temporal_reference["start_ms"])
                reference_end = int(temporal_reference["end_ms"])
                temporal = {
                    "iou": _temporal_iou(
                        candidate.start_ms,
                        candidate.end_ms,
                        reference_start,
                        reference_end,
                    ),
                    "start_error_ms": abs(candidate.start_ms - reference_start),
                    "end_error_ms": abs(candidate.end_ms - reference_end),
                }

        result = {
            "query_id": case.query_id,
            "category": case.category,
            "text": case.text,
            "positive": bool(case.acceptable_scene_ids),
            "acceptable_scene_ids": case.acceptable_scene_ids,
            "scene_ids": scene_ids,
            "correct_at_1": bool(scene_ids)
            and scene_ids[0] in case.acceptable_scene_ids,
            "correct_at_3": any(
                scene_id in case.acceptable_scene_ids for scene_id in scene_ids[:3]
            ),
            "reciprocal_rank": reciprocal_rank,
            "latency_ms": elapsed_ms,
            "agent_status": response.trace.status,
            "trace_id": response.trace.trace_id,
            "phase_latency_ms": response.trace.phase_latency_ms,
            "temporal_reference": temporal_reference,
            "temporal": temporal,
            "hits": [
                {
                    "scene_id": hit.candidate.scene_id,
                    "start_ms": hit.candidate.start_ms,
                    "end_ms": hit.candidate.end_ms,
                    "final_score": hit.final_score,
                    "verdict": hit.verification.verdict,
                    "direct_evidence": hit.verification.direct_evidence,
                    "reason": hit.verification.reason,
                    "evidence_refs": hit.candidate.evidence_refs,
                    "is_acceptable": hit.candidate.scene_id
                    in case.acceptable_scene_ids,
                }
                for hit in hits
            ],
        }
        results.append(result)
        replay_material.append(
            {
                "query_id": case.query_id,
                "query_spec": response.trace.query_spec.model_dump(mode="json"),
                "scene_ids": scene_ids,
            }
        )

    replay_pass = True
    if deterministic_replay:
        replay: list[dict[str, Any]] = []
        replay_agent = ShotSeekAgent(database_path=database)
        for case in cases:
            response = replay_agent.search(
                case.text,
                top_k=top_k,
                planner_mode=planner_mode,
                verifier_mode=verifier_mode,
                allow_network=False,
            )
            replay.append(
                {
                    "query_id": case.query_id,
                    "query_spec": response.trace.query_spec.model_dump(mode="json"),
                    "scene_ids": [
                        hit.candidate.scene_id for hit in response.hits
                    ],
                }
            )
        replay_pass = replay == replay_material

    metrics = _result_metrics(results)
    category_metrics = _category_metrics(results)
    selected = thresholds or BenchmarkThresholds()
    temporal = metrics["temporal"]
    boundary_medians = [
        value
        for value in (
            temporal["median_start_error_ms"],
            temporal["median_end_error_ms"],
        )
        if value is not None
    ]
    database_ok = _database_integrity(database)
    gates = {
        "all_queries_executed": metrics["query_count"] == len(cases),
        "database_integrity_ok": database_ok,
        "recall_at_1_at_least_threshold": (
            metrics["recall_at_1"] >= selected.recall_at_1
        ),
        "recall_at_3_at_least_threshold": (
            metrics["recall_at_3"] >= selected.recall_at_3
        ),
        "evidence_support_rate_at_least_threshold": (
            metrics["evidence_support_rate"] >= selected.evidence_support_rate
        ),
        "direct_evidence_rate_at_least_threshold": (
            metrics["direct_evidence_rate"] >= selected.direct_evidence_rate
        ),
        "query_p95_at_most_threshold": (
            metrics["latency_ms"]["p95"] <= selected.p95_latency_ms
        ),
        "negative_false_positive_queries_zero": (
            metrics["negative_false_positive_query_count"] == 0
        ),
        "temporal_median_boundary_error_at_most_threshold": (
            not boundary_medians
            or max(boundary_medians) <= selected.median_boundary_error_ms
        ),
        "deterministic_replay": replay_pass,
    }
    evaluation = {
        "schema_version": "shotseek-benchmark-v1",
        "split": split,
        "generated_at": datetime.now(UTC).isoformat(),
        "pass": all(gates.values()),
        "provenance": {
            "git_commit": _git_commit(root),
            "database_path": str(database.relative_to(root)),
            "database_sha256": _sha256_file(database),
            "query_paths": [str(path.relative_to(root)) for path in paths],
            "dataset_sha256": _dataset_sha256(root, paths),
            "planner_mode": planner_mode,
            "verifier_mode": verifier_mode,
            "top_k": top_k,
            "network_calls": 0,
        },
        "thresholds": {
            "recall_at_1": selected.recall_at_1,
            "recall_at_3": selected.recall_at_3,
            "evidence_support_rate": selected.evidence_support_rate,
            "direct_evidence_rate": selected.direct_evidence_rate,
            "p95_latency_ms": selected.p95_latency_ms,
            "median_boundary_error_ms": selected.median_boundary_error_ms,
        },
        "metrics": metrics,
        "category_metrics": category_metrics,
        "gates": gates,
    }
    _dump_json(output / "results.json", results)
    _dump_json(output / "evaluation.json", evaluation)
    _atomic_write(output / "report.md", _markdown_report(evaluation, results))
    _atomic_write(output / "report.html", _html_report(evaluation, results))
    return BenchmarkRunResult(output_dir=output, evaluation=evaluation)
