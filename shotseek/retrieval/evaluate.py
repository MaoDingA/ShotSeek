"""Deterministic 15-query M1C evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from shotseek.retrieval.query_rules import plan_query
from shotseek.retrieval.sqlite_index import search

EXPECTED_CATEGORIES = {
    "exact_dialogue": 4,
    "visual": 4,
    "multimodal": 3,
    "temporal": 2,
    "negative": 2,
}


def load_query_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"query_id", "category", "text", "acceptable_scene_ids"}
        if set(item) != required:
            raise ValueError(f"invalid query keys on line {line_number}")
        if not isinstance(item["acceptable_scene_ids"], list):
            raise ValueError(f"acceptable_scene_ids must be a list on line {line_number}")
        cases.append(item)
    query_ids = [str(item["query_id"]) for item in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be unique")
    categories = Counter(str(item["category"]) for item in cases)
    if dict(categories) != EXPECTED_CATEGORIES:
        raise ValueError(
            f"query categories must equal {EXPECTED_CATEGORIES}, got {dict(categories)}"
        )
    return cases


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_evaluation(
    database_path: Path,
    cases: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    query_results: list[dict[str, Any]] = []
    stable_first: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for case in cases:
        spec = plan_query(str(case["text"]))
        started = perf_counter()
        hits = search(database_path, spec, top_k=top_k)
        latency_ms = (perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        hit_payload = [hit.model_dump(mode="json") for hit in hits]
        stable_first.append({"query_id": case["query_id"], "hits": hit_payload})
        acceptable = set(case["acceptable_scene_ids"])
        scene_ids = [hit.scene_id for hit in hits]
        query_results.append(
            {
                "query_id": case["query_id"],
                "category": case["category"],
                "query": case["text"],
                "query_spec": spec.model_dump(mode="json"),
                "acceptable_scene_ids": case["acceptable_scene_ids"],
                "hits": hit_payload,
                "latency_ms": latency_ms,
                "hit_at_1": bool(acceptable and scene_ids[:1] and scene_ids[0] in acceptable),
                "hit_at_3": bool(acceptable.intersection(scene_ids[:3])),
            }
        )
    stable_second = [
        {
            "query_id": case["query_id"],
            "hits": [
                hit.model_dump(mode="json")
                for hit in search(database_path, str(case["text"]), top_k=top_k)
            ],
        }
        for case in cases
    ]
    positives = [
        item for item in query_results if item["acceptable_scene_ids"]
    ]
    exact = [
        item for item in positives if item["category"] == "exact_dialogue"
    ]
    negatives = [
        item for item in query_results if item["category"] == "negative"
    ]
    negative_high_confidence_false_positives = sum(
        any(float(hit["score"]) >= 0.80 for hit in item["hits"])
        for item in negatives
    )
    recall_at_1 = sum(bool(item["hit_at_1"]) for item in positives) / len(positives)
    recall_at_3 = sum(bool(item["hit_at_3"]) for item in positives) / len(positives)
    exact_recall_at_1 = sum(bool(item["hit_at_1"]) for item in exact) / len(exact)
    deterministic = stable_first == stable_second
    stable_sha = hashlib.sha256(
        json.dumps(
            stable_first,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    metrics = {
        "query_count": len(cases),
        "executed_query_count": len(query_results),
        "positive_query_count": len(positives),
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "exact_dialogue_recall_at_1": exact_recall_at_1,
        "negative_high_confidence_false_positive_count": negative_high_confidence_false_positives,
        "query_p95_ms": _p95(latencies_ms),
        "query_max_ms": max(latencies_ms, default=0.0),
        "deterministic_replay": deterministic,
    }
    gates = {
        "all_15_queries_executed": len(query_results) == 15,
        "category_matrix_exact": dict(Counter(item["category"] for item in cases))
        == EXPECTED_CATEGORIES,
        "recall_at_1_at_least_0_60": recall_at_1 >= 0.60,
        "recall_at_3_at_least_0_80": recall_at_3 >= 0.80,
        "exact_dialogue_recall_at_1_is_1": exact_recall_at_1 == 1.0,
        "negative_high_confidence_false_positives_zero": negative_high_confidence_false_positives
        == 0,
        "query_p95_below_1000_ms": metrics["query_p95_ms"] < 1000.0,
        "deterministic_replay": deterministic,
    }
    return {
        "schema_version": "m1c-evaluation-v1",
        "category_counts": dict(Counter(item["category"] for item in cases)),
        "metrics": metrics,
        "gates": gates,
        "deterministic_results_sha256": stable_sha,
        "pass": all(gates.values()),
        "query_results": query_results,
    }
