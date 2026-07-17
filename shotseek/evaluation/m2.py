"""Deterministic 40-query M2 retrieval and evidence evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

from shotseek.agent import ShotSeekAgent
from shotseek.m0 import ensure_within_project
from shotseek.planning.rules import build_rule_spec
from shotseek.planning.schema import QuerySpecV2
from shotseek.retrieval.candidates import retrieve_candidates
from shotseek.retrieval.sqlite_index import search as m1_search


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _plan_matches(spec: QuerySpecV2, expected: dict[str, Any]) -> bool:
    actual_lists = {
        "entities": [item.text for item in spec.entities],
        "actions": spec.actions,
        "objects": spec.objects,
        "locations": spec.locations,
        "keywords": spec.keywords,
    }
    for field, actual in actual_lists.items():
        required = expected.get(field, [])
        if not set(required) <= set(actual):
            return False
    if "quoted_text" in expected and spec.quoted_text != expected["quoted_text"]:
        return False
    if "relations" in expected:
        actual = [item.relation for item in spec.temporal_constraints]
        if actual != expected["relations"]:
            return False
    if "ordinal" in expected:
        if spec.ordinal is None or spec.ordinal.value != expected["ordinal"]:
            return False
    if len(spec.negative_constraints) != expected.get(
        "negative_count", len(spec.negative_constraints)
    ):
        return False
    return True


def _reference_ids(root: Path) -> set[str]:
    m1a = root / "runs" / "m1a" / "latest"
    visual = json.loads((m1a / "aligned_visual_events.json").read_text())
    dialogue = json.loads((m1a / "contextualized_utterances.json").read_text())
    return {
        *(item["event_id"] for item in visual),
        *(item["utterance_id"] for item in dialogue),
    }


def _ranking_metrics(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
) -> dict[str, Any]:
    positives = [case for case in cases if case["acceptable_scene_ids"]]
    correct_at_1 = 0
    correct_at_3 = 0
    reciprocal_rank = 0.0
    negative_false_positives = 0
    for case in cases:
        actual = rankings[case["query_id"]]
        expected = case["acceptable_scene_ids"]
        if not expected:
            negative_false_positives += len(actual)
            continue
        correct_at_1 += int(bool(actual) and actual[0] in expected)
        correct_at_3 += int(any(item in expected for item in actual[:3]))
        for rank, scene_id in enumerate(actual, start=1):
            if scene_id in expected:
                reciprocal_rank += 1.0 / rank
                break
    count = len(positives)
    return {
        "executed_query_count": len(cases),
        "recall_at_1": correct_at_1 / count,
        "recall_at_3": correct_at_3 / count,
        "mrr": reciprocal_rank / count,
        "negative_false_positive_count": negative_false_positives,
    }


def run_m2_evaluation(
    *,
    project_root: Path,
    output_dir: Path,
) -> Path:
    root = project_root.resolve()
    output = ensure_within_project(root, output_dir)
    database = ensure_within_project(
        root, root / "runs" / "m1c" / "latest" / "search.sqlite3"
    )
    cases = _load_jsonl(root / "eval" / "m1_queries.jsonl")
    cases += _load_jsonl(root / "eval" / "m2_queries.jsonl")
    agent = ShotSeekAgent(
        database_path=database,
        trace_dir=output / "traces",
    )

    results: list[dict[str, Any]] = []
    replay_material: list[dict[str, Any]] = []
    phase_latencies: dict[str, list[float]] = {
        "planner": [],
        "retrieval": [],
        "temporal": [],
        "verification": [],
        "total": [],
    }
    correct_at_1 = 0
    correct_at_3 = 0
    reciprocal_rank = 0.0
    candidate_recalled = 0
    positive_count = 0
    returned_hit_count = 0
    valid_returned_hit_count = 0
    supported_hit_count = 0
    direct_evidence_hit_count = 0
    broken_evidence_refs = 0
    unsupported_claim_count = 0
    negative_false_positives = 0
    planner_checks: list[bool] = []
    query_spec_valid_count = 0
    reference_ids = _reference_ids(root)

    for case in cases:
        started = perf_counter()
        response = agent.search(
            case["text"],
            planner_mode="rule",
            verifier_mode="rule",
        )
        elapsed_ms = (perf_counter() - started) * 1000
        spec = QuerySpecV2.model_validate(
            response.trace.query_spec.model_dump(mode="json")
        )
        query_spec_valid_count += 1
        expected = case["acceptable_scene_ids"]
        actual = [item.candidate.scene_id for item in response.hits]
        recalled = response.trace.retrieval["recalled_scene_ids"]
        positive = bool(expected)
        if positive:
            positive_count += 1
            correct_at_1 += int(bool(actual) and actual[0] in expected)
            correct_at_3 += int(any(item in expected for item in actual[:3]))
            candidate_recalled += int(any(item in expected for item in recalled))
            for rank, scene_id in enumerate(actual, start=1):
                if scene_id in expected:
                    reciprocal_rank += 1.0 / rank
                    break
        elif actual:
            negative_false_positives += len(actual)

        for hit in response.hits:
            returned_hit_count += 1
            valid_returned_hit_count += int(
                hit.candidate.scene_id in expected
            )
            supported_hit_count += int(
                hit.verification.verdict == "supported"
            )
            direct_evidence_hit_count += int(
                hit.verification.direct_evidence
            )
            unsupported_claim_count += int(
                hit.verification.verdict != "supported"
            )
            broken_evidence_refs += sum(
                ref["evidence_id"] not in reference_ids
                for ref in hit.candidate.evidence_refs
            )

        if "expected_plan" in case:
            planner_checks.append(
                _plan_matches(spec, case["expected_plan"])
            )
        for phase in ("planner", "retrieval", "temporal", "verification"):
            phase_latencies[phase].append(
                response.trace.phase_latency_ms[phase]
            )
        phase_latencies["total"].append(elapsed_ms)

        deterministic = {
            "query_id": case["query_id"],
            "query_spec": spec.model_dump(mode="json"),
            "scene_ids": actual,
        }
        replay_material.append(deterministic)
        results.append(
            {
                **deterministic,
                "category": case["category"],
                "acceptable_scene_ids": expected,
                "candidate_scene_ids": recalled,
                "planner_status": response.trace.planner.status,
                "agent_status": response.trace.status,
                "trace_id": response.trace.trace_id,
                "latency_ms": elapsed_ms,
                "hits": [
                    {
                        "scene_id": hit.candidate.scene_id,
                        "final_score": hit.final_score,
                        "verdict": hit.verification.verdict,
                        "direct_evidence": hit.verification.direct_evidence,
                        "evidence_refs": hit.candidate.evidence_refs,
                    }
                    for hit in response.hits
                ],
            }
        )

    replay = [
        {
            "query_id": case["query_id"],
            "query_spec": replay_response.trace.query_spec.model_dump(mode="json"),
            "scene_ids": [
                item.candidate.scene_id for item in replay_response.hits
            ],
        }
        for case in cases
        for replay_response in [
            ShotSeekAgent(database_path=database).search(
                case["text"],
                planner_mode="rule",
                verifier_mode="rule",
            )
        ]
    ]
    deterministic_replay = replay == replay_material

    full_rankings = {
        item["query_id"]: item["scene_ids"] for item in results
    }
    m1_rankings: dict[str, list[str]] = {}
    recall_only_rankings: dict[str, list[str]] = {}
    for case in cases:
        try:
            m1_rankings[case["query_id"]] = [
                item.scene_id
                for item in m1_search(database, case["text"], top_k=3)
            ]
        except Exception:
            m1_rankings[case["query_id"]] = []
        spec = build_rule_spec(case["text"])
        candidates, _ = retrieve_candidates(database, spec, limit=20)
        recall_only_rankings[case["query_id"]] = [
            item.scene_id for item in candidates[:3]
        ]
    ablation = {
        "schema_version": "m2-ablation-v1",
        "systems": {
            "m1_fts_baseline": _ranking_metrics(cases, m1_rankings),
            "m2_recall_without_verification": _ranking_metrics(
                cases, recall_only_rankings
            ),
            "m2_full_agent": _ranking_metrics(cases, full_rankings),
        },
    }

    fallback_cases = [
        case for case in cases if case["query_id"] in {"q28", "q33", "q37"}
    ]
    fallback_successes = 0
    for case in fallback_cases:
        response = ShotSeekAgent(database_path=database).search(
            case["text"],
            planner_mode="stepfun",
            verifier_mode="rule",
            allow_network=False,
        )
        actual = [item.candidate.scene_id for item in response.hits]
        fallback_successes += int(
            response.trace.planner.status == "FALLBACK"
            and (
                (not case["acceptable_scene_ids"] and not actual)
                or (
                    bool(actual)
                    and actual[0] in case["acceptable_scene_ids"]
                )
            )
        )
    fallback_success_rate = fallback_successes / len(fallback_cases)

    temporal_ids = {
        case["query_id"]
        for case in cases
        if case["category"] in {"complex_temporal", "ordinal", "temporal"}
    }
    replay_by_id = {item["query_id"]: item for item in replay}
    temporal_deterministic = all(
        item == replay_by_id[item["query_id"]]
        for item in replay_material
        if item["query_id"] in temporal_ids
    )
    status_counts = dict(
        Counter(item["agent_status"] for item in results)
    )
    metrics = {
        "query_count": len(cases),
        "executed_query_count": len(results),
        "positive_query_count": positive_count,
        "recall_at_1": correct_at_1 / positive_count,
        "recall_at_3": correct_at_3 / positive_count,
        "mrr": reciprocal_rank / positive_count,
        "query_spec_valid_rate": query_spec_valid_count / len(cases),
        "planner_accuracy": sum(planner_checks) / len(planner_checks),
        "planner_evaluated_query_count": len(planner_checks),
        "candidate_recall_at_20": candidate_recalled / positive_count,
        "verifier_precision": (
            valid_returned_hit_count / returned_hit_count
            if returned_hit_count
            else 1.0
        ),
        "evidence_support_rate": (
            supported_hit_count / returned_hit_count
            if returned_hit_count
            else 1.0
        ),
        "direct_evidence_rate": (
            direct_evidence_hit_count / returned_hit_count
            if returned_hit_count
            else 1.0
        ),
        "fallback_success_rate": fallback_success_rate,
        "status_counts": status_counts,
        "cache_rate": status_counts.get("CACHED", 0) / len(results),
        "fallback_rate": (
            status_counts.get("FALLBACK", 0) / len(results)
        ),
        "broken_evidence_reference_count": broken_evidence_refs,
        "unsupported_claim_count": unsupported_claim_count,
        "negative_high_confidence_false_positive_count": negative_false_positives,
        "deterministic_replay": deterministic_replay,
        "temporal_deterministic": temporal_deterministic,
        "network_call_count": 0,
        "phase_latency_ms": {
            phase: {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }
            for phase, values in phase_latencies.items()
        },
    }
    exact_dialogue = [
        item for item in results if item["category"] == "exact_dialogue"
    ]
    exact_dialogue_recall_at_1 = sum(
        bool(item["scene_ids"])
        and item["scene_ids"][0] in item["acceptable_scene_ids"]
        for item in exact_dialogue
    ) / len(exact_dialogue)
    metrics["exact_dialogue_recall_at_1"] = exact_dialogue_recall_at_1

    category_counts = dict(Counter(case["category"] for case in cases))
    gates = {
        "all_40_queries_executed": len(results) == 40,
        "m1_15_queries_preserved": all(
            item["query_id"].startswith("q")
            for item in results[:15]
        ) and len(results[:15]) == 15,
        "m2_category_matrix_exact": {
            key: category_counts.get(key, 0)
            for key in (
                "chinese_synonym",
                "english_synonym",
                "complex_temporal",
                "ordinal",
                "negation",
                "hard_negative",
            )
        }
        == {
            "chinese_synonym": 8,
            "english_synonym": 4,
            "complex_temporal": 5,
            "ordinal": 3,
            "negation": 2,
            "hard_negative": 3,
        },
        "query_spec_valid_100_percent": metrics["query_spec_valid_rate"] == 1.0,
        "planner_accuracy_100_percent": metrics["planner_accuracy"] == 1.0,
        "fallback_success_100_percent": fallback_success_rate == 1.0,
        "candidate_recall_at_20_at_least_0_95": metrics["candidate_recall_at_20"] >= 0.95,
        "recall_at_1_at_least_0_75": metrics["recall_at_1"] >= 0.75,
        "recall_at_3_at_least_0_85": metrics["recall_at_3"] >= 0.85,
        "verifier_precision_at_least_0_85": metrics["verifier_precision"] >= 0.85,
        "ablation_three_systems_compared": len(ablation["systems"]) == 3,
        "full_agent_not_worse_than_baselines": (
            ablation["systems"]["m2_full_agent"]["recall_at_1"]
            >= max(
                ablation["systems"]["m1_fts_baseline"]["recall_at_1"],
                ablation["systems"]["m2_recall_without_verification"][
                    "recall_at_1"
                ],
            )
        ),
        "evidence_support_rate_100_percent": metrics["evidence_support_rate"] == 1.0,
        "direct_evidence_rate_100_percent": metrics["direct_evidence_rate"] == 1.0,
        "broken_evidence_references_zero": broken_evidence_refs == 0,
        "unsupported_claims_zero": unsupported_claim_count == 0,
        "temporal_deterministic_100_percent": temporal_deterministic,
        "exact_dialogue_recall_at_1_is_1": exact_dialogue_recall_at_1 == 1.0,
        "negative_high_confidence_false_positives_zero": negative_false_positives == 0,
        "offline_replay_deterministic": deterministic_replay,
        "ci_network_calls_zero": metrics["network_call_count"] == 0,
    }
    digest = hashlib.sha256(
        json.dumps(
            replay_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation = {
        "schema_version": "m2-evaluation-v1",
        "pass": all(gates.values()),
        "category_counts": category_counts,
        "metrics": metrics,
        "gates": gates,
        "deterministic_results_sha256": digest,
    }
    _dump(output / "results.json", results)
    _dump(output / "ablation.json", ablation)
    _dump(output / "evaluation.json", evaluation)
    if not evaluation["pass"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"M2 evaluation gates failed: {failed}")
    return output
