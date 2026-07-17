"""M2 two-stage candidate recall with field-aware evidence scores."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from shotseek.planning.rules import expand_aliases
from shotseek.planning.schema import AnchorSpec, QuerySpecV2
from shotseek.verification.schema import CandidateScene, ScoreComponents

TOKEN_RE = re.compile(r"[a-z0-9']+|[\u4e00-\u9fff]+", re.IGNORECASE)
CANONICAL = {
    "indoors": "indoor",
    "outdoors": "outdoor",
    "looking": "look",
    "looks": "look",
    "speaking": "speak",
    "aiming": "aim",
    "aims": "aim",
    "reaching": "reach",
    "standing": "stand",
    "operating": "operate",
    "faces": "face",
    "facing": "face",
    "moving": "move",
    "moves": "move",
    "appears": "appear",
    "observes": "observe",
    "buildings": "building",
    "rooftops": "rooftop",
    "robots": "robot",
    "monitors": "monitor",
}


def normalized_tokens(value: str) -> list[str]:
    result: list[str] = []
    for token in TOKEN_RE.findall(expand_aliases(value)):
        canonical = CANONICAL.get(token.lower().strip("'"), token.lower().strip("'"))
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def _constraint_tokens(spec: QuerySpecV2 | AnchorSpec) -> dict[str, list[str]]:
    return {
        "entity": normalized_tokens(
            " ".join(item.text for item in spec.entities)
        ),
        "action": normalized_tokens(" ".join(spec.actions)),
        "object": normalized_tokens(" ".join(spec.objects)),
        "location": normalized_tokens(" ".join(spec.locations)),
        "keyword": normalized_tokens(" ".join(spec.keywords)),
    }


def _coverage(needles: list[str], haystack: set[str]) -> float:
    if not needles:
        return 1.0
    return sum(token in haystack for token in needles) / len(needles)


def _all_requested(tokens_by_field: dict[str, list[str]]) -> list[str]:
    result: list[str] = []
    for values in tokens_by_field.values():
        for value in values:
            if value not in result:
                result.append(value)
    return result


def _fts_rowids(connection: sqlite3.Connection, terms: list[str]) -> dict[int, float]:
    if not terms:
        return {}
    query = " OR ".join(f'"{term}"' for term in terms)
    rows = connection.execute(
        """
        SELECT rowid, bm25(
            scene_fts, 0.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0, 5.0
        ) AS rank
        FROM scene_fts
        WHERE scene_fts MATCH ?
        ORDER BY rank ASC, rowid ASC
        """,
        (query,),
    )
    return {int(row[0]): float(row[1]) for row in rows}


def retrieve_candidates(
    database_path: Path,
    spec: QuerySpecV2,
    *,
    limit: int = 20,
) -> tuple[list[CandidateScene], dict[str, Any]]:
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    requested = _constraint_tokens(spec)
    all_terms = _all_requested(requested)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        fts_rows = _fts_rowids(connection, all_terms)
        ranked_fts = sorted(
            fts_rows,
            key=lambda rowid: (fts_rows[rowid], rowid),
        )
        fts_quality = {
            rowid: 1.0 - (position / max(1, len(ranked_fts)))
            for position, rowid in enumerate(ranked_fts)
        }
        rows = list(connection.execute("SELECT rowid, * FROM scene ORDER BY start_ms, scene_id"))
    candidates: list[CandidateScene] = []
    route_counts = {
        "exact_dialogue": 0,
        "strict_and": 0,
        "relaxed_or": 0,
        "all_scenes": 0,
    }
    for row in rows:
        scene = json.loads(row["scene_json"])
        dialogue = str(row["dialogue"])
        dialogue_exact = (
            bool(spec.quoted_text)
            and spec.quoted_text.lower() in dialogue.lower()
        )
        if spec.quoted_text and not dialogue_exact:
            continue
        entity_haystack = set(
            normalized_tokens(" ".join(scene["characters"]) + " " + scene["summary"])
        )
        action_haystack = set(
            normalized_tokens(" ".join(scene["actions"]) + " " + scene["summary"])
        )
        object_haystack = set(
            normalized_tokens(" ".join(scene["objects"]) + " " + scene["summary"])
        )
        location_haystack = set(
            normalized_tokens((scene["location"] or "") + " " + scene["summary"])
        )
        visual_haystack = (
            entity_haystack
            | action_haystack
            | object_haystack
            | location_haystack
            | set(normalized_tokens(" ".join(scene["visible_text"])))
        )
        entity_score = _coverage(requested["entity"], entity_haystack)
        action_score = _coverage(requested["action"], action_haystack)
        object_score = _coverage(requested["object"], object_haystack)
        location_score = _coverage(requested["location"], location_haystack)
        keyword_score = _coverage(requested["keyword"], visual_haystack)
        requested_groups = [
            value
            for key, value in (
                ("entity", entity_score),
                ("action", action_score),
                ("object", object_score),
                ("location", location_score),
                ("keyword", keyword_score),
            )
            if requested[key]
        ]
        field_coverage = (
            sum(requested_groups) / len(requested_groups)
            if requested_groups
            else 1.0
        )
        lexical_score = (
            0.90 * field_coverage + 0.10 * fts_quality[int(row["rowid"])]
            if int(row["rowid"]) in fts_quality
            else field_coverage
        )
        visual_groups = [
            value
            for key, value in (
                ("action", action_score),
                ("object", object_score),
                ("location", location_score),
                ("keyword", keyword_score),
            )
            if requested[key]
        ]
        visual_score = (
            sum(visual_groups) / len(visual_groups)
            if visual_groups
            else entity_score if requested["entity"] else 1.0
        )
        matched_terms = sum(token in visual_haystack for token in all_terms)
        if dialogue_exact:
            route = "exact_dialogue"
        elif all_terms and matched_terms == len(all_terms):
            route = "strict_and"
        elif int(row["rowid"]) in fts_rows or matched_terms:
            route = "relaxed_or"
        elif not all_terms:
            route = "all_scenes"
        else:
            continue
        route_counts[route] += 1
        dialogue_score = 1.0 if dialogue_exact else 0.0
        retrieval_score = min(
            1.0,
            0.30 * lexical_score
            + 0.30 * visual_score
            + 0.20 * entity_score
            + 0.20 * dialogue_score,
        )
        components = ScoreComponents(
            lexical_score=lexical_score,
            dialogue_score=dialogue_score,
            visual_score=visual_score,
            entity_score=entity_score,
            temporal_score=1.0 if not spec.temporal_constraints else 0.0,
            evidence_coverage=0.0,
            boundary_quality=1.0 if scene["shot_ids"] else 0.0,
            contradiction_penalty=0.0,
        )
        candidates.append(
            CandidateScene(
                scene_id=scene["scene_id"],
                start_ms=scene["start_ms"],
                end_ms=scene["end_ms"],
                start_frame=scene["start_frame"],
                end_frame=scene["end_frame"],
                summary=scene["summary"],
                characters=scene["characters"],
                actions=scene["actions"],
                objects=scene["objects"],
                location=scene["location"],
                visible_text=scene["visible_text"],
                dialogue=dialogue,
                shot_ids=scene["shot_ids"],
                evidence_refs=scene["evidence_refs"],
                retrieval_route=route,
                retrieval_score=retrieval_score,
                components=components,
            )
        )
    candidates.sort(
        key=lambda item: (-item.retrieval_score, item.start_ms, item.scene_id)
    )
    return candidates[:limit], {
        "routes": [key for key, count in route_counts.items() if count],
        "route_counts": route_counts,
        "candidate_count": min(len(candidates), limit),
        "candidate_limit": limit,
    }
