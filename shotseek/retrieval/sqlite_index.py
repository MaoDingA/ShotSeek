"""SQLite FTS5 index and stable M1 ranking."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from shotseek.media.schema import ContextualizedUtterance
from shotseek.retrieval.query_rules import plan_query
from shotseek.retrieval.schema import QuerySpec, SearchHit
from shotseek.scenes.schema import Scene


def _joined(values: list[str]) -> str:
    return " ".join(value.strip() for value in values if value.strip())


def build_index(
    database_path: Path,
    scenes: list[Scene],
    utterances: list[ContextualizedUtterance],
) -> dict[str, int | bool]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    utterance_by_id = {item.utterance_id: item for item in utterances}
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE scene (
                scene_id TEXT PRIMARY KEY,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                summary TEXT NOT NULL,
                dialogue TEXT NOT NULL,
                scene_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE scene_fts USING fts5(
                scene_id UNINDEXED,
                summary,
                characters,
                actions,
                objects,
                location,
                visible_text,
                dialogue,
                tokenize='porter unicode61'
            );
            """
        )
        for scene in scenes:
            dialogue = _joined(
                [
                    utterance_by_id[utterance_id].text
                    for utterance_id in scene.utterance_ids
                ]
            )
            scene_json = json.dumps(
                scene.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            cursor = connection.execute(
                """
                INSERT INTO scene(
                    scene_id, start_ms, end_ms, start_frame, end_frame,
                    summary, dialogue, scene_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene.scene_id,
                    scene.start_ms,
                    scene.end_ms,
                    scene.start_frame,
                    scene.end_frame,
                    scene.summary,
                    dialogue,
                    scene_json,
                ),
            )
            rowid = cursor.lastrowid
            connection.execute(
                """
                INSERT INTO scene_fts(
                    rowid, scene_id, summary, characters, actions, objects,
                    location, visible_text, dialogue
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    scene.scene_id,
                    scene.summary,
                    _joined(scene.characters),
                    _joined(scene.actions),
                    _joined(scene.objects),
                    scene.location or "",
                    _joined(scene.visible_text),
                    dialogue,
                ),
            )
        connection.execute("INSERT INTO scene_fts(scene_fts) VALUES('optimize')")
        connection.commit()
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        fts_count = connection.execute(
            "SELECT count(*) FROM scene_fts"
        ).fetchone()[0]
    return {
        "scene_count": len(scenes),
        "fts_row_count": int(fts_count),
        "integrity_check_pass": integrity == "ok",
    }


def _fts_query(terms: list[str]) -> str:
    return " AND ".join(f'"{term}"' for term in terms)


def _candidate_rows(
    connection: sqlite3.Connection,
    spec: QuerySpec,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    parameters: list[Any] = []
    use_fts = bool(spec.terms)
    if use_fts:
        clauses.append("scene_fts MATCH ?")
        parameters.append(_fts_query(spec.terms))
    if spec.quoted_text:
        clauses.append("instr(lower(scene.dialogue), lower(?)) > 0")
        parameters.append(spec.quoted_text)
    if not clauses:
        return []
    rank_expression = (
        "bm25(scene_fts, 0.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0, 5.0)"
        if use_fts
        else "0.0"
    )
    join = (
        "JOIN scene_fts ON scene_fts.rowid = scene.rowid" if use_fts else ""
    )
    sql = f"""
        SELECT scene.*, {rank_expression} AS fts_rank
        FROM scene
        {join}
        WHERE {" AND ".join(clauses)}
        ORDER BY fts_rank ASC, scene.start_ms ASC, scene.scene_id ASC
    """
    return list(connection.execute(sql, parameters))


def search(
    database_path: Path,
    query: str | QuerySpec,
    *,
    top_k: int = 3,
) -> list[SearchHit]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    spec = query if isinstance(query, QuerySpec) else plan_query(query)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = _candidate_rows(connection, spec)
        anchor_end_ms: int | None = None
        if spec.temporal_relation == "after":
            anchor_spec = QuerySpec(
                raw_query=" ".join(spec.anchor_terms),
                normalized_query=" ".join(spec.anchor_terms),
                terms=spec.anchor_terms,
            )
            anchors = _candidate_rows(connection, anchor_spec)
            if not anchors:
                return []
            anchor_end_ms = int(anchors[0]["end_ms"])
            rows = [
                row for row in rows if int(row["start_ms"]) >= anchor_end_ms
            ]
        if spec.ordinal is not None:
            rows.sort(key=lambda row: (int(row["start_ms"]), str(row["scene_id"])))

    if spec.temporal_relation or spec.ordinal:
        match_type = "temporal"
        score = 0.90
    elif spec.quoted_text and spec.terms:
        match_type = "multimodal"
        score = 0.95
    elif spec.quoted_text:
        match_type = "dialogue"
        score = 1.0
    else:
        match_type = "visual"
        score = 0.85
    hits: list[SearchHit] = []
    for row in rows[:top_k]:
        scene = json.loads(row["scene_json"])
        hits.append(
            SearchHit(
                scene_id=row["scene_id"],
                start_ms=row["start_ms"],
                end_ms=row["end_ms"],
                start_frame=row["start_frame"],
                end_frame=row["end_frame"],
                summary=row["summary"],
                score=score,
                match_type=match_type,
                evidence_refs=scene["evidence_refs"],
            )
        )
    return hits
