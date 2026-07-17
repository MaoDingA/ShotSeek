"""SQLite-backed idempotent Runtime Registry and state machine."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shotseek.runtime.schema import (
    ALLOWED_TRANSITIONS,
    STAGE_STATES,
    TERMINAL_STATES,
    ArtifactRecord,
    JobEvent,
    JobRecord,
    JobState,
    VideoRecord,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _video_id(sha256: str) -> str:
    return f"video_{sha256[:16]}"


def _job_id() -> str:
    return f"job_{uuid.uuid4().hex[:20]}"


def _artifact_id() -> str:
    return f"artifact_{uuid.uuid4().hex[:20]}"


class RuntimeRegistry:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS video (
                    video_id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    proxy_path TEXT,
                    audio_path TEXT,
                    duration_ms INTEGER,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    bytes INTEGER NOT NULL,
                    scene_count INTEGER NOT NULL DEFAULT 0,
                    search_db_path TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job (
                    job_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES video(video_id),
                    state TEXT NOT NULL,
                    progress REAL NOT NULL,
                    current_stage TEXT NOT NULL,
                    completed_units INTEGER NOT NULL,
                    total_units INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    cancel_requested INTEGER NOT NULL,
                    error_code TEXT,
                    message TEXT NOT NULL,
                    resume_state TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_state_idx
                    ON job(state, created_at);
                CREATE TABLE IF NOT EXISTS job_event (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES job(job_id),
                    state TEXT NOT NULL,
                    progress REAL NOT NULL,
                    completed_units INTEGER NOT NULL,
                    total_units INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifact (
                    artifact_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES video(video_id),
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    prompt_version TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(video_id, kind, cache_key)
                );
                """
            )

    @staticmethod
    def _video(row: sqlite3.Row) -> VideoRecord:
        return VideoRecord.model_validate(dict(row))

    @staticmethod
    def _job(row: sqlite3.Row) -> JobRecord:
        payload = dict(row)
        payload["cancel_requested"] = bool(payload["cancel_requested"])
        return JobRecord.model_validate(payload)

    def register_video(
        self,
        *,
        sha256: str,
        original_filename: str,
        source_path: str,
        bytes: int,
    ) -> tuple[VideoRecord, bool]:
        now = _now()
        video_id = _video_id(sha256)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM video WHERE sha256=?", (sha256,)
            ).fetchone()
            if existing is not None:
                return self._video(existing), False
            connection.execute(
                """
                INSERT INTO video(
                    video_id, sha256, original_filename, source_path, bytes,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'REGISTERED', ?, ?)
                """,
                (
                    video_id,
                    sha256,
                    original_filename,
                    source_path,
                    bytes,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM video WHERE video_id=?", (video_id,)
            ).fetchone()
            return self._video(row), True

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM video WHERE video_id=?", (video_id,)
            ).fetchone()
        return self._video(row) if row is not None else None

    def list_videos(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM video ORDER BY created_at DESC, video_id"
            ).fetchall()
        return [self._video(row) for row in rows]

    def update_video(self, video_id: str, **updates: Any) -> VideoRecord:
        allowed = {
            "proxy_path",
            "audio_path",
            "duration_ms",
            "width",
            "height",
            "fps",
            "scene_count",
            "search_db_path",
            "status",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unsupported video fields: {sorted(unknown)}")
        if not updates:
            record = self.get_video(video_id)
            if record is None:
                raise KeyError(video_id)
            return record
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE video SET {assignments} WHERE video_id=?",
                [*updates.values(), video_id],
            )
            if cursor.rowcount != 1:
                raise KeyError(video_id)
            row = connection.execute(
                "SELECT * FROM video WHERE video_id=?", (video_id,)
            ).fetchone()
        return self._video(row)

    def create_job(self, video_id: str) -> JobRecord:
        if self.get_video(video_id) is None:
            raise KeyError(video_id)
        now = _now()
        job_id = _job_id()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job(
                    job_id, video_id, state, progress, current_stage,
                    completed_units, total_units, retry_count, cancel_requested,
                    error_code, message, resume_state, created_at, updated_at
                ) VALUES (?, ?, 'CREATED', 0, 'created', 0, 0, 0, 0,
                          NULL, '任务已创建', NULL, ?, ?)
                """,
                (job_id, video_id, now, now),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                state=JobState.CREATED,
                progress=0.0,
                completed_units=0,
                total_units=0,
                message="任务已创建",
                created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM job WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._job(row)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM job WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._job(row) if row is not None else None

    def list_jobs(self) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM job ORDER BY created_at DESC, job_id"
            ).fetchall()
        return [self._job(row) for row in rows]

    def latest_job_for_video(self, video_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM job WHERE video_id=?
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """,
                (video_id,),
            ).fetchone()
        return self._job(row) if row is not None else None

    def active_job_for_video(self, video_id: str) -> JobRecord | None:
        placeholders = ",".join("?" for _ in TERMINAL_STATES)
        terminal = [state.value for state in TERMINAL_STATES]
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM job
                WHERE video_id=? AND state NOT IN ({placeholders})
                ORDER BY created_at DESC, job_id DESC
                LIMIT 1
                """,
                [video_id, *terminal],
            ).fetchone()
        return self._job(row) if row is not None else None

    def claim_next_job(self) -> JobRecord | None:
        """Atomically claim the oldest queued job for the single media worker."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM job
                WHERE state='QUEUED'
                ORDER BY created_at, job_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            current = self._job(row)
            if current.cancel_requested:
                next_state = JobState.CANCELLED
                message = "任务已取消"
            else:
                next_state = current.resume_state or JobState.PROBING
                if next_state not in STAGE_STATES:
                    next_state = JobState.PROBING
                message = f"开始 {next_state.value}"
            now = _now()
            connection.execute(
                """
                UPDATE job SET state=?, current_stage=?, message=?,
                    resume_state=NULL, updated_at=?
                WHERE job_id=? AND state='QUEUED'
                """,
                (
                    next_state.value,
                    next_state.value.lower(),
                    message,
                    now,
                    current.job_id,
                ),
            )
            self._insert_event(
                connection,
                job_id=current.job_id,
                state=next_state,
                progress=current.progress,
                completed_units=current.completed_units,
                total_units=current.total_units,
                message=message,
                created_at=now,
            )
            updated = connection.execute(
                "SELECT * FROM job WHERE job_id=?", (current.job_id,)
            ).fetchone()
        return self._job(updated)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        state: JobState,
        progress: float,
        completed_units: int,
        total_units: int,
        message: str,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO job_event(
                job_id, state, progress, completed_units, total_units,
                message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                state.value,
                progress,
                completed_units,
                total_units,
                message,
                created_at,
            ),
        )

    def transition(
        self,
        job_id: str,
        state: JobState,
        *,
        progress: float | None = None,
        current_stage: str | None = None,
        completed_units: int | None = None,
        total_units: int | None = None,
        message: str = "",
        error_code: str | None = None,
        resume_state: JobState | None = None,
        increment_retry: bool = False,
        force: bool = False,
    ) -> JobRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM job WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = self._job(row)
            if not force and state not in ALLOWED_TRANSITIONS[current.state]:
                raise ValueError(
                    f"invalid job transition: {current.state.value} -> {state.value}"
                )
            now = _now()
            next_progress = (
                1.0 if state == JobState.READY
                else current.progress if progress is None
                else progress
            )
            values = {
                "state": state.value,
                "progress": next_progress,
                "current_stage": current_stage or state.value.lower(),
                "completed_units": (
                    current.completed_units
                    if completed_units is None
                    else completed_units
                ),
                "total_units": current.total_units if total_units is None else total_units,
                "retry_count": current.retry_count + int(increment_retry),
                "error_code": error_code,
                "message": message or state.value,
                "resume_state": resume_state.value if resume_state else None,
                "updated_at": now,
            }
            connection.execute(
                """
                UPDATE job SET
                    state=:state, progress=:progress,
                    current_stage=:current_stage,
                    completed_units=:completed_units,
                    total_units=:total_units,
                    retry_count=:retry_count,
                    error_code=:error_code, message=:message,
                    resume_state=:resume_state, updated_at=:updated_at
                WHERE job_id=:job_id
                """,
                {**values, "job_id": job_id},
            )
            self._insert_event(
                connection,
                job_id=job_id,
                state=state,
                progress=next_progress,
                completed_units=values["completed_units"],
                total_units=values["total_units"],
                message=values["message"],
                created_at=now,
            )
            updated = connection.execute(
                "SELECT * FROM job WHERE job_id=?", (job_id,)
            ).fetchone()
        return self._job(updated)

    def update_progress(
        self,
        job_id: str,
        *,
        completed_units: int,
        total_units: int,
        progress: float,
        message: str,
    ) -> JobRecord:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(job_id)
        return self.transition(
            job_id,
            current.state,
            progress=progress,
            current_stage=current.current_stage,
            completed_units=completed_units,
            total_units=total_units,
            message=message,
            force=True,
        )

    def request_cancel(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE job SET cancel_requested=1, updated_at=? WHERE job_id=?",
                (_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state in TERMINAL_STATES:
            return job
        return job

    def cancel_if_requested(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if not job.cancel_requested or job.state in TERMINAL_STATES:
            return False
        self.transition(
            job_id,
            JobState.CANCELLED,
            message="任务已取消",
            force=True,
        )
        return True

    def events(self, job_id: str, *, after: int = 0) -> list[JobEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM job_event
                WHERE job_id=? AND event_id>?
                ORDER BY event_id
                """,
                (job_id, after),
            ).fetchall()
        return [JobEvent.model_validate(dict(row)) for row in rows]

    def recover_incomplete_jobs(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        for job in self.list_jobs():
            if job.state in TERMINAL_STATES or job.state == JobState.QUEUED:
                continue
            recovered.append(
                self.transition(
                    job.job_id,
                    JobState.QUEUED,
                    progress=job.progress,
                    current_stage="queued",
                    completed_units=job.completed_units,
                    total_units=job.total_units,
                    message=f"服务重启，等待从 {job.state.value} 恢复",
                    resume_state=job.state if job.state in STAGE_STATES else job.resume_state,
                    force=True,
                )
            )
        return recovered

    def add_artifact(
        self,
        *,
        video_id: str,
        kind: str,
        path: str,
        cache_key: str,
        schema_version: str,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> ArtifactRecord:
        existing = self.find_artifact(video_id, kind, cache_key)
        if existing is not None:
            return existing
        artifact_id = _artifact_id()
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifact(
                    artifact_id, video_id, kind, path, cache_key,
                    schema_version, provider, model, prompt_version,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    video_id,
                    kind,
                    path,
                    cache_key,
                    schema_version,
                    provider,
                    model,
                    prompt_version,
                    status,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM artifact WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        return ArtifactRecord.model_validate(dict(row))

    def find_artifact(
        self, video_id: str, kind: str, cache_key: str
    ) -> ArtifactRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM artifact
                WHERE video_id=? AND kind=? AND cache_key=?
                """,
                (video_id, kind, cache_key),
            ).fetchone()
        return ArtifactRecord.model_validate(dict(row)) if row else None

    def list_artifacts(self, video_id: str) -> list[ArtifactRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM artifact
                WHERE video_id=?
                ORDER BY created_at, artifact_id
                """,
                (video_id,),
            ).fetchall()
        return [ArtifactRecord.model_validate(dict(row)) for row in rows]

    def diagnostics(self) -> dict[str, Any]:
        with self._connect() as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            counts = {
                table: int(
                    connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                )
                for table in ("video", "job", "job_event", "artifact")
            }
        return {
            "schema_version": "runtime-registry-v1",
            "integrity_check": integrity,
            "counts": counts,
            "database_path": str(self.database_path),
        }
