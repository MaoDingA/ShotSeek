"""Strict Production Runtime contracts for jobs, videos and artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class JobState(StrEnum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PROBING = "PROBING"
    TRANSCODING = "TRANSCODING"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    DETECTING_SHOTS = "DETECTING_SHOTS"
    CHUNKING = "CHUNKING"
    ANALYZING_VISUAL = "ANALYZING_VISUAL"
    ANALYZING_ASR = "ANALYZING_ASR"
    ALIGNING = "ALIGNING"
    BUILDING_SCENES = "BUILDING_SCENES"
    INDEXING = "INDEXING"
    RETRYING = "RETRYING"
    READY = "READY"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {JobState.READY, JobState.PARTIAL, JobState.FAILED, JobState.CANCELLED}
)

STAGE_STATES = (
    JobState.PROBING,
    JobState.TRANSCODING,
    JobState.EXTRACTING_AUDIO,
    JobState.DETECTING_SHOTS,
    JobState.CHUNKING,
    JobState.ANALYZING_VISUAL,
    JobState.ANALYZING_ASR,
    JobState.ALIGNING,
    JobState.BUILDING_SCENES,
    JobState.INDEXING,
)

ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.CREATED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.QUEUED: frozenset({JobState.PROBING, JobState.CANCELLED}),
    JobState.PROBING: frozenset(
        {JobState.TRANSCODING, JobState.RETRYING, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.TRANSCODING: frozenset(
        {
            JobState.EXTRACTING_AUDIO,
            JobState.RETRYING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.EXTRACTING_AUDIO: frozenset(
        {
            JobState.DETECTING_SHOTS,
            JobState.RETRYING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.DETECTING_SHOTS: frozenset(
        {JobState.CHUNKING, JobState.RETRYING, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.CHUNKING: frozenset(
        {
            JobState.ANALYZING_VISUAL,
            JobState.RETRYING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.ANALYZING_VISUAL: frozenset(
        {
            JobState.ANALYZING_ASR,
            JobState.RETRYING,
            JobState.PARTIAL,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.ANALYZING_ASR: frozenset(
        {
            JobState.ALIGNING,
            JobState.RETRYING,
            JobState.PARTIAL,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.ALIGNING: frozenset(
        {
            JobState.BUILDING_SCENES,
            JobState.RETRYING,
            JobState.FAILED,
            JobState.CANCELLED,
        }
    ),
    JobState.BUILDING_SCENES: frozenset(
        {JobState.INDEXING, JobState.RETRYING, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.INDEXING: frozenset(
        {JobState.READY, JobState.RETRYING, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.RETRYING: frozenset(
        {*STAGE_STATES, JobState.PARTIAL, JobState.FAILED, JobState.CANCELLED}
    ),
    JobState.READY: frozenset(),
    JobState.PARTIAL: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.FAILED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.CANCELLED: frozenset({JobState.QUEUED}),
}


class VideoRecord(RuntimeModel):
    video_id: str = Field(pattern=r"^video_[a-f0-9]{16}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_filename: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    proxy_path: str | None = None
    audio_path: str | None = None
    duration_ms: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    bytes: int = Field(gt=0)
    scene_count: int = Field(default=0, ge=0)
    search_db_path: str | None = None
    status: Literal["REGISTERED", "PROCESSING", "READY", "PARTIAL", "FAILED"]
    created_at: str
    updated_at: str


class JobRecord(RuntimeModel):
    job_id: str = Field(pattern=r"^job_[a-f0-9]{20}$")
    video_id: str = Field(pattern=r"^video_[a-f0-9]{16}$")
    state: JobState
    progress: float = Field(ge=0.0, le=1.0)
    current_stage: str
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    cancel_requested: bool
    error_code: str | None
    message: str
    resume_state: JobState | None
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def validate_progress(self) -> "JobRecord":
        if self.completed_units > self.total_units and self.total_units:
            raise ValueError("completed_units cannot exceed total_units")
        if self.state == JobState.READY and self.progress != 1.0:
            raise ValueError("READY jobs must have progress=1")
        return self


class JobEvent(RuntimeModel):
    event_id: int = Field(gt=0)
    job_id: str
    state: JobState
    progress: float = Field(ge=0.0, le=1.0)
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    message: str
    created_at: str


class ArtifactRecord(RuntimeModel):
    artifact_id: str = Field(pattern=r"^artifact_[a-f0-9]{20}$")
    video_id: str
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    cache_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    schema_version: str = Field(min_length=1)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    status: Literal["LIVE", "CACHED", "GENERATED", "PARTIAL", "FAILED"]
    created_at: str
