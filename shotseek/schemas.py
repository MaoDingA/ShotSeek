"""Frozen M0 schemas for provider evidence and the unified timeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class UploadedFile(FrozenModel):
    provider: str = "stepfun"
    file_id: str = Field(min_length=1)
    file_uri: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(min_length=1)


class VisualEvent(FrozenModel):
    event_id: str = Field(min_length=1)
    approx_start_ms: int = Field(ge=0)
    approx_end_ms: int = Field(gt=0)
    summary: str = Field(min_length=1)
    characters: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    location: str | None = None
    visible_text: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "stepfun_vision"
    model: str = Field(min_length=1)
    chunk_id: str = "chunk_000"

    @model_validator(mode="after")
    def validate_time_range(self) -> "VisualEvent":
        if self.approx_end_ms <= self.approx_start_ms:
            raise ValueError("approx_end_ms must be greater than approx_start_ms")
        return self


class WordTimestamp(FrozenModel):
    text: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_time_range(self) -> "WordTimestamp":
        if self.end_ms <= self.start_ms:
            raise ValueError("word end_ms must be greater than start_ms")
        return self


class Utterance(FrozenModel):
    utterance_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    speaker_id: str | None = None
    words: list[WordTimestamp] = Field(default_factory=list)
    source: str = "stepfun_asr"

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("utterance text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "Utterance":
        if self.end_ms <= self.start_ms:
            raise ValueError("utterance end_ms must be greater than start_ms")
        for word in self.words:
            if word.start_ms < self.start_ms or word.end_ms > self.end_ms:
                raise ValueError("word timestamps must remain inside the utterance")
        return self


class EvidenceKind(str, Enum):
    VISUAL = "visual"
    DIALOGUE = "dialogue"


class BoundaryStatus(str, Enum):
    APPROXIMATE = "approximate"
    ASR_TIMESTAMP = "asr_timestamp"


class EvidenceSpan(FrozenModel):
    evidence_id: str = Field(min_length=1)
    kind: EvidenceKind
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_ref: str = Field(min_length=1)
    boundary_status: BoundaryStatus

    @model_validator(mode="after")
    def validate_time_range(self) -> "EvidenceSpan":
        if self.end_ms <= self.start_ms:
            raise ValueError("evidence end_ms must be greater than start_ms")
        return self


class VideoInfo(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=1)
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    video_codec: str = Field(min_length=1)
    audio_codec: str | None = None
    audio_channels: int | None = Field(default=None, gt=0)


class RunManifest(FrozenModel):
    run_id: str = Field(min_length=1)
    mode: str = Field(pattern=r"^(live|fixture)$")
    created_at: str = Field(min_length=1)
    video: VideoInfo
    models: dict[str, str]
    versions: dict[str, str]
    inputs: dict[str, Any]


class RunReport(FrozenModel):
    run_id: str = Field(min_length=1)
    mode: str = Field(pattern=r"^(live|fixture)$")
    status: str = Field(pattern=r"^(pass|partial|failed)$")
    video: dict[str, Any]
    models: dict[str, str]
    versions: dict[str, str]
    metrics: dict[str, int | float | bool]
    errors: list[str] = Field(default_factory=list)
