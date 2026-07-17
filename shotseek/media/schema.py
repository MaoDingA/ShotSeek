"""Strict M1A media and alignment contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VideoContract(ContractModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    fps_num: int = Field(gt=0)
    fps_den: int = Field(gt=0)
    time_base_num: int = Field(gt=0)
    time_base_den: int = Field(gt=0)
    video_codec: str = Field(min_length=1)
    cfr: bool


class ShotBoundary(ContractModel):
    boundary_id: str = Field(min_length=1)
    frame: int = Field(gt=0)
    timestamp_ms: int = Field(gt=0)
    pts: int = Field(ge=0)
    scene_score: float = Field(ge=0.0, le=1.0)
    detector: str = Field(min_length=1)


class Shot(ContractModel):
    shot_id: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    duration_frames: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "Shot":
        if self.end_frame <= self.start_frame or self.end_ms <= self.start_ms:
            raise ValueError("shot range must be positive")
        if self.duration_frames != self.end_frame - self.start_frame:
            raise ValueError("duration_frames does not match frame range")
        return self


class AlignedVisualEvent(ContractModel):
    event_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_start_ms: int = Field(ge=0)
    raw_local_start_ms: int = Field(ge=0)
    raw_local_end_ms: int = Field(gt=0)
    raw_global_start_ms: int = Field(ge=0)
    raw_global_end_ms: int = Field(gt=0)
    final_start_ms: int = Field(ge=0)
    final_end_ms: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    shot_ids: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)
    characters: list[str]
    actions: list[str]
    objects: list[str]
    location: str | None
    visible_text: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)
    model: str = Field(min_length=1)
    boundary_strategy: str = "shot_first"
    start_delta_frames: int
    end_delta_frames: int


class ContextualizedUtterance(ContractModel):
    utterance_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    speaker_id: str | None
    words: list[dict[str, Any]]
    source: str = Field(min_length=1)
    shot_ids: list[str] = Field(min_length=1)


class M1AManifest(ContractModel):
    schema_version: str = "m1a-v1"
    input_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fixture_sha256: dict[str, str]
    detector: dict[str, Any]
    interval_semantics: str = "half_open"
    boundary_semantics: str = "first_frame_of_new_shot"
    network_calls: int = 0
