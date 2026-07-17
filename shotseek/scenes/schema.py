"""Strict, evidence-referenced M1B scene contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SceneModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceRef(SceneModel):
    kind: Literal["visual", "dialogue"]
    evidence_id: str = Field(min_length=1)


class Scene(SceneModel):
    schema_version: str = "scene-v1"
    scene_id: str = Field(pattern=r"^scene_[0-9]{4}$")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    shot_ids: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)
    characters: list[str]
    actions: list[str]
    objects: list[str]
    location: str | None
    visible_text: list[str]
    visual_event_id: str = Field(min_length=1)
    utterance_ids: list[str]
    evidence_refs: list[EvidenceRef] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    boundary_strategy: Literal["shot_first"] = "shot_first"

    @model_validator(mode="after")
    def validate_contract(self) -> "Scene":
        if self.end_ms <= self.start_ms or self.end_frame <= self.start_frame:
            raise ValueError("scene range must be positive")
        if len(self.shot_ids) != len(set(self.shot_ids)):
            raise ValueError("scene shot_ids must be unique")
        if len(self.utterance_ids) != len(set(self.utterance_ids)):
            raise ValueError("scene utterance_ids must be unique")
        visual_refs = [
            ref.evidence_id for ref in self.evidence_refs if ref.kind == "visual"
        ]
        dialogue_refs = [
            ref.evidence_id for ref in self.evidence_refs if ref.kind == "dialogue"
        ]
        if visual_refs != [self.visual_event_id]:
            raise ValueError("scene must contain exactly its visual evidence reference")
        if dialogue_refs != self.utterance_ids:
            raise ValueError("dialogue evidence references must match utterance_ids")
        return self
