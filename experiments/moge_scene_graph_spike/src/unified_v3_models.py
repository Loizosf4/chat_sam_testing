"""Blender-neutral data contracts for the unified clean V3 compiler."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfidenceClass(str, Enum):
    automatic_high_confidence = "automatic_high_confidence"
    automatic_with_ambiguity = "automatic_with_ambiguity"
    user_review_recommended = "user_review_recommended"
    yaw_unobservable = "yaw_unobservable"
    insufficient_geometry = "insufficient_geometry"


class Transform(BaseModel):
    center: list[float] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    rotation_matrix: list[list[float]]
    quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)

    @field_validator("center","dimensions","quaternion_wxyz")
    @classmethod
    def finite_vectors(cls,value:list[float])->list[float]:
        if not all(math.isfinite(item) for item in value):raise ValueError("transform vectors must be finite")
        return value

    @model_validator(mode="after")
    def validate_transform(self)->"Transform":
        if any(value<=0 for value in self.dimensions):raise ValueError("transform dimensions must be positive")
        if len(self.rotation_matrix)!=3 or any(len(row)!=3 for row in self.rotation_matrix):raise ValueError("rotation_matrix must be 3x3")
        if not all(math.isfinite(value) for row in self.rotation_matrix for value in row):raise ValueError("rotation_matrix must be finite")
        norm=math.sqrt(sum(value*value for value in self.quaternion_wxyz))
        if abs(norm-1.0)>1e-6:raise ValueError("quaternion_wxyz must be normalized")
        return self


class UnifiedObject(BaseModel):
    model_config = ConfigDict(extra="allow")
    object_id: str
    semantic_label: str
    primitive_type: str = "cube"
    transform: Transform
    support_target: str | None
    confidence_classification: ConfidenceClass
    final_pose_confidence: float = Field(ge=0, le=1)
    normal_frame: dict[str, Any]
    validation_metrics: dict[str, Any]
    ambiguity: dict[str, Any]


class UnifiedScenePlan(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    mode: str
    scene_id: str
    room_proxies: list[dict[str, Any]]
    camera_candidates: list[dict[str, Any]]
    semantic_objects: list[UnifiedObject]
    semantic_object_count: int

    @model_validator(mode="after")
    def validate_object_identity(self)->"UnifiedScenePlan":
        ids=[item.object_id for item in self.semantic_objects]
        if self.semantic_object_count!=len(ids):raise ValueError("semantic_object_count does not match semantic_objects")
        if len(set(ids))!=len(ids):raise ValueError("semantic object IDs must be unique")
        known=set(ids)|{str(item.get("plane_id")) for item in self.room_proxies}
        for item in self.semantic_objects:
            if item.support_target==item.object_id:raise ValueError(f"object {item.object_id} cannot support itself")
            if item.support_target is not None and item.support_target not in known:raise ValueError(f"unresolved support target: {item.support_target}")
        return self

