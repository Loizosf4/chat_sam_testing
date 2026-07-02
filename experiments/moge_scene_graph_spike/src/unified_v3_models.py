"""Blender-neutral data contracts for the unified clean V3 compiler."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

