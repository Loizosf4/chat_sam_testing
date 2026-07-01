"""Strict typed output contract for semantic VLM scene-graph review."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Confidence = float
Predicate = Literal[
    "supported_by",
    "attached_to",
    "near",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "above",
    "below",
    "image_occludes",
    "image_occluded_by",
    "unknown_support",
]


class GeneratorInfo(StrictModel):
    provider: Literal["openai"]
    requested_model: str = Field(min_length=1)
    returned_model: str = Field(min_length=1)
    prompt_version: Literal["scene_graph_review_v1"]


class CameraReview(StrictModel):
    recommended_model: Literal["perspective", "orthographic", "unresolved"]
    confidence: Confidence = Field(ge=0.0, le=1.0)
    evidence: list[str]
    contradictory_evidence: list[str]
    test_both_in_blender: bool
    requires_user_review: bool


class StructuralPlaneReview(StrictModel):
    plane_id: str = Field(min_length=1)
    semantic_label: str = Field(min_length=1)
    decision: Literal["accept_equation", "reject", "uncertain"]
    confidence: Confidence = Field(ge=0.0, le=1.0)
    extent_policy: Literal["equation_only", "visible_image_extent", "requires_reconstruction_review"]
    reasoning_summary: str = Field(min_length=1)
    warnings: list[str]
    requires_user_review: bool


class ObjectReview(StrictModel):
    object_id: str = Field(min_length=1)
    original_label: str = Field(min_length=1)
    reviewed_label: str = Field(min_length=1)
    label_decision: Literal["accept", "correct", "uncertain"]
    semantic_category: str = Field(min_length=1)
    primitive_policy: Literal["one_cube"]
    geometry_strategy: Literal[
        "retained_obb",
        "canonical_aabb",
        "support_aligned_proxy",
        "wall_attached_thin_proxy",
        "visible_surface_proxy",
        "requires_manual_review",
    ]
    geometry_source_reference: str = Field(min_length=1)
    support_summary: str = Field(min_length=1)
    attachment_summary: str = Field(min_length=1)
    occlusion_summary: str = Field(min_length=1)
    confidence: Confidence = Field(ge=0.0, le=1.0)
    uncertainty: list[str]
    requires_user_review: bool


class RelationshipReview(StrictModel):
    candidate_id: str = Field(min_length=1)
    subject_object_id: str = Field(min_length=1)
    predicate: Predicate
    target_id: str | None
    deterministic_confidence: Confidence = Field(ge=0.0, le=1.0)
    decision: Literal["accept", "reject", "uncertain"]
    reviewed_confidence: Confidence = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(min_length=1)
    contradiction_summary: str
    requires_user_review: bool


class ProposedHypothesis(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    subject_object_id: str = Field(min_length=1)
    predicate: Predicate
    target_id: str | None
    status: Literal["unverified"]
    evidence_type: Literal["visual", "semantic"]
    evidence_summary: str = Field(min_length=1)
    confidence: Confidence = Field(ge=0.0, le=1.0)
    requires_user_review: Literal[True]


class SemanticSceneGraph(StrictModel):
    schema_version: Literal["1.0"]
    scene_id: str = Field(min_length=1)
    generator: GeneratorInfo
    camera_review: CameraReview
    structural_plane_reviews: list[StructuralPlaneReview]
    objects: list[ObjectReview]
    relationship_reviews: list[RelationshipReview]
    proposed_hypotheses: list[ProposedHypothesis]
    global_uncertainties: list[str]
    requires_user_review: bool
    review_summary: str = Field(min_length=1)
