"""Strict typed models for the Blender-neutral primitive scene plan."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AllowedObjectId = Literal[
    "e971e9fbbf5746eea108d07bfbb5764a",
    "b9858c1f45f443a88d84229a1824d339",
    "9171a7b4d4e142ca936f2564200d0bdb",
    "1064a3a09c4c427e80caa227d17a7cc1",
    "4ec402f9a71d4ea7925b487ac7e11249",
    "49fd02b50a914e8198bacf4644bc377f",
]
AllowedPlaneId = Literal["plane_floor", "plane_left_wall", "plane_right_wall"]
RelationshipDecision = Literal["accept", "reject", "uncertain"]
ConstraintStrength = Literal["hard", "soft", "review_only", "ignored_for_transform"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceReference(StrictModel):
    role: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CoordinateSystemDefinition(StrictModel):
    name: str
    handedness: Literal["right_handed"]
    x_axis: str
    y_axis: str
    z_axis: str
    origin: str
    units: str
    source_to_canonical_matrix: list[list[float]] = Field(min_length=4, max_length=4)
    canonical_to_source_matrix: list[list[float]] = Field(min_length=4, max_length=4)


class PerspectiveParameters(StrictModel):
    normalized_intrinsics: list[list[float]] = Field(min_length=3, max_length=3)
    horizontal_fov_degrees: float = Field(gt=0.0, lt=180.0)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)


class OrthographicParameters(StrictModel):
    pixels_per_canonical_unit: float = Field(gt=0.0)
    orthographic_scale_vertical: float = Field(gt=0.0)
    image_center_offset_pixels: list[float] = Field(min_length=2, max_length=2)
    fit_rmse_pixels: float = Field(ge=0.0)
    fit_median_error_pixels: float = Field(ge=0.0)
    correspondence_count: int = Field(gt=0)
    fitting_method: str


class CameraCandidate(StrictModel):
    camera_id: Literal["camera_perspective_moge", "camera_orthographic_fitted"]
    camera_type: Literal["perspective", "orthographic"]
    camera_to_canonical_transform: list[list[float]] = Field(min_length=4, max_length=4)
    canonical_to_camera_transform: list[list[float]] = Field(min_length=4, max_length=4)
    perspective: PerspectiveParameters | None
    orthographic: OrthographicParameters | None
    fit_confidence: float = Field(ge=0.0, le=1.0)
    source_evidence: list[str]
    warnings: list[str]
    requires_user_review: bool


class StructuralProxy(StrictModel):
    plane_id: AllowedPlaneId
    semantic_label: Literal["floor", "left_wall", "right_wall"]
    primitive_type: Literal["cube"]
    canonical_plane_normal: list[float] = Field(min_length=3, max_length=3)
    canonical_plane_offset: float
    center: list[float] = Field(min_length=3, max_length=3)
    rotation_quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)
    rotation_matrix: list[list[float]] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    transform_matrix: list[list[float]] = Field(min_length=4, max_length=4)
    equation_confidence: float = Field(ge=0.0, le=1.0)
    extent_confidence: float = Field(ge=0.0, le=1.0)
    extent_policy: Literal["bright_near_plane_robust_clip"]
    thickness_policy: str
    warnings: list[str]
    requires_user_review: bool


class SemanticPrimitive(StrictModel):
    object_id: AllowedObjectId
    semantic_label: str
    primitive_type: Literal["cube"]
    geometry_strategy: Literal[
        "retained_obb",
        "support_aligned_proxy",
        "visible_surface_proxy",
        "wall_attached_thin_proxy",
    ]
    geometry_source: list[str]
    center: list[float] = Field(min_length=3, max_length=3)
    rotation_quaternion_wxyz: list[float] = Field(min_length=4, max_length=4)
    rotation_matrix: list[list[float]] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    transform_matrix: list[list[float]] = Field(min_length=4, max_length=4)
    support_target: str | None
    constraint_strength: ConstraintStrength
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: list[str]
    requires_user_review: bool


class ConstraintClassification(StrictModel):
    candidate_id: str = Field(pattern=r"^rel_[0-9a-f]{24}$")
    subject_object_id: AllowedObjectId
    predicate: str
    target_id: str | None
    decision: RelationshipDecision
    reviewed_confidence: float = Field(ge=0.0, le=1.0)
    classification: ConstraintStrength
    influences_transform: bool
    source_relationship: str
    rationale: str
    requires_user_review: bool


class ProjectionMetrics(StrictModel):
    camera_id: Literal["camera_perspective_moge", "camera_orthographic_fitted"]
    object_id: AllowedObjectId
    projected_corners_xy: list[list[float]] = Field(min_length=8, max_length=8)
    projected_bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    source_mask_bbox_iou: float = Field(ge=0.0, le=1.0)
    projected_hull_mask_iou: float = Field(ge=0.0, le=1.0)
    centroid_error_pixels: float = Field(ge=0.0)
    projected_area_ratio: float = Field(ge=0.0)
    visibility_valid: bool
    quality: Literal["good", "marginal", "failed"]
    warnings: list[str]


class ProjectionValidation(StrictModel):
    shared_geometry_across_cameras: Literal[True]
    metrics: list[ProjectionMetrics] = Field(min_length=12, max_length=12)
    perspective_mean_bbox_iou: float = Field(ge=0.0, le=1.0)
    perspective_mean_hull_iou: float = Field(ge=0.0, le=1.0)
    perspective_mean_centroid_error_pixels: float = Field(ge=0.0)
    orthographic_mean_bbox_iou: float = Field(ge=0.0, le=1.0)
    orthographic_mean_hull_iou: float = Field(ge=0.0, le=1.0)
    orthographic_mean_centroid_error_pixels: float = Field(ge=0.0)
    recommended_first_blender_camera: Literal["camera_perspective_moge", "camera_orthographic_fitted"]
    other_camera_must_still_be_tested: Literal[True]


class QualityGateResult(StrictModel):
    gate: str
    passed: bool
    details: str


class CompilationStatus(StrictModel):
    status: Literal["ready_for_blender_execution", "needs_compiler_revision", "blocked"]
    deterministic: Literal[True]
    quality_gates_passed: bool
    rejected_quality_gates: list[str]
    blender_objects_created: Literal[False]


class PrimitiveScenePlan(StrictModel):
    schema_version: Literal["1.0"]
    scene_id: Literal["office_test"]
    source_references: list[SourceReference] = Field(min_length=7, max_length=7)
    coordinate_system: CoordinateSystemDefinition
    unit_scale_warning: str
    camera_candidates: list[CameraCandidate] = Field(min_length=2, max_length=2)
    structural_proxies: list[StructuralProxy] = Field(min_length=3, max_length=3)
    semantic_primitives: list[SemanticPrimitive] = Field(min_length=6, max_length=6)
    constraint_classifications: list[ConstraintClassification] = Field(min_length=98, max_length=98)
    projection_validation: ProjectionValidation
    uncertainties: list[str]
    user_review_requirements: list[str]
    quality_gates: list[QualityGateResult]
    compilation_status: CompilationStatus
