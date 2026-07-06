"""Pydantic models for the browser/backend/Blender-neutral scene package."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CURRENT_SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
STABLE_OBJECT_ID_PATTERN = (
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$"
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        use_enum_values=True,
    )


class ApprovalStatus(str, Enum):
    unreviewed = "unreviewed"
    approved = "approved"
    rejected = "rejected"
    needs_revision = "needs_revision"


class SupportType(str, Enum):
    floor = "floor"
    wall = "wall"
    object = "object"
    ceiling = "ceiling"
    unknown = "unknown"


class OcclusionState(str, Enum):
    visible = "visible"
    partially_occluded = "partially_occluded"
    fully_occluded = "fully_occluded"
    unknown = "unknown"


class GenerationState(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    stale = "stale"


class GeometryInvalidationStatus(str, Enum):
    not_computed = "not_computed"
    valid = "valid"
    invalidated = "invalidated"
    recompute_required = "recompute_required"


class EditOperation(str, Enum):
    initial_sam_export = "initial_sam_export"
    merge = "merge"
    subtract = "subtract"
    fill_holes = "fill_holes"
    remove_small_components = "remove_small_components"
    smooth = "smooth"
    manual = "manual"
    imported = "imported"
    other = "other"


class AuthorSource(str, Enum):
    sam = "sam"
    frontend = "frontend"
    backend = "backend"
    import_adapter = "import_adapter"
    blender = "blender"
    unknown = "unknown"


class CameraType(str, Enum):
    perspective = "perspective"
    orthographic = "orthographic"


class ArtifactReference(ContractModel):
    artifact_id: str = Field(pattern=ID_PATTERN)
    url: str
    media_type: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("url")
    @classmethod
    def application_url_only(cls, value: str) -> str:
        if value.startswith(("/artifacts/", "/api/artifacts/", "https://", "http://")):
            return value
        raise ValueError("url must be an application artifact URL or HTTP(S) URL")


class ImageDimensions(ContractModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SourceImage(ContractModel):
    artifact_id: str = Field(pattern=ID_PATTERN)
    url: str
    original_filename: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("url")
    @classmethod
    def application_url_only(cls, value: str) -> str:
        return ArtifactReference.application_url_only(value)

    @field_validator("original_filename")
    @classmethod
    def filename_only(cls, value: str | None) -> str | None:
        if value is not None and ("/" in value or "\\" in value):
            raise ValueError("original_filename must not contain a path")
        return value


class CoordinateSystem(ContractModel):
    coordinate_system_id: str = Field(pattern=ID_PATTERN)
    handedness: Literal["right", "left"]
    up_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    units: Literal["meters", "pixels", "normalized"]
    origin: str
    transform_to_world: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transform_to_world")
    @classmethod
    def matrix_is_4x4(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is not None and (len(value) != 4 or any(len(row) != 4 for row in value)):
            raise ValueError("transform_to_world must be a 4x4 matrix")
        return value


class Transform(ContractModel):
    center: list[float] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(min_length=4, max_length=4)

    @field_validator("dimensions")
    @classmethod
    def dimensions_are_positive(cls, value: list[float]) -> list[float]:
        if any(component <= 0 for component in value):
            raise ValueError("all dimensions must be positive")
        return value

    @field_validator("quaternion")
    @classmethod
    def quaternion_is_normalized(cls, value: list[float]) -> list[float]:
        norm = math.sqrt(sum(component * component for component in value))
        if not 0.999 <= norm <= 1.001:
            raise ValueError("quaternion must be normalized")
        return value


class CameraCandidate(ContractModel):
    camera_id: str = Field(pattern=ID_PATTERN)
    camera_type: CameraType
    transform: list[list[float]]
    normalized_intrinsics: list[list[float]] | None = None
    field_of_view_x_degrees: float | None = Field(default=None, gt=0, lt=180)
    confidence: float = Field(ge=0, le=1)
    provisional: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transform")
    @classmethod
    def transform_is_4x4(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("camera transform must be a 4x4 matrix")
        return value

    @field_validator("normalized_intrinsics")
    @classmethod
    def intrinsics_are_3x3(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is not None and (len(value) != 3 or any(len(row) != 3 for row in value)):
            raise ValueError("normalized_intrinsics must be a 3x3 matrix")
        return value


class StructuralRoomProxy(ContractModel):
    proxy_id: str = Field(pattern=ID_PATTERN)
    semantic: Literal["floor", "wall", "ceiling", "room_boundary"]
    display_name: str
    transform: Transform
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationStatus(ContractModel):
    overall: GenerationState
    stages: dict[str, GenerationState] = Field(default_factory=dict)
    updated_at: datetime
    errors: list[str] = Field(default_factory=list)


class MaskQualityReport(ContractModel):
    binary: bool
    non_empty: bool
    area_pixels: int = Field(ge=0)
    bbox_xyxy: list[int] | None = Field(default=None, min_length=4, max_length=4)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class MaskRevision(ContractModel):
    revision_id: str = Field(pattern=STABLE_OBJECT_ID_PATTERN)
    object_id: str = Field(pattern=STABLE_OBJECT_ID_PATTERN)
    revision_number: int = Field(ge=1)
    parent_revision: str | None = Field(default=None, pattern=STABLE_OBJECT_ID_PATTERN)
    binary_mask_hash: str = Field(pattern=SHA256_PATTERN)
    edit_operation: EditOperation
    author: str
    source: AuthorSource
    timestamp: datetime
    quality_report: MaskQualityReport
    geometry_invalidation_status: GeometryInvalidationStatus


class SemanticObject(ContractModel):
    object_id: str = Field(pattern=STABLE_OBJECT_ID_PATTERN)
    version: int = Field(ge=1)
    semantic_label: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=192)
    mask_filename: str
    mask_url: str
    mask_revision: str = Field(pattern=STABLE_OBJECT_ID_PATTERN)
    mask_hash: str = Field(pattern=SHA256_PATTERN)
    overlay_url: str | None = None
    approval_status: ApprovalStatus
    quality_warnings: list[str] = Field(default_factory=list)
    geometry_confidence: float = Field(ge=0, le=1)
    orientation_confidence: float = Field(ge=0, le=1)
    placement_confidence: float = Field(ge=0, le=1)
    center: list[float] = Field(min_length=3, max_length=3)
    dimensions: list[float] = Field(min_length=3, max_length=3)
    quaternion: list[float] = Field(min_length=4, max_length=4)
    support_target: str | None = None
    support_type: SupportType
    occlusion_state: OcclusionState
    blender_object_identifier: str | None = None
    selected_asset_identifier: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mask_filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value:
            raise ValueError("mask_filename must be a filename, not a path")
        return value

    @field_validator("mask_url", "overlay_url")
    @classmethod
    def application_urls_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ArtifactReference.application_url_only(value)

    @model_validator(mode="after")
    def valid_transform(self) -> "SemanticObject":
        Transform(center=self.center, dimensions=self.dimensions, quaternion=self.quaternion)
        if self.support_target == self.object_id:
            raise ValueError("an object cannot support itself")
        return self


def _assert_no_filesystem_paths(value: Any, location: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_filesystem_paths(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_filesystem_paths(child, f"{location}[{index}]")
    elif isinstance(value, str):
        lower = value.lower()
        allowed_application_path = value.startswith(("/artifacts/", "/api/artifacts/"))
        if lower.startswith("file://") or _WINDOWS_PATH.match(value):
            raise ValueError(f"filesystem path is forbidden at {location}")
        if value.startswith("/") and not allowed_application_path:
            raise ValueError(f"filesystem path is forbidden at {location}")


class Scene(ContractModel):
    schema_version: Literal[CURRENT_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
    package_revision: int = Field(ge=1)
    scene_id: str = Field(pattern=ID_PATTERN)
    source_image: SourceImage
    image_dimensions: ImageDimensions
    coordinate_systems: list[CoordinateSystem] = Field(min_length=2)
    camera_candidates: list[CameraCandidate] = Field(min_length=1)
    selected_camera: str = Field(pattern=ID_PATTERN)
    structural_room_proxies: list[StructuralRoomProxy]
    generation_status: GenerationStatus
    artifact_urls: dict[str, ArtifactReference]
    semantic_objects: list[SemanticObject]
    mask_revisions: list[MaskRevision]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> "Scene":
        object_ids = [item.object_id for item in self.semantic_objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("duplicate object IDs are not allowed")

        camera_ids = [camera.camera_id for camera in self.camera_candidates]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("duplicate camera IDs are not allowed")
        if self.selected_camera not in set(camera_ids):
            raise ValueError("selected_camera must reference a camera candidate")

        proxy_ids = [proxy.proxy_id for proxy in self.structural_room_proxies]
        if len(proxy_ids) != len(set(proxy_ids)):
            raise ValueError("duplicate structural proxy IDs are not allowed")
        if set(proxy_ids) & set(object_ids):
            raise ValueError("object and structural proxy IDs must be disjoint")

        revision_ids = [revision.revision_id for revision in self.mask_revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise ValueError("duplicate mask revision IDs are not allowed")
        revisions = {revision.revision_id: revision for revision in self.mask_revisions}
        revision_numbers: set[tuple[str, int]] = set()
        for revision in self.mask_revisions:
            key = (revision.object_id, revision.revision_number)
            if key in revision_numbers:
                raise ValueError("duplicate mask revision number for object")
            revision_numbers.add(key)
            if revision.object_id not in set(object_ids):
                raise ValueError("mask revision references an unknown object")
            if revision.parent_revision is not None:
                parent = revisions.get(revision.parent_revision)
                if parent is None or parent.object_id != revision.object_id:
                    raise ValueError("mask revision parent must reference the same object")
                if parent.revision_number >= revision.revision_number:
                    raise ValueError("mask revision parent must precede its child")

        proxies = {proxy.proxy_id: proxy for proxy in self.structural_room_proxies}
        for item in self.semantic_objects:
            revision = revisions.get(item.mask_revision)
            if revision is None or revision.object_id != item.object_id:
                raise ValueError("object mask_revision must reference its own revision")
            if revision.binary_mask_hash != item.mask_hash:
                raise ValueError("object mask_hash must match its current mask revision")
            if item.support_type == SupportType.object:
                if item.support_target not in set(object_ids):
                    raise ValueError("object support target must reference a semantic object")
            elif item.support_type in {SupportType.floor, SupportType.wall, SupportType.ceiling}:
                proxy = proxies.get(item.support_target or "")
                if proxy is None:
                    raise ValueError("structural support target must reference a room proxy")
                expected = item.support_type
                if expected == SupportType.wall and proxy.semantic != "wall":
                    raise ValueError("wall support must reference a wall proxy")
                if expected != SupportType.wall and proxy.semantic != expected:
                    raise ValueError(f"{expected} support must reference a matching proxy")
            elif item.support_target is not None:
                raise ValueError("unknown support type must not carry a support target")

        _assert_no_filesystem_paths(self.model_dump(mode="json"))
        return self
