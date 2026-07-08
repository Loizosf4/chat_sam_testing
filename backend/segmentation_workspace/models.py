"""Strict Pydantic models for pre-MoGe segmentation workspaces."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CURRENT_SCHEMA_VERSION = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
UUID_PATTERN = (
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})$"
)
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        use_enum_values=True,
    )


class WorkspaceStatus(str, Enum):
    draft = "draft"
    ready_for_review = "ready_for_review"
    finalized = "finalized"


class SegmentationObjectStatus(str, Enum):
    draft = "draft"
    ready = "ready"
    finalized = "finalized"


class ImageDimensions(ContractModel):
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)


class SourceImage(ContractModel):
    image_id: str = Field(pattern=UUID_PATTERN)
    url: str
    original_filename: str = Field(min_length=1, max_length=255)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("url")
    @classmethod
    def application_url_only(cls, value: str) -> str:
        if value.startswith("/api/segmentation-artifacts/"):
            return value
        if value.startswith("file://") or _WINDOWS_PATH.match(value):
            raise ValueError("url must not be a filesystem path")
        raise ValueError("url must be an application-controlled segmentation artifact URL")

    @field_validator("original_filename")
    @classmethod
    def filename_only(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {"", ".", ".."}:
            raise ValueError("original_filename must be a basename")
        return value


def _non_empty_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("value cannot be empty")
    return stripped


class SegmentationObject(ContractModel):
    object_id: str = Field(pattern=UUID_PATTERN)
    object_version: int = Field(ge=1, strict=True)
    semantic_label: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=192)
    status: SegmentationObjectStatus = SegmentationObjectStatus.draft
    created_at: datetime
    updated_at: datetime

    @field_validator("semantic_label", "display_name")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        return _non_empty_text(value)


class SegmentationWorkspace(ContractModel):
    schema_version: Literal[CURRENT_SCHEMA_VERSION] = CURRENT_SCHEMA_VERSION
    workspace_id: str = Field(pattern=UUID_PATTERN)
    workspace_revision: int = Field(ge=1, strict=True)
    created_at: datetime
    updated_at: datetime
    source_image: SourceImage
    image_dimensions: ImageDimensions
    status: WorkspaceStatus = WorkspaceStatus.draft
    objects: list[SegmentationObject] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_object_ids(self) -> "SegmentationWorkspace":
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("duplicate object IDs are not allowed")
        return self
