"""Browser-safe Pydantic models for managed reconstruction jobs."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import ContractModel, SHA256_PATTERN, UUID_PATTERN


RECONSTRUCTION_JOB_SCHEMA_VERSION = "1.0.0"
RECONSTRUCTION_ARTIFACT_PREFIX = "/api/segmentation-artifacts/reconstruction-results/"
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ReconstructionJobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    interrupted = "interrupted"


class ReconstructionJobStage(str, Enum):
    queued = "queued"
    validating = "validating"
    moge = "moge"
    compiling = "compiling"
    publishing = "publishing"
    complete = "complete"
    failed = "failed"
    interrupted = "interrupted"


PROGRESS_BY_STAGE: dict[str, int] = {
    "queued": 0,
    "validating": 5,
    "moge": 10,
    "compiling": 60,
    "publishing": 95,
    "complete": 100,
    "failed": 100,
    "interrupted": 100,
}


class ReconstructionJobRequest(ContractModel):
    expected_workspace_revision: int = Field(ge=1, strict=True)
    expected_export_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    resolution_level: int = Field(ge=1, le=9, strict=True)
    num_tokens: int | None = Field(default=None, ge=1, strict=True)


class ReconstructionJobError(ContractModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    stage: ReconstructionJobStage
    retryable: bool


class ReconstructionArtifactSet(ContractModel):
    result_manifest_url: str
    unified_scene_plan_url: str
    compilation_report_url: str
    compilation_markdown_url: str
    room_plan_url: str
    camera_candidates_url: str
    object_pose_report_url: str
    placement_report_url: str
    collision_report_url: str
    confidence_report_url: str
    support_graph_url: str
    overview_url: str | None = None
    projected_primitives_url: str | None = None
    room_camera_url: str | None = None
    confidence_overview_url: str | None = None
    ambiguity_overview_url: str | None = None
    blender_manifest_url: str
    moge_geometry_url: str
    moge_summary_url: str
    depth_preview_url: str | None = None
    normal_preview_url: str | None = None
    valid_mask_preview_url: str | None = None

    @field_validator("*")
    @classmethod
    def artifact_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.startswith(RECONSTRUCTION_ARTIFACT_PREFIX):
            return value
        if value.startswith("file://") or _WINDOWS_PATH.match(value) or value.startswith("/tmp/"):
            raise ValueError("reconstruction artifact URLs must not be filesystem paths")
        raise ValueError("reconstruction artifact URLs must be application-controlled reconstruction URLs")


class ReconstructionJobResult(ContractModel):
    scene_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    semantic_object_count: int = Field(ge=1, strict=True)
    object_ids: list[str] = Field(min_length=1)
    compilation_passed: bool
    artifacts: ReconstructionArtifactSet

    @model_validator(mode="after")
    def validate_object_count(self) -> "ReconstructionJobResult":
        if len(self.object_ids) != self.semantic_object_count:
            raise ValueError("semantic_object_count must equal object_ids length")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("duplicate result object IDs are not allowed")
        return self


class ReconstructionJobRecord(ContractModel):
    schema_version: Literal[RECONSTRUCTION_JOB_SCHEMA_VERSION] = RECONSTRUCTION_JOB_SCHEMA_VERSION
    job_id: str = Field(pattern=UUID_PATTERN)
    job_version: int = Field(ge=1, strict=True)
    workspace_id: str = Field(pattern=UUID_PATTERN)
    export_id: str = Field(pattern=UUID_PATTERN)
    export_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    source_image_id: str = Field(pattern=UUID_PATTERN)
    status: ReconstructionJobStatus
    stage: ReconstructionJobStage
    progress_percent: int = Field(ge=0, le=100, strict=True)
    created_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    finished_at: datetime | None = None
    request: ReconstructionJobRequest
    export_was_stale_at_start: bool
    semantic_object_count: int = Field(ge=1, strict=True)
    object_ids: list[str] = Field(min_length=1)
    scene_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    result: ReconstructionJobResult | None = None
    error: ReconstructionJobError | None = None

    @model_validator(mode="after")
    def validate_record(self) -> "ReconstructionJobRecord":
        if len(self.object_ids) != self.semantic_object_count:
            raise ValueError("semantic_object_count must equal object_ids length")
        if len(self.object_ids) != len(set(self.object_ids)):
            raise ValueError("duplicate object IDs are not allowed")
        expected_progress = PROGRESS_BY_STAGE[str(self.stage)]
        if self.progress_percent != expected_progress:
            raise ValueError("progress_percent must match the current stage")
        if self.status == "queued":
            if self.stage != "queued" or self.started_at is not None or self.finished_at is not None or self.result or self.error:
                raise ValueError("queued jobs must not have started, finished, result, or error state")
        elif self.status == "running":
            if self.stage not in {"validating", "moge", "compiling", "publishing"} or self.started_at is None or self.finished_at is not None or self.result or self.error:
                raise ValueError("running jobs require active stage and started_at only")
        elif self.status == "succeeded":
            if self.stage != "complete" or self.finished_at is None or self.result is None or self.error is not None:
                raise ValueError("succeeded jobs require complete stage, finished_at, and result")
        elif self.status in {"failed", "interrupted"}:
            expected_stage = "failed" if self.status == "failed" else "interrupted"
            if self.stage != expected_stage or self.finished_at is None or self.error is None or self.result is not None:
                raise ValueError("failed/interrupted jobs require terminal stage, finished_at, and error")
        return self
