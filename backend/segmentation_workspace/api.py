"""HTTP API for pre-MoGe segmentation workspaces."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend import sam_engine

from .models import SamPromptPoint
from . import service
from .store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


ROOT_DIR = Path(__file__).resolve().parents[2]
STORE = SegmentationWorkspaceStore(
    os.environ.get("SEGMENTATION_WORKSPACE_ROOT", ROOT_DIR / "data" / "segmentation_workspaces")
)
MAX_UPLOAD_BYTES = 128 * 1024 * 1024


class WorkspaceObjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_label: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=192)
    expected_workspace_revision: int = Field(ge=1)


class WorkspaceObjectUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_label: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=192)
    expected_workspace_revision: int = Field(ge=1)
    expected_object_version: int = Field(ge=1)


class SamPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_revision: int = Field(ge=1)
    points: list[SamPromptPoint] = Field(default_factory=list)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    base_prompt_revision: int | None = Field(default=None, ge=0)
    base_candidate_index: int | None = Field(default=None, ge=0)
    multimask_output: bool | None = None


class SamCandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_revision: int = Field(ge=1)
    candidate_index: int = Field(ge=0)
    expected_workspace_revision: int = Field(ge=1)
    expected_object_version: int = Field(ge=1)


router = APIRouter(prefix="/api/segmentation-workspaces", tags=["segmentation-workspaces"])
artifact_router = APIRouter(prefix="/api/segmentation-artifacts", tags=["segmentation-artifacts"])


def _error(exc: SegmentationWorkspaceStoreError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def _sam_error(exc: sam_engine.SamEngineError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    total = 0
    with destination.open("wb") as stream:
        while block := await upload.read(1024 * 1024):
            total += len(block)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "upload is too large")
            stream.write(block)


@router.post("", status_code=201)
async def create_workspace(source_image: UploadFile = File(...)) -> dict[str, Any]:
    if not source_image.filename:
        raise HTTPException(400, "missing source image filename")
    temporary = Path(tempfile.mkdtemp(prefix="segmentation-workspace-"))
    try:
        source = temporary / "source-image"
        await _save_upload(source_image, source)
        try:
            workspace = STORE.create_workspace(source, source_image.filename)
        except SegmentationWorkspaceStoreError as exc:
            raise _error(exc) from exc
        return workspace.model_dump(mode="json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@router.get("/{workspace_id}")
def get_workspace(workspace_id: str) -> dict[str, Any]:
    try:
        return STORE.get_workspace(workspace_id).model_dump(mode="json")
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.delete("/{workspace_id}")
def delete_workspace(
    workspace_id: str,
    expected_workspace_revision: int = Query(..., ge=1),
) -> dict[str, str]:
    try:
        STORE.delete_workspace(workspace_id, expected_workspace_revision)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    service.LOGITS_CACHE.clear_workspace(workspace_id)
    return {"workspace_id": workspace_id, "status": "deleted"}


@router.post("/{workspace_id}/objects", status_code=201)
def create_object(workspace_id: str, payload: WorkspaceObjectCreate) -> dict[str, Any]:
    try:
        workspace = STORE.create_object(
            workspace_id,
            semantic_label=payload.semantic_label,
            display_name=payload.display_name,
            expected_workspace_revision=payload.expected_workspace_revision,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    return workspace.model_dump(mode="json")


@router.patch("/{workspace_id}/objects/{object_id}")
def update_object(workspace_id: str, object_id: str, payload: WorkspaceObjectUpdate) -> dict[str, Any]:
    try:
        workspace = STORE.update_object(
            workspace_id,
            object_id,
            semantic_label=payload.semantic_label,
            display_name=payload.display_name,
            expected_workspace_revision=payload.expected_workspace_revision,
            expected_object_version=payload.expected_object_version,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    return workspace.model_dump(mode="json")


@router.delete("/{workspace_id}/objects/{object_id}")
def delete_object(
    workspace_id: str,
    object_id: str,
    expected_workspace_revision: int = Query(..., ge=1),
    expected_object_version: int = Query(..., ge=1),
) -> dict[str, Any]:
    try:
        workspace = STORE.delete_object(
            workspace_id,
            object_id,
            expected_workspace_revision=expected_workspace_revision,
            expected_object_version=expected_object_version,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    service.LOGITS_CACHE.clear_object(workspace_id, object_id)
    return workspace.model_dump(mode="json")


@router.post("/{workspace_id}/prepare-sam")
def prepare_sam(workspace_id: str) -> dict[str, Any]:
    try:
        return service.prepare_workspace_sam(STORE, workspace_id)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    except sam_engine.SamEngineError as exc:
        raise _sam_error(exc) from exc


@router.post("/{workspace_id}/objects/{object_id}/predict")
def predict_candidates(workspace_id: str, object_id: str, payload: SamPredictRequest) -> dict[str, Any]:
    try:
        return service.predict_workspace_candidates(
            STORE,
            workspace_id,
            object_id,
            prompt_revision=payload.prompt_revision,
            points=payload.points,
            box=payload.box,
            base_prompt_revision=payload.base_prompt_revision,
            base_candidate_index=payload.base_candidate_index,
            multimask_output=payload.multimask_output,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    except sam_engine.SamEngineError as exc:
        raise _sam_error(exc) from exc


@router.post("/{workspace_id}/objects/{object_id}/select-candidate")
def select_candidate(workspace_id: str, object_id: str, payload: SamCandidateSelection) -> dict[str, Any]:
    try:
        return service.select_workspace_candidate(
            STORE,
            workspace_id,
            object_id,
            prompt_revision=payload.prompt_revision,
            candidate_index=payload.candidate_index,
            expected_workspace_revision=payload.expected_workspace_revision,
            expected_object_version=payload.expected_object_version,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.delete("/{workspace_id}/objects/{object_id}/sam-draft")
def clear_sam_draft(
    workspace_id: str,
    object_id: str,
    expected_workspace_revision: int = Query(..., ge=1),
    expected_object_version: int = Query(..., ge=1),
) -> dict[str, Any]:
    try:
        return service.clear_workspace_sam_draft(
            STORE,
            workspace_id,
            object_id,
            expected_workspace_revision=expected_workspace_revision,
            expected_object_version=expected_object_version,
        )
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@artifact_router.get("/{kind}/{identifier:path}")
def get_segmentation_artifact(kind: str, identifier: str) -> FileResponse:
    try:
        path, media_type = STORE.resolve_artifact(kind, identifier)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store" if kind == "sam-candidates" else "private, max-age=3600",
        },
    )
