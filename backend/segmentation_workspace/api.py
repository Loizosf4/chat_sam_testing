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


router = APIRouter(prefix="/api/segmentation-workspaces", tags=["segmentation-workspaces"])
artifact_router = APIRouter(prefix="/api/segmentation-artifacts", tags=["segmentation-artifacts"])


def _error(exc: SegmentationWorkspaceStoreError) -> HTTPException:
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
    return workspace.model_dump(mode="json")


@artifact_router.get("/{kind}/{identifier:path}")
def get_segmentation_artifact(kind: str, identifier: str) -> FileResponse:
    try:
        path, media_type = STORE.resolve_artifact(kind, identifier)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc
    return FileResponse(
        path,
        media_type=media_type,
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"},
    )
