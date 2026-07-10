"""HTTP API for pre-MoGe segmentation workspaces."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend import sam_engine
from backend import moge_engine, reconstruction_engine

from .models import SamPromptPoint
from . import manual_masks
from . import service
from .reconstruction_jobs import ReconstructionJobManager, ReconstructionJobStore
from .reconstruction_models import ReconstructionJobRequest
from .store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


ROOT_DIR = Path(__file__).resolve().parents[2]
STORE = SegmentationWorkspaceStore(
    os.environ.get("SEGMENTATION_WORKSPACE_ROOT", ROOT_DIR / "data" / "segmentation_workspaces")
)
RECONSTRUCTION_JOBS = ReconstructionJobStore(STORE)
RECONSTRUCTION_MANAGER = ReconstructionJobManager(RECONSTRUCTION_JOBS)
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


class ExpectedExportObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(min_length=1, max_length=64)
    expected_object_version: int = Field(ge=1)


class CreateWorkspaceExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_workspace_revision: int = Field(ge=1)
    expected_objects: list[ExpectedExportObject] = Field(min_length=1)
    include_previews: bool = True

    @model_validator(mode="after")
    def validate_unique_objects(self) -> "CreateWorkspaceExportRequest":
        object_ids = [item.object_id for item in self.expected_objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("duplicate expected object IDs are not allowed")
        return self


router = APIRouter(prefix="/api/segmentation-workspaces", tags=["segmentation-workspaces"])
artifact_router = APIRouter(prefix="/api/segmentation-artifacts", tags=["segmentation-artifacts"])
reconstruction_router = APIRouter(prefix="/api/segmentation-reconstruction", tags=["segmentation-reconstruction"])


def _error(exc: SegmentationWorkspaceStoreError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def _sam_error(exc: sam_engine.SamEngineError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


def _validate_reconstruction_environment() -> None:
    try:
        moge_engine.validate_model_config()
    except moge_engine.MogeEngineError as exc:
        message = reconstruction_engine.sanitize_user_message(exc)
        raise HTTPException(status_code=503, detail=message) from exc
    try:
        reconstruction_engine.reconstruction_python(validate_exists=True)
    except reconstruction_engine.ReconstructionEngineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile, destination: Path, *, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    total = 0
    with destination.open("wb") as stream:
        while block := await upload.read(1024 * 1024):
            total += len(block)
            if total > max_bytes:
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


def _export_response(workspace_id: str, export_id: str) -> dict[str, Any]:
    record = STORE.get_export(workspace_id, export_id)
    response = record.model_dump(mode="json")
    response["is_stale"] = STORE.export_is_stale(workspace_id, export_id)
    return response


@router.post("/{workspace_id}/exports", status_code=201)
def create_export(workspace_id: str, payload: CreateWorkspaceExportRequest) -> dict[str, Any]:
    try:
        record = STORE.create_export(
            workspace_id,
            expected_workspace_revision=payload.expected_workspace_revision,
            expected_objects=[item.model_dump() for item in payload.expected_objects],
            include_previews=payload.include_previews,
        )
        return _export_response(workspace_id, record.export_id)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.get("/{workspace_id}/exports")
def list_exports(workspace_id: str) -> list[dict[str, Any]]:
    try:
        return [_export_response(workspace_id, record.export_id) for record in STORE.list_exports(workspace_id)]
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.get("/{workspace_id}/exports/{export_id}")
def get_export(workspace_id: str, export_id: str) -> dict[str, Any]:
    try:
        return _export_response(workspace_id, export_id)
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.post("/{workspace_id}/exports/{export_id}/reconstructions", status_code=202)
def start_reconstruction(workspace_id: str, export_id: str, payload: ReconstructionJobRequest) -> dict[str, Any]:
    _validate_reconstruction_environment()
    try:
        job = RECONSTRUCTION_JOBS.create_job(workspace_id, export_id, payload)
        RECONSTRUCTION_MANAGER.submit(job)
        return job.model_dump(mode="json")
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.get("/{workspace_id}/reconstructions")
def list_reconstructions(workspace_id: str) -> list[dict[str, Any]]:
    try:
        return [job.model_dump(mode="json") for job in RECONSTRUCTION_JOBS.list_jobs(workspace_id)]
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@router.get("/{workspace_id}/reconstructions/{job_id}")
def get_reconstruction(workspace_id: str, job_id: str) -> dict[str, Any]:
    try:
        return RECONSTRUCTION_JOBS.get_job(workspace_id, job_id).model_dump(mode="json")
    except SegmentationWorkspaceStoreError as exc:
        raise _error(exc) from exc


@reconstruction_router.get("/health")
def reconstruction_health() -> dict[str, Any]:
    moge_status = moge_engine.get_status()
    compiler_status = reconstruction_engine.compiler_health()
    counts = RECONSTRUCTION_JOBS.active_count_by_status()
    moge_configured = bool(moge_status.get("configured"))
    compiler_configured = bool(compiler_status.get("compiler_configured"))
    error = None
    if not moge_configured:
        error = "MoGe is not configured."
    elif not compiler_status.get("configured"):
        error = compiler_status.get("error") or "Reconstruction compiler is not configured."
    return {
        "configured": moge_configured and bool(compiler_status.get("configured")),
        "moge_configured": moge_configured,
        "compiler_configured": compiler_configured,
        "worker_running": bool(moge_status.get("worker_running")),
        "device": moge_status.get("device") if moge_configured else None,
        "model": moge_status.get("model") if moge_configured else None,
        "queue_running": counts["running"],
        "queue_queued": counts["queued"],
        "error": error,
    }


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


@router.put("/{workspace_id}/objects/{object_id}/manual-mask")
async def save_manual_mask(
    workspace_id: str,
    object_id: str,
    edited_mask: UploadFile = File(...),
    base_prompt_revision: int = Form(..., ge=1),
    base_candidate_index: int = Form(..., ge=0),
    expected_workspace_revision: int = Form(..., ge=1),
    expected_object_version: int = Form(..., ge=1),
    expected_manual_revision: int = Form(..., ge=0),
) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix="segmentation-manual-mask-"))
    try:
        uploaded = temporary / "edited-mask"
        await _save_upload(
            edited_mask,
            uploaded,
            max_bytes=manual_masks.MAX_MANUAL_MASK_UPLOAD_BYTES,
        )
        try:
            return manual_masks.save_manual_mask(
                STORE,
                workspace_id,
                object_id,
                edited_mask_path=uploaded,
                base_prompt_revision=base_prompt_revision,
                base_candidate_index=base_candidate_index,
                expected_workspace_revision=expected_workspace_revision,
                expected_object_version=expected_object_version,
                expected_manual_revision=expected_manual_revision,
            )
        except SegmentationWorkspaceStoreError as exc:
            raise _error(exc) from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@router.delete("/{workspace_id}/objects/{object_id}/manual-mask")
def clear_manual_mask(
    workspace_id: str,
    object_id: str,
    expected_workspace_revision: int = Query(..., ge=1),
    expected_object_version: int = Query(..., ge=1),
    expected_manual_revision: int = Query(..., ge=0),
) -> dict[str, Any]:
    try:
        return manual_masks.clear_manual_mask(
            STORE,
            workspace_id,
            object_id,
            expected_workspace_revision=expected_workspace_revision,
            expected_object_version=expected_object_version,
            expected_manual_revision=expected_manual_revision,
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
            "Cache-Control": (
                "no-store"
                if kind in {"sam-candidates", "manual-masks"}
                else "private, max-age=31536000, immutable"
                if kind in {"workspace-exports", "reconstruction-results"}
                else "private, max-age=3600"
            ),
        },
    )
