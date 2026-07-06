"""HTTP API for scene-package import, review, revision, and reconstruction."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .models import ApprovalStatus, EditOperation, Transform
from .store import SceneStore, SceneStoreError, valid_id


ROOT_DIR = Path(__file__).resolve().parents[2]
STORE = SceneStore(os.environ.get("SCENE_PACKAGE_ROOT", ROOT_DIR / "data" / "scene_packages"))
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_FILES = 1000


class LabelUpdate(BaseModel):
    semantic_label: str = Field(min_length=1, max_length=128)
    display_name: str | None = Field(default=None, min_length=1, max_length=192)
    expected_package_revision: int = Field(ge=1)
    expected_object_version: int = Field(ge=1)


class ApprovalUpdate(BaseModel):
    approval_status: ApprovalStatus
    expected_package_revision: int = Field(ge=1)
    expected_object_version: int = Field(ge=1)


class CompiledTransform(BaseModel):
    object_id: str
    expected_object_version: int = Field(ge=1)
    expected_mask_revision: str
    transform: Transform
    blender_object_identifier: str | None = Field(default=None, max_length=192)


class ManifestUpdate(BaseModel):
    expected_package_revision: int = Field(ge=1)
    transforms: list[CompiledTransform] = Field(min_length=1)


router = APIRouter(prefix="/api", tags=["scene-packages"])
artifact_router = APIRouter(tags=["scene-artifacts"])


def _error(exc: SceneStoreError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


async def _save_upload(upload: UploadFile, destination: Path, allowed_types: set[str]) -> None:
    if upload.content_type not in allowed_types:
        raise HTTPException(415, f"unsupported Content-Type {upload.content_type!r}")
    total = 0
    with destination.open("wb") as stream:
        while block := await upload.read(1024 * 1024):
            total += len(block)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "upload is too large")
            stream.write(block)


def _extract_sam_archive(archive: Path, destination: Path) -> Path:
    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if len(entries) > MAX_ARCHIVE_FILES or sum(item.file_size for item in entries) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "SAM archive is too large")
            for item in entries:
                name = PurePosixPath(item.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in item.filename:
                    raise HTTPException(400, "SAM archive contains an unsafe path")
                if item.is_dir():
                    continue
                target = (destination / Path(*name.parts)).resolve()
                if destination.resolve() not in target.parents:
                    raise HTTPException(400, "SAM archive path escapes extraction root")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as exc:
        raise HTTPException(400, "invalid SAM ZIP archive") from exc
    candidates = list(destination.rglob("metadata.json"))
    if len(candidates) != 1:
        raise HTTPException(400, "SAM archive must contain exactly one metadata.json")
    return candidates[0].parent


@router.post("/scenes/import", status_code=201)
async def import_scene(
    sam_export: UploadFile = File(..., description="ZIP containing metadata.json, masks, previews, and quality reports"),
    unified_manifest: UploadFile = File(..., description="Unified V3 scene manifest JSON"),
    source_image: UploadFile = File(...),
) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix="scene-import-"))
    try:
        archive = temporary / "sam-export.zip"
        manifest = temporary / "unified-scene-plan.json"
        image = temporary / "source-image"
        await _save_upload(sam_export, archive, {"application/zip", "application/x-zip-compressed"})
        await _save_upload(unified_manifest, manifest, {"application/json"})
        await _save_upload(source_image, image, {"image/png", "image/jpeg", "image/webp"})
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(400, "Unified manifest must be a JSON object") from exc
        sam_dir = _extract_sam_archive(archive, temporary / "sam")
        try:
            scene = STORE.import_scene(sam_dir, manifest, image)
        except SceneStoreError as exc:
            raise _error(exc) from exc
        return scene.model_dump(mode="json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@router.get("/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict[str, Any]:
    try:
        return STORE.get(scene_id).model_dump(mode="json")
    except SceneStoreError as exc:
        raise _error(exc) from exc


@router.get("/scenes/{scene_id}/objects/{object_id}")
def get_object(scene_id: str, object_id: str) -> dict[str, Any]:
    try:
        scene = STORE.get(scene_id)
    except SceneStoreError as exc:
        raise _error(exc) from exc
    item = next((item for item in scene.semantic_objects if item.object_id == object_id), None)
    if item is None:
        raise HTTPException(404, "object not found")
    return item.model_dump(mode="json")


@router.patch("/scenes/{scene_id}/objects/{object_id}/label")
def update_label(scene_id: str, object_id: str, payload: LabelUpdate) -> dict[str, Any]:
    changes = {"semantic_label": payload.semantic_label, "display_name": payload.display_name or payload.semantic_label.replace("_", " ").title()}
    try:
        scene = STORE.update_object(scene_id, object_id, payload.expected_package_revision, payload.expected_object_version, changes)
    except SceneStoreError as exc:
        raise _error(exc) from exc
    return scene.model_dump(mode="json")


@router.patch("/scenes/{scene_id}/objects/{object_id}/approval")
def update_approval(scene_id: str, object_id: str, payload: ApprovalUpdate) -> dict[str, Any]:
    try:
        scene = STORE.update_object(scene_id, object_id, payload.expected_package_revision, payload.expected_object_version, {"approval_status": payload.approval_status.value})
    except SceneStoreError as exc:
        raise _error(exc) from exc
    return scene.model_dump(mode="json")


@router.post("/scenes/{scene_id}/objects/{object_id}/mask-revisions", status_code=201)
async def create_mask_revision(
    scene_id: str,
    object_id: str,
    resulting_mask: UploadFile = File(...),
    edit_delta: UploadFile = File(..., description="JSON brush-edit delta"),
    expected_package_revision: int = Form(...),
    expected_object_version: int = Form(...),
    expected_mask_revision: str = Form(...),
    author: str = Form(..., min_length=1, max_length=192),
    operation: EditOperation = Form(EditOperation.manual),
) -> dict[str, Any]:
    if not valid_id(object_id):
        raise HTTPException(400, "invalid object ID")
    temporary = Path(tempfile.mkdtemp(prefix="mask-revision-"))
    try:
        mask = temporary / "mask.png"
        delta_path = temporary / "delta.json"
        await _save_upload(resulting_mask, mask, {"image/png"})
        await _save_upload(edit_delta, delta_path, {"application/json"})
        try:
            delta = json.loads(delta_path.read_text(encoding="utf-8"))
            if not isinstance(delta, dict):
                raise ValueError
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(400, "edit_delta must be a JSON object") from exc
        try:
            scene = STORE.create_revision(
                scene_id, object_id, mask, delta,
                expected_package_revision=expected_package_revision,
                expected_object_version=expected_object_version,
                expected_mask_revision=expected_mask_revision,
                author=author, operation=operation.value,
            )
        except SceneStoreError as exc:
            raise _error(exc) from exc
        return scene.model_dump(mode="json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@router.get("/scenes/{scene_id}/objects/{object_id}/mask-revisions")
def revision_history(scene_id: str, object_id: str) -> dict[str, Any]:
    try:
        scene = STORE.get(scene_id)
    except SceneStoreError as exc:
        raise _error(exc) from exc
    if not any(item.object_id == object_id for item in scene.semantic_objects):
        raise HTTPException(404, "object not found")
    revisions = sorted((r for r in scene.mask_revisions if r.object_id == object_id), key=lambda r: r.revision_number)
    return {"scene_id": scene_id, "object_id": object_id, "package_revision": scene.package_revision, "revisions": [r.model_dump(mode="json") for r in revisions]}


@router.put("/scenes/{scene_id}/scene-manifest")
def update_scene_manifest(scene_id: str, payload: ManifestUpdate) -> dict[str, Any]:
    try:
        scene = STORE.import_transforms(scene_id, payload.expected_package_revision, [item.model_dump(mode="json") for item in payload.transforms])
    except SceneStoreError as exc:
        raise _error(exc) from exc
    return scene.model_dump(mode="json")


@router.get("/scenes/{scene_id}/reconstruction-status")
def reconstruction_status(scene_id: str) -> dict[str, Any]:
    try:
        scene = STORE.get(scene_id)
    except SceneStoreError as exc:
        raise _error(exc) from exc
    stale = [item.object_id for item in scene.semantic_objects if next(r for r in scene.mask_revisions if r.revision_id == item.mask_revision).geometry_invalidation_status in {"invalidated", "recompute_required"}]
    return {"scene_id": scene_id, "package_revision": scene.package_revision, "generation_status": scene.generation_status.model_dump(mode="json"), "stale_object_ids": stale}


def _artifact(kind: str, identifier: str) -> FileResponse:
    try:
        path, media_type = STORE.resolve_artifact(kind, identifier)
    except SceneStoreError as exc:
        raise _error(exc) from exc
    return FileResponse(path, media_type=media_type, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"})


@artifact_router.get("/artifacts/{kind}/{identifier:path}")
def get_artifact(kind: str, identifier: str) -> FileResponse:
    return _artifact(kind, identifier)


@router.get("/artifacts/{kind}/{identifier:path}")
def get_api_artifact(kind: str, identifier: str) -> FileResponse:
    return _artifact(kind, identifier)


@router.get("/scenes/{scene_id}/objects/{object_id}/mask")
def get_current_mask(scene_id: str, object_id: str) -> FileResponse:
    item = get_object(scene_id, object_id)
    return _artifact("masks", f"{object_id}/{item['mask_revision']}")


@router.get("/scenes/{scene_id}/objects/{object_id}/overlay")
def get_current_overlay(scene_id: str, object_id: str) -> FileResponse:
    item = get_object(scene_id, object_id)
    if not item["overlay_url"]:
        raise HTTPException(404, "overlay not available for current revision")
    return _artifact("mask-overlays", f"{object_id}/{item['mask_revision']}")
