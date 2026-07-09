"""Manual mask correction service helpers for segmentation workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .models import ManualMaskState, SegmentationObject, SegmentationWorkspace
from .store import MAX_IMAGE_PIXELS, SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


MAX_MANUAL_MASK_UPLOAD_BYTES = 16 * 1024 * 1024


def decode_uploaded_manual_mask(path: Path, *, width: int, height: int) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise SegmentationWorkspaceStoreError("unsupported mask dimensions", 413)
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise SegmentationWorkspaceStoreError("edited_mask must be a PNG image", 415)
            image.load()
            if image.mode not in {"1", "L"}:
                raise SegmentationWorkspaceStoreError("edited_mask must be a grayscale binary PNG", 415)
            if image.size != (width, height):
                raise SegmentationWorkspaceStoreError("edited_mask dimensions do not match the source image")
            pixels = np.asarray(image)
    except SegmentationWorkspaceStoreError:
        raise
    except Exception as exc:
        raise SegmentationWorkspaceStoreError("invalid edited_mask PNG", 400) from exc
    if pixels.ndim != 2:
        raise SegmentationWorkspaceStoreError("edited_mask must decode to a two-dimensional mask")
    if pixels.dtype == np.bool_:
        return pixels.astype(bool)
    unique = set(np.unique(pixels).tolist())
    if not unique.issubset({0, 255}):
        raise SegmentationWorkspaceStoreError("edited_mask must contain only 0 and 255 pixels")
    return pixels == 255


def save_manual_mask(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    *,
    edited_mask_path: Path,
    base_prompt_revision: int,
    base_candidate_index: int,
    expected_workspace_revision: int,
    expected_object_version: int,
    expected_manual_revision: int,
) -> dict[str, Any]:
    workspace = store.get_workspace(workspace_id)
    _object_by_id(workspace, object_id)
    edited_mask = decode_uploaded_manual_mask(
        edited_mask_path,
        width=workspace.image_dimensions.width,
        height=workspace.image_dimensions.height,
    )
    updated = store.save_manual_mask(
        workspace_id,
        object_id,
        edited_mask=edited_mask,
        base_prompt_revision=base_prompt_revision,
        base_candidate_index=base_candidate_index,
        expected_workspace_revision=expected_workspace_revision,
        expected_object_version=expected_object_version,
        expected_manual_revision=expected_manual_revision,
    )
    return manual_mask_response(updated, object_id)


def clear_manual_mask(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    *,
    expected_workspace_revision: int,
    expected_object_version: int,
    expected_manual_revision: int,
) -> dict[str, Any]:
    updated = store.clear_manual_mask(
        workspace_id,
        object_id,
        expected_workspace_revision=expected_workspace_revision,
        expected_object_version=expected_object_version,
        expected_manual_revision=expected_manual_revision,
    )
    return manual_mask_response(updated, object_id)


def manual_mask_response(workspace: SegmentationWorkspace, object_id: str) -> dict[str, Any]:
    item = _object_by_id(workspace, object_id)
    return {
        "workspace_revision": workspace.workspace_revision,
        "object_version": item.object_version,
        "object_id": item.object_id,
        "manual_mask": item.manual_mask.model_dump(mode="json"),
    }


def _object_by_id(workspace: SegmentationWorkspace, object_id: str) -> SegmentationObject:
    item = next((obj for obj in workspace.objects if obj.object_id == object_id), None)
    if item is None:
        raise SegmentationWorkspaceStoreError("object not found", 404)
    if item.manual_mask is None:
        item.manual_mask = ManualMaskState()
    return item
