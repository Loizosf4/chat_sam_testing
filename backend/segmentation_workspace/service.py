"""Interactive SAM prompting service for segmentation workspaces."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any

from backend import sam_engine

from .models import SamPromptPoint, SegmentationObject, SegmentationWorkspace
from .store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


MAX_PROMPT_POINTS = 256


class SamLogitsCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._items: OrderedDict[tuple[str, str, int, int], Any] = OrderedDict()

    def get(self, workspace_id: str, object_id: str, prompt_revision: int, candidate_index: int) -> Any | None:
        key = (workspace_id, object_id, prompt_revision, candidate_index)
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def replace_object_revision(
        self,
        workspace_id: str,
        object_id: str,
        prompt_revision: int,
        logits_by_candidate: list[Any],
    ) -> None:
        with self._lock:
            self.clear_object(workspace_id, object_id)
            for candidate_index, logits in enumerate(logits_by_candidate):
                self._items[(workspace_id, object_id, prompt_revision, candidate_index)] = logits
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear_object(self, workspace_id: str, object_id: str) -> None:
        with self._lock:
            for key in list(self._items):
                if key[0] == workspace_id and key[1] == object_id:
                    self._items.pop(key, None)

    def clear_workspace(self, workspace_id: str) -> None:
        with self._lock:
            for key in list(self._items):
                if key[0] == workspace_id:
                    self._items.pop(key, None)


LOGITS_CACHE = SamLogitsCache()


def _object_by_id(workspace: SegmentationWorkspace, object_id: str) -> SegmentationObject:
    item = next((obj for obj in workspace.objects if obj.object_id == object_id), None)
    if item is None:
        raise SegmentationWorkspaceStoreError("object not found", 404)
    return item


def _image_key(workspace: SegmentationWorkspace) -> str:
    return f"segmentation-workspace:{workspace.workspace_id}:{workspace.source_image.image_id}:{workspace.source_image.sha256}"


def prepare_workspace_sam(store: SegmentationWorkspaceStore, workspace_id: str) -> dict[str, Any]:
    workspace = store.get_workspace(workspace_id)
    source_path, _media_type = store.resolve_artifact("source-images", workspace.source_image.image_id)
    status = sam_engine.load_model()
    prepared = sam_engine.prepare_image_from_path(_image_key(workspace), source_path)
    return {
        "workspace_id": workspace.workspace_id,
        "image_id": workspace.source_image.image_id,
        "sam_ready": True,
        "model_type": status.get("model_type", ""),
        "device": status.get("device", ""),
        "width": prepared["width"],
        "height": prepared["height"],
    }


def predict_workspace_candidates(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    *,
    prompt_revision: int,
    points: list[SamPromptPoint],
    box: list[float] | None,
    base_prompt_revision: int | None,
    base_candidate_index: int | None,
    multimask_output: bool | None,
) -> dict[str, Any]:
    workspace = store.get_workspace(workspace_id)
    item = _object_by_id(workspace, object_id)
    _validate_prediction_state(workspace, item, prompt_revision)
    clean_points = _validate_prompt_geometry(workspace, points, box)
    source_path, _media_type = store.resolve_artifact("source-images", workspace.source_image.image_id)
    image_key = _image_key(workspace)
    sam_engine.load_model()

    mask_input = None
    if base_prompt_revision is not None and base_candidate_index is not None:
        mask_input = LOGITS_CACHE.get(workspace_id, object_id, base_prompt_revision, base_candidate_index)

    resolved_multimask = multimask_output if multimask_output is not None else mask_input is None
    prediction = sam_engine.predict_candidates(
        image_key,
        image_path=source_path,
        points=[[point.x, point.y] for point in clean_points],
        point_labels=[point.label for point in clean_points],
        box=box,
        mask_input=mask_input,
        multimask_output=resolved_multimask,
    )
    selected = _highest_score_index(prediction.scores)
    updated = store.persist_sam_prediction(
        workspace_id,
        object_id,
        prompt_revision=prompt_revision,
        points=[point.model_dump(mode="json") for point in clean_points],
        box=box,
        candidate_masks=list(prediction.masks),
        candidate_scores=prediction.scores,
        candidate_areas=prediction.areas,
        candidate_bboxes=prediction.bboxes,
        selected_candidate_index=selected,
        prepared_image_key=image_key,
    )
    logits_by_candidate = _split_logits(prediction.logits, len(prediction.scores))
    LOGITS_CACHE.replace_object_revision(workspace_id, object_id, prompt_revision, logits_by_candidate)
    updated_item = _object_by_id(updated, object_id)
    draft = updated_item.sam_draft
    return {
        "workspace_revision": updated.workspace_revision,
        "object_version": updated_item.object_version,
        "object_id": updated_item.object_id,
        "prompt_revision": draft.prompt_revision,
        "points": [point.model_dump(mode="json") for point in draft.points],
        "box": draft.box,
        "candidates": [candidate.model_dump(mode="json") for candidate in draft.candidates],
        "selected_candidate_index": draft.selected_candidate_index,
        "prediction_ms": prediction.elapsed_ms,
        "used_mask_input": mask_input is not None,
    }


def select_workspace_candidate(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    *,
    prompt_revision: int,
    candidate_index: int,
    expected_workspace_revision: int,
    expected_object_version: int,
) -> dict[str, Any]:
    workspace = store.select_sam_candidate(
        workspace_id,
        object_id,
        prompt_revision=prompt_revision,
        candidate_index=candidate_index,
        expected_workspace_revision=expected_workspace_revision,
        expected_object_version=expected_object_version,
    )
    item = _object_by_id(workspace, object_id)
    return {
        "workspace_revision": workspace.workspace_revision,
        "object_version": item.object_version,
        "object_id": item.object_id,
        "prompt_revision": item.sam_draft.prompt_revision,
        "selected_candidate_index": item.sam_draft.selected_candidate_index,
        "candidates": [candidate.model_dump(mode="json") for candidate in item.sam_draft.candidates],
    }


def clear_workspace_sam_draft(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    *,
    expected_workspace_revision: int,
    expected_object_version: int,
) -> dict[str, Any]:
    workspace = store.clear_sam_draft(
        workspace_id,
        object_id,
        expected_workspace_revision=expected_workspace_revision,
        expected_object_version=expected_object_version,
    )
    LOGITS_CACHE.clear_object(workspace_id, object_id)
    item = _object_by_id(workspace, object_id)
    return {
        "workspace_revision": workspace.workspace_revision,
        "object_version": item.object_version,
        "object_id": item.object_id,
        "sam_draft": item.sam_draft.model_dump(mode="json"),
    }


def _validate_prediction_state(
    workspace: SegmentationWorkspace,
    item: SegmentationObject,
    prompt_revision: int,
) -> None:
    if workspace.status == "finalized":
        raise SegmentationWorkspaceStoreError("workspace is finalized", 409)
    if item.status == "finalized":
        raise SegmentationWorkspaceStoreError("object is finalized", 409)
    if prompt_revision <= item.sam_draft.prompt_revision:
        raise SegmentationWorkspaceStoreError(
            f"stale prompt revision: incoming {prompt_revision}, current {item.sam_draft.prompt_revision}",
            409,
        )


def _validate_prompt_geometry(
    workspace: SegmentationWorkspace,
    points: list[SamPromptPoint],
    box: list[float] | None,
) -> list[SamPromptPoint]:
    if len(points) > MAX_PROMPT_POINTS:
        raise SegmentationWorkspaceStoreError("too many prompt points")
    if not points and box is None:
        raise SegmentationWorkspaceStoreError("at least one point or box is required")
    width = workspace.image_dimensions.width
    height = workspace.image_dimensions.height
    for point in points:
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            raise SegmentationWorkspaceStoreError("point coordinates must be finite")
        if point.x < 0 or point.y < 0 or point.x >= width or point.y >= height:
            raise SegmentationWorkspaceStoreError("point coordinates are outside the source image")
    if box is not None:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise SegmentationWorkspaceStoreError("box must contain four finite coordinates")
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            raise SegmentationWorkspaceStoreError("box coordinates are outside the source image or unordered")
    elif not any(point.label == 1 for point in points):
        raise SegmentationWorkspaceStoreError("at least one positive point is required unless a box is supplied")
    return points


def _highest_score_index(scores: list[float]) -> int:
    if not scores:
        raise SegmentationWorkspaceStoreError("SAM returned no scores", 500)
    return max(range(len(scores)), key=lambda index: scores[index])


def _split_logits(logits: Any, count: int) -> list[Any]:
    if logits is None:
        return [None] * count
    try:
        if len(logits) == count:
            return [_mask_input_shape(logits[index]) for index in range(count)]
    except TypeError:
        pass
    return [_mask_input_shape(logits) for _ in range(count)]


def _mask_input_shape(logit: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        return logit
    array = np.asarray(logit)
    if array.ndim == 2:
        return array[None, :, :]
    return array
