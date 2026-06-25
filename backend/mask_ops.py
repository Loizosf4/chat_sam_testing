import base64
import io
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT_DIR / "data" / "masks"


class MaskOpsError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def merge_masks(mask_ids: list[str], label: str | None = None) -> dict[str, Any]:
    if not mask_ids:
        raise MaskOpsError("mask_ids must contain at least one mask_id.")

    masks = [_load_mask(mask_id) for mask_id in mask_ids]
    _validate_same_shape(masks)

    merged = np.logical_or.reduce(masks)
    return _save_mask_result(merged, label)


def subtract_masks(
    base_mask_id: str,
    subtract_mask_ids: list[str],
    label: str | None = None,
) -> dict[str, Any]:
    if not base_mask_id:
        raise MaskOpsError("base_mask_id is required.")
    if not subtract_mask_ids:
        raise MaskOpsError("subtract_mask_ids must contain at least one mask_id.")

    base_mask = _load_mask(base_mask_id)
    subtract_masks_loaded = [_load_mask(mask_id) for mask_id in subtract_mask_ids]
    _validate_same_shape([base_mask, *subtract_masks_loaded])

    subtract_union = np.logical_or.reduce(subtract_masks_loaded)
    cleaned = np.logical_and(base_mask, np.logical_not(subtract_union))
    return _save_mask_result(cleaned, label)


def fill_holes(mask_id: str, label: str | None = None) -> dict[str, Any]:
    mask = _load_mask(mask_id)
    filled = _fill_holes(mask)
    return _save_mask_result(filled, label)


def remove_small_components(
    mask_id: str,
    min_area: int = 100,
    label: str | None = None,
) -> dict[str, Any]:
    if min_area < 1:
        raise MaskOpsError("min_area must be at least 1.")

    mask = _load_mask(mask_id)
    cleaned = _remove_small_components(mask, min_area)
    return _save_mask_result(cleaned, label)


def smooth_mask(
    mask_id: str,
    kernel_size: int = 3,
    label: str | None = None,
) -> dict[str, Any]:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise MaskOpsError("kernel_size must be an odd integer greater than or equal to 3.")

    mask = _load_mask(mask_id)
    smoothed = _majority_filter(mask, kernel_size)
    return _save_mask_result(smoothed, label)


def _load_mask(mask_id: str) -> np.ndarray:
    clean_mask_id = (mask_id or "").strip()
    if not clean_mask_id:
        raise MaskOpsError("mask_id is required.")

    mask_path = MASK_DIR / f"{clean_mask_id}.png"
    if not mask_path.is_file():
        raise MaskOpsError(f"No mask found for mask_id '{clean_mask_id}'.")

    try:
        with Image.open(mask_path) as image:
            return np.array(image.convert("L")) > 127
    except Exception as exc:
        raise MaskOpsError(f"Failed to load mask '{clean_mask_id}': {exc}") from exc


def _validate_same_shape(masks: list[np.ndarray]) -> None:
    if not masks:
        raise MaskOpsError("At least one mask is required.")

    expected_shape = masks[0].shape
    for mask in masks[1:]:
        if mask.shape != expected_shape:
            raise MaskOpsError("All masks must have the same width and height.")


def _save_mask_result(mask: np.ndarray, label: str | None) -> dict[str, Any]:
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    mask_id = uuid4().hex
    mask_path = MASK_DIR / f"{mask_id}.png"
    mask_image = _mask_to_image(mask)
    mask_image.save(mask_path)

    png_buffer = io.BytesIO()
    mask_image.save(png_buffer, format="PNG")

    return {
        "mask_id": mask_id,
        "label": label,
        "area": int(mask.sum()),
        "bbox": _mask_bbox(mask),
        "png_base64": base64.b64encode(png_buffer.getvalue()).decode("ascii"),
    }


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]

    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    background = np.logical_not(mask)
    outside = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue_if_background(y: int, x: int) -> None:
        if background[y, x] and not outside[y, x]:
            outside[y, x] = True
            queue.append((y, x))

    for x in range(width):
        enqueue_if_background(0, x)
        enqueue_if_background(height - 1, x)

    for y in range(height):
        enqueue_if_background(y, 0)
        enqueue_if_background(y, width - 1)

    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= next_y < height and 0 <= next_x < width:
                enqueue_if_background(next_y, next_x)

    holes = np.logical_and(background, np.logical_not(outside))
    return np.logical_or(mask, holes)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    cleaned = np.zeros_like(mask, dtype=bool)

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            component = []
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                component.append((y, x))

                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and not visited[next_y, next_x]
                        and mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))

            if len(component) >= min_area:
                for y, x in component:
                    cleaned[y, x] = True

    return cleaned


def _majority_filter(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    height, width = mask.shape
    smoothed = np.zeros_like(mask, dtype=bool)
    threshold = (kernel_size * kernel_size) // 2 + 1

    for y in range(height):
        for x in range(width):
            window = padded[y:y + kernel_size, x:x + kernel_size]
            smoothed[y, x] = int(window.sum()) >= threshold

    return smoothed
