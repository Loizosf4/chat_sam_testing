import base64
import io
import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
MASK_DIR = ROOT_DIR / "data" / "masks"
EXPORT_DIR = ROOT_DIR / "data" / "exports"
IMAGE_INDEX_PATH = IMAGE_DIR / "images.json"


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


def register_image(
    image_id: str,
    original_filename: str,
    stored_filename: str,
    width: int,
    height: int,
) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_index = _read_image_index()
    image_index[image_id] = {
        "image_id": image_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "width": width,
        "height": height,
    }
    _write_image_index(image_index)


def export_masks(
    image_id: str,
    masks: list[dict[str, str | None]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    clean_image_id = (image_id or "").strip()
    if not clean_image_id:
        raise MaskOpsError("image_id is required.")
    if not masks:
        raise MaskOpsError("masks must contain at least one final object mask.")

    image_info = _get_image_info(clean_image_id)
    expected_shape = (image_info["height"], image_info["width"])
    exported_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    export_folder = _create_export_folder(output_dir)

    used_filenames: dict[str, int] = {}
    exported_masks = []
    exported_files = []

    for mask_ref in masks:
        mask_id = (mask_ref.get("mask_id") or "").strip()
        label = (mask_ref.get("label") or "").strip() or mask_id
        if not mask_id:
            raise MaskOpsError("Each mask entry must include mask_id.")

        mask = _load_mask(mask_id)
        if mask.shape != expected_shape:
            raise MaskOpsError(
                f"Mask '{mask_id}' dimensions do not match uploaded image dimensions."
            )

        filename = _unique_png_filename(label, used_filenames)
        output_path = export_folder / filename
        _mask_to_image(mask).save(output_path)

        exported_files.append(filename)
        exported_masks.append(
            {
                "label": label,
                "mask_id": mask_id,
                "filename": filename,
                "area": int(mask.sum()),
                "bbox": _mask_bbox(mask),
            }
        )

    metadata = {
        "image_id": clean_image_id,
        "original_filename": image_info["original_filename"],
        "width": image_info["width"],
        "height": image_info["height"],
        "exported_at": exported_at,
        "masks": exported_masks,
    }

    metadata_path = export_folder / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    exported_files.append("metadata.json")

    return {
        "export_path": str(export_folder),
        "files": exported_files,
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }


def get_mask_path(mask_id: str) -> Path:
    clean_mask_id = (mask_id or "").strip()
    if not clean_mask_id:
        raise MaskOpsError("mask_id is required.")

    mask_path = MASK_DIR / f"{clean_mask_id}.png"
    if not mask_path.is_file():
        raise MaskOpsError(f"No mask found for mask_id '{clean_mask_id}'.")

    return mask_path


def _load_mask(mask_id: str) -> np.ndarray:
    clean_mask_id = (mask_id or "").strip()
    if not clean_mask_id:
        raise MaskOpsError("mask_id is required.")

    mask_path = get_mask_path(clean_mask_id)

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


def _read_image_index() -> dict[str, Any]:
    if not IMAGE_INDEX_PATH.is_file():
        return {}

    try:
        return json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaskOpsError(f"Image index is invalid JSON: {exc}") from exc


def _write_image_index(image_index: dict[str, Any]) -> None:
    IMAGE_INDEX_PATH.write_text(json.dumps(image_index, indent=2), encoding="utf-8")


def _get_image_info(image_id: str) -> dict[str, Any]:
    image_index = _read_image_index()
    if image_id in image_index:
        info = image_index[image_id]
        image_path = IMAGE_DIR / info["stored_filename"]
        if not image_path.is_file():
            raise MaskOpsError(f"Uploaded image file is missing for image_id '{image_id}'.")
        return info

    matches = list(IMAGE_DIR.glob(f"{image_id}.*"))
    if not matches:
        raise MaskOpsError(f"No uploaded image found for image_id '{image_id}'.")

    image_path = matches[0]
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise MaskOpsError(f"Failed to read uploaded image '{image_id}': {exc}") from exc

    return {
        "image_id": image_id,
        "original_filename": image_path.name,
        "stored_filename": image_path.name,
        "width": width,
        "height": height,
    }


def _create_export_folder(output_dir: str | None = None) -> Path:
    if output_dir:
        export_folder = Path(output_dir).expanduser()
        if not export_folder.is_absolute():
            export_folder = (Path.cwd() / export_folder).resolve()
        export_folder.mkdir(parents=True, exist_ok=True)
        return export_folder

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = EXPORT_DIR / timestamp

    suffix = 2
    while export_folder.exists():
        export_folder = EXPORT_DIR / f"{timestamp}_{suffix}"
        suffix += 1

    export_folder.mkdir(parents=True)
    return export_folder


def _unique_png_filename(label: str, used_filenames: dict[str, int]) -> str:
    base_name = _safe_filename_stem(label)
    count = used_filenames.get(base_name, 0) + 1
    used_filenames[base_name] = count

    if count == 1:
        return f"{base_name}.png"

    return f"{base_name}_{count}.png"


def _safe_filename_stem(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip().lower())
    safe = safe.strip("_")
    return safe or "mask"
