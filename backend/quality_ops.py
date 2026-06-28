import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend import mask_ops


DEFAULT_THRESHOLDS = {
    "small_area_percent": 0.1,
    "large_area_percent": 75.0,
    "many_components_count": 5,
    "small_component_percent": 0.05,
    "overlap_percent": 5.0,
    "bbox_iou": 0.9,
}


class QualityOpsError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_mask_quality_report(
    image_id: str,
    masks: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown diagnostics for a final mask set."""
    clean_image_id = _clean_image_id(image_id)
    if not masks:
        raise QualityOpsError("masks must contain at least one mask entry.")

    image_info = _get_image_info(clean_image_id)
    image_area = int(image_info["width"] * image_info["height"])
    expected_shape = (image_info["height"], image_info["width"])
    loaded_masks = []

    for index, mask_ref in enumerate(masks):
        if not isinstance(mask_ref, dict):
            raise QualityOpsError("Each mask entry must be an object.")

        mask_id = _clean_mask_id(mask_ref.get("mask_id"))
        label = _clean_label(mask_ref.get("label"), mask_id)
        mask = _load_mask(mask_id)
        if mask.shape != expected_shape:
            raise QualityOpsError(
                f"Mask '{mask_id}' dimensions do not match uploaded image dimensions."
            )

        components = _connected_components(mask)
        area = int(mask.sum())
        loaded_masks.append(
            {
                "index": index,
                "mask_id": mask_id,
                "label": label,
                "mask": mask,
                "metrics": _mask_metrics(mask=mask, components=components, image_area=image_area),
            }
        )

    warnings = []
    for mask_ref in loaded_masks:
        warnings.extend(_mask_warnings(mask_ref, image_area))

    pairwise_overlaps = _pairwise_overlaps(loaded_masks)
    for overlap in pairwise_overlaps:
        warnings.extend(_overlap_warnings(overlap))

    bbox_comparisons = _bbox_comparisons(loaded_masks)
    for comparison in bbox_comparisons:
        warnings.extend(_bbox_warnings(comparison))

    report = {
        "image_id": clean_image_id,
        "image": {
            "width": image_info["width"],
            "height": image_info["height"],
            "area": image_area,
            "original_filename": image_info["original_filename"],
        },
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "thresholds": DEFAULT_THRESHOLDS,
        "masks": [
            {
                "mask_id": mask_ref["mask_id"],
                "label": mask_ref["label"],
                **mask_ref["metrics"],
            }
            for mask_ref in loaded_masks
        ],
        "pairwise_overlaps": pairwise_overlaps,
        "bbox_comparisons": bbox_comparisons,
        "summary": {
            "mask_count": len(loaded_masks),
            "warnings": warnings,
        },
    }

    report_dir = _resolve_report_dir(output_dir)
    report_path = report_dir / "mask_quality_report.json"
    markdown_path = report_dir / "mask_quality_report.md"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "summary": report["summary"],
    }


def _mask_metrics(mask: np.ndarray, components: list[int], image_area: int) -> dict[str, Any]:
    area = int(mask.sum())
    bbox = mask_ops._mask_bbox(mask)
    small_component_min_area = _small_component_min_area(image_area)
    small_component_count = sum(1 for component_area in components if component_area < small_component_min_area)

    return {
        "area": area,
        "bbox": bbox,
        "percent_image_area": _percent(area, image_area),
        "connected_component_count": len(components),
        "largest_component_area": max(components, default=0),
        "small_component_count": small_component_count,
        "touches_image_border": _touches_image_border(mask),
    }


def _pairwise_overlaps(loaded_masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = []
    for left_index, left in enumerate(loaded_masks):
        for right in loaded_masks[left_index + 1:]:
            overlap_pixels = int(np.logical_and(left["mask"], right["mask"]).sum())
            left_area = left["metrics"]["area"]
            right_area = right["metrics"]["area"]
            smaller_area = max(1, min(left_area, right_area))
            union_area = int(np.logical_or(left["mask"], right["mask"]).sum())
            overlaps.append(
                {
                    "mask_a": left["mask_id"],
                    "label_a": left["label"],
                    "mask_b": right["mask_id"],
                    "label_b": right["label"],
                    "overlap_pixels": overlap_pixels,
                    "overlap_percent_of_smaller_mask": _percent(overlap_pixels, smaller_area),
                    "overlap_percent_of_union": _percent(overlap_pixels, union_area),
                }
            )

    return overlaps


def _bbox_comparisons(loaded_masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for left_index, left in enumerate(loaded_masks):
        for right in loaded_masks[left_index + 1:]:
            comparisons.append(
                {
                    "mask_a": left["mask_id"],
                    "label_a": left["label"],
                    "mask_b": right["mask_id"],
                    "label_b": right["label"],
                    "bbox_iou": _bbox_iou(left["metrics"]["bbox"], right["metrics"]["bbox"]),
                }
            )

    return comparisons


def _mask_warnings(mask_ref: dict[str, Any], image_area: int) -> list[dict[str, str]]:
    metrics = mask_ref["metrics"]
    warnings = []
    label = mask_ref["label"]

    if metrics["connected_component_count"] > DEFAULT_THRESHOLDS["many_components_count"]:
        warnings.append(
            _warning(
                label,
                f"Mask has many disconnected components ({metrics['connected_component_count']}).",
            )
        )

    if metrics["small_component_count"] > 0:
        warnings.append(
            _warning(
                label,
                f"Mask has {metrics['small_component_count']} small disconnected component(s).",
            )
        )

    if metrics["percent_image_area"] < DEFAULT_THRESHOLDS["small_area_percent"]:
        warnings.append(
            _warning(
                label,
                f"Mask area is unusually small ({metrics['percent_image_area']}% of image).",
            )
        )

    if metrics["percent_image_area"] > DEFAULT_THRESHOLDS["large_area_percent"]:
        warnings.append(
            _warning(
                label,
                f"Mask area is unusually large ({metrics['percent_image_area']}% of image).",
            )
        )

    if metrics["touches_image_border"]:
        warnings.append(_warning(label, "Mask touches image border."))

    return warnings


def _overlap_warnings(overlap: dict[str, Any]) -> list[dict[str, str]]:
    if overlap["overlap_pixels"] == 0:
        return []

    overlap_percent = overlap["overlap_percent_of_smaller_mask"]
    if overlap_percent <= DEFAULT_THRESHOLDS["overlap_percent"]:
        return []

    return [
        _warning(
            overlap["label_a"],
            f"Mask overlaps {overlap['label_b']} by {overlap_percent}% of the smaller mask.",
        ),
        _warning(
            overlap["label_b"],
            f"Mask overlaps {overlap['label_a']} by {overlap_percent}% of the smaller mask.",
        ),
    ]


def _bbox_warnings(comparison: dict[str, Any]) -> list[dict[str, str]]:
    if comparison["bbox_iou"] < DEFAULT_THRESHOLDS["bbox_iou"]:
        return []

    return [
        _warning(
            comparison["label_a"],
            f"Mask bbox is nearly identical to {comparison['label_b']} (IoU {comparison['bbox_iou']}).",
        ),
        _warning(
            comparison["label_b"],
            f"Mask bbox is nearly identical to {comparison['label_a']} (IoU {comparison['bbox_iou']}).",
        ),
    ]


def _connected_components(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    component_areas = []

    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not mask[start_y, start_x]:
                continue

            area = 0
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                area += 1

                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and not visited[next_y, next_x]
                        and mask[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))

            component_areas.append(area)

    return component_areas


def _touches_image_border(mask: np.ndarray) -> bool:
    if not mask.any():
        return False

    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _bbox_iou(left_bbox: list[int], right_bbox: list[int]) -> float:
    if left_bbox == [0, 0, 0, 0] or right_bbox == [0, 0, 0, 0]:
        return 0.0

    left_x1, left_y1, left_x2, left_y2 = left_bbox
    right_x1, right_y1, right_x2, right_y2 = right_bbox
    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(left_y1, right_y1)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(left_y2, right_y2)
    if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
        return 0.0

    intersection_area = (intersection_x2 - intersection_x1 + 1) * (intersection_y2 - intersection_y1 + 1)
    left_area = (left_x2 - left_x1 + 1) * (left_y2 - left_y1 + 1)
    right_area = (right_x2 - right_x1 + 1) * (right_y2 - right_y1 + 1)
    union_area = left_area + right_area - intersection_area
    return round(intersection_area / union_area, 4) if union_area else 0.0


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mask Quality Report",
        "",
        f"- Image ID: `{report['image_id']}`",
        f"- Mask count: {report['summary']['mask_count']}",
        f"- Generated at: {report['generated_at']}",
        "",
        "## Warnings",
        "",
    ]

    if report["summary"]["warnings"]:
        for warning in report["summary"]["warnings"]:
            lines.append(f"- {warning['severity']}: {warning['label']} - {warning['message']}")
    else:
        lines.append("- No warnings.")

    lines.extend(["", "## Masks", ""])
    for mask_ref in report["masks"]:
        lines.append(
            "- "
            f"{mask_ref['label']}: area {mask_ref['area']} px "
            f"({mask_ref['percent_image_area']}%), components {mask_ref['connected_component_count']}, "
            f"bbox {mask_ref['bbox']}"
        )

    lines.extend(["", "## Pairwise Overlaps", ""])
    if report["pairwise_overlaps"]:
        for overlap in report["pairwise_overlaps"]:
            lines.append(
                "- "
                f"{overlap['label_a']} / {overlap['label_b']}: "
                f"{overlap['overlap_pixels']} px, "
                f"{overlap['overlap_percent_of_smaller_mask']}% of smaller mask"
            )
    else:
        lines.append("- None.")

    return "\n".join(lines) + "\n"


def _resolve_report_dir(output_dir: str | None) -> Path:
    if output_dir:
        report_dir = Path(output_dir).expanduser()
        if not report_dir.is_absolute():
            report_dir = (Path.cwd() / report_dir).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        return report_dir

    try:
        report_dir = mask_ops._create_export_folder(None) / "quality"
    except mask_ops.MaskOpsError as exc:
        raise QualityOpsError(str(exc), status_code=exc.status_code) from exc

    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _get_image_info(image_id: str) -> dict[str, Any]:
    try:
        return mask_ops._get_image_info(image_id)
    except mask_ops.MaskOpsError as exc:
        raise QualityOpsError(str(exc), status_code=exc.status_code) from exc


def _load_mask(mask_id: str) -> np.ndarray:
    try:
        return mask_ops._load_mask(mask_id)
    except mask_ops.MaskOpsError as exc:
        raise QualityOpsError(str(exc), status_code=exc.status_code) from exc


def _small_component_min_area(image_area: int) -> int:
    return max(4, int(image_area * DEFAULT_THRESHOLDS["small_component_percent"] / 100.0))


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0

    return round((numerator / denominator) * 100.0, 2)


def _warning(label: str, message: str) -> dict[str, str]:
    return {
        "label": label,
        "severity": "warning",
        "message": message,
    }


def _clean_image_id(image_id: Any) -> str:
    clean_image_id = (str(image_id) if image_id is not None else "").strip()
    if not clean_image_id:
        raise QualityOpsError("image_id is required.")

    return clean_image_id


def _clean_mask_id(mask_id: Any) -> str:
    clean_mask_id = (str(mask_id) if mask_id is not None else "").strip()
    if not clean_mask_id:
        raise QualityOpsError("mask_id is required.")

    return clean_mask_id


def _clean_label(label: Any, fallback: str) -> str:
    clean_label = (str(label) if label is not None else "").strip()
    return clean_label or fallback
