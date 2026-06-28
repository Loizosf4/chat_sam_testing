from pathlib import Path
from shutil import copyfile
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from PIL import Image

from backend import mask_ops, preview_ops, quality_ops, sam_engine


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

mcp = FastMCP("Local SAM Mask Editor")


@mcp.tool()
def sam_health() -> dict:
    """Return local SAM server status without loading the model."""
    return sam_engine.get_status()


@mcp.tool()
def sam_load_model(
    checkpoint_path: str | None = None,
    model_type: str | None = None,
    device: str | None = None,
) -> dict:
    """Load the configured local SAM model."""
    try:
        return sam_engine.load_model(
            checkpoint_path=_empty_to_none(checkpoint_path),
            model_type=_empty_to_none(model_type),
            device=_empty_to_none(device),
        )
    except sam_engine.SamEngineError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def sam_register_image(image_path: str) -> dict:
    """Register an existing local image file for later SAM prediction."""
    source_path = Path(image_path).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()

    if not source_path.is_file():
        raise ValueError(f"Image path does not point to a valid local file: {source_path}")

    extension = source_path.suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type '{extension}'.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_id = uuid4().hex
    stored_filename = f"{image_id}{extension}"
    stored_path = IMAGE_DIR / stored_filename

    try:
        copyfile(source_path, stored_path)
        with Image.open(stored_path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise ValueError(f"Failed to register image: {exc}") from exc

    mask_ops.register_image(
        image_id=image_id,
        original_filename=source_path.name,
        stored_filename=stored_filename,
        width=width,
        height=height,
    )

    return {
        "image_id": image_id,
        "filename": source_path.name,
        "width": width,
        "height": height,
        "path": str(stored_path),
    }


@mcp.tool()
def sam_set_image(image_id: str) -> dict:
    """Set a registered image as the active image in the SAM predictor."""
    try:
        return sam_engine.set_image(image_id)
    except sam_engine.SamEngineError as exc:
        if "Call /load_model first" in str(exc):
            raise ValueError("SAM is not loaded. Call sam_load_model first.") from exc
        raise ValueError(str(exc)) from exc


@mcp.tool()
def sam_predict(
    image_id: str,
    points: list[dict[str, Any]] | None = None,
    box: dict[str, Any] | None = None,
    multimask_output: bool = True,
) -> dict:
    """Run SAM prediction and return mask metadata plus local PNG paths."""
    point_coords, point_labels = _parse_prompt_points(points)
    box_values = _parse_prompt_box(box)

    try:
        prediction = sam_engine.predict(
            image_id=image_id,
            points=point_coords,
            point_labels=point_labels,
            box=box_values,
            multimask_output=multimask_output,
        )
    except sam_engine.SamEngineError as exc:
        raise ValueError(_mcp_error(str(exc))) from exc

    return {
        "image_id": prediction["image_id"],
        "masks": [_mask_result_for_mcp(mask) for mask in prediction["masks"]],
    }


@mcp.tool()
def sam_merge_masks(mask_ids: list[str], label: str | None = None) -> dict:
    """Merge binary mask PNGs into a new object mask."""
    try:
        result = mask_ops.merge_masks(mask_ids=mask_ids, label=_empty_to_none(label))
    except mask_ops.MaskOpsError as exc:
        raise ValueError(str(exc)) from exc

    return _mask_result_for_mcp(result)


@mcp.tool()
def sam_subtract_masks(
    base_mask_id: str,
    subtract_mask_ids: list[str],
    label: str | None = None,
) -> dict:
    """Subtract one or more masks from a base mask."""
    try:
        result = mask_ops.subtract_masks(
            base_mask_id=base_mask_id,
            subtract_mask_ids=subtract_mask_ids,
            label=_empty_to_none(label),
        )
    except mask_ops.MaskOpsError as exc:
        raise ValueError(str(exc)) from exc

    return _mask_result_for_mcp(result)


@mcp.tool()
def sam_refine_mask(
    mask_id: str,
    operation: str,
    label: str | None = None,
    min_area: int = 100,
    kernel_size: int = 3,
) -> dict:
    """Refine a mask with fill_holes, remove_small_components, or smooth."""
    try:
        if operation == "fill_holes":
            result = mask_ops.fill_holes(mask_id=mask_id, label=_empty_to_none(label))
        elif operation == "remove_small_components":
            result = mask_ops.remove_small_components(
                mask_id=mask_id,
                min_area=min_area,
                label=_empty_to_none(label),
            )
        elif operation == "smooth":
            result = mask_ops.smooth_mask(
                mask_id=mask_id,
                kernel_size=kernel_size,
                label=_empty_to_none(label),
            )
        else:
            raise ValueError("operation must be fill_holes, remove_small_components, or smooth.")
    except mask_ops.MaskOpsError as exc:
        raise ValueError(str(exc)) from exc

    return _mask_result_for_mcp(result)


@mcp.tool()
def sam_export_masks(
    image_id: str,
    masks: list[dict[str, Any]],
    output_dir: str | None = None,
    include_previews: bool = False,
) -> dict:
    """Export selected masks as PNGs plus metadata.json."""
    if not masks:
        raise ValueError("masks must contain at least one final object mask.")

    try:
        result = mask_ops.export_masks(
            image_id=image_id,
            masks=[
                {
                    "mask_id": _string_value(mask.get("mask_id")),
                    "label": _optional_string_value(mask.get("label")),
                    "color": _optional_string_value(mask.get("color")),
                }
                for mask in masks
            ],
            output_dir=_empty_to_none(output_dir),
            include_previews=include_previews,
        )
    except mask_ops.MaskOpsError as exc:
        raise ValueError(str(exc)) from exc

    response = {
        "export_dir": result["export_path"],
        "files": result["files"],
        "metadata_path": result["metadata_path"],
    }

    if include_previews:
        response["preview_dir"] = result["preview_dir"]
        response["combined_preview_path"] = result["combined_preview_path"]

    return response


@mcp.tool()
def sam_preview_mask_overlay(
    image_id: str,
    mask_id: str,
    label: str | None = None,
    color: str = "#ff3366",
    alpha: float = 0.55,
    outline_width: int = 3,
    output_dir: str | None = None,
) -> dict:
    """Create a colored source-image overlay preview for one mask."""
    try:
        result = preview_ops.create_mask_overlay_preview(
            image_id=image_id,
            mask_id=mask_id,
            label=_empty_to_none(label),
            color=_empty_to_none(color) or "#ff3366",
            alpha=alpha,
            outline_width=outline_width,
            output_dir=_empty_to_none(output_dir),
        )
    except preview_ops.PreviewOpsError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "preview_path": result["path"],
        "image_id": result["image_id"],
        "mask_id": result["mask_id"],
        "label": result["label"],
    }


@mcp.tool()
def sam_preview_masks_overlay(
    image_id: str,
    masks: list[dict[str, Any]],
    alpha: float = 0.55,
    outline_width: int = 3,
    output_dir: str | None = None,
) -> dict:
    """Create a combined colored source-image overlay preview for multiple masks."""
    if not masks:
        raise ValueError("masks must contain at least one mask entry.")

    try:
        result = preview_ops.create_combined_overlay_preview(
            image_id=image_id,
            masks=[_preview_mask_ref_for_mcp(mask) for mask in masks],
            alpha=alpha,
            outline_width=outline_width,
            output_dir=_empty_to_none(output_dir),
        )
    except preview_ops.PreviewOpsError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "preview_path": result["path"],
        "image_id": result["image_id"],
        "mask_count": len(result["masks"]),
    }


@mcp.tool()
def sam_preview_candidate_contact_sheet(
    image_id: str,
    candidates: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict:
    """Create a contact sheet for visually comparing candidate masks."""
    if not candidates:
        raise ValueError("candidates must contain at least one candidate group.")

    try:
        result = preview_ops.create_candidate_contact_sheet(
            image_id=image_id,
            candidates=[_preview_candidate_group_for_mcp(candidate) for candidate in candidates],
            output_dir=_empty_to_none(output_dir),
        )
    except preview_ops.PreviewOpsError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "contact_sheet_path": result["path"],
        "image_id": result["image_id"],
        "groups": _unique_groups_from_panels(result["panels"]),
    }


@mcp.tool()
def sam_mask_quality_report(
    image_id: str,
    masks: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict:
    """Write mask quality diagnostics and return report paths plus summary warnings."""
    if not masks:
        raise ValueError("masks must contain at least one mask entry.")

    try:
        result = quality_ops.create_mask_quality_report(
            image_id=image_id,
            masks=[
                {
                    "mask_id": _string_value(mask.get("mask_id")) if isinstance(mask, dict) else "",
                    "label": _optional_string_value(mask.get("label")) if isinstance(mask, dict) else None,
                }
                for mask in masks
            ],
            output_dir=_empty_to_none(output_dir),
        )
    except quality_ops.QualityOpsError as exc:
        raise ValueError(str(exc)) from exc

    return {
        "report_path": result["report_path"],
        "markdown_path": result["markdown_path"],
        "summary": result["summary"],
    }


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _string_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None

    return _empty_to_none(str(value))


def _parse_prompt_points(
    points: list[dict[str, Any]] | None,
) -> tuple[list[list[float]], list[int]]:
    if not points:
        return [], []

    point_coords = []
    point_labels = []
    for point in points:
        try:
            x = float(point["x"])
            y = float(point["y"])
            label = int(point["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each point must include numeric x, y, and label fields.") from exc

        if label not in (0, 1):
            raise ValueError("Point labels must be 1 for positive or 0 for negative.")

        point_coords.append([x, y])
        point_labels.append(label)

    return point_coords, point_labels


def _parse_prompt_box(box: dict[str, Any] | None) -> list[float] | None:
    if box is None:
        return None

    try:
        return [
            float(box["x1"]),
            float(box["y1"]),
            float(box["x2"]),
            float(box["y2"]),
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("box must include numeric x1, y1, x2, and y2 fields.") from exc


def _mask_result_for_mcp(mask: dict[str, Any]) -> dict:
    result = {
        "mask_id": mask["mask_id"],
        "area": mask["area"],
        "bbox": mask["bbox"],
        "path": str(mask_ops.get_mask_path(mask["mask_id"])),
    }

    if "label" in mask:
        result["label"] = mask["label"]
    if "score" in mask:
        result["score"] = mask["score"]

    return result


def _preview_mask_ref_for_mcp(mask: dict[str, Any]) -> dict[str, str | None]:
    if not isinstance(mask, dict):
        raise ValueError("Each mask entry must be an object.")

    return {
        "mask_id": _string_value(mask.get("mask_id")),
        "label": _optional_string_value(mask.get("label")),
        "color": _optional_string_value(mask.get("color")),
    }


def _preview_candidate_group_for_mcp(candidate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError("Each candidate group must be an object.")

    masks = candidate.get("masks")
    if not isinstance(masks, list) or not masks:
        raise ValueError("Each candidate group must include at least one mask entry.")

    return {
        "group_label": _optional_string_value(candidate.get("group_label")),
        "masks": [_preview_mask_ref_for_mcp(mask) for mask in masks],
    }


def _unique_groups_from_panels(panels: list[dict[str, Any]]) -> list[str]:
    groups = []
    seen = set()
    for panel in panels:
        group_label = _string_value(panel.get("group_label"))
        if group_label and group_label not in seen:
            groups.append(group_label)
            seen.add(group_label)

    return groups


def _mcp_error(message: str) -> str:
    return message.replace("Call /load_model first", "Call sam_load_model first")


if __name__ == "__main__":
    mcp.run("stdio")
