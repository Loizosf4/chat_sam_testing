import math
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from backend import mask_ops


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
DEFAULT_ALPHA = 0.55
DEFAULT_OUTLINE_WIDTH = 3
DEFAULT_PALETTE = [
    "#ff3366",
    "#33ccff",
    "#ffcc33",
    "#66dd66",
    "#aa66ff",
    "#ff8833",
]


class PreviewOpsError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_mask_overlay_preview(
    image_id: str,
    mask_id: str,
    label: str | None = None,
    color: str = "#ff3366",
    alpha: float = DEFAULT_ALPHA,
    outline_width: int = DEFAULT_OUTLINE_WIDTH,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create a colored visual preview for one binary mask."""
    image = _load_source_image(image_id)
    mask = _load_mask(mask_id)
    _validate_mask_matches_image(mask_id, mask, image)

    clean_label = _clean_label(label, mask_id)
    color_rgb = _parse_hex_color(color)
    preview = _overlay_masks(
        image=image,
        masks=[
            {
                "mask": mask,
                "color": color_rgb,
            }
        ],
        alpha=_validate_alpha(alpha),
        outline_width=_validate_outline_width(outline_width),
    )

    preview_dir = _resolve_preview_dir(output_dir)
    output_path = _unique_output_path(
        preview_dir,
        f"{_safe_filename_stem(clean_label)}_{_short_id(mask_id)}_overlay.png",
    )
    preview.save(output_path)

    return {
        "image_id": _clean_image_id(image_id),
        "mask_id": _clean_mask_id(mask_id),
        "label": clean_label,
        "path": str(output_path),
        "width": preview.width,
        "height": preview.height,
    }


def create_combined_overlay_preview(
    image_id: str,
    masks: list[dict[str, Any]],
    alpha: float = DEFAULT_ALPHA,
    outline_width: int = DEFAULT_OUTLINE_WIDTH,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create a colored visual preview for multiple binary masks."""
    if not masks:
        raise PreviewOpsError("masks must contain at least one mask entry.")

    image = _load_source_image(image_id)
    overlay_entries = []
    result_masks = []

    for index, mask_ref in enumerate(masks):
        mask_id = _clean_mask_id(_string_value(mask_ref.get("mask_id")))
        label = _clean_label(_optional_string_value(mask_ref.get("label")), mask_id)
        color = _parse_hex_color(
            _optional_string_value(mask_ref.get("color")) or DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
        )
        mask = _load_mask(mask_id)
        _validate_mask_matches_image(mask_id, mask, image)

        overlay_entries.append({"mask": mask, "color": color})
        result_masks.append(
            {
                "mask_id": mask_id,
                "label": label,
                "color": _rgb_to_hex(color),
            }
        )

    preview = _overlay_masks(
        image=image,
        masks=overlay_entries,
        alpha=_validate_alpha(alpha),
        outline_width=_validate_outline_width(outline_width),
    )

    preview_dir = _resolve_preview_dir(output_dir)
    output_path = _unique_output_path(
        preview_dir,
        f"{_safe_filename_stem(_clean_image_id(image_id))}_combined_overlay.png",
    )
    preview.save(output_path)

    return {
        "image_id": _clean_image_id(image_id),
        "masks": result_masks,
        "path": str(output_path),
        "width": preview.width,
        "height": preview.height,
    }


def create_candidate_contact_sheet(
    image_id: str,
    candidates: list[dict[str, Any]],
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Create a contact sheet showing candidate masks over the source image."""
    if not candidates:
        raise PreviewOpsError("candidates must contain at least one candidate group.")

    image = _load_source_image(image_id)
    panels = []

    for group_index, group in enumerate(candidates):
        group_label = _optional_string_value(group.get("group_label")) or f"group_{group_index}"
        group_masks = group.get("masks")
        if not isinstance(group_masks, list) or not group_masks:
            raise PreviewOpsError("Each candidate group must include at least one mask entry.")

        for mask_index, mask_ref in enumerate(group_masks):
            if not isinstance(mask_ref, dict):
                raise PreviewOpsError("Each candidate mask entry must be an object.")

            mask_id = _clean_mask_id(_string_value(mask_ref.get("mask_id")))
            label = _clean_label(_optional_string_value(mask_ref.get("label")), f"candidate_{mask_index}")
            color = _parse_hex_color(
                _optional_string_value(mask_ref.get("color"))
                or DEFAULT_PALETTE[(group_index + mask_index) % len(DEFAULT_PALETTE)]
            )
            mask = _load_mask(mask_id)
            _validate_mask_matches_image(mask_id, mask, image)
            panels.append(
                {
                    "image": _draw_panel_label(
                        _overlay_masks(
                            image=image,
                            masks=[{"mask": mask, "color": color}],
                            alpha=DEFAULT_ALPHA,
                            outline_width=DEFAULT_OUTLINE_WIDTH,
                        ),
                        f"{group_label}: {label}",
                    ),
                    "mask_id": mask_id,
                    "label": label,
                    "group_label": group_label,
                }
            )

    contact_sheet = _build_contact_sheet([panel["image"] for panel in panels])
    preview_dir = _resolve_preview_dir(output_dir)
    output_path = _unique_output_path(
        preview_dir,
        f"{_safe_filename_stem(_clean_image_id(image_id))}_candidate_contact_sheet.png",
    )
    contact_sheet.save(output_path)

    return {
        "image_id": _clean_image_id(image_id),
        "path": str(output_path),
        "width": contact_sheet.width,
        "height": contact_sheet.height,
        "panels": [
            {
                "group_label": panel["group_label"],
                "label": panel["label"],
                "mask_id": panel["mask_id"],
            }
            for panel in panels
        ],
    }


def _load_source_image(image_id: str) -> Image.Image:
    clean_image_id = _clean_image_id(image_id)

    try:
        image_info = mask_ops._get_image_info(clean_image_id)
    except mask_ops.MaskOpsError as exc:
        raise PreviewOpsError(str(exc), status_code=exc.status_code) from exc

    image_path = IMAGE_DIR / image_info["stored_filename"]
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise PreviewOpsError(f"Failed to load source image '{clean_image_id}': {exc}") from exc


def _load_mask(mask_id: str) -> np.ndarray:
    clean_mask_id = _clean_mask_id(mask_id)
    try:
        return mask_ops._load_mask(clean_mask_id)
    except mask_ops.MaskOpsError as exc:
        raise PreviewOpsError(str(exc), status_code=exc.status_code) from exc


def _overlay_masks(
    image: Image.Image,
    masks: list[dict[str, Any]],
    alpha: float,
    outline_width: int,
) -> Image.Image:
    output = np.array(image.convert("RGB"), dtype=np.float32)

    for mask_ref in masks:
        mask = mask_ref["mask"]
        color = np.array(mask_ref["color"], dtype=np.float32)
        output[mask] = (output[mask] * (1.0 - alpha)) + (color * alpha)

    for mask_ref in masks:
        if outline_width < 1:
            continue

        outline = _mask_outline(mask_ref["mask"], outline_width)
        output[outline] = np.array(mask_ref["color"], dtype=np.float32)

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def _mask_outline(mask: np.ndarray, outline_width: int) -> np.ndarray:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    filter_size = outline_width * 2 + 1
    dilated = np.array(mask_image.filter(ImageFilter.MaxFilter(filter_size))) > 127
    eroded = np.array(mask_image.filter(ImageFilter.MinFilter(filter_size))) > 127
    return np.logical_and(dilated, np.logical_not(eroded))


def _build_contact_sheet(panel_images: list[Image.Image]) -> Image.Image:
    if not panel_images:
        raise PreviewOpsError("At least one contact sheet panel is required.")

    panel_width, panel_height = panel_images[0].size
    count = len(panel_images)
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    gutter = max(8, min(panel_width, panel_height) // 80)

    sheet_width = (columns * panel_width) + ((columns + 1) * gutter)
    sheet_height = (rows * panel_height) + ((rows + 1) * gutter)
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(24, 24, 24))

    for index, panel in enumerate(panel_images):
        row = index // columns
        column = index % columns
        x = gutter + column * (panel_width + gutter)
        y = gutter + row * (panel_height + gutter)
        sheet.paste(panel, (x, y))

    return sheet


def _draw_panel_label(image: Image.Image, label: str) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    font = ImageFont.load_default()
    label_text = label.strip() or "candidate"
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    padding = max(6, min(panel.size) // 80)
    x = padding
    y = padding
    box = (
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding,
    )
    draw.rectangle(box, fill=(0, 0, 0, 180))
    draw.text((x, y), label_text, fill=(255, 255, 255, 255), font=font)
    return panel


def _resolve_preview_dir(output_dir: str | None) -> Path:
    if output_dir:
        preview_dir = Path(output_dir).expanduser()
        if not preview_dir.is_absolute():
            preview_dir = (Path.cwd() / preview_dir).resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        return preview_dir

    try:
        export_folder = mask_ops._create_export_folder(None)
    except mask_ops.MaskOpsError as exc:
        raise PreviewOpsError(str(exc), status_code=exc.status_code) from exc

    preview_dir = export_folder / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir


def _unique_output_path(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate

    return directory / f"{stem}_{uuid4().hex}{suffix}"


def _validate_mask_matches_image(mask_id: str, mask: np.ndarray, image: Image.Image) -> None:
    expected_shape = (image.height, image.width)
    if mask.shape != expected_shape:
        raise PreviewOpsError(
            f"Mask '{mask_id}' dimensions do not match source image dimensions."
        )


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    if not isinstance(color, str):
        raise PreviewOpsError("color must be a hex string such as '#ff3366'.")

    clean_color = color.strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", clean_color):
        raise PreviewOpsError("color must be a 6-digit hex string such as '#ff3366'.")

    return (
        int(clean_color[1:3], 16),
        int(clean_color[3:5], 16),
        int(clean_color[5:7], 16),
    )


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _validate_alpha(alpha: float) -> float:
    try:
        parsed_alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise PreviewOpsError("alpha must be a number between 0 and 1.") from exc

    if parsed_alpha < 0 or parsed_alpha > 1:
        raise PreviewOpsError("alpha must be between 0 and 1.")

    return parsed_alpha


def _validate_outline_width(outline_width: int) -> int:
    try:
        parsed_width = int(outline_width)
    except (TypeError, ValueError) as exc:
        raise PreviewOpsError("outline_width must be a non-negative integer.") from exc

    if parsed_width < 0:
        raise PreviewOpsError("outline_width must be a non-negative integer.")

    return parsed_width


def _clean_image_id(image_id: str) -> str:
    clean_image_id = (image_id or "").strip()
    if not clean_image_id:
        raise PreviewOpsError("image_id is required.")

    return clean_image_id


def _clean_mask_id(mask_id: str) -> str:
    clean_mask_id = (mask_id or "").strip()
    if not clean_mask_id:
        raise PreviewOpsError("mask_id is required.")

    return clean_mask_id


def _clean_label(label: str | None, fallback: str) -> str:
    return (label or "").strip() or fallback


def _safe_filename_stem(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip().lower())
    safe = safe.strip("_")
    return safe or "preview"


def _short_id(value: str) -> str:
    clean_value = re.sub(r"[^A-Za-z0-9]+", "", value)
    return clean_value[:8] or "mask"


def _string_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value)


def _optional_string_value(value: Any) -> str | None:
    if value is None:
        return None

    stripped = str(value).strip()
    return stripped or None
