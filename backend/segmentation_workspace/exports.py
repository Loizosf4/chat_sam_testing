"""Immutable segmentation workspace export snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend import preview_ops, quality_ops

from .models import SegmentationExportRecord, SegmentationWorkspace


EXPORT_KIND = "workspace-exports"
EXPORT_PREFIX = "/api/segmentation-artifacts/workspace-exports/"
ARTIFACT_PREFIX = "/api/segmentation-artifacts/"
RESERVED_WINDOWS_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class SegmentationExportError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_export_snapshot(
    *,
    workspace: SegmentationWorkspace,
    source_image_path: Path,
    resolve_artifact: Callable[[str], tuple[Path, str]],
    export_dir: Path,
    export_id: str,
    created_at: datetime,
    published_workspace_revision: int,
) -> tuple[SegmentationExportRecord, dict[str, dict[str, str]]]:
    """Write a complete immutable export into export_dir and return its record/index entries."""
    effective_masks = resolve_effective_masks(workspace, resolve_artifact)
    if not effective_masks:
        raise SegmentationExportError("at least one semantic object is required for export")

    export_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = export_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    image = _load_source_image(
        source_image_path,
        width=workspace.image_dimensions.width,
        height=workspace.image_dimensions.height,
    )
    used_filenames: set[str] = set()
    metadata_masks: list[dict[str, Any]] = []
    record_masks: list[dict[str, Any]] = []
    quality_masks: list[dict[str, Any]] = []
    overlay_entries: list[dict[str, Any]] = []

    for index, entry in enumerate(effective_masks):
        color = preview_ops.DEFAULT_PALETTE[index % len(preview_ops.DEFAULT_PALETTE)]
        color_rgb = _hex_to_rgb(color)
        filename = _unique_mask_filename(entry["semantic_label"], entry["object_id"], used_filenames)
        mask_path = export_dir / filename
        _write_binary_png(mask_path, entry["mask"])
        mask_hash = sha256_file(mask_path)
        preview_filename = f"{Path(filename).stem}_overlay.png"
        preview_path = preview_dir / preview_filename
        preview_ops._overlay_masks(
            image=image,
            masks=[{"mask": entry["mask"], "color": color_rgb}],
            alpha=preview_ops.DEFAULT_ALPHA,
            outline_width=preview_ops.DEFAULT_OUTLINE_WIDTH,
        ).save(preview_path, format="PNG")

        metadata_masks.append(
            {
                "label": entry["semantic_label"],
                "mask_id": entry["object_id"],
                "filename": filename,
                "area": entry["area_pixels"],
                "bbox": entry["bbox_xyxy"],
                "color": color,
                "preview_path": f"previews/{preview_filename}",
            }
        )
        record_masks.append(
            {
                "object_id": entry["object_id"],
                "object_version": entry["object_version"],
                "semantic_label": entry["semantic_label"],
                "display_name": entry["display_name"],
                "source_kind": entry["source_kind"],
                "source_prompt_revision": entry["source_prompt_revision"],
                "source_candidate_index": entry["source_candidate_index"],
                "source_manual_revision": entry["source_manual_revision"],
                "filename": filename,
                "mask_url": _artifact_url(workspace.workspace_id, export_id, f"masks/{entry['object_id']}"),
                "preview_url": _artifact_url(workspace.workspace_id, export_id, f"previews/{entry['object_id']}"),
                "mask_sha256": mask_hash,
                "area_pixels": entry["area_pixels"],
                "bbox_xyxy": entry["bbox_xyxy"],
            }
        )
        quality_masks.append(
            {
                "mask_id": entry["object_id"],
                "label": entry["semantic_label"],
                "mask": entry["mask"],
            }
        )
        overlay_entries.append({"mask": entry["mask"], "color": color_rgb})

    combined_preview_path = preview_dir / "all_masks_overlay.png"
    preview_ops._overlay_masks(
        image=image,
        masks=overlay_entries,
        alpha=preview_ops.DEFAULT_ALPHA,
        outline_width=preview_ops.DEFAULT_OUTLINE_WIDTH,
    ).save(combined_preview_path, format="PNG")

    exported_at = created_at.replace(microsecond=0).isoformat()
    metadata = {
        "image_id": workspace.source_image.image_id,
        "original_filename": workspace.source_image.original_filename,
        "width": workspace.image_dimensions.width,
        "height": workspace.image_dimensions.height,
        "exported_at": exported_at,
        "masks": metadata_masks,
        "preview_dir": "previews",
        "combined_preview_path": "previews/all_masks_overlay.png",
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    quality_ops.create_mask_quality_report_from_arrays(
        image_id=workspace.source_image.image_id,
        original_filename=workspace.source_image.original_filename,
        width=workspace.image_dimensions.width,
        height=workspace.image_dimensions.height,
        masks=quality_masks,
        output_dir=export_dir,
    )
    quality_json_path = export_dir / "mask_quality_report.json"
    quality_markdown_path = export_dir / "mask_quality_report.md"
    quality_report = json.loads(quality_json_path.read_text(encoding="utf-8"))
    warning_count = len(quality_report["summary"]["warnings"])

    archive_path = export_dir / "segmentation-export.zip"
    _write_export_zip(export_dir, archive_path)
    archive_hash = sha256_file(archive_path)

    record = SegmentationExportRecord(
        export_id=export_id,
        created_at=created_at,
        created_from_workspace_revision=workspace.workspace_revision,
        published_workspace_revision=published_workspace_revision,
        masks=record_masks,
        metadata_url=_artifact_url(workspace.workspace_id, export_id, "metadata"),
        quality_report_url=_artifact_url(workspace.workspace_id, export_id, "quality-json"),
        quality_markdown_url=_artifact_url(workspace.workspace_id, export_id, "quality-markdown"),
        combined_preview_url=_artifact_url(workspace.workspace_id, export_id, "combined-preview"),
        archive_url=_artifact_url(workspace.workspace_id, export_id, "archive"),
        archive_sha256=archive_hash,
        warning_count=warning_count,
        mask_count=len(record_masks),
        total_mask_area=sum(mask["area_pixels"] for mask in record_masks),
    )
    entries = _artifact_entries(workspace.workspace_id, export_id, record, export_dir)
    return record, entries


def resolve_effective_masks(
    workspace: SegmentationWorkspace,
    resolve_artifact: Callable[[str], tuple[Path, str]],
) -> list[dict[str, Any]]:
    resolved = []
    for item in workspace.objects:
        draft = item.sam_draft
        manual = item.manual_mask
        source_kind = "sam_candidate"
        source_manual_revision = 0
        if manual.manual_revision > 0:
            if manual.base_prompt_revision != draft.prompt_revision or manual.base_candidate_index != draft.selected_candidate_index:
                raise SegmentationExportError(
                    f"manual mask base mismatch for object {item.object_id} ({item.display_name})"
                )
            if not manual.composite_mask_url:
                raise SegmentationExportError(f"manual composite is missing for object {item.object_id} ({item.display_name})")
            artifact_key = _artifact_key_from_url(manual.composite_mask_url)
            source_kind = "manual_composite"
            source_manual_revision = manual.manual_revision
            source_prompt_revision = manual.base_prompt_revision or 0
            source_candidate_index = manual.base_candidate_index or 0
        else:
            if draft.selected_candidate_index is None:
                raise SegmentationExportError(f"object {item.object_id} ({item.display_name}) has no selected SAM candidate")
            selected = next(
                (candidate for candidate in draft.candidates if candidate.candidate_index == draft.selected_candidate_index),
                None,
            )
            if selected is None:
                raise SegmentationExportError(f"selected SAM candidate is missing for object {item.object_id} ({item.display_name})")
            artifact_key = _artifact_key_from_url(selected.mask_url)
            source_prompt_revision = draft.prompt_revision
            source_candidate_index = selected.candidate_index

        artifact_path, media_type = resolve_artifact(artifact_key)
        if media_type != "image/png":
            raise SegmentationExportError(f"effective mask is not a PNG for object {item.object_id} ({item.display_name})", 500)
        mask = _load_binary_png(
            artifact_path,
            width=workspace.image_dimensions.width,
            height=workspace.image_dimensions.height,
            object_id=item.object_id,
            display_name=item.display_name,
        )
        area = int(mask.sum())
        if area <= 0:
            raise SegmentationExportError(f"effective mask is empty for object {item.object_id} ({item.display_name})")
        resolved.append(
            {
                "object_id": item.object_id,
                "object_version": item.object_version,
                "semantic_label": item.semantic_label,
                "display_name": item.display_name,
                "source_kind": source_kind,
                "source_prompt_revision": source_prompt_revision,
                "source_candidate_index": source_candidate_index,
                "source_manual_revision": source_manual_revision,
                "mask": mask,
                "mask_sha256": sha256_file(artifact_path),
                "area_pixels": area,
                "bbox_xyxy": _bbox(mask),
            }
        )
    return resolved


def export_is_stale(
    workspace: SegmentationWorkspace,
    record: SegmentationExportRecord,
    resolve_artifact: Callable[[str], tuple[Path, str]],
) -> bool:
    if {item.object_id for item in workspace.objects} != {mask.object_id for mask in record.masks}:
        return True
    try:
        current = {item["object_id"]: item for item in resolve_effective_masks(workspace, resolve_artifact)}
    except Exception:
        return True
    for mask in record.masks:
        item = current.get(mask.object_id)
        if item is None:
            return True
        if (
            item["object_version"] != mask.object_version
            or item["semantic_label"] != mask.semantic_label
            or item["source_kind"] != mask.source_kind
            or item["source_prompt_revision"] != mask.source_prompt_revision
            or item["source_candidate_index"] != mask.source_candidate_index
            or item["source_manual_revision"] != mask.source_manual_revision
            or item["mask_sha256"] != mask.mask_sha256
        ):
            return True
    return False


def _artifact_entries(
    workspace_id: str,
    export_id: str,
    record: SegmentationExportRecord,
    export_dir: Path,
) -> dict[str, dict[str, str]]:
    entries = {
        f"{EXPORT_KIND}/{workspace_id}/{export_id}/archive": _entry(export_dir / "segmentation-export.zip", f"exports/{export_id}/segmentation-export.zip", "application/zip"),
        f"{EXPORT_KIND}/{workspace_id}/{export_id}/metadata": _entry(export_dir / "metadata.json", f"exports/{export_id}/metadata.json", "application/json"),
        f"{EXPORT_KIND}/{workspace_id}/{export_id}/quality-json": _entry(export_dir / "mask_quality_report.json", f"exports/{export_id}/mask_quality_report.json", "application/json"),
        f"{EXPORT_KIND}/{workspace_id}/{export_id}/quality-markdown": _entry(export_dir / "mask_quality_report.md", f"exports/{export_id}/mask_quality_report.md", "text/markdown; charset=utf-8"),
        f"{EXPORT_KIND}/{workspace_id}/{export_id}/combined-preview": _entry(export_dir / "previews" / "all_masks_overlay.png", f"exports/{export_id}/previews/all_masks_overlay.png", "image/png"),
    }
    for mask in record.masks:
        entries[f"{EXPORT_KIND}/{workspace_id}/{export_id}/masks/{mask.object_id}"] = _entry(
            export_dir / mask.filename,
            f"exports/{export_id}/{mask.filename}",
            "image/png",
        )
        preview_name = Path(mask.preview_url).name
        entries[f"{EXPORT_KIND}/{workspace_id}/{export_id}/previews/{mask.object_id}"] = _entry(
            export_dir / "previews" / f"{Path(mask.filename).stem}_overlay.png",
            f"exports/{export_id}/previews/{Path(mask.filename).stem}_overlay.png",
            "image/png",
        )
        assert preview_name == mask.object_id
    return entries


def _entry(path: Path, relative_path: str, media_type: str) -> dict[str, str]:
    return {
        "path": relative_path,
        "media_type": media_type,
        "sha256": sha256_file(path),
    }


def _write_export_zip(export_dir: Path, archive_path: Path) -> None:
    members = [
        path
        for path in export_dir.rglob("*")
        if path.is_file() and path.resolve() != archive_path.resolve()
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(members, key=lambda item: item.relative_to(export_dir).as_posix()):
            relative = path.relative_to(export_dir).as_posix()
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise SegmentationExportError("unsafe export ZIP member", 500)
            archive.write(path, relative)


def _load_source_image(path: Path, *, width: int, height: int) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise SegmentationExportError("source image dimensions do not match workspace metadata", 500)
            return image.convert("RGB")
    except SegmentationExportError:
        raise
    except Exception as exc:
        raise SegmentationExportError("source image could not be loaded", 500) from exc


def _load_binary_png(path: Path, *, width: int, height: int, object_id: str, display_name: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise SegmentationExportError(f"effective mask is not a PNG for object {object_id} ({display_name})", 500)
            if image.mode not in {"1", "L"}:
                raise SegmentationExportError(f"effective mask must be grayscale for object {object_id} ({display_name})", 400)
            if image.size != (width, height):
                raise SegmentationExportError(f"effective mask dimensions do not match source image for object {object_id} ({display_name})")
            pixels = np.asarray(image)
    except SegmentationExportError:
        raise
    except Exception as exc:
        raise SegmentationExportError(f"effective mask could not be decoded for object {object_id} ({display_name})", 400) from exc
    if pixels.ndim != 2:
        raise SegmentationExportError(f"effective mask must be two-dimensional for object {object_id} ({display_name})")
    unique = set(np.unique(pixels).tolist())
    if not unique.issubset({0, 255, False, True}):
        raise SegmentationExportError(f"effective mask is not binary for object {object_id} ({display_name})")
    return pixels.astype(bool) if pixels.dtype == np.bool_ else pixels == 255


def _write_binary_png(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, format="PNG")


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _artifact_key_from_url(url: str) -> str:
    if not url.startswith(ARTIFACT_PREFIX):
        raise SegmentationExportError("effective mask URL is not a segmentation artifact URL")
    return url[len(ARTIFACT_PREFIX) :]


def _artifact_url(workspace_id: str, export_id: str, suffix: str) -> str:
    return f"{EXPORT_PREFIX}{workspace_id}/{export_id}/{suffix}"


def _unique_mask_filename(label: str, object_id: str, used: set[str]) -> str:
    stem = _safe_filename_stem(label)
    candidate = f"{stem}.png"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    suffix = re.sub(r"[^A-Za-z0-9]", "", object_id)[:8] or "object"
    candidate = f"{stem}_{suffix}.png"
    counter = 2
    while candidate.lower() in used:
        candidate = f"{stem}_{suffix}_{counter}.png"
        counter += 1
    used.add(candidate.lower())
    return candidate


def _safe_filename_stem(label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip().lower()).strip("._-")
    if not stem or stem in RESERVED_WINDOWS_NAMES:
        stem = "mask"
    return stem[:96].rstrip("._-") or "mask"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
