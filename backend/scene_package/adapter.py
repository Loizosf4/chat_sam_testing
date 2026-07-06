"""Adapters from final SAM exports and Unified V3.1.1 clean manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from .models import (
    ArtifactReference,
    CameraCandidate,
    CameraType,
    CoordinateSystem,
    GenerationState,
    GenerationStatus,
    GeometryInvalidationStatus,
    ImageDimensions,
    MaskQualityReport,
    MaskRevision,
    OcclusionState,
    Scene,
    SemanticObject,
    SourceImage,
    StructuralRoomProxy,
    SupportType,
    Transform,
)
from .serialization import dump_scene


REVISION_NAMESPACE = UUID("36b9ab3a-6035-58a8-b544-283304f1d15b")


class SceneAdapterError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneAdapterError(f"cannot read JSON input {path.name!r}") from exc
    if not isinstance(value, dict):
        raise SceneAdapterError(f"JSON input {path.name!r} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_url(kind: str, identifier: str) -> str:
    return f"/artifacts/{kind}/{identifier}"


def _mask_revision_id(object_id: str, mask_hash: str) -> str:
    return uuid5(REVISION_NAMESPACE, f"{object_id}:{mask_hash}:1").hex


def _as_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _display_name(label: str) -> str:
    return label.replace("_", " ").strip().title()


def _support_type(value: Any) -> SupportType:
    try:
        return SupportType(str(value))
    except ValueError:
        return SupportType.unknown


def _orientation_confidence(item: dict[str, Any]) -> float:
    normal_frame = item.get("normal_frame")
    if isinstance(normal_frame, dict):
        value = normal_frame.get("orientation_confidence")
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    return max(0.0, min(1.0, float(item.get("final_pose_confidence", 0.0))))


def _quality_warnings(item: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    classification = str(item.get("confidence_classification", ""))
    if classification not in {"", "automatic_high_confidence"}:
        warnings.append(classification)
    placement = str(item.get("placement_classification", ""))
    if placement in {"placement_review_required", "placement_invalid"}:
        warnings.append(placement)
    for gate in item.get("placement_hard_gates") or []:
        warnings.append(f"placement_gate:{gate}")
    if item.get("collision_warnings"):
        warnings.append("collision_warning")
    return sorted(set(warnings))


def _room_proxy(raw: dict[str, Any]) -> StructuralRoomProxy:
    source_semantic = str(raw.get("semantic", "room_boundary"))
    if source_semantic == "floor":
        semantic = "floor"
    elif "wall" in source_semantic:
        semantic = "wall"
    elif source_semantic == "ceiling":
        semantic = "ceiling"
    else:
        semantic = "room_boundary"
    return StructuralRoomProxy(
        proxy_id=str(raw["plane_id"]),
        semantic=semantic,
        display_name=_display_name(source_semantic),
        transform=Transform(
            center=raw["center"],
            dimensions=raw["dimensions"],
            quaternion=raw["quaternion_wxyz"],
        ),
        confidence=float(raw.get("confidence", 0.0)),
        metadata={
            "source_semantic": source_semantic,
            "extent_policy": raw.get("extent_policy"),
        },
    )


def _camera(raw: dict[str, Any]) -> CameraCandidate:
    camera_type = CameraType(str(raw.get("type", "perspective")))
    transform = raw.get("matrix_world")
    if not transform:
        raise SceneAdapterError(f"camera {raw.get('camera_id')!r} has no matrix_world")
    metadata = {
        key: raw[key]
        for key in (
            "canonical_to_camera_transform",
            "framing_shift_pixels",
            "orientation_correction_degrees_xyz",
            "focal_scale",
            "structural_reprojection_rmse_pixels",
            "orthographic_scale",
        )
        if key in raw
    }
    return CameraCandidate(
        camera_id=str(raw["camera_id"]),
        camera_type=camera_type,
        transform=transform,
        normalized_intrinsics=raw.get("normalized_intrinsics"),
        field_of_view_x_degrees=raw.get("field_of_view_x_degrees"),
        confidence=float(raw.get("confidence", 0.0)),
        provisional=bool(raw.get("provisional", True)),
        metadata=metadata,
    )


def adapt_scene_package(
    sam_metadata_path: str | Path,
    unified_manifest_path: str | Path,
    *,
    source_image_path: str | Path | None = None,
) -> Scene:
    """Combine immutable source artifacts into one validated scene package."""
    sam_path = Path(sam_metadata_path)
    unified_path = Path(unified_manifest_path)
    sam = _read_json(sam_path)
    unified = _read_json(unified_path)
    if str(unified.get("mode")) != "clean_reconstruction":
        raise SceneAdapterError("Unified input must be a clean_reconstruction manifest")
    if int(unified.get("semantic_object_count", -1)) != len(
        unified.get("semantic_objects", [])
    ):
        raise SceneAdapterError("Unified semantic_object_count does not match its objects")

    sam_masks = sam.get("masks")
    if not isinstance(sam_masks, list):
        raise SceneAdapterError("SAM metadata must contain a masks array")
    sam_by_id: dict[str, dict[str, Any]] = {}
    for raw in sam_masks:
        if not isinstance(raw, dict) or not raw.get("mask_id"):
            raise SceneAdapterError("every SAM mask must contain mask_id")
        object_id = str(raw["mask_id"])
        if object_id in sam_by_id:
            raise SceneAdapterError(f"duplicate SAM mask_id {object_id!r}")
        sam_by_id[object_id] = raw

    timestamp = _as_timestamp(sam.get("exported_at"))
    revisions: list[MaskRevision] = []
    objects: list[SemanticObject] = []
    for raw_object in unified.get("semantic_objects", []):
        object_id = str(raw_object.get("object_id", ""))
        mask = sam_by_id.get(object_id)
        if mask is None:
            raise SceneAdapterError(
                f"Unified object {object_id!r} has no matching final SAM mask"
            )
        filename = str(mask.get("filename", ""))
        mask_path = sam_path.parent / filename
        if not mask_path.is_file():
            raise SceneAdapterError(f"final SAM mask file {filename!r} is missing")
        mask_hash = _sha256(mask_path)
        revision_id = _mask_revision_id(object_id, mask_hash)
        warnings = _quality_warnings(raw_object)
        revisions.append(
            MaskRevision(
                revision_id=revision_id,
                object_id=object_id,
                revision_number=1,
                parent_revision=None,
                binary_mask_hash=mask_hash,
                edit_operation="initial_sam_export",
                author="SAM final export adapter",
                source="import_adapter",
                timestamp=timestamp,
                quality_report=MaskQualityReport(
                    binary=True,
                    non_empty=int(mask.get("area", 0)) > 0,
                    area_pixels=int(mask.get("area", 0)),
                    bbox_xyxy=mask.get("bbox"),
                    warnings=warnings,
                    metrics={},
                ),
                geometry_invalidation_status=GeometryInvalidationStatus.valid,
            )
        )
        transform = raw_object.get("transform") or {}
        occlusion = raw_object.get("occlusion") or {}
        is_occluded = bool(occlusion.get("partially_occluded"))
        support_type = _support_type(raw_object.get("support_type"))
        objects.append(
            SemanticObject(
                object_id=object_id,
                version=1,
                semantic_label=str(raw_object.get("semantic_label") or mask.get("label")),
                display_name=_display_name(
                    str(raw_object.get("semantic_label") or mask.get("label"))
                ),
                mask_filename=filename,
                mask_url=_artifact_url("masks", f"{object_id}/{revision_id}"),
                mask_revision=revision_id,
                mask_hash=mask_hash,
                overlay_url=_artifact_url("mask-overlays", f"{object_id}/{revision_id}"),
                approval_status="unreviewed",
                quality_warnings=warnings,
                geometry_confidence=float(raw_object.get("geometry_confidence", 0.0)),
                orientation_confidence=_orientation_confidence(raw_object),
                placement_confidence=float(raw_object.get("final_pose_confidence", 0.0)),
                center=transform["center"],
                dimensions=transform["dimensions"],
                quaternion=transform["quaternion_wxyz"],
                support_target=raw_object.get("support_target") or None,
                support_type=support_type,
                occlusion_state=(
                    OcclusionState.partially_occluded
                    if is_occluded
                    else OcclusionState.visible
                ),
                blender_object_identifier=None,
                selected_asset_identifier=None,
                metadata={
                    "primitive_type": raw_object.get("primitive_type"),
                    "source_confidence_classification": raw_object.get(
                        "confidence_classification"
                    ),
                    "source_placement_classification": raw_object.get(
                        "placement_classification"
                    ),
                    "support_confidence": raw_object.get("support_confidence"),
                    "orientation_method": raw_object.get("orientation_method"),
                    "hidden_geometry_policy": occlusion.get("hidden_geometry_policy"),
                },
            )
        )

    if len(objects) != len(sam_masks):
        missing = sorted(set(sam_by_id) - {item.object_id for item in objects})
        raise SceneAdapterError(
            "final SAM and Unified object sets differ; unconsumed SAM IDs: "
            + ", ".join(missing)
        )

    scene_id = str(unified["scene_id"])
    image_id = str(sam["image_id"])
    source_path = Path(source_image_path) if source_image_path is not None else None
    source_hash = _sha256(source_path) if source_path and source_path.is_file() else None
    raw_coordinate = unified.get("coordinate_system") or {}
    artifacts = {
        "sam_export_manifest": ArtifactReference(
            artifact_id=f"sam-manifest:{scene_id}",
            url=_artifact_url("scene-inputs", f"{scene_id}/sam-manifest"),
            media_type="application/json",
            sha256=_sha256(sam_path),
        ),
        "unified_scene_manifest": ArtifactReference(
            artifact_id=f"unified-manifest:{scene_id}",
            url=_artifact_url("scene-inputs", f"{scene_id}/unified-v3-1-1"),
            media_type="application/json",
            sha256=_sha256(unified_path),
        ),
        "moge_geometry": ArtifactReference(
            artifact_id=f"moge-geometry:{scene_id}",
            url=_artifact_url("scene-inputs", f"{scene_id}/moge-geometry"),
            media_type="application/octet-stream",
        ),
    }
    return Scene(
        package_revision=1,
        scene_id=scene_id,
        source_image=SourceImage(
            artifact_id=f"source-image:{image_id}",
            url=_artifact_url("source-images", image_id),
            original_filename=sam.get("original_filename"),
            sha256=source_hash,
        ),
        image_dimensions=ImageDimensions(
            width=int(sam["width"]), height=int(sam["height"])
        ),
        coordinate_systems=[
            CoordinateSystem(
                coordinate_system_id="canonical_world",
                handedness="right",
                up_axis="+Z",
                forward_axis="-Y",
                units="meters",
                origin="joint room reconstruction origin",
                metadata={
                    "absolute_scale_verified": bool(
                        raw_coordinate.get("absolute_scale_verified", False)
                    )
                },
            ),
            CoordinateSystem(
                coordinate_system_id="image_pixels",
                handedness="right",
                up_axis="-Y",
                forward_axis="+Z",
                units="pixels",
                origin="top-left pixel center",
            ),
            CoordinateSystem(
                coordinate_system_id="moge_raw",
                handedness="right",
                up_axis="-Y",
                forward_axis="+Z",
                units="meters",
                origin="MoGe camera frame",
                transform_to_world=raw_coordinate.get("raw_moge_to_canonical"),
            ),
        ],
        camera_candidates=[_camera(item) for item in unified["camera_candidates"]],
        selected_camera=str(unified["provisional_camera_id"]),
        structural_room_proxies=[
            _room_proxy(item) for item in unified.get("room_proxies", [])
        ],
        generation_status=GenerationStatus(
            overall=GenerationState.completed,
            stages={
                "sam_masks": GenerationState.completed,
                "moge_geometry": GenerationState.completed,
                "primitive_transforms": GenerationState.completed,
                "blender_objects": GenerationState.pending,
                "asset_replacement": GenerationState.pending,
            },
            updated_at=timestamp,
            errors=[],
        ),
        artifact_urls=artifacts,
        semantic_objects=objects,
        mask_revisions=revisions,
        metadata={
            "source_contract": "Unified V3.1.1 clean scene manifest",
            "source_mode": unified.get("mode"),
            "input_manifest_sha256": unified.get("input_manifest_sha256"),
            "uncertainties": unified.get("uncertainties", []),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sam_metadata", type=Path)
    parser.add_argument("unified_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-image", type=Path)
    args = parser.parse_args()
    scene = adapt_scene_package(
        args.sam_metadata,
        args.unified_manifest,
        source_image_path=args.source_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump_scene(scene), encoding="utf-8")


if __name__ == "__main__":
    main()
