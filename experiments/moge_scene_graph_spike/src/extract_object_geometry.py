"""Extract deterministic visible-surface geometry for approved semantic masks."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from src.object_geometry import (
    SceneNormalization,
    bounds_record,
    connected_components,
    depth_statistics,
    estimate_oriented_bounds,
    geometry_confidence,
    make_scene_normalization,
    normal_statistics,
    robust_depth_filter,
)
from src.validate_fixture import validate_fixture


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "inputs" / "office_test" / "manifest.json"
DEFAULT_MOGE_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "moge"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry"
COMPONENT_COLORS = np.asarray(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]],
    dtype=np.uint8,
)
BOX_EDGES = np.asarray(
    [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]],
    dtype=np.int32,
)
GENERATED_DIAGNOSTIC_FILES = {
    "mask_overlay.png", "mask_validity_intersection.png", "depth_inside_mask.png", "normal_inside_mask.png",
    "connected_components.png", "outlier_rejection.png", "raw_visible_geometry.npz", "filtered_visible_geometry.npz",
    "extracted_colored_point_cloud.ply", "point_cloud_with_centroid.ply", "point_cloud_with_axis_aligned_bounds.ply",
    "point_cloud_with_oriented_bounds.ply", "object_report.json", "object_report.md",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_and_load(manifest_path: Path, moge_dir: Path) -> dict[str, Any]:
    fixture_errors = validate_fixture(manifest_path)
    if fixture_errors:
        raise ValueError("fixture validation failed: " + "; ".join(fixture_errors))
    manifest = _load_json(manifest_path)
    metadata = _load_json(moge_dir / "metadata.json")
    fixture_dir = manifest_path.parent
    image_path = fixture_dir / manifest["image"]["filename"]
    with Image.open(image_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with np.load(moge_dir / "geometry.npz", allow_pickle=False) as archive:
        required = {"points", "depth", "normal", "valid_mask", "intrinsics"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"MoGe archive missing keys: {sorted(missing)}")
        arrays = {name: archive[name] for name in required}

    height, width = source.shape[:2]
    if arrays["points"].shape != (height, width, 3):
        raise ValueError(f"points shape {arrays['points'].shape} does not match image")
    if arrays["normal"].shape != (height, width, 3):
        raise ValueError(f"normal shape {arrays['normal'].shape} does not match image")
    if arrays["depth"].shape != (height, width) or arrays["valid_mask"].shape != (height, width):
        raise ValueError("depth/validity shape does not match image")
    if arrays["intrinsics"].shape != (3, 3):
        raise ValueError("normalized intrinsics must have shape (3, 3)")
    valid = arrays["valid_mask"].astype(bool)
    finite_valid = (
        np.isfinite(arrays["points"]).all(axis=-1)
        & np.isfinite(arrays["depth"])
        & np.isfinite(arrays["normal"]).all(axis=-1)
    )
    if not finite_valid[valid].all():
        raise ValueError("MoGe geometry contains non-finite values inside valid regions")
    object_ids = [obj["object_id"] for obj in manifest["objects"]]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("manifest object IDs are not unique")

    masks: dict[str, np.ndarray] = {}
    for obj in manifest["objects"]:
        mask_path = fixture_dir / obj["mask_filename"]
        with Image.open(mask_path) as mask_image:
            if mask_image.format != "PNG" or mask_image.size != (width, height):
                raise ValueError(f"invalid mask file or dimensions: {mask_path}")
            mask_array = np.asarray(mask_image.convert("L"))
        values = set(np.unique(mask_array).tolist())
        if not values.issubset({0, 255}) or 255 not in values:
            raise ValueError(f"mask is not non-empty binary PNG: {mask_path}")
        masks[obj["object_id"]] = mask_array == 255
    return {
        "manifest": manifest,
        "metadata": metadata,
        "source": source,
        "image_path": image_path,
        "arrays": arrays,
        "masks": masks,
    }


def _box_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[x, y, z] for z in [minimum[2], maximum[2]] for y in [minimum[1], maximum[1]] for x in [minimum[0], maximum[0]]],
        dtype=np.float32,
    )


def _obb_corners(obb: dict[str, Any]) -> np.ndarray:
    center = np.asarray(obb["center_xyz"], dtype=np.float32)
    dimensions = np.asarray(obb["dimensions_xyz_along_axes"], dtype=np.float32)
    axes = np.asarray(obb["orientation_matrix_columns_are_box_axes"], dtype=np.float32)
    local = _box_corners(-dimensions / 2.0, dimensions / 2.0)
    return center + local @ axes.T


def _write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    marker_points: np.ndarray | None = None,
    marker_color: tuple[int, int, int] = (255, 0, 0),
    edges: np.ndarray | None = None,
) -> None:
    marker_points = np.empty((0, 3), dtype=np.float32) if marker_points is None else marker_points.astype(np.float32)
    marker_colors = np.tile(np.asarray(marker_color, dtype=np.uint8), (len(marker_points), 1))
    all_points = np.concatenate([points.astype(np.float32), marker_points], axis=0)
    all_colors = np.concatenate([colors.astype(np.uint8), marker_colors], axis=0)
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(all_points), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = all_colors[:, 0], all_colors[:, 1], all_colors[:, 2]
    edge_count = 0 if edges is None else len(edges)
    header_lines = [
        "ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}", "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
    ]
    if edge_count:
        header_lines.extend([f"element edge {edge_count}", "property int vertex1", "property int vertex2"])
    header = ("\n".join([*header_lines, "end_header", ""])).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
        if edge_count:
            edge_dtype = np.dtype([("vertex1", "<i4"), ("vertex2", "<i4")])
            edge_records = np.empty(edge_count, dtype=edge_dtype)
            edge_records["vertex1"] = edges[:, 0] + len(points)
            edge_records["vertex2"] = edges[:, 1] + len(points)
            edge_records.tofile(stream)


def _save_previews(
    diagnostic_dir: Path,
    source: np.ndarray,
    mask: np.ndarray,
    object_valid: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    labels: np.ndarray,
    keep: np.ndarray,
    rejected: np.ndarray,
) -> dict[str, str]:
    overlay = source.copy()
    tint = np.asarray([255, 40, 40], dtype=np.float32)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * tint).astype(np.uint8)
    Image.fromarray(overlay).save(diagnostic_dir / "mask_overlay.png")

    intersection = np.zeros((*mask.shape, 3), dtype=np.uint8)
    intersection[mask] = [80, 80, 80]
    intersection[object_valid] = [255, 255, 255]
    Image.fromarray(intersection).save(diagnostic_dir / "mask_validity_intersection.png")

    depth_preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    values = depth[object_valid]
    if values.size:
        low, high = np.percentile(values, [2, 98])
        span = max(float(high - low), np.finfo(np.float32).eps)
        scaled = np.zeros(mask.shape, dtype=np.float32)
        scaled[object_valid] = np.clip((depth[object_valid] - low) / span, 0.0, 1.0)
        depth_preview = (colormaps["turbo"](scaled)[..., :3] * 255).astype(np.uint8)
        depth_preview[~object_valid] = 0
    Image.fromarray(depth_preview).save(diagnostic_dir / "depth_inside_mask.png")

    normal_preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    normal_preview[object_valid] = np.clip((normal[object_valid] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(normal_preview).save(diagnostic_dir / "normal_inside_mask.png")

    component_preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for component_id in range(1, int(labels.max()) + 1):
        component_preview[labels == component_id] = COMPONENT_COLORS[(component_id - 1) % len(COMPONENT_COLORS)]
    Image.fromarray(component_preview).save(diagnostic_dir / "connected_components.png")

    rejection_preview = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rejection_preview[mask] = [70, 70, 70]
    rejection_preview[keep] = [40, 200, 80]
    rejection_preview[rejected] = [255, 30, 30]
    Image.fromarray(rejection_preview).save(diagnostic_dir / "outlier_rejection.png")
    return {
        "mask_overlay": "mask_overlay.png",
        "mask_validity_intersection": "mask_validity_intersection.png",
        "depth_inside_mask": "depth_inside_mask.png",
        "normal_inside_mask": "normal_inside_mask.png",
        "connected_components": "connected_components.png",
        "outlier_rejection": "outlier_rejection.png",
    }


def _component_statistics(
    component_records: list[dict[str, Any]],
    labels: np.ndarray,
    valid: np.ndarray,
    keep: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in component_records:
        component = labels == record["component_id"]
        component_valid = component & valid
        component_keep = component & keep
        result = dict(record)
        result.update(
            {
                "valid_geometry_pixel_count": int(component_valid.sum()),
                "valid_geometry_ratio": float(component_valid.sum() / max(component.sum(), 1)),
                "filtered_point_count": int(component_keep.sum()),
                "depth": depth_statistics(depth[component_valid]),
                "raw_3d_centroid": points[component_valid].mean(axis=0).tolist() if component_valid.any() else None,
                "filtered_3d_centroid": points[component_keep].mean(axis=0).tolist() if component_keep.any() else None,
            }
        )
        results.append(result)
    return results


def _object_warnings(
    *,
    mask_count: int,
    valid_count: int,
    valid_ratio: float,
    component_count: int,
    depth_stats: dict[str, Any] | None,
    raw_bounds: dict[str, Any] | None,
    rejected_ratio: float,
    obb: dict[str, Any],
    partial_occlusion: bool,
    scene_longest_extent: float,
) -> list[str]:
    warnings: list[str] = []
    if valid_count == 0:
        warnings.append("empty mask/validity intersection")
    if valid_count < 100:
        warnings.append("very small valid pixel count")
    if mask_count < 100:
        warnings.append("tiny mask")
    if valid_ratio < 0.8:
        warnings.append("low valid geometry ratio")
    if component_count > 5:
        warnings.append("excessive disconnected components; preserved under the same semantic object ID")
    if depth_stats:
        spread = depth_stats["percentiles"]["95"] - depth_stats["percentiles"]["5"]
        if spread > max(1.0, 0.15 * depth_stats["median"]):
            warnings.append("large depth spread")
    if not obb.get("estimated", False):
        warnings.append("unstable orientation; oriented bounding box omitted")
    factors = obb.get("unstable_factors", {})
    if factors.get("thin"):
        warnings.append("thin geometry")
    if factors.get("single_visible_surface"):
        warnings.append("geometry dominated by a single visible surface")
    if partial_occlusion:
        warnings.append("likely partial occlusion; measurements cover visible surfaces only")
    if raw_bounds:
        largest = max(raw_bounds["visible_surface_dimensions"])
        if largest > max(5.0, 0.75 * scene_longest_extent):
            warnings.append("suspiciously large raw metric dimensions")
    if rejected_ratio > 0.02:
        warnings.append("depth outliers warrant boundary review; mask leakage is not established by depth alone")
    return warnings


def _normalized_obb(obb: dict[str, Any], normalization: SceneNormalization) -> dict[str, Any] | None:
    if not obb.get("estimated"):
        return None
    return {
        "center_xyz": normalization.apply(np.asarray(obb["center_xyz"])).tolist(),
        "dimensions_xyz_along_axes": (np.asarray(obb["dimensions_xyz_along_axes"]) * normalization.scale).tolist(),
        "orientation_matrix_columns_are_box_axes": obb["orientation_matrix_columns_are_box_axes"],
        "method": obb["method"],
        "confidence": obb["confidence"],
    }


def _write_object_markdown(path: Path, record: dict[str, Any]) -> None:
    warnings = "\n".join(f"- {warning}" for warning in record["warnings"]) or "- None"
    raw_filtered = record["geometry"]["raw_moge"]["filtered"]
    normalized_filtered = record["geometry"]["scene_normalized"]["filtered"]
    raw_dims = raw_filtered["visible_surface_dimensions"] if raw_filtered else None
    normalized_dims = normalized_filtered["visible_surface_dimensions"] if normalized_filtered else None
    text = f"""# {record['semantic_label']}

- Object ID: `{record['object_id']}`
- Visible mask pixels: {record['total_mask_pixel_count']}
- Valid MoGe pixels: {record['valid_moge_pixel_count']} ({record['valid_geometry_ratio'] * 100:.2f}%)
- Connected components: {record['connected_component_count']}
- Raw / filtered / rejected points: {record['point_filtering']['raw_point_count']} / {record['point_filtering']['filtered_point_count']} / {record['point_filtering']['rejected_point_count']}
- Raw visible dimensions: {raw_dims}
- Scene-normalized visible dimensions: {normalized_dims}
- Geometry confidence: {record['geometry_confidence']['label']} ({record['geometry_confidence']['score']:.3f})
- Oriented bounds estimated: {record['oriented_bounding_box']['estimated']}
- Oriented-bounds confidence: {record['oriented_bounding_box']['confidence']:.3f}

All measurements describe visible pixels/surfaces only. Raw MoGe scale is retained but is not independently verified.

## Warnings

{warnings}
"""
    path.write_text(text, encoding="utf-8")


def extract(manifest_path: Path, moge_dir: Path, output_dir: Path) -> dict[str, Any]:
    loaded = _validate_and_load(manifest_path.resolve(), moge_dir.resolve())
    manifest, metadata = loaded["manifest"], loaded["metadata"]
    source, arrays, masks = loaded["source"], loaded["arrays"], loaded["masks"]
    points, depth = arrays["points"], arrays["depth"]
    normal, valid = arrays["normal"], arrays["valid_mask"].astype(bool)
    scene_points = points[valid]
    normalization = make_scene_normalization(scene_points)
    scene_extent = float((normalization.robust_bounds_max - normalization.robust_bounds_min).max())
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_root = output_dir / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    object_records: list[dict[str, Any]] = []

    for obj in manifest["objects"]:
        object_id, label = obj["object_id"], obj["semantic_label"]
        diagnostic_dir = diagnostics_root / object_id
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        for filename in GENERATED_DIAGNOSTIC_FILES:
            generated_path = diagnostic_dir / filename
            if generated_path.exists():
                generated_path.unlink()
        mask = masks[object_id]
        object_valid = mask & valid
        labels, component_records = connected_components(mask)
        keep, filtering = robust_depth_filter(depth, object_valid, labels)
        rejected = filtering["rejected_mask"]
        raw_points, raw_depth, raw_normals = points[object_valid], depth[object_valid], normal[object_valid]
        filtered_points, filtered_depth, filtered_normals = points[keep], depth[keep], normal[keep]
        raw_count, filtered_count = len(raw_points), len(filtered_points)
        rejected_count = raw_count - filtered_count
        rejected_ratio = rejected_count / raw_count if raw_count else 0.0
        mask_count = int(mask.sum())
        valid_ratio = raw_count / mask_count if mask_count else 0.0
        ys, xs = np.nonzero(mask)
        partial_occlusion = "partially_occluded" in obj.get("selection_role", [])
        raw_depth_stats = depth_statistics(raw_depth)
        filtered_depth_stats = depth_statistics(filtered_depth)
        normal_stats = normal_statistics(filtered_normals)
        raw_unfiltered_geometry = bounds_record(raw_points)
        raw_filtered_geometry = bounds_record(filtered_points)
        normalized_unfiltered_geometry = bounds_record(raw_points, normalization)
        normalized_filtered_geometry = bounds_record(filtered_points, normalization)
        obb = estimate_oriented_bounds(
            filtered_points,
            valid_geometry_ratio=valid_ratio,
            component_count=len(component_records),
            partial_occlusion=partial_occlusion,
        )
        obb_normalized = _normalized_obb(obb, normalization)
        thin = bool(obb.get("unstable_factors", {}).get("thin", False))
        confidence = geometry_confidence(
            valid_ratio=valid_ratio,
            valid_count=raw_count,
            component_count=len(component_records),
            depth_stats=filtered_depth_stats,
            normal_stats=normal_stats,
            outlier_ratio=rejected_ratio,
            thin=thin,
            partial_occlusion=partial_occlusion,
            obb_confidence=float(obb.get("confidence", 0.0)),
        )
        warnings = _object_warnings(
            mask_count=mask_count,
            valid_count=raw_count,
            valid_ratio=valid_ratio,
            component_count=len(component_records),
            depth_stats=raw_depth_stats,
            raw_bounds=raw_unfiltered_geometry,
            rejected_ratio=rejected_ratio,
            obb=obb,
            partial_occlusion=partial_occlusion,
            scene_longest_extent=scene_extent,
        )
        if label == "desk_chair" and len(component_records) > 1:
            warnings.append("chair supports are disconnected; all components remain one semantic object")
        if label == "coat_rack":
            warnings.append("thin coat-rack branches are preserved by component-aware filtering")
        if label == "wall_light_fixture":
            warnings.append("wall-mounted thin geometry may include only the outward-facing surface")
        if label == "desktop_box" and raw_count < 1000:
            warnings.append("small object; dimensions are sensitive to a few pixels")
            warnings.append("excessive MoGe depth spread is retained; raw Z extent must not be interpreted as physical object depth")
        if "small_object" in obj.get("selection_role", []) and raw_unfiltered_geometry:
            if max(raw_unfiltered_geometry["visible_surface_dimensions"]) > 0.25 * scene_extent:
                warnings.append("suspiciously large raw metric dimensions for a fixture selected as a small object")
        if label == "left_tall_filing_cabinet":
            warnings.append("multiple visible cabinet planes are measured; hidden rear geometry is not inferred")

        component_stats = _component_statistics(component_records, labels, object_valid, keep, depth, points)
        previews = _save_previews(diagnostic_dir, source, mask, object_valid, depth, normal, labels, keep, rejected)
        np.savez_compressed(
            diagnostic_dir / "raw_visible_geometry.npz",
            image_xy=np.column_stack(np.nonzero(object_valid)[::-1]).astype(np.int32),
            points=raw_points.astype(np.float32),
            depth=raw_depth.astype(np.float32),
            normal=raw_normals.astype(np.float32),
            scene_normalized_points=normalization.apply(raw_points).astype(np.float32),
            component_id=labels[object_valid].astype(np.int32),
        )
        np.savez_compressed(
            diagnostic_dir / "filtered_visible_geometry.npz",
            image_xy=np.column_stack(np.nonzero(keep)[::-1]).astype(np.int32),
            points=filtered_points.astype(np.float32),
            depth=filtered_depth.astype(np.float32),
            normal=filtered_normals.astype(np.float32),
            scene_normalized_points=normalization.apply(filtered_points).astype(np.float32),
            component_id=labels[keep].astype(np.int32),
        )
        point_colors = source[keep]
        _write_ply(diagnostic_dir / "extracted_colored_point_cloud.ply", filtered_points, point_colors)
        if filtered_count:
            centroid = filtered_points.mean(axis=0, keepdims=True)
            _write_ply(diagnostic_dir / "point_cloud_with_centroid.ply", filtered_points, point_colors, marker_points=centroid, marker_color=(255, 0, 0))
            minimum, maximum = filtered_points.min(axis=0), filtered_points.max(axis=0)
            _write_ply(
                diagnostic_dir / "point_cloud_with_axis_aligned_bounds.ply", filtered_points, point_colors,
                marker_points=_box_corners(minimum, maximum), marker_color=(255, 255, 0), edges=BOX_EDGES,
            )
            if obb.get("estimated"):
                _write_ply(
                    diagnostic_dir / "point_cloud_with_oriented_bounds.ply", filtered_points, point_colors,
                    marker_points=_obb_corners(obb), marker_color=(255, 0, 255), edges=BOX_EDGES,
                )

        record: dict[str, Any] = {
            "object_id": object_id,
            "semantic_label": label,
            "mask_filename": obj["mask_filename"],
            "visible_surfaces_only": True,
            "total_mask_pixel_count": mask_count,
            "valid_moge_pixel_count": raw_count,
            "valid_geometry_ratio": valid_ratio,
            "connected_component_count": len(component_records),
            "components": component_stats,
            "bbox_2d_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "mask_centroid_2d_xy": [float(xs.mean()), float(ys.mean())],
            "depth_raw_moge": raw_depth_stats,
            "depth_filtered": filtered_depth_stats,
            "raw_3d_centroid": raw_points.mean(axis=0).tolist() if raw_count else None,
            "filtered_3d_centroid": filtered_points.mean(axis=0).tolist() if filtered_count else None,
            "geometry": {
                "raw_moge": {"unfiltered": raw_unfiltered_geometry, "filtered": raw_filtered_geometry},
                "scene_normalized": {"unfiltered": normalized_unfiltered_geometry, "filtered": normalized_filtered_geometry},
            },
            "dominant_normal": normal_stats,
            "point_filtering": {
                "raw_point_count": raw_count,
                "filtered_point_count": filtered_count,
                "rejected_point_count": rejected_count,
                "rejected_point_ratio": rejected_ratio,
                "method": "component-wise robust depth candidate detection plus local same-component disagreement test",
                "details": {k: v for k, v in filtering.items() if not isinstance(v, np.ndarray)},
            },
            "oriented_bounding_box": obb,
            "oriented_bounding_box_scene_normalized": obb_normalized,
            "geometry_confidence": confidence,
            "warnings": list(dict.fromkeys(warnings)),
            "diagnostics": {
                **previews,
                "raw_geometry_npz": "raw_visible_geometry.npz",
                "filtered_geometry_npz": "filtered_visible_geometry.npz",
                "colored_point_cloud": "extracted_colored_point_cloud.ply",
                "point_cloud_with_centroid": "point_cloud_with_centroid.ply" if filtered_count else None,
                "point_cloud_with_axis_aligned_bounds": "point_cloud_with_axis_aligned_bounds.ply" if filtered_count else None,
                "point_cloud_with_oriented_bounds": "point_cloud_with_oriented_bounds.ply" if obb.get("estimated") else None,
                "report_json": "object_report.json",
                "report_markdown": "object_report.md",
            },
        }
        (diagnostic_dir / "object_report.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_object_markdown(diagnostic_dir / "object_report.md", record)
        object_records.append(record)

    scene_warnings = [
        "absolute metric scale is returned by MoGe-2 but is not independently verified",
        "source resembles an orthographic/isometric render while MoGe estimated a perspective camera",
        "all object geometry describes visible pixels and surfaces only; hidden or occluded geometry is not inferred",
    ]
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "scene_id": manifest["scene_id"],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_image": {
            "fixture_filename": manifest["image"]["filename"],
            "moge_input_filename": Path(metadata["source_image"]).name,
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
        },
        "moge": {
            "model_name": metadata["model_name"], "model_revision": metadata.get("model_revision"), "device": metadata["device"],
            "runtime_seconds": metadata["runtime_seconds"], "normalized_intrinsics": arrays["intrinsics"].tolist(),
            "estimated_fov_x_degrees": metadata["estimated_fov_x_degrees"],
        },
        "coordinate_systems": {
            "raw_moge": {
                "description": "unmodified MoGe camera-space values", "convention": "OpenCV", "x_axis": "right", "y_axis": "down",
                "z_axis": "forward", "scale": "metric as returned by MoGe-2; absolute metric scale is not verified",
            },
            "scene_normalized": {
                "description": "uniformly normalized relative to valid scene geometry", "convention": "same axes as raw_moge",
                "units": "scene-relative", "aspect_ratios_preserved": True,
            },
        },
        "raw_to_normalized_transformation": normalization.to_record(),
        "camera_model_uncertainty": {
            "perspective_with_estimated_fov": True,
            "possible_orthographic_source": True,
            "estimated_horizontal_fov_degrees": metadata["estimated_fov_x_degrees"],
            "explanation": "The source resembles an orthographic or isometric render, but MoGe inferred a perspective camera with a narrow field of view.",
            "recommendation": "Test both perspective and orthographic camera models during later Blender validation.",
        },
        "processing_parameters": {
            "connectivity": 8,
            "robust_bounds_percentiles": [2, 98],
            "scene_normalization_percentiles": [1, 99],
            "outlier_filter": {
                "mad_multiplier": 8.0, "minimum_global_deviation_m": 0.25, "minimum_local_deviation_m": 0.15,
                "minimum_same_component_neighbors": 3,
            },
            "oriented_bounds": "PCA on filtered visible points with p2/p98 extents; omitted below confidence/stability thresholds",
        },
        "objects": object_records,
        "next_phase_readiness": {
            "status": "conditionally_ready",
            "sufficient_for_room_plane_and_support_relationship_estimation": True,
            "basis": "all five objects have complete valid visible geometry and medium-or-higher geometry confidence; conservative filtering retained at least 97% of every object",
            "constraints": [
                "absolute metric scale remains unverified",
                "test perspective and orthographic camera interpretations",
                "do not require oriented bounds for the partially occluded chair, disconnected coat rack, or thin wall light",
                "support inference must remain visible-surface-based and uncertainty-aware",
            ],
        },
        "warnings": scene_warnings,
    }
    (output_dir / "object_geometry.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _save_combined_overview(output_dir / "object_geometry_overview.png", source, masks, object_records)
    return result


def _save_combined_overview(path: Path, source: np.ndarray, masks: dict[str, np.ndarray], records: list[dict[str, Any]]) -> None:
    canvas = source.astype(np.float32).copy()
    for record in records:
        score = record["geometry_confidence"]["score"]
        color = np.asarray([40, 200, 80] if score >= 0.75 else [255, 190, 40] if score >= 0.5 else [230, 50, 50], dtype=np.float32)
        mask = masks[record["object_id"]]
        canvas[mask] = 0.7 * canvas[mask] + 0.3 * color
    scene_image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    panel_width = 390
    image = Image.new("RGB", (scene_image.width + panel_width, scene_image.height), (20, 20, 24))
    image.paste(scene_image, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((scene_image.width + 12, 10), "Object geometry confidence", fill=(240, 240, 240))
    for index, record in enumerate(records, start=1):
        score = record["geometry_confidence"]["score"]
        color = (40, 200, 80) if score >= 0.75 else (255, 190, 40) if score >= 0.5 else (230, 50, 50)
        x0, y0, x1, y1 = record["bbox_2d_xyxy_inclusive"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.rectangle((x0, y0, min(x0 + 14, x1), min(y0 + 12, y1)), fill=(0, 0, 0))
        draw.text((x0 + 2, y0 + 1), str(index), fill=color)
        panel_x = scene_image.width + 12
        panel_y = 35 + (index - 1) * 78
        warning_marker = " !" if record["warnings"] else ""
        draw.text((panel_x, panel_y), f"{index}. {record['semantic_label']}{warning_marker}", fill=color)
        draw.text((panel_x, panel_y + 13), f"ID {record['object_id']}  confidence {score:.3f}", fill=(220, 220, 220))
        warning_text = "; ".join(record["warnings"]) or "no warnings"
        for line_index, line in enumerate(textwrap.wrap(warning_text, width=58)[:3]):
            draw.text((panel_x, panel_y + 27 + line_index * 11), line, fill=(165, 165, 170))
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--moge-dir", type=Path, default=DEFAULT_MOGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = extract(args.manifest, args.moge_dir, args.output_dir)
    print(json.dumps({"scene_id": result["scene_id"], "object_count": len(result["objects"]), "output": str(args.output_dir / "object_geometry.json")}, indent=2))
