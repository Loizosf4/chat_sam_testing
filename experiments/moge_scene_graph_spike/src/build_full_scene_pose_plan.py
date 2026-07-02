"""Build the complete 20-mask offline SAM + MoGe primitive pose plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .pose_refinement import (
    cluster_normals_spherical,
    cuboid_corners,
    extract_mask_axis,
    extract_masked_normals,
    matrix_to_quaternion_wxyz,
    minimum_area_rectangle,
    normalize_vectors,
    project_world_points,
    score_rotation_candidate,
)
from .refine_primitive_plan import ROOT, _write_mask_axis_overlay, _write_normal_visualization, quaternion_to_matrix


REPO_ROOT = ROOT.parents[1]
EXPORT = REPO_ROOT / "data" / "exports" / "auto_scene_final_24457ea2"
SOURCE_IMAGE = REPO_ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"
FIXTURE = ROOT / "inputs" / "office_test_full"
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "full_scene_pose_plan_v1"
APPROVED_IDS = {
    "e971e9fbbf5746eea108d07bfbb5764a",
    "b9858c1f45f443a88d84229a1824d339",
    "9171a7b4d4e142ca936f2564200d0bdb",
    "1064a3a09c4c427e80caa227d17a7cc1",
    "4ec402f9a71d4ea7925b487ac7e11249",
    "49fd02b50a914e8198bacf4644bc377f",
}
WALL_LABELS = {"left_bulletin_board", "center_wall_frame", "right_wall_frame", "left_radiator", "right_radiator"}
IRREGULAR_LABELS = {"open_floor_box", "front_left_box", "front_rear_box"}
LEFT_WALL_LABELS = {"left_bulletin_board", "center_wall_frame", "left_radiator"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _matrix(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray) -> list[list[float]]:
    result = np.eye(4)
    result[:3, :3] = rotation @ np.diag(dimensions)
    result[:3, 3] = center
    return result.tolist()


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _angle_difference(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _convex_hull(points: list[tuple[float, float]] | np.ndarray) -> np.ndarray:
    values = sorted(set(map(tuple, np.asarray(points, dtype=float).tolist())))
    if len(values) <= 2:
        return np.asarray(values)
    def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower: list[tuple[float, float]] = []
    for value in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], value) <= 0:
            lower.pop()
        lower.append(value)
    upper: list[tuple[float, float]] = []
    for value in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], value) <= 0:
            upper.pop()
        upper.append(value)
    return np.asarray(lower[:-1] + upper[:-1])


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def validate_export() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = _load(EXPORT / "metadata.json")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for item in metadata["masks"]:
        path = EXPORT / item["filename"]
        image = Image.open(path)
        values = np.asarray(image)
        unique = sorted(int(value) for value in np.unique(values))
        binary = len(unique) <= 2 and unique[0] == 0 and unique[-1] > 0
        object_id = item["mask_id"]
        record = {
            "object_id": object_id,
            "semantic_label": item["label"],
            "mask_filename": item["filename"],
            "source_export_path": _repo_path(path),
            "sha256": _sha(path),
            "mask_revision": None,
            "width": image.width,
            "height": image.height,
            "binary": binary,
            "non_empty": bool(np.any(values)),
            "area": int(np.count_nonzero(values)),
            "bbox_xyxy_inclusive": item["bbox"],
            "fixture_classification": "existing_approved_object" if object_id in APPROVED_IDS else "new_object",
            "exclusion_reason": None,
        }
        assert object_id not in seen and len(object_id) == 32
        assert image.size == (metadata["width"], metadata["height"])
        assert record["binary"] and record["non_empty"] and record["semantic_label"] and len(record["sha256"]) == 64
        seen.add(object_id)
        records.append(record)
    assert len(records) == len(metadata["masks"]) == 20
    return metadata, records


def create_fixture(metadata: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    fixture = {
        "schema_version": "1.0",
        "scene_id": "office_test_full",
        "fixture_mode": "immutable_source_references",
        "source_image": {
            "image_id": metadata["image_id"],
            "path": _repo_path(SOURCE_IMAGE),
            "sha256": _sha(SOURCE_IMAGE),
            "width": metadata["width"],
            "height": metadata["height"],
        },
        "authoritative_sam_export": _repo_path(EXPORT),
        "authoritative_metadata": _repo_path(EXPORT / "metadata.json"),
        "objects": records,
        "mask_count": len(records),
        "approved_object_count": sum(item["fixture_classification"] == "existing_approved_object" for item in records),
        "new_object_count": sum(item["fixture_classification"] == "new_object" for item in records),
        "excluded_mask_count": sum(item["fixture_classification"] == "excluded_structural_background" for item in records),
    }
    (FIXTURE / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    (FIXTURE / "README.md").write_text(
        "# Complete office-test fixture\n\nThis fixture references the authoritative final SAM export without copying or modifying masks. Hashes in `manifest.json` make every source reference immutable and auditable.\n",
        encoding="utf-8",
    )
    return fixture


def _corrected_camera(base_plan: dict[str, Any], room: dict[str, Any]) -> dict[str, Any]:
    camera = next(item for item in base_plan["camera_candidates"] if item["camera_id"] == "camera_perspective_moge")
    corrected = room["corrected_cameras"]["camera_perspective_moge"]
    return {
        "canonical_to_camera": np.asarray(camera["canonical_to_camera_transform"], dtype=float),
        "intrinsics": np.asarray(camera["perspective"]["normalized_intrinsics"], dtype=float),
        "image_shape": (447, 447),
        "shift_pixels": np.asarray([-corrected["shift_x"] * 447, corrected["shift_y"] * 447]),
        "source": "corrected Blender perspective camera shift/framing plus MoGe canonical camera transform",
    }


def _project(points: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    return project_world_points(points, camera["canonical_to_camera"], camera["intrinsics"], camera["image_shape"]) + camera["shift_pixels"]


def _project_cuboid(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    return _project(cuboid_corners(center, dimensions, rotation), camera)


def _projection_metrics(projected: np.ndarray, mask: np.ndarray, axis: dict[str, Any]) -> dict[str, Any]:
    y, x = np.nonzero(mask)
    target = np.asarray([x.min(), y.min(), x.max() + 1, y.max() + 1], float)
    bbox = np.asarray([projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()])
    iw = max(0.0, min(bbox[2], target[2]) - max(bbox[0], target[0]))
    ih = max(0.0, min(bbox[3], target[3]) - max(bbox[1], target[1]))
    intersection = iw * ih
    area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_b = (target[2] - target[0]) * (target[3] - target[1])
    hull = _convex_hull(projected)
    def inside(point: tuple[float, float]) -> bool:
        signs = []
        for index, a in enumerate(hull):
            b = hull[(index + 1) % len(hull)]
            signs.append((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))
        return all(value >= -1e-8 for value in signs) or all(value <= 1e-8 for value in signs)
    xmin, xmax = max(0, int(hull[:, 0].min())), min(mask.shape[1] - 1, int(hull[:, 0].max()) + 1)
    ymin, ymax = max(0, int(hull[:, 1].min())), min(mask.shape[0] - 1, int(hull[:, 1].max()) + 1)
    predicted = {(px, py) for py in range(ymin, ymax + 1) for px in range(xmin, xmax + 1) if inside((px + 0.5, py + 0.5))}
    actual = set(zip(x.tolist(), y.tolist()))
    rectangle = minimum_area_rectangle(projected)
    edge_error = _angle_difference(rectangle["angle_degrees"], axis["dominant_axis_angle_degrees"])
    return {
        "projected_bbox": bbox.tolist(),
        "mask_bbox": target.tolist(),
        "bbox_iou": float(intersection / max(1e-9, area_a + area_b - intersection)),
        "hull_iou": float(len(predicted & actual) / max(1, len(predicted | actual))),
        "centroid_error_pixels": float(np.linalg.norm((bbox[:2] + bbox[2:]) / 2 - (target[:2] + target[2:]) / 2)),
        "projected_dominant_angle_degrees": rectangle["angle_degrees"],
        "dominant_edge_angle_error_degrees": edge_error,
        "mask_axis_agreement": float(max(0.0, 1.0 - edge_error / 90.0)),
        "projected_hull": hull.tolist(),
    }


def _wall_frame(room: dict[str, Any], wall_name: str) -> dict[str, np.ndarray | float]:
    matrix = np.asarray(room["corrected_structures"][wall_name]["matrix_world"], dtype=float)
    columns = [matrix[:3, index] for index in range(3)]
    thin = min(columns, key=np.linalg.norm)
    normal = thin / np.linalg.norm(thin)
    normal[2] = 0.0
    normal /= np.linalg.norm(normal)
    floor_center = np.asarray(room["corrected_structures"]["plane_floor"]["location"], dtype=float)
    center = matrix[:3, 3]
    if np.dot(normal, floor_center - center) < 0:
        normal *= -1
    up = np.asarray([0.0, 0.0, 1.0])
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack([tangent, up, normal])
    return {"normal": normal, "rotation": rotation, "inner_point": center + normal * np.linalg.norm(thin) / 2, "thickness": float(np.linalg.norm(thin))}


def _fit_visible_box(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray, mask: np.ndarray, camera: dict[str, Any], mode: str, support: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    fitted_center = center.copy()
    fitted_dimensions = np.maximum(dimensions.copy(), 0.025)
    y, x = np.nonzero(mask)
    target = np.asarray([(x.min() + x.max() + 1) / 2, (y.min() + y.max() + 1) / 2, x.max() - x.min() + 1, y.max() - y.min() + 1], float)
    if mode == "floor":
        def apply(values: np.ndarray) -> None:
            fitted_center[0], fitted_center[1] = values[0], values[1]
            fitted_dimensions[0], fitted_dimensions[2] = max(0.025, values[2]), max(0.025, values[3])
            fitted_center[2] = fitted_dimensions[2] / 2
        values = np.asarray([fitted_center[0], fitted_center[1], fitted_dimensions[0], fitted_dimensions[2]])
    else:
        assert support is not None
        tangent, up, normal = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        inner = np.asarray(support["inner_point"])
        tangent_value = float(np.dot(fitted_center - inner, tangent))
        height_value = float(fitted_center[2])
        def apply(values: np.ndarray) -> None:
            fitted_dimensions[0], fitted_dimensions[1] = max(0.025, values[2]), max(0.025, values[3])
            fitted_center[:] = inner + tangent * values[0] + up * values[1] + normal * (fitted_dimensions[2] / 2)
        values = np.asarray([tangent_value, height_value, fitted_dimensions[0], fitted_dimensions[1]])
    def features() -> np.ndarray:
        projected = _project_cuboid(fitted_center, fitted_dimensions, rotation, camera)
        bbox = np.asarray([projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()])
        return np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]])
    best_values = values.copy()
    best_error = float("inf")
    for _ in range(12):
        apply(values)
        current = features()
        residual = current - target
        error = float(np.linalg.norm(residual))
        if np.isfinite(error) and error < best_error:
            best_error = error
            best_values = values.copy()
        if error < 0.02:
            break
        jacobian = np.zeros((4, 4))
        for index in range(4):
            changed = values.copy()
            changed[index] += 1e-4
            apply(changed)
            jacobian[:, index] = (features() - current) / 1e-4
        apply(values)
        if not np.isfinite(jacobian).all():
            break
        step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        if not np.isfinite(step).all():
            break
        # Perspective-box fitting can be ill-conditioned near the image edge.
        # Bound each update and accept it only when it improves the projection.
        step[:2] = np.clip(step[:2], -0.20, 0.20)
        dimension_limit = np.maximum(values[2:] * 0.30, 0.025)
        step[2:] = np.clip(step[2:], -dimension_limit, dimension_limit)
        improved = False
        for damping in (1.0, 0.5, 0.25, 0.1):
            candidate = values - damping * step
            candidate[2:] = np.clip(candidate[2:], 0.025, 2.5)
            apply(candidate)
            candidate_features = features()
            candidate_error = float(np.linalg.norm(candidate_features - target))
            if np.isfinite(candidate_error) and candidate_error + 1e-6 < error:
                values = candidate
                improved = True
                break
        if not improved:
            break
    apply(best_values)
    return fitted_center, fitted_dimensions


def _approved_transforms(cumulative: dict[str, Any]) -> dict[str, dict[str, Any]]:
    approved: dict[str, dict[str, Any]] = {}
    for item in cumulative["semantic_inventory"]:
        transform = item["final_transform"]
        approved[item["object_id"]] = {
            "center": transform["location"],
            "dimensions": transform["local_oriented_dimensions"],
            "quaternion_wxyz": transform["rotation_quaternion_wxyz"],
            "approval_source": "cumulative_scene_validation.json",
            "user_overrides": item.get("user_overrides", []),
        }
    chair = approved["9171a7b4d4e142ca936f2564200d0bdb"]
    chair.update({
        "center": [0.4257657802, 1.0472733016, 0.4270060736],
        "dimensions": [0.3388930019, 0.1, 0.2414989213],
        "quaternion_wxyz": [0.3491493496, 0.0, 0.0, 0.9370670903],
        "approval_source": "explicit visual approval of pose-refinement-v2 chair",
        "user_overrides": [],
    })
    return approved


def _geometry_record(mask: np.ndarray, point_map: np.ndarray, normal_map: np.ndarray, depth_map: np.ndarray, valid: np.ndarray, raw_to_canonical: np.ndarray, raw_to_normalized: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    selection = mask & valid & np.isfinite(point_map).all(axis=2) & np.isfinite(depth_map)
    raw_points = point_map[selection]
    depths = depth_map[selection]
    median = float(np.median(depths))
    mad = float(np.median(np.abs(depths - median)))
    threshold = max(0.2, 6.0 * 1.4826 * mad)
    keep = np.abs(depths - median) <= threshold
    raw_points, depths = raw_points[keep], depths[keep]
    points = _transform(raw_points, raw_to_canonical)
    normalized = _transform(raw_points, raw_to_normalized)
    normals, extraction = extract_masked_normals(normal_map, mask, valid, raw_to_canonical, boundary_width=2)
    clusters = cluster_normals_spherical(normals)
    labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    center = np.median(points, axis=0)
    p2, p98 = np.percentile(points, [2, 98], axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    local = centered @ eigenvectors
    lo, hi = np.percentile(local, [2, 98], axis=0)
    obb_confidence = float(np.clip((eigenvalues[0] - eigenvalues[-1]) / max(1e-9, eigenvalues[0]), 0.0, 1.0))
    record = {
        "mask_pixel_count": int(mask.sum()),
        "valid_pixel_count": int(selection.sum()),
        "valid_pixel_percentage": float(selection.sum() / max(1, mask.sum()) * 100),
        "component_count": int(component_count),
        "visible_point_count": int(len(points)),
        "rejected_outlier_count": int(len(keep) - int(keep.sum())),
        "robust_center_canonical": center.tolist(),
        "raw_visible_aabb": {"min": raw_points.min(0).tolist(), "max": raw_points.max(0).tolist()},
        "normalized_visible_aabb": {"min": normalized.min(0).tolist(), "max": normalized.max(0).tolist()},
        "canonical_robust_aabb_p2_p98": {"min": p2.tolist(), "max": p98.tolist(), "dimensions": (p98 - p2).tolist()},
        "candidate_obb": {"center": center.tolist(), "rotation_matrix": eigenvectors.tolist(), "dimensions_p2_p98": (hi - lo).tolist(), "eigenvalues": eigenvalues.tolist(), "confidence": obb_confidence, "stable": obb_confidence >= 0.25},
        "depth_statistics": {"minimum": float(depths.min()), "median": float(np.median(depths)), "maximum": float(depths.max()), "p10": float(np.percentile(depths, 10)), "p90": float(np.percentile(depths, 90)), "mad": mad},
        "normal_extraction": extraction,
        "normal_clusters": clusters,
        "geometry_confidence": float(np.clip(0.45 * selection.sum() / max(1, mask.sum()) + 0.25 * min(1.0, len(points) / 1000) + 0.3 * max((item["confidence"] for item in clusters), default=0.0), 0.0, 1.0)),
        "warnings": [],
    }
    if component_count > 1:
        record["warnings"].append(f"{component_count} disconnected visible components retained as one semantic object")
    if record["valid_pixel_percentage"] < 90:
        record["warnings"].append("limited valid MoGe coverage")
    return record, points, depths, center, extraction, clusters


def _pose_new(label: str, geometry: dict[str, Any], points: np.ndarray, depths: np.ndarray, mask: np.ndarray, axis: dict[str, Any], clusters: list[dict[str, Any]], camera: dict[str, Any], room: dict[str, Any]) -> dict[str, Any]:
    wall = label in WALL_LABELS
    irregular = label in IRREGULAR_LABELS
    center = np.asarray(geometry["robust_center_canonical"], dtype=float)
    support_target = "plane_floor"
    strategy = "irregular_object_coarse_proxy" if irregular else "gravity_upright_proxy"
    wall_support: dict[str, Any] | None = None
    if wall:
        wall_name = "plane_left_wall" if label in LEFT_WALL_LABELS else "plane_right_wall"
        wall_support = _wall_frame(room, wall_name)
        rotation = np.asarray(wall_support["rotation"])
        local = (points - center) @ rotation
        lo, hi = np.percentile(local, [5, 95], axis=0)
        dimensions = np.maximum(hi - lo, [0.08, 0.08, 0.03])
        dimensions[2] = float(np.clip(np.percentile(depths, 90) - np.percentile(depths, 10), 0.03, 0.08))
        inner = np.asarray(wall_support["inner_point"])
        normal = np.asarray(wall_support["normal"])
        center = center + (dimensions[2] / 2 - float(np.dot(center - inner, normal))) * normal
        center, dimensions = _fit_visible_box(center, dimensions, rotation, mask, camera, "wall", wall_support)
        support_target = wall_name
        strategy = "wall_attached_thin_proxy"
        normal_agreement = float(abs(np.dot(rotation[:, 2], normal)))
        gravity_error = 0.0
    else:
        centered_xy = points[:, :2] - np.median(points[:, :2], axis=0)
        _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
        base_yaw = math.atan2(vh[0, 1], vh[0, 0])
        dominant_vertical = next((item for item in clusters if item["classification"] == "vertical"), None)
        normal = np.asarray(dominant_vertical["center"]) if dominant_vertical else None
        candidate_records = []
        robust_dimensions = None
        for index, yaw in enumerate((base_yaw, base_yaw + math.pi / 2, base_yaw + math.pi, base_yaw + 3 * math.pi / 2)):
            candidate = _rz(yaw)
            local = (points - center) @ candidate
            lo, hi = np.percentile(local, [5, 95], axis=0)
            dimensions = np.maximum(hi - lo, [0.08, 0.08, 0.08])
            dimensions[1] = float(np.clip(dimensions[1], 0.06, max(0.12, dimensions[0])))
            score = score_rotation_candidate(candidate, center=center, dimensions=dimensions, target_axis_angle=axis["dominant_axis_angle_degrees"], target_width=axis["oriented_width_pixels"], target_height=axis["oriented_height_pixels"], canonical_to_camera=camera["canonical_to_camera"], intrinsics=camera["intrinsics"], image_shape=camera["image_shape"], dominant_normal=normal, gravity_up=np.asarray([0, 0, 1]))
            candidate_records.append((score["score"], index, candidate, dimensions, score))
        _, _, rotation, dimensions, selected_score = min(candidate_records, key=lambda item: (item[0], item[1]))
        center[2] = dimensions[2] / 2
        center, dimensions = _fit_visible_box(center, dimensions, rotation, mask, camera, "floor")
        normal_agreement = float(max(0.0, 1.0 - selected_score["normal_alignment_error_degrees"] / 90.0)) if normal is not None else 0.25
        gravity_error = 0.0
    projected = _project_cuboid(center, dimensions, rotation, camera)
    metrics = _projection_metrics(projected, mask, axis)
    pose_confidence = float(np.clip(0.28 * geometry["geometry_confidence"] + 0.22 * axis["orientation_confidence"] + 0.2 * normal_agreement + 0.2 * metrics["mask_axis_agreement"] + 0.1 * min(1.0, metrics["bbox_iou"] / 0.7), 0.0, 1.0))
    if irregular:
        pose_confidence *= 0.72
    contact = {"target": support_target, "estimated_gap": 0.0, "consistent": True, "method": "constructed deterministic contact"}
    return {
        "center": center.tolist(),
        "dimensions": dimensions.tolist(),
        "rotation_matrix": rotation.tolist(),
        "rotation_quaternion_wxyz": matrix_to_quaternion_wxyz(rotation),
        "transform_matrix": _matrix(center, dimensions, rotation),
        "primitive_type": "cube",
        "pose_strategy": strategy,
        "support_target": support_target,
        "projection_metrics": metrics,
        "masked_normal_agreement": normal_agreement,
        "gravity_alignment_error_degrees": gravity_error,
        "support_contact": contact,
        "pose_confidence": pose_confidence,
        "collision_risks": [],
    }


def _draw_outline(image: Image.Image, points: list[list[float]], color: tuple[int, int, int], width: int = 2) -> None:
    draw = ImageDraw.Draw(image)
    hull = _convex_hull(points)
    draw.line([tuple(point) for point in [*hull, hull[0]]], fill=color, width=width)


def _diagnostics(source: Image.Image, mask: np.ndarray, axis: dict[str, Any], clusters: list[dict[str, Any]], pose: dict[str, Any], output: Path, label: str, object_id: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    overlay = source.copy().convert("RGBA")
    tint = Image.new("RGBA", source.size, (255, 50, 80, 0))
    tint.putalpha(Image.fromarray(np.uint8(mask) * 110))
    overlay = Image.alpha_composite(overlay, tint).convert("RGB")
    ImageDraw.Draw(overlay).text((6, 6), f"{label}\n{object_id}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    overlay.save(output / "source_mask_overlay.png")
    _write_mask_axis_overlay(mask, axis, output / "dominant_mask_axes.png")
    _write_normal_visualization(clusters, output / "masked_normal_visualization.png")
    projected = source.copy().convert("RGB")
    _draw_outline(projected, pose["projection_metrics"]["projected_hull"], (40, 255, 110), 3)
    projected.save(output / "projected_primitive_overlay.png")


def _global_visuals(source: Image.Image, inventory: list[dict[str, Any]], poses: list[dict[str, Any]], output: Path) -> None:
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 117), (128, 128, 128)]
    numbered = source.copy().convert("RGBA")
    axes_image = source.copy().convert("RGB")
    all_projected = source.copy().convert("RGB")
    batch = source.copy().convert("RGB")
    review = source.copy().convert("RGB")
    for index, (item, pose) in enumerate(zip(inventory, poses), start=1):
        mask = _mask(REPO_ROOT / item["source_export_path"])
        tint = Image.new("RGBA", source.size, (*colors[index - 1], 0))
        tint.putalpha(Image.fromarray(np.uint8(mask) * 75))
        numbered = Image.alpha_composite(numbered, tint)
        y, x = np.nonzero(mask)
        ImageDraw.Draw(numbered).text((int(np.median(x)), int(np.median(y))), str(index), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        axis = pose["mask_axis"]
        cx, cy = axis["center_xy"]
        angle = math.radians(axis["dominant_axis_angle_degrees"])
        half = axis["oriented_width_pixels"] / 2
        ImageDraw.Draw(axes_image).line([(cx - math.cos(angle) * half, cy - math.sin(angle) * half), (cx + math.cos(angle) * half, cy + math.sin(angle) * half)], fill=colors[index - 1], width=2)
        _draw_outline(all_projected, pose["projection_metrics"]["projected_hull"], colors[index - 1], 2)
        if pose["classification"] == "batch_ready":
            _draw_outline(batch, pose["projection_metrics"]["projected_hull"], colors[index - 1], 3)
        if pose["classification"] == "blender_review_required":
            _draw_outline(review, pose["projection_metrics"]["projected_hull"], colors[index - 1], 3)
    numbered.convert("RGB").save(output / "numbered_full_object_overlay.png")
    axes_image.save(output / "all_mask_axes_overlay.png")
    all_projected.save(output / "all_projected_primitives_overlay.png")
    batch.save(output / "batch_ready_overlay.png")
    review.save(output / "review_required_overlay.png")
    normal = Image.new("RGB", (900, 700), (18, 18, 22))
    draw = ImageDraw.Draw(normal)
    draw.text((20, 15), "Normal/orientation confidence summary", fill=(255, 255, 255))
    for index, pose in enumerate(poses):
        y = 48 + index * 31
        confidence = pose["pose_confidence"]
        draw.text((20, y), f"{index + 1:02d} {pose['semantic_label'][:28]}", fill=(220, 220, 225))
        draw.rectangle((310, y, 310 + int(500 * confidence), y + 16), fill=(40, 200, 240))
        draw.text((820, y), f"{confidence:.2f}", fill=(220, 220, 225))
    normal.save(output / "normal_orientation_summary.png")
    panels = [numbered.convert("RGB"), axes_image, all_projected, batch, review]
    overview = Image.new("RGB", (447 * 3, 447 * 2), (10, 10, 12))
    for index, panel in enumerate(panels):
        overview.paste(panel, ((index % 3) * 447, (index // 3) * 447))
    overview.save(output / "full_scene_plan_overview.png")


def build(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    metadata, records = validate_export()
    fixture = create_fixture(metadata, records)
    output.mkdir(parents=True, exist_ok=True)
    base_plan = _load(ROOT / "outputs" / "office_test" / "primitive_plan" / "primitive_scene_plan.json")
    scene_evidence = _load(ROOT / "outputs" / "office_test" / "geometry" / "scene_evidence.json")
    room = _load(ROOT / "outputs" / "office_test" / "blender_execution" / "room_corrected" / "room_camera_validation.json")
    cumulative = _load(ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "05_06_chair_box" / "final" / "cumulative_scene_validation.json")
    approved = _approved_transforms(cumulative)
    camera = _corrected_camera(base_plan, room)
    raw_to_canonical = np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"], dtype=float)
    raw_to_normalized = np.asarray(scene_evidence["transforms"]["raw_moge_to_scene_normalized"], dtype=float)
    with np.load(ROOT / "outputs" / "office_test" / "moge" / "geometry.npz") as data:
        point_map, normal_map, depth_map, valid = data["points"], data["normal"], data["depth"], data["valid_mask"]
    source = Image.open(SOURCE_IMAGE).convert("RGB")
    geometry_records: list[dict[str, Any]] = []
    pose_records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for item in records:
        object_id, label = item["object_id"], item["semantic_label"]
        mask = _mask(REPO_ROOT / item["source_export_path"])
        geometry, points, depths, robust_center, extraction, clusters = _geometry_record(mask, point_map, normal_map, depth_map, valid, raw_to_canonical, raw_to_normalized)
        axis = extract_mask_axis(mask)
        y, x = np.nonzero(mask)
        geometry.update({"object_id": object_id, "semantic_label": label, "mask_centroid_xy": [float(x.mean()), float(y.mean())], "mask_axis": axis, "projected_mask_convex_hull": _convex_hull(np.column_stack([x, y])).tolist()})
        if object_id in approved:
            accepted = approved[object_id]
            center = np.asarray(accepted["center"]);dimensions = np.asarray(accepted["dimensions"]);rotation = quaternion_to_matrix(accepted["quaternion_wxyz"])
            projected = _project_cuboid(center, dimensions, rotation, camera)
            pose = {
                "center": center.tolist(), "dimensions": dimensions.tolist(), "rotation_matrix": rotation.tolist(), "rotation_quaternion_wxyz": accepted["quaternion_wxyz"], "transform_matrix": _matrix(center, dimensions, rotation), "primitive_type": "cube", "pose_strategy": "approved_transform_preserved", "support_target": next((x["support_contact_status"] for x in cumulative["semantic_inventory"] if x["object_id"] == object_id), "approved"), "projection_metrics": _projection_metrics(projected, mask, axis), "masked_normal_agreement": None, "gravity_alignment_error_degrees": 0.0, "support_contact": {"consistent": True, "method": "approved Blender validation"}, "pose_confidence": 1.0, "collision_risks": [], "approval_source": accepted["approval_source"], "user_overrides": accepted["user_overrides"], "classification": "existing_approved_object",
            }
        else:
            pose = _pose_new(label, geometry, points, depths, mask, axis, clusters, camera, room)
            pose_conf = pose["pose_confidence"];metrics = pose["projection_metrics"]
            collision_warning = []
            # Conservative image-space overlap warning against protected masks.
            for protected in records:
                if protected["object_id"] not in APPROVED_IDS:
                    continue
                a, b = np.asarray(item["bbox_xyxy_inclusive"]), np.asarray(protected["bbox_xyxy_inclusive"])
                iw=max(0,min(a[2],b[2])-max(a[0],b[0])+1);ih=max(0,min(a[3],b[3])-max(a[1],b[1])+1)
                if iw*ih/min(item["area"],protected["area"])>0.15:
                    collision_warning.append(f"2D overlap with protected {protected['semantic_label']} requires Blender collision review")
            pose["collision_risks"] = collision_warning
            if metrics["bbox_iou"] >= 0.62 and metrics["centroid_error_pixels"] <= 5 and pose_conf >= 0.48 and not irregular_or_weak(label, geometry):
                classification = "batch_ready"
            elif metrics["bbox_iou"] >= 0.42 and geometry["valid_pixel_percentage"] >= 75 and pose_conf >= 0.28:
                classification = "blender_review_required"
            else:
                classification = "insufficient_geometry"
            if label in IRREGULAR_LABELS:
                classification = "blender_review_required" if classification != "insufficient_geometry" else classification
            if collision_warning and classification == "batch_ready":
                classification = "blender_review_required"
            pose["classification"] = classification
        pose.update({"object_id": object_id, "semantic_label": label, "mask_filename": item["mask_filename"], "mask_axis": axis, "geometry_confidence": geometry["geometry_confidence"], "warnings": list(geometry["warnings"])})
        if pose["classification"] == "existing_approved_object":
            pose["warnings"].append("protected approved transform; not eligible for batch update")
        geometry_records.append(geometry)
        pose_records.append(pose)
        inventory.append({**item, "inventory_index": len(inventory) + 1, "plan_classification": pose["classification"]})
        if object_id not in APPROVED_IDS:
            diagnostic = output / "per_object" / object_id
            _diagnostics(source, mask, axis, clusters, pose, diagnostic, label, object_id)
            (diagnostic / "geometry_pose_summary.json").write_text(json.dumps({"geometry": geometry, "pose": pose, "warnings": [*geometry["warnings"], *pose["collision_risks"]]}, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Protected audit records exact source transforms; nothing is mutated offline.
    protected = []
    for object_id, transform in approved.items():
        pose = next(item for item in pose_records if item["object_id"] == object_id)
        proposed = {key: pose[key] for key in ("center", "dimensions", "rotation_quaternion_wxyz")}
        current = {key: transform[key] for key in ("center", "dimensions", "quaternion_wxyz")}
        protected.append({"object_id": object_id, "current": current, "planned": {"center": proposed["center"], "dimensions": proposed["dimensions"], "quaternion_wxyz": proposed["rotation_quaternion_wxyz"]}, "maximum_absolute_delta": float(max(np.max(np.abs(np.asarray(current[key]) - np.asarray({"center": proposed["center"], "dimensions": proposed["dimensions"], "quaternion_wxyz": proposed["rotation_quaternion_wxyz"]}[key]))) for key in current)), "passed": True})
    audit = {"schema_version": "1.0", "protected_objects": protected, "room_geometry": {"status": "unchanged_reference", "source": "room_camera_validation.json"}, "corrected_perspective_camera": {"status": "unchanged_reference", "source": "room_camera_validation.json"}, "passed": all(item["passed"] and item["maximum_absolute_delta"] <= 2e-6 for item in protected)}

    classifications = {name: [item for item in pose_records if item["classification"] == name] for name in ("batch_ready", "blender_review_required", "insufficient_geometry", "excluded_structure", "existing_approved_object")}
    plan = {"schema_version": "1.0", "scene_id": "office_test_full", "source_fixture": "inputs/office_test_full/manifest.json", "source_fixture_sha256": _sha(FIXTURE / "manifest.json"), "camera": {"source": camera["source"], "shift_pixels": camera["shift_pixels"].tolist()}, "structural_planes_source": "outputs/office_test/geometry/scene_evidence.json", "objects": pose_records, "object_count": len(pose_records), "approved_object_count": len(classifications["existing_approved_object"]), "new_object_count": len(pose_records) - len(classifications["existing_approved_object"])}
    class_report = {"schema_version": "1.0", "counts": {key: len(value) for key, value in classifications.items()}, "classifications": {key: [{"object_id": item["object_id"], "semantic_label": item["semantic_label"], "reason": classification_reason(item)} for item in value] for key, value in classifications.items()}}
    batches = _batch_manifest(classifications, approved)
    quality = {"schema_version": "1.0", "quality_gates": {"all_20_masks_in_inventory": len(inventory) == 20, "unique_stable_ids": len({item["object_id"] for item in inventory}) == 20, "all_masks_valid": all(item["binary"] and item["non_empty"] and item["width"] == 447 and item["height"] == 447 for item in inventory), "every_new_object_classified_once": sum(len(classifications[key]) for key in ("batch_ready", "blender_review_required", "insufficient_geometry", "excluded_structure")) == 14, "approved_transforms_unchanged": audit["passed"], "one_cube_per_object": all(item["primitive_type"] == "cube" for item in pose_records), "no_mask_silently_excluded": len(inventory) == len(records)}, "warnings": [warning for item in pose_records for warning in [*item["warnings"], *item["collision_risks"]]], "passed": False}
    quality["passed"] = all(quality["quality_gates"].values())
    outputs = {
        "full_object_inventory.json": {"schema_version": "1.0", "scene_id": "office_test_full", "source_image": fixture["source_image"], "objects": inventory, "object_count": len(inventory)},
        "full_object_geometry.json": {"schema_version": "1.0", "scene_id": "office_test_full", "objects": geometry_records, "object_count": len(geometry_records)},
        "full_scene_pose_plan.json": plan,
        "protected_transform_audit.json": audit,
        "classification_report.json": class_report,
        "blender_batch_manifest.json": batches,
        "quality_report.json": quality,
    }
    for filename, value in outputs.items():
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(output, outputs)
    _global_visuals(source, inventory, pose_records, output)
    return {"fixture": fixture, "outputs": outputs}


def irregular_or_weak(label: str, geometry: dict[str, Any]) -> bool:
    return label in IRREGULAR_LABELS or geometry["component_count"] > 3 or geometry["geometry_confidence"] < 0.45


def classification_reason(item: dict[str, Any]) -> str:
    if item["classification"] == "existing_approved_object":
        return "Approved transform is frozen and excluded from new-object batching."
    if item["classification"] == "batch_ready":
        return "Conservative projection, geometry, orientation, and support thresholds passed."
    if item["classification"] == "blender_review_required":
        return "Plausible coarse proxy, but irregular shape, overlap, support, or orientation needs visual review."
    if item["classification"] == "insufficient_geometry":
        return "Valid-pixel, projection, or pose-confidence evidence is below conservative thresholds."
    return "Authoritative metadata identifies structure/background rather than a semantic object."


def _batch_manifest(groups: dict[str, list[dict[str, Any]]], approved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ready = groups["batch_ready"]
    floor = [item for item in ready if item["support_target"] == "plane_floor"]
    wall = [item for item in ready if item["support_target"] in {"plane_left_wall", "plane_right_wall"}]
    batches = []
    if floor:
        batches.append({"batch_id": "floor_supported_cabinets_boxes", "order": 1, "object_ids": [item["object_id"] for item in floor], "policy": "small controlled batch; validate each object independently"})
    if wall:
        batches.append({"batch_id": "wall_mounted_decorations_radiators", "order": 2, "object_ids": [item["object_id"] for item in wall], "policy": "validate corrected wall attachment and conservative thickness"})
    return {"schema_version": "1.0", "objects_already_present_and_frozen": [{"object_id": key, **value} for key, value in approved.items()], "batch_ready_objects": [{key: item[key] for key in ("object_id", "semantic_label", "center", "dimensions", "rotation_quaternion_wxyz", "support_target", "pose_confidence", "collision_risks")} for item in ready], "individual_review_objects": [{key: item[key] for key in ("object_id", "semantic_label", "center", "dimensions", "rotation_quaternion_wxyz", "support_target", "pose_confidence", "collision_risks")} for item in groups["blender_review_required"]], "insufficient_geometry": [{"object_id": item["object_id"], "semantic_label": item["semantic_label"]} for item in groups["insufficient_geometry"]], "excluded_objects": [{"object_id": item["object_id"], "semantic_label": item["semantic_label"]} for item in groups["excluded_structure"]], "recommended_batches": batches, "recommended_ordering": ["floor-supported cabinets and closed boxes", "wall-mounted decorations and radiators", "weak or irregular objects individually"]}


def _write_markdown(output: Path, values: dict[str, dict[str, Any]]) -> None:
    inventory = values["full_object_inventory.json"]
    lines = ["# Full object inventory", "", f"Authoritative final SAM masks: {inventory['object_count']}", "", "| # | Label | Object ID | Fixture classification |", "|---:|---|---|---|"]
    for item in inventory["objects"]:
        lines.append(f"| {item['inventory_index']} | {item['semantic_label']} | `{item['object_id']}` | {item['fixture_classification']} |")
    (output / "full_object_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plan = values["full_scene_pose_plan.json"]
    lines = ["# Full scene pose plan", "", f"- Objects: {plan['object_count']}", f"- Approved/frozen: {plan['approved_object_count']}", f"- New: {plan['new_object_count']}", "", "| Label | Classification | Strategy | IoU | Confidence |", "|---|---|---|---:|---:|"]
    for item in plan["objects"]:
        lines.append(f"| {item['semantic_label']} | {item['classification']} | {item['pose_strategy']} | {item['projection_metrics']['bbox_iou']:.3f} | {item['pose_confidence']:.3f} |")
    (output / "full_scene_pose_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    classification = values["classification_report.json"]
    lines = ["# Classification report", ""]
    for name, items in classification["classifications"].items():
        lines += [f"## {name} ({len(items)})", ""] + [f"- {item['semantic_label']} (`{item['object_id']}`): {item['reason']}" for item in items] + [""]
    (output / "classification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality = values["quality_report.json"]
    lines = ["# Quality report", "", f"- Result: {'PASS' if quality['passed'] else 'FAIL'}", "", "## Gates", ""] + [f"- {key}: {'PASS' if passed else 'FAIL'}" for key, passed in quality["quality_gates"].items()] + ["", "## Warnings", ""] + [f"- {warning}" for warning in quality["warnings"]]
    (output / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build(args.output_dir.resolve())
    report = result["outputs"]["classification_report.json"]
    print(json.dumps({"mask_count": result["fixture"]["mask_count"], "classification_counts": report["counts"], "output": str(args.output_dir)}, indent=2))
