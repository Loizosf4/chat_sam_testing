"""Create the deterministic offline v2 primitive pose-refinement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .pose_refinement import (
    cluster_normals_spherical,
    conservative_scale_fit,
    cuboid_corners,
    extract_mask_axis,
    extract_masked_normals,
    matrix_to_quaternion_wxyz,
    minimum_area_rectangle,
    normalize_vectors,
    orthonormal_basis_from_forward_up,
    project_world_points,
    projected_cuboid_rectangle,
    rotation_candidates_from_basis,
    score_rotation_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "pose_refinement_v2"
OBJECT_IDS = {
    "left_tall_filing_cabinet": "e971e9fbbf5746eea108d07bfbb5764a",
    "desk": "b9858c1f45f443a88d84229a1824d339",
    "desk_chair": "9171a7b4d4e142ca936f2564200d0bdb",
    "desktop_box": "1064a3a09c4c427e80caa227d17a7cc1",
    "coat_rack": "4ec402f9a71d4ea7925b487ac7e11249",
    "wall_light_fixture": "49fd02b50a914e8198bacf4644bc377f",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_to_matrix(quaternion: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    length = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / length, x / length, y / length, z / length
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _transform_matrix(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(dimensions)
    matrix[:3, 3] = center
    return matrix.tolist()


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _camera(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    camera = next(item for item in plan["camera_candidates"] if item["camera_id"] == "camera_perspective_moge")
    perspective = camera["perspective"]
    return (
        np.asarray(camera["canonical_to_camera_transform"], dtype=np.float64),
        np.asarray(perspective["normalized_intrinsics"], dtype=np.float64),
        (int(perspective["image_height"]), int(perspective["image_width"])),
    )


def _project(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray, camera: tuple[np.ndarray, np.ndarray, tuple[int, int]]) -> np.ndarray:
    extrinsics, intrinsics, shape = camera
    return project_world_points(cuboid_corners(center, dimensions, rotation), extrinsics, intrinsics, shape)


def _bbox_metrics(projected: np.ndarray, mask: np.ndarray) -> dict[str, float | list[float]]:
    y, x = np.nonzero(mask)
    target = np.asarray([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float64)
    bbox = np.asarray([projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()])
    intersection = max(0.0, min(bbox[2], target[2]) - max(bbox[0], target[0])) * max(0.0, min(bbox[3], target[3]) - max(bbox[1], target[1]))
    area_a = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    area_b = (target[2] - target[0]) * (target[3] - target[1])
    return {
        "projected_bbox": bbox.tolist(),
        "mask_bbox": target.tolist(),
        "bbox_iou": float(intersection / max(1e-9, area_a + area_b - intersection)),
        "centroid_error_pixels": float(np.linalg.norm((bbox[:2] + bbox[2:]) / 2 - (target[:2] + target[2:]) / 2)),
    }


def _fit_projected_center_xy(
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation: np.ndarray,
    mask: np.ndarray,
    camera: tuple[np.ndarray, np.ndarray, tuple[int, int]],
    *,
    iterations: int = 8,
) -> np.ndarray:
    """Solve only canonical XY so the projected cuboid center matches the mask."""
    fitted = np.asarray(center, dtype=np.float64).copy()
    y, x = np.nonzero(mask)
    target = np.asarray([(x.min() + x.max() + 1) / 2, (y.min() + y.max() + 1) / 2])

    def projected_center(value: np.ndarray) -> np.ndarray:
        projected = _project(value, dimensions, rotation, camera)
        return np.asarray([(projected[:, 0].min() + projected[:, 0].max()) / 2, (projected[:, 1].min() + projected[:, 1].max()) / 2])

    for _ in range(iterations):
        current = projected_center(fitted)
        residual = current - target
        if np.linalg.norm(residual) < 1e-5:
            break
        jacobian = np.zeros((2, 2))
        for axis in range(2):
            perturbed = fitted.copy()
            perturbed[axis] += 1e-4
            jacobian[:, axis] = (projected_center(perturbed) - current) / 1e-4
        fitted[:2] -= np.linalg.solve(jacobian, residual)
    return fitted


def _normal_candidate(clusters: list[dict[str, Any]], classification: str) -> np.ndarray | None:
    eligible = [item for item in clusters if item["classification"] == classification]
    if not eligible:
        return None
    return np.asarray(max(eligible, key=lambda item: (item["confidence"], item["point_count"]))["center"], dtype=np.float64)


def _pca_horizontal_normal(points: np.ndarray) -> np.ndarray:
    centered = points[:, :2] - np.median(points[:, :2], axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray([vh[-1, 0], vh[-1, 1], 0.0])
    normal /= np.linalg.norm(normal)
    return normal


def _draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int, int, int], width: int = 2) -> None:
    ordered = minimum_area_rectangle(points)["corners"]
    draw.line([tuple(map(float, point)) for point in [*ordered, ordered[0]]], fill=color, width=width)


def _write_mask_axis_overlay(mask: np.ndarray, axis: dict[str, Any], path: Path) -> None:
    image = Image.fromarray(np.uint8(mask) * 80, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    if axis.get("valid"):
        corners = [tuple(point) for point in axis["corners_xy"]]
        draw.line([*corners, corners[0]], fill=(255, 170, 0), width=2)
        cx, cy = axis["center_xy"]
        length = axis["oriented_width_pixels"] / 2
        angle = math.radians(axis["dominant_axis_angle_degrees"])
        direction = (math.cos(angle) * length, math.sin(angle) * length)
        draw.line([(cx - direction[0], cy - direction[1]), (cx + direction[0], cy + direction[1])], fill=(0, 255, 255), width=2)
    image.save(path)


def _write_normal_visualization(clusters: list[dict[str, Any]], path: Path) -> None:
    image = Image.new("RGB", (512, 256), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    origin = (128, 128)
    draw.ellipse((28, 28, 228, 228), outline=(90, 90, 100), width=2)
    colors = [(0, 220, 255), (255, 170, 0), (120, 255, 120), (255, 100, 180)]
    for index, cluster in enumerate(clusters):
        x, y, z = cluster["center"]
        end = (origin[0] + 95 * x, origin[1] - 95 * y)
        color = colors[index % len(colors)]
        draw.line([origin, end], fill=color, width=max(2, int(8 * cluster["coverage_ratio"])))
        draw.text((270, 28 + index * 48), f"{index}: {cluster['classification']} cov={cluster['coverage_ratio']:.3f} p90={cluster['angular_dispersion_degrees']['p90']:.1f}", fill=color)
        draw.rectangle((270, 48 + index * 48, 270 + int(180 * (z + 1) / 2), 56 + index * 48), fill=color)
    image.save(path)


def _write_projection(source: Image.Image, original: np.ndarray, refined: np.ndarray, path: Path, label: str) -> None:
    image = source.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    _draw_polyline(draw, original, (255, 70, 70), 2)
    _draw_polyline(draw, refined, (60, 255, 120), 2)
    draw.rectangle((4, 4, 250, 24), fill=(0, 0, 0))
    draw.text((8, 7), f"{label}: red=original green=refined", fill=(255, 255, 255))
    image.save(path)


def _accepted_transforms() -> dict[str, dict[str, Any]]:
    cabinet = _load_json(ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "01_filing_cabinet" / "final" / "object_validation.json")["final_blender_transform"]
    desk_report = _load_json(ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "02_desk" / "approved_candidate_c" / "object_validation.json")
    desk = desk_report["approved_transform"]
    yaw = math.radians(desk["yaw_degrees"])
    return {
        "left_tall_filing_cabinet": {"center": cabinet["location"], "dimensions": cabinet["dimensions"], "quaternion": cabinet["rotation_quaternion_wxyz"], "source": "accepted_filing_cabinet_validation"},
        "desk": {"center": desk["center"], "dimensions": desk["dimensions"], "quaternion": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], "source": "approved_candidate_c"},
    }


def _right_wall_rotation(room_report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(room_report["corrected_structures"]["plane_right_wall"]["matrix_world"], dtype=np.float64)
    columns = [matrix[:3, index] for index in range(3)]
    normal = min(columns, key=np.linalg.norm)
    normal /= np.linalg.norm(normal)
    if normal[0] > 0:
        normal *= -1.0
    up = np.asarray([0.0, 0.0, 1.0])
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent)
    rotation = np.column_stack([tangent, up, normal])
    if np.linalg.det(rotation) < 0:
        tangent *= -1
        rotation = np.column_stack([tangent, up, normal])
    return rotation, normal


def refine_plan(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run the full deterministic offline refinement and write all artifacts."""
    manifest = _load_json(ROOT / "inputs" / "office_test" / "manifest.json")
    base_plan = _load_json(ROOT / "outputs" / "office_test" / "primitive_plan" / "primitive_scene_plan.json")
    scene_evidence = _load_json(ROOT / "outputs" / "office_test" / "geometry" / "scene_evidence.json")
    object_geometry = _load_json(ROOT / "outputs" / "office_test" / "geometry" / "object_geometry.json")
    room_report = _load_json(ROOT / "outputs" / "office_test" / "blender_execution" / "room_corrected" / "room_camera_validation.json")
    accepted = _accepted_transforms()
    camera = _camera(base_plan)
    raw_to_canonical = np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"], dtype=np.float64)
    with np.load(ROOT / "outputs" / "office_test" / "moge" / "geometry.npz") as geometry:
        point_map = np.asarray(geometry["points"], dtype=np.float64)
        depth_map = np.asarray(geometry["depth"], dtype=np.float64)
        normal_map = np.asarray(geometry["normal"], dtype=np.float64)
        valid_map = np.asarray(geometry["valid_mask"], dtype=bool)
    source = Image.open(ROOT / "inputs" / "office_test" / "office_scene.jpg").convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_object_dir = output_dir / "per_object"
    per_object_dir.mkdir(exist_ok=True)
    original_by_id = {item["object_id"]: item for item in base_plan["semantic_primitives"]}
    manifest_by_id = {item["object_id"]: item for item in manifest["objects"]}
    geometry_by_id = {item["object_id"]: item for item in object_geometry["objects"]}
    refined_items: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    gravity = np.asarray([0.0, 0.0, 1.0])
    desk_accepted_rotation = quaternion_to_matrix(accepted["desk"]["quaternion"])
    wall_rotation, wall_normal = _right_wall_rotation(room_report)

    for label, object_id in OBJECT_IDS.items():
        original = original_by_id[object_id]
        mask = _mask(ROOT / "inputs" / "office_test" / manifest_by_id[object_id]["mask_filename"])
        axis = extract_mask_axis(mask)
        normals, normal_extraction = extract_masked_normals(normal_map, mask, valid_map, raw_to_canonical, boundary_width=2)
        clusters = cluster_normals_spherical(normals)
        selection = mask & valid_map & np.isfinite(point_map).all(axis=2)
        raw_points = point_map[selection]
        points = _transform_points(raw_points, raw_to_canonical)
        depths = depth_map[selection]
        center = np.median(points, axis=0)
        original_center = np.asarray(original["center"], dtype=np.float64)
        original_dimensions = np.asarray(original["dimensions"], dtype=np.float64)
        original_rotation = np.asarray(original["rotation_matrix"], dtype=np.float64)
        method = "masked_normals_gravity_mask_axis"
        fallback_reason: str | None = None
        candidates: list[dict[str, Any]] = []
        selected_rotation = original_rotation.copy()
        selected_dimensions = original_dimensions.copy()
        selected_center = center.copy()
        confidence = float(np.clip(axis.get("orientation_confidence", 0.0) * (clusters[0]["confidence"] if clusters else 0.0), 0.0, 1.0))
        scale_record: dict[str, Any] = {"original_dimensions": original_dimensions.tolist(), "refined_dimensions": original_dimensions.tolist(), "policy": "unchanged"}
        orientation_evidence_details: dict[str, Any] = {}

        if label in accepted:
            record = accepted[label]
            selected_center = np.asarray(record["center"], dtype=np.float64)
            selected_dimensions = np.asarray(record["dimensions"], dtype=np.float64)
            selected_rotation = quaternion_to_matrix(record["quaternion"])
            method = "accepted_transform_preserved_with_offline_evidence_comparison"
            fallback_reason = f"{record['source']} is authoritative and must not be overwritten"
            confidence = 1.0
            scale_record = {"original_dimensions": original_dimensions.tolist(), "refined_dimensions": selected_dimensions.tolist(), "policy": "accepted transform preserved", "correction": (selected_dimensions - original_dimensions).tolist()}
            if label == "desk":
                support_record = next(item for item in scene_evidence["object_support_surface_candidates"] if item.get("object_id") == object_id)["visible_support_patch"]
                automatic = minimum_area_rectangle(np.asarray(support_record["boundary_canonical_xy"], dtype=np.float64))
                accepted_yaw = math.degrees(math.atan2(selected_rotation[1, 0], selected_rotation[0, 0]))
                automatic_yaw = float(automatic["angle_degrees"])
                orientation_evidence_details = {
                    "automatic_tabletop_support_yaw_degrees": automatic_yaw,
                    "approved_candidate_c_yaw_degrees": accepted_yaw,
                    "absolute_yaw_difference_degrees": abs((automatic_yaw - accepted_yaw + 90.0) % 180.0 - 90.0),
                    "support_patch_point_count": support_record["point_count"],
                    "support_patch_confidence": support_record["confidence"],
                }
        elif label == "desk_chair":
            dominant = _normal_candidate(clusters, "vertical")
            if dominant is None:
                dominant = _pca_horizontal_normal(points)
                fallback_reason = "no stable vertical normal cluster; robust visible-point plane used"
            dominant[2] = 0.0
            dominant /= np.linalg.norm(dominant)
            basis = orthonormal_basis_from_forward_up(dominant, gravity)
            projected_points = (points - center) @ basis
            spans = np.percentile(projected_points, 90, axis=0) - np.percentile(projected_points, 10, axis=0)
            dimensions = np.asarray([max(0.18, spans[0]), 0.08, max(0.20, spans[2])])
            for index, rotation in enumerate(rotation_candidates_from_basis(basis)):
                score = score_rotation_candidate(rotation, center=center, dimensions=dimensions, target_axis_angle=axis["dominant_axis_angle_degrees"], target_width=axis["oriented_width_pixels"], target_height=axis["oriented_height_pixels"], canonical_to_camera=camera[0], intrinsics=camera[1], image_shape=camera[2], dominant_normal=dominant, gravity_up=gravity)
                candidates.append({"candidate_id": f"chair_rotation_{index}", "rotation_matrix": rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(rotation), **score})
            chosen = min(range(len(candidates)), key=lambda index: (candidates[index]["score"], candidates[index]["candidate_id"]))
            selected_rotation = np.asarray(candidates[chosen]["rotation_matrix"])
            selected_dimensions = dimensions
            selected_center = center
            projected = projected_cuboid_rectangle(selected_center, selected_dimensions, selected_rotation, *camera)
            selected_dimensions, scale_record = conservative_scale_fit(selected_dimensions, projected, axis, thickness_axis=1, depth_samples=depths)
            selected_dimensions[1] = min(selected_dimensions[1], 0.10)
            scale_record["refined_dimensions"] = selected_dimensions.tolist()
            method = "visible_backrest_normal_plane_plus_gravity_and_mask_edges"
            confidence = float(np.clip(0.45 * axis["orientation_confidence"] + 0.55 * max(item["confidence"] for item in clusters), 0.0, 1.0)) if clusters else axis["orientation_confidence"] * 0.5
        elif label == "desktop_box":
            selected_rotation = desk_accepted_rotation.copy()
            selected_dimensions = original_dimensions.copy()
            selected_center = np.asarray([np.median(points[:, 0]), np.median(points[:, 1]), accepted["desk"]["center"][2] + accepted["desk"]["dimensions"][2] / 2 + original_dimensions[2] / 2])
            selected_center = _fit_projected_center_xy(selected_center, selected_dimensions, selected_rotation, mask, camera)
            desk_center = np.asarray(accepted["desk"]["center"], dtype=np.float64)
            desk_dimensions = np.asarray(accepted["desk"]["dimensions"], dtype=np.float64)
            local = desk_accepted_rotation.T @ (selected_center - desk_center)
            local[0] = np.clip(local[0], -desk_dimensions[0] / 2 + selected_dimensions[0] / 2 + 0.005, desk_dimensions[0] / 2 - selected_dimensions[0] / 2 - 0.005)
            local[1] = np.clip(local[1], -desk_dimensions[1] / 2 + selected_dimensions[1] / 2 + 0.005, desk_dimensions[1] / 2 - selected_dimensions[1] / 2 - 0.005)
            selected_center[:2] = (desk_center + desk_accepted_rotation @ local)[:2]
            candidates.append({"candidate_id": "desk_aligned", "rotation_matrix": selected_rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(selected_rotation), "score": 0.0, "support_consistency": 1.0})
            method = "top_side_normal_clusters_plus_approved_desk_support"
            confidence = float(np.clip(0.5 * axis["orientation_confidence"] + 0.5 * sum(item["coverage_ratio"] for item in clusters[:2]), 0.0, 1.0))
            scale_record = {"original_dimensions": original_dimensions.tolist(), "refined_dimensions": selected_dimensions.tolist(), "policy": "corrected conservative depth preserved; bottom placed on approved Candidate C top"}
        elif label == "coat_rack":
            selected_rotation = np.eye(3)
            selected_center = original_center.copy()
            selected_dimensions = original_dimensions.copy()
            method = "gravity_primary_disconnected_component_fallback"
            fallback_reason = "disconnected components and normal disagreement make precise yaw unreliable"
            confidence = float(min(0.45, axis["orientation_confidence"]))
            candidates.append({"candidate_id": "gravity_aligned_no_precise_yaw", "rotation_matrix": selected_rotation.tolist(), "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "score": 0.5})
        elif label == "wall_light_fixture":
            selected_rotation = wall_rotation
            selected_center = original_center.copy()
            selected_dimensions = original_dimensions.copy()
            method = "corrected_right_wall_normal_plus_mask_long_axis"
            confidence = float(np.clip(0.65 + 0.35 * axis["orientation_confidence"], 0.0, 1.0))
            candidates.append({"candidate_id": "corrected_wall_aligned", "rotation_matrix": selected_rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(selected_rotation), "score": 0.0, "wall_normal": wall_normal.tolist()})
            scale_record = {"original_dimensions": original_dimensions.tolist(), "refined_dimensions": selected_dimensions.tolist(), "policy": "conservative compiled thickness retained; in-wall axes corrected"}
        else:
            selected_center = original_center.copy()
            dominant = _normal_candidate(clusters, "vertical")
            if dominant is not None:
                basis = orthonormal_basis_from_forward_up(dominant, gravity)
                for index, rotation in enumerate(rotation_candidates_from_basis(basis)):
                    score = score_rotation_candidate(rotation, center=selected_center, dimensions=original_dimensions, target_axis_angle=axis["dominant_axis_angle_degrees"], target_width=axis["oriented_width_pixels"], target_height=axis["oriented_height_pixels"], canonical_to_camera=camera[0], intrinsics=camera[1], image_shape=camera[2], dominant_normal=dominant, gravity_up=gravity)
                    candidates.append({"candidate_id": f"rotation_{index}", "rotation_matrix": rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(rotation), **score})
                selected_rotation = np.asarray(min(candidates, key=lambda item: (item["score"], item["candidate_id"]))["rotation_matrix"])
            else:
                fallback_reason = "no usable vertical normal cluster; original accepted-scale orientation retained"

        if not candidates:
            candidates.append({"candidate_id": "preserved_transform", "rotation_matrix": selected_rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(selected_rotation), "score": 0.0})
        quaternion = matrix_to_quaternion_wxyz(selected_rotation)
        refined = deepcopy(original)
        refined.update({
            "center": selected_center.tolist(),
            "dimensions": selected_dimensions.tolist(),
            "rotation_matrix": selected_rotation.tolist(),
            "rotation_quaternion_wxyz": quaternion,
            "transform_matrix": _transform_matrix(selected_center, selected_dimensions, selected_rotation),
            "orientation_method": method,
            "dominant_normals": clusters,
            "mask_axis": {key: value for key, value in axis.items() if key != "components"},
            "rotation_candidates": candidates,
            "selected_rotation": {"rotation_matrix": selected_rotation.tolist(), "quaternion_wxyz": quaternion, "candidate_id": min(candidates, key=lambda item: (item["score"], item["candidate_id"]))["candidate_id"] if candidates else "preserved_transform"},
            "orientation_confidence": confidence,
            "scale_refinement": scale_record,
            "fallback_reason": fallback_reason,
            "orientation_evidence_details": orientation_evidence_details,
        })
        original_projected = _project(original_center, original_dimensions, original_rotation, camera)
        refined_projected = _project(selected_center, selected_dimensions, selected_rotation, camera)
        comparison = {"original": _bbox_metrics(original_projected, mask), "refined": _bbox_metrics(refined_projected, mask)}
        object_output = per_object_dir / object_id
        object_output.mkdir(exist_ok=True)
        (object_output / "normal_clusters.json").write_text(json.dumps({"extraction": normal_extraction, "clusters": clusters}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (object_output / "rotation_candidates.json").write_text(json.dumps({"candidates": candidates, "selected_rotation": refined["selected_rotation"]}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_mask_axis_overlay(mask, axis, object_output / "mask_axis_overlay.png")
        _write_normal_visualization(clusters, object_output / "normal_cluster_visualization.png")
        _write_projection(source, original_projected, refined_projected, object_output / "projection_comparison.png", label)
        reports.append({
            "object_id": object_id,
            "semantic_label": label,
            "original_transform": {"center": original["center"], "dimensions": original["dimensions"], "quaternion_wxyz": original["rotation_quaternion_wxyz"]},
            "refined_transform": {"center": refined["center"], "dimensions": refined["dimensions"], "quaternion_wxyz": refined["rotation_quaternion_wxyz"]},
            "normal_cluster_confidence": max((item["confidence"] for item in clusters), default=0.0),
            "mask_axis_confidence": axis.get("orientation_confidence", 0.0),
            "orientation_confidence": confidence,
            "projection_comparison": comparison,
            "scale_change": (selected_dimensions - original_dimensions).tolist(),
            "fallback_reason": fallback_reason,
        })
        refined_items.append(refined)

    refined_plan = deepcopy(base_plan)
    refined_plan["schema_version"] = "2.0"
    refined_plan["base_plan_schema_version"] = base_plan["schema_version"]
    refined_plan["refinement_version"] = "pose_refinement_v2"
    refined_plan["refinement_method"] = "deterministic_sam_moge_normals_points_depth_gravity_support_camera"
    refined_plan["semantic_primitives"] = refined_items
    refined_plan["source_plan_path"] = "outputs/office_test/primitive_plan/primitive_scene_plan.json"
    refined_plan["source_plan_sha256"] = _sha256(ROOT / refined_plan["source_plan_path"])
    refined_plan["approved_transform_policy"] = {"filing_cabinet": "preserved", "desk_candidate_c": "preserved"}
    refined_path = output_dir / "refined_primitive_scene_plan_v2.json"
    refined_path.write_text(json.dumps(refined_plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    regressions = []
    for item in reports:
        original_iou = item["projection_comparison"]["original"]["bbox_iou"]
        refined_iou = item["projection_comparison"]["refined"]["bbox_iou"]
        if refined_iou < original_iou - 0.03:
            regressions.append({
                "object_id": item["object_id"],
                "semantic_label": item["semantic_label"],
                "metric": "perspective_bbox_iou",
                "original": original_iou,
                "refined": refined_iou,
                "reason": "accepted checkpoint transform is authoritative" if item["semantic_label"] in accepted else "orientation/support refinement tradeoff",
            })
    report = {
        "schema_version": "2.0",
        "scene_id": "office_test",
        "deterministic": True,
        "offline_only": True,
        "object_count": len(refined_items),
        "stable_object_ids_preserved": {item["object_id"] for item in refined_items} == set(OBJECT_IDS.values()),
        "source_lineage_preserved": refined_plan["source_references"] == base_plan["source_references"],
        "accepted_transforms_preserved": {
            "left_tall_filing_cabinet": next(item for item in reports if item["semantic_label"] == "left_tall_filing_cabinet")["refined_transform"] == {"center": accepted["left_tall_filing_cabinet"]["center"], "dimensions": accepted["left_tall_filing_cabinet"]["dimensions"], "quaternion_wxyz": matrix_to_quaternion_wxyz(quaternion_to_matrix(accepted["left_tall_filing_cabinet"]["quaternion"]))},
            "desk": next(item for item in reports if item["semantic_label"] == "desk")["refined_transform"]["center"] == accepted["desk"]["center"],
        },
        "objects": reports,
        "regressions": regressions,
        "ready_for_offline_evaluation": True,
    }
    (output_dir / "pose_refinement_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir, refined_plan, report)
    _write_before_after(source, refined_items, original_by_id, camera, output_dir / "before_after_comparison.png")
    return {"plan": refined_plan, "report": report}


def _write_before_after(source: Image.Image, refined: list[dict[str, Any]], originals: dict[str, dict[str, Any]], camera: tuple[np.ndarray, np.ndarray, tuple[int, int]], path: Path) -> None:
    image = source.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [(255, 80, 80), (80, 255, 120), (80, 180, 255), (255, 190, 40), (220, 90, 255), (40, 255, 255)]
    for index, item in enumerate(refined):
        original = originals[item["object_id"]]
        op = _project(np.asarray(original["center"]), np.asarray(original["dimensions"]), np.asarray(original["rotation_matrix"]), camera)
        rp = _project(np.asarray(item["center"]), np.asarray(item["dimensions"]), np.asarray(item["rotation_matrix"]), camera)
        _draw_polyline(draw, op, (150, 150, 150), 1)
        _draw_polyline(draw, rp, colors[index], 2)
    draw.rectangle((3, 3, 310, 22), fill=(0, 0, 0))
    draw.text((7, 6), "gray=original; color=refined deterministic cuboids", fill=(255, 255, 255))
    image.save(path)


def _write_markdown(output_dir: Path, plan: dict[str, Any], report: dict[str, Any]) -> None:
    lines = ["# Refined primitive scene plan v2", "", "Deterministic offline pose refinement using SAM masks, numeric MoGe normals/points/depth, gravity, support planes, and the corrected perspective camera.", "", "## Objects", ""]
    for item in plan["semantic_primitives"]:
        lines += [f"### {item['semantic_label']}", "", f"- Object ID: `{item['object_id']}`", f"- Orientation method: {item['orientation_method']}", f"- Orientation confidence: {item['orientation_confidence']:.4f}", f"- Center: {item['center']}", f"- Dimensions: {item['dimensions']}", f"- Quaternion WXYZ: {item['rotation_quaternion_wxyz']}", f"- Fallback: {item['fallback_reason']}", ""]
    (output_dir / "refined_primitive_scene_plan_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_lines = ["# Pose refinement report", "", f"- Stable IDs preserved: {report['stable_object_ids_preserved']}", f"- Source lineage preserved: {report['source_lineage_preserved']}", f"- Ready for offline evaluation: {report['ready_for_offline_evaluation']}", "", "| Object | Normal confidence | Mask-axis confidence | Original IoU | Refined IoU |", "|---|---:|---:|---:|---:|"]
    for item in report["objects"]:
        report_lines.append(f"| {item['semantic_label']} | {item['normal_cluster_confidence']:.3f} | {item['mask_axis_confidence']:.3f} | {item['projection_comparison']['original']['bbox_iou']:.3f} | {item['projection_comparison']['refined']['bbox_iou']:.3f} |")
    (output_dir / "pose_refinement_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = refine_plan(args.output_dir.resolve())
    print(json.dumps({"output": str(args.output_dir), "objects": len(result["plan"]["semantic_primitives"]), "ready": result["report"]["ready_for_offline_evaluation"]}, indent=2))
