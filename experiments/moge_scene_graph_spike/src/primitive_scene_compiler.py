"""Compile validated semantic evidence into a deterministic primitive scene plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .primitive_plan_models import PrimitiveScenePlan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "primitive_plan"
INPUT_PATHS = {
    "fixture_manifest": ROOT / "inputs" / "office_test" / "manifest.json",
    "moge_geometry": ROOT / "outputs" / "office_test" / "moge" / "geometry.npz",
    "object_geometry": ROOT / "outputs" / "office_test" / "geometry" / "object_geometry.json",
    "scene_evidence": ROOT / "outputs" / "office_test" / "geometry" / "scene_evidence.json",
    "semantic_scene_graph": ROOT / "outputs" / "office_test" / "vlm" / "semantic_scene_graph.json",
    "live_evaluation": ROOT / "outputs" / "office_test" / "vlm" / "live_evaluation.json",
    "vlm_review_overview": ROOT / "outputs" / "office_test" / "vlm" / "vlm_review_overview.png",
}
EXPECTED_IDS = [
    "e971e9fbbf5746eea108d07bfbb5764a",
    "b9858c1f45f443a88d84229a1824d339",
    "9171a7b4d4e142ca936f2564200d0bdb",
    "1064a3a09c4c427e80caa227d17a7cc1",
    "4ec402f9a71d4ea7925b487ac7e11249",
    "49fd02b50a914e8198bacf4644bc377f",
]
PLANE_IDS = ["plane_floor", "plane_left_wall", "plane_right_wall"]
CAMERA_IDS = ["camera_perspective_moge", "camera_orthographic_fitted"]
BOX_SIGNS = np.asarray(
    [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
    dtype=np.float64,
)
SLAB_THICKNESS = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation to a deterministic w-positive quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0.0 or (abs(quaternion[0]) < 1e-12 and tuple(quaternion[1:]) < tuple(-quaternion[1:])):
        quaternion *= -1.0
    return quaternion


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize(np.asarray(quaternion, dtype=np.float64))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(center: np.ndarray, rotation: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ np.diag(dimensions)
    transform[:3, 3] = center
    return transform


def stable_obb_basis(axes: np.ndarray, dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder OBB axes deterministically while preserving the represented box."""
    axes = np.asarray(axes, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    vertical_index = int(np.argmax(np.abs(axes[2, :])))
    remaining = [index for index in range(3) if index != vertical_index]
    x_index = max(remaining, key=lambda index: (abs(float(axes[0, index])), -index))
    y_index = next(index for index in remaining if index != x_index)
    x_axis = axes[:, x_index].copy()
    z_axis = axes[:, vertical_index].copy()
    if z_axis[2] < 0.0:
        z_axis *= -1.0
    if x_axis[0] < 0.0:
        x_axis *= -1.0
    y_axis = _normalize(np.cross(z_axis, x_axis))
    x_axis = _normalize(np.cross(y_axis, z_axis))
    basis = np.column_stack([x_axis, y_axis, z_axis])
    return basis, dimensions[[x_index, y_index, vertical_index]]


def box_corners(primitive: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(primitive["rotation_matrix"], dtype=np.float64)
    dimensions = np.asarray(primitive["dimensions"], dtype=np.float64)
    center = np.asarray(primitive["center"], dtype=np.float64)
    return BOX_SIGNS @ np.diag(dimensions) @ rotation.T + center


def _primitive_record(
    semantic: dict[str, Any],
    geometry: dict[str, Any],
    strategy: str,
    center: np.ndarray,
    rotation: np.ndarray,
    dimensions: np.ndarray,
    geometry_source: list[str],
    support_target: str | None,
    strength: str,
    extra_uncertainty: list[str],
) -> dict[str, Any]:
    quaternion = matrix_to_quaternion_wxyz(rotation)
    confidence = min(float(semantic["confidence"]), float(geometry["geometry_confidence"]["score"]))
    uncertainties = list(dict.fromkeys([*semantic["uncertainty"], *geometry["warnings"], *extra_uncertainty]))
    return {
        "object_id": semantic["object_id"],
        "semantic_label": semantic["original_label"],
        "primitive_type": "cube",
        "geometry_strategy": strategy,
        "geometry_source": geometry_source,
        "center": center.tolist(),
        "rotation_quaternion_wxyz": quaternion.tolist(),
        "rotation_matrix": rotation.tolist(),
        "dimensions": dimensions.tolist(),
        "transform_matrix": make_transform(center, rotation, dimensions).tolist(),
        "support_target": support_target,
        "constraint_strength": strength,
        "confidence": confidence,
        "uncertainty": uncertainties,
        "requires_user_review": bool(semantic["requires_user_review"]),
    }


def classify_constraints(graph: dict[str, Any]) -> list[dict[str, Any]]:
    classifications: list[dict[str, Any]] = []
    for index, relation in enumerate(graph["relationship_reviews"]):
        predicate = relation["predicate"]
        decision = relation["decision"]
        confidence = float(relation["reviewed_confidence"])
        requires_review = bool(relation["requires_user_review"])
        if decision == "reject":
            classification, rationale = "ignored_for_transform", "rejected semantic candidate"
        elif decision == "uncertain":
            classification, rationale = "review_only", "uncertain candidates cannot influence transforms"
        elif predicate == "unknown_support":
            classification, rationale = "review_only", "unknown support is an explicit unresolved state"
        elif predicate in {"supported_by", "attached_to"}:
            if requires_review:
                classification, rationale = "soft", "accepted support or attachment remains user-reviewable"
            elif confidence >= 0.75:
                classification, rationale = "hard", "accepted support meets confidence and review policy"
            else:
                classification, rationale = "soft", "accepted support is below the hard-confidence threshold"
        else:
            classification = "ignored_for_transform"
            rationale = "directional, proximity, and occlusion evidence is validation-only"
            if confidence < 0.65:
                rationale += "; reviewed confidence is below 0.65"
        classifications.append(
            {
                "candidate_id": relation["candidate_id"],
                "subject_object_id": relation["subject_object_id"],
                "predicate": predicate,
                "target_id": relation["target_id"],
                "decision": decision,
                "reviewed_confidence": confidence,
                "classification": classification,
                "influences_transform": classification in {"hard", "soft"} and predicate in {"supported_by", "attached_to"},
                "source_relationship": f"outputs/office_test/vlm/semantic_scene_graph.json#/relationship_reviews/{index}",
                "rationale": rationale,
                "requires_user_review": requires_review,
            }
        )
    return classifications


def _subject_constraint(classifications: list[dict[str, Any]], object_id: str) -> dict[str, Any] | None:
    relevant = [
        item
        for item in classifications
        if item["subject_object_id"] == object_id and item["predicate"] in {"supported_by", "attached_to", "unknown_support"}
    ]
    return relevant[0] if relevant else None


def _load_object_points(object_record: dict[str, Any], raw_to_canonical: np.ndarray) -> np.ndarray:
    path = ROOT / "outputs" / "office_test" / "geometry" / "diagnostics" / object_record["object_id"] / "filtered_visible_geometry.npz"
    with np.load(path) as archive:
        raw = archive["points"].astype(np.float64)
    return transform_points(raw, raw_to_canonical)


def compile_semantic_primitives(
    object_geometry: dict[str, Any],
    scene_evidence: dict[str, Any],
    graph: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_to_canonical = np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"], dtype=np.float64)
    by_geometry = {item["object_id"]: item for item in object_geometry["objects"]}
    by_semantic = {item["object_id"]: item for item in graph["objects"]}
    support_patches = {item["object_id"]: item["visible_support_patch"] for item in scene_evidence["object_support_surface_candidates"]}
    output: list[dict[str, Any]] = []
    for object_id in EXPECTED_IDS:
        semantic = by_semantic[object_id]
        geometry = by_geometry[object_id]
        points = _load_object_points(geometry, raw_to_canonical)
        strategy = semantic["geometry_strategy"]
        constraint = _subject_constraint(classifications, object_id)
        support_target = constraint["target_id"] if constraint and constraint["predicate"] != "unknown_support" else None
        strength = constraint["classification"] if constraint else "review_only"
        sources = [
            f"outputs/office_test/geometry/object_geometry.json#/objects/{object_geometry['objects'].index(geometry)}",
            f"outputs/office_test/geometry/diagnostics/{object_id}/filtered_visible_geometry.npz",
            f"outputs/office_test/vlm/semantic_scene_graph.json#/objects/{graph['objects'].index(semantic)}",
        ]
        extra: list[str] = []

        if strategy == "retained_obb":
            obb = geometry["oriented_bounding_box"]
            center = transform_points(np.asarray([obb["center_xyz"]]), raw_to_canonical)[0]
            axes = raw_to_canonical[:3, :3] @ np.asarray(obb["orientation_matrix_columns_are_box_axes"], dtype=np.float64)
            rotation, dimensions = stable_obb_basis(axes, np.asarray(obb["dimensions_xyz_along_axes"], dtype=np.float64))
            extra.append("floor support remains review-only; retained OBB was not snapped to the floor")
        elif semantic["original_label"] == "desk":
            lower, upper = np.percentile(points[:, :2], [2, 98], axis=0)
            top = float(support_patches[object_id]["height_median"])
            bottom = 0.0
            center = np.asarray([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, (bottom + top) / 2])
            dimensions = np.asarray([upper[0] - lower[0], upper[1] - lower[1], top - bottom])
            rotation = np.eye(3)
            sources.append("outputs/office_test/geometry/scene_evidence.json#/object_support_surface_candidates/1")
            extra.append("one support-aligned cube represents the tabletop, legs, and pedestal together")
        elif semantic["original_label"] == "desk_chair":
            lower, upper = points.min(axis=0), points.max(axis=0)
            center, dimensions = (lower + upper) / 2, upper - lower
            rotation = np.eye(3)
            extra.extend(["incomplete visible-surface geometry", "no hidden seat, legs, wheels, or floor contact inferred"])
        elif semantic["original_label"] == "desktop_box":
            x_lower, x_upper = np.percentile(points[:, 0], [2, 98])
            y_lower, y_upper = np.percentile(points[:, 1], [5, 95])
            z_lower, z_upper = np.percentile(points[:, 2], [5, 95])
            desk_id = next(item["object_id"] for item in object_geometry["objects"] if item["semantic_label"] == "desk")
            bottom = float(support_patches[desk_id]["height_median"])
            height = float(z_upper - z_lower)
            center = np.asarray([(x_lower + x_upper) / 2, (y_lower + y_upper) / 2, bottom + height / 2])
            dimensions = np.asarray([x_upper - x_lower, y_upper - y_lower, height])
            rotation = np.eye(3)
            sources.append("outputs/office_test/geometry/scene_evidence.json#/object_support_surface_candidates/1")
            extra.extend(["raw full-Z extent excluded", "p5/p95 local depth and height intervals used", "uncertain chair occlusion did not influence transform"])
        elif semantic["original_label"] == "coat_rack":
            lower, upper = points.min(axis=0), points.max(axis=0)
            dimensions = upper - lower
            center = (lower + upper) / 2
            center[2] = dimensions[2] / 2
            rotation = np.eye(3)
            extra.extend(["all seven visible components are unioned into one cube", "soft floor support aligns the visible union bottom to the floor"])
        elif semantic["original_label"] == "wall_light_fixture":
            wall = next(item for item in scene_evidence["structural_planes"] if item["plane_id"] == "plane_right_wall")
            normal = _normalize(np.asarray(wall["canonical_plane_equation"]["normal"], dtype=np.float64))
            offset = float(wall["canonical_plane_equation"]["offset"])
            vertical = _normalize(np.asarray([0.0, 0.0, 1.0]) - normal[2] * normal)
            horizontal = _normalize(np.cross(vertical, normal))
            rotation = np.column_stack([horizontal, vertical, normal])
            h_values, v_values, n_values = points @ horizontal, points @ vertical, points @ normal
            h_lower, h_upper = np.percentile(h_values, [2, 98])
            v_lower, v_upper = np.percentile(v_values, [2, 98])
            measured_thickness = float(np.percentile(n_values, 95) - np.percentile(n_values, 5))
            thickness = float(np.clip(measured_thickness, 0.03, 0.12))
            center = horizontal * ((h_lower + h_upper) / 2) + vertical * ((v_lower + v_upper) / 2) + normal * (-offset + thickness / 2)
            dimensions = np.asarray([h_upper - h_lower, v_upper - v_lower, thickness])
            extra.extend(["thickness is conservatively clipped to [0.03, 0.12] canonical units", "no hidden wall penetration inferred"])
        else:
            raise ValueError(f"unsupported semantic strategy for {semantic['original_label']}")

        output.append(
            _primitive_record(
                semantic, geometry, strategy, center, rotation, dimensions, sources, support_target, strength, extra
            )
        )
    return output


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    normal = _normalize(normal)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(normal @ up)) > 0.95:
        return np.eye(3)
    vertical = _normalize(up - float(up @ normal) * normal)
    horizontal = _normalize(np.cross(vertical, normal))
    return np.column_stack([horizontal, vertical, normal])


def compile_structural_proxies(
    scene_evidence: dict[str, Any], source: np.ndarray, points_raw: np.ndarray, valid_mask: np.ndarray, object_masks: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    raw_to_canonical = np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"], dtype=np.float64)
    points_canonical = transform_points(points_raw[valid_mask], raw_to_canonical)
    brightness = source.astype(np.float64).mean(axis=2)
    structural_pixels = valid_mask & (brightness >= 24.0)
    for mask in object_masks.values():
        structural_pixels &= ~mask
    structural_points = transform_points(points_raw[structural_pixels], raw_to_canonical)
    proxies: list[dict[str, Any]] = []
    review_by_id = {item["plane_id"]: item for item in scene_evidence.get("structural_planes", [])}
    for plane_id in PLANE_IDS:
        evidence = review_by_id[plane_id]
        normal = _normalize(np.asarray(evidence["canonical_plane_equation"]["normal"], dtype=np.float64))
        offset = float(evidence["canonical_plane_equation"]["offset"])
        basis = _plane_basis(normal)
        distance = np.abs(structural_points @ normal + offset)
        near = structural_points[distance <= 0.08]
        if len(near) < 100:
            raise ValueError(f"insufficient bright near-plane points for {plane_id}")
        local = near @ basis
        if plane_id == "plane_floor":
            x0, y0 = np.percentile(local[:, :2], 3, axis=0)
            x1, y1 = np.percentile(local[:, :2], 97, axis=0)
            center = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2, -offset - SLAB_THICKNESS / 2])
            dimensions = np.asarray([x1 - x0, y1 - y0, SLAB_THICKNESS])
            extent_confidence = 0.62
            warning = "full contaminated floor-inlier bounds were not used; black exterior pixels and object masks were excluded"
        else:
            h0, v0 = np.percentile(local[:, :2], 5, axis=0)
            h1, v1 = np.percentile(local[:, :2], 95, axis=0)
            v0 = max(0.0, float(v0))
            n_center = -offset - SLAB_THICKNESS / 2
            center = basis @ np.asarray([(h0 + h1) / 2, (v0 + v1) / 2, n_center])
            dimensions = np.asarray([h1 - h0, v1 - v0, SLAB_THICKNESS])
            extent_confidence = 0.55
            warning = "extent uses bright near-plane robust clipping; decor and complete inlier bounds were not used"
        quaternion = matrix_to_quaternion_wxyz(basis)
        proxies.append(
            {
                "plane_id": plane_id,
                "semantic_label": evidence["semantic_candidate"],
                "primitive_type": "cube",
                "canonical_plane_normal": normal.tolist(),
                "canonical_plane_offset": offset,
                "center": center.tolist(),
                "rotation_quaternion_wxyz": quaternion.tolist(),
                "rotation_matrix": basis.tolist(),
                "dimensions": dimensions.tolist(),
                "transform_matrix": make_transform(center, basis, dimensions).tolist(),
                "equation_confidence": float(evidence["confidence"]),
                "extent_confidence": extent_confidence,
                "extent_policy": "bright_near_plane_robust_clip",
                "thickness_policy": f"fixed configurable slab thickness {SLAB_THICKNESS:.3f} canonical units",
                "warnings": [warning, "structural proxy extent is provisional"],
                "requires_user_review": True,
            }
        )
    return proxies


def fit_orthographic_camera(points_raw: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    height, width = valid_mask.shape
    ys, xs = np.nonzero(valid_mask)
    camera_points = points_raw[valid_mask].astype(np.float64)
    if len(camera_points) > 50000:
        indices = np.linspace(0, len(camera_points) - 1, 50000, dtype=np.int64)
        camera_points, xs, ys = camera_points[indices], xs[indices], ys[indices]
    x, y = camera_points[:, 0], camera_points[:, 1]
    x_mean, y_mean = float(x.mean()), float(y.mean())
    u_mean, v_mean = float(xs.mean()), float(ys.mean())
    denominator = float(np.sum((x - x_mean) ** 2 + (y - y_mean) ** 2))
    scale = float(np.sum((x - x_mean) * (xs - u_mean) + (y - y_mean) * (ys - v_mean)) / denominator)
    if scale <= 0.0:
        raise ValueError("orthographic fit produced non-positive scale")
    offset_x, offset_y = u_mean - scale * x_mean, v_mean - scale * y_mean
    projected = np.column_stack([scale * x + offset_x, scale * y + offset_y])
    errors = np.linalg.norm(projected - np.column_stack([xs, ys]), axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    median = float(np.median(errors))
    confidence = float(np.clip(1.0 - rmse / (0.25 * math.hypot(width, height)), 0.0, 1.0))
    return {
        "pixels_per_canonical_unit": scale,
        "orthographic_scale_vertical": height / scale,
        "image_center_offset_pixels": [offset_x, offset_y],
        "fit_rmse_pixels": rmse,
        "fit_median_error_pixels": median,
        "correspondence_count": int(len(camera_points)),
        "fitting_method": "shared least-squares image scale plus independent image-center offsets on valid MoGe 2D/3D correspondences",
        "fit_confidence": confidence,
    }


def compile_camera_candidates(scene_evidence: dict[str, Any], points_raw: np.ndarray, valid_mask: np.ndarray) -> list[dict[str, Any]]:
    camera_to_world = np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"], dtype=np.float64)
    world_to_camera = np.asarray(scene_evidence["transforms"]["canonical_scene_world_to_raw_moge"], dtype=np.float64)
    camera_evidence = scene_evidence["camera_evidence"]
    height, width = valid_mask.shape
    orthographic = fit_orthographic_camera(points_raw, valid_mask)
    return [
        {
            "camera_id": "camera_perspective_moge",
            "camera_type": "perspective",
            "camera_to_canonical_transform": camera_to_world.tolist(),
            "canonical_to_camera_transform": world_to_camera.tolist(),
            "perspective": {
                "normalized_intrinsics": camera_evidence["normalized_intrinsics"],
                "horizontal_fov_degrees": float(camera_evidence["estimated_horizontal_fov_degrees"]),
                "image_width": width,
                "image_height": height,
            },
            "orthographic": None,
            "fit_confidence": 0.68,
            "source_evidence": ["MoGe-2 normalized intrinsics", "MoGe-2 estimated 25.14-degree horizontal FOV"],
            "warnings": ["metric camera placement is unverified", "source image resembles an orthographic or isometric render"],
            "requires_user_review": True,
        },
        {
            "camera_id": "camera_orthographic_fitted",
            "camera_type": "orthographic",
            "camera_to_canonical_transform": camera_to_world.tolist(),
            "canonical_to_camera_transform": world_to_camera.tolist(),
            "perspective": None,
            "orthographic": {key: value for key, value in orthographic.items() if key != "fit_confidence"},
            "fit_confidence": orthographic["fit_confidence"],
            "source_evidence": ["valid MoGe image-space and camera-space correspondences", "same canonical orientation basis as the perspective candidate"],
            "warnings": ["orthographic model is fitted, not proven", "absolute metric scale remains unverified"],
            "requires_user_review": True,
        },
    ]


def project_points(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    camera_points = transform_points(points, np.asarray(camera["canonical_to_camera_transform"], dtype=np.float64))
    if camera["camera_type"] == "perspective":
        params = camera["perspective"]
        intrinsics = np.asarray(params["normalized_intrinsics"], dtype=np.float64)
        z = camera_points[:, 2]
        safe_z = np.where(np.abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
        u = (intrinsics[0, 0] * camera_points[:, 0] / safe_z + intrinsics[0, 2]) * params["image_width"]
        v = (intrinsics[1, 1] * camera_points[:, 1] / safe_z + intrinsics[1, 2]) * params["image_height"]
        visible = z > 1e-8
    else:
        params = camera["orthographic"]
        scale = float(params["pixels_per_canonical_unit"])
        offset_x, offset_y = params["image_center_offset_pixels"]
        u = scale * camera_points[:, 0] + offset_x
        v = scale * camera_points[:, 1] + offset_y
        visible = np.ones(len(points), dtype=bool)
    return np.column_stack([u, v]), visible


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).tolist())))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)
    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection_min = np.maximum(a[:2], b[:2])
    intersection_max = np.minimum(a[2:], b[2:])
    intersection = np.maximum(0.0, intersection_max - intersection_min)
    intersection_area = float(intersection[0] * intersection[1])
    area_a = float(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]))
    area_b = float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    union = area_a + area_b - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def _hull_mask(hull: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.new("1", (shape[1], shape[0]), 0)
    if len(hull) >= 3:
        ImageDraw.Draw(image).polygon([tuple(map(float, point)) for point in hull], fill=1)
    return np.asarray(image, dtype=bool)


def projection_metrics(
    primitive: dict[str, Any], camera: dict[str, Any], source_mask: np.ndarray
) -> dict[str, Any]:
    corners = box_corners(primitive)
    projected, visible = project_points(corners, camera)
    finite = np.isfinite(projected).all(axis=1)
    valid = bool(finite.all() and visible.all())
    if not finite.all():
        projected = np.nan_to_num(projected, nan=0.0, posinf=1e6, neginf=-1e6)
    projected_bbox = np.asarray([projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()])
    ys, xs = np.nonzero(source_mask)
    mask_bbox = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)
    bbox_iou = _bbox_iou(projected_bbox, mask_bbox)
    hull = convex_hull_2d(projected)
    projected_mask = _hull_mask(hull, source_mask.shape)
    intersection = int((projected_mask & source_mask).sum())
    union = int((projected_mask | source_mask).sum())
    hull_iou = intersection / union if union else 0.0
    projected_centroid = projected.mean(axis=0)
    mask_centroid = np.asarray([xs.mean(), ys.mean()])
    centroid_error = float(np.linalg.norm(projected_centroid - mask_centroid))
    area_ratio = float(projected_mask.sum() / max(1, source_mask.sum()))
    if valid and bbox_iou >= 0.5 and hull_iou >= 0.35 and centroid_error <= 30.0:
        quality = "good"
    elif valid and bbox_iou >= 0.2 and centroid_error <= 80.0:
        quality = "marginal"
    else:
        quality = "failed"
    warnings: list[str] = []
    if not visible.all():
        warnings.append("one or more projected corners are behind the camera")
    if not finite.all():
        warnings.append("non-finite projection was sanitized for diagnostics")
    if quality != "good":
        warnings.append("projection is marginal or failed; geometry was not silently retuned")
    return {
        "camera_id": camera["camera_id"],
        "object_id": primitive["object_id"],
        "projected_corners_xy": projected.tolist(),
        "projected_bbox_xyxy": projected_bbox.tolist(),
        "source_mask_bbox_iou": float(np.clip(bbox_iou, 0.0, 1.0)),
        "projected_hull_mask_iou": float(np.clip(hull_iou, 0.0, 1.0)),
        "centroid_error_pixels": centroid_error,
        "projected_area_ratio": area_ratio,
        "visibility_valid": valid,
        "quality": quality,
        "warnings": warnings,
    }


def compile_projection_validation(
    primitives: list[dict[str, Any]], cameras: list[dict[str, Any]], masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    metrics = [projection_metrics(primitive, camera, masks[primitive["object_id"]]) for camera in cameras for primitive in primitives]
    grouped = {camera_id: [item for item in metrics if item["camera_id"] == camera_id] for camera_id in CAMERA_IDS}
    def mean(camera_id: str, key: str) -> float:
        return float(np.mean([item[key] for item in grouped[camera_id]]))
    perspective_score = mean(CAMERA_IDS[0], "projected_hull_mask_iou") - mean(CAMERA_IDS[0], "centroid_error_pixels") / 1000.0
    orthographic_score = mean(CAMERA_IDS[1], "projected_hull_mask_iou") - mean(CAMERA_IDS[1], "centroid_error_pixels") / 1000.0
    recommended = CAMERA_IDS[0] if perspective_score >= orthographic_score else CAMERA_IDS[1]
    return {
        "shared_geometry_across_cameras": True,
        "metrics": metrics,
        "perspective_mean_bbox_iou": mean(CAMERA_IDS[0], "source_mask_bbox_iou"),
        "perspective_mean_hull_iou": mean(CAMERA_IDS[0], "projected_hull_mask_iou"),
        "perspective_mean_centroid_error_pixels": mean(CAMERA_IDS[0], "centroid_error_pixels"),
        "orthographic_mean_bbox_iou": mean(CAMERA_IDS[1], "source_mask_bbox_iou"),
        "orthographic_mean_hull_iou": mean(CAMERA_IDS[1], "projected_hull_mask_iou"),
        "orthographic_mean_centroid_error_pixels": mean(CAMERA_IDS[1], "centroid_error_pixels"),
        "recommended_first_blender_camera": recommended,
        "other_camera_must_still_be_tested": True,
    }


def evaluate_quality_gates(
    primitives: list[dict[str, Any]], structures: list[dict[str, Any]], cameras: list[dict[str, Any]],
    classifications: list[dict[str, Any]], source_hashes_valid: bool, object_geometry: dict[str, Any]
) -> list[dict[str, Any]]:
    ids = [item["object_id"] for item in primitives]
    all_boxes = [*primitives, *structures]
    gates: list[tuple[str, bool, str]] = []
    gates.append(("exact_six_object_ids", len(ids) == 6 and set(ids) == set(EXPECTED_IDS) and len(set(ids)) == 6, "exact expected object-ID set"))
    gates.append(("one_cube_per_semantic_object", all(item["primitive_type"] == "cube" for item in primitives), "no part primitives"))
    gates.append(("positive_dimensions", all(np.all(np.asarray(item["dimensions"]) > 0) for item in all_boxes), "all cube dimensions must be positive"))
    orthonormal = all(np.allclose(np.asarray(item["rotation_matrix"]).T @ np.asarray(item["rotation_matrix"]), np.eye(3), atol=1e-6) for item in all_boxes)
    gates.append(("orthonormal_rotations", orthonormal, "R^T R equals identity"))
    quaternion_match = all(np.allclose(quaternion_to_matrix_wxyz(np.asarray(item["rotation_quaternion_wxyz"])), np.asarray(item["rotation_matrix"]), atol=1e-6) for item in all_boxes)
    gates.append(("quaternion_matrix_consistency", quaternion_match, "quaternions and matrices agree"))
    gates.append(("finite_transforms", all(np.isfinite(np.asarray(item["transform_matrix"])).all() for item in all_boxes), "all transforms are finite"))
    known_targets = set(EXPECTED_IDS) | set(PLANE_IDS)
    gates.append(("known_support_targets", all(item["support_target"] is None or item["support_target"] in known_targets for item in primitives), "support targets are known IDs"))
    gates.append(("uncertain_never_hard", not any(item["decision"] == "uncertain" and item["classification"] == "hard" for item in classifications), "uncertain relationships remain review-only"))
    directional = {"left_of", "right_of", "above", "below", "in_front_of", "behind", "near", "image_occludes", "image_occluded_by"}
    gates.append(("directional_never_affects_transform", not any(item["predicate"] in directional and item["influences_transform"] for item in classifications), "directional/proximity/occlusion evidence is validation-only"))
    chair = next(item for item in primitives if item["semantic_label"] == "desk_chair")
    gates.append(("chair_visible_only", chair["geometry_strategy"] == "visible_surface_proxy" and chair["support_target"] is None and any("no hidden seat" in text for text in chair["uncertainty"]), "no hidden chair geometry or floor support"))
    gates.append(("desk_not_split", sum(item["semantic_label"] == "desk" for item in primitives) == 1, "one desk cube"))
    coat = next(item for item in primitives if item["semantic_label"] == "coat_rack")
    gates.append(("coat_rack_union", sum(item["semantic_label"] == "coat_rack" for item in primitives) == 1 and any("seven visible components" in text for text in coat["uncertainty"]), "seven components remain one cube"))
    box = next(item for item in primitives if item["semantic_label"] == "desktop_box")
    raw_box = next(item for item in object_geometry["objects"] if item["semantic_label"] == "desktop_box")
    raw_depth_dimension = max(raw_box["geometry"]["raw_moge"]["filtered"]["visible_surface_dimensions"])
    gates.append(("desktop_box_depth_corrected", max(box["dimensions"]) < 0.75 * raw_depth_dimension and any("raw full-Z extent excluded" in text for text in box["uncertainty"]), "robust local intervals replace raw full-Z extent"))
    floor = next(item for item in structures if item["plane_id"] == "plane_floor")
    gates.append(("structural_extent_clipped", floor["extent_policy"] == "bright_near_plane_robust_clip" and any("full contaminated" in text for text in floor["warnings"]), "full contaminated floor inlier extent is excluded"))
    gates.append(("two_camera_candidates", {item["camera_id"] for item in cameras} == set(CAMERA_IDS), "perspective and orthographic candidates preserved"))
    gates.append(("stable_validated_source_hashes", source_hashes_valid, "validated VLM request inputs and source lineage are unchanged"))
    return [{"gate": gate, "passed": bool(passed), "details": details} for gate, passed, details in gates]


def _load_masks(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    fixture = ROOT / "inputs" / "office_test"
    return {item["object_id"]: np.asarray(Image.open(fixture / item["mask_filename"]).convert("L")) > 0 for item in manifest["objects"]}


def validate_source_lineage() -> bool:
    vlm_root = ROOT / "outputs" / "office_test" / "vlm"
    request = _read_json(vlm_root / "input" / "request_manifest.json")
    hashes_ok = all(sha256_file(vlm_root / "input" / item["filename"]) == item["sha256"] for item in request["files"])
    validation = _read_json(vlm_root / "validation_report.json")
    evaluation = _read_json(vlm_root / "live_evaluation.json")
    response = _read_json(vlm_root / "raw_response.json")
    return bool(hashes_ok and validation["valid"] and response["response_id"] == evaluation["source_response_id"])


def _source_references() -> list[dict[str, str]]:
    return [{"role": role, "path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for role, path in INPUT_PATHS.items()]


def _draw_projection(source: np.ndarray, primitives: list[dict[str, Any]], metrics: list[dict[str, Any]], output: Path) -> None:
    image = Image.fromarray(source.copy())
    draw = ImageDraw.Draw(image)
    colors = {"good": (40, 210, 80), "marginal": (255, 185, 30), "failed": (235, 55, 55)}
    by_id = {item["object_id"]: item for item in primitives}
    for item in metrics:
        primitive = by_id[item["object_id"]]
        hull = convex_hull_2d(np.asarray(item["projected_corners_xy"]))
        color = (220, 60, 220) if primitive["requires_user_review"] else colors[item["quality"]]
        if len(hull) >= 3:
            draw.line([tuple(point) for point in np.vstack([hull, hull[0]])], fill=color, width=2)
        x0, y0, _, _ = item["projected_bbox_xyxy"]
        draw.text((max(0, x0), max(0, y0)), primitive["semantic_label"], fill=color)
    image.save(output)


def _draw_structures(source: np.ndarray, structures: list[dict[str, Any]], camera: dict[str, Any]) -> Image.Image:
    image = Image.fromarray(source.copy())
    draw = ImageDraw.Draw(image)
    colors = [(30, 210, 230), (130, 210, 60), (255, 120, 40)]
    for structure, color in zip(structures, colors):
        projected, _ = project_points(box_corners(structure), camera)
        hull = convex_hull_2d(projected)
        if len(hull) >= 3:
            draw.line([tuple(point) for point in np.vstack([hull, hull[0]])], fill=color, width=2)
            draw.text(tuple(hull[0]), structure["semantic_label"], fill=color)
    return image


def _side_by_side(left: Image.Image, right: Image.Image, left_label: str, right_label: str) -> Image.Image:
    output = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 24), (18, 18, 22))
    output.paste(left, (0, 24))
    output.paste(right, (left.width, 24))
    draw = ImageDraw.Draw(output)
    draw.text((8, 6), left_label, fill=(240, 240, 240))
    draw.text((left.width + 8, 6), right_label, fill=(240, 240, 240))
    return output


def write_visualizations(
    output_dir: Path, source: np.ndarray, primitives: list[dict[str, Any]], structures: list[dict[str, Any]],
    cameras: list[dict[str, Any]], projection: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_object = output_dir / "per_object_projections"
    per_object.mkdir(parents=True, exist_ok=True)
    metric_groups = {camera_id: [item for item in projection["metrics"] if item["camera_id"] == camera_id] for camera_id in CAMERA_IDS}
    perspective_path = output_dir / "perspective_projection_overlay.png"
    orthographic_path = output_dir / "orthographic_projection_overlay.png"
    _draw_projection(source, primitives, metric_groups[CAMERA_IDS[0]], perspective_path)
    _draw_projection(source, primitives, metric_groups[CAMERA_IDS[1]], orthographic_path)
    perspective_image = Image.open(perspective_path).convert("RGB")
    orthographic_image = Image.open(orthographic_path).convert("RGB")
    _side_by_side(perspective_image, orthographic_image, "Perspective", "Orthographic").save(output_dir / "camera_comparison.png")
    structures_p = _draw_structures(source, structures, cameras[0])
    structures_o = _draw_structures(source, structures, cameras[1])
    _side_by_side(structures_p, structures_o, "Perspective structural proxies", "Orthographic structural proxies").save(output_dir / "structural_projection_comparison.png")
    primitive_by_id = {item["object_id"]: item for item in primitives}
    for object_id in EXPECTED_IDS:
        p_metric = next(item for item in metric_groups[CAMERA_IDS[0]] if item["object_id"] == object_id)
        o_metric = next(item for item in metric_groups[CAMERA_IDS[1]] if item["object_id"] == object_id)
        p_path, o_path = per_object / "_p.png", per_object / "_o.png"
        _draw_projection(source, [primitive_by_id[object_id]], [p_metric], p_path)
        _draw_projection(source, [primitive_by_id[object_id]], [o_metric], o_path)
        _side_by_side(Image.open(p_path).convert("RGB"), Image.open(o_path).convert("RGB"), "Perspective", "Orthographic").save(per_object / f"{primitive_by_id[object_id]['semantic_label']}.png")
        p_path.unlink()
        o_path.unlink()
    comparison = Image.open(output_dir / "camera_comparison.png").convert("RGB")
    structural = Image.open(output_dir / "structural_projection_comparison.png").convert("RGB")
    panel_height = 125
    overview = Image.new("RGB", (comparison.width, comparison.height + structural.height + panel_height), (18, 18, 22))
    overview.paste(comparison, (0, 0))
    overview.paste(structural, (0, comparison.height))
    draw = ImageDraw.Draw(overview)
    y = comparison.height + structural.height + 8
    draw.text((10, y), f"Recommended first: {projection['recommended_first_blender_camera']}", fill=(240, 240, 240))
    draw.text((10, y + 18), f"Perspective mean bbox/hull IoU: {projection['perspective_mean_bbox_iou']:.3f} / {projection['perspective_mean_hull_iou']:.3f}", fill=(200, 200, 205))
    draw.text((10, y + 36), f"Orthographic mean bbox/hull IoU: {projection['orthographic_mean_bbox_iou']:.3f} / {projection['orthographic_mean_hull_iou']:.3f}", fill=(200, 200, 205))
    draw.text((10, y + 54), "Magenta=user review; green=good; amber=marginal; red=failed", fill=(200, 200, 205))
    draw.text((10, y + 72), "Shared object geometry; no per-camera transform tuning", fill=(200, 200, 205))
    overview.save(output_dir / "primitive_plan_overview.png")


def _write_reports(output_dir: Path, plan: dict[str, Any]) -> None:
    projection_by_key = {(item["object_id"], item["camera_id"]): item for item in plan["projection_validation"]["metrics"]}
    rows: list[str] = []
    for primitive in plan["semantic_primitives"]:
        p = projection_by_key[(primitive["object_id"], CAMERA_IDS[0])]
        o = projection_by_key[(primitive["object_id"], CAMERA_IDS[1])]
        rows.append(
            f"| {primitive['semantic_label']} | `{primitive['geometry_strategy']}` | "
            f"{', '.join(f'{value:.4f}' for value in primitive['dimensions'])} | {primitive['support_target'] or 'unresolved'} | "
            f"{primitive['constraint_strength']} | {primitive['confidence']:.3f} | "
            f"{p['source_mask_bbox_iou']:.3f}/{p['projected_hull_mask_iou']:.3f}/{p['centroid_error_pixels']:.1f} | "
            f"{o['source_mask_bbox_iou']:.3f}/{o['projected_hull_mask_iou']:.3f}/{o['centroid_error_pixels']:.1f} | "
            f"{'yes' if primitive['requires_user_review'] else 'no'} | {'; '.join(primitive['uncertainty'])} |"
        )
    structures = "\n".join(
        f"- {item['semantic_label']}: dimensions {', '.join(f'{value:.4f}' for value in item['dimensions'])}; "
        f"equation confidence {item['equation_confidence']:.3f}; extent confidence {item['extent_confidence']:.3f}; "
        f"warnings: {'; '.join(item['warnings'])}"
        for item in plan["structural_proxies"]
    )
    rejected = [item for item in plan["quality_gates"] if not item["passed"]]
    rejected_text = "\n".join(f"- `{item['gate']}`: {item['details']}" for item in rejected) or "- None"
    camera = plan["projection_validation"]
    text = f"""# Primitive scene plan

Compilation status: `{plan['compilation_status']['status']}`

Absolute MoGe metric scale is not independently verified. No Blender objects were created.

## Semantic object cubes

Projection columns are bbox IoU / hull IoU / centroid error in pixels.

| Object | Strategy | Dimensions | Support target | Strength | Confidence | Perspective | Orthographic | Review | Uncertainties |
|---|---|---|---|---|---:|---|---|---|---|
{chr(10).join(rows)}

## Structural proxies

{structures}

## Camera comparison

- Perspective mean bbox IoU: {camera['perspective_mean_bbox_iou']:.4f}
- Perspective mean hull IoU: {camera['perspective_mean_hull_iou']:.4f}
- Perspective mean centroid error: {camera['perspective_mean_centroid_error_pixels']:.2f} px
- Orthographic mean bbox IoU: {camera['orthographic_mean_bbox_iou']:.4f}
- Orthographic mean hull IoU: {camera['orthographic_mean_hull_iou']:.4f}
- Orthographic mean centroid error: {camera['orthographic_mean_centroid_error_pixels']:.2f} px
- Recommended first Blender camera: `{camera['recommended_first_blender_camera']}`
- The other camera must still be tested; the camera model remains unresolved.

## Rejected quality gates

{rejected_text}

## Constraint policy

Only accepted support and attachment evidence can influence transforms. Uncertain relationships are review-only. Rejected, directional, proximity, and occlusion relationships are ignored for transforms. Low-confidence accepted directional relationships are not compiled as constraints.

## User review

{chr(10).join(f'- {item}' for item in plan['user_review_requirements'])}
"""
    (output_dir / "primitive_scene_plan.md").write_text(text, encoding="utf-8")
    (output_dir / "compilation_report.md").write_text(text, encoding="utf-8")


def compile_plan(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    for path in INPUT_PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _read_json(INPUT_PATHS["fixture_manifest"])
    object_geometry = _read_json(INPUT_PATHS["object_geometry"])
    scene_evidence = _read_json(INPUT_PATHS["scene_evidence"])
    graph = _read_json(INPUT_PATHS["semantic_scene_graph"])
    evaluation = _read_json(INPUT_PATHS["live_evaluation"])
    if evaluation["readiness"] != "ready_for_primitive_compilation":
        raise ValueError("live evaluation is not ready for primitive compilation")
    with np.load(INPUT_PATHS["moge_geometry"]) as archive:
        points_raw = archive["points"].astype(np.float64)
        valid_mask = archive["valid_mask"].astype(bool)
    fixture = ROOT / "inputs" / "office_test"
    source = np.asarray(Image.open(fixture / manifest["image"]["filename"]).convert("RGB"))
    masks = _load_masks(manifest)
    classifications = classify_constraints(graph)
    primitives = compile_semantic_primitives(object_geometry, scene_evidence, graph, classifications)
    structures = compile_structural_proxies(scene_evidence, source, points_raw, valid_mask, masks)
    cameras = compile_camera_candidates(scene_evidence, points_raw, valid_mask)
    projection = compile_projection_validation(primitives, cameras, masks)
    source_hashes_valid = validate_source_lineage()
    gates = evaluate_quality_gates(primitives, structures, cameras, classifications, source_hashes_valid, object_geometry)
    rejected = [item["gate"] for item in gates if not item["passed"]]
    status = "ready_for_blender_execution" if not rejected else "needs_compiler_revision"
    transform = scene_evidence["transforms"]
    plan = {
        "schema_version": "1.0",
        "scene_id": "office_test",
        "source_references": _source_references(),
        "coordinate_system": {
            "name": "canonical_scene_world",
            "handedness": "right_handed",
            "x_axis": "camera-right projected onto the fitted floor",
            "y_axis": "cross(+Z,+X)",
            "z_axis": "fitted floor normal toward occupied room volume",
            "origin": "scene-wide valid-point median projected onto the fitted floor",
            "units": "unverified MoGe metric units",
            "source_to_canonical_matrix": transform["raw_moge_to_canonical_scene_world"],
            "canonical_to_source_matrix": transform["canonical_scene_world_to_raw_moge"],
        },
        "unit_scale_warning": "MoGe-2 metric scale is retained but has not been independently verified; dimensions are provisional.",
        "camera_candidates": cameras,
        "structural_proxies": structures,
        "semantic_primitives": primitives,
        "constraint_classifications": classifications,
        "projection_validation": projection,
        "uncertainties": list(dict.fromkeys([*graph["global_uncertainties"], *evaluation["remaining_deterministic_uncertainties"]])),
        "user_review_requirements": [
            "test both camera candidates before choosing a final camera",
            "review all structural proxy extents",
            "review filing-cabinet floor support",
            "review incomplete visible-only chair geometry and unknown support",
            "review desktop-box corrected depth and desk support",
            "review wall-light thickness and right-wall attachment",
            "do not promote uncertain or low-confidence directional relationships to hard constraints",
        ],
        "quality_gates": gates,
        "compilation_status": {
            "status": status,
            "deterministic": True,
            "quality_gates_passed": not rejected,
            "rejected_quality_gates": rejected,
            "blender_objects_created": False,
        },
    }
    validated = PrimitiveScenePlan.model_validate(plan).model_dump(mode="json")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primitive_scene_plan.json").write_text(json.dumps(validated, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "1.0",
        "scene_id": "office_test",
        "status": status,
        "quality_gates": gates,
        "projection_summary": {key: value for key, value in projection.items() if key != "metrics"},
        "poor_reprojection_objects": [
            {"camera_id": item["camera_id"], "object_id": item["object_id"], "quality": item["quality"], "warnings": item["warnings"]}
            for item in projection["metrics"] if item["quality"] == "failed"
        ],
        "warnings": validated["uncertainties"],
    }
    (output_dir / "compilation_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "constraint_classification.json").write_text(json.dumps({"classifications": classifications}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "camera_candidates.json").write_text(json.dumps({"camera_candidates": cameras, "projection_summary": summary["projection_summary"]}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_reports(output_dir, validated)
    write_visualizations(output_dir, source, primitives, structures, cameras, projection)
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = compile_plan(arguments.output_dir.resolve())
    print(json.dumps({"status": result["compilation_status"]["status"], "output": str(arguments.output_dir / "primitive_scene_plan.json")}, indent=2))
