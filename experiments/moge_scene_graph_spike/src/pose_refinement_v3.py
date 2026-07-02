"""Normal-first, audit-only cuboid orientation estimation for the office spike.

Normals are the primary orientation signal.  This module intentionally contains
no Blender integration and never performs per-yaw bbox center/scale fitting.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .pose_refinement import matrix_to_quaternion_wxyz


EPS = 1e-9
SIDE_DOT_MAX = 0.35
UP_DOT_MIN = 0.80
CLUSTER_THRESHOLD_DEGREES = 12.0
ORTHOGONALITY_THRESHOLD_DEGREES = 12.0


def angle_distance_180(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def yaw_from_quaternion(quaternion_wxyz: list[float]) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_rotation(yaw_degrees: float) -> np.ndarray:
    angle = math.radians(yaw_degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def boundary_distance_weights(mask: np.ndarray, maximum: int = 8) -> np.ndarray:
    """Small deterministic interior-distance transform without scipy."""
    current = np.asarray(mask, bool).copy()
    distance = np.zeros(current.shape, dtype=float)
    for level in range(1, maximum + 1):
        distance[current] = level
        padded = np.pad(current, 1)
        current = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        if not current.any():
            break
    return 0.35 + 0.65 * distance / max(1.0, float(distance.max()))


def _axial_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    radians = np.radians(2.0 * angles)
    vector = np.asarray([(weights * np.cos(radians)).sum(), (weights * np.sin(radians)).sum()])
    return float((math.degrees(math.atan2(vector[1], vector[0])) / 2.0) % 180.0)


def cluster_azimuths_mod180(
    azimuth_degrees: np.ndarray,
    weights: np.ndarray | None = None,
    threshold_degrees: float = CLUSTER_THRESHOLD_DEGREES,
) -> list[dict[str, Any]]:
    """Cluster plane-normal azimuths with n and -n treated identically."""
    angles = np.mod(np.asarray(azimuth_degrees, dtype=float).reshape(-1), 180.0)
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    angles, weights = angles[keep], weights[keep]
    if not len(angles):
        return []
    centers: list[float] = []
    assignments = np.zeros(len(angles), dtype=int)
    for angle in angles[np.argsort(angles)]:
        distances = [angle_distance_180(angle, center) for center in centers]
        if not distances or min(distances) > threshold_degrees:
            centers.append(float(angle))
    for _ in range(12):
        assignments = np.asarray([int(np.argmin([angle_distance_180(a, c) for c in centers])) for a in angles])
        updated = [_axial_mean(angles[assignments == index], weights[assignments == index]) for index in range(len(centers)) if np.any(assignments == index)]
        if len(updated) == len(centers) and max(angle_distance_180(a, b) for a, b in zip(updated, centers)) < 1e-7:
            centers = updated
            break
        centers = updated
    total = float(weights.sum())
    records = []
    for index, center in enumerate(centers):
        selection = assignments == index
        errors = np.asarray([angle_distance_180(a, center) for a in angles[selection]])
        cluster_weight = float(weights[selection].sum())
        records.append({
            "cluster_id": index,
            "azimuth_degrees_mod180": float(center),
            "sample_count": int(selection.sum()),
            "weight": cluster_weight,
            "coverage": cluster_weight / max(EPS, total),
            "dispersion_median_degrees": float(np.median(errors)),
            "dispersion_p90_degrees": float(np.percentile(errors, 90)),
        })
    records.sort(key=lambda item: (-item["weight"], item["azimuth_degrees_mod180"]))
    return records


def estimate_horizontal_frame(
    normals_world: np.ndarray,
    pixel_yx: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    gravity_up: np.ndarray = np.asarray([0.0, 0.0, 1.0]),
) -> dict[str, Any]:
    normals = np.asarray(normals_world, dtype=float)
    lengths = np.linalg.norm(normals, axis=1)
    keep = np.isfinite(normals).all(axis=1) & (lengths > EPS)
    normals = normals[keep] / lengths[keep, None]
    up = np.asarray(gravity_up, dtype=float); up /= np.linalg.norm(up)
    dot = normals @ up
    side = np.abs(dot) <= SIDE_DOT_MAX
    upward = np.abs(dot) >= UP_DOT_MIN
    weights = np.ones(int(side.sum()), dtype=float)
    if mask is not None and pixel_yx is not None and side.any():
        coords = np.asarray(pixel_yx, dtype=int)[keep][side]
        distance = boundary_distance_weights(mask)
        weights *= distance[coords[:, 0], coords[:, 1]]
    horizontal = normals[side].copy()
    horizontal[:, 2] = 0.0
    hlen = np.linalg.norm(horizontal[:, :2], axis=1)
    valid_h = hlen > EPS
    horizontal, weights = horizontal[valid_h], weights[valid_h]
    horizontal[:, :2] /= hlen[valid_h, None]
    angles = np.degrees(np.arctan2(horizontal[:, 1], horizontal[:, 0])) % 180.0
    clusters = cluster_azimuths_mod180(angles, weights)
    for cluster in clusters:
        cluster["reliable"] = bool(cluster["coverage"] >= 0.05 and cluster["dispersion_p90_degrees"] <= 20.0)
    reliable = [item for item in clusters if item["reliable"]]
    yaw_seeds = sorted({round(item["azimuth_degrees_mod180"], 8) for item in reliable} | {round((item["azimuth_degrees_mod180"] - 90.0) % 180.0, 8) for item in reliable})
    best_yaw, best_residual = None, 180.0
    for yaw in yaw_seeds:
        residual = sum(item["weight"] * min(angle_distance_180(item["azimuth_degrees_mod180"], yaw), angle_distance_180(item["azimuth_degrees_mod180"], yaw + 90.0)) for item in reliable) / max(EPS, sum(item["weight"] for item in reliable))
        if residual < best_residual:
            best_yaw, best_residual = float(yaw), float(residual)
    orthogonality_error = None
    if len(reliable) >= 2:
        orthogonality_error = abs(angle_distance_180(reliable[0]["azimuth_degrees_mod180"], reliable[1]["azimuth_degrees_mod180"]) - 90.0)
    side_coverage = float(side.sum() / max(1, len(normals)))
    ambiguity = "none" if len(reliable) >= 2 and (orthogonality_error or 0) <= ORTHOGONALITY_THRESHOLD_DEGREES else "single_face_90_degree" if len(reliable) == 1 else "insufficient"
    confidence = side_coverage * min(1.0, len(reliable) / 2.0) * max(0.0, 1.0 - best_residual / 25.0)
    if ambiguity == "single_face_90_degree": confidence *= 0.65
    return {
        "thresholds": {"side_abs_dot_gravity_max": SIDE_DOT_MAX, "up_abs_dot_gravity_min": UP_DOT_MIN, "cluster_threshold_degrees": CLUSTER_THRESHOLD_DEGREES},
        "numeric_normal_count": int(len(normals)),
        "up_or_down_count": int(upward.sum()),
        "side_normal_count": int(side.sum()),
        "oblique_or_unreliable_count": int((~side & ~upward).sum()),
        "side_normal_coverage": side_coverage,
        "clusters": clusters,
        "reliable_cluster_count": len(reliable),
        "supported_visible_faces": len(reliable),
        "selected_horizontal_axes_degrees": None if best_yaw is None else [best_yaw, (best_yaw + 90.0) % 180.0],
        "normal_derived_yaw_degrees": best_yaw,
        "normal_frame_residual_degrees": None if best_yaw is None else best_residual,
        "orthogonality_error_degrees": orthogonality_error,
        "ambiguity": ambiguity,
        "orientation_confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def classify_image_edge_families(
    angles_degrees: np.ndarray,
    lengths: np.ndarray,
    projected_vertical_angle_degrees: float,
    vertical_tolerance_degrees: float = 12.0,
) -> dict[str, Any]:
    angles = np.mod(np.asarray(angles_degrees, dtype=float), 180.0)
    lengths = np.asarray(lengths, dtype=float)
    vertical = np.asarray([angle_distance_180(a, projected_vertical_angle_degrees) <= vertical_tolerance_degrees for a in angles])
    horizontal_clusters = cluster_azimuths_mod180(angles[~vertical], lengths[~vertical], threshold_degrees=10.0)
    return {
        "projected_vertical_angle_degrees": float(projected_vertical_angle_degrees % 180.0),
        "vertical_segment_count": int(vertical.sum()),
        "horizontal_segment_count": int((~vertical).sum()),
        "vertical_angles_degrees": angles[vertical].tolist(),
        "horizontal_clusters": horizontal_clusters,
    }


def extract_mask_edge_families(mask: np.ndarray, projected_vertical_angle_degrees: float) -> tuple[dict[str, Any], list[dict[str, float]]]:
    mask = np.asarray(mask, bool)
    padded = np.pad(mask, 1)
    boundary = mask & ~(
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    y, x = np.nonzero(boundary)
    points = np.column_stack([x, y])
    segments = []
    for point in points[::max(1, len(points) // 300)]:
        delta = points - point
        nearby = points[(delta[:, 0] ** 2 + delta[:, 1] ** 2) <= 36]
        if len(nearby) < 5:
            continue
        centered = nearby - nearby.mean(axis=0)
        _, values, vh = np.linalg.svd(centered, full_matrices=False)
        coherence = float(values[0] / max(EPS, values.sum()))
        if coherence < 0.72:
            continue
        direction = vh[0]
        segments.append({"x": float(point[0]), "y": float(point[1]), "angle_degrees": float(math.degrees(math.atan2(direction[1], direction[0])) % 180), "length": float(values[0]), "coherence": coherence})
    families = classify_image_edge_families(np.asarray([s["angle_degrees"] for s in segments]), np.asarray([s["length"] * s["coherence"] for s in segments]), projected_vertical_angle_degrees) if segments else classify_image_edge_families(np.asarray([]), np.asarray([]), projected_vertical_angle_degrees)
    return families, segments


def footprint_yaw_observable(dimensions: np.ndarray, ratio_threshold: float = 1.15) -> bool:
    dims = np.sort(np.asarray(dimensions, dtype=float)[:2])
    return bool(dims[1] / max(EPS, dims[0]) >= ratio_threshold)


def generate_yaw_candidates(frame: dict[str, Any], edge_families: dict[str, Any], previous_yaw: float, maximum: int = 4) -> list[dict[str, Any]]:
    proposals: list[tuple[float, str]] = []
    for cluster in frame["clusters"]:
        if cluster["reliable"]:
            proposals += [(cluster["azimuth_degrees_mod180"], "side_normal"), ((cluster["azimuth_degrees_mod180"] + 90.0) % 180.0, "side_normal_axis_swap")]
    if frame["normal_derived_yaw_degrees"] is not None:
        proposals.insert(0, (frame["normal_derived_yaw_degrees"], "manhattan_normal_frame"))
    for cluster in edge_families.get("horizontal_clusters", [])[:2]:
        proposals.append((cluster.get("world_yaw_degrees", cluster["azimuth_degrees_mod180"]), "projected_mask_edge_family"))
    proposals.append((previous_yaw % 180.0, "previous_pose_fallback"))
    proposals.append(((previous_yaw + 90.0) % 180.0, "previous_pose_axis_swap"))
    unique=[]
    for yaw, source in proposals:
        if all(angle_distance_180(yaw, item["yaw_degrees"]) > 1.0 for item in unique):
            unique.append({"candidate_id": f"candidate_{len(unique)+1}", "yaw_degrees": float(yaw % 180.0), "source": source})
        if len(unique) >= maximum: break
    return unique


def robust_dimensions_in_frame(points_world: np.ndarray, yaw_degrees: float, minimum: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    rotation = yaw_rotation(yaw_degrees)
    local = np.asarray(points_world) @ rotation
    lower, upper = np.percentile(local, [2, 98], axis=0)
    dimensions = np.maximum(upper - lower, minimum)
    center_local = (lower + upper) / 2
    center = center_local @ rotation.T
    center[2] = dimensions[2] / 2.0
    return center, dimensions


def orientation_score(metrics: dict[str, float], gates: dict[str, bool] | None = None) -> dict[str, Any]:
    """Normals dominate; high bbox/hull overlap cannot rescue wrong yaw."""
    normal = max(0.0, 1.0 - metrics["normal_residual_degrees"] / 30.0)
    edge = max(0.0, 1.0 - metrics["horizontal_edge_error_degrees"] / 30.0)
    hull = float(np.clip(metrics["hull_iou"], 0, 1))
    support = max(0.0, 1.0 - metrics["support_error"] / 0.05)
    centroid = max(0.0, 1.0 - metrics["centroid_error_pixels"] / 30.0)
    collision = float(np.clip(metrics.get("collision_risk", 0.0), 0, 1))
    score = 0.35 * normal + 0.25 * edge + 0.15 * hull + 0.10 * support + 0.10 * centroid + 0.05 * (1.0 - collision)
    hard = []
    if metrics["horizontal_edge_error_degrees"] > 15.0: hard.append("horizontal_edge_error_above_15_degrees")
    if metrics.get("normal_coverage", 1.0) < 0.05: hard.append("insufficient_side_normal_coverage")
    if metrics["normal_residual_degrees"] > 20.0: hard.append("normal_frame_residual_above_20_degrees")
    for name, failed in (gates or {}).items():
        if failed: hard.append(name)
    return {"score": float(score), "orientation_valid": not hard, "hard_review_gates": hard, "weights": {"normal":0.35,"horizontal_edge":0.25,"hull":0.15,"support":0.10,"centroid":0.10,"collision":0.05}}


def transform_from_yaw(center: np.ndarray, dimensions: np.ndarray, yaw: float) -> dict[str, Any]:
    rotation = yaw_rotation(yaw)
    return {"center": np.asarray(center).tolist(), "dimensions": np.asarray(dimensions).tolist(), "rotation_matrix": rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(rotation), "yaw_degrees": float(yaw)}
