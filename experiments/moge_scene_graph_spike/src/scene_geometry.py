"""Deterministic structural-plane, coordinate, and support-evidence algorithms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull


def orient_plane_toward_point(normal: np.ndarray, offset: float, interior_point: np.ndarray) -> tuple[np.ndarray, float]:
    """Orient a unit plane normal toward a known occupied/interior point."""
    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    offset = float(offset)
    signed = float(normal @ interior_point + offset)
    if signed < 0 or (abs(signed) < 1e-12 and tuple(normal) < tuple(-normal)):
        normal, offset = -normal, -offset
    return normal, offset


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold: float,
    iterations: int = 500,
    seed: int = 0,
    max_sample_points: int = 20000,
) -> dict[str, Any]:
    """Fit a plane robustly and deterministically, retaining full-set inliers."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return {"success": False, "reason": "fewer than three 3D points"}
    if not np.isfinite(points).all():
        return {"success": False, "reason": "non-finite input points"}
    rng = np.random.default_rng(seed)
    if len(points) > max_sample_points:
        sample_indices = np.sort(rng.choice(len(points), max_sample_points, replace=False))
        sample = points[sample_indices]
    else:
        sample = points
    best_count = 0
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    for _ in range(iterations):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        offset = -float(normal @ tri[0])
        count = int((np.abs(sample @ normal + offset) <= threshold).sum())
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset
    if best_normal is None or best_count < 3:
        return {"success": False, "reason": "RANSAC did not find a non-degenerate plane"}

    preliminary = np.abs(points @ best_normal + best_offset) <= threshold
    if preliminary.sum() < 3:
        return {"success": False, "reason": "full-set plane has fewer than three inliers"}
    centroid = points[preliminary].mean(axis=0)
    _, singular_values, vh = np.linalg.svd(points[preliminary] - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    residuals = np.abs(points @ normal + offset)
    inliers = residuals <= threshold
    inlier_residuals = residuals[inliers]
    return {
        "success": True,
        "normal": normal,
        "offset": offset,
        "inlier_mask": inliers,
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "residuals": residuals,
        "residual_statistics": {
            "median": float(np.median(inlier_residuals)),
            "p90": float(np.percentile(inlier_residuals, 90)),
            "p95": float(np.percentile(inlier_residuals, 95)),
            "p99": float(np.percentile(inlier_residuals, 99)),
            "maximum": float(inlier_residuals.max()),
        },
        "singular_values": singular_values.tolist(),
        "threshold": threshold,
    }


def deterministic_normal_clusters(normals: np.ndarray, k: int = 8, iterations: int = 20, max_fit: int = 25000) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means with deterministic farthest-point initialization."""
    normals = np.asarray(normals, dtype=np.float64)
    if len(normals) < k:
        raise ValueError("normal clustering needs at least k samples")
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if len(unit) > max_fit:
        sample = unit[np.linspace(0, len(unit) - 1, max_fit, dtype=np.int64)]
    else:
        sample = unit
    mean = sample.mean(axis=0)
    mean /= np.linalg.norm(mean)
    first = int(np.argmax(sample @ mean))
    centers = [sample[first]]
    for _ in range(1, k):
        similarity = np.max(sample @ np.stack(centers).T, axis=1)
        centers.append(sample[int(np.argmin(similarity))])
    centers_array = np.stack(centers)
    for _ in range(iterations):
        labels = np.argmax(sample @ centers_array.T, axis=1)
        updated = centers_array.copy()
        for cluster_id in range(k):
            members = sample[labels == cluster_id]
            if len(members):
                center = members.mean(axis=0)
                updated[cluster_id] = center / np.linalg.norm(center)
        if np.allclose(updated, centers_array, atol=1e-8):
            centers_array = updated
            break
        centers_array = updated
    full_labels = np.argmax(unit @ centers_array.T, axis=1)
    return full_labels.astype(np.int32), centers_array


def _extent_record(points: np.ndarray) -> dict[str, list[float]]:
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    return {"min": minimum.tolist(), "max": maximum.tolist(), "dimensions": (maximum - minimum).tolist()}


def estimate_structural_planes(
    points: np.ndarray,
    normals: np.ndarray,
    image_xy: np.ndarray,
    image_shape: tuple[int, int],
    interior_point: np.ndarray,
    *,
    k: int = 8,
    threshold: float = 0.05,
    minimum_cluster_points: int = 1000,
    minimum_inliers: int = 1800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cluster normals, robustly fit planes, then classify substantial room surfaces."""
    height, width = image_shape
    labels, centers = deterministic_normal_clusters(normals, k=k)
    candidates: list[dict[str, Any]] = []
    for cluster_id in range(k):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) < minimum_cluster_points:
            continue
        fit = fit_plane_ransac(points[cluster_indices], threshold=threshold, seed=4000 + cluster_id)
        if not fit["success"] or fit["inlier_count"] < minimum_inliers:
            continue
        inlier_indices = cluster_indices[fit["inlier_mask"]]
        inlier_points = points[inlier_indices]
        xy = image_xy[inlier_indices]
        normal, offset = orient_plane_toward_point(fit["normal"], fit["offset"], interior_point)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        cx, cy = xy.mean(axis=0)
        bbox_area_ratio = float((x1 - x0 + 1) * (y1 - y0 + 1) / (width * height))
        coverage = float(len(inlier_indices) / (width * height))
        floor_roi = float((xy[:, 1] >= 0.50 * height).mean())
        upper_roi = float((xy[:, 1] <= 0.64 * height).mean())
        left_roi = float((xy[:, 0] <= 0.53 * width).mean())
        right_roi = float((xy[:, 0] >= 0.47 * width).mean())
        candidate = {
            "candidate_id": f"normal_cluster_{cluster_id}",
            "cluster_id": cluster_id,
            "cluster_center_normal": centers[cluster_id].tolist(),
            "normal": normal,
            "offset": offset,
            "inlier_indices": inlier_indices,
            "inlier_count": int(len(inlier_indices)),
            "cluster_point_count": int(len(cluster_indices)),
            "inlier_ratio": fit["inlier_ratio"],
            "residual_statistics": fit["residual_statistics"],
            "image_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            "image_centroid_xy": [float(cx), float(cy)],
            "image_bbox_area_ratio": bbox_area_ratio,
            "image_coverage_ratio": coverage,
            "extent_3d": _extent_record(inlier_points),
            "roi_fractions": {"floor_lower": floor_roi, "upper": upper_roi, "left": left_roi, "right": right_roi},
        }
        support_factor = min(1.0, len(inlier_indices) / 12000.0)
        residual_factor = max(0.0, 1.0 - fit["residual_statistics"]["p95"] / threshold)
        coverage_factor = min(1.0, bbox_area_ratio / 0.25)
        candidate["fit_confidence"] = float(0.4 * support_factor + 0.35 * residual_factor + 0.25 * coverage_factor)
        candidates.append(candidate)

    def score(candidate: dict[str, Any], semantic: str) -> float:
        n = np.asarray(candidate["normal"])
        cx = candidate["image_centroid_xy"][0] / width
        roi = candidate["roi_fractions"]
        base = candidate["fit_confidence"]
        if semantic == "floor":
            orientation = abs(float(n[1]))
            return base * roi["floor_lower"] * orientation
        orientation = abs(float(n[0]))
        side = roi["left"] if semantic == "left_wall" else roi["right"]
        centroid_side = max(0.2, 1.0 - cx) if semantic == "left_wall" else max(0.2, cx)
        return base * roi["upper"] * side * centroid_side * orientation

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for semantic in ["floor", "left_wall", "right_wall"]:
        ranked = sorted(((score(c, semantic), c) for c in candidates if c["candidate_id"] not in used), key=lambda item: (-item[0], item[1]["candidate_id"]))
        if not ranked or ranked[0][0] < 0.12:
            continue
        semantic_score, candidate = ranked[0]
        used.add(candidate["candidate_id"])
        record = dict(candidate)
        record["semantic_candidate"] = semantic
        record["classification_score"] = float(semantic_score)
        record["plane_id"] = f"plane_{semantic}"
        selected.append(record)

    # Add orientation relationships to classification confidence without forcing missing planes.
    for plane in selected:
        other_angles: list[float] = []
        for other in selected:
            if other is plane:
                continue
            dot = abs(float(np.dot(plane["normal"], other["normal"])))
            other_angles.append(math.degrees(math.acos(np.clip(dot, -1.0, 1.0))))
        orthogonality = float(np.mean([max(0.0, 1.0 - abs(angle - 90.0) / 30.0) for angle in other_angles])) if other_angles else 0.0
        plane["orientation_relationship_score"] = orthogonality
        plane["confidence"] = float(np.clip(0.75 * plane["fit_confidence"] + 0.25 * orthogonality, 0.0, 1.0))
        plane["classification_evidence"] = {
            "normal_cluster": plane["candidate_id"],
            "fit_confidence": plane["fit_confidence"],
            "classification_score": plane["classification_score"],
            "roi_fractions": plane["roi_fractions"],
            "orientation_relationship_score": orthogonality,
            "substantial_coverage": plane["inlier_count"] >= minimum_inliers and plane["image_bbox_area_ratio"] >= 0.08,
        }
        plane["warnings"] = []
        if plane["image_bbox_area_ratio"] < 0.15:
            plane["warnings"].append("limited 2D image extent")
        if plane["confidence"] < 0.55:
            plane["warnings"].append("plane classification confidence is limited")

    diagnostics = {
        "candidate_count": len(candidates),
        "normal_cluster_centers": centers.tolist(),
        "unclassified_candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"normal", "inlier_indices"}
            }
            for candidate in candidates
            if candidate["candidate_id"] not in used
        ],
    }
    return selected, diagnostics


def construct_canonical_transform(floor_normal: np.ndarray, floor_offset: float, occupied_point: np.ndarray) -> dict[str, Any]:
    """Build a right-handed raw-camera to canonical-world rigid transform."""
    z_axis, floor_offset = orient_plane_toward_point(floor_normal, floor_offset, occupied_point)
    camera_right = np.asarray([1.0, 0.0, 0.0])
    x_axis = camera_right - float(camera_right @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        raise ValueError("camera-right is parallel to floor normal")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation_canonical_to_raw = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation_canonical_to_raw) < 0:
        y_axis *= -1
        rotation_canonical_to_raw = np.column_stack([x_axis, y_axis, z_axis])
    signed_distance = float(z_axis @ occupied_point + floor_offset)
    origin_raw = occupied_point - signed_distance * z_axis
    raw_to_canonical = np.eye(4, dtype=np.float64)
    raw_to_canonical[:3, :3] = rotation_canonical_to_raw.T
    raw_to_canonical[:3, 3] = -rotation_canonical_to_raw.T @ origin_raw
    canonical_to_raw = np.eye(4, dtype=np.float64)
    canonical_to_raw[:3, :3] = rotation_canonical_to_raw
    canonical_to_raw[:3, 3] = origin_raw
    return {
        "raw_to_canonical": raw_to_canonical,
        "canonical_to_raw": canonical_to_raw,
        "origin_raw": origin_raw,
        "axes_raw": {"x": x_axis, "y": y_axis, "z": z_axis},
        "determinant": float(np.linalg.det(rotation_canonical_to_raw)),
        "floor_offset_oriented": floor_offset,
    }


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = np.asarray(normals, dtype=np.float64) @ matrix[:3, :3].T
    return transformed / np.linalg.norm(transformed, axis=1, keepdims=True)


def transform_plane(normal: np.ndarray, offset: float, raw_to_target: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform a plane under a rigid or uniform-scale affine transform."""
    inverse = np.linalg.inv(raw_to_target)
    coefficients = np.concatenate([np.asarray(normal, dtype=np.float64), [float(offset)]])
    transformed = inverse.T @ coefficients
    length = np.linalg.norm(transformed[:3])
    return transformed[:3] / length, float(transformed[3] / length)


def plane_intersection(plane_a: tuple[np.ndarray, float], plane_b: tuple[np.ndarray, float]) -> dict[str, Any]:
    n1, d1 = plane_a
    n2, d2 = plane_b
    direction = np.cross(n1, n2)
    sine = float(np.linalg.norm(direction))
    if sine < 1e-6:
        return {"stable": False, "reason": "planes are parallel or nearly parallel", "condition_sine": sine}
    direction /= sine
    system = np.vstack([n1, n2, direction])
    rhs = np.asarray([-d1, -d2, 0.0])
    point = np.linalg.solve(system, rhs)
    return {
        "stable": True,
        "point": point,
        "direction": direction,
        "condition_sine": sine,
        "angle_degrees": math.degrees(math.asin(np.clip(sine, 0.0, 1.0))),
    }


def detect_upward_support_patch(
    points_canonical: np.ndarray,
    normals_canonical: np.ndarray,
    image_xy: np.ndarray,
    *,
    up_dot_minimum: float = 0.82,
    top_height_quantile: float = 0.60,
    minimum_points: int = 30,
    plane_threshold: float = 0.035,
) -> dict[str, Any]:
    """Detect a visible, upward-facing top patch without using an object AABB."""
    if len(points_canonical) < minimum_points:
        return {"detected": False, "confidence": 0.0, "reason": "insufficient visible points"}
    up_alignment = normals_canonical[:, 2]
    height_cutoff = float(np.quantile(points_canonical[:, 2], top_height_quantile))
    candidates = (up_alignment >= up_dot_minimum) & (points_canonical[:, 2] >= height_cutoff)
    if int(candidates.sum()) < minimum_points:
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "insufficient upward-facing high points",
            "candidate_point_count": int(candidates.sum()),
        }
    candidate_points = points_canonical[candidates]
    fit = fit_plane_ransac(candidate_points, threshold=plane_threshold, seed=991)
    if not fit["success"] or fit["inlier_count"] < minimum_points:
        return {"detected": False, "confidence": 0.0, "reason": "support-patch plane fit failed"}
    inlier_local = fit["inlier_mask"]
    patch_points = candidate_points[inlier_local]
    patch_xy = image_xy[candidates][inlier_local]
    normal = fit["normal"]
    offset = fit["offset"]
    if normal[2] < 0:
        normal, offset = -normal, -offset
    boundary_xy: list[list[float]] = []
    if len(patch_points) >= 3:
        try:
            hull = ConvexHull(patch_points[:, :2])
            boundary_xy = patch_points[hull.vertices, :2].tolist()
        except Exception:
            boundary_xy = []
    coverage = float(len(patch_points) / len(points_canonical))
    alignment = float(np.median(normals_canonical[candidates][inlier_local, 2]))
    residual_factor = max(0.0, 1.0 - fit["residual_statistics"]["p95"] / plane_threshold)
    confidence = float(np.clip(0.35 * min(1.0, len(patch_points) / 500.0) + 0.25 * min(1.0, coverage / 0.2) + 0.2 * alignment + 0.2 * residual_factor, 0.0, 1.0))
    return {
        "detected": confidence >= 0.45,
        "confidence": confidence,
        "point_count": int(len(patch_points)),
        "coverage_ratio_of_visible_object": coverage,
        "plane_canonical": {"normal": normal.tolist(), "offset": float(offset)},
        "residual_statistics": fit["residual_statistics"],
        "height_median": float(np.median(patch_points[:, 2])),
        "image_bbox_xyxy": [int(patch_xy[:, 0].min()), int(patch_xy[:, 1].min()), int(patch_xy[:, 0].max()), int(patch_xy[:, 1].max())],
        "boundary_canonical_xy": boundary_xy,
        "parameters": {
            "up_dot_minimum": up_dot_minimum,
            "top_height_quantile": top_height_quantile,
            "minimum_points": minimum_points,
            "plane_threshold": plane_threshold,
        },
        "point_selection_mask": candidates,
        "patch_candidate_inlier_mask": inlier_local,
    }


def support_distance_evidence(
    subject_points: np.ndarray,
    support_normal: np.ndarray,
    support_offset: float,
    support_boundary_xy: list[list[float]] | None,
    *,
    lower_quantile: float = 0.08,
    distance_threshold: float = 0.10,
) -> dict[str, Any]:
    """Measure robust lower-point distance and horizontal patch overlap."""
    if len(subject_points) == 0:
        return {"supported": False, "confidence": 0.0, "reason": "no subject points"}
    height_cut = float(np.quantile(subject_points[:, 2], lower_quantile))
    lower = subject_points[subject_points[:, 2] <= height_cut]
    signed = lower @ support_normal + support_offset
    absolute = np.abs(signed)
    horizontal_overlap = 0.0
    if support_boundary_xy and len(support_boundary_xy) >= 3:
        polygon = np.asarray(support_boundary_xy)
        try:
            from matplotlib.path import Path as MplPath

            horizontal_overlap = float(MplPath(polygon).contains_points(lower[:, :2], radius=1e-9).mean())
        except Exception:
            horizontal_overlap = 0.0
    distance = float(np.median(absolute))
    close_ratio = float((absolute <= distance_threshold).mean())
    distance_factor = max(0.0, 1.0 - distance / distance_threshold)
    confidence = float(np.clip(0.55 * distance_factor + 0.25 * close_ratio + 0.20 * horizontal_overlap, 0.0, 1.0))
    return {
        "supported": confidence >= 0.45 and distance <= distance_threshold and (horizontal_overlap >= 0.10 or support_boundary_xy is None),
        "confidence": confidence,
        "lower_point_count": int(len(lower)),
        "lower_height_quantile": lower_quantile,
        "median_absolute_plane_distance": distance,
        "signed_distance_median": float(np.median(signed)),
        "distance_p95": float(np.percentile(absolute, 95)),
        "close_point_ratio": close_ratio,
        "horizontal_overlap_ratio": horizontal_overlap,
        "thresholds": {"distance": distance_threshold, "minimum_horizontal_overlap": 0.10},
    }


def wall_attachment_evidence(points: np.ndarray, wall_normal: np.ndarray, wall_offset: float, *, distance_threshold: float = 0.12) -> dict[str, Any]:
    if len(points) == 0:
        return {"attached": False, "confidence": 0.0, "reason": "no visible points"}
    distances = np.abs(points @ wall_normal + wall_offset)
    median = float(np.median(distances))
    p20 = float(np.percentile(distances, 20))
    near_ratio = float((distances <= distance_threshold).mean())
    confidence = float(np.clip(0.55 * max(0.0, 1.0 - p20 / distance_threshold) + 0.45 * near_ratio, 0.0, 1.0))
    return {
        "attached": confidence >= 0.45 and p20 <= distance_threshold,
        "confidence": confidence,
        "distance_median": median,
        "distance_p20": p20,
        "distance_p95": float(np.percentile(distances, 95)),
        "near_point_ratio": near_ratio,
        "distance_threshold": distance_threshold,
    }


def opencv_to_blender_camera_local_matrix() -> np.ndarray:
    """Map OpenCV camera axes (X right,Y down,Z forward) to Blender camera local (X right,Y up,-Z forward)."""
    return np.diag([1.0, -1.0, -1.0, 1.0])


def unknown_support_candidate(subject_object_id: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_object_id": subject_object_id,
        "predicate": "unknown_support",
        "target_id": None,
        "confidence": 1.0,
        "evidence": evidence,
        "thresholds_used": {},
        "evidence_source": ["visible_geometry", "occlusion_metadata"],
        "contradictions": [],
        "uncertainty": reason,
        "requires_vlm_review": True,
        "requires_user_review": True,
    }


def validate_relationship_targets(relationships: list[dict[str, Any]], object_ids: set[str], plane_ids: set[str]) -> None:
    allowed = object_ids | plane_ids
    for relationship in relationships:
        subject = relationship["subject_object_id"]
        target = relationship.get("target_id")
        if subject not in object_ids:
            raise ValueError(f"relationship subject is absent from fixture: {subject}")
        if target is not None and target not in allowed:
            raise ValueError(f"relationship target is absent from fixture/planes: {target}")
