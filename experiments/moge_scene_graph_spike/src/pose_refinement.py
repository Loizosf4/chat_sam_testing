"""Deterministic mask, normal, pose, and projection refinement primitives.

The functions in this module are Blender-neutral and operate only on numeric SAM
masks, MoGe arrays, and explicit camera/coordinate transforms.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull


EPS = 1e-9


def normalize_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite unit vectors and the row mask retained from the input."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    lengths = np.linalg.norm(values, axis=1)
    keep = np.isfinite(values).all(axis=1) & np.isfinite(lengths) & (lengths > EPS)
    return values[keep] / lengths[keep, None], keep


def erode_mask(mask: np.ndarray, boundary_width: int = 2, minimum_retained_ratio: float = 0.2) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove a narrow boundary band, falling back when erosion is destructive."""
    binary = np.asarray(mask, dtype=bool)
    if boundary_width <= 0 or not binary.any():
        return binary.copy(), {"boundary_width": 0, "fallback": False, "retained_ratio": 1.0}
    eroded = ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool), iterations=boundary_width)
    ratio = float(eroded.sum() / max(1, binary.sum()))
    fallback = not eroded.any() or ratio < minimum_retained_ratio
    return (binary.copy() if fallback else eroded), {
        "boundary_width": boundary_width,
        "fallback": fallback,
        "retained_ratio": 1.0 if fallback else ratio,
    }


def transform_normals_rotation_only(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform normals with only the rotational part of a source-to-world matrix."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape not in {(3, 3), (4, 4)}:
        raise ValueError("transform must be 3x3 or 4x4")
    rotation = matrix[:3, :3]
    transformed = np.asarray(normals, dtype=np.float64) @ rotation.T
    unit, _ = normalize_vectors(transformed)
    return unit


def extract_masked_normals(
    normal_map: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    camera_to_canonical: np.ndarray,
    *,
    boundary_width: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract numeric interior normals and convert them to canonical coordinates."""
    normals = np.asarray(normal_map, dtype=np.float64)
    if normals.shape[:2] != np.asarray(mask).shape or normals.shape[2] != 3:
        raise ValueError("normal map and mask dimensions differ")
    interior, erosion = erode_mask(mask, boundary_width)
    selection = interior & np.asarray(valid_mask, dtype=bool)
    selected = normals[selection]
    selected_unit, kept = normalize_vectors(selected)
    transformed = transform_normals_rotation_only(selected_unit, camera_to_canonical)
    return transformed, {
        "mask_pixels": int(np.asarray(mask, dtype=bool).sum()),
        "interior_pixels": int(interior.sum()),
        "selected_pixels": int(selection.sum()),
        "valid_numeric_normals": int(len(transformed)),
        "invalid_numeric_normals": int(len(selected) - int(kept.sum())),
        "erosion": erosion,
    }


def _farthest_centers(unit: np.ndarray, k: int) -> np.ndarray:
    mean = unit.mean(axis=0)
    mean /= max(EPS, np.linalg.norm(mean))
    centers = [unit[int(np.argmax(unit @ mean))]]
    while len(centers) < k:
        similarity = np.max(unit @ np.stack(centers).T, axis=1)
        centers.append(unit[int(np.argmin(similarity))])
    return np.stack(centers)


def _spherical_kmeans(unit: np.ndarray, k: int, iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    centers = _farthest_centers(unit, k)
    for _ in range(iterations):
        labels = np.argmax(unit @ centers.T, axis=1)
        updated = centers.copy()
        for index in range(k):
            members = unit[labels == index]
            if len(members):
                center = members.mean(axis=0)
                updated[index] = center / max(EPS, np.linalg.norm(center))
        if np.allclose(updated, centers, atol=1e-10):
            centers = updated
            break
        centers = updated
    return np.argmax(unit @ centers.T, axis=1).astype(np.int32), centers


def cluster_normals_spherical(
    normals: np.ndarray,
    *,
    max_clusters: int = 4,
    target_p90_degrees: float = 18.0,
    minimum_coverage: float = 0.025,
) -> list[dict[str, Any]]:
    """Adaptively cluster incompatible unit normals without averaging them together."""
    unit, _ = normalize_vectors(normals)
    if not len(unit):
        return []
    chosen: tuple[np.ndarray, np.ndarray] | None = None
    for k in range(1, min(max_clusters, len(unit)) + 1):
        labels, centers = _spherical_kmeans(unit, k)
        residual = np.degrees(np.arccos(np.clip(np.sum(unit * centers[labels], axis=1), -1.0, 1.0)))
        chosen = labels, centers
        if float(np.percentile(residual, 90)) <= target_p90_degrees:
            break
    assert chosen is not None
    labels, centers = chosen
    records: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        member = unit[labels == index]
        coverage = float(len(member) / len(unit))
        if coverage < minimum_coverage:
            continue
        angles = np.degrees(np.arccos(np.clip(member @ center, -1.0, 1.0)))
        up = abs(float(center[2]))
        classification = "upward" if up >= 0.75 else "vertical" if up <= 0.35 else "oblique"
        dispersion = float(np.percentile(angles, 90))
        confidence = float(np.clip(coverage * max(0.0, 1.0 - dispersion / 45.0), 0.0, 1.0))
        records.append({
            "cluster_id": index,
            "center": center.tolist(),
            "point_count": int(len(member)),
            "coverage_ratio": coverage,
            "angular_dispersion_degrees": {
                "median": float(np.median(angles)),
                "p90": dispersion,
                "maximum": float(angles.max()),
            },
            "confidence": confidence,
            "classification": classification,
        })
    return sorted(records, key=lambda item: (-item["point_count"], item["cluster_id"]))


def _angle_difference_180(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


def minimum_area_rectangle(points_xy: np.ndarray) -> dict[str, Any]:
    """Fit a deterministic minimum-area rectangle to 2D points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("minimum-area rectangle needs at least two points")
    unique = np.unique(points, axis=0)
    if len(unique) == 2 or np.linalg.matrix_rank(unique - unique.mean(axis=0)) < 2:
        centered = unique - unique.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        coordinates = centered @ direction
        low, high = float(coordinates.min()), float(coordinates.max())
        center = unique.mean(axis=0) + direction * (low + high) / 2.0
        normal = np.asarray([-direction[1], direction[0]])
        half_width, half_height = max(0.5, (high - low) / 2.0), 0.5
        corners = np.asarray([
            center - direction * half_width - normal * half_height,
            center + direction * half_width - normal * half_height,
            center + direction * half_width + normal * half_height,
            center - direction * half_width + normal * half_height,
        ])
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        angle = (angle + 90.0) % 180.0 - 90.0
        return {"center": center.tolist(), "width": float(2 * half_width), "height": 1.0, "angle_degrees": angle, "corners": corners.tolist()}
    hull = unique[ConvexHull(unique).vertices]
    best: tuple[float, np.ndarray, np.ndarray, float, np.ndarray] | None = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        angle = math.atan2(edge[1], edge[0])
        rotation = np.asarray([[math.cos(-angle), -math.sin(-angle)], [math.sin(-angle), math.cos(-angle)]])
        rotated = hull @ rotation.T
        minimum, maximum = rotated.min(axis=0), rotated.max(axis=0)
        dimensions = maximum - minimum
        area = float(np.prod(dimensions))
        center_rotated = (minimum + maximum) / 2.0
        inverse = rotation.T
        corners_rotated = np.asarray([
            [minimum[0], minimum[1]], [maximum[0], minimum[1]],
            [maximum[0], maximum[1]], [minimum[0], maximum[1]],
        ])
        candidate = area, center_rotated @ inverse.T, dimensions, math.degrees(angle), corners_rotated @ inverse.T
        if best is None or area < best[0] - 1e-9 or (abs(area - best[0]) <= 1e-9 and candidate[3] < best[3]):
            best = candidate
    assert best is not None
    _, center, dimensions, angle, corners = best
    width, height = map(float, dimensions)
    if height > width:
        width, height = height, width
        angle += 90.0
    angle = (angle + 90.0) % 180.0 - 90.0
    return {"center": center.tolist(), "width": width, "height": height, "angle_degrees": angle, "corners": corners.tolist()}


def extract_mask_axis(mask: np.ndarray) -> dict[str, Any]:
    """Extract contour-scale orientation with disconnected-component confidence reduction."""
    binary = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.int8))
    components: list[dict[str, Any]] = []
    all_y, all_x = np.nonzero(binary)
    if len(all_x) < 2:
        return {"valid": False, "fallback_reason": "fewer than two mask pixels", "orientation_confidence": 0.0}
    overall = minimum_area_rectangle(np.column_stack([all_x, all_y]))
    for component_id in range(1, count + 1):
        y, x = np.nonzero(labels == component_id)
        if len(x) < 2:
            continue
        fitted = minimum_area_rectangle(np.column_stack([x, y]))
        fitted["component_id"] = component_id
        fitted["pixel_count"] = int(len(x))
        components.append(fitted)
    total = sum(item["pixel_count"] for item in components) or 1
    disagreement = sum(item["pixel_count"] * _angle_difference_180(item["angle_degrees"], overall["angle_degrees"]) for item in components) / total
    anisotropy = float(np.clip(1.0 - overall["height"] / max(EPS, overall["width"]), 0.0, 1.0))
    agreement = float(np.clip(1.0 - disagreement / 45.0, 0.0, 1.0))
    fragmentation = 1.0 / math.sqrt(max(1, count))
    confidence = float(np.clip((0.25 + 0.75 * anisotropy) * agreement * fragmentation, 0.0, 1.0))
    return {
        "valid": True,
        "dominant_axis_angle_degrees": overall["angle_degrees"],
        "secondary_axis_angle_degrees": (overall["angle_degrees"] + 90.0) % 180.0,
        "oriented_width_pixels": overall["width"],
        "oriented_height_pixels": overall["height"],
        "center_xy": overall["center"],
        "corners_xy": overall["corners"],
        "component_count": int(count),
        "component_weighted_disagreement_degrees": float(disagreement),
        "orientation_confidence": confidence,
        "components": components,
        "method": "component-weighted minimum-area rectangles plus full-mask convex hull",
    }


def orthonormal_basis_from_forward_up(forward: np.ndarray, gravity_up: np.ndarray) -> np.ndarray:
    """Construct right/forward/up columns as a right-handed orthonormal basis."""
    f = np.asarray(forward, dtype=np.float64)
    f /= max(EPS, np.linalg.norm(f))
    gravity = np.asarray(gravity_up, dtype=np.float64)
    gravity /= max(EPS, np.linalg.norm(gravity))
    up = gravity - float(gravity @ f) * f
    if np.linalg.norm(up) < 1e-7:
        raise ValueError("forward is parallel to gravity")
    up /= np.linalg.norm(up)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    basis = np.column_stack([right, f, up])
    if np.linalg.det(basis) < 0:
        right *= -1.0
        basis = np.column_stack([right, f, up])
    return basis


def rotation_candidates_from_basis(basis: np.ndarray) -> list[np.ndarray]:
    """Generate deterministic 90/180-degree ambiguity candidates around local up."""
    base = np.asarray(basis, dtype=np.float64)
    candidates: list[np.ndarray] = []
    for degrees in (0.0, 90.0, 180.0, 270.0):
        angle = math.radians(degrees)
        local = np.asarray([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        candidate = base @ local
        if np.linalg.det(candidate) > 0.0:
            candidates.append(candidate)
    return candidates


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    """Convert a proper 3x3 rotation matrix to a canonical-sign quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1.0
    return q.tolist()


def cuboid_corners(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    signs = np.asarray([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    return np.asarray(center) + (signs * np.asarray(dimensions)) @ np.asarray(rotation).T


def project_camera_points(points_camera: np.ndarray, normalized_intrinsics: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Project positive-Z camera points using normalized intrinsics."""
    points = np.asarray(points_camera, dtype=np.float64)
    if np.any(points[:, 2] <= EPS):
        raise ValueError("camera projection requires positive depth")
    k = np.asarray(normalized_intrinsics, dtype=np.float64)
    normalized = points[:, :2] / points[:, 2, None]
    uv = normalized @ k[:2, :2].T + k[:2, 2]
    height, width = image_shape
    return uv * np.asarray([width, height])


def project_world_points(points: np.ndarray, canonical_to_camera: np.ndarray, normalized_intrinsics: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(canonical_to_camera, dtype=np.float64)
    camera = np.asarray(points, dtype=np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    return project_camera_points(camera, normalized_intrinsics, image_shape)


def projected_cuboid_rectangle(center: np.ndarray, dimensions: np.ndarray, rotation: np.ndarray, canonical_to_camera: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    corners = cuboid_corners(center, dimensions, rotation)
    projected = project_world_points(corners, canonical_to_camera, intrinsics, image_shape)
    rectangle = minimum_area_rectangle(projected)
    rectangle["projected_corners"] = projected.tolist()
    return rectangle


def score_rotation_candidate(
    rotation: np.ndarray,
    *,
    center: np.ndarray,
    dimensions: np.ndarray,
    target_axis_angle: float,
    target_width: float,
    target_height: float,
    canonical_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    dominant_normal: np.ndarray | None,
    gravity_up: np.ndarray,
) -> dict[str, float]:
    projected = projected_cuboid_rectangle(center, dimensions, rotation, canonical_to_camera, intrinsics, image_shape)
    edge_error = _angle_difference_180(projected["angle_degrees"], target_axis_angle)
    width_error = abs(projected["width"] - target_width) / max(1.0, target_width)
    height_error = abs(projected["height"] - target_height) / max(1.0, target_height)
    normal_error = 0.0 if dominant_normal is None else math.degrees(math.acos(np.clip(abs(float(np.dot(rotation[:, 1], dominant_normal))), -1.0, 1.0)))
    gravity_error = math.degrees(math.acos(np.clip(float(np.dot(rotation[:, 2], gravity_up)), -1.0, 1.0)))
    total = 0.35 * edge_error / 90.0 + 0.2 * normal_error / 90.0 + 0.2 * gravity_error / 90.0 + 0.125 * width_error + 0.125 * height_error
    return {"score": float(total), "projected_edge_angle_error_degrees": float(edge_error), "normal_alignment_error_degrees": float(normal_error), "gravity_alignment_error_degrees": float(gravity_error), "projected_width_relative_error": float(width_error), "projected_height_relative_error": float(height_error)}


def conservative_scale_fit(current_dimensions: np.ndarray, projected_rectangle: dict[str, Any], mask_axis: dict[str, Any], *, thickness_axis: int = 1, depth_samples: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively fit two visible dimensions and clamp depth-derived thickness."""
    dimensions = np.asarray(current_dimensions, dtype=np.float64).copy()
    old = dimensions.copy()
    ratios = np.asarray([
        mask_axis["oriented_width_pixels"] / max(EPS, projected_rectangle["width"]),
        mask_axis["oriented_height_pixels"] / max(EPS, projected_rectangle["height"]),
    ])
    ratios = np.clip(ratios, 0.7, 1.3)
    visible_axes = [index for index in range(3) if index != thickness_axis]
    dimensions[visible_axes[0]] *= ratios[0]
    dimensions[visible_axes[1]] *= ratios[1]
    depth_spread = None
    if depth_samples is not None and len(depth_samples) >= 5:
        depth_spread = float(np.percentile(depth_samples, 90) - np.percentile(depth_samples, 10))
        lower = max(0.025, 0.05 * max(dimensions[visible_axes]))
        upper = max(lower, 0.35 * max(dimensions[visible_axes]))
        dimensions[thickness_axis] = float(np.clip(depth_spread, lower, upper))
    dimensions = np.maximum(dimensions, 1e-4)
    return dimensions, {"original_dimensions": old.tolist(), "refined_dimensions": dimensions.tolist(), "projected_scale_ratios_clamped": ratios.tolist(), "robust_depth_spread": depth_spread, "thickness_axis": thickness_axis, "policy": "visible projected axes fitted within 0.7-1.3; thickness clipped to 5-35% of visible maximum"}
