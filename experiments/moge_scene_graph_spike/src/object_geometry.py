"""Deterministic geometry algorithms for visible semantic-object masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage


PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Label 8-connected visible mask components in deterministic scan order."""
    labels, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    records: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        ys, xs = np.nonzero(component)
        records.append(
            {
                "component_id": component_id,
                "pixel_count": int(component.sum()),
                "bbox_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
            }
        )
    return labels, records


def robust_depth_filter(
    depth: np.ndarray,
    object_valid: np.ndarray,
    component_labels: np.ndarray,
    *,
    mad_multiplier: float = 8.0,
    minimum_global_deviation_m: float = 0.25,
    minimum_local_deviation_m: float = 0.15,
    minimum_neighbors: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject only globally extreme samples that also disagree with local peers.

    Statistics are computed per visible component. A candidate is rejected only
    when at least ``minimum_neighbors`` same-component 8-neighbors exist and its
    depth disagrees with their median. This preserves isolated/thin components.
    """
    keep = object_valid.copy()
    candidates = np.zeros_like(object_valid, dtype=bool)
    rejected = np.zeros_like(object_valid, dtype=bool)
    component_parameters: list[dict[str, Any]] = []
    for component_id in range(1, int(component_labels.max()) + 1):
        component_valid = object_valid & (component_labels == component_id)
        values = depth[component_valid]
        if values.size < 8:
            component_parameters.append(
                {"component_id": component_id, "valid_points": int(values.size), "filter_applied": False, "reason": "fewer than 8 valid samples"}
            )
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(minimum_global_deviation_m, mad_multiplier * robust_sigma)
        component_candidates = component_valid & (np.abs(depth - median) > threshold)
        candidates |= component_candidates
        for y, x in np.argwhere(component_candidates):
            y0, y1 = max(0, y - 1), min(depth.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(depth.shape[1], x + 2)
            neighborhood = component_valid[y0:y1, x0:x1].copy()
            neighborhood[y - y0, x - x0] = False
            neighbor_depths = depth[y0:y1, x0:x1][neighborhood]
            if neighbor_depths.size >= minimum_neighbors:
                local_deviation = abs(float(depth[y, x]) - float(np.median(neighbor_depths)))
                if local_deviation > minimum_local_deviation_m:
                    rejected[y, x] = True
        component_parameters.append(
            {
                "component_id": component_id,
                "valid_points": int(values.size),
                "filter_applied": True,
                "median_depth_m": median,
                "mad_m": mad,
                "robust_sigma_m": robust_sigma,
                "global_candidate_threshold_m": threshold,
                "candidate_count": int(component_candidates.sum()),
                "rejected_count": int((rejected & (component_labels == component_id)).sum()),
            }
        )
    keep[rejected] = False
    return keep, {
        "candidate_mask": candidates,
        "rejected_mask": rejected,
        "component_parameters": component_parameters,
        "parameters": {
            "mad_multiplier": mad_multiplier,
            "minimum_global_deviation_m": minimum_global_deviation_m,
            "minimum_local_deviation_m": minimum_local_deviation_m,
            "minimum_neighbors": minimum_neighbors,
        },
    }


@dataclass(frozen=True)
class SceneNormalization:
    origin: np.ndarray
    scale: float
    robust_bounds_min: np.ndarray
    robust_bounds_max: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) * self.scale

    def invert(self, points: np.ndarray) -> np.ndarray:
        return points / self.scale + self.origin

    def to_record(self) -> dict[str, Any]:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] *= self.scale
        matrix[:3, 3] = -self.origin * self.scale
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] /= self.scale
        inverse[:3, 3] = self.origin
        return {
            "rule": "origin is center of scene-wide p1/p99 camera-space bounds; uniform scale maps their longest side to 1.0",
            "origin_raw_moge_xyz": self.origin.tolist(),
            "scale_factor": self.scale,
            "scene_robust_bounds_raw_moge": {"min": self.robust_bounds_min.tolist(), "max": self.robust_bounds_max.tolist()},
            "forward_formula": "scene_normalized = (raw_moge - origin_raw_moge_xyz) * scale_factor",
            "inverse_formula": "raw_moge = scene_normalized / scale_factor + origin_raw_moge_xyz",
            "raw_to_normalized_matrix_4x4": matrix.tolist(),
            "normalized_to_raw_matrix_4x4": inverse.tolist(),
            "preserves_aspect_ratios": True,
            "reversible": True,
        }


def make_scene_normalization(scene_points: np.ndarray) -> SceneNormalization:
    if scene_points.ndim != 2 or scene_points.shape[1] != 3 or len(scene_points) == 0:
        raise ValueError("scene_points must have shape (N, 3) with N > 0")
    if not np.isfinite(scene_points).all():
        raise ValueError("scene_points contain non-finite values")
    low, high = np.percentile(scene_points, [1, 99], axis=0)
    extent = high - low
    longest = float(extent.max())
    if longest <= 0:
        raise ValueError("scene geometry has zero robust extent")
    return SceneNormalization(origin=(low + high) / 2.0, scale=1.0 / longest, robust_bounds_min=low, robust_bounds_max=high)


def bounds_record(points: np.ndarray, normalization: SceneNormalization | None = None) -> dict[str, Any] | None:
    if len(points) == 0:
        return None
    working = normalization.apply(points) if normalization else points
    minimum, maximum = working.min(axis=0), working.max(axis=0)
    robust_min, robust_max = np.percentile(working, [2, 98], axis=0)
    return {
        "centroid": working.mean(axis=0).tolist(),
        "axis_aligned_bounds": {"min": minimum.tolist(), "max": maximum.tolist()},
        "robust_bounds_p2_p98": {"min": robust_min.tolist(), "max": robust_max.tolist()},
        "visible_surface_dimensions": (maximum - minimum).tolist(),
        "robust_visible_surface_dimensions": (robust_max - robust_min).tolist(),
    }


def normal_statistics(normals: np.ndarray) -> dict[str, Any] | None:
    if len(normals) == 0:
        return None
    dominant = normals.mean(axis=0)
    magnitude = float(np.linalg.norm(dominant))
    if magnitude < 1e-8:
        return {"dominant_normal_xyz": None, "mean_resultant_length": 0.0, "angular_dispersion_degrees": None}
    dominant /= magnitude
    angles = np.degrees(np.arccos(np.clip(normals @ dominant, -1.0, 1.0)))
    return {
        "dominant_normal_xyz": dominant.tolist(),
        "mean_resultant_length": magnitude,
        "angular_dispersion_degrees": {
            "median": float(np.median(angles)),
            "p90": float(np.percentile(angles, 90)),
            "p95": float(np.percentile(angles, 95)),
        },
    }


def estimate_oriented_bounds(
    points: np.ndarray,
    *,
    valid_geometry_ratio: float = 1.0,
    component_count: int = 1,
    partial_occlusion: bool = False,
) -> dict[str, Any]:
    """Estimate a conservative visible-surface PCA box or decline safely."""
    warnings: list[str] = []
    if len(points) < 100:
        return {
            "estimated": False,
            "method": "PCA on filtered visible camera-space points with p2/p98 extents",
            "confidence": 0.0,
            "warnings": ["insufficient points for stable oriented bounds"],
        }
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, axes = eigenvalues[order], eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    largest = max(float(eigenvalues[0]), np.finfo(float).eps)
    ratios = eigenvalues / largest
    first_separation = float((eigenvalues[0] - eigenvalues[1]) / largest)
    second_separation = float((eigenvalues[1] - eigenvalues[2]) / largest)
    projected = centered @ axes
    low, high = np.percentile(projected, [2, 98], axis=0)
    dimensions = high - low
    local_center = (low + high) / 2.0
    center = points.mean(axis=0) + axes @ local_center

    thin = bool(ratios[2] < 0.003 or dimensions.min() / max(float(dimensions.max()), 1e-9) < 0.02)
    single_surface = bool(ratios[2] < 0.01)
    degenerate_axes = bool(first_separation < 0.08 or second_separation < 0.03)
    if thin:
        warnings.append("visible geometry is thin; thickness and the minor axis are unstable")
    if single_surface:
        warnings.append("geometry is dominated by a single visible surface")
    if degenerate_axes:
        warnings.append("PCA eigenvalues do not separate orientation axes reliably")
    if component_count > 3:
        warnings.append("multiple disconnected components can destabilize a single oriented box")
    if partial_occlusion:
        warnings.append("partial occlusion limits oriented-bound coverage")
    if valid_geometry_ratio < 0.7:
        warnings.append("low valid geometry ratio reduces oriented-bound stability")

    point_factor = min(1.0, math.log10(len(points)) / 4.0)
    separation_factor = float(np.clip((first_separation + second_separation) / 0.8, 0.0, 1.0))
    confidence = point_factor * separation_factor * min(1.0, valid_geometry_ratio / 0.9)
    if thin:
        confidence *= 0.55
    if component_count > 1:
        confidence *= max(0.45, 1.0 - 0.08 * (component_count - 1))
    if partial_occlusion:
        confidence *= 0.65
    reliable = confidence >= 0.55 and not degenerate_axes and not (thin and partial_occlusion)
    record: dict[str, Any] = {
        "estimated": reliable,
        "method": "PCA on filtered visible camera-space points with p2/p98 extents",
        "confidence": float(confidence),
        "eigenvalues": eigenvalues.tolist(),
        "normalized_eigenvalues": ratios.tolist(),
        "degeneracy_warnings": warnings,
        "unstable_factors": {
            "thin": thin,
            "disconnected": component_count > 1,
            "partially_occluded": partial_occlusion,
            "single_visible_surface": single_surface,
        },
    }
    if reliable:
        record.update(
            {
                "center_xyz": center.tolist(),
                "dimensions_xyz_along_axes": dimensions.tolist(),
                "orientation_matrix_columns_are_box_axes": axes.tolist(),
            }
        )
    else:
        record["warning"] = "oriented bounding box omitted because visible geometry does not support a stable estimate"
    return record


def depth_statistics(values: np.ndarray) -> dict[str, Any] | None:
    if len(values) == 0:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "median": median,
        "mean": float(values.mean()),
        "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))},
        "robust_spread": {"mad": mad, "robust_sigma_1_4826_mad": 1.4826 * mad, "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))},
    }


def geometry_confidence(
    *,
    valid_ratio: float,
    valid_count: int,
    component_count: int,
    depth_stats: dict[str, Any] | None,
    normal_stats: dict[str, Any] | None,
    outlier_ratio: float,
    thin: bool,
    partial_occlusion: bool,
    obb_confidence: float,
) -> dict[str, Any]:
    size_factor = float(np.clip(math.log10(max(valid_count, 1)) / 3.0, 0.0, 1.0))
    component_factor = max(0.45, 1.0 - 0.08 * max(0, component_count - 1))
    if depth_stats:
        depth_relative_spread = depth_stats["robust_spread"]["robust_sigma_1_4826_mad"] / max(depth_stats["median"], 1e-6)
        depth_factor = float(np.clip(1.0 - depth_relative_spread / 0.12, 0.0, 1.0))
    else:
        depth_factor = 0.0
    if normal_stats and normal_stats["angular_dispersion_degrees"]:
        normal_factor = float(np.clip(1.0 - normal_stats["angular_dispersion_degrees"]["p90"] / 90.0, 0.0, 1.0))
    else:
        normal_factor = 0.0
    outlier_factor = float(np.clip(1.0 - outlier_ratio / 0.1, 0.0, 1.0))
    score = (
        0.22 * valid_ratio
        + 0.14 * size_factor
        + 0.10 * component_factor
        + 0.16 * depth_factor
        + 0.14 * normal_factor
        + 0.10 * outlier_factor
        + 0.14 * obb_confidence
    )
    if thin:
        score *= 0.88
    if partial_occlusion:
        score *= 0.82
    score = float(np.clip(score, 0.0, 1.0))
    label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return {
        "score": score,
        "label": label,
        "factors": {
            "valid_geometry_ratio": valid_ratio,
            "mask_size": size_factor,
            "component_consistency": component_factor,
            "depth_consistency": depth_factor,
            "normal_consistency": normal_factor,
            "outlier_retention": outlier_factor,
            "thinness_penalty_applied": thin,
            "partial_occlusion_penalty_applied": partial_occlusion,
            "oriented_bounds_confidence": obb_confidence,
        },
    }
