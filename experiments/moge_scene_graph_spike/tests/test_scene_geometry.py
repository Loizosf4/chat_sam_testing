import math

import numpy as np
import pytest

from src.scene_geometry import (
    construct_canonical_transform,
    detect_upward_support_patch,
    estimate_structural_planes,
    fit_plane_ransac,
    opencv_to_blender_camera_local_matrix,
    orient_plane_toward_point,
    support_distance_evidence,
    transform_plane,
    transform_points,
    unknown_support_candidate,
    validate_relationship_targets,
    wall_attachment_evidence,
)


def _plane_points(normal: np.ndarray, offset: float, count: int, rng: np.random.Generator) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    seed = np.asarray([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
    axis_a = np.cross(normal, seed)
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(normal, axis_a)
    origin = -offset * normal
    uv = rng.uniform(-1.5, 1.5, size=(count, 2))
    return origin + uv[:, :1] * axis_a + uv[:, 1:] * axis_b


def test_robust_plane_fitting_tolerates_outliers() -> None:
    rng = np.random.default_rng(4)
    inliers = np.column_stack([rng.uniform(-2, 2, 600), rng.uniform(-2, 2, 600), rng.normal(0, 0.002, 600)])
    outliers = rng.uniform(-2, 2, size=(150, 3))
    result = fit_plane_ransac(np.vstack([inliers, outliers]), threshold=0.01, seed=7)
    assert result["success"]
    assert result["inlier_count"] >= 580
    assert result["residual_statistics"]["p95"] < 0.006


def test_plane_normal_orientation_is_deterministic_and_interior_facing() -> None:
    normal, offset = orient_plane_toward_point(np.asarray([0.0, 0.0, -1.0]), 0.0, np.asarray([0.0, 0.0, 2.0]))
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert offset == 0.0
    normal2, offset2 = orient_plane_toward_point(-normal, -offset, np.asarray([0.0, 0.0, 2.0]))
    assert np.allclose(normal2, normal)
    assert offset2 == offset


def test_structural_plane_classification_finds_floor_and_two_walls() -> None:
    rng = np.random.default_rng(9)
    definitions = [
        ("floor", np.asarray([0.0, -0.9, -0.435]), (0, 260, 447, 447)),
        ("left_wall", np.asarray([0.69, 0.18, -0.70]), (20, 75, 215, 260)),
        ("right_wall", np.asarray([-0.71, 0.23, -0.66]), (225, 75, 425, 265)),
    ]
    points, normals, pixels = [], [], []
    for _, normal, (x0, y0, x1, y1) in definitions:
        normal = normal / np.linalg.norm(normal)
        points.append(_plane_points(normal, 5.0, 2400, rng) + rng.normal(0, 0.003, size=(2400, 1)) * normal)
        normals.append(np.tile(normal, (2400, 1)) + rng.normal(0, 0.005, size=(2400, 3)))
        pixels.append(np.column_stack([rng.integers(x0, x1, 2400), rng.integers(y0, y1, 2400)]))
    planes, _ = estimate_structural_planes(
        np.vstack(points), np.vstack(normals), np.vstack(pixels), (447, 447), np.asarray([0.0, 0.0, 0.0]),
        k=3, threshold=0.02, minimum_cluster_points=500, minimum_inliers=500,
    )
    assert {plane["semantic_candidate"] for plane in planes} == {"floor", "left_wall", "right_wall"}


def test_canonical_transform_is_right_handed_reversible_and_makes_floor_horizontal() -> None:
    floor_normal = np.asarray([0.0, -0.9, -0.435])
    floor_normal /= np.linalg.norm(floor_normal)
    occupied = -4.0 * floor_normal + 1.2 * floor_normal
    transform = construct_canonical_transform(floor_normal, 4.0, occupied)
    assert transform["determinant"] == pytest.approx(1.0, abs=1e-12)
    canonical_normal, canonical_offset = transform_plane(floor_normal, 4.0, transform["raw_to_canonical"])
    assert np.allclose(canonical_normal, [0.0, 0.0, 1.0], atol=1e-10)
    assert canonical_offset == pytest.approx(0.0, abs=1e-10)
    assert transform_points(occupied[None], transform["raw_to_canonical"])[0, 2] > 0
    rng = np.random.default_rng(2)
    points = rng.normal(size=(100, 3))
    canonical = transform_points(points, transform["raw_to_canonical"])
    restored = transform_points(canonical, transform["canonical_to_raw"])
    assert np.allclose(restored, points, atol=1e-12)
    assert np.allclose(np.linalg.norm(canonical[1:] - canonical[:-1], axis=1), np.linalg.norm(points[1:] - points[:-1], axis=1))


def test_opencv_to_blender_camera_local_conversion_is_correct() -> None:
    matrix = opencv_to_blender_camera_local_matrix()
    vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    converted = transform_points(vectors, matrix)
    assert np.allclose(converted, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)


def test_support_distance_uses_lower_points_and_patch_overlap() -> None:
    rng = np.random.default_rng(3)
    lower = np.column_stack([rng.uniform(-0.3, 0.3, 100), rng.uniform(-0.3, 0.3, 100), rng.normal(1.01, 0.005, 100)])
    upper = np.column_stack([rng.uniform(-0.3, 0.3, 500), rng.uniform(-0.3, 0.3, 500), rng.uniform(1.2, 2.0, 500)])
    boundary = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    evidence = support_distance_evidence(np.vstack([lower, upper]), np.asarray([0.0, 0.0, 1.0]), -1.0, boundary, distance_threshold=0.08)
    assert evidence["supported"]
    assert evidence["horizontal_overlap_ratio"] > 0.9


def test_support_patch_detection_finds_upward_tabletop() -> None:
    x, y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-0.5, 0.5, 20))
    points = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    pixels = np.column_stack([np.arange(len(points)) % 100, np.arange(len(points)) // 100])
    patch = detect_upward_support_patch(points, normals, pixels, top_height_quantile=0.5)
    assert patch["detected"]
    assert patch["confidence"] > 0.7
    assert patch["plane_canonical"]["normal"][2] > 0.99


def test_wall_attachment_uses_visible_point_to_plane_distance() -> None:
    rng = np.random.default_rng(10)
    points = np.column_stack([rng.normal(0.02, 0.005, 300), rng.uniform(-1, 1, 300), rng.uniform(0, 2, 300)])
    evidence = wall_attachment_evidence(points, np.asarray([1.0, 0.0, 0.0]), 0.0, distance_threshold=0.08)
    assert evidence["attached"]
    assert evidence["confidence"] > 0.7


def test_occluded_object_can_retain_unknown_support() -> None:
    candidate = unknown_support_candidate("chair", "base is occluded", {"visible_part": "backrest"})
    assert candidate["predicate"] == "unknown_support"
    assert candidate["target_id"] is None
    assert candidate["requires_user_review"]


def test_desktop_depth_discontinuity_does_not_override_local_lower_contact() -> None:
    rng = np.random.default_rng(12)
    contact = np.column_stack([rng.uniform(-0.2, 0.2, 120), rng.uniform(-0.2, 0.2, 120), rng.normal(0.51, 0.004, 120)])
    bad_depth_extent = np.column_stack([rng.uniform(-0.2, 0.2, 20), rng.uniform(-0.2, 0.2, 20), np.full(20, 5.0)])
    boundary = [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]]
    evidence = support_distance_evidence(np.vstack([contact, bad_depth_extent]), np.asarray([0.0, 0.0, 1.0]), -0.5, boundary, distance_threshold=0.08)
    assert evidence["supported"]
    assert evidence["median_absolute_plane_distance"] < 0.03


def test_disconnected_coat_rack_points_can_provide_base_evidence() -> None:
    components = [
        np.column_stack([np.full(40, x), np.linspace(-0.1, 0.1, 40), np.linspace(0.0, 1.0, 40)])
        for x in [-0.15, 0.0, 0.15]
    ]
    evidence = support_distance_evidence(np.vstack(components), np.asarray([0.0, 0.0, 1.0]), 0.0, None, lower_quantile=0.1, distance_threshold=0.12)
    assert evidence["supported"]


def test_relationship_validation_prevents_missing_ids() -> None:
    valid = [{"subject_object_id": "a", "predicate": "near", "target_id": "b"}]
    validate_relationship_targets(valid, {"a", "b"}, {"plane_floor"})
    invalid = [{"subject_object_id": "a", "predicate": "near", "target_id": "missing"}]
    with pytest.raises(ValueError, match="absent"):
        validate_relationship_targets(invalid, {"a", "b"}, {"plane_floor"})
