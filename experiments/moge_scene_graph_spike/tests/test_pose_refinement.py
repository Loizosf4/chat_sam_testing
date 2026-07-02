from __future__ import annotations

import numpy as np
import pytest

from src.pose_refinement import (
    cluster_normals_spherical,
    conservative_scale_fit,
    erode_mask,
    extract_mask_axis,
    extract_masked_normals,
    matrix_to_quaternion_wxyz,
    normalize_vectors,
    orthonormal_basis_from_forward_up,
    project_camera_points,
    rotation_candidates_from_basis,
)


def test_numeric_normal_extraction_and_boundary_erosion() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    valid = np.ones_like(mask)
    normals = np.zeros((9, 9, 3), dtype=float)
    normals[..., 2] = 2.0
    normals[1, :, :] = np.nan
    extracted, report = extract_masked_normals(normals, mask, valid, np.eye(4), boundary_width=1)
    assert len(extracted) == 25
    assert np.allclose(extracted, [0.0, 0.0, 1.0])
    assert report["erosion"]["fallback"] is False


def test_boundary_erosion_falls_back_for_thin_mask() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 1:6] = True
    eroded, report = erode_mask(mask, boundary_width=2)
    assert np.array_equal(eroded, mask)
    assert report["fallback"] is True


def test_spherical_clusters_do_not_average_incompatible_normals() -> None:
    normals = np.vstack([np.tile([0.0, 0.0, 1.0], (80, 1)), np.tile([0.0, 1.0, 0.0], (20, 1))])
    clusters = cluster_normals_spherical(normals, max_clusters=3, target_p90_degrees=5.0)
    assert len(clusters) == 2
    centers = np.asarray([item["center"] for item in clusters])
    assert np.max(np.abs(centers @ np.asarray([0.0, 0.0, 1.0]))) == pytest.approx(1.0)
    assert {item["classification"] for item in clusters} == {"upward", "vertical"}


def test_camera_to_world_normal_conversion_uses_rotation_only() -> None:
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [100, 200, 300]
    normal_map = np.zeros((3, 3, 3), dtype=float)
    normal_map[..., 0] = 1
    result, _ = extract_masked_normals(normal_map, np.ones((3, 3), bool), np.ones((3, 3), bool), transform, boundary_width=0)
    assert np.allclose(result, [0, 1, 0])


def test_orthonormal_right_handed_basis() -> None:
    basis = orthonormal_basis_from_forward_up(np.asarray([0.2, 1.0, 0.1]), np.asarray([0.0, 0.0, 1.0]))
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    assert matrix_to_quaternion_wxyz(basis)[0] >= 0.0


def test_mask_axis_and_disconnected_confidence() -> None:
    connected = np.zeros((80, 80), dtype=bool)
    connected[30:40, 10:65] = True
    fragmented = connected.copy()
    fragmented[30:40, 36:40] = False
    a = extract_mask_axis(connected)
    b = extract_mask_axis(fragmented)
    assert a["oriented_width_pixels"] > a["oriented_height_pixels"]
    assert abs(a["dominant_axis_angle_degrees"]) < 1.0
    assert b["component_count"] == 2
    assert b["orientation_confidence"] < a["orientation_confidence"]


def test_rotation_candidates_cover_90_and_180_ambiguity() -> None:
    basis = orthonormal_basis_from_forward_up(np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]))
    candidates = rotation_candidates_from_basis(basis)
    assert len(candidates) == 4
    assert all(np.linalg.det(item) == pytest.approx(1.0) for item in candidates)
    assert np.dot(candidates[0][:, 0], candidates[2][:, 0]) == pytest.approx(-1.0)


def test_camera_reprojection() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    intrinsics = np.asarray([[2.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1]])
    projected = project_camera_points(points, intrinsics, (100, 200))
    assert np.allclose(projected[0], [100, 50])
    assert np.allclose(projected[1], [300, 50])


def test_conservative_scale_fit_clamps_thickness() -> None:
    dimensions, report = conservative_scale_fit(
        np.asarray([1.0, 2.0, 1.0]),
        {"width": 100.0, "height": 100.0},
        {"oriented_width_pixels": 200.0, "oriented_height_pixels": 10.0},
        thickness_axis=1,
        depth_samples=np.asarray([1.0, 1.1, 1.2, 50.0, 100.0]),
    )
    assert np.all(dimensions > 0)
    assert dimensions[0] == pytest.approx(1.3)
    assert dimensions[2] == pytest.approx(0.7)
    assert dimensions[1] <= 0.35 * max(dimensions[0], dimensions[2])
    assert report["robust_depth_spread"] is not None


def test_normalize_vectors_rejects_invalid_rows() -> None:
    unit, keep = normalize_vectors(np.asarray([[2, 0, 0], [0, 0, 0], [np.nan, 1, 0]]))
    assert keep.tolist() == [True, False, False]
    assert np.allclose(unit, [[1, 0, 0]])

