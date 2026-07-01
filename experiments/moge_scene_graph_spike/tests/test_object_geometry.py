import numpy as np
import pytest

from src.object_geometry import (
    connected_components,
    estimate_oriented_bounds,
    make_scene_normalization,
    robust_depth_filter,
)


def test_empty_mask_validity_intersection_is_failure_safe() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    object_valid = np.zeros_like(mask)
    labels, components = connected_components(mask)
    keep, details = robust_depth_filter(np.ones((8, 8)), object_valid, labels)
    assert len(components) == 1
    assert not keep.any()
    assert not details["rejected_mask"].any()


def test_tiny_mask_is_preserved() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    labels, components = connected_components(mask)
    keep, _ = robust_depth_filter(np.ones((7, 7)), mask, labels)
    assert len(components) == 1
    assert keep.sum() == 1


def test_disconnected_components_remain_in_one_semantic_mask() -> None:
    semantic_mask = np.zeros((10, 10), dtype=bool)
    semantic_mask[1:3, 1:3] = True
    semantic_mask[7:9, 7:9] = True
    labels, components = connected_components(semantic_mask)
    assert len(components) == 2
    assert sum(component["pixel_count"] for component in components) == semantic_mask.sum()
    assert np.array_equal(labels > 0, semantic_mask)


def test_thin_structure_is_not_removed_by_depth_filter() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 6] = True
    depth = np.ones((12, 12), dtype=np.float32)
    labels, _ = connected_components(mask)
    keep, details = robust_depth_filter(depth, mask, labels)
    assert np.array_equal(keep, mask)
    assert not details["rejected_mask"].any()


def test_isolated_depth_outlier_is_rejected_conservatively() -> None:
    mask = np.ones((9, 9), dtype=bool)
    depth = np.ones((9, 9), dtype=np.float32)
    depth[4, 4] = 10.0
    labels, _ = connected_components(mask)
    keep, details = robust_depth_filter(depth, mask, labels)
    assert details["candidate_mask"][4, 4]
    assert details["rejected_mask"][4, 4]
    assert keep.sum() == mask.sum() - 1


def test_non_finite_scene_geometry_is_rejected() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [np.inf, 0.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        make_scene_normalization(points)


def test_coordinate_axes_and_normalization_are_consistent_and_reversible() -> None:
    points = np.asarray([[-2.0, -1.0, 2.0], [2.0, 3.0, 10.0], [0.5, 1.0, 6.0]], dtype=np.float64)
    normalization = make_scene_normalization(points)
    normalized = normalization.apply(points)
    restored = normalization.invert(normalized)
    assert normalization.scale > 0
    assert np.allclose(restored, points, atol=1e-12)
    assert np.all(np.sign(normalized[1] - normalized[0]) == np.sign(points[1] - points[0]))


def test_oriented_bounds_fail_safely_for_too_few_points() -> None:
    result = estimate_oriented_bounds(np.zeros((20, 3), dtype=np.float32))
    assert result["estimated"] is False
    assert result["confidence"] == 0.0
    assert "center_xyz" not in result


def test_oriented_bounds_do_not_force_partial_thin_geometry() -> None:
    x = np.linspace(-1.0, 1.0, 500)
    points = np.column_stack([x, np.zeros_like(x), np.ones_like(x)])
    result = estimate_oriented_bounds(points, partial_occlusion=True)
    assert result["estimated"] is False
    assert result["unstable_factors"]["thin"] is True
    assert "warning" in result
