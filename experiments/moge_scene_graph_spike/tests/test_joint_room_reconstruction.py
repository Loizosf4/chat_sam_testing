import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.joint_room_reconstruction import (
    exclude_exterior_floor_points,
    intersect_three_planes,
    recover_finite_floor_quadrilateral,
    refine_perspective_camera,
    run_office_joint_room,
)


@pytest.fixture(scope="session")
def joint_room_output(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("joint_room") / "result"
    run_office_joint_room(output)
    return output


def test_exact_three_plane_intersection():
    point = intersect_three_planes([
        (np.asarray([0.0, 0.0, 1.0]), -2.0),
        (np.asarray([1.0, 0.0, 0.0]), -3.0),
        (np.asarray([0.0, 1.0, 0.0]), 4.0),
    ])
    np.testing.assert_allclose(point, [3.0, -4.0, 2.0], atol=1e-12)


def test_finite_floor_quadrilateral_recovery():
    shape = (160, 160)
    source = Image.new("RGB", shape[::-1], (30, 35, 55))
    draw = ImageDraw.Draw(source)
    left = [(80, 20), (10, 45), (15, 100), (80, 75)]
    right = [(80, 20), (150, 45), (145, 100), (80, 75)]
    floor = [(80, 75), (15, 100), (80, 145), (145, 100)]
    draw.polygon(left, fill=(205, 95, 45)); draw.polygon(right, fill=(215, 105, 50)); draw.polygon(floor, fill=(225, 185, 125))
    plane_masks = {}
    for name, polygon in (("plane_left_wall", left), ("plane_right_wall", right), ("plane_floor", floor)):
        mask = Image.new("1", shape[::-1]); ImageDraw.Draw(mask).polygon(polygon, fill=1); plane_masks[name] = np.asarray(mask, bool)
    depth = np.ones(shape); normals = np.zeros((*shape, 3)); valid = np.ones(shape, bool)
    result = recover_finite_floor_quadrilateral(np.asarray(source), plane_masks, depth, normals, valid)
    recovered = np.asarray(result["floor_polygon"])
    assert recovered.shape == (4, 2)
    np.testing.assert_allclose(recovered[1], [15, 100], atol=8)
    np.testing.assert_allclose(recovered[2], [80, 145], atol=8)
    np.testing.assert_allclose(recovered[3], [145, 100], atol=8)


def test_exterior_floor_plane_points_are_excluded():
    corners = np.asarray([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float)
    points = np.asarray([[.5, .5, 0], [1.9, .9, 0], [3, .5, 0], [-1, 0, 0]], float)
    selected = exclude_exterior_floor_points(points, corners, tolerance=0)
    np.testing.assert_array_equal(selected, [True, True, False, False])


def test_camera_refinement_is_deterministic():
    world = np.asarray([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]], float)
    world_to_camera = np.eye(4); world_to_camera[2, 3] = 5
    intrinsics = np.asarray([[1.8, 0, .5], [0, 1.8, .5], [0, 0, 1]], float)
    from src.pose_refinement import project_world_points
    pixels = project_world_points(world, world_to_camera, intrinsics, (200, 200)) + np.asarray([2.0, -1.0])
    first = refine_perspective_camera(world, pixels, world_to_camera, intrinsics, (200, 200))
    second = refine_perspective_camera(world, pixels, world_to_camera, intrinsics, (200, 200))
    np.testing.assert_allclose(first["world_to_camera"], second["world_to_camera"], atol=1e-12)
    np.testing.assert_allclose(first["intrinsics"], second["intrinsics"], atol=1e-12)
    assert first["rmse_pixels"] == pytest.approx(second["rmse_pixels"], abs=1e-12)


def test_shared_corner_and_floor_wall_contact(joint_room_output):
    room = json.loads((joint_room_output / "room_plan.json").read_text(encoding="utf-8"))
    report = json.loads((joint_room_output / "structural_fit_report.json").read_text(encoding="utf-8"))
    proxies = {item["plane_id"]: item for item in room["room_proxies"]}
    floor_top = proxies["plane_floor"]["center"][2] + proxies["plane_floor"]["dimensions"][2] / 2
    assert floor_top == pytest.approx(0, abs=1e-12)
    for name in ("plane_left_wall", "plane_right_wall"):
        wall = proxies[name]
        assert wall["center"][2] - wall["dimensions"][1] / 2 == pytest.approx(floor_top, abs=1e-12)
    assert report["shared_corner_residual_scene_units"] < 1e-9
    assert report["wall_to_wall_gap_scene_units"] < .005
    assert report["floor_wall_contact_residual_scene_units"] < .005


def test_structural_reprojection_metrics_and_exterior_rejection(joint_room_output):
    report = json.loads((joint_room_output / "structural_fit_report.json").read_text(encoding="utf-8"))
    assert report["passed"]
    assert report["source_floor_polygon_projected_iou"] > .75
    assert report["source_wall_polygon_projected_iou"]["mean"] > .75
    assert report["platform_spill_ratio"] < .20
    assert report["structural_corner_reprojection_rmse_pixels"] < 20
    assert report["floor_inlier_exclusion"]["excluded_exterior"] > 0


def test_no_approved_blender_dependency(joint_room_output):
    room = json.loads((joint_room_output / "room_plan.json").read_text(encoding="utf-8"))
    audit = room["input_audit"]
    assert audit["approved_blender_dependency"] is False
    assert audit["forbidden_matches"] == []
    assert all("blender_execution" not in path.lower() and ".blend" not in path.lower() for path in audit["read_paths"])
