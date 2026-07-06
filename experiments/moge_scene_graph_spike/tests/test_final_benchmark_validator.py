from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from src.final_benchmark_validator import (
    create_corrected_comparison,
    obb_collides,
    thickness_aware_wall_validation,
    transform_delta,
    zero_below_tolerance,
)


ROOT = Path(__file__).resolve().parents[1]


def _snapshot(center, dimensions, rotation, *, properties=None):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=float) @ np.diag(np.asarray(dimensions))
    matrix[:3, 3] = center
    return {
        "matrix_world": matrix.tolist(),
        "custom_properties": properties or {},
    }


def test_sub_micrometre_deltas_are_numerical_zero():
    assert zero_below_tolerance(5.9e-8) == 0.0
    assert zero_below_tolerance(-9.999e-7) == 0.0
    assert zero_below_tolerance(1e-6) == 1e-6


def test_transform_validator_preserves_raw_delta_but_normalizes_reported_delta():
    transform = {
        "center": [0.0, 0.0, 0.0],
        "dimensions": [1.0, 2.0, 3.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    snapshot = _snapshot([5e-8, 0.0, 0.0], transform["dimensions"], np.eye(3))
    result = transform_delta(snapshot, transform)
    assert result["transform_delta_max"] == 5e-8
    assert result["tolerance_normalized_transform_delta"] == 0.0
    assert result["within_tolerance"] is True


def test_wall_validator_does_not_call_sqrt2_half_thickness_a_gap():
    manifest = json.loads((ROOT / "clean_room_v3_1_1/blender_one_batch_manifest.json").read_text())
    rooms = {item["plane_id"]: item for item in manifest["room_proxies"]}
    snapshots = {
        key: _snapshot(value["center"], value["dimensions"], value["rotation_matrix"])
        for key, value in rooms.items()
    }
    result = thickness_aware_wall_validation(
        snapshots["plane_left_wall"], snapshots["plane_right_wall"],
        rooms["plane_left_wall"], rooms["plane_right_wall"],
    )
    assert result["structural_plane_gap"] == 0.0
    assert result["visible_inner_face_gap"] == 0.0
    assert math.isclose(result["expected_finite_thickness_corner_overlap"], math.sqrt(2) * 0.02)
    assert result["expected_overlap_classification"] == "expected_geometry_not_a_wall_gap"


def test_corrected_comparison_decodes_and_preserves_source_panel(tmp_path):
    source = Image.new("RGB", (10, 10), "black")
    for x in range(2, 8):
        for y in range(2, 8):
            source.putpixel((x, y), (220, 80, 30))
    render = Image.new("RGB", (10, 10), (40, 60, 80))
    source_path, render_path, output_path = tmp_path / "source.jpg", tmp_path / "render.png", tmp_path / "comparison.png"
    source.save(source_path)
    render.save(render_path)
    report = create_corrected_comparison(source_path, render_path, output_path)
    with Image.open(output_path) as combined:
        assert combined.size == (20, 10)
        assert combined.getpixel((5, 5))[0] > 150
    assert report["source_panel_black"] is False


def test_obb_contact_is_not_collision_but_penetration_is():
    identity = np.eye(3)
    first = _snapshot([0, 0, 0], [1, 1, 1], identity)
    contact = _snapshot([1, 0, 0], [1, 1, 1], identity)
    overlap = _snapshot([0.9, 0, 0], [1, 1, 1], identity)
    assert obb_collides(first, contact) is False
    assert obb_collides(first, overlap) is True
