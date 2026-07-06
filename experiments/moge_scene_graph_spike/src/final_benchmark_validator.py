"""Final, read-only validation for the Unified V3.1.1 Blender benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

NUMERICAL_TOLERANCE = 1e-6


def zero_below_tolerance(value: float, tolerance: float = NUMERICAL_TOLERANCE) -> float:
    """Normalize a signed numerical residual to zero inside the accepted band."""
    return 0.0 if abs(float(value)) < tolerance else float(value)


def max_abs_delta(actual: Any, expected: Any) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))


def _unit(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decompose_cube(snapshot: dict[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(snapshot["matrix_world"], dtype=float)
    columns = matrix[:3, :3]
    dimensions = np.linalg.norm(columns, axis=0)
    half_dimensions = dimensions / 2.0
    axes = columns / dimensions
    return {
        "center": matrix[:3, 3],
        "dimensions": dimensions,
        "half_dimensions": half_dimensions,
        "axes": axes,
        "matrix_world": matrix,
    }


def _quaternion_matrix_wxyz(quaternion: Any) -> np.ndarray:
    w, x, y, z = _unit(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_delta(snapshot: dict[str, Any], manifest_transform: dict[str, Any]) -> dict[str, float]:
    cube = _decompose_cube(snapshot)
    expected_rotation = _quaternion_matrix_wxyz(manifest_transform["quaternion_wxyz"])
    raw = {
        "center_delta_max_abs": max_abs_delta(cube["center"], manifest_transform["center"]),
        "dimensions_delta_max_abs": max_abs_delta(cube["dimensions"], manifest_transform["dimensions"]),
        "rotation_matrix_delta_max_abs": max_abs_delta(cube["axes"], expected_rotation),
    }
    raw["transform_delta_max"] = max(raw.values())
    return {
        **raw,
        "tolerance_normalized_transform_delta": zero_below_tolerance(raw["transform_delta_max"]),
        "within_tolerance": raw["transform_delta_max"] < NUMERICAL_TOLERANCE,
    }


def _line_intersection_2d(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float) -> np.ndarray:
    return np.linalg.solve(np.vstack((n1[:2], n2[:2])), np.asarray([d1, d2], dtype=float))


def thickness_aware_wall_validation(
    left: dict[str, Any],
    right: dict[str, Any],
    manifest_left: dict[str, Any],
    manifest_right: dict[str, Any],
) -> dict[str, Any]:
    """Compare wall supporting planes and their visible inner faces, not mesh corners."""
    walls = []
    for snapshot, manifest in ((left, manifest_left), (right, manifest_right)):
        cube = _decompose_cube(snapshot)
        expected_normal = _unit(manifest["plane_equation"]["normal"])
        normal_axis = cube["axes"][:, 2]
        normal = normal_axis if np.dot(normal_axis, expected_normal) >= 0 else -normal_axis
        inner_face_center = cube["center"] - normal * cube["half_dimensions"][2]
        tangent = cube["axes"][:, 0]
        endpoints = [
            inner_face_center - tangent * cube["half_dimensions"][0],
            inner_face_center + tangent * cube["half_dimensions"][0],
        ]
        walls.append(
            {
                "normal": normal,
                "plane_distance": float(np.dot(normal, inner_face_center)),
                "inner_face_center": inner_face_center,
                "inner_face_endpoints": endpoints,
                "thickness": float(cube["dimensions"][2]),
                "half_thickness": float(cube["half_dimensions"][2]),
            }
        )

    manifest_normals = [_unit(item["plane_equation"]["normal"]) for item in (manifest_left, manifest_right)]
    manifest_distances = [-float(item["plane_equation"]["offset"]) for item in (manifest_left, manifest_right)]
    intended_corner_xy = _line_intersection_2d(
        manifest_normals[0], manifest_distances[0], manifest_normals[1], manifest_distances[1]
    )
    intended_corner = np.asarray([*intended_corner_xy, walls[0]["inner_face_center"][2]])

    structural_raw = max(
        abs(float(np.dot(wall["normal"], intended_corner) - wall["plane_distance"])) for wall in walls
    )
    inner_face_raw = min(
        float(np.linalg.norm(a - b))
        for a in walls[0]["inner_face_endpoints"]
        for b in walls[1]["inner_face_endpoints"]
    )
    expected_overlap = math.hypot(walls[0]["half_thickness"], walls[1]["half_thickness"])
    return {
        "method": "supporting wall planes plus visible inner-face endpoints",
        "structural_plane_gap_raw": structural_raw,
        "structural_plane_gap": zero_below_tolerance(structural_raw),
        "visible_inner_face_gap_raw": inner_face_raw,
        "visible_inner_face_gap": zero_below_tolerance(inner_face_raw),
        "finite_cube_thickness": {
            "left_wall": walls[0]["thickness"],
            "right_wall": walls[1]["thickness"],
        },
        "expected_finite_thickness_corner_overlap": expected_overlap,
        "expected_overlap_classification": "expected_geometry_not_a_wall_gap",
        "legacy_mesh_corner_value_excluded_from_gap": expected_overlap,
    }


def floor_wall_contact(floor: dict[str, Any], walls: list[dict[str, Any]]) -> dict[str, Any]:
    floor_cube = _decompose_cube(floor)
    floor_normal = floor_cube["axes"][:, 2]
    if floor_normal[2] < 0:
        floor_normal = -floor_normal
    floor_face = floor_cube["center"] + floor_normal * floor_cube["half_dimensions"][2]
    floor_height = float(np.dot(floor_normal, floor_face))
    raw = {}
    for wall in walls:
        cube = _decompose_cube(wall)
        bottom = min(
            float(np.dot(floor_normal, cube["center"] + cube["axes"][:, 1] * sign * cube["half_dimensions"][1]))
            for sign in (-1, 1)
        )
        raw[wall["custom_properties"]["plane_id"]] = bottom - floor_height
    normalized = {key: zero_below_tolerance(value) for key, value in raw.items()}
    return {"raw_residuals": raw, "residuals": normalized, "maximum_contact_gap": max(map(abs, normalized.values()))}


def vertical_support_gap(lower: dict[str, Any], upper: dict[str, Any]) -> dict[str, float]:
    def extrema(snapshot: dict[str, Any]) -> tuple[float, float]:
        cube = _decompose_cube(snapshot)
        radius = sum(abs(cube["axes"][2, index]) * cube["half_dimensions"][index] for index in range(3))
        return float(cube["center"][2] - radius), float(cube["center"][2] + radius)

    _, lower_top = extrema(lower)
    upper_bottom, _ = extrema(upper)
    raw = upper_bottom - lower_top
    return {"raw": raw, "gap": zero_below_tolerance(raw)}


def obb_collides(a_snapshot: dict[str, Any], b_snapshot: dict[str, Any], tolerance: float = NUMERICAL_TOLERANCE) -> bool:
    """15-axis separating-axis test; touching within tolerance is not a collision."""
    a, b = _decompose_cube(a_snapshot), _decompose_cube(b_snapshot)
    rotation = a["axes"].T @ b["axes"]
    translation = a["axes"].T @ (b["center"] - a["center"])
    absolute = np.abs(rotation) + 1e-12
    ah, bh = a["half_dimensions"], b["half_dimensions"]

    for i in range(3):
        if abs(translation[i]) >= ah[i] + float(np.dot(bh, absolute[i, :])) - tolerance:
            return False
    for j in range(3):
        projected = abs(float(np.dot(translation, rotation[:, j])))
        if projected >= float(np.dot(ah, absolute[:, j])) + bh[j] - tolerance:
            return False
    for i, j in product(range(3), repeat=2):
        if 1.0 - rotation[i, j] * rotation[i, j] < 1e-12:
            continue
        ra = ah[(i + 1) % 3] * absolute[(i + 2) % 3, j] + ah[(i + 2) % 3] * absolute[(i + 1) % 3, j]
        rb = bh[(j + 1) % 3] * absolute[i, (j + 2) % 3] + bh[(j + 2) % 3] * absolute[i, (j + 1) % 3]
        projected = abs(translation[(i + 2) % 3] * rotation[(i + 1) % 3, j] - translation[(i + 1) % 3] * rotation[(i + 2) % 3, j])
        if projected >= ra + rb - tolerance:
            return False
    return True


def create_corrected_comparison(source_path: Path, orthographic_path: Path, output_path: Path) -> dict[str, Any]:
    """Compose with Pillow so the source pixels are explicitly decoded and preserved."""
    with Image.open(source_path) as source_file, Image.open(orthographic_path) as render_file:
        source = source_file.convert("RGB")
        render = render_file.convert("RGB")
        source = source.resize((render.height, render.height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (source.width + render.width, render.height))
        canvas.paste(source, (0, 0))
        canvas.paste(render, (source.width, 0))
        canvas.save(output_path)
        source_array = np.asarray(source)
    non_black_ratio = float(np.mean(np.max(source_array, axis=2) > 8))
    return {
        "source_panel_non_black_pixel_ratio": non_black_ratio,
        "source_panel_black": non_black_ratio < 0.01,
        "generation": "Pillow RGB decode, resize, and paste",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _object_map(snapshot: dict[str, Any], role: str, key: str) -> dict[str, dict[str, Any]]:
    return {
        str(item["custom_properties"][key]): item
        for item in snapshot["objects"]
        if item["custom_properties"].get("object_role") == role
    }


def validate(project: Path, checkpoint_snapshot: Path) -> dict[str, Any]:
    output = project / "outputs/office_test/blender_execution/unified_v3_1_1_clean_one_batch_01"
    checkpoint = output / "unified_v3_1_1_clean_one_batch.blend"
    manifest_path = project / "clean_room_v3_1_1/blender_one_batch_manifest.json"
    source_path = project / "inputs/office_test/office_scene.jpg"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = json.loads(checkpoint_snapshot.read_text(encoding="utf-8"))
    semantic = _object_map(snapshot, "semantic_primitive", "semantic_id")
    rooms = _object_map(snapshot, "room_proxy", "plane_id")
    manifest_semantic = {item["object_id"]: item for item in manifest["semantic_primitives"]}
    manifest_rooms = {item["plane_id"]: item for item in manifest["room_proxies"]}

    transform_records = []
    for object_id, expected in manifest_semantic.items():
        delta = transform_delta(semantic[object_id], expected["transform"])
        transform_records.append({"semantic_id": object_id, "semantic_label": expected["semantic_label"], **delta})
    room_transform_records = []
    for plane_id, expected in manifest_rooms.items():
        delta = transform_delta(rooms[plane_id], expected)
        room_transform_records.append({"plane_id": plane_id, **delta})

    # Cameras have no object_role property, so select them directly by camera_id.
    cameras = {
        str(item["custom_properties"]["camera_id"]): item
        for item in snapshot["objects"]
        if "camera_id" in item["custom_properties"]
    }
    camera_transform_records = []
    expected_camera_matrix = manifest["provisional_camera"]["matrix_world"]
    for camera_id in ("joint_perspective", "joint_orthographic"):
        raw_delta = max_abs_delta(cameras[camera_id]["matrix_world"], expected_camera_matrix)
        camera_transform_records.append(
            {
                "camera_id": camera_id,
                "matrix_world_delta_max_abs": raw_delta,
                "tolerance_normalized_transform_delta": zero_below_tolerance(raw_delta),
                "within_tolerance": raw_delta < NUMERICAL_TOLERANCE,
            }
        )
    maximum_raw_delta = max(
        [record["transform_delta_max"] for record in transform_records]
        + [record["transform_delta_max"] for record in room_transform_records]
        + [record["matrix_world_delta_max_abs"] for record in camera_transform_records]
    )

    walls = thickness_aware_wall_validation(
        rooms["plane_left_wall"], rooms["plane_right_wall"],
        manifest_rooms["plane_left_wall"], manifest_rooms["plane_right_wall"],
    )
    floor_contact = floor_wall_contact(
        rooms["plane_floor"], [rooms["plane_left_wall"], rooms["plane_right_wall"]]
    )
    desk_id = next(key for key, value in manifest_semantic.items() if value["semantic_label"] == "desk")
    box_id = next(key for key, value in manifest_semantic.items() if value["semantic_label"] == "desktop_box")
    desktop_gap = vertical_support_gap(semantic[desk_id], semantic[box_id])

    collisions = []
    for (a_id, a), (b_id, b) in combinations(semantic.items(), 2):
        if obb_collides(a, b):
            collisions.append({"object_a": a_id, "object_b": b_id})

    clean_room = json.loads((output / "clean_room_compliance_report.json").read_text(encoding="utf-8"))
    comparison = create_corrected_comparison(
        source_path, output / "orthographic.png", output / "corrected_orthographic_comparison.png"
    )
    checkpoint_hash = _sha256(checkpoint)
    gates = {
        "20_unique_semantic_objects": len(semantic) == 20 and len(set(semantic)) == 20,
        "clean_room_compliance_passed": clean_room.get("result") == "PASS",
        "desktop_box_support_gap_zero": desktop_gap["gap"] == 0.0,
        "structural_wall_gap_zero": walls["structural_plane_gap"] == 0.0,
        "floor_wall_contact_zero": floor_contact["maximum_contact_gap"] == 0.0,
        "transform_differences_within_1e_6": maximum_raw_delta < NUMERICAL_TOLERANCE,
        "no_collisions": not collisions,
        "orthographic_source_panel_present": not comparison["source_panel_black"],
    }
    accepted = all(gates.values())
    validation = {
        "schema_version": "1.0",
        "benchmark": "Unified V3.1.1 clean",
        "status": "accepted" if accepted else "rejected",
        "validation_mode": "read_only_existing_blender_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "semantic_object_count": len(semantic),
        "unique_semantic_id_count": len(set(semantic)),
        "transform_validation": {
            "maximum_raw_transform_delta": maximum_raw_delta,
            "maximum_tolerance_normalized_transform_delta": zero_below_tolerance(maximum_raw_delta),
            "all_within_tolerance": maximum_raw_delta < NUMERICAL_TOLERANCE,
            "objects": transform_records,
            "room_proxies": room_transform_records,
            "cameras": camera_transform_records,
        },
        "wall_validation": walls,
        "floor_wall_contact": floor_contact,
        "desktop_box_support": desktop_gap,
        "collision_validation": {"method": "15-axis OBB SAT; tolerance contact is non-collision", "collision_count": len(collisions), "collisions": collisions},
        "clean_room_compliance": clean_room,
        "orthographic_comparison": comparison,
        "acceptance_gates": gates,
    }
    acceptance = {
        "benchmark": "Unified V3.1.1 clean",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "accepted": accepted,
        "frozen_checkpoint_sha256": checkpoint_hash,
        "criteria": gates,
    }
    tolerance_report = {
        "tolerance": NUMERICAL_TOLERANCE,
        "comparison_rule": "absolute deltas strictly below 1e-6 are reported as numerical zero",
        "maximum_raw_transform_delta": maximum_raw_delta,
        "maximum_reported_transform_delta": zero_below_tolerance(maximum_raw_delta),
        "raw_structural_plane_gap": walls["structural_plane_gap_raw"],
        "reported_structural_plane_gap": walls["structural_plane_gap"],
        "raw_visible_inner_face_gap": walls["visible_inner_face_gap_raw"],
        "reported_visible_inner_face_gap": walls["visible_inner_face_gap"],
        "raw_desktop_box_support_gap": desktop_gap["raw"],
        "reported_desktop_box_support_gap": desktop_gap["gap"],
        "raw_floor_wall_contact_residuals": floor_contact["raw_residuals"],
        "reported_floor_wall_contact_residuals": floor_contact["residuals"],
        "per_object_transform_deltas": transform_records,
    }
    _write_json(output / "final_benchmark_validation.json", validation)
    _write_json(output / "benchmark_acceptance.json", acceptance)
    _write_json(output / "numerical_tolerance_report.json", tolerance_report)

    (output / "final_benchmark_validation.md").write_text(
        f"""# Unified V3.1.1 final benchmark validation

**Status: {'ACCEPTED' if accepted else 'REJECTED'}**

- Unique semantic objects: {len(set(semantic))} / 20
- Clean-room compliance: {clean_room.get('result')}
- Desktop-box support gap: {desktop_gap['gap']:.9g} (raw {desktop_gap['raw']:.9g})
- Structural-plane gap: {walls['structural_plane_gap']:.9g} (raw {walls['structural_plane_gap_raw']:.9g})
- Visible inner-face gap: {walls['visible_inner_face_gap']:.9g} (raw {walls['visible_inner_face_gap_raw']:.9g})
- Expected finite-thickness corner overlap: {walls['expected_finite_thickness_corner_overlap']:.9g} (diagnostic, not a gap)
- Floor/wall contact: {floor_contact['maximum_contact_gap']:.9g}
- Maximum transform delta: {zero_below_tolerance(maximum_raw_delta):.9g} (raw {maximum_raw_delta:.9g}; tolerance 1e-6)
- Collisions: {len(collisions)}
- Checkpoint SHA-256: `{checkpoint_hash}`

Wall validation uses supporting planes and visible inner faces. The legacy `sqrt(2) * half-thickness` mesh-corner measurement is retained only as expected finite-thickness overlap and is excluded from the structural wall gap.
""",
        encoding="utf-8",
    )
    (output / "benchmark_acceptance.md").write_text(
        "# Benchmark acceptance\n\n"
        f"**{'ACCEPTED' if accepted else 'REJECTED'}**\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}" for name, passed in gates.items())
        + f"\n\nFrozen checkpoint SHA-256: `{checkpoint_hash}`\n",
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint-snapshot", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.project.resolve(), args.checkpoint_snapshot.resolve())
    print(result["status"].upper())


if __name__ == "__main__":
    main()
