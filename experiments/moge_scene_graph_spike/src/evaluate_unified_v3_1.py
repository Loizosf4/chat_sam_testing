"""Read-only comparison of unified V3.1 against the preceding clean benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .joint_room_reconstruction import polygon_iou
from .pose_refinement import cuboid_corners, project_world_points
from .unified_v3_scene_compiler import ROOT, _hull


PREVIOUS = ROOT / "outputs" / "office_test" / "unified_v3_clean"
CURRENT = ROOT / "outputs" / "office_test" / "unified_v3_1_clean"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _camera(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    candidate = next(item for item in plan["camera_candidates"] if item["type"] == "perspective")
    world_to_camera = np.asarray(candidate.get("canonical_to_camera_transform", np.linalg.inv(np.asarray(plan["coordinate_system"]["raw_moge_to_canonical"]))), float)
    return world_to_camera, np.asarray(candidate["normalized_intrinsics"], float), (447, 447)


def _proxy_polygon(proxy: dict[str, Any], camera: tuple[np.ndarray, np.ndarray, tuple[int, int]]) -> np.ndarray:
    corners = cuboid_corners(np.asarray(proxy["center"]), np.asarray(proxy["dimensions"]), np.asarray(proxy["rotation_matrix"]))
    return _hull(project_world_points(corners, *camera))


def compare_clean_outputs(current: Path = CURRENT, previous: Path = PREVIOUS) -> dict[str, Any]:
    """Compare two clean plans without opening any approved-reference artifact."""
    old = _load(previous / "unified_scene_plan.json"); new = _load(current / "unified_scene_plan.json")
    old_by = {item["object_id"]: item for item in old["semantic_objects"]}; new_by = {item["object_id"]: item for item in new["semantic_objects"]}
    objects = []
    for object_id, candidate in new_by.items():
        baseline = old_by[object_id]
        old_iou, new_iou = baseline["validation_metrics"]["bbox_iou"], candidate["validation_metrics"]["bbox_iou"]
        old_centroid, new_centroid = baseline["validation_metrics"]["centroid_error_pixels"], candidate["validation_metrics"]["centroid_error_pixels"]
        score = (new_iou - old_iou) + .02 * (old_centroid - new_centroid)
        classification = "improved" if score > .03 else "regressed" if score < -.03 else "stable"
        objects.append({
            "object_id": object_id, "semantic_label": candidate["semantic_label"], "classification": classification,
            "before": {"bbox_iou": old_iou, "centroid_error_pixels": old_centroid, "support_type": baseline.get("support_type"), "support_target": baseline.get("support_target"), "dimensions": baseline["transform"]["dimensions"]},
            "after": {"bbox_iou": new_iou, "centroid_error_pixels": new_centroid, "support_type": candidate["support_type"], "support_target": candidate["support_target"], "dimensions": candidate["transform"]["dimensions"], "placement_classification": candidate["placement_classification"]},
        })

    room = _load(current / "room_plan.json"); report = room["structural_fit_report"]
    source = room["structural_landmarks"]["image"]
    old_room = {item["plane_id"]: item for item in old["room_proxies"]}; new_room = {item["plane_id"]: item for item in new["room_proxies"]}
    old_camera = _camera(old)
    old_floor_iou, old_floor_spill = polygon_iou(np.asarray(source["floor_polygon"]), _proxy_polygon(old_room["plane_floor"], old_camera), old_camera[2])
    old_left_iou, _ = polygon_iou(np.asarray(source["left_wall_polygon"]), _proxy_polygon(old_room["plane_left_wall"], old_camera), old_camera[2])
    old_right_iou, _ = polygon_iou(np.asarray(source["right_wall_polygon"]), _proxy_polygon(old_room["plane_right_wall"], old_camera), old_camera[2])
    result = {
        "schema_version": "1.0", "comparison_mode": "clean_before_approved_audit",
        "read_paths": [str((previous / "unified_scene_plan.json").resolve()), str((current / "unified_scene_plan.json").resolve()), str((current / "room_plan.json").resolve())],
        "approved_reference_reads": [],
        "objects": objects,
        "object_counts": {name: sum(item["classification"] == name for item in objects) for name in ("improved", "stable", "regressed")},
        "room": {
            "floor": {"before_dimensions": old_room["plane_floor"]["dimensions"], "after_dimensions": new_room["plane_floor"]["dimensions"], "before_polygon_iou": old_floor_iou, "after_polygon_iou": report["source_floor_polygon_projected_iou"], "before_spill_ratio": old_floor_spill, "after_spill_ratio": report["platform_spill_ratio"]},
            "left_wall": {"before_dimensions": old_room["plane_left_wall"]["dimensions"], "after_dimensions": new_room["plane_left_wall"]["dimensions"], "before_polygon_iou": old_left_iou, "after_polygon_iou": report["source_wall_polygon_projected_iou"]["left"]},
            "right_wall": {"before_dimensions": old_room["plane_right_wall"]["dimensions"], "after_dimensions": new_room["plane_right_wall"]["dimensions"], "before_polygon_iou": old_right_iou, "after_polygon_iou": report["source_wall_polygon_projected_iou"]["right"]},
            "shared_corner_residual": report["shared_corner_residual_scene_units"], "floor_wall_contact_residual": report["floor_wall_contact_residual_scene_units"],
        },
    }
    (current / "clean_benchmark_comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    by_label = {item["semantic_label"]: item for item in objects}
    lines = ["# Unified V3.1 clean comparison", "", "Approved references were not read.", ""]
    for label in ("desktop_box", "desk_chair", "desk"):
        item = by_label[label]; lines.append(f"- {label}: {item['classification']}; bbox IoU {item['before']['bbox_iou']:.3f} -> {item['after']['bbox_iou']:.3f}; centroid {item['before']['centroid_error_pixels']:.2f} -> {item['after']['centroid_error_pixels']:.2f} px")
    lines.extend(["", f"- Floor IoU: {old_floor_iou:.3f} -> {report['source_floor_polygon_projected_iou']:.3f}", f"- Floor spill: {old_floor_spill:.3f} -> {report['platform_spill_ratio']:.3f}", f"- Object totals: {result['object_counts']}"])
    (current / "clean_benchmark_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(compare_clean_outputs(), indent=2))
