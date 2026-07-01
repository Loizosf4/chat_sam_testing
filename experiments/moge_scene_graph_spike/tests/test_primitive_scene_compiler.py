from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.primitive_plan_models import PrimitiveScenePlan
from src.primitive_scene_compiler import (
    CAMERA_IDS,
    EXPECTED_IDS,
    INPUT_PATHS,
    ROOT,
    compile_plan,
    evaluate_quality_gates,
    quaternion_to_matrix_wxyz,
    sha256_file,
    transform_points,
    validate_source_lineage,
)


@pytest.fixture(scope="session")
def compiled(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    output = tmp_path_factory.mktemp("primitive_plan")
    return compile_plan(output), output


def by_label(plan: dict, label: str) -> dict:
    return next(item for item in plan["semantic_primitives"] if item["semantic_label"] == label)


def test_exact_six_object_preservation(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    assert len(plan["semantic_primitives"]) == 6
    assert {item["object_id"] for item in plan["semantic_primitives"]} == set(EXPECTED_IDS)


def test_one_cube_per_semantic_object(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    assert all(item["primitive_type"] == "cube" for item in plan["semantic_primitives"])
    assert len({item["object_id"] for item in plan["semantic_primitives"]}) == 6


def test_deterministic_compilation(compiled: tuple[dict, Path], tmp_path: Path) -> None:
    first, _ = compiled
    second = compile_plan(tmp_path / "second")
    assert first == second


def test_retained_obb_conversion(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    filing = by_label(plan, "left_tall_filing_cabinet")
    assert filing["geometry_strategy"] == "retained_obb"
    assert not np.allclose(filing["rotation_matrix"], np.eye(3))
    assert filing["rotation_quaternion_wxyz"][0] >= 0.0
    assert filing["constraint_strength"] == "review_only"


def test_support_aligned_desk(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    desk = by_label(plan, "desk")
    assert desk["geometry_strategy"] == "support_aligned_proxy"
    assert np.allclose(desk["rotation_matrix"], np.eye(3))
    assert desk["center"][2] - desk["dimensions"][2] / 2 == pytest.approx(0.0, abs=1e-8)
    assert desk["constraint_strength"] == "hard"


def test_desktop_box_depth_correction(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    box = by_label(plan, "desktop_box")
    object_geometry = json.loads(INPUT_PATHS["object_geometry"].read_text(encoding="utf-8"))
    raw = next(item for item in object_geometry["objects"] if item["semantic_label"] == "desktop_box")
    raw_longest = max(raw["geometry"]["raw_moge"]["filtered"]["visible_surface_dimensions"])
    assert max(box["dimensions"]) < 0.75 * raw_longest
    assert any("raw full-Z extent excluded" in warning for warning in box["uncertainty"])
    assert box["constraint_strength"] == "soft"


def test_visible_only_chair_proxy(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    chair = by_label(plan, "desk_chair")
    evidence = json.loads(INPUT_PATHS["scene_evidence"].read_text(encoding="utf-8"))
    transform = np.asarray(evidence["transforms"]["raw_moge_to_canonical_scene_world"])
    archive = ROOT / "outputs" / "office_test" / "geometry" / "diagnostics" / chair["object_id"] / "filtered_visible_geometry.npz"
    with np.load(archive) as data:
        points = transform_points(data["points"], transform)
    assert np.allclose(chair["dimensions"], points.max(axis=0) - points.min(axis=0))
    assert chair["support_target"] is None
    assert any("no hidden seat" in warning for warning in chair["uncertainty"])


def test_disconnected_coat_rack_union(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    coat = by_label(plan, "coat_rack")
    object_geometry = json.loads(INPUT_PATHS["object_geometry"].read_text(encoding="utf-8"))
    source = next(item for item in object_geometry["objects"] if item["semantic_label"] == "coat_rack")
    assert source["connected_component_count"] == 7
    assert sum(item["semantic_label"] == "coat_rack" for item in plan["semantic_primitives"]) == 1
    assert any("seven visible components" in warning for warning in coat["uncertainty"])


def test_wall_aligned_light_proxy(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    light = by_label(plan, "wall_light_fixture")
    evidence = json.loads(INPUT_PATHS["scene_evidence"].read_text(encoding="utf-8"))
    wall = next(item for item in evidence["structural_planes"] if item["plane_id"] == "plane_right_wall")
    normal = np.asarray(wall["canonical_plane_equation"]["normal"])
    normal /= np.linalg.norm(normal)
    assert np.allclose(np.asarray(light["rotation_matrix"])[:, 2], normal)
    assert 0.03 <= light["dimensions"][2] <= 0.12
    assert light["constraint_strength"] == "soft"


def test_constraint_classification_policy(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    counts = {name: sum(item["classification"] == name for item in plan["constraint_classifications"]) for name in ("hard", "soft", "review_only", "ignored_for_transform")}
    assert counts == {"hard": 1, "soft": 3, "review_only": 36, "ignored_for_transform": 58}
    assert not any(item["predicate"] in {"left_of", "right_of", "above", "below", "near", "image_occludes"} and item["influences_transform"] for item in plan["constraint_classifications"])


def test_rejects_uncertain_hard_constraint(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    classifications = json.loads(json.dumps(plan["constraint_classifications"]))
    uncertain = next(item for item in classifications if item["decision"] == "uncertain")
    uncertain["classification"] = "hard"
    uncertain["influences_transform"] = True
    object_geometry = json.loads(INPUT_PATHS["object_geometry"].read_text(encoding="utf-8"))
    gates = evaluate_quality_gates(plan["semantic_primitives"], plan["structural_proxies"], plan["camera_candidates"], classifications, True, object_geometry)
    gate = next(item for item in gates if item["gate"] == "uncertain_never_hard")
    assert gate["passed"] is False


def test_structural_extent_clipping(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    floor = next(item for item in plan["structural_proxies"] if item["plane_id"] == "plane_floor")
    assert floor["extent_policy"] == "bright_near_plane_robust_clip"
    assert any("full contaminated floor-inlier bounds were not used" in warning for warning in floor["warnings"])
    assert floor["dimensions"][2] == pytest.approx(0.05)


def test_perspective_projection(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    metrics = [item for item in plan["projection_validation"]["metrics"] if item["camera_id"] == CAMERA_IDS[0]]
    assert len(metrics) == 6
    assert all(item["visibility_valid"] for item in metrics)
    assert plan["projection_validation"]["perspective_mean_bbox_iou"] > 0.6


def test_orthographic_fitting(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    camera = next(item for item in plan["camera_candidates"] if item["camera_id"] == CAMERA_IDS[1])
    assert camera["orthographic"]["pixels_per_canonical_unit"] > 0
    assert camera["orthographic"]["fit_rmse_pixels"] > 0
    assert camera["orthographic"]["correspondence_count"] > 1000
    assert camera["perspective"] is None


def test_quaternion_matrix_consistency(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    for item in [*plan["semantic_primitives"], *plan["structural_proxies"]]:
        assert np.allclose(quaternion_to_matrix_wxyz(np.asarray(item["rotation_quaternion_wxyz"])), item["rotation_matrix"], atol=1e-7)


def test_positive_dimensions(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    assert all(np.all(np.asarray(item["dimensions"]) > 0) for item in [*plan["semantic_primitives"], *plan["structural_proxies"]])


def test_stable_source_hashes(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    assert validate_source_lineage()
    for reference in plan["source_references"]:
        assert sha256_file(ROOT / reference["path"]) == reference["sha256"]


def test_camera_candidate_preservation(compiled: tuple[dict, Path]) -> None:
    plan, _ = compiled
    assert {item["camera_id"] for item in plan["camera_candidates"]} == set(CAMERA_IDS)
    assert plan["projection_validation"]["other_camera_must_still_be_tested"] is True


def test_plan_schema_and_required_artifacts(compiled: tuple[dict, Path]) -> None:
    plan, output = compiled
    PrimitiveScenePlan.model_validate(plan)
    required = {
        "primitive_scene_plan.json",
        "primitive_scene_plan.md",
        "compilation_report.json",
        "compilation_report.md",
        "constraint_classification.json",
        "camera_candidates.json",
        "perspective_projection_overlay.png",
        "orthographic_projection_overlay.png",
        "structural_projection_comparison.png",
        "camera_comparison.png",
        "primitive_plan_overview.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert len(list((output / "per_object_projections").glob("*.png"))) == 6
