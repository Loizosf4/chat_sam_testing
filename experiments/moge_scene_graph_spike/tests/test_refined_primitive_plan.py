from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from src.refine_primitive_plan import OBJECT_IDS, ROOT, refine_plan


@pytest.fixture(scope="session")
def refinement(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    output = tmp_path_factory.mktemp("pose_refinement_v2")
    return refine_plan(output), output


def by_label(result: dict, label: str) -> dict:
    return next(item for item in result["plan"]["semantic_primitives"] if item["semantic_label"] == label)


def test_refined_schema_and_stable_ids(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    schema = json.loads((ROOT / "schemas" / "refined_primitive_scene_plan.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(result["plan"])
    objects = result["plan"]["semantic_primitives"]
    assert len(objects) == 6
    assert {item["object_id"] for item in objects} == set(OBJECT_IDS.values())
    assert all(item["primitive_type"] == "cube" for item in objects)


def test_source_lineage_and_approved_transforms_preserved(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    base = json.loads((ROOT / "outputs" / "office_test" / "primitive_plan" / "primitive_scene_plan.json").read_text(encoding="utf-8"))
    assert result["plan"]["source_references"] == base["source_references"]
    cabinet = by_label(result, "left_tall_filing_cabinet")
    accepted_cabinet = json.loads((ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "01_filing_cabinet" / "final" / "object_validation.json").read_text(encoding="utf-8"))["final_blender_transform"]
    assert np.allclose(cabinet["center"], accepted_cabinet["location"])
    assert np.allclose(cabinet["dimensions"], accepted_cabinet["dimensions"])
    desk = by_label(result, "desk")
    accepted_desk = json.loads((ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "02_desk" / "approved_candidate_c" / "object_validation.json").read_text(encoding="utf-8"))["approved_transform"]
    assert np.allclose(desk["center"], accepted_desk["center"])
    assert np.allclose(desk["dimensions"], accepted_desk["dimensions"])


def test_chair_has_evidence_rotation_not_identity(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    chair = by_label(result, "desk_chair")
    assert chair["orientation_method"] == "visible_backrest_normal_plane_plus_gravity_and_mask_edges"
    assert not np.allclose(chair["rotation_matrix"], np.eye(3))
    assert chair["support_target"] is None
    assert chair["dimensions"][1] <= 0.10
    assert len(chair["rotation_candidates"]) >= 4


def test_desk_yaw_matches_candidate_c(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    desk = by_label(result, "desk")
    matrix = np.asarray(desk["rotation_matrix"])
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    assert yaw == pytest.approx(-48.327316, abs=1e-6)


def test_wall_light_and_box_support_policy(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    wall = by_label(result, "wall_light_fixture")
    assert wall["orientation_method"].startswith("corrected_right_wall")
    assert wall["dimensions"][2] < 0.05
    box = by_label(result, "desktop_box")
    desk = by_label(result, "desk")
    assert box["center"][2] - box["dimensions"][2] / 2 == pytest.approx(desk["center"][2] + desk["dimensions"][2] / 2)


def test_deterministic_output(tmp_path: Path) -> None:
    first = refine_plan(tmp_path / "a")
    second = refine_plan(tmp_path / "b")
    assert first["plan"] == second["plan"]
    assert first["report"] == second["report"]


def test_required_diagnostics_exist(refinement: tuple[dict, Path]) -> None:
    _, output = refinement
    required = {
        "refined_primitive_scene_plan_v2.json",
        "refined_primitive_scene_plan_v2.md",
        "pose_refinement_report.json",
        "pose_refinement_report.md",
        "before_after_comparison.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    for object_id in OBJECT_IDS.values():
        names = {path.name for path in (output / "per_object" / object_id).iterdir()}
        assert {
            "mask_axis_overlay.png",
            "normal_clusters.json",
            "normal_cluster_visualization.png",
            "rotation_candidates.json",
            "projection_comparison.png",
        } <= names


def test_no_identity_fallback_with_usable_evidence(refinement: tuple[dict, Path]) -> None:
    result, _ = refinement
    for item in result["plan"]["semantic_primitives"]:
        if item["semantic_label"] in {"desk_chair", "wall_light_fixture"}:
            assert not np.allclose(item["rotation_matrix"], np.eye(3))

