import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src import build_full_scene_pose_plan as full_plan


JSON_SCHEMAS = {
    "full_object_inventory.json": "full_object_inventory.schema.json",
    "full_object_geometry.json": "full_object_geometry.schema.json",
    "full_scene_pose_plan.json": "full_scene_pose_plan.schema.json",
    "protected_transform_audit.json": "protected_transform_audit.schema.json",
    "classification_report.json": "classification_report.schema.json",
    "blender_batch_manifest.json": "blender_batch_manifest.schema.json",
    "quality_report.json": "full_scene_quality_report.schema.json",
}
GLOBAL_VISUALS = {
    "numbered_full_object_overlay.png",
    "all_mask_axes_overlay.png",
    "all_projected_primitives_overlay.png",
    "batch_ready_overlay.png",
    "review_required_overlay.png",
    "normal_orientation_summary.png",
    "full_scene_plan_overview.png",
}
DIAGNOSTICS = {
    "source_mask_overlay.png",
    "masked_normal_visualization.png",
    "dominant_mask_axes.png",
    "projected_primitive_overlay.png",
    "geometry_pose_summary.json",
}
CHAIR_ID = "9171a7b4d4e142ca936f2564200d0bdb"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("full_scene_pose_plan")
    first, second = root / "first", root / "second"
    full_plan.build(first)
    full_plan.build(second)
    return first, second


def test_complete_fixture_is_auditable_and_exactly_scoped(generated):
    first, _ = generated
    manifest = load(full_plan.FIXTURE / "manifest.json")
    inventory = load(first / "full_object_inventory.json")
    assert manifest["mask_count"] == inventory["object_count"] == 20
    assert manifest["approved_object_count"] == 6
    assert manifest["new_object_count"] == 14
    assert manifest["excluded_mask_count"] == 0
    ids = [item["object_id"] for item in inventory["objects"]]
    assert len(ids) == len(set(ids)) == 20
    for item in manifest["objects"]:
        source = full_plan.REPO_ROOT / item["source_export_path"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]


def test_outputs_are_deterministic(generated):
    first, second = generated
    for filename in JSON_SCHEMAS:
        assert load(first / filename) == load(second / filename)


def test_all_machine_readable_outputs_validate(generated):
    first, _ = generated
    manifest_schema = load(full_plan.ROOT / "schemas" / "office_test_full_manifest.schema.json")
    jsonschema.validate(load(full_plan.FIXTURE / "manifest.json"), manifest_schema)
    for filename, schema_name in JSON_SCHEMAS.items():
        jsonschema.validate(load(first / filename), load(full_plan.ROOT / "schemas" / schema_name))


def test_every_object_has_one_defensible_classification(generated):
    first, _ = generated
    report = load(first / "classification_report.json")
    groups = report["classifications"]
    flattened = [item["object_id"] for items in groups.values() for item in items]
    assert len(flattened) == len(set(flattened)) == 20
    assert len(groups["existing_approved_object"]) == 6
    assert sum(len(groups[name]) for name in ("batch_ready", "blender_review_required", "insufficient_geometry", "excluded_structure")) == 14
    assert report["counts"]["insufficient_geometry"] == 0
    assert report["counts"]["excluded_structure"] == 0


def test_transforms_are_finite_positive_orthonormal_and_right_handed(generated):
    first, _ = generated
    plan = load(first / "full_scene_pose_plan.json")
    for item in plan["objects"]:
        center = np.asarray(item["center"], dtype=float)
        dimensions = np.asarray(item["dimensions"], dtype=float)
        rotation = np.asarray(item["rotation_matrix"], dtype=float)
        quaternion = np.asarray(item["rotation_quaternion_wxyz"], dtype=float)
        assert np.isfinite(center).all()
        assert np.isfinite(dimensions).all() and (dimensions > 0).all()
        assert np.isfinite(rotation).all() and np.isfinite(quaternion).all()
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-6)


def test_protected_transforms_and_approved_chair_are_exact(generated):
    first, _ = generated
    audit = load(first / "protected_transform_audit.json")
    assert audit["passed"]
    assert len(audit["protected_objects"]) == 6
    assert max(item["maximum_absolute_delta"] for item in audit["protected_objects"]) <= 2e-6
    chair = next(item for item in audit["protected_objects"] if item["object_id"] == CHAIR_ID)
    assert chair["planned"]["center"] == [0.4257657802, 1.0472733016, 0.4270060736]
    assert chair["planned"]["dimensions"] == [0.3388930019, 0.1, 0.2414989213]
    assert chair["planned"]["quaternion_wxyz"] == [0.3491493496, 0.0, 0.0, 0.9370670903]


def test_required_visuals_and_per_object_diagnostics_exist(generated):
    first, _ = generated
    assert GLOBAL_VISUALS <= {item.name for item in first.iterdir()}
    inventory = load(first / "full_object_inventory.json")
    new_ids = [item["object_id"] for item in inventory["objects"] if item["fixture_classification"] == "new_object"]
    assert len(new_ids) == 14
    for object_id in new_ids:
        diagnostic = first / "per_object" / object_id
        assert DIAGNOSTICS <= {item.name for item in diagnostic.iterdir()}


def test_quality_gates_and_blender_manifest_exclude_frozen_objects(generated):
    first, _ = generated
    quality = load(first / "quality_report.json")
    assert quality["passed"] and all(quality["quality_gates"].values())
    batch = load(first / "blender_batch_manifest.json")
    frozen = {item["object_id"] for item in batch["objects_already_present_and_frozen"]}
    scheduled = {item["object_id"] for item in batch["batch_ready_objects"]}
    scheduled |= {item["object_id"] for item in batch["individual_review_objects"]}
    assert len(frozen) == 6
    assert frozen.isdisjoint(scheduled)
    assert len(scheduled) == 14
