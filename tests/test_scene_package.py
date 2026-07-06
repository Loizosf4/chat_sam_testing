from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from backend.scene_package.adapter import adapt_scene_package
from backend.scene_package.export_schema import SCHEMA_PATH, export_schema
from backend.scene_package.migrations import migrate_document
from backend.scene_package.models import Scene
from backend.scene_package.serialization import (
    RevisionConflictError,
    dump_scene,
    load_scene,
    validate_optimistic_lock,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "backend" / "scene_package" / "fixtures" / "office-scene-v1.json"
SAM_METADATA = ROOT / "data" / "exports" / "auto_scene_final_24457ea2" / "metadata.json"
UNIFIED_MANIFEST = (
    ROOT
    / "experiments"
    / "moge_scene_graph_spike"
    / "outputs"
    / "office_test"
    / "unified_v3_1_1_clean"
    / "unified_scene_plan.json"
)
SOURCE_IMAGE = ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"


@pytest.fixture
def fixture_document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def scene(fixture_document: dict) -> Scene:
    return Scene.model_validate(fixture_document)


def test_office_fixture_validates_against_models_and_json_schema(
    fixture_document: dict,
) -> None:
    Scene.model_validate(fixture_document)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(fixture_document),
        key=lambda error: list(error.absolute_path),
    )
    assert errors == []


def test_checked_in_schema_matches_typed_models(tmp_path: Path) -> None:
    generated = export_schema(tmp_path / "scene-package-v1.schema.json")
    assert json.loads(generated.read_text(encoding="utf-8")) == json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )


def test_serialization_round_trip_is_lossless(scene: Scene) -> None:
    restored = load_scene(dump_scene(scene))
    assert restored.model_dump(mode="json") == scene.model_dump(mode="json")


def test_migration_from_preview_contract_is_non_mutating(
    fixture_document: dict,
) -> None:
    preview = deepcopy(fixture_document)
    preview["schema_version"] = "0.9.0"
    preview["revision"] = preview.pop("package_revision")
    preview["objects"] = preview.pop("semantic_objects")
    preview["revisions"] = preview.pop("mask_revisions")
    preview["selected_camera_id"] = preview.pop("selected_camera")
    original = deepcopy(preview)

    migrated = migrate_document(preview)

    assert preview == original
    assert migrated["schema_version"] == "1.0.0"
    assert Scene.model_validate(migrated).scene_id == fixture_document["scene_id"]


def test_optimistic_revision_checks(scene: Scene) -> None:
    item = scene.semantic_objects[0]
    validate_optimistic_lock(
        scene,
        expected_package_revision=scene.package_revision,
        expected_object_versions={item.object_id: item.version},
        expected_mask_revisions={item.object_id: item.mask_revision},
    )
    with pytest.raises(RevisionConflictError, match="scene revision conflict"):
        validate_optimistic_lock(scene, expected_package_revision=99)
    with pytest.raises(RevisionConflictError, match="version conflict"):
        validate_optimistic_lock(
            scene,
            expected_package_revision=scene.package_revision,
            expected_object_versions={item.object_id: item.version + 1},
        )
    with pytest.raises(RevisionConflictError, match="mask revision conflict"):
        validate_optimistic_lock(
            scene,
            expected_package_revision=scene.package_revision,
            expected_mask_revisions={item.object_id: "0" * 32},
        )


def test_duplicate_object_ids_are_rejected(fixture_document: dict) -> None:
    document = deepcopy(fixture_document)
    duplicate = deepcopy(document["semantic_objects"][0])
    document["semantic_objects"].append(duplicate)
    with pytest.raises(ValidationError, match="duplicate object IDs"):
        Scene.model_validate(document)


def test_invalid_support_reference_is_rejected(fixture_document: dict) -> None:
    document = deepcopy(fixture_document)
    item = document["semantic_objects"][0]
    item["support_type"] = "object"
    item["support_target"] = "f" * 32
    with pytest.raises(ValidationError, match="object support target"):
        Scene.model_validate(document)


def test_mask_revision_must_belong_to_object(fixture_document: dict) -> None:
    document = deepcopy(fixture_document)
    document["semantic_objects"][0]["mask_revision"] = document["semantic_objects"][1][
        "mask_revision"
    ]
    with pytest.raises(ValidationError, match="own revision"):
        Scene.model_validate(document)


def test_filesystem_paths_are_rejected(fixture_document: dict) -> None:
    document = deepcopy(fixture_document)
    document["metadata"]["debug_path"] = r"C:\private\scene.json"
    with pytest.raises(ValidationError, match="filesystem path is forbidden"):
        Scene.model_validate(document)


def test_office_adapter_is_deterministic_and_preserves_stable_ids() -> None:
    if not (SAM_METADATA.is_file() and UNIFIED_MANIFEST.is_file()):
        pytest.skip("authoritative office source artifacts are not present")
    adapted = adapt_scene_package(
        SAM_METADATA,
        UNIFIED_MANIFEST,
        source_image_path=SOURCE_IMAGE if SOURCE_IMAGE.is_file() else None,
    )
    fixture = load_scene(FIXTURE)
    assert [item.object_id for item in adapted.semantic_objects] == [
        item.object_id for item in fixture.semantic_objects
    ]
    assert [item.mask_revision for item in adapted.semantic_objects] == [
        item.mask_revision for item in fixture.semantic_objects
    ]
    assert adapted.model_dump(mode="json") == fixture.model_dump(mode="json")
    serialized = dump_scene(adapted)
    assert "C:\\" not in serialized
    assert "file://" not in serialized
