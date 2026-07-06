from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from blender_addon.protocol import ProtocolError, SCHEMA_VERSION, error_response, validate_envelope
from blender_addon.sync_core import MappingState, assert_fresh_scene_revision, plan_sync, semantic_revision, transform_revision


ROOT = Path(__file__).resolve().parents[1]
OFFICE = ROOT / "backend" / "scene_package" / "fixtures" / "office-scene-v1.json"


def office_scene() -> dict:
    return json.loads(OFFICE.read_text(encoding="utf-8"))


def test_protocol_handshake_envelope_and_error_response() -> None:
    request = validate_envelope({
        "schema_version": SCHEMA_VERSION,
        "message_id": "message-1",
        "type": "handshake",
        "scene_id": "office",
        "revision": 1,
        "payload": {"client": "website"},
    })
    assert request["type"] == "handshake"
    error = error_response(request, ProtocolError("stale_revision", "old scene"))
    assert error["correlation_id"] == "message-1"
    assert error["payload"]["code"] == "stale_revision"


def test_protocol_rejects_schema_mismatch() -> None:
    with pytest.raises(ProtocolError, match="not supported") as failure:
        validate_envelope({"schema_version": "2.0.0", "type": "handshake", "payload": {}})
    assert failure.value.code == "schema_version_mismatch"


def test_stale_scene_revision_is_rejected() -> None:
    with pytest.raises(ProtocolError) as failure:
        assert_fresh_scene_revision(6, 7)
    assert failure.value.code == "stale_revision"
    assert failure.value.details == {"blender_revision": 7}


def test_mapping_creates_office_scene_once_and_then_is_idempotent() -> None:
    scene = office_scene()
    initial = plan_sync(scene, [])
    assert len(initial["create"]) == len(scene["semantic_objects"]) + len(scene["structural_room_proxies"])
    existing = [MappingState(item["semantic_id"], item["kind"], item["revision"], f"sam_{item['semantic_id']}") for item in initial["create"]]
    repeated = plan_sync(scene, existing)
    assert repeated["create"] == []
    assert repeated["update"] == []
    assert len(repeated["unchanged"]) == len(existing)


def test_only_changed_scene_package_object_is_updated() -> None:
    scene = office_scene()
    first = plan_sync(scene, [])
    existing = [MappingState(item["semantic_id"], item["kind"], item["revision"], f"sam_{item['semantic_id']}") for item in first["create"]]
    changed = deepcopy(scene)
    target = changed["semantic_objects"][4]
    target["version"] += 1
    target["center"][0] += 0.1
    result = plan_sync(changed, existing)
    assert [item["semantic_id"] for item in result["update"]] == [target["object_id"]]
    assert result["create"] == []


def test_duplicate_ids_in_package_and_blender_are_rejected() -> None:
    scene = office_scene()
    duplicate = deepcopy(scene["semantic_objects"][0])
    scene["semantic_objects"].append(duplicate)
    with pytest.raises(ProtocolError) as package_failure:
        plan_sync(scene, [])
    assert package_failure.value.code == "duplicate_semantic_id"

    scene = office_scene()
    object_id = scene["semantic_objects"][0]["object_id"]
    existing = [MappingState(object_id, "semantic_object", "1:a", "A"), MappingState(object_id, "semantic_object", "1:a", "B")]
    with pytest.raises(ProtocolError) as blender_failure:
        plan_sync(scene, existing)
    assert blender_failure.value.details["object_ids"] == [object_id]


def test_unrelated_user_objects_are_outside_the_sync_plan() -> None:
    scene = office_scene()
    result = plan_sync(scene, [MappingState("user-created-unrelated", "unmanaged", "", "My Sculpture")])
    ids = {item["semantic_id"] for item in result["create"] + result["update"] + result["unchanged"]}
    assert "user-created-unrelated" not in ids


def test_mask_and_transform_revisions_are_stable() -> None:
    item = office_scene()["semantic_objects"][0]
    assert semantic_revision(item) == semantic_revision(deepcopy(item))
    changed = deepcopy(item)
    changed["mask_revision"] = "f" * 32
    assert semantic_revision(changed) != semantic_revision(item)
    assert len(transform_revision(item)) == 16
