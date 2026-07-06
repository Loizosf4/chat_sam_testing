"""Headless Blender acceptance smoke test for the V3.1.1 office scene."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import bpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blender_addon import commands, event_bus, object_utils, selection_listener
from blender_addon.protocol import ProtocolError


scene_path = ROOT / "backend" / "scene_package" / "fixtures" / "office-scene-v1.json"
scene = json.loads(scene_path.read_text(encoding="utf-8"))
output_dir = ROOT / "docs" / "demonstration"
output_dir.mkdir(parents=True, exist_ok=True)

# This object represents arbitrary user content and must survive all synchronization.
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(8.0, 8.0, 8.0))
unrelated = bpy.context.object
unrelated.name = "User_Unrelated_Sculpture"

request = {
    "schema_version": "1.0.0",
    "message_id": "blender-smoke-initial",
    "type": "scene_sync",
    "scene_id": scene["scene_id"],
    "revision": scene["package_revision"],
    "payload": {
        "scene_package": {
            key: scene[key]
            for key in ("schema_version", "package_revision", "scene_id", "structural_room_proxies", "semantic_objects")
        },
        "stale_object_ids": [],
    },
}
first = commands.dispatch(request)
request["message_id"] = "blender-smoke-repeat"
second = commands.dispatch(request)

changed_scene = deepcopy(scene)
changed_scene["package_revision"] = 2
changed_scene["semantic_objects"][0]["version"] += 1
changed_scene["semantic_objects"][0]["center"][0] += 0.1
changed_request = deepcopy(request)
changed_request["message_id"] = "blender-smoke-one-change"
changed_request["revision"] = 2
changed_request["payload"]["scene_package"] = {
    key: changed_scene[key]
    for key in ("schema_version", "package_revision", "scene_id", "structural_room_proxies", "semantic_objects")
}
changed = commands.dispatch(changed_request)
assert changed["payload"]["updated"] == [changed_scene["semantic_objects"][0]["object_id"]]

try:
    commands.dispatch(request)
except ProtocolError as exc:
    stale_error_code = exc.code
else:
    raise AssertionError("older scene revision was accepted")

target_id = scene["semantic_objects"][0]["object_id"]
selected = commands.dispatch({
    "schema_version": "1.0.0",
    "message_id": "blender-smoke-select",
    "type": "select_highlight",
    "scene_id": scene["scene_id"],
    "object_id": target_id,
    "revision": changed_scene["package_revision"],
    "payload": {"mask_revision": scene["semantic_objects"][0]["mask_revision"]},
})

managed = object_utils.managed_objects(scene["scene_id"])
assert len(managed) == 23
assert bpy.data.objects.get(unrelated.name) is unrelated
assert unrelated.get(object_utils.ID_PROPERTY) is None
assert second["payload"]["created"] == []
assert second["payload"]["updated"] == []
assert len(second["payload"]["unchanged"]) == 23
assert selected["payload"]["selected"]["object_id"] == target_id
assert bpy.context.view_layer.objects.active.get(object_utils.ID_PROPERTY) == target_id

event_bus.clear()
selection_listener.refresh_baseline()
target = bpy.context.view_layer.objects.active
selection_listener._on_depsgraph_update(bpy.context.scene, SimpleNamespace(updates=[]))
target.location.x += 0.05
selection_listener._on_depsgraph_update(bpy.context.scene, SimpleNamespace(updates=[SimpleNamespace(id=target)]))
_latest, emitted_events = event_bus.since(0)
event_types = [event["type"] for event in emitted_events]
assert "selection_changed" in event_types
assert "transform_update" in event_types

bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "accepted-office-v3.1.1-sync.blend"))
report = {
    "status": "passed",
    "blender_version": bpy.app.version_string,
    "scene_id": scene["scene_id"],
    "managed_object_count": len(managed),
    "semantic_object_count": sum(obj.get(object_utils.KIND_PROPERTY) == "semantic_object" for obj in managed),
    "room_proxy_count": sum(obj.get(object_utils.KIND_PROPERTY) == "room_proxy" for obj in managed),
    "unrelated_object_preserved": bpy.data.objects.get(unrelated.name) is unrelated,
    "repeat_sync_created": len(second["payload"]["created"]),
    "repeat_sync_updated": len(second["payload"]["updated"]),
    "repeat_sync_unchanged": len(second["payload"]["unchanged"]),
    "selective_update_ids": changed["payload"]["updated"],
    "stale_scene_error_code": stale_error_code,
    "selected_semantic_id": target_id,
    "reported_event_types": event_types,
    "saved_sync_status": object_utils.sync_status(),
    "blend_file": "accepted-office-v3.1.1-sync.blend",
}
(output_dir / "accepted-office-v3.1.1-blender-smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
