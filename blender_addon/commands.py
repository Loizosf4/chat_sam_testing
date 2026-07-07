"""Allowlisted semantic synchronization command handlers."""

from __future__ import annotations

from typing import Any, Callable

import bpy

from . import event_bus, object_utils
from .protocol import ProtocolError, response
from .sync_core import MappingState, assert_fresh_scene_revision, plan_sync


def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    handler = COMMANDS.get(request["type"])
    if handler is None:
        raise ProtocolError("unsupported_message", f"unsupported message type: {request['type']}")
    return handler(request)


def handshake(request: dict[str, Any]) -> dict[str, Any]:
    status = object_utils.sync_status()
    return response(request, "sync_ack", {
        "addon_version": "1.0.1",
        "capabilities": ["scene_sync", "select_highlight", "selection_changed", "transform_update"],
        "sync_status": status,
    }, revision=status.get("revision"))


def scene_sync(request: dict[str, Any]) -> dict[str, Any]:
    scene = request["payload"].get("scene_package")
    if not isinstance(scene, dict):
        raise ProtocolError("invalid_scene_package", "payload.scene_package is required")
    scene_id = request.get("scene_id") or scene.get("scene_id")
    revision = request.get("revision") or scene.get("package_revision")
    if scene_id != scene.get("scene_id") or revision != scene.get("package_revision"):
        raise ProtocolError("scene_mismatch", "envelope scene ID/revision does not match the scene package")

    current = object_utils.sync_status()
    if current.get("scene_id") == scene_id:
        assert_fresh_scene_revision(revision, current.get("revision"))

    existing = [MappingState(**item) for item in object_utils.mappings(scene_id)]
    plan = plan_sync(scene, existing)
    created: list[str] = []
    updated: list[str] = []
    for bucket, target in (("create", created), ("update", updated)):
        for operation in plan[bucket]:
            obj, was_created = object_utils.ensure_cube(scene_id, operation["semantic_id"])
            object_utils.apply_item(obj, scene_id, operation)
            target.append(operation["semantic_id"])
            if was_created and bucket == "update":
                created.append(operation["semantic_id"])

    unchanged = [operation["semantic_id"] for operation in plan["unchanged"]]
    stale = request["payload"].get("stale_object_ids", [])
    details = {"created": created, "updated": updated, "unchanged": unchanged, "stale_object_ids": stale}
    object_utils.save_sync_status(scene_id, revision, "synchronized", details)
    from . import selection_listener

    selection_listener.refresh_baseline()
    return response(request, "sync_ack", details, revision=revision)


def select_highlight(request: dict[str, Any]) -> dict[str, Any]:
    scene_id = request.get("scene_id")
    object_id = request.get("object_id")
    if not scene_id or not object_id:
        raise ProtocolError("invalid_identifier", "scene_id and object_id are required")
    obj = object_utils.find_unique(scene_id, object_id)
    if obj is None:
        raise ProtocolError("mapping_not_found", f"semantic object not found: {object_id}")
    requested_mask_revision = request["payload"].get("mask_revision")
    if requested_mask_revision and requested_mask_revision != obj.get("sam_mask_revision"):
        raise ProtocolError("stale_mask_revision", "website mask revision differs from Blender", details={"blender_mask_revision": obj.get("sam_mask_revision")})
    obj = object_utils.select_and_highlight(scene_id, object_id)
    return response(request, "sync_ack", {"selected": object_utils.object_state(obj)})


def sync_ack(request: dict[str, Any]) -> dict[str, Any]:
    return response(request, "sync_ack", {"accepted": True})


# Compatibility adapters for the reference add-on's core command names.
def compatibility(request: dict[str, Any]) -> dict[str, Any]:
    payload = request["payload"]
    request = dict(request)
    request["scene_id"] = request.get("scene_id") or payload.get("scene_id")
    request["object_id"] = request.get("object_id") or payload.get("object_id")
    if request["type"] in {"select_object", "highlight_object"}:
        return select_highlight(request)
    if request["type"] == "get_selected_object":
        obj = bpy.context.view_layer.objects.active
        return response(request, "sync_ack", {"selected": object_utils.object_state(obj) if obj and obj.get(object_utils.ID_PROPERTY) else None})
    if request["type"] == "get_scene_state":
        return response(request, "sync_ack", {"sync_status": object_utils.sync_status(), "objects": [object_utils.object_state(obj) for obj in object_utils.managed_objects(request.get("scene_id"))]})
    raise ProtocolError("legacy_command_requires_scene_sync", f"{request['type']} is superseded by scene_sync")


COMMANDS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "handshake": handshake,
    "scene_sync": scene_sync,
    "select_highlight": select_highlight,
    "sync_ack": sync_ack,
    "select_object": compatibility,
    "highlight_object": compatibility,
    "get_selected_object": compatibility,
    "get_scene_state": compatibility,
    "create_scene": compatibility,
    "create_proxy_object": compatibility,
    "update_proxy_transform": compatibility,
}
