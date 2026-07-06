"""Report managed Blender selection and transform changes to the website."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import bpy

from . import event_bus, object_utils
from .protocol import SCHEMA_VERSION


_last_selected: tuple[str | None, str | None] = (None, None)
_transform_baseline: dict[str, str] = {}


def _event(message_type: str, obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "message_id": str(uuid4()),
        "type": message_type,
        "scene_id": obj.get(object_utils.SCENE_PROPERTY),
        "object_id": obj.get(object_utils.ID_PROPERTY),
        "revision": bpy.context.scene.get("sam_scene_revision"),
        "payload": {
            "object_version": obj.get("sam_object_version"),
            "mask_revision": obj.get("sam_mask_revision"),
            "transform": object_utils.object_state(obj)["transform"],
        },
    }


def refresh_baseline() -> None:
    global _transform_baseline
    _transform_baseline = {
        str(obj.get(object_utils.ID_PROPERTY)): object_utils.transform_signature(obj)
        for obj in object_utils.managed_objects()
    }


def _on_depsgraph_update(_scene: bpy.types.Scene, depsgraph: Any) -> None:
    global _last_selected
    active = bpy.context.view_layer.objects.active
    selected = (
        (str(active.get(object_utils.SCENE_PROPERTY)), str(active.get(object_utils.ID_PROPERTY)))
        if active and active.select_get() and active.get(object_utils.ID_PROPERTY)
        else (None, None)
    )
    if selected != _last_selected:
        _last_selected = selected
        if active and selected[1]:
            event_bus.publish(_event("selection_changed", active))

    for update in depsgraph.updates:
        obj = getattr(update, "id", None)
        if not isinstance(obj, bpy.types.Object) or not obj.get(object_utils.ID_PROPERTY):
            continue
        semantic_id = str(obj.get(object_utils.ID_PROPERTY))
        signature = object_utils.transform_signature(obj)
        previous = _transform_baseline.get(semantic_id)
        _transform_baseline[semantic_id] = signature
        if previous is not None and previous != signature:
            event_bus.publish(_event("transform_update", obj))


def install_selection_listener() -> None:
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    refresh_baseline()


def remove_selection_listener() -> None:
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
