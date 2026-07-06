"""Blender object ownership, stable-ID mapping, transforms, and status storage."""

from __future__ import annotations

import json
from math import degrees
from typing import Any

import bpy


COLLECTION_PREFIX = "SAM Scene "
ID_PROPERTY = "sam_semantic_id"
SCENE_PROPERTY = "sam_scene_id"
KIND_PROPERTY = "sam_kind"
REVISION_PROPERTY = "sam_revision"
HIGHLIGHT_MATERIAL = "SAM_Selected_Semantic_Object"
DEFAULT_MATERIAL = "SAM_Semantic_Primitive"
ROOM_MATERIAL = "SAM_Room_Proxy"


def collection_name(scene_id: str) -> str:
    return f"{COLLECTION_PREFIX}{scene_id}"


def get_or_create_collection(scene_id: str) -> bpy.types.Collection:
    name = collection_name(scene_id)
    collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if bpy.context.scene.collection.children.get(name) is None:
        bpy.context.scene.collection.children.link(collection)
    return collection


def managed_objects(scene_id: str | None = None) -> list[bpy.types.Object]:
    return [
        obj for obj in bpy.data.objects
        if obj.get(ID_PROPERTY) and (scene_id is None or obj.get(SCENE_PROPERTY) == scene_id)
    ]


def mappings(scene_id: str) -> list[dict[str, str]]:
    return [
        {
            "semantic_id": str(obj.get(ID_PROPERTY)),
            "kind": str(obj.get(KIND_PROPERTY, "semantic_object")),
            "revision": str(obj.get(REVISION_PROPERTY, "")),
            "object_name": obj.name,
        }
        for obj in managed_objects(scene_id)
    ]


def find_unique(scene_id: str, semantic_id: str) -> bpy.types.Object | None:
    matches = [obj for obj in managed_objects(scene_id) if obj.get(ID_PROPERTY) == semantic_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate semantic ID in Blender: {semantic_id}")
    return matches[0] if matches else None


def ensure_cube(scene_id: str, semantic_id: str) -> tuple[bpy.types.Object, bool]:
    existing = find_unique(scene_id, semantic_id)
    if existing is not None:
        if existing.type != "MESH":
            raise ValueError(f"managed object {semantic_id} is not a mesh")
        return existing, False
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.object
    obj.name = f"sam_{semantic_id}"
    link_to_collection(obj, scene_id)
    return obj, True


def link_to_collection(obj: bpy.types.Object, scene_id: str) -> None:
    collection = get_or_create_collection(scene_id)
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)
    for other in list(obj.users_collection):
        if other != collection:
            other.objects.unlink(obj)


def apply_item(obj: bpy.types.Object, scene_id: str, operation: dict[str, Any]) -> None:
    item = operation["item"]
    kind = operation["kind"]
    transform = item.get("transform") or item
    center = transform["center"]
    dimensions = transform["dimensions"]
    quaternion = transform["quaternion"]
    obj.location = tuple(float(value) for value in center)
    obj.scale = tuple(float(value) for value in dimensions)
    # Scene packages use wxyz, as does mathutils.Quaternion.
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = tuple(float(value) for value in quaternion)
    obj[ID_PROPERTY] = operation["semantic_id"]
    obj[SCENE_PROPERTY] = scene_id
    obj[KIND_PROPERTY] = kind
    obj[REVISION_PROPERTY] = operation["revision"]
    obj["sam_schema_version"] = "1.0.0"
    if kind == "semantic_object":
        obj["sam_object_version"] = int(item.get("version", 1))
        obj["sam_mask_revision"] = str(item.get("mask_revision", ""))
        obj["sam_semantic_label"] = str(item.get("semantic_label", ""))
        obj["sam_status"] = "primitive"
        set_material(obj, DEFAULT_MATERIAL, (0.35, 0.58, 0.92, 1.0))
    else:
        obj["sam_proxy_semantic"] = str(item.get("semantic", "room_boundary"))
        obj["sam_status"] = "room_proxy"
        set_material(obj, ROOM_MATERIAL, (0.22, 0.24, 0.28, 1.0))
    link_to_collection(obj, scene_id)


def set_material(obj: bpy.types.Object, name: str, color: tuple[float, float, float, float]) -> None:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = color
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.color = color
    obj.show_name = name == HIGHLIGHT_MATERIAL


def reset_highlights(scene_id: str) -> None:
    for obj in managed_objects(scene_id):
        if obj.get(KIND_PROPERTY) == "room_proxy":
            set_material(obj, ROOM_MATERIAL, (0.22, 0.24, 0.28, 1.0))
        else:
            set_material(obj, DEFAULT_MATERIAL, (0.35, 0.58, 0.92, 1.0))


def select_and_highlight(scene_id: str, semantic_id: str) -> bpy.types.Object:
    obj = find_unique(scene_id, semantic_id)
    if obj is None:
        raise ValueError(f"semantic object not found: {semantic_id}")
    bpy.ops.object.select_all(action="DESELECT")
    reset_highlights(scene_id)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    set_material(obj, HIGHLIGHT_MATERIAL, (1.0, 0.65, 0.05, 1.0))
    return obj


def object_state(obj: bpy.types.Object) -> dict[str, Any]:
    rotation = obj.rotation_quaternion if obj.rotation_mode == "QUATERNION" else obj.rotation_euler.to_quaternion()
    return {
        "object_name": obj.name,
        "scene_id": obj.get(SCENE_PROPERTY),
        "object_id": obj.get(ID_PROPERTY),
        "kind": obj.get(KIND_PROPERTY),
        "revision": obj.get(REVISION_PROPERTY),
        "object_version": obj.get("sam_object_version"),
        "mask_revision": obj.get("sam_mask_revision"),
        "transform": {
            "center": [round(float(value), 6) for value in obj.location],
            "dimensions": [round(float(value), 6) for value in obj.scale],
            "quaternion": [round(float(value), 8) for value in rotation],
        },
        "selected": obj.select_get(),
    }


def transform_signature(obj: bpy.types.Object) -> str:
    state = object_state(obj)["transform"]
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def save_sync_status(scene_id: str, revision: int, status: str, details: dict[str, Any]) -> None:
    scene = bpy.context.scene
    scene["sam_scene_id"] = scene_id
    scene["sam_scene_revision"] = int(revision)
    scene["sam_sync_status"] = status
    scene["sam_sync_details"] = json.dumps(details, sort_keys=True)


def sync_status() -> dict[str, Any]:
    scene = bpy.context.scene
    raw = scene.get("sam_sync_details", "{}")
    try:
        details = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        details = {}
    return {
        "scene_id": scene.get("sam_scene_id"),
        "revision": scene.get("sam_scene_revision"),
        "status": scene.get("sam_sync_status", "never_synced"),
        "details": details,
    }
