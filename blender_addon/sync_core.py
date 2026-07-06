"""Pure synchronization planning, independent of Blender's Python runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .protocol import ProtocolError


@dataclass(frozen=True)
class MappingState:
    semantic_id: str
    kind: str
    revision: str
    object_name: str


def transform_revision(item: dict[str, Any]) -> str:
    transform = item.get("transform") or {
        "center": item.get("center"),
        "dimensions": item.get("dimensions"),
        "quaternion": item.get("quaternion"),
    }
    encoded = json.dumps(transform, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def semantic_revision(item: dict[str, Any]) -> str:
    return f"{int(item.get('version', 1))}:{item.get('mask_revision', '')}:{transform_revision(item)}"


def assert_fresh_scene_revision(incoming: int, stored: int | None) -> None:
    if stored is not None and incoming < stored:
        raise ProtocolError("stale_revision", "incoming scene revision is older than Blender", details={"blender_revision": stored})


def validate_scene_package(scene: dict[str, Any]) -> None:
    required = ("schema_version", "package_revision", "scene_id", "semantic_objects", "structural_room_proxies")
    missing = [field for field in required if field not in scene]
    if missing:
        raise ProtocolError("invalid_scene_package", f"scene package is missing: {', '.join(missing)}")
    all_ids: list[str] = []
    for item in scene["semantic_objects"]:
        all_ids.append(str(item.get("object_id", "")))
    for item in scene["structural_room_proxies"]:
        all_ids.append(str(item.get("proxy_id", "")))
    duplicates = sorted({item_id for item_id in all_ids if all_ids.count(item_id) > 1})
    if duplicates:
        raise ProtocolError("duplicate_semantic_id", "scene package contains duplicate semantic IDs", details={"object_ids": duplicates})


def plan_sync(scene: dict[str, Any], existing: Iterable[MappingState]) -> dict[str, Any]:
    validate_scene_package(scene)
    existing_by_id: dict[str, MappingState] = {}
    duplicates: set[str] = set()
    for mapping in existing:
        if mapping.semantic_id in existing_by_id:
            duplicates.add(mapping.semantic_id)
        existing_by_id[mapping.semantic_id] = mapping
    if duplicates:
        raise ProtocolError("duplicate_semantic_id", "Blender contains duplicate semantic IDs", details={"object_ids": sorted(duplicates)})

    desired: list[tuple[str, str, str, dict[str, Any]]] = []
    for proxy in scene["structural_room_proxies"]:
        desired.append((proxy["proxy_id"], "room_proxy", transform_revision(proxy), proxy))
    for item in scene["semantic_objects"]:
        desired.append((item["object_id"], "semantic_object", semantic_revision(item), item))

    create, update, unchanged = [], [], []
    for semantic_id, kind, revision, item in desired:
        current = existing_by_id.get(semantic_id)
        operation = {"semantic_id": semantic_id, "kind": kind, "revision": revision, "item": item}
        if current is None:
            create.append(operation)
        elif current.kind != kind or current.revision != revision:
            update.append(operation)
        else:
            unchanged.append(operation)
    return {"create": create, "update": update, "unchanged": unchanged}
