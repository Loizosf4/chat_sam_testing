"""Serialization and optimistic concurrency checks for scene packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .migrations import migrate_document
from .models import Scene


class RevisionConflictError(RuntimeError):
    pass


def load_scene(value: str | bytes | Path | Mapping[str, object]) -> Scene:
    if isinstance(value, Path):
        raw = json.loads(value.read_text(encoding="utf-8"))
    elif isinstance(value, bytes):
        raw = json.loads(value.decode("utf-8"))
    elif isinstance(value, str):
        raw = json.loads(value)
    else:
        raw = dict(value)
    return Scene.model_validate(migrate_document(raw))


def dump_scene(scene: Scene, *, indent: int | None = 2) -> str:
    return scene.model_dump_json(indent=indent, exclude_none=False) + ("\n" if indent else "")


def validate_optimistic_lock(
    scene: Scene,
    *,
    expected_package_revision: int,
    expected_object_versions: Mapping[str, int] | None = None,
    expected_mask_revisions: Mapping[str, str] | None = None,
) -> None:
    """Raise before a write when any caller-supplied revision is stale."""
    if scene.package_revision != expected_package_revision:
        raise RevisionConflictError(
            f"scene revision conflict: expected {expected_package_revision}, "
            f"current {scene.package_revision}"
        )
    by_id = {item.object_id: item for item in scene.semantic_objects}
    for object_id, expected in (expected_object_versions or {}).items():
        current = by_id.get(object_id)
        if current is None:
            raise RevisionConflictError(f"unknown object {object_id!r}")
        if current.version != expected:
            raise RevisionConflictError(
                f"object {object_id} version conflict: expected {expected}, "
                f"current {current.version}"
            )
    for object_id, expected in (expected_mask_revisions or {}).items():
        current = by_id.get(object_id)
        if current is None:
            raise RevisionConflictError(f"unknown object {object_id!r}")
        if current.mask_revision != expected:
            raise RevisionConflictError(
                f"object {object_id} mask revision conflict: expected {expected}, "
                f"current {current.mask_revision}"
            )
