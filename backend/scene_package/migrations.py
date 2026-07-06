"""Ordered, explicit migrations for persisted scene-package documents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .models import CURRENT_SCHEMA_VERSION


class UnsupportedSchemaVersion(ValueError):
    pass


def _preview_0_9_0_to_1_0_0(document: dict[str, Any]) -> dict[str, Any]:
    """Migrate the documented pre-release field names to the stable v1 names."""
    migrated = deepcopy(document)
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    if "revision" in migrated:
        migrated["package_revision"] = migrated.pop("revision")
    if "objects" in migrated:
        migrated["semantic_objects"] = migrated.pop("objects")
    if "revisions" in migrated:
        migrated["mask_revisions"] = migrated.pop("revisions")
    if "selected_camera_id" in migrated:
        migrated["selected_camera"] = migrated.pop("selected_camera_id")
    return migrated


Migration = tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]
MIGRATIONS: dict[str, Migration] = {
    "0.9.0": (CURRENT_SCHEMA_VERSION, _preview_0_9_0_to_1_0_0),
}


def migrate_document(
    document: dict[str, Any], target_version: str = CURRENT_SCHEMA_VERSION
) -> dict[str, Any]:
    """Return a migrated copy; source documents are never mutated."""
    migrated = deepcopy(document)
    version = migrated.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedSchemaVersion("scene package is missing schema_version")
    visited: set[str] = set()
    while version != target_version:
        if version in visited or version not in MIGRATIONS:
            raise UnsupportedSchemaVersion(
                f"cannot migrate scene package from {version!r} to {target_version!r}"
            )
        visited.add(version)
        next_version, migration = MIGRATIONS[version]
        migrated = migration(migrated)
        version = next_version
    return migrated
