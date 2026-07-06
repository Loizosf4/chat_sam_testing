"""Versioned shared scene-package contract."""

from .migrations import CURRENT_SCHEMA_VERSION, migrate_document
from .models import MaskRevision, Scene, SemanticObject
from .serialization import (
    RevisionConflictError,
    dump_scene,
    load_scene,
    validate_optimistic_lock,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MaskRevision",
    "RevisionConflictError",
    "Scene",
    "SemanticObject",
    "dump_scene",
    "load_scene",
    "migrate_document",
    "validate_optimistic_lock",
]
