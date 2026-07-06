"""Protocol validation shared by the Blender command server and unit tests."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = "1.0.0"
SUPPORTED_TYPES = {
    "handshake",
    "scene_sync",
    "select_highlight",
    "transform_update",
    "sync_ack",
    "selection_changed",
    "error",
    # Reference-addon compatibility commands.
    "create_scene",
    "create_proxy_object",
    "update_proxy_transform",
    "select_object",
    "highlight_object",
    "get_selected_object",
    "get_scene_state",
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def validate_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_envelope", "message must be a JSON object")
    message_type = value.get("type") or value.get("command")
    if not isinstance(message_type, str) or message_type not in SUPPORTED_TYPES:
        raise ProtocolError("unsupported_message", f"unsupported message type: {message_type!r}")
    schema_version = value.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ProtocolError(
            "schema_version_mismatch",
            f"schema version {schema_version!r} is not supported",
            details={"supported_schema_versions": [SCHEMA_VERSION]},
        )
    payload = value.get("payload", {})
    if not isinstance(payload, dict):
        raise ProtocolError("invalid_payload", "payload must be a JSON object")
    result = dict(value)
    result["type"] = message_type
    result["schema_version"] = schema_version
    result["message_id"] = str(value.get("message_id") or uuid4())
    result["payload"] = payload
    for field in ("scene_id", "object_id"):
        field_value = result.get(field)
        if field_value is not None and (not isinstance(field_value, str) or not ID_RE.fullmatch(field_value)):
            raise ProtocolError("invalid_identifier", f"{field} is invalid")
    revision = result.get("revision")
    if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 1):
        raise ProtocolError("invalid_revision", "revision must be a positive integer")
    return result


def response(
    request: dict[str, Any],
    message_type: str,
    payload: dict[str, Any] | None = None,
    *,
    revision: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "message_id": str(uuid4()),
        "correlation_id": request.get("message_id"),
        "type": message_type,
        "scene_id": request.get("scene_id"),
        "object_id": request.get("object_id"),
        "revision": revision if revision is not None else request.get("revision"),
        "payload": payload or {},
    }


def error_response(request: dict[str, Any] | None, error: ProtocolError | Exception) -> dict[str, Any]:
    request = request or {}
    code = error.code if isinstance(error, ProtocolError) else "internal_error"
    details = error.details if isinstance(error, ProtocolError) else {}
    return response(
        request,
        "error",
        {"code": code, "message": str(error), "details": details, "retryable": code in {"timeout", "disconnected"}},
    )
