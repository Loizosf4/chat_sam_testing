"""Same-origin proxy for the localhost-only Blender semantic bridge."""

from __future__ import annotations

import http.client
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response


BLENDER_HOST = "127.0.0.1"
BLENDER_PORT = int(os.environ.get("SAM_BLENDER_PORT", "8765"))
MAX_BODY_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 16

router = APIRouter(prefix="/api/blender-sync", tags=["blender-sync"])


def _forward(method: str, path: str, body: bytes | None = None) -> tuple[int, bytes, str]:
    connection = http.client.HTTPConnection(BLENDER_HOST, BLENDER_PORT, timeout=TIMEOUT_SECONDS)
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read(MAX_BODY_BYTES + 1)
        if len(content) > MAX_BODY_BYTES:
            raise HTTPException(502, "Blender bridge response is too large")
        return response.status, content, response.getheader("Content-Type", "application/json")
    except (ConnectionError, OSError, TimeoutError, http.client.HTTPException) as exc:
        raise HTTPException(503, "Blender bridge is not reachable") from exc
    finally:
        connection.close()


def _response(status: int, content: bytes, media_type: str) -> Response:
    return Response(content=content, status_code=status, media_type=media_type.split(";", 1)[0])


@router.get("/health")
def blender_health() -> Response:
    return _response(*_forward("GET", "/health"))


@router.get("/events")
def blender_events(since: int = Query(default=0, ge=0)) -> Response:
    return _response(*_forward("GET", f"/events?since={since}"))


@router.post("/command")
async def blender_command(request: Request) -> Response:
    if "application/json" not in request.headers.get("content-type", ""):
        raise HTTPException(415, "Content-Type must be application/json")
    body = await request.body()
    if not body or len(body) > MAX_BODY_BYTES:
        raise HTTPException(413 if body else 400, "invalid Blender command size")
    try:
        value: Any = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, "Blender command must be valid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "Blender command must be a JSON object")
    return _response(*_forward("POST", "/command", body))
