from __future__ import annotations

import json

from fastapi.testclient import TestClient

from backend import blender_sync_proxy
from backend.main import app


def test_proxy_forwards_health_and_events(monkeypatch) -> None:
    calls = []

    def forward(method: str, path: str, body: bytes | None = None):
        calls.append((method, path, body))
        payload = {"ok": True} if path == "/health" else {"latest_sequence": 0, "events": []}
        return 200, json.dumps(payload).encode(), "application/json"

    monkeypatch.setattr(blender_sync_proxy, "_forward", forward)
    client = TestClient(app)
    assert client.get("/api/blender-sync/health").json() == {"ok": True}
    assert client.get("/api/blender-sync/events?since=7").json()["latest_sequence"] == 0
    assert calls[:2] == [("GET", "/health", None), ("GET", "/events?since=7", None)]


def test_proxy_forwards_command_without_rewriting(monkeypatch) -> None:
    captured = {}

    def forward(method: str, path: str, body: bytes | None = None):
        captured.update(method=method, path=path, body=body)
        return 200, b'{"type":"sync_ack","payload":{}}', "application/json"

    monkeypatch.setattr(blender_sync_proxy, "_forward", forward)
    client = TestClient(app)
    message = {"schema_version": "1.0.0", "type": "handshake", "payload": {}}
    response = client.post("/api/blender-sync/command", json=message)
    assert response.status_code == 200
    assert captured["method"] == "POST" and captured["path"] == "/command"
    assert json.loads(captured["body"]) == message


def test_proxy_rejects_invalid_command_before_forwarding(monkeypatch) -> None:
    monkeypatch.setattr(blender_sync_proxy, "_forward", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not forward")))
    client = TestClient(app)
    assert client.post("/api/blender-sync/command", content=b"[]", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/api/blender-sync/command", content=b"{}", headers={"Content-Type": "text/plain"}).status_code == 415
