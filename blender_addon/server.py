"""Local-only HTTP JSON server based on the reference add-on transport."""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import bpy

from . import commands, event_bus
from .protocol import ProtocolError, error_response, validate_envelope


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 8 * 1024 * 1024
_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None
_task_queue: queue.Queue["CommandTask"] = queue.Queue()
_timer_registered = False


@dataclass
class CommandTask:
    request: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class BlenderCommandHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BlenderCommandHandler(BaseHTTPRequestHandler):
    server_version = "SAMSemanticBridge/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(200, {"ok": True, "status": "running", "schema_version": "1.0.0"})
            return
        if parsed.path == "/events":
            try:
                sequence = max(0, int(parse_qs(parsed.query).get("since", ["0"])[0]))
            except ValueError:
                self._send_json(400, error_response(None, ProtocolError("invalid_sequence", "since must be an integer")))
                return
            latest, events = event_bus.since(sequence)
            self._send_json(200, {"schema_version": "1.0.0", "latest_sequence": latest, "events": events})
            return
        self._send_json(404, error_response(None, ProtocolError("not_found", "unknown endpoint")))

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/command":
            self._send_json(404, error_response(None, ProtocolError("not_found", "use POST /command")))
            return
        if "application/json" not in self.headers.get("Content-Type", ""):
            self._send_json(415, error_response(None, ProtocolError("invalid_content_type", "Content-Type must be application/json")))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
            request = validate_envelope(raw)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError, ProtocolError) as exc:
            protocol_error = exc if isinstance(exc, ProtocolError) else ProtocolError("invalid_request", "request body is invalid")
            self._send_json(400, error_response(None, protocol_error))
            return
        task = CommandTask(request=request)
        _task_queue.put(task)
        if not task.event.wait(timeout=15.0):
            self._send_json(504, error_response(request, ProtocolError("timeout", "command timed out in Blender")))
            return
        status = 200 if task.response and task.response.get("type") != "error" else 409
        self._send_json(status, task.response or error_response(request, RuntimeError("empty response")))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[SAM Semantic Bridge] {format % args}")

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(encoded)


def start_server(host: str = DEFAULT_HOST, port: int | None = None) -> dict[str, Any]:
    global _server, _server_thread, _timer_registered
    port = port or int(os.getenv("SAM_BLENDER_PORT", str(DEFAULT_PORT)))
    if host != DEFAULT_HOST:
        raise ValueError("only localhost is supported")
    if _server is not None:
        return {"host": host, "port": _server.server_address[1], "status": "already_running"}
    _server = BlenderCommandHTTPServer((host, port), BlenderCommandHandler)
    _server_thread = threading.Thread(target=_server.serve_forever, name="SAMSemanticBridgeServer", daemon=True)
    _server_thread.start()
    if not _timer_registered:
        bpy.app.timers.register(_process_command_queue, persistent=True)
        _timer_registered = True
    print(f"[SAM Semantic Bridge] Listening on http://{host}:{port}")
    return {"host": host, "port": port, "status": "running"}


def stop_server() -> None:
    global _server, _server_thread, _timer_registered
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
    if _server_thread is not None:
        _server_thread.join(timeout=2.0)
        _server_thread = None
    _timer_registered = False


def _process_command_queue() -> float | None:
    while True:
        try:
            task = _task_queue.get_nowait()
        except queue.Empty:
            break
        try:
            task.response = commands.dispatch(task.request)
        except ProtocolError as exc:
            task.response = error_response(task.request, exc)
        except ValueError as exc:
            task.response = error_response(task.request, ProtocolError("mapping_error", str(exc)))
        except Exception as exc:
            task.response = error_response(task.request, exc)
        task.event.set()
    return 0.05 if _server is not None else None
