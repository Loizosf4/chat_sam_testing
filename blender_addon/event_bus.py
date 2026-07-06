"""Bounded in-memory event journal exposed by GET /events."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

from .protocol import SCHEMA_VERSION


_lock = threading.Lock()
_sequence = 0
_events: deque[dict[str, Any]] = deque(maxlen=512)


def publish(event: dict[str, Any]) -> dict[str, Any]:
    global _sequence
    with _lock:
        _sequence += 1
        value = {"schema_version": SCHEMA_VERSION, "sequence": _sequence, **event}
        _events.append(value)
        return dict(value)


def since(sequence: int) -> tuple[int, list[dict[str, Any]]]:
    with _lock:
        return _sequence, [dict(item) for item in _events if item["sequence"] > sequence]


def clear() -> None:
    global _sequence
    with _lock:
        _sequence = 0
        _events.clear()
