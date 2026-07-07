"""Blender add-on entrypoint for semantic scene-package synchronization."""

from __future__ import annotations

bl_info = {
    "name": "SAM Semantic Scene Bridge",
    "author": "CYENS SAM",
    "version": (1, 0, 1),
    "blender": (4, 0, 0),
    "category": "Object",
    "description": "Synchronize scene-package semantic objects with the mask review website.",
}


def register() -> None:
    from . import selection_listener, server

    server.start_server()
    selection_listener.install_selection_listener()


def unregister() -> None:
    from . import selection_listener, server

    selection_listener.remove_selection_listener()
    server.stop_server()
