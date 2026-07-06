"""Read-only checkpoint extractor used by the final benchmark validator.

Run with Blender in background mode.  This script never saves the blend file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _plain(value):
    if hasattr(value, "to_list"):
        return value.to_list()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    objects = []
    for obj in bpy.data.objects:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.location),
                "dimensions": list(obj.dimensions),
                "rotation_mode": obj.rotation_mode,
                "rotation_quaternion_wxyz": list(obj.rotation_quaternion),
                "matrix_world": [list(row) for row in obj.matrix_world],
                "custom_properties": {
                    key: _plain(obj[key]) for key in obj.keys() if key != "_RNA_UI"
                },
            }
        )

    payload = {
        "blend_filepath": bpy.data.filepath,
        "objects": objects,
        "scene_camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
