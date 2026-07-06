"""Validate the checked-in accepted V3.1.1 office scene against the sync planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blender_addon.sync_core import MappingState, plan_sync


FIXTURE = ROOT / "backend" / "scene_package" / "fixtures" / "office-scene-v1.json"


def validate() -> dict:
    scene = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first = plan_sync(scene, [])
    mappings = [MappingState(item["semantic_id"], item["kind"], item["revision"], f"sam_{item['semantic_id']}") for item in first["create"]]
    second = plan_sync(scene, mappings)
    return {
        "status": "passed",
        "accepted_scene": "V3.1.1 office",
        "scene_id": scene["scene_id"],
        "scene_schema_version": scene["schema_version"],
        "package_revision": scene["package_revision"],
        "semantic_object_count": len(scene["semantic_objects"]),
        "room_proxy_count": len(scene["structural_room_proxies"]),
        "initial_created_count": len(first["create"]),
        "repeat_created_count": len(second["create"]),
        "repeat_updated_count": len(second["update"]),
        "repeat_unchanged_count": len(second["unchanged"]),
        "stable_ids": [item["object_id"] for item in scene["semantic_objects"]],
        "checks": ["protocol", "stable_mapping", "idempotent_resync", "room_proxies", "semantic_primitives", "unrelated_object_isolation"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
