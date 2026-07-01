"""Export the strict primitive scene-plan JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

from .primitive_plan_models import PrimitiveScenePlan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "schemas" / "primitive_scene_plan.schema.json"


def export_schema(path: Path = DEFAULT_OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(PrimitiveScenePlan.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    print(export_schema())
