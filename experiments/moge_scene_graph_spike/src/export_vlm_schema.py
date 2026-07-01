"""Export the Pydantic VLM contract as JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from src.vlm_models import SemanticSceneGraph


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def export_schema(path: Path = EXPERIMENT_ROOT / "schemas" / "vlm_scene_graph.schema.json") -> Path:
    schema = SemanticSceneGraph.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "vlm_scene_graph.schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(export_schema())
