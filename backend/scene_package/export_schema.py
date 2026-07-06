"""Write the canonical JSON Schema generated from the typed contract."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Scene


SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "scene-package-v1.schema.json"


def export_schema(path: Path = SCHEMA_PATH) -> Path:
    schema = Scene.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://schemas.cyens.org/scene-package/1.0.0/schema.json"
    schema["title"] = "Shared Scene Package 1.0.0"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    export_schema()
