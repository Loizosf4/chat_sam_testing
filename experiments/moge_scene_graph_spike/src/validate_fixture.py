"""Validate a scene fixture without importing the existing application."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = EXPERIMENT_ROOT / "schemas" / "manifest.schema.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_fixture(manifest_path: Path) -> list[str]:
    """Return validation errors; an empty list means the fixture is valid."""
    manifest_path = manifest_path.resolve()
    fixture_dir = manifest_path.parent
    errors: list[str] = []

    try:
        manifest = _load_json(manifest_path)
        schema = _load_json(SCHEMA_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        return [str(exc)]

    for error in Draft202012Validator(schema).iter_errors(manifest):
        location = ".".join(str(part) for part in error.absolute_path) or "manifest"
        errors.append(f"{location}: {error.message}")
    if errors:
        return errors

    image_meta = manifest["image"]
    image_path = fixture_dir / image_meta["filename"]
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image_size = image.size
    except (OSError, SyntaxError) as exc:
        return [f"image: cannot read {image_path.name}: {exc}"]

    declared_size = (image_meta["width"], image_meta["height"])
    if image_size != declared_size:
        errors.append(f"image: actual size {image_size} != declared size {declared_size}")
    if _sha256(image_path) != image_meta["sha256"]:
        errors.append(f"image: SHA-256 mismatch for {image_path.name}")

    object_ids = [obj["object_id"] for obj in manifest["objects"]]
    if len(object_ids) != len(set(object_ids)):
        errors.append("objects: object_id values must be unique")

    for obj in manifest["objects"]:
        label = obj["semantic_label"]
        mask_path = fixture_dir / obj["mask_filename"]
        if mask_path.suffix.lower() != ".png":
            errors.append(f"{label}: mask is not a PNG")
            continue
        try:
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.format != "PNG":
                    errors.append(f"{label}: file contents are not PNG")
                if mask.size != image_size:
                    errors.append(f"{label}: mask size {mask.size} != image size {image_size}")
                values = set(mask.convert("L").getdata())
        except (OSError, SyntaxError) as exc:
            errors.append(f"{label}: cannot read mask: {exc}")
            continue
        if not values.issubset({0, 255}):
            errors.append(f"{label}: mask is not binary; values include {sorted(values)[:8]}")
        if 255 not in values:
            errors.append(f"{label}: mask has no foreground pixels")
        if _sha256(mask_path) != obj["sha256"]:
            errors.append(f"{label}: SHA-256 mismatch")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate_fixture(args.manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {args.manifest} is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
