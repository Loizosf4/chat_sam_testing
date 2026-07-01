"""Validate persisted MoGe numerical outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_KEYS = {"points", "depth", "normal", "valid_mask", "intrinsics"}


def validate_output(output_dir: Path) -> list[str]:
    errors: list[str] = []
    metadata_path = output_dir / "metadata.json"
    archive_path = output_dir / "geometry.npz"
    if not metadata_path.is_file() or not archive_path.is_file():
        return ["metadata.json and geometry.npz must exist"]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    width = metadata["source_image_dimensions"]["width"]
    height = metadata["source_image_dimensions"]["height"]
    with np.load(archive_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = REQUIRED_KEYS - keys
        if missing:
            errors.append(f"missing keys: {sorted(missing)}")
            return errors
        points, depth = data["points"], data["depth"]
        valid, intrinsics = data["valid_mask"], data["intrinsics"]
        if points.shape != (height, width, 3):
            errors.append(f"points shape is {points.shape}")
        if depth.shape != (height, width):
            errors.append(f"depth shape is {depth.shape}")
        if valid.shape != (height, width):
            errors.append(f"valid_mask shape is {valid.shape}")
        if intrinsics.shape != (3, 3):
            errors.append(f"intrinsics shape is {intrinsics.shape}")
        if not np.any(valid):
            errors.append("valid_mask has no valid pixels")
        elif not np.isfinite(points[valid]).all():
            errors.append("points contain non-finite values in valid regions")
        if np.any(valid) and not np.isfinite(depth[valid]).all():
            errors.append("depth contains non-finite values in valid regions")
        if np.any(valid) and not (depth[valid] > 0).all():
            errors.append("depth is not positive in every valid region")
        if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            errors.append("intrinsics are invalid")
        normal = data["normal"]
        if normal.shape != (height, width, 3):
            errors.append(f"normal shape is {normal.shape}")
        elif not np.isfinite(normal[valid]).all():
            errors.append("normal contains non-finite values in valid regions")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=EXPERIMENT_ROOT / "outputs" / "office_test" / "moge")
    args = parser.parse_args()
    errors = validate_output(args.output_dir)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: {args.output_dir} contains valid MoGe outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
