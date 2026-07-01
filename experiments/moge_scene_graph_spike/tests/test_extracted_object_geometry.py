import json
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry"
OUTPUT_PATH = GEOMETRY_DIR / "object_geometry.json"


@unittest.skipUnless(OUTPUT_PATH.exists(), "run object geometry extraction first")
def test_extracted_object_geometry_is_self_consistent() -> None:
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    objects = data["objects"]
    assert len(objects) == 6
    assert len({obj["object_id"] for obj in objects}) == len(objects)

    transform = data["raw_to_normalized_transformation"]
    forward = np.asarray(transform["raw_to_normalized_matrix_4x4"])
    inverse = np.asarray(transform["normalized_to_raw_matrix_4x4"])
    assert np.allclose(inverse @ forward, np.eye(4), atol=1e-10)

    for obj in objects:
        assert sum(component["pixel_count"] for component in obj["components"]) == obj["total_mask_pixel_count"]
        assert obj["connected_component_count"] == len(obj["components"])
        assert obj["point_filtering"]["raw_point_count"] >= obj["point_filtering"]["filtered_point_count"]
        if not obj["oriented_bounding_box"]["estimated"]:
            assert "center_xyz" not in obj["oriented_bounding_box"]

        diagnostic_dir = GEOMETRY_DIR / "diagnostics" / obj["object_id"]
        for filename in obj["diagnostics"].values():
            if filename is not None:
                assert (diagnostic_dir / filename).is_file()
        with np.load(diagnostic_dir / "raw_visible_geometry.npz", allow_pickle=False) as raw:
            assert raw["points"].shape == (obj["point_filtering"]["raw_point_count"], 3)
            assert raw["scene_normalized_points"].shape == raw["points"].shape
