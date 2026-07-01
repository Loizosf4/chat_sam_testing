import json
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry" / "scene_evidence.json"


@unittest.skipUnless(OUTPUT.exists(), "run deterministic scene evidence first")
def test_scene_evidence_output_is_self_consistent() -> None:
    data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    assert {plane["semantic_candidate"] for plane in data["structural_planes"]} == {"floor", "left_wall", "right_wall"}
    assert len(data["object_geometry_references"]) == 6
    transforms = data["transforms"]
    forward = np.asarray(transforms["raw_moge_to_canonical_scene_world"])
    inverse = np.asarray(transforms["canonical_scene_world_to_raw_moge"])
    assert np.allclose(inverse @ forward, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(forward[:3, :3]), 1.0, atol=1e-10)
    floor = next(plane for plane in data["structural_planes"] if plane["semantic_candidate"] == "floor")
    assert np.allclose(floor["canonical_plane_equation"]["normal"], [0, 0, 1], atol=1e-8)
    assert abs(floor["canonical_plane_equation"]["offset"]) < 1e-8
    assert data["quality_checks"]["canonical_distance_preservation_max_error"] < 1e-5
    assert data["quality_checks"]["scene_normalized_distance_scale_max_error"] < 1e-5

    relationships = data["relationship_candidates"]
    labels = {item["semantic_label"]: item["object_id"] for item in data["object_geometry_references"]}
    assert any(r["subject_object_id"] == labels["desktop_box"] and r["predicate"] == "supported_by" and r["target_id"] == labels["desk"] for r in relationships)
    assert any(r["subject_object_id"] == labels["desk_chair"] and r["predicate"] == "unknown_support" for r in relationships)
    valid_targets = set(labels.values()) | {plane["plane_id"] for plane in data["structural_planes"]}
    assert all(r["target_id"] is None or r["target_id"] in valid_targets for r in relationships)
    diagnostics = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry"
    for relative_path in data["diagnostic_artifacts"].values():
        assert (diagnostics / relative_path).is_file()
