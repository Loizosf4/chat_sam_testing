from pathlib import Path

import json
import numpy as np
from PIL import Image

from src.object_geometry import connected_components
from src.validate_fixture import validate_fixture


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_office_fixture_is_valid() -> None:
    manifest = EXPERIMENT_ROOT / "inputs" / "office_test" / "manifest.json"
    assert validate_fixture(manifest) == []


def test_desk_fixture_is_binary_three_component_and_nonoverlapping() -> None:
    fixture = EXPERIMENT_ROOT / "inputs" / "office_test"
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    desk_record = next(obj for obj in manifest["objects"] if obj["semantic_label"] == "desk")
    desk = np.asarray(Image.open(fixture / desk_record["mask_filename"]).convert("L"))
    assert desk.shape == (447, 447)
    assert set(np.unique(desk).tolist()) == {0, 255}
    _, components = connected_components(desk == 255)
    assert len(components) == 3
    for obj in manifest["objects"]:
        if obj["object_id"] == desk_record["object_id"]:
            continue
        other = np.asarray(Image.open(fixture / obj["mask_filename"]).convert("L")) == 255
        assert int(((desk == 255) & other).sum()) <= 2
