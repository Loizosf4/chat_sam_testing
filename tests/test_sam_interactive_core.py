from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend import sam_engine


class FakePredictor:
    def __init__(self) -> None:
        self.set_image_calls = 0
        self.predict_calls: list[dict] = []
        self.height = 0
        self.width = 0

    def set_image(self, image_array: np.ndarray) -> None:
        self.set_image_calls += 1
        self.height, self.width = image_array.shape[:2]

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        count = 3 if kwargs.get("multimask_output") else 1
        masks = np.zeros((count, self.height, self.width), dtype=bool)
        for index in range(count):
            masks[index, index : index + 2, index : index + 3] = True
        scores = np.array([0.2, 0.9, 0.5][:count], dtype=np.float32)
        logits = np.ones((count, 4, 4), dtype=np.float32)
        return masks, scores, logits


def _image(path: Path, size: tuple[int, int] = (7, 5)) -> Path:
    Image.new("RGB", size, (12, 90, 180)).save(path, format="PNG")
    return path


def _install_fake_predictor(monkeypatch) -> FakePredictor:
    predictor = FakePredictor()
    monkeypatch.setattr(sam_engine, "_state", sam_engine.SamState(predictor=predictor, model_type="vit_b", device="cpu"))
    return predictor


def test_internal_candidate_prediction_returns_metadata_and_logits(tmp_path: Path, monkeypatch) -> None:
    predictor = _install_fake_predictor(monkeypatch)
    image = _image(tmp_path / "source.png")
    sam_engine.prepare_image_from_path("workspace-image", image)
    mask_input = np.ones((1, 4, 4), dtype=np.float32)

    prediction = sam_engine.predict_candidates(
        "workspace-image",
        points=[[1, 1], [3, 2]],
        point_labels=[1, 0],
        mask_input=mask_input,
        multimask_output=False,
    )

    assert prediction.masks.shape == (1, 5, 7)
    assert prediction.scores == [0.20000000298023224]
    assert prediction.areas == [6]
    assert prediction.bboxes == [[0, 0, 2, 1]]
    assert prediction.logits.shape == (1, 4, 4)
    assert predictor.predict_calls[-1]["mask_input"] is mask_input


def test_trusted_image_preparation_caches_by_image_key(tmp_path: Path, monkeypatch) -> None:
    predictor = _install_fake_predictor(monkeypatch)
    image = _image(tmp_path / "source.png")

    sam_engine.prepare_image_from_path("workspace-image", image)
    sam_engine.prepare_image_from_path("workspace-image", image)
    sam_engine.prepare_image_from_path("workspace-image-2", image)

    assert predictor.set_image_calls == 2


def test_predict_candidates_reprepares_trusted_path_after_workspace_switch(tmp_path: Path, monkeypatch) -> None:
    predictor = _install_fake_predictor(monkeypatch)
    image_a = _image(tmp_path / "a.png", size=(7, 5))
    image_b = _image(tmp_path / "b.png", size=(4, 3))

    sam_engine.prepare_image_from_path("workspace-a", image_a)
    sam_engine.prepare_image_from_path("workspace-b", image_b)
    prediction = sam_engine.predict_candidates(
        "workspace-a",
        image_path=image_a,
        points=[[1, 1]],
        point_labels=[1],
        multimask_output=False,
    )
    repeated = sam_engine.predict_candidates(
        "workspace-a",
        image_path=image_a,
        points=[[1, 1]],
        point_labels=[1],
        multimask_output=False,
    )

    assert prediction.masks.shape == (1, 5, 7)
    assert repeated.masks.shape == (1, 5, 7)
    assert predictor.set_image_calls == 3


def test_legacy_predict_still_persists_masks_and_response_shape(tmp_path: Path, monkeypatch) -> None:
    _install_fake_predictor(monkeypatch)
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    monkeypatch.setattr(sam_engine, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(sam_engine, "MASK_DIR", mask_dir)
    _image(image_dir / "legacy-image.png")

    response = sam_engine.predict(
        "legacy-image",
        points=[[1, 1]],
        point_labels=[1],
        multimask_output=True,
    )

    assert response["image_id"] == "legacy-image"
    assert len(response["masks"]) == 3
    for mask in response["masks"]:
        assert {"mask_id", "score", "area", "bbox", "png_base64"} <= set(mask)
        assert (mask_dir / f"{mask['mask_id']}.png").is_file()
        assert Image.open(io.BytesIO(base64.b64decode(mask["png_base64"]))).size == (7, 5)
