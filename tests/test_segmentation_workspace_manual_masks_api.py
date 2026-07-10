from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import sam_engine
from backend.main import app
from backend.segmentation_workspace import api, service
from backend.segmentation_workspace.store import SegmentationWorkspaceStore


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "STORE", SegmentationWorkspaceStore(tmp_path / "workspaces"))
    monkeypatch.setattr(service, "LOGITS_CACHE", service.SamLogitsCache())
    monkeypatch.setattr(service.sam_engine, "load_model", lambda: {"model_type": "vit_b", "device": "cpu"})
    monkeypatch.setattr(
        service.sam_engine,
        "prepare_image_from_path",
        lambda image_key, path: {"image_set": True, "image_id": image_key, "width": 10, "height": 8},
    )

    def predict_candidates(*args, **kwargs):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, 1:4, 1:5] = True
        masks[1, 4:6, 4:8] = True
        return sam_engine.CandidatePrediction(
            image_key=args[0],
            width=10,
            height=8,
            masks=masks,
            scores=[0.9, 0.2],
            areas=[int(mask.sum()) for mask in masks],
            bboxes=[[1, 1, 4, 3], [4, 4, 7, 5]],
            logits=np.ones((2, 4, 4), dtype=np.float32),
            elapsed_ms=2.0,
        )

    monkeypatch.setattr(service.sam_engine, "predict_candidates", predict_candidates)
    return TestClient(app)


def _source_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (10, 8), (200, 40, 90)).save(output, format="PNG")
    return output.getvalue()


def _mask_png(mask: np.ndarray, mode: str = "L") -> bytes:
    output = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode=mode).save(output, format="PNG")
    return output.getvalue()


def _workspace_with_prediction(client: TestClient) -> tuple[dict, dict]:
    workspace = client.post(
        "/api/segmentation-workspaces",
        files={"source_image": ("office.png", _source_png(), "image/png")},
    ).json()
    workspace = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "chair",
            "display_name": "Chair",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    ).json()
    item = workspace["objects"][0]
    predicted = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/predict",
        json={"prompt_revision": 1, "points": [{"x": 1, "y": 1, "label": 1}]},
    )
    assert predicted.status_code == 200, predicted.text
    workspace = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}").json()
    return workspace, workspace["objects"][0]


def _save_manual(client: TestClient, workspace: dict, item: dict, edited: np.ndarray):
    return client.put(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/manual-mask",
        data={
            "base_prompt_revision": str(item["sam_draft"]["prompt_revision"]),
            "base_candidate_index": str(item["sam_draft"]["selected_candidate_index"]),
            "expected_workspace_revision": str(workspace["workspace_revision"]),
            "expected_object_version": str(item["object_version"]),
            "expected_manual_revision": str(item["manual_mask"]["manual_revision"]),
        },
        files={"edited_mask": ("edited.png", _mask_png(edited), "image/png")},
    )


def test_multipart_manual_save_artifacts_and_clear(client: TestClient, tmp_path: Path) -> None:
    workspace, item = _workspace_with_prediction(client)
    base = np.zeros((8, 10), dtype=bool)
    base[1:4, 1:5] = True
    edited = base.copy()
    edited[0, 0] = True
    edited[1, 1] = False

    saved = _save_manual(client, workspace, item, edited)

    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert set(body) == {"workspace_revision", "object_version", "object_id", "manual_mask"}
    assert body["object_id"] == item["object_id"]
    assert body["manual_mask"]["manual_revision"] == 1
    assert body["manual_mask"]["area_pixels"] == int(edited.sum())
    serialized = json.dumps(body)
    assert "file://" not in serialized
    assert str(tmp_path) not in serialized
    assert "base64" not in serialized

    artifact = client.get(body["manual_mask"]["composite_mask_url"])
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "image/png"
    assert artifact.headers["cache-control"] == "no-store"
    assert artifact.headers["x-content-type-options"] == "nosniff"

    cleared = client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/manual-mask",
        params={
            "expected_workspace_revision": body["workspace_revision"],
            "expected_object_version": body["object_version"],
            "expected_manual_revision": body["manual_mask"]["manual_revision"],
        },
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["manual_mask"]["manual_revision"] == 0
    assert client.get(body["manual_mask"]["composite_mask_url"]).status_code == 404


def test_manual_api_conflicts_missing_and_invalid_uploads(client: TestClient) -> None:
    workspace, item = _workspace_with_prediction(client)
    edited = np.zeros((8, 10), dtype=bool)

    missing = "11111111111141118111111111111111"
    assert client.put(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{missing}/manual-mask",
        data={
            "base_prompt_revision": "1",
            "base_candidate_index": "0",
            "expected_workspace_revision": str(workspace["workspace_revision"]),
            "expected_object_version": "1",
            "expected_manual_revision": "0",
        },
        files={"edited_mask": ("edited.png", _mask_png(edited), "image/png")},
    ).status_code == 404

    stale = client.put(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/manual-mask",
        data={
            "base_prompt_revision": "1",
            "base_candidate_index": "0",
            "expected_workspace_revision": "1",
            "expected_object_version": str(item["object_version"]),
            "expected_manual_revision": "0",
        },
        files={"edited_mask": ("edited.png", _mask_png(edited), "image/png")},
    )
    assert stale.status_code == 409

    bad_png = client.put(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/manual-mask",
        data={
            "base_prompt_revision": "1",
            "base_candidate_index": "0",
            "expected_workspace_revision": str(workspace["workspace_revision"]),
            "expected_object_version": str(item["object_version"]),
            "expected_manual_revision": "0",
        },
        files={"edited_mask": ("edited.png", b"not a png", "image/png")},
    )
    assert bad_png.status_code == 400

    rgb = io.BytesIO()
    Image.new("RGB", (10, 8), (255, 0, 0)).save(rgb, format="PNG")
    bad_mode = client.put(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/manual-mask",
        data={
            "base_prompt_revision": "1",
            "base_candidate_index": "0",
            "expected_workspace_revision": str(workspace["workspace_revision"]),
            "expected_object_version": str(item["object_version"]),
            "expected_manual_revision": "0",
        },
        files={"edited_mask": ("edited.png", rgb.getvalue(), "image/png")},
    )
    assert bad_mode.status_code == 415


def test_manual_api_blocks_sam_changes_and_sam_clear_clears_manual(client: TestClient) -> None:
    workspace, item = _workspace_with_prediction(client)
    edited = np.zeros((8, 10), dtype=bool)
    edited[1:4, 1:5] = True
    saved = _save_manual(client, workspace, item, edited).json()

    blocked_prediction = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/predict",
        json={"prompt_revision": 2, "points": [{"x": 2, "y": 2, "label": 1}]},
    )
    assert blocked_prediction.status_code == 409

    blocked_select = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/select-candidate",
        json={
            "prompt_revision": 1,
            "candidate_index": 1,
            "expected_workspace_revision": saved["workspace_revision"],
            "expected_object_version": saved["object_version"],
        },
    )
    assert blocked_select.status_code == 409

    cleared_sam = client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}/sam-draft",
        params={
            "expected_workspace_revision": saved["workspace_revision"],
            "expected_object_version": saved["object_version"],
        },
    )
    assert cleared_sam.status_code == 200, cleared_sam.text
    body = cleared_sam.json()
    assert body["sam_draft"]["prompt_revision"] == 0
    current = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}").json()["objects"][0]
    assert current["manual_mask"]["manual_revision"] == 0
    assert client.get(saved["manual_mask"]["composite_mask_url"]).status_code == 404
