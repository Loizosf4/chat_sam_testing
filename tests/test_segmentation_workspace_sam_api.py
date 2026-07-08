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
        count = 1 if kwargs.get("multimask_output") is False else 3
        masks = np.zeros((count, 8, 10), dtype=bool)
        for index in range(count):
            masks[index, index : index + 2, index : index + 3] = True
        return sam_engine.CandidatePrediction(
            image_key=args[0],
            width=10,
            height=8,
            masks=masks,
            scores=[0.2, 0.9, 0.4][:count],
            areas=[int(mask.sum()) for mask in masks],
            bboxes=[[index, index, index + 2, index + 1] for index in range(count)],
            logits=np.ones((count, 4, 4), dtype=np.float32),
            elapsed_ms=2.0,
        )

    monkeypatch.setattr(service.sam_engine, "predict_candidates", predict_candidates)
    return TestClient(app)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (10, 8), (200, 40, 90)).save(output, format="PNG")
    return output.getvalue()


def _workspace_with_object(client: TestClient) -> tuple[dict, dict]:
    workspace = client.post(
        "/api/segmentation-workspaces",
        files={"source_image": ("office.png", _png_bytes(), "image/png")},
    ).json()
    workspace = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "chair",
            "display_name": "Chair",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    ).json()
    return workspace, workspace["objects"][0]


def test_prepare_predict_artifact_select_and_clear_api(client: TestClient) -> None:
    workspace, item = _workspace_with_object(client)
    workspace_id = workspace["workspace_id"]
    object_id = item["object_id"]

    prepared = client.post(f"/api/segmentation-workspaces/{workspace_id}/prepare-sam")
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["sam_ready"] is True
    assert "checkpoint" not in json.dumps(prepared.json()).lower()

    predicted = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict",
        json={
            "prompt_revision": 1,
            "points": [{"x": 1, "y": 1, "label": 1}, {"x": 5, "y": 2, "label": 0}],
            "box": None,
        },
    )
    assert predicted.status_code == 200, predicted.text
    body = predicted.json()
    assert body["selected_candidate_index"] == 1
    assert len(body["candidates"]) == 3
    assert "png_base64" not in json.dumps(body)

    artifact = client.get(body["candidates"][1]["mask_url"])
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "image/png"
    assert artifact.headers["cache-control"] == "no-store"

    selected = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/select-candidate",
        json={
            "prompt_revision": 1,
            "candidate_index": 2,
            "expected_workspace_revision": body["workspace_revision"],
            "expected_object_version": body["object_version"],
        },
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["selected_candidate_index"] == 2

    cleared = client.delete(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/sam-draft",
        params={
            "expected_workspace_revision": selected.json()["workspace_revision"],
            "expected_object_version": selected.json()["object_version"],
        },
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["sam_draft"]["prompt_revision"] == 0
    assert client.get(body["candidates"][1]["mask_url"]).status_code == 404


def test_api_missing_invalid_and_stale_prompt_behaviour(client: TestClient) -> None:
    workspace, item = _workspace_with_object(client)
    workspace_id = workspace["workspace_id"]
    object_id = item["object_id"]
    missing = "11111111111141118111111111111111"

    assert client.post(f"/api/segmentation-workspaces/{missing}/prepare-sam").status_code == 404
    assert client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{missing}/predict",
        json={"prompt_revision": 1, "points": [{"x": 1, "y": 1, "label": 1}]},
    ).status_code == 404

    invalid = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict",
        json={"prompt_revision": 1, "points": [{"x": 99, "y": 1, "label": 1}]},
    )
    assert invalid.status_code == 400

    invalid_label = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict",
        json={"prompt_revision": 1, "points": [{"x": 1, "y": 1, "label": 2}]},
    )
    assert invalid_label.status_code == 422

    first = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict",
        json={"prompt_revision": 1, "points": [{"x": 1, "y": 1, "label": 1}]},
    )
    assert first.status_code == 200
    stale = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/predict",
        json={"prompt_revision": 1, "points": [{"x": 2, "y": 2, "label": 1}]},
    )
    assert stale.status_code == 409

    bad_select = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{object_id}/select-candidate",
        json={
            "prompt_revision": 1,
            "candidate_index": 9,
            "expected_workspace_revision": first.json()["workspace_revision"],
            "expected_object_version": first.json()["object_version"],
        },
    )
    assert bad_select.status_code == 404


def test_deleting_object_clears_only_that_object_logits(client: TestClient) -> None:
    workspace, item = _workspace_with_object(client)
    workspace_id = workspace["workspace_id"]
    other = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/objects",
        json={
            "semantic_label": "desk",
            "display_name": "Desk",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    ).json()["objects"][1]
    current = client.get(f"/api/segmentation-workspaces/{workspace_id}").json()
    service.LOGITS_CACHE.replace_object_revision(workspace_id, item["object_id"], 1, ["item-logits"])
    service.LOGITS_CACHE.replace_object_revision(workspace_id, other["object_id"], 1, ["other-logits"])

    deleted = client.delete(
        f"/api/segmentation-workspaces/{workspace_id}/objects/{item['object_id']}",
        params={
            "expected_workspace_revision": current["workspace_revision"],
            "expected_object_version": item["object_version"],
        },
    )

    assert deleted.status_code == 200, deleted.text
    assert service.LOGITS_CACHE.get(workspace_id, item["object_id"], 1, 0) is None
    assert service.LOGITS_CACHE.get(workspace_id, other["object_id"], 1, 0) == "other-logits"


def test_deleting_workspace_clears_workspace_logits_only(client: TestClient) -> None:
    workspace, item = _workspace_with_object(client)
    other_workspace, other_item = _workspace_with_object(client)
    service.LOGITS_CACHE.replace_object_revision(workspace["workspace_id"], item["object_id"], 1, ["item-logits"])
    service.LOGITS_CACHE.replace_object_revision(
        other_workspace["workspace_id"],
        other_item["object_id"],
        1,
        ["other-workspace-logits"],
    )

    deleted = client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}",
        params={"expected_workspace_revision": workspace["workspace_revision"]},
    )

    assert deleted.status_code == 200, deleted.text
    assert service.LOGITS_CACHE.get(workspace["workspace_id"], item["object_id"], 1, 0) is None
    assert (
        service.LOGITS_CACHE.get(other_workspace["workspace_id"], other_item["object_id"], 1, 0)
        == "other-workspace-logits"
    )
