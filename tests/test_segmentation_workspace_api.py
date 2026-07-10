from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.segmentation_workspace import api
from backend.segmentation_workspace.store import SegmentationWorkspaceStore


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "STORE", SegmentationWorkspaceStore(tmp_path / "workspaces"))
    return TestClient(app)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (10, 7), (200, 40, 90)).save(output, format="PNG")
    return output.getvalue()


def _create_workspace(client: TestClient) -> dict:
    response = client.post(
        "/api/segmentation-workspaces",
        files={"source_image": ("office.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _assert_browser_safe(payload: dict, tmp_path: Path | None = None) -> None:
    serialized = json.dumps(payload)
    assert "file://" not in serialized
    assert "source/source" not in serialized
    if tmp_path is not None:
        assert str(tmp_path) not in serialized


def test_create_retrieve_and_source_image_artifact(client: TestClient, tmp_path: Path) -> None:
    workspace = _create_workspace(client)
    _assert_browser_safe(workspace, tmp_path)
    assert workspace["workspace_revision"] == 1
    assert workspace["source_image"]["media_type"] == "image/png"

    fetched = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == workspace

    image = client.get(workspace["source_image"]["url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_object_create_update_delete_and_workspace_delete(client: TestClient, tmp_path: Path) -> None:
    workspace = _create_workspace(client)
    created = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "office_chair",
            "display_name": "Office Chair",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    )
    assert created.status_code == 201, created.text
    with_object = created.json()
    item = with_object["objects"][0]
    assert with_object["workspace_revision"] == 2

    renamed = client.patch(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}",
        json={
            "semantic_label": "desk_chair",
            "display_name": "Desk Chair",
            "expected_workspace_revision": with_object["workspace_revision"],
            "expected_object_version": item["object_version"],
        },
    )
    assert renamed.status_code == 200, renamed.text
    updated = renamed.json()
    updated_item = updated["objects"][0]
    assert updated_item["object_id"] == item["object_id"]
    assert updated_item["object_version"] == 2
    assert updated["workspace_revision"] == 3

    deleted_object = client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}",
        params={
            "expected_workspace_revision": updated["workspace_revision"],
            "expected_object_version": updated_item["object_version"],
        },
    )
    assert deleted_object.status_code == 200, deleted_object.text
    without_object = deleted_object.json()
    assert without_object["objects"] == []
    assert without_object["workspace_revision"] == 4
    _assert_browser_safe(without_object, tmp_path)

    deleted_workspace = client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}",
        params={"expected_workspace_revision": without_object["workspace_revision"]},
    )
    assert deleted_workspace.status_code == 200
    assert deleted_workspace.json()["status"] == "deleted"
    assert client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}").status_code == 404


def test_api_missing_resources_and_stale_revisions(client: TestClient) -> None:
    missing = "11111111111141118111111111111111"
    assert client.get(f"/api/segmentation-workspaces/{missing}").status_code == 404

    workspace = _create_workspace(client)
    created = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "chair",
            "display_name": "Chair",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    ).json()
    item = created["objects"][0]

    stale_workspace = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "desk",
            "display_name": "Desk",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    )
    assert stale_workspace.status_code == 409

    stale_object = client.patch(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}",
        json={
            "semantic_label": "desk_chair",
            "display_name": "Desk Chair",
            "expected_workspace_revision": created["workspace_revision"],
            "expected_object_version": item["object_version"] + 1,
        },
    )
    assert stale_object.status_code == 409

    assert client.delete(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{missing}",
        params={
            "expected_workspace_revision": created["workspace_revision"],
            "expected_object_version": 1,
        },
    ).status_code == 404


def test_api_rejects_invalid_upload_and_does_not_expose_paths(client: TestClient, tmp_path: Path) -> None:
    invalid = client.post(
        "/api/segmentation-workspaces",
        files={"source_image": ("fake.png", b"not an image", "image/png")},
    )
    assert invalid.status_code == 400

    workspace = _create_workspace(client)
    _assert_browser_safe(workspace, tmp_path)
    traversal = client.get("/api/segmentation-artifacts/source-images/../../workspace.json")
    assert traversal.status_code in {400, 404}
