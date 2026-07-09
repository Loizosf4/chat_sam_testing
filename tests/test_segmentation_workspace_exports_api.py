from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
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


def _png_bytes(size: tuple[int, int] = (10, 8)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (80, 120, 200)).save(output, format="PNG")
    return output.getvalue()


def _workspace_with_candidate(client: TestClient) -> dict:
    created = client.post(
        "/api/segmentation-workspaces",
        files={"source_image": ("source.png", _png_bytes(), "image/png")},
    )
    assert created.status_code == 201, created.text
    workspace = created.json()
    object_response = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects",
        json={
            "semantic_label": "chair",
            "display_name": "Chair",
            "expected_workspace_revision": workspace["workspace_revision"],
        },
    )
    assert object_response.status_code == 201, object_response.text
    workspace = object_response.json()
    item = workspace["objects"][0]
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:5, 2:7] = True
    api.STORE.persist_sam_prediction(
        workspace["workspace_id"],
        item["object_id"],
        prompt_revision=1,
        points=[{"x": 2, "y": 2, "label": 1}],
        box=None,
        candidate_masks=[mask],
        candidate_scores=[0.9],
        candidate_areas=[0],
        candidate_bboxes=[[0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    fetched = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}")
    assert fetched.status_code == 200
    return fetched.json()


def _expected_objects(workspace: dict) -> list[dict[str, object]]:
    return [
        {"object_id": item["object_id"], "expected_object_version": item["object_version"]}
        for item in workspace["objects"]
    ]


def test_create_list_read_and_download_export_artifacts(client: TestClient, tmp_path: Path) -> None:
    workspace = _workspace_with_candidate(client)
    response = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports",
        json={
            "expected_workspace_revision": workspace["workspace_revision"],
            "expected_objects": _expected_objects(workspace),
            "include_previews": True,
        },
    )
    assert response.status_code == 201, response.text
    record = response.json()
    assert record["is_stale"] is False
    assert record["mask_count"] == 1
    assert record["masks"][0]["object_id"] == workspace["objects"][0]["object_id"]
    assert "file://" not in json.dumps(record)
    assert str(tmp_path) not in json.dumps(record)
    assert "base64" not in json.dumps(record).lower()

    listed = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports")
    assert listed.status_code == 200
    assert listed.json()[0]["export_id"] == record["export_id"]
    fetched = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports/{record['export_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["archive_sha256"] == record["archive_sha256"]

    for url, content_type in [
        (record["metadata_url"], "application/json"),
        (record["quality_report_url"], "application/json"),
        (record["quality_markdown_url"], "text/markdown; charset=utf-8"),
        (record["combined_preview_url"], "image/png"),
        (record["archive_url"], "application/zip"),
        (record["masks"][0]["mask_url"], "image/png"),
        (record["masks"][0]["preview_url"], "image/png"),
    ]:
        artifact = client.get(url)
        assert artifact.status_code == 200, url
        assert artifact.headers["content-type"] == content_type
        assert artifact.headers["x-content-type-options"] == "nosniff"
        assert "immutable" in artifact.headers["cache-control"]

    archive = client.get(record["archive_url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        names = set(bundle.namelist())
    assert "metadata.json" in names
    assert "segmentation-export.zip" not in names


def test_export_api_conflicts_and_missing_resources(client: TestClient) -> None:
    workspace = _workspace_with_candidate(client)
    endpoint = f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports"

    stale = client.post(
        endpoint,
        json={
            "expected_workspace_revision": workspace["workspace_revision"] - 1,
            "expected_objects": _expected_objects(workspace),
        },
    )
    assert stale.status_code == 409

    stale_object = client.post(
        endpoint,
        json={
            "expected_workspace_revision": workspace["workspace_revision"],
            "expected_objects": [
                {
                    "object_id": workspace["objects"][0]["object_id"],
                    "expected_object_version": workspace["objects"][0]["object_version"] + 1,
                }
            ],
        },
    )
    assert stale_object.status_code == 409

    duplicate = client.post(
        endpoint,
        json={
            "expected_workspace_revision": workspace["workspace_revision"],
            "expected_objects": [_expected_objects(workspace)[0], _expected_objects(workspace)[0]],
        },
    )
    assert duplicate.status_code == 422

    missing_object = client.post(
        endpoint,
        json={
            "expected_workspace_revision": workspace["workspace_revision"],
            "expected_objects": [
                {"object_id": "11111111111141118111111111111111", "expected_object_version": 1}
            ],
        },
    )
    assert missing_object.status_code == 409
    assert client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports/11111111111141118111111111111111").status_code == 404
    assert client.get("/api/segmentation-artifacts/workspace-exports/../../workspace.json").status_code in {400, 404}


def test_export_api_reports_stale_after_label_change(client: TestClient) -> None:
    workspace = _workspace_with_candidate(client)
    created = client.post(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports",
        json={
            "expected_workspace_revision": workspace["workspace_revision"],
            "expected_objects": _expected_objects(workspace),
        },
    ).json()
    current = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}").json()
    item = current["objects"][0]
    renamed = client.patch(
        f"/api/segmentation-workspaces/{workspace['workspace_id']}/objects/{item['object_id']}",
        json={
            "semantic_label": "renamed_chair",
            "display_name": "Renamed Chair",
            "expected_workspace_revision": current["workspace_revision"],
            "expected_object_version": item["object_version"],
        },
    )
    assert renamed.status_code == 200

    fetched = client.get(f"/api/segmentation-workspaces/{workspace['workspace_id']}/exports/{created['export_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["is_stale"] is True
