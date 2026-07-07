from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.scene_package import api
from backend.scene_package.store import SceneStore


ROOT = Path(__file__).resolve().parents[1]
SAM_DIR = ROOT / "data" / "exports" / "auto_scene_final_24457ea2"
UNIFIED = ROOT / "experiments" / "moge_scene_graph_spike" / "outputs" / "office_test" / "unified_v3_1_1_clean" / "unified_scene_plan.json"
SOURCE = ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "STORE", SceneStore(tmp_path / "packages"))
    return TestClient(app)


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in SAM_DIR.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(SAM_DIR).as_posix())
    return output.getvalue()


def _import(client: TestClient) -> dict:
    if not (SAM_DIR.is_dir() and UNIFIED.is_file() and SOURCE.is_file()):
        pytest.skip("office source fixture is not present")
    response = client.post(
        "/api/scenes/import",
        files={
            "sam_export": ("sam.zip", _archive(), "application/zip"),
            "unified_manifest": ("unified.json", UNIFIED.read_bytes(), "application/json"),
            "source_image": ("office.jpg", SOURCE.read_bytes(), "image/jpeg"),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_office_import_returns_browser_safe_package_and_artifacts(client: TestClient) -> None:
    scene = _import(client)
    assert len(scene["semantic_objects"]) == 20
    serialized = json.dumps(scene)
    assert "C:\\" not in serialized and "file://" not in serialized

    assert client.get(scene["source_image"]["url"]).status_code == 200
    item = scene["semantic_objects"][0]
    mask = client.get(item["mask_url"])
    assert mask.status_code == 200
    assert mask.headers["content-type"] == "image/png"
    assert client.get(item["overlay_url"]).status_code == 200
    assert client.get("/artifacts/masks/../../scene.json").status_code in {400, 404}


def test_label_approval_revision_history_and_stale_write(client: TestClient) -> None:
    scene = _import(client)
    scene_id = scene["scene_id"]
    item = scene["semantic_objects"][0]
    label = client.patch(
        f"/api/scenes/{scene_id}/objects/{item['object_id']}/label",
        json={
            "semantic_label": "notice_board",
            "expected_package_revision": scene["package_revision"],
            "expected_object_version": item["version"],
        },
    )
    assert label.status_code == 200
    updated = label.json()
    updated_item = next(obj for obj in updated["semantic_objects"] if obj["object_id"] == item["object_id"])
    assert updated_item["semantic_label"] == "notice_board"

    stale = client.patch(
        f"/api/scenes/{scene_id}/objects/{item['object_id']}/approval",
        json={
            "approval_status": "approved",
            "expected_package_revision": scene["package_revision"],
            "expected_object_version": item["version"],
        },
    )
    assert stale.status_code == 409

    current_mask = client.get(updated_item["mask_url"]).content
    edited_image = Image.open(io.BytesIO(current_mask)).convert("L")
    edited_image.putpixel((0, 0), 255 if edited_image.getpixel((0, 0)) == 0 else 0)
    edited_mask = io.BytesIO()
    edited_image.save(edited_mask, format="PNG")
    revision = client.post(
        f"/api/scenes/{scene_id}/objects/{item['object_id']}/mask-revisions",
        data={
            "expected_package_revision": updated["package_revision"],
            "expected_object_version": updated_item["version"],
            "expected_mask_revision": updated_item["mask_revision"],
            "author": "integration-test",
            "operation": "manual",
        },
        files={
            "resulting_mask": ("mask.png", edited_mask.getvalue(), "image/png"),
            "edit_delta": ("delta.json", b'{"strokes": []}', "application/json"),
        },
    )
    assert revision.status_code == 201, revision.text
    revised = revision.json()
    revised_item = next(obj for obj in revised["semantic_objects"] if obj["object_id"] == item["object_id"])
    assert revised_item["mask_revision"] != item["mask_revision"]
    assert revised_item["approval_status"] == "needs_revision"
    assert revised["generation_status"]["stages"]["primitive_transforms"] == "stale"

    history = client.get(f"/api/scenes/{scene_id}/objects/{item['object_id']}/mask-revisions")
    assert [entry["revision_number"] for entry in history.json()["revisions"]] == [1, 2]
    old_mask = client.get(item["mask_url"])
    assert old_mask.status_code == 200
    assert old_mask.content == current_mask


def test_transform_import_clears_object_geometry_staleness(client: TestClient) -> None:
    scene = _import(client)
    item = scene["semantic_objects"][0]
    response = client.put(
        f"/api/scenes/{scene['scene_id']}/scene-manifest",
        json={
            "expected_package_revision": scene["package_revision"],
            "transforms": [{
                "object_id": item["object_id"],
                "expected_object_version": item["version"],
                "expected_mask_revision": item["mask_revision"],
                "transform": {
                    "center": item["center"],
                    "dimensions": item["dimensions"],
                    "quaternion": item["quaternion"],
                },
                "blender_object_identifier": "Office.NoticeBoard",
            }],
        },
    )
    assert response.status_code == 200, response.text
    status = client.get(f"/api/scenes/{scene['scene_id']}/reconstruction-status")
    assert status.status_code == 200
    assert item["object_id"] not in status.json()["stale_object_ids"]


def test_canvas_rgba_binary_mask_is_normalized_to_grayscale(client: TestClient) -> None:
    scene = _import(client)
    item = scene["semantic_objects"][0]
    current = Image.open(io.BytesIO(client.get(item["mask_url"]).content)).convert("RGBA")
    red, green, blue, alpha = current.getpixel((0, 0))
    value = 0 if red else 255
    current.putpixel((0, 0), (value, value, value, alpha))
    encoded = io.BytesIO()
    current.save(encoded, format="PNG")

    response = client.post(
        f"/api/scenes/{scene['scene_id']}/objects/{item['object_id']}/mask-revisions",
        data={
            "expected_package_revision": scene["package_revision"],
            "expected_object_version": item["version"],
            "expected_mask_revision": item["mask_revision"],
            "author": "canvas-regression-test",
            "operation": "manual",
        },
        files={
            "resulting_mask": ("mask.png", encoded.getvalue(), "image/png"),
            "edit_delta": ("delta.json", b'{"strokes": []}', "application/json"),
        },
    )
    assert response.status_code == 201, response.text
    revised = next(obj for obj in response.json()["semantic_objects"] if obj["object_id"] == item["object_id"])
    persisted = Image.open(io.BytesIO(client.get(revised["mask_url"]).content))
    assert persisted.mode == "L"


def test_import_rejects_zip_traversal(client: TestClient) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("../metadata.json", "{}")
    response = client.post(
        "/api/scenes/import",
        files={
            "sam_export": ("sam.zip", output.getvalue(), "application/zip"),
            "unified_manifest": ("unified.json", b"{}", "application/json"),
            "source_image": ("office.jpg", SOURCE.read_bytes(), "image/jpeg"),
        },
    )
    assert response.status_code == 400
