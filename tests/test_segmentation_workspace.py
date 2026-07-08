from __future__ import annotations

import io
import json
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image, features
from pydantic import ValidationError

from backend.segmentation_workspace.models import SegmentationWorkspace
from backend.segmentation_workspace.store import (
    SegmentationWorkspaceStore,
    SegmentationWorkspaceStoreError,
)


def _image(path: Path, image_format: str = "PNG", size: tuple[int, int] = (8, 6)) -> Path:
    image = Image.new("RGB", size, (24, 80, 160))
    image.save(path, format=image_format)
    return path


def _store(tmp_path: Path) -> SegmentationWorkspaceStore:
    return SegmentationWorkspaceStore(tmp_path / "workspaces")


def test_create_workspace_from_valid_png_and_reload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")

    assert workspace.schema_version == "1.0.0"
    assert workspace.workspace_revision == 1
    assert workspace.status == "draft"
    assert workspace.image_dimensions.width == 8
    assert workspace.image_dimensions.height == 6
    assert workspace.source_image.media_type == "image/png"
    assert workspace.source_image.url.startswith("/api/segmentation-artifacts/source-images/")

    reloaded = store.get_workspace(workspace.workspace_id)
    assert reloaded.model_dump(mode="json") == workspace.model_dump(mode="json")


def test_create_workspace_from_valid_jpeg_and_webp_if_supported(tmp_path: Path) -> None:
    store = _store(tmp_path)
    jpeg = store.create_workspace(_image(tmp_path / "source.jpg", "JPEG"), "camera-upload.jpeg")
    assert jpeg.source_image.media_type == "image/jpeg"

    if not features.check("webp"):
        pytest.skip("Pillow WebP support is not available")
    webp = store.create_workspace(_image(tmp_path / "source.webp", "WEBP"), "source.webp")
    assert webp.source_image.media_type == "image/webp"


def test_reject_unsupported_or_invalid_image_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    gif = _image(tmp_path / "source.gif", "GIF")
    with pytest.raises(SegmentationWorkspaceStoreError, match="unsupported image type"):
        store.create_workspace(gif, "source.gif")

    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(SegmentationWorkspaceStoreError, match="invalid image file"):
        store.create_workspace(invalid, "invalid.png")


def test_source_artifact_resolves_and_browser_data_exposes_no_paths(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    identifier = workspace.source_image.url.rsplit("/", 1)[-1]

    artifact_path, media_type = store.resolve_artifact("source-images", identifier)

    assert media_type == "image/png"
    assert artifact_path.name == "source.png"
    serialized = json.dumps(workspace.model_dump(mode="json"))
    assert str(tmp_path) not in serialized
    assert "file://" not in serialized
    assert "source/source" not in serialized


def test_object_create_update_delete_revision_and_identity_rules(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    created = store.create_object(
        workspace.workspace_id,
        "office_chair",
        "Office Chair",
        expected_workspace_revision=workspace.workspace_revision,
    )
    item = created.objects[0]
    UUID(item.object_id)
    assert item.object_version == 1
    assert item.status == "draft"
    assert created.workspace_revision == 2

    updated = store.update_object(
        workspace.workspace_id,
        item.object_id,
        "desk_chair",
        "Desk Chair",
        expected_workspace_revision=created.workspace_revision,
        expected_object_version=item.object_version,
    )
    updated_item = updated.objects[0]
    assert updated_item.object_id == item.object_id
    assert updated_item.semantic_label == "desk_chair"
    assert updated_item.object_version == 2
    assert updated.workspace_revision == 3

    second = store.create_object(
        workspace.workspace_id,
        "monitor",
        "Monitor",
        expected_workspace_revision=updated.workspace_revision,
    )
    remaining_ids = [obj.object_id for obj in second.objects if obj.object_id != item.object_id]
    deleted = store.delete_object(
        workspace.workspace_id,
        item.object_id,
        expected_workspace_revision=second.workspace_revision,
        expected_object_version=updated_item.object_version,
    )
    assert [obj.object_id for obj in deleted.objects] == remaining_ids
    assert deleted.workspace_revision == 5


def test_duplicate_object_ids_are_rejected_by_model(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = store.create_object(workspace.workspace_id, "chair", "Chair", 1)
    document = workspace.model_dump(mode="json")
    document["objects"].append(dict(document["objects"][0]))

    with pytest.raises(ValidationError, match="duplicate object IDs"):
        SegmentationWorkspace.model_validate(document)


def test_stale_workspace_and_object_versions_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    updated = store.create_object(workspace.workspace_id, "chair", "Chair", 1)
    item = updated.objects[0]

    with pytest.raises(SegmentationWorkspaceStoreError, match="workspace revision conflict"):
        store.create_object(workspace.workspace_id, "desk", "Desk", 1)

    with pytest.raises(SegmentationWorkspaceStoreError, match="object version conflict"):
        store.update_object(
            workspace.workspace_id,
            item.object_id,
            "desk_chair",
            "Desk Chair",
            expected_workspace_revision=updated.workspace_revision,
            expected_object_version=item.object_version + 1,
        )


def test_path_traversal_unsafe_ids_and_artifact_hash_mismatch_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")

    with pytest.raises(SegmentationWorkspaceStoreError, match="unsafe workspace ID"):
        store.get_workspace("../workspace")
    with pytest.raises(SegmentationWorkspaceStoreError, match="invalid artifact identifier"):
        store.resolve_artifact("source-images", "../source")

    identifier = workspace.source_image.url.rsplit("/", 1)[-1]
    artifact_path, _ = store.resolve_artifact("source-images", identifier)
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact integrity check failed"):
        store.resolve_artifact("source-images", identifier)


def test_delete_workspace_removes_only_managed_directory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_workspace(_image(tmp_path / "first.png"), "first.png")
    second = store.create_workspace(_image(tmp_path / "second.png"), "second.png")
    first_dir = store.root / first.workspace_id
    second_dir = store.root / second.workspace_id

    store.delete_workspace(first.workspace_id, expected_workspace_revision=first.workspace_revision)

    assert not first_dir.exists()
    assert second_dir.is_dir()
    assert store.get_workspace(second.workspace_id).workspace_id == second.workspace_id
