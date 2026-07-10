from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from backend.scene_package.adapter import adapt_scene_package
from backend.segmentation_workspace import exports as export_service
from backend.segmentation_workspace.models import SegmentationExportMask, SegmentationWorkspace
from backend.segmentation_workspace.store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


def _image(path: Path, size: tuple[int, int] = (10, 8)) -> Path:
    Image.new("RGB", size, (30, 120, 200)).save(path, format="PNG")
    return path


def _store_with_masks(tmp_path: Path) -> tuple[SegmentationWorkspaceStore, str]:
    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = store.create_object(workspace.workspace_id, "chair", "Chair", workspace.workspace_revision)
    workspace = store.create_object(workspace.workspace_id, "chair", "Second Chair", workspace.workspace_revision)
    first, second = workspace.objects
    first_mask = np.zeros((8, 10), dtype=bool)
    first_mask[1:4, 1:5] = True
    second_mask = np.zeros((8, 10), dtype=bool)
    second_mask[4:7, 5:9] = True
    store.persist_sam_prediction(
        workspace.workspace_id,
        first.object_id,
        prompt_revision=1,
        points=[{"x": 1, "y": 1, "label": 1}],
        box=None,
        candidate_masks=[first_mask],
        candidate_scores=[0.9],
        candidate_areas=[0],
        candidate_bboxes=[[0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    store.persist_sam_prediction(
        workspace.workspace_id,
        second.object_id,
        prompt_revision=1,
        points=[{"x": 6, "y": 5, "label": 1}],
        box=None,
        candidate_masks=[second_mask],
        candidate_scores=[0.8],
        candidate_areas=[0],
        candidate_bboxes=[[0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    return store, workspace.workspace_id


def _expected_objects(workspace: SegmentationWorkspace) -> list[dict[str, object]]:
    return [
        {"object_id": item.object_id, "expected_object_version": item.object_version}
        for item in workspace.objects
    ]


def _create_export(store: SegmentationWorkspaceStore, workspace_id: str):
    workspace = store.get_workspace(workspace_id)
    return store.create_export(
        workspace_id,
        expected_workspace_revision=workspace.workspace_revision,
        expected_objects=_expected_objects(workspace),
    )


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) == 255


def _minimal_unified(scene_id: str, object_ids: list[str]) -> dict:
    return {
        "mode": "clean_reconstruction",
        "scene_id": scene_id,
        "semantic_object_count": len(object_ids),
        "provisional_camera_id": "camera_main",
        "coordinate_system": {"absolute_scale_verified": False},
        "camera_candidates": [
            {
                "camera_id": "camera_main",
                "type": "perspective",
                "matrix_world": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "confidence": 1.0,
                "provisional": True,
            }
        ],
        "room_proxies": [],
        "semantic_objects": [
            {
                "object_id": object_id,
                "semantic_label": f"object_{index}",
                "transform": {
                    "center": [float(index), 0.0, 0.0],
                    "dimensions": [1.0, 1.0, 1.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "geometry_confidence": 1.0,
                "final_pose_confidence": 1.0,
                "support_type": "unknown",
            }
            for index, object_id in enumerate(object_ids)
        ],
    }


def test_existing_workspace_json_without_exports_still_loads(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    path = store.root / workspace_id / "workspace.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("exports", None)
    path.write_text(json.dumps(document), encoding="utf-8")

    reloaded = store.get_workspace(workspace_id)

    assert reloaded.exports == []


def test_export_model_rejects_filesystem_urls() -> None:
    with pytest.raises(ValidationError, match="filesystem"):
        SegmentationExportMask(
            object_id="11111111111141118111111111111111",
            object_version=1,
            semantic_label="chair",
            display_name="Chair",
            source_kind="sam_candidate",
            source_prompt_revision=1,
            source_candidate_index=0,
            source_manual_revision=0,
            filename="chair.png",
            mask_url=r"C:\temp\chair.png",
            preview_url="/api/segmentation-artifacts/workspace-exports/w/e/previews/o",
            mask_sha256="0" * 64,
            area_pixels=1,
            bbox_xyxy=[0, 0, 0, 0],
        )


def test_create_export_writes_metadata_quality_previews_zip_and_record(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    before = store.get_workspace(workspace_id)

    record = _create_export(store, workspace_id)
    updated = store.get_workspace(workspace_id)

    assert updated.workspace_revision == before.workspace_revision + 1
    assert [item.object_version for item in updated.objects] == [item.object_version for item in before.objects]
    assert updated.objects[0].sam_draft == before.objects[0].sam_draft
    assert len(updated.exports) == 1
    assert updated.exports[0].export_id == record.export_id
    assert record.created_from_workspace_revision == before.workspace_revision
    assert record.published_workspace_revision == updated.workspace_revision
    assert record.mask_count == 2
    assert record.masks[0].object_id == before.objects[0].object_id
    assert record.masks[0].source_kind == "sam_candidate"
    assert record.masks[0].mask_sha256
    assert len({mask.filename.lower() for mask in record.masks}) == 2

    export_dir = store.root / workspace_id / "exports" / record.export_id
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["masks"][0]["mask_id"] == before.objects[0].object_id
    assert metadata["masks"][0]["filename"] == record.masks[0].filename
    assert metadata["masks"][0]["area"] == int(_read_mask(export_dir / record.masks[0].filename).sum())
    assert (export_dir / "mask_quality_report.json").is_file()
    assert (export_dir / "mask_quality_report.md").is_file()
    assert (export_dir / "previews" / "all_masks_overlay.png").is_file()
    with Image.open(export_dir / "previews" / "all_masks_overlay.png") as preview:
        assert preview.size == (10, 8)
    with zipfile.ZipFile(export_dir / "segmentation-export.zip") as archive:
        names = set(archive.namelist())
    assert "metadata.json" in names
    assert "mask_quality_report.json" in names
    assert "mask_quality_report.md" in names
    assert "previews/all_masks_overlay.png" in names
    assert "segmentation-export.zip" not in names
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
    assert str(tmp_path) not in json.dumps(record.model_dump(mode="json"))


def test_manual_composite_takes_priority_and_base_mismatch_is_rejected(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    workspace = store.get_workspace(workspace_id)
    item = workspace.objects[0]
    edited = np.zeros((8, 10), dtype=bool)
    edited[0:2, 0:2] = True
    saved = store.save_manual_mask(
        workspace_id,
        item.object_id,
        edited_mask=edited,
        base_prompt_revision=item.sam_draft.prompt_revision,
        base_candidate_index=item.sam_draft.selected_candidate_index or 0,
        expected_workspace_revision=workspace.workspace_revision,
        expected_object_version=item.object_version,
        expected_manual_revision=0,
    )

    record = _create_export(store, workspace_id)
    exported = next(mask for mask in record.masks if mask.object_id == item.object_id)
    assert exported.source_kind == "manual_composite"
    assert exported.source_manual_revision == 1
    export_dir = store.root / workspace_id / "exports" / record.export_id
    np.testing.assert_array_equal(_read_mask(export_dir / exported.filename), edited)

    document = saved.model_dump(mode="json")
    document["objects"][0]["manual_mask"]["base_prompt_revision"] = 99
    (store.root / workspace_id / "workspace.json").write_text(json.dumps(document), encoding="utf-8")
    broken = store.get_workspace(workspace_id)
    with pytest.raises(SegmentationWorkspaceStoreError, match="base mismatch"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=broken.workspace_revision,
            expected_objects=_expected_objects(broken),
        )


def test_missing_selected_candidate_is_rejected_without_mutation(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    before = store.get_workspace(workspace_id)
    path = store.root / workspace_id / "workspace.json"
    document = before.model_dump(mode="json")
    document["objects"][0]["sam_draft"]["selected_candidate_index"] = None
    path.write_text(json.dumps(document), encoding="utf-8")
    broken = store.get_workspace(workspace_id)
    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))

    with pytest.raises(SegmentationWorkspaceStoreError, match="no selected"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=broken.workspace_revision,
            expected_objects=_expected_objects(broken),
        )

    assert store.get_workspace(workspace_id).workspace_revision == broken.workspace_revision
    assert json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8")) == index_before


def test_missing_effective_mask_artifact_is_rejected(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    workspace = store.get_workspace(workspace_id)
    candidate = workspace.objects[0].sam_draft.candidates[0]
    identifier = candidate.mask_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    index_path = store.root / workspace_id / "artifact-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop(f"sam-candidates/{identifier}")
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        _create_export(store, workspace_id)


def test_mask_hash_mismatch_wrong_dimensions_non_binary_and_empty_are_rejected(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    workspace = store.get_workspace(workspace_id)
    candidate = workspace.objects[0].sam_draft.candidates[0]
    identifier = candidate.mask_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    path, _ = store.resolve_artifact("sam-candidates", identifier)

    path.write_bytes(b"tampered")
    with pytest.raises(SegmentationWorkspaceStoreError, match="integrity"):
        _create_export(store, workspace_id)

    Image.new("L", (9, 8), 255).save(path, format="PNG")
    index_path = store.root / workspace_id / "artifact-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index[f"sam-candidates/{identifier}"]["sha256"] = export_service.sha256_file(path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(SegmentationWorkspaceStoreError, match="dimensions"):
        _create_export(store, workspace_id)

    Image.fromarray(np.full((8, 10), 128, dtype=np.uint8), mode="L").save(path, format="PNG")
    index[f"sam-candidates/{identifier}"]["sha256"] = export_service.sha256_file(path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(SegmentationWorkspaceStoreError, match="not binary"):
        _create_export(store, workspace_id)

    Image.new("L", (10, 8), 0).save(path, format="PNG")
    index[f"sam-candidates/{identifier}"]["sha256"] = export_service.sha256_file(path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(SegmentationWorkspaceStoreError, match="empty"):
        _create_export(store, workspace_id)


def test_export_metadata_is_scene_adapter_compatible(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    record = _create_export(store, workspace_id)
    export_dir = store.root / workspace_id / "exports" / record.export_id
    unified = _minimal_unified("segmentation_export_test", [mask.object_id for mask in record.masks])
    unified_path = tmp_path / "unified.json"
    unified_path.write_text(json.dumps(unified), encoding="utf-8")
    source_path, _ = store.resolve_artifact("source-images", store.get_workspace(workspace_id).source_image.image_id)

    scene = adapt_scene_package(export_dir / "metadata.json", unified_path, source_image_path=source_path)

    assert [item.object_id for item in scene.semantic_objects] == [mask.object_id for mask in record.masks]
    assert [revision.object_id for revision in scene.mask_revisions] == [mask.object_id for mask in record.masks]
    assert all((export_dir / item.mask_filename).is_file() for item in scene.semantic_objects)


def test_export_staleness_and_immutability_after_edits_and_deletion(tmp_path: Path) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    record = _create_export(store, workspace_id)
    archive_identifier = record.archive_url.split("/api/segmentation-artifacts/workspace-exports/", 1)[1]
    archive_path, _ = store.resolve_artifact("workspace-exports", archive_identifier)

    assert store.export_is_stale(workspace_id, record.export_id) is False
    second = _create_export(store, workspace_id)
    assert second.export_id != record.export_id
    assert store.export_is_stale(workspace_id, record.export_id) is False

    workspace = store.get_workspace(workspace_id)
    item = workspace.objects[0]
    store.update_object(
        workspace_id,
        item.object_id,
        "renamed_chair",
        "Renamed Chair",
        expected_workspace_revision=workspace.workspace_revision,
        expected_object_version=item.object_version,
    )
    assert store.export_is_stale(workspace_id, record.export_id) is True

    workspace = store.get_workspace(workspace_id)
    item = workspace.objects[0]
    store.delete_object(
        workspace_id,
        item.object_id,
        expected_workspace_revision=workspace.workspace_revision,
        expected_object_version=item.object_version,
    )
    assert archive_path.is_file()
    assert store.resolve_artifact("workspace-exports", archive_identifier)[0] == archive_path


def test_optimistic_concurrency_and_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, workspace_id = _store_with_masks(tmp_path)
    workspace = store.get_workspace(workspace_id)

    with pytest.raises(SegmentationWorkspaceStoreError, match="workspace revision conflict"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=workspace.workspace_revision - 1,
            expected_objects=_expected_objects(workspace),
        )
    with pytest.raises(SegmentationWorkspaceStoreError, match="object version conflict"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=workspace.workspace_revision,
            expected_objects=[
                {"object_id": item.object_id, "expected_object_version": item.object_version + 1}
                for item in workspace.objects
            ],
        )
    with pytest.raises(SegmentationWorkspaceStoreError, match="duplicate"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=workspace.workspace_revision,
            expected_objects=[_expected_objects(workspace)[0], _expected_objects(workspace)[0]],
        )

    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))
    workspace_before = store.get_workspace(workspace_id).model_dump(mode="json")

    def fail_write_index(_workspace_id, _index):
        raise RuntimeError("simulated index write failure")

    monkeypatch.setattr(store, "_write_index", fail_write_index)
    with pytest.raises(RuntimeError, match="simulated"):
        store.create_export(
            workspace_id,
            expected_workspace_revision=workspace.workspace_revision,
            expected_objects=_expected_objects(workspace),
        )

    assert store.get_workspace(workspace_id).model_dump(mode="json") == workspace_before
    assert json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8")) == index_before
    exports_dir = store.root / workspace_id / "exports"
    assert not exports_dir.exists() or not [path for path in exports_dir.iterdir() if not path.name.startswith(".")]
