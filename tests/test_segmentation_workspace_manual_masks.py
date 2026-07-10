from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from backend.segmentation_workspace import service
from backend.segmentation_workspace.manual_masks import decode_uploaded_manual_mask
from backend.segmentation_workspace.models import ManualMaskState, SamPromptPoint, SegmentationWorkspace
from backend.segmentation_workspace.store import (
    SegmentationWorkspaceStore,
    SegmentationWorkspaceStoreError,
    derive_manual_layers,
)


def _image(path: Path, size: tuple[int, int] = (10, 8)) -> Path:
    Image.new("RGB", size, (30, 120, 200)).save(path, format="PNG")
    return path


def _mask_png(path: Path, mask: np.ndarray, mode: str = "L") -> Path:
    Image.fromarray(mask.astype(np.uint8) * 255, mode=mode).save(path, format="PNG")
    return path


def _store_with_candidate(tmp_path: Path) -> tuple[SegmentationWorkspaceStore, str, str, np.ndarray]:
    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = store.create_object(workspace.workspace_id, "chair", "Chair", workspace.workspace_revision)
    base = np.zeros((8, 10), dtype=bool)
    base[1:4, 1:5] = True
    alternate = np.zeros((8, 10), dtype=bool)
    alternate[4:6, 4:8] = True
    workspace = store.persist_sam_prediction(
        workspace.workspace_id,
        workspace.objects[0].object_id,
        prompt_revision=1,
        points=[{"x": 1, "y": 1, "label": 1}],
        box=None,
        candidate_masks=[base, alternate],
        candidate_scores=[0.9, 0.2],
        candidate_areas=[0, 0],
        candidate_bboxes=[[0, 0, 0, 0], [0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    return store, workspace.workspace_id, workspace.objects[0].object_id, base


def _save_manual(
    store: SegmentationWorkspaceStore,
    workspace_id: str,
    object_id: str,
    edited: np.ndarray,
) -> SegmentationWorkspace:
    workspace = store.get_workspace(workspace_id)
    item = workspace.objects[0]
    return store.save_manual_mask(
        workspace_id,
        object_id,
        edited_mask=edited,
        base_prompt_revision=item.sam_draft.prompt_revision,
        base_candidate_index=item.sam_draft.selected_candidate_index or 0,
        expected_workspace_revision=workspace.workspace_revision,
        expected_object_version=item.object_version,
        expected_manual_revision=item.manual_mask.manual_revision,
    )


def _identifier(url: str, kind: str = "manual-masks") -> str:
    return url.split(f"/api/segmentation-artifacts/{kind}/", 1)[1]


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image) == 255


def test_existing_workspace_json_without_manual_state_still_loads(tmp_path: Path) -> None:
    store, workspace_id, _object_id, _base = _store_with_candidate(tmp_path)
    path = store.root / workspace_id / "workspace.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["objects"][0].pop("manual_mask")
    path.write_text(json.dumps(document), encoding="utf-8")

    reloaded = store.get_workspace(workspace_id)

    assert reloaded.objects[0].manual_mask.manual_revision == 0
    assert reloaded.objects[0].manual_mask.bbox_xyxy == [0, 0, 0, 0]


def test_manual_mask_model_rejects_partial_active_state_and_paths() -> None:
    with pytest.raises(ValidationError, match="missing"):
        ManualMaskState(manual_revision=1, base_prompt_revision=1, base_candidate_index=0)
    with pytest.raises(ValidationError, match="filesystem"):
        ManualMaskState(
            manual_revision=1,
            base_prompt_revision=1,
            base_candidate_index=0,
            add_mask_url="C:/temp/add.png",
            remove_mask_url="/api/segmentation-artifacts/manual-masks/w/o/1/remove",
            composite_mask_url="/api/segmentation-artifacts/manual-masks/w/o/1/composite",
            updated_at="2026-01-01T00:00:00Z",
        )


def test_layer_derivation_matches_formula() -> None:
    base = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    edited = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)

    add, remove, composite, area, bbox = derive_manual_layers(base, edited)

    np.testing.assert_array_equal(add, edited & ~base)
    np.testing.assert_array_equal(remove, base & ~edited)
    np.testing.assert_array_equal(composite, (base | add) & ~remove)
    assert not np.any(add & remove)
    assert area == 3
    assert bbox == [0, 0, 2, 1]


def test_save_manual_mask_persists_binary_layers_and_metadata(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    edited = base.copy()
    edited[0, 0] = True
    edited[1, 1] = False

    saved = _save_manual(store, workspace_id, object_id, edited)
    item = saved.objects[0]

    assert item.object_id == object_id
    assert item.manual_mask.manual_revision == 1
    assert item.manual_mask.base_prompt_revision == 1
    assert item.manual_mask.base_candidate_index == 0
    assert saved.workspace_revision == 4
    assert item.object_version == 3
    assert item.manual_mask.area_pixels == int(edited.sum())
    assert item.manual_mask.bbox_xyxy == [0, 0, 4, 3]
    assert str(tmp_path) not in json.dumps(saved.model_dump(mode="json"))

    add_path, add_type = store.resolve_artifact("manual-masks", _identifier(item.manual_mask.add_mask_url))
    remove_path, remove_type = store.resolve_artifact("manual-masks", _identifier(item.manual_mask.remove_mask_url))
    composite_path, composite_type = store.resolve_artifact(
        "manual-masks",
        _identifier(item.manual_mask.composite_mask_url),
    )
    assert add_type == remove_type == composite_type == "image/png"
    np.testing.assert_array_equal(_read_mask(add_path), edited & ~base)
    np.testing.assert_array_equal(_read_mask(remove_path), base & ~edited)
    np.testing.assert_array_equal(_read_mask(composite_path), edited)


def test_empty_composite_metadata_uses_zero_bbox(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)

    saved = _save_manual(store, workspace_id, object_id, np.zeros_like(base))

    assert saved.objects[0].manual_mask.area_pixels == 0
    assert saved.objects[0].manual_mask.bbox_xyxy == [0, 0, 0, 0]


def test_replacement_removes_old_revision_after_success(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    first = _save_manual(store, workspace_id, object_id, base.copy())
    first_url = first.objects[0].manual_mask.composite_mask_url
    second_mask = base.copy()
    second_mask[7, 9] = True

    second = _save_manual(store, workspace_id, object_id, second_mask)

    assert second.objects[0].manual_mask.manual_revision == 2
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store.resolve_artifact("manual-masks", _identifier(first_url))
    assert store.resolve_artifact("manual-masks", _identifier(second.objects[0].manual_mask.composite_mask_url))
    manual_root = store.root / workspace_id / "objects" / object_id / "manual"
    assert sorted(path.name for path in manual_root.iterdir()) == ["2"]


def test_manual_save_validation_failures_leave_state_unchanged(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    before = store.get_workspace(workspace_id).model_dump(mode="json")
    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))

    with pytest.raises(SegmentationWorkspaceStoreError, match="obsolete SAM prompt"):
        store.save_manual_mask(
            workspace_id,
            object_id,
            edited_mask=base,
            base_prompt_revision=99,
            base_candidate_index=0,
            expected_workspace_revision=before["workspace_revision"],
            expected_object_version=before["objects"][0]["object_version"],
            expected_manual_revision=0,
        )
    with pytest.raises(SegmentationWorkspaceStoreError, match="workspace revision conflict"):
        store.save_manual_mask(
            workspace_id,
            object_id,
            edited_mask=base,
            base_prompt_revision=1,
            base_candidate_index=0,
            expected_workspace_revision=1,
            expected_object_version=before["objects"][0]["object_version"],
            expected_manual_revision=0,
        )

    assert store.get_workspace(workspace_id).model_dump(mode="json") == before
    assert json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8")) == index_before


def test_manual_save_rejects_missing_candidate_and_finalized_object(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    workspace_path = store.root / workspace_id / "workspace.json"
    document = json.loads(workspace_path.read_text(encoding="utf-8"))
    document["objects"][0]["sam_draft"]["selected_candidate_index"] = None
    workspace_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SegmentationWorkspaceStoreError, match="selected SAM candidate"):
        store.save_manual_mask(
            workspace_id,
            object_id,
            edited_mask=base,
            base_prompt_revision=1,
            base_candidate_index=0,
            expected_workspace_revision=document["workspace_revision"],
            expected_object_version=document["objects"][0]["object_version"],
            expected_manual_revision=0,
        )

    document["objects"][0]["sam_draft"]["selected_candidate_index"] = 0
    document["objects"][0]["status"] = "finalized"
    workspace_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SegmentationWorkspaceStoreError, match="object is finalized"):
        store.save_manual_mask(
            workspace_id,
            object_id,
            edited_mask=base,
            base_prompt_revision=1,
            base_candidate_index=0,
            expected_workspace_revision=document["workspace_revision"],
            expected_object_version=document["objects"][0]["object_version"],
            expected_manual_revision=0,
        )


def test_decode_rejects_invalid_wrong_size_rgb_and_nonbinary_masks(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not a png")
    with pytest.raises(SegmentationWorkspaceStoreError, match="invalid"):
        decode_uploaded_manual_mask(invalid, width=10, height=8)

    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (10, 8), (255, 0, 0)).save(rgb, format="PNG")
    with pytest.raises(SegmentationWorkspaceStoreError, match="grayscale"):
        decode_uploaded_manual_mask(rgb, width=10, height=8)

    wrong = _mask_png(tmp_path / "wrong.png", np.zeros((7, 10), dtype=bool))
    with pytest.raises(SegmentationWorkspaceStoreError, match="dimensions"):
        decode_uploaded_manual_mask(wrong, width=10, height=8)

    nonbinary = tmp_path / "nonbinary.png"
    Image.fromarray(np.full((8, 10), 128, dtype=np.uint8), mode="L").save(nonbinary, format="PNG")
    with pytest.raises(SegmentationWorkspaceStoreError, match="0 and 255"):
        decode_uploaded_manual_mask(nonbinary, width=10, height=8)


def test_clear_manual_mask_preserves_sam_draft_and_removes_artifacts(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    saved = _save_manual(store, workspace_id, object_id, base.copy())
    item = saved.objects[0]
    composite_url = item.manual_mask.composite_mask_url

    cleared = store.clear_manual_mask(
        workspace_id,
        object_id,
        expected_workspace_revision=saved.workspace_revision,
        expected_object_version=item.object_version,
        expected_manual_revision=item.manual_mask.manual_revision,
    )

    assert cleared.objects[0].manual_mask.manual_revision == 0
    assert cleared.objects[0].sam_draft.prompt_revision == 1
    assert cleared.objects[0].sam_draft.selected_candidate_index == 0
    assert cleared.workspace_revision == saved.workspace_revision + 1
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store.resolve_artifact("manual-masks", _identifier(composite_url))


def test_manual_mask_blocks_sam_changes_until_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    saved = _save_manual(store, workspace_id, object_id, base.copy())
    calls: list[str] = []
    monkeypatch.setattr(service.sam_engine, "load_model", lambda: calls.append("load") or {})

    with pytest.raises(SegmentationWorkspaceStoreError, match="Clear manual mask"):
        service.predict_workspace_candidates(
            store,
            workspace_id,
            object_id,
            prompt_revision=2,
            points=[SamPromptPoint(x=1, y=1, label=1)],
            box=None,
            base_prompt_revision=None,
            base_candidate_index=None,
            multimask_output=None,
        )
    assert calls == []

    with pytest.raises(SegmentationWorkspaceStoreError, match="Clear manual mask"):
        store.persist_sam_prediction(
            workspace_id,
            object_id,
            prompt_revision=2,
            points=[{"x": 1, "y": 1, "label": 1}],
            box=None,
            candidate_masks=[base],
            candidate_scores=[0.5],
            candidate_areas=[0],
            candidate_bboxes=[[0, 0, 0, 0]],
            selected_candidate_index=0,
            prepared_image_key="workspace",
        )

    with pytest.raises(SegmentationWorkspaceStoreError, match="different SAM candidate"):
        store.select_sam_candidate(
            workspace_id,
            object_id,
            prompt_revision=1,
            candidate_index=1,
            expected_workspace_revision=saved.workspace_revision,
            expected_object_version=saved.objects[0].object_version,
        )


def test_clear_sam_draft_and_object_delete_remove_manual_index_entries(tmp_path: Path) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    saved = _save_manual(store, workspace_id, object_id, base.copy())
    manual_url = saved.objects[0].manual_mask.composite_mask_url
    cleared = store.clear_sam_draft(
        workspace_id,
        object_id,
        expected_workspace_revision=saved.workspace_revision,
        expected_object_version=saved.objects[0].object_version,
    )
    assert cleared.objects[0].manual_mask.manual_revision == 0
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store.resolve_artifact("manual-masks", _identifier(manual_url))

    second = _store_with_candidate(tmp_path / "other")
    store2, workspace_id2, object_id2, base2 = second
    saved2 = _save_manual(store2, workspace_id2, object_id2, base2.copy())
    manual_url2 = saved2.objects[0].manual_mask.composite_mask_url
    store2.delete_object(
        workspace_id2,
        object_id2,
        expected_workspace_revision=saved2.workspace_revision,
        expected_object_version=saved2.objects[0].object_version,
    )
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store2.resolve_artifact("manual-masks", _identifier(manual_url2))


def test_manual_save_failure_restores_old_revision_and_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, workspace_id, object_id, base = _store_with_candidate(tmp_path)
    first = _save_manual(store, workspace_id, object_id, base.copy())
    first_url = first.objects[0].manual_mask.composite_mask_url
    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))

    def fail_save(_workspace) -> None:
        raise RuntimeError("simulated workspace write failure")

    monkeypatch.setattr(store, "_save", fail_save)
    edited = base.copy()
    edited[0, 0] = True
    with pytest.raises(RuntimeError, match="simulated workspace write failure"):
        store.save_manual_mask(
            workspace_id,
            object_id,
            edited_mask=edited,
            base_prompt_revision=1,
            base_candidate_index=0,
            expected_workspace_revision=first.workspace_revision,
            expected_object_version=first.objects[0].object_version,
            expected_manual_revision=1,
        )

    reloaded = store.get_workspace(workspace_id)
    assert reloaded.objects[0].manual_mask.manual_revision == 1
    assert store.resolve_artifact("manual-masks", _identifier(first_url))
    assert json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8")) == index_before
    manual_root = store.root / workspace_id / "objects" / object_id / "manual"
    assert sorted(path.name for path in manual_root.iterdir()) == ["1"]
