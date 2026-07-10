from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import sam_engine
from backend.segmentation_workspace import service
from backend.segmentation_workspace.models import SamPromptPoint
from backend.segmentation_workspace.store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


def _image(path: Path, size: tuple[int, int] = (10, 8)) -> Path:
    Image.new("RGB", size, (30, 120, 200)).save(path, format="PNG")
    return path


def _store_with_object(tmp_path: Path) -> tuple[SegmentationWorkspaceStore, str, str]:
    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = store.create_object(workspace.workspace_id, "chair", "Chair", workspace.workspace_revision)
    return store, workspace.workspace_id, workspace.objects[0].object_id


def _fake_prediction(count: int = 3) -> sam_engine.CandidatePrediction:
    masks = np.zeros((count, 8, 10), dtype=bool)
    for index in range(count):
        masks[index, index : index + 2, index : index + 3] = True
    return sam_engine.CandidatePrediction(
        image_key="workspace",
        width=10,
        height=8,
        masks=masks,
        scores=[0.2, 0.95, 0.5][:count],
        areas=[int(mask.sum()) for mask in masks],
        bboxes=[[index, index, index + 2, index + 1] for index in range(count)],
        logits=np.ones((count, 4, 4), dtype=np.float32),
        elapsed_ms=1.25,
    )


def _patch_sam(monkeypatch, calls: list[dict]) -> None:
    monkeypatch.setattr(service.sam_engine, "load_model", lambda: {"model_type": "vit_b", "device": "cpu"})
    monkeypatch.setattr(
        service.sam_engine,
        "prepare_image_from_path",
        lambda image_key, path: {"image_set": True, "image_id": image_key, "width": 10, "height": 8},
    )

    def predict_candidates(*args, **kwargs):
        calls.append(kwargs)
        return _fake_prediction(1 if kwargs.get("multimask_output") is False else 3)

    monkeypatch.setattr(service.sam_engine, "predict_candidates", predict_candidates)
    monkeypatch.setattr(service, "LOGITS_CACHE", service.SamLogitsCache())


def test_prepare_sam_resolves_managed_source_image(tmp_path: Path, monkeypatch) -> None:
    store, workspace_id, _object_id = _store_with_object(tmp_path)
    seen: dict[str, Path] = {}
    monkeypatch.setattr(service.sam_engine, "load_model", lambda: {"model_type": "vit_b", "device": "cpu"})

    def prepare(image_key: str, path: Path) -> dict:
        seen["path"] = path
        return {"image_set": True, "image_id": image_key, "width": 10, "height": 8}

    monkeypatch.setattr(service.sam_engine, "prepare_image_from_path", prepare)

    response = service.prepare_workspace_sam(store, workspace_id)

    assert response["sam_ready"] is True
    assert response["model_type"] == "vit_b"
    assert seen["path"].name == "source.png"
    assert str(tmp_path) not in json.dumps(response)


def test_prediction_persists_candidates_and_cache_then_replaces_old_revision(tmp_path: Path, monkeypatch) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    calls: list[dict] = []
    _patch_sam(monkeypatch, calls)

    first = service.predict_workspace_candidates(
        store,
        workspace_id,
        object_id,
        prompt_revision=1,
        points=[SamPromptPoint(x=1, y=1, label=1), SamPromptPoint(x=4, y=2, label=0)],
        box=None,
        base_prompt_revision=None,
        base_candidate_index=None,
        multimask_output=None,
    )

    assert len(first["candidates"]) == 3
    assert first["selected_candidate_index"] == 1
    assert first["object_id"] == object_id
    assert "png_base64" not in json.dumps(first)
    assert calls[-1]["point_labels"] == [1, 0]
    first_url = first["candidates"][0]["mask_url"]
    first_identifier = first_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    first_path, media_type = store.resolve_artifact("sam-candidates", first_identifier)
    assert media_type == "image/png"
    persisted = Image.open(first_path)
    assert persisted.mode == "L"
    assert persisted.size == (10, 8)
    assert sorted(np.unique(np.array(persisted)).tolist()) == [0, 255]
    assert first["candidates"][0]["area_pixels"] == 6
    assert first["candidates"][0]["bbox_xyxy"] == [0, 0, 2, 1]

    second = service.predict_workspace_candidates(
        store,
        workspace_id,
        object_id,
        prompt_revision=2,
        points=[SamPromptPoint(x=1, y=1, label=1), SamPromptPoint(x=5, y=3, label=0)],
        box=None,
        base_prompt_revision=1,
        base_candidate_index=1,
        multimask_output=None,
    )

    assert second["used_mask_input"] is True
    assert calls[-1]["mask_input"] is not None
    assert calls[-1]["multimask_output"] is False
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store.resolve_artifact("sam-candidates", first_identifier)


def test_store_derives_candidate_metadata_from_pixels(tmp_path: Path) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 3:7] = True

    workspace = store.persist_sam_prediction(
        workspace_id,
        object_id,
        prompt_revision=1,
        points=[{"x": 1, "y": 1, "label": 1}],
        box=None,
        candidate_masks=[mask],
        candidate_scores=[0.1],
        candidate_areas=[999],
        candidate_bboxes=[[9, 9, 9, 9]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )

    candidate = workspace.objects[0].sam_draft.candidates[0]
    assert candidate.area_pixels == 12
    assert candidate.bbox_xyxy == [3, 2, 6, 4]
    identifier = candidate.mask_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    path, _media_type = store.resolve_artifact("sam-candidates", identifier)
    persisted = Image.open(path)
    assert persisted.size == (10, 8)
    assert sorted(np.unique(np.array(persisted)).tolist()) == [0, 255]


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((7, 10), dtype=bool), "dimensions"),
        (np.zeros((8, 9), dtype=bool), "dimensions"),
        (np.zeros((8, 10, 1), dtype=bool), "two-dimensional"),
    ],
)
def test_store_rejects_malformed_candidate_without_state_change(
    tmp_path: Path,
    mask: np.ndarray,
    message: str,
) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    before = store.get_workspace(workspace_id).model_dump(mode="json")
    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))

    with pytest.raises(SegmentationWorkspaceStoreError, match=message):
        store.persist_sam_prediction(
            workspace_id,
            object_id,
            prompt_revision=1,
            points=[{"x": 1, "y": 1, "label": 1}],
            box=None,
            candidate_masks=[mask],
            candidate_scores=[0.1],
            candidate_areas=[0],
            candidate_bboxes=[[0, 0, 0, 0]],
            selected_candidate_index=0,
            prepared_image_key="workspace",
        )

    after = store.get_workspace(workspace_id).model_dump(mode="json")
    index_after = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))
    assert after == before
    assert index_after == index_before
    sam_root = store.root / workspace_id / "objects" / object_id / "sam"
    assert not sam_root.exists() or not list(sam_root.iterdir())


def test_persist_failure_restores_old_index_and_removes_new_candidate_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    first_mask = np.zeros((8, 10), dtype=bool)
    first_mask[1:3, 1:4] = True
    workspace = store.persist_sam_prediction(
        workspace_id,
        object_id,
        prompt_revision=1,
        points=[{"x": 1, "y": 1, "label": 1}],
        box=None,
        candidate_masks=[first_mask],
        candidate_scores=[0.5],
        candidate_areas=[0],
        candidate_bboxes=[[0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    old_candidate = workspace.objects[0].sam_draft.candidates[0]
    old_identifier = old_candidate.mask_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    old_path, _media_type = store.resolve_artifact("sam-candidates", old_identifier)
    index_before = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))

    def fail_save(_workspace) -> None:
        raise RuntimeError("simulated workspace write failure")

    monkeypatch.setattr(store, "_save", fail_save)
    second_mask = np.zeros((8, 10), dtype=bool)
    second_mask[4:6, 4:8] = True

    with pytest.raises(RuntimeError, match="simulated workspace write failure"):
        store.persist_sam_prediction(
            workspace_id,
            object_id,
            prompt_revision=2,
            points=[{"x": 2, "y": 2, "label": 1}],
            box=None,
            candidate_masks=[second_mask],
            candidate_scores=[0.9],
            candidate_areas=[0],
            candidate_bboxes=[[0, 0, 0, 0]],
            selected_candidate_index=0,
            prepared_image_key="workspace",
        )

    reloaded = store.get_workspace(workspace_id)
    assert reloaded.workspace_revision == workspace.workspace_revision
    assert reloaded.objects[0].object_version == workspace.objects[0].object_version
    assert reloaded.objects[0].sam_draft.prompt_revision == 1
    assert store.resolve_artifact("sam-candidates", old_identifier)[0] == old_path
    index_after = json.loads((store.root / workspace_id / "artifact-index.json").read_text(encoding="utf-8"))
    assert index_after == index_before
    sam_root = store.root / workspace_id / "objects" / object_id / "sam"
    assert (sam_root / "1").is_dir()
    assert not (sam_root / "2").exists()
    assert [path for path in sam_root.iterdir() if path.name.startswith(".")] == []


def test_missing_logits_falls_back_to_point_only_prediction(tmp_path: Path, monkeypatch) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    calls: list[dict] = []
    _patch_sam(monkeypatch, calls)

    response = service.predict_workspace_candidates(
        store,
        workspace_id,
        object_id,
        prompt_revision=1,
        points=[SamPromptPoint(x=1, y=1, label=1)],
        box=None,
        base_prompt_revision=99,
        base_candidate_index=2,
        multimask_output=None,
    )

    assert response["used_mask_input"] is False
    assert calls[-1]["mask_input"] is None
    assert calls[-1]["multimask_output"] is True


def test_stale_prompt_select_clear_and_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    calls: list[dict] = []
    _patch_sam(monkeypatch, calls)
    predicted = service.predict_workspace_candidates(
        store,
        workspace_id,
        object_id,
        prompt_revision=1,
        points=[SamPromptPoint(x=1, y=1, label=1)],
        box=None,
        base_prompt_revision=None,
        base_candidate_index=None,
        multimask_output=None,
    )

    with pytest.raises(SegmentationWorkspaceStoreError, match="stale prompt revision"):
        service.predict_workspace_candidates(
            store,
            workspace_id,
            object_id,
            prompt_revision=1,
            points=[SamPromptPoint(x=2, y=2, label=1)],
            box=None,
            base_prompt_revision=None,
            base_candidate_index=None,
            multimask_output=None,
        )

    selected = service.select_workspace_candidate(
        store,
        workspace_id,
        object_id,
        prompt_revision=1,
        candidate_index=2,
        expected_workspace_revision=predicted["workspace_revision"],
        expected_object_version=predicted["object_version"],
    )
    assert selected["selected_candidate_index"] == 2
    assert len(calls) == 1

    current = store.get_workspace(workspace_id)
    item = current.objects[0]
    candidate_url = item.sam_draft.candidates[0].mask_url
    candidate_identifier = candidate_url.split("/api/segmentation-artifacts/sam-candidates/", 1)[1]
    candidate_path, _media_type = store.resolve_artifact("sam-candidates", candidate_identifier)
    candidate_path.write_bytes(b"tampered")
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact integrity check failed"):
        store.resolve_artifact("sam-candidates", candidate_identifier)
    Image.new("L", (10, 8), 0).save(candidate_path)

    cleared = service.clear_workspace_sam_draft(
        store,
        workspace_id,
        object_id,
        expected_workspace_revision=selected["workspace_revision"],
        expected_object_version=selected["object_version"],
    )
    assert cleared["sam_draft"]["prompt_revision"] == 0
    assert cleared["sam_draft"]["candidates"] == []
    with pytest.raises(SegmentationWorkspaceStoreError, match="artifact not found"):
        store.resolve_artifact("sam-candidates", candidate_identifier)


def test_prediction_validation_rejects_bad_prompts(tmp_path: Path, monkeypatch) -> None:
    store, workspace_id, object_id = _store_with_object(tmp_path)
    calls: list[dict] = []
    _patch_sam(monkeypatch, calls)

    with pytest.raises(SegmentationWorkspaceStoreError, match="outside the source image"):
        service.predict_workspace_candidates(
            store,
            workspace_id,
            object_id,
            prompt_revision=1,
            points=[SamPromptPoint(x=99, y=1, label=1)],
            box=None,
            base_prompt_revision=None,
            base_candidate_index=None,
            multimask_output=None,
        )
    assert calls == []
