from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import reconstruction_engine
from backend.segmentation_workspace.reconstruction_jobs import ReconstructionJobManager, ReconstructionJobStore
from backend.segmentation_workspace.reconstruction_models import ReconstructionJobRequest
from backend.segmentation_workspace.store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


def _image(path: Path, size: tuple[int, int] = (10, 8)) -> Path:
    Image.new("RGB", size, (30, 120, 200)).save(path, format="PNG")
    return path


def _store_with_export(tmp_path: Path) -> tuple[SegmentationWorkspaceStore, str, str]:
    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = store.create_object(workspace.workspace_id, "chair", "Chair", workspace.workspace_revision)
    obj = workspace.objects[0]
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:4, 1:5] = True
    store.persist_sam_prediction(
        workspace.workspace_id,
        obj.object_id,
        prompt_revision=1,
        points=[{"x": 1, "y": 1, "label": 1}],
        box=None,
        candidate_masks=[mask],
        candidate_scores=[0.9],
        candidate_areas=[0],
        candidate_bboxes=[[0, 0, 0, 0]],
        selected_candidate_index=0,
        prepared_image_key="workspace",
    )
    workspace = store.get_workspace(workspace.workspace_id)
    record = store.create_export(
        workspace.workspace_id,
        expected_workspace_revision=workspace.workspace_revision,
        expected_objects=[{"object_id": item.object_id, "expected_object_version": item.object_version} for item in workspace.objects],
    )
    return store, workspace.workspace_id, record.export_id


def _request(store: SegmentationWorkspaceStore, workspace_id: str, export_id: str, *, revision: int | None = None, sha: str | None = None) -> ReconstructionJobRequest:
    workspace = store.get_workspace(workspace_id)
    record = next(item for item in workspace.exports if item.export_id == export_id)
    return ReconstructionJobRequest(
        expected_workspace_revision=revision or workspace.workspace_revision,
        expected_export_archive_sha256=sha or record.archive_sha256,
        resolution_level=9,
        num_tokens=None,
    )


def _fake_moge(*, image_path: Path, output_dir: Path, requested_outputs: list[str], resolution_level: int, num_tokens: int | None) -> dict:
    assert image_path.is_file()
    assert requested_outputs == reconstruction_engine.REQUIRED_MOGE_OUTPUTS
    assert resolution_level == 9
    output_dir.mkdir(parents=True)
    np.savez(
        output_dir / "geometry.npz",
        points=np.zeros((8, 10, 3), np.float32),
        depth=np.ones((8, 10), np.float32),
        valid_mask=np.ones((8, 10), bool),
        intrinsics=np.eye(3, dtype=np.float32),
        normal=np.zeros((8, 10, 3), np.float32),
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_image": str(image_path),
                "output_dir": str(output_dir),
                "output_paths": {"geometry_npz": str(output_dir / "geometry.npz")},
                "source_image_dimensions": {"width": 10, "height": 8},
                "array_shapes": {"points": [8, 10, 3]},
                "estimated_fov_x_degrees": 60.0,
                "runtime_seconds": 0.1,
                "device": "cuda",
                "model": "moge-2-vitl-normal",
            }
        )
    )
    Image.new("L", (10, 8), 1).save(output_dir / "depth_preview.png")
    Image.new("RGB", (10, 8), (127, 127, 255)).save(output_dir / "normal_preview.png")
    Image.new("L", (10, 8), 255).save(output_dir / "valid_mask_preview.png")
    return {"success": True}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_compiler(*, sam_dir: Path, source_image: Path, moge_dir: Path, output_dir: Path, scene_id: str, handoff_dir: Path, expected_object_ids: list[str]) -> reconstruction_engine.CompilerResult:
    assert (sam_dir / "metadata.json").is_file()
    assert source_image.is_file()
    assert (moge_dir / "geometry.npz").is_file()
    output_dir.mkdir(parents=True)
    handoff_dir.mkdir(parents=True)
    objects = [{"object_id": object_id, "semantic_label": f"object_{index}"} for index, object_id in enumerate(expected_object_ids)]
    _write_json(output_dir / "unified_scene_plan.json", {"scene_id": scene_id, "semantic_object_count": len(objects), "semantic_objects": objects})
    _write_json(output_dir / "compilation_report.json", {"object_count": len(objects), "passed": False})
    for name in (
        "room_plan.json",
        "camera_candidates.json",
        "object_pose_report.json",
        "placement_report.json",
        "collision_report.json",
        "confidence_report.json",
        "support_graph.json",
    ):
        _write_json(output_dir / name, {"schema_version": "1.0"})
    (output_dir / "compilation_report.md").write_text("# Compilation\n", encoding="utf-8")
    Image.new("RGB", (10, 8)).save(output_dir / "clean_scene_plan_overview.png")
    Image.new("RGB", (10, 8)).save(output_dir / "projected_primitives_overlay.png")
    Image.new("RGB", (10, 8)).save(output_dir / "room_and_camera_overlay.png")
    Image.new("RGB", (10, 8)).save(output_dir / "confidence_overview.png")
    Image.new("RGB", (10, 8)).save(output_dir / "ambiguity_overview.png")
    _write_json(handoff_dir / "blender_one_batch_manifest.json", {"semantic_primitive_count": len(objects), "semantic_primitives": objects})
    return reconstruction_engine.CompilerResult(scene_id=scene_id, object_count=len(objects), compilation_passed=False, summary={})


def test_valid_export_creates_queued_job_without_workspace_revision_change(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    before = store.get_workspace(workspace_id)
    job = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    after = store.get_workspace(workspace_id)

    assert job.status == "queued"
    assert job.job_version == 1
    assert job.export_was_stale_at_start is False
    assert job.object_ids == [before.objects[0].object_id]
    assert after.workspace_revision == before.workspace_revision
    assert after.objects[0].object_version == before.objects[0].object_version


def test_creation_rejects_revision_hash_duplicate_active_and_queue_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    with pytest.raises(SegmentationWorkspaceStoreError, match="workspace revision conflict"):
        jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id, revision=1))
    with pytest.raises(SegmentationWorkspaceStoreError, match="archive hash conflict"):
        jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id, sha="0" * 64))
    jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    with pytest.raises(SegmentationWorkspaceStoreError, match="already active"):
        jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    monkeypatch.setenv("RECONSTRUCTION_MAX_QUEUED_JOBS", "1")
    store2, workspace2, export2 = _store_with_export(tmp_path / "second")
    jobs2 = ReconstructionJobStore(store2)
    jobs2.create_job(workspace2, export2, _request(store2, workspace2, export2))
    with pytest.raises(SegmentationWorkspaceStoreError, match="queue limit"):
        jobs2.create_job(workspace2, export2, _request(store2, workspace2, export2))


def test_failed_or_interrupted_jobs_permit_retry(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    first = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    jobs.interrupt_job(first)
    retry = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    assert retry.job_id != first.job_id


def test_manager_runs_job_to_success_and_publishes_browser_safe_artifacts(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    job = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    manager = ReconstructionJobManager(jobs, moge_runner=_fake_moge, compiler_runner=_fake_compiler)
    manager._run_job(job.job_id, workspace_id)

    completed = jobs.get_job(workspace_id, job.job_id)
    assert completed.status == "succeeded"
    assert completed.job_version > job.job_version
    assert completed.result.compilation_passed is False
    serialized = completed.model_dump_json()
    assert str(tmp_path) not in serialized
    assert "source_image" not in json.loads((store.root / workspace_id / "reconstructions" / job.job_id / "moge" / "moge-summary.json").read_text())
    manifest = json.loads((store.root / workspace_id / "reconstructions" / job.job_id / "reconstruction-result-manifest.json").read_text())
    assert manifest["object_ids"] == completed.object_ids
    assert manifest["moge_geometry_sha256"]
    identifier = completed.result.artifacts.unified_scene_plan_url.split("/api/segmentation-artifacts/reconstruction-results/", 1)[1]
    path, media_type = store.resolve_artifact("reconstruction-results", identifier)
    assert path.name == "unified_scene_plan.json"
    assert media_type == "application/json"


def test_moge_failure_persists_sanitized_failed_job_and_removes_staging(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    job = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))

    def failing_moge(**_kwargs):
        raise reconstruction_engine.ReconstructionEngineError(r"failed at C:\models\secret.pt", code="moge_failed")

    manager = ReconstructionJobManager(jobs, moge_runner=failing_moge, compiler_runner=_fake_compiler)
    manager._run_job(job.job_id, workspace_id)
    failed = jobs.get_job(workspace_id, job.job_id)

    assert failed.status == "failed"
    assert failed.error.code == "moge_failed"
    assert "C:\\models" not in failed.error.message
    recon_root = store.root / workspace_id / "reconstructions"
    assert not recon_root.exists() or not [path for path in recon_root.iterdir() if path.name.startswith(".")]


def test_restart_reconciles_active_jobs_and_terminal_jobs_remain(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    queued = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    (store.root / workspace_id / "reconstructions" / f".{queued.job_id}.staging").mkdir(parents=True)
    restarted = ReconstructionJobStore(store)

    interrupted = restarted.get_job(workspace_id, queued.job_id)
    assert interrupted.status == "interrupted"
    assert interrupted.error.retryable is True
    assert not (store.root / workspace_id / "reconstructions" / f".{queued.job_id}.staging").exists()


def test_active_job_blocks_workspace_deletion_but_object_editing_and_export_anchoring_continue(tmp_path: Path) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    job = jobs.create_job(workspace_id, export_id, _request(store, workspace_id, export_id))
    workspace = store.get_workspace(workspace_id)

    with pytest.raises(SegmentationWorkspaceStoreError, match="active reconstruction job"):
        store.delete_workspace(workspace_id, workspace.workspace_revision)

    item = workspace.objects[0]
    updated = store.update_object(
        workspace_id,
        item.object_id,
        "renamed",
        "Renamed",
        workspace.workspace_revision,
        item.object_version,
    )
    assert updated.workspace_revision == workspace.workspace_revision + 1
    inputs = jobs.prepare_execution_inputs(job)
    assert inputs.export_id == export_id
    assert inputs.object_ids == job.object_ids
