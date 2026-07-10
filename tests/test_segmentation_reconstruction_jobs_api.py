from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from backend import reconstruction_engine
from backend.main import app
from backend.segmentation_workspace import api
from backend.segmentation_workspace.reconstruction_jobs import ReconstructionJobManager, ReconstructionJobStore
from backend.segmentation_workspace.store import SegmentationWorkspaceStore


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


def _payload(store: SegmentationWorkspaceStore, workspace_id: str, export_id: str, **overrides) -> dict:
    workspace = store.get_workspace(workspace_id)
    record = next(item for item in workspace.exports if item.export_id == export_id)
    payload = {
        "expected_workspace_revision": workspace.workspace_revision,
        "expected_export_archive_sha256": record.archive_sha256,
        "resolution_level": 9,
        "num_tokens": None,
    }
    payload.update(overrides)
    return payload


def _fake_moge(*, image_path: Path, output_dir: Path, requested_outputs: list[str], resolution_level: int, num_tokens: int | None) -> dict:
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
    return {"success": True}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_compiler(*, sam_dir: Path, source_image: Path, moge_dir: Path, output_dir: Path, scene_id: str, handoff_dir: Path, expected_object_ids: list[str]) -> reconstruction_engine.CompilerResult:
    output_dir.mkdir(parents=True)
    handoff_dir.mkdir(parents=True)
    objects = [{"object_id": object_id, "semantic_label": "chair"} for object_id in expected_object_ids]
    _write_json(output_dir / "unified_scene_plan.json", {"scene_id": scene_id, "semantic_object_count": len(objects), "semantic_objects": objects})
    _write_json(output_dir / "compilation_report.json", {"object_count": len(objects), "passed": True})
    for name in ("room_plan.json", "camera_candidates.json", "object_pose_report.json", "placement_report.json", "collision_report.json", "confidence_report.json", "support_graph.json"):
        _write_json(output_dir / name, {"schema_version": "1.0"})
    (output_dir / "compilation_report.md").write_text("# Compilation\n", encoding="utf-8")
    Image.new("RGB", (10, 8)).save(output_dir / "clean_scene_plan_overview.png")
    _write_json(handoff_dir / "blender_one_batch_manifest.json", {"semantic_primitive_count": len(objects), "semantic_primitives": objects})
    return reconstruction_engine.CompilerResult(scene_id=scene_id, object_count=len(objects), compilation_passed=True, summary={})


class NoopManager:
    def submit(self, _job) -> None:
        return None


@pytest.fixture
def configured_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    monkeypatch.setattr(api, "STORE", store)
    monkeypatch.setattr(api, "RECONSTRUCTION_JOBS", jobs)
    monkeypatch.setattr(api, "RECONSTRUCTION_MANAGER", NoopManager())
    monkeypatch.setattr(api, "_validate_reconstruction_environment", lambda: None)
    return TestClient(app), store, jobs, workspace_id, export_id


def test_start_list_read_missing_and_conflicts(configured_api) -> None:
    client, store, _jobs, workspace_id, export_id = configured_api
    response = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["status"] == "queued"
    assert queued["job_version"] == 1
    assert "result" in queued and queued["result"] is None

    listed = client.get(f"/api/segmentation-workspaces/{workspace_id}/reconstructions")
    assert listed.status_code == 200
    assert listed.json()[0]["job_id"] == queued["job_id"]

    fetched = client.get(f"/api/segmentation-workspaces/{workspace_id}/reconstructions/{queued['job_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == queued["job_id"]

    duplicate = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )
    assert duplicate.status_code == 409
    missing = client.get(f"/api/segmentation-workspaces/{workspace_id}/reconstructions/{'9' * 32}")
    assert missing.status_code == 404
    stale = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id, expected_workspace_revision=1),
    )
    assert stale.status_code == 409


def test_start_rejects_unconfigured_environment_before_persisting_job(configured_api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, store, jobs, workspace_id, export_id = configured_api

    def fail_config() -> None:
        raise HTTPException(status_code=503, detail="MoGe configuration is incomplete.")

    monkeypatch.setattr(api, "_validate_reconstruction_environment", fail_config)
    response = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )
    assert response.status_code == 503
    assert jobs.list_jobs(workspace_id) == []


def test_start_submission_failure_returns_503_and_marks_job_failed(configured_api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, store, jobs, workspace_id, export_id = configured_api

    class RaisingExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError(r"executor failed at C:\secret\worker")

    manager = ReconstructionJobManager(jobs, executor=RaisingExecutor(), moge_runner=_fake_moge, compiler_runner=_fake_compiler)
    monkeypatch.setattr(api, "RECONSTRUCTION_MANAGER", manager)
    response = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )

    assert response.status_code == 503
    assert "could not be submitted" in response.json()["detail"]
    assert r"C:\secret" not in response.text
    records = jobs.list_jobs(workspace_id)
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].error.code == "queue_submission_failed"
    assert jobs.active_count() == 0


def test_health_response_is_browser_safe(monkeypatch: pytest.MonkeyPatch, configured_api) -> None:
    client, _store, jobs, _workspace_id, _export_id = configured_api
    monkeypatch.setattr(api.moge_engine, "get_status", lambda: {"configured": True, "worker_running": False, "device": "cuda", "model": "moge-2-vitl-normal", "python": r"C:\secret\python.exe"})
    monkeypatch.setattr(api.reconstruction_engine, "compiler_health", lambda: {"configured": True, "compiler_configured": True, "error": None, "python": r"C:\secret\recon.exe"})
    response = client.get("/api/segmentation-reconstruction/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["queue_running"] == 0
    assert "python" not in payload
    assert "C:\\secret" not in json.dumps(payload)


def test_inline_success_polling_and_artifact_api_are_browser_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, workspace_id, export_id = _store_with_export(tmp_path)
    jobs = ReconstructionJobStore(store)
    real_manager = ReconstructionJobManager(jobs, moge_runner=_fake_moge, compiler_runner=_fake_compiler)

    class InlineManager:
        def submit(self, job) -> None:
            real_manager._run_job(job.job_id, job.workspace_id)

    monkeypatch.setattr(api, "STORE", store)
    monkeypatch.setattr(api, "RECONSTRUCTION_JOBS", jobs)
    monkeypatch.setattr(api, "RECONSTRUCTION_MANAGER", InlineManager())
    monkeypatch.setattr(api, "_validate_reconstruction_environment", lambda: None)
    client = TestClient(app)

    started = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )
    assert started.status_code == 202, started.text
    job_id = started.json()["job_id"]
    fetched = client.get(f"/api/segmentation-workspaces/{workspace_id}/reconstructions/{job_id}")
    assert fetched.status_code == 200
    completed = fetched.json()
    assert completed["status"] == "succeeded"
    assert completed["result"]["artifacts"]["result_manifest_url"].startswith("/api/segmentation-artifacts/reconstruction-results/")
    assert str(tmp_path) not in json.dumps(completed)
    assert "base64" not in json.dumps(completed)

    manifest = client.get(completed["result"]["artifacts"]["result_manifest_url"])
    assert manifest.status_code == 200
    assert manifest.headers["cache-control"] == "private, max-age=31536000, immutable"
    assert manifest.headers["x-content-type-options"] == "nosniff"
    manifest_json = manifest.json()
    assert manifest_json["scene_id"] == completed["scene_id"]
    assert str(tmp_path) not in json.dumps(manifest_json)

    raw_moge_metadata = client.get(f"/api/segmentation-artifacts/reconstruction-results/{workspace_id}/{job_id}/metadata")
    assert raw_moge_metadata.status_code == 404
    traversal = client.get("/api/segmentation-artifacts/reconstruction-results/../../workspace.json")
    assert traversal.status_code in {400, 404}


def test_active_job_blocks_workspace_delete_through_api(configured_api) -> None:
    client, store, _jobs, workspace_id, export_id = configured_api
    started = client.post(
        f"/api/segmentation-workspaces/{workspace_id}/exports/{export_id}/reconstructions",
        json=_payload(store, workspace_id, export_id),
    )
    assert started.status_code == 202
    workspace = store.get_workspace(workspace_id)
    deleted = client.delete(
        f"/api/segmentation-workspaces/{workspace_id}",
        params={"expected_workspace_revision": workspace.workspace_revision},
    )
    assert deleted.status_code == 409
