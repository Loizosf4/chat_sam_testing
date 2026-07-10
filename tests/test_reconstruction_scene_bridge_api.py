from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.segmentation_workspace import api as segmentation_api
from backend.segmentation_workspace.review_scene import ReconstructionReviewSceneBridge

from test_reconstruction_scene_bridge import build_successful_bridge


def _install_bridge(tmp_path: Path, monkeypatch, *, compilation_passed: bool = True):
    workspace_store, jobs, scene_store, bridge, job = build_successful_bridge(tmp_path, compilation_passed=compilation_passed)
    monkeypatch.setattr(segmentation_api, "STORE", workspace_store)
    monkeypatch.setattr(segmentation_api, "RECONSTRUCTION_JOBS", jobs)
    monkeypatch.setattr(segmentation_api, "REVIEW_SCENES", ReconstructionReviewSceneBridge(workspace_store, jobs, scene_store))
    return workspace_store, jobs, scene_store, bridge, job


def test_review_scene_api_create_status_and_idempotency(tmp_path: Path, monkeypatch) -> None:
    _workspace_store, _jobs, _scene_store, _bridge, job = _install_bridge(tmp_path, monkeypatch)
    client = TestClient(app)

    status = client.get(f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene")
    assert status.status_code == 200
    assert status.json()["imported"] is False
    assert status.json()["scene_url"] is None

    created = client.post(
        f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene",
        json={"expected_job_version": job.job_version},
    )
    assert created.status_code == 201
    assert created.json()["created"] is True
    assert created.json()["scene_url"] == f"/?scene={job.scene_id}"

    again = client.post(
        f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene",
        json={"expected_job_version": job.job_version},
    )
    assert again.status_code == 200
    assert again.json()["created"] is False


def test_review_scene_api_requires_acknowledgement_and_forbids_unknown_fields(tmp_path: Path, monkeypatch) -> None:
    _workspace_store, _jobs, _scene_store, _bridge, job = _install_bridge(tmp_path, monkeypatch, compilation_passed=False)
    client = TestClient(app)

    unknown = client.post(
        f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene",
        json={"expected_job_version": job.job_version, "path": "C:\\secret"},
    )
    assert unknown.status_code == 422

    blocked = client.post(
        f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene",
        json={"expected_job_version": job.job_version},
    )
    assert blocked.status_code == 409
    assert "quality gates require review" in blocked.json()["detail"]

    created = client.post(
        f"/api/segmentation-workspaces/{job.workspace_id}/reconstructions/{job.job_id}/review-scene",
        json={"expected_job_version": job.job_version, "acknowledge_review_required": True},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["compilation_passed"] is False
    forbidden = ("C:\\", "/tmp/", "file://", "Traceback", "python.exe")
    assert not any(item in created.text for item in forbidden)
