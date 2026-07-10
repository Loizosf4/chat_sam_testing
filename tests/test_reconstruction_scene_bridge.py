from __future__ import annotations

import json
import threading
from queue import Queue
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import reconstruction_engine
from backend.scene_package.store import SceneStore
from backend.segmentation_workspace.reconstruction_jobs import ReconstructionJobManager, ReconstructionJobStore
from backend.segmentation_workspace.reconstruction_models import ReconstructionJobRequest
from backend.segmentation_workspace.review_scene import (
    CreateReviewSceneRequest,
    ReconstructionReviewSceneBridge,
    ReviewSceneBridgeError,
)
from backend.segmentation_workspace.store import SegmentationWorkspaceStore


def _image(path: Path, size: tuple[int, int] = (10, 8)) -> Path:
    Image.new("RGB", size, (30, 120, 200)).save(path, format="PNG")
    return path


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fake_moge(*, image_path: Path, output_dir: Path, requested_outputs: list[str], resolution_level: int, num_tokens: int | None) -> dict:
    del requested_outputs, resolution_level, num_tokens
    output_dir.mkdir(parents=True)
    np.savez(
        output_dir / "geometry.npz",
        points=np.zeros((8, 10, 3), np.float32),
        depth=np.ones((8, 10), np.float32),
        valid_mask=np.ones((8, 10), bool),
        intrinsics=np.eye(3, dtype=np.float32),
        normal=np.zeros((8, 10, 3), np.float32),
    )
    _write_json(output_dir / "metadata.json", {"source_image": str(image_path), "output_paths": {"geometry_npz": str(output_dir / "geometry.npz")}})
    Image.new("L", (10, 8), 1).save(output_dir / "depth_preview.png")
    Image.new("RGB", (10, 8), (127, 127, 255)).save(output_dir / "normal_preview.png")
    Image.new("L", (10, 8), 255).save(output_dir / "valid_mask_preview.png")
    return {"success": True}


def _compiler(compilation_passed: bool = True):
    def fake_compiler(
        *,
        sam_dir: Path,
        source_image: Path,
        moge_dir: Path,
        output_dir: Path,
        scene_id: str,
        handoff_dir: Path,
        expected_object_ids: list[str],
    ) -> reconstruction_engine.CompilerResult:
        del sam_dir, source_image, moge_dir
        output_dir.mkdir(parents=True)
        handoff_dir.mkdir(parents=True)
        objects = [
            {
                "object_id": object_id,
                "semantic_label": f"object_{index}",
                "primitive_type": "box",
                "geometry_confidence": 0.9,
                "final_pose_confidence": 0.8,
                "support_type": "unknown",
                "support_target": None,
                "transform": {"center": [index + 1.0, 0.0, 0.5], "dimensions": [1.0, 1.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "occlusion": {"partially_occluded": False},
            }
            for index, object_id in enumerate(expected_object_ids)
        ]
        _write_json(
            output_dir / "unified_scene_plan.json",
            {
                "mode": "clean_reconstruction",
                "scene_id": scene_id,
                "input_manifest_sha256": "a" * 64,
                "semantic_object_count": len(objects),
                "semantic_objects": objects,
                "coordinate_system": {"absolute_scale_verified": True},
                "camera_candidates": [
                    {
                        "camera_id": "camera_main",
                        "type": "perspective",
                        "matrix_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        "confidence": 0.9,
                        "provisional": True,
                    }
                ],
                "provisional_camera_id": "camera_main",
                "room_proxies": [],
                "uncertainties": [],
            },
        )
        _write_json(output_dir / "compilation_report.json", {"object_count": len(objects), "passed": compilation_passed})
        for name in ("room_plan.json", "camera_candidates.json", "object_pose_report.json", "placement_report.json", "collision_report.json", "confidence_report.json", "support_graph.json"):
            _write_json(output_dir / name, {"schema_version": "1.0"})
        (output_dir / "compilation_report.md").write_text("# Compilation\n", encoding="utf-8")
        _write_json(handoff_dir / "blender_one_batch_manifest.json", {"semantic_primitive_count": len(objects), "semantic_primitives": objects})
        return reconstruction_engine.CompilerResult(scene_id=scene_id, object_count=len(objects), compilation_passed=compilation_passed, summary={})

    return fake_compiler


def build_successful_bridge(tmp_path: Path, *, compilation_passed: bool = True):
    workspace_store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    workspace = workspace_store.create_workspace(_image(tmp_path / "source.png"), "source.png")
    workspace = workspace_store.create_object(workspace.workspace_id, "chair", "Chair", workspace.workspace_revision)
    obj = workspace.objects[0]
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:4, 1:5] = True
    workspace_store.persist_sam_prediction(
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
    workspace = workspace_store.get_workspace(workspace.workspace_id)
    export = workspace_store.create_export(
        workspace.workspace_id,
        expected_workspace_revision=workspace.workspace_revision,
        expected_objects=[{"object_id": item.object_id, "expected_object_version": item.object_version} for item in workspace.objects],
    )
    jobs = ReconstructionJobStore(workspace_store)
    job = jobs.create_job(
        workspace.workspace_id,
        export.export_id,
        ReconstructionJobRequest(
            expected_workspace_revision=workspace_store.get_workspace(workspace.workspace_id).workspace_revision,
            expected_export_archive_sha256=export.archive_sha256,
            resolution_level=9,
            num_tokens=None,
        ),
    )
    manager = ReconstructionJobManager(jobs, moge_runner=_fake_moge, compiler_runner=_compiler(compilation_passed))
    manager._run_job(job.job_id, workspace.workspace_id)
    completed = jobs.get_job(workspace.workspace_id, job.job_id)
    scene_store = SceneStore(tmp_path / "scenes")
    bridge = ReconstructionReviewSceneBridge(workspace_store, jobs, scene_store)
    return workspace_store, jobs, scene_store, bridge, completed


def test_create_review_scene_copies_artifacts_and_is_idempotent(tmp_path: Path) -> None:
    workspace_store, _jobs, scene_store, bridge, job = build_successful_bridge(tmp_path)
    request = CreateReviewSceneRequest(expected_job_version=job.job_version)

    created = bridge.create(job.workspace_id, job.job_id, request)
    again = bridge.create(job.workspace_id, job.job_id, request)

    assert created.created is True
    assert again.created is False
    assert again.scene_id == created.scene_id == job.scene_id
    scene = scene_store.get(job.scene_id)
    assert [item.object_id for item in scene.semantic_objects] == job.object_ids
    assert scene.metadata["import_origin"] == "segmentation_reconstruction_job"
    assert scene.metadata["reconstruction_job_id"] == job.job_id
    assert "C:\\" not in scene.model_dump_json()
    assert "/tmp/" not in scene.model_dump_json()
    for key in ("reconstruction_result_manifest", "compilation_report", "compilation_report_markdown", "moge_geometry", "moge_summary"):
        ref = scene.artifact_urls[key]
        path, _media_type = scene_store.resolve_artifact("scene-inputs", ref.url.split("/artifacts/scene-inputs/", 1)[1])
        assert path.is_file()

    workspace = workspace_store.get_workspace(job.workspace_id)
    workspace_store.delete_workspace(job.workspace_id, workspace.workspace_revision)
    assert scene_store.get(job.scene_id).scene_id == job.scene_id
    for item in scene.semantic_objects:
        assert scene_store.resolve_artifact("masks", f"{item.object_id}/{item.mask_revision}")[0].is_file()


def test_review_required_rejects_until_acknowledged(tmp_path: Path) -> None:
    _workspace_store, _jobs, _scene_store, bridge, job = build_successful_bridge(tmp_path, compilation_passed=False)

    with pytest.raises(ReviewSceneBridgeError, match="quality gates require review") as exc:
        bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version))

    assert exc.value.status_code == 409
    result = bridge.create(
        job.workspace_id,
        job.job_id,
        CreateReviewSceneRequest(expected_job_version=job.job_version, acknowledge_review_required=True),
    )
    assert result.imported is True


def test_rejects_job_version_conflict_and_non_success(tmp_path: Path) -> None:
    workspace_store, jobs, scene_store, bridge, job = build_successful_bridge(tmp_path)

    with pytest.raises(ReviewSceneBridgeError, match="version conflict") as exc:
        bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version - 1))
    assert exc.value.status_code == 409

    queued = jobs.create_job(
        job.workspace_id,
        job.export_id,
        ReconstructionJobRequest(
            expected_workspace_revision=workspace_store.get_workspace(job.workspace_id).workspace_revision,
            expected_export_archive_sha256=job.export_archive_sha256,
            resolution_level=9,
            num_tokens=None,
        ),
    )
    blocked = ReconstructionReviewSceneBridge(workspace_store, jobs, scene_store)
    with pytest.raises(ReviewSceneBridgeError, match="successful completed"):
        blocked.status(queued.workspace_id, queued.job_id)


def test_collision_with_unrelated_scene_is_rejected(tmp_path: Path) -> None:
    _workspace_store, _jobs, scene_store, bridge, job = build_successful_bridge(tmp_path)
    request = CreateReviewSceneRequest(expected_job_version=job.job_version)
    scene = bridge.create(job.workspace_id, job.job_id, request)
    scene_path = scene_store.root / scene.scene_id / "scene.json"
    document = json.loads(scene_path.read_text(encoding="utf-8"))
    document["metadata"]["reconstruction_job_id"] = "0" * 32
    scene_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReviewSceneBridgeError, match="unrelated provenance") as exc:
        bridge.create(job.workspace_id, job.job_id, request)
    assert exc.value.status_code == 409


def test_reverification_catches_missing_unified_and_artifact_hash_mismatch(tmp_path: Path) -> None:
    workspace_store, _jobs, _scene_store, bridge, job = build_successful_bridge(tmp_path)
    unified_key = job.result.artifacts.unified_scene_plan_url.split("/api/segmentation-artifacts/", 1)[1]
    unified_path, _ = workspace_store._resolve_workspace_artifact(job.workspace_id, unified_key)
    unified_path.unlink()

    with pytest.raises(ReviewSceneBridgeError):
        bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version))

    workspace_store, _jobs, _scene_store, bridge, job = build_successful_bridge(tmp_path / "hash")
    geometry_key = job.result.artifacts.moge_geometry_url.split("/api/segmentation-artifacts/", 1)[1]
    geometry_path, _ = workspace_store._resolve_workspace_artifact(job.workspace_id, geometry_key)
    geometry_path.write_bytes(b"corrupted")
    with pytest.raises(ReviewSceneBridgeError, match="integrity"):
        bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version))


def test_status_reports_absent_and_existing_import(tmp_path: Path) -> None:
    _workspace_store, _jobs, _scene_store, bridge, job = build_successful_bridge(tmp_path)

    absent = bridge.status(job.workspace_id, job.job_id)
    assert absent.imported is False
    assert absent.scene_url is None

    bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version))
    present = bridge.status(job.workspace_id, job.job_id)
    assert present.imported is True
    assert present.scene_url == f"/?scene={job.scene_id}"


def test_concurrent_imports_share_one_scene(tmp_path: Path) -> None:
    _workspace_store, _jobs, scene_store, bridge, job = build_successful_bridge(tmp_path)
    start = threading.Barrier(3)
    results: Queue[object] = Queue()

    def create() -> None:
        try:
            start.wait(timeout=3)
            results.put(bridge.create(job.workspace_id, job.job_id, CreateReviewSceneRequest(expected_job_version=job.job_version)))
        except Exception as exc:
            results.put(exc)

    threads = [threading.Thread(target=create, daemon=True), threading.Thread(target=create, daemon=True)]
    for thread in threads:
        thread.start()
    start.wait(timeout=3)
    for thread in threads:
        thread.join(timeout=3)

    values = [results.get_nowait() for _ in threads]
    assert all(not isinstance(value, Exception) for value in values)
    assert {value.scene_id for value in values} == {job.scene_id}
    assert sorted(value.created for value in values) == [False, True]
    assert len([path for path in scene_store.root.iterdir() if path.is_dir() and not path.name.startswith(".")]) == 1
    assert not [path for path in scene_store.root.iterdir() if path.name.startswith(".")]
