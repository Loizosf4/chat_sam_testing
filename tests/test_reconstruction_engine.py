from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend import reconstruction_engine
from backend.segmentation_workspace.reconstruction_models import (
    ReconstructionArtifactSet,
    ReconstructionJobError,
    ReconstructionJobRecord,
    ReconstructionJobRequest,
    ReconstructionJobResult,
)


def _urls() -> dict[str, str]:
    base = "/api/segmentation-artifacts/reconstruction-results/w/j"
    return {
        "result_manifest_url": f"{base}/result-manifest",
        "unified_scene_plan_url": f"{base}/scene-plan",
        "compilation_report_url": f"{base}/compilation-report",
        "compilation_markdown_url": f"{base}/compilation-markdown",
        "room_plan_url": f"{base}/room-plan",
        "camera_candidates_url": f"{base}/camera-candidates",
        "object_pose_report_url": f"{base}/object-pose-report",
        "placement_report_url": f"{base}/placement-report",
        "collision_report_url": f"{base}/collision-report",
        "confidence_report_url": f"{base}/confidence-report",
        "support_graph_url": f"{base}/support-graph",
        "blender_manifest_url": f"{base}/blender-manifest",
        "moge_geometry_url": f"{base}/moge-geometry",
        "moge_summary_url": f"{base}/moge-summary",
    }


def _request() -> ReconstructionJobRequest:
    return ReconstructionJobRequest(
        expected_workspace_revision=4,
        expected_export_archive_sha256="a" * 64,
        resolution_level=9,
        num_tokens=None,
    )


def _base_record(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    data = {
        "job_id": "1" * 32,
        "job_version": 1,
        "workspace_id": "2" * 32,
        "export_id": "3" * 32,
        "export_archive_sha256": "a" * 64,
        "source_image_id": "4" * 32,
        "status": "queued",
        "stage": "queued",
        "progress_percent": 0,
        "created_at": now,
        "updated_at": now,
        "request": _request(),
        "export_was_stale_at_start": False,
        "semantic_object_count": 1,
        "object_ids": ["object-a"],
        "scene_id": "seg-22222222-33333333-11111111",
    }
    data.update(overrides)
    return data


def test_job_models_validate_queued_running_succeeded_failed_and_reject_bad_combinations() -> None:
    queued = ReconstructionJobRecord.model_validate(_base_record())
    assert queued.status == "queued"

    running = ReconstructionJobRecord.model_validate(
        _base_record(
            job_version=2,
            status="running",
            stage="moge",
            progress_percent=10,
            started_at=datetime.now(timezone.utc),
        )
    )
    assert running.stage == "moge"

    result = ReconstructionJobResult(
        scene_id="seg-22222222-33333333-11111111",
        semantic_object_count=1,
        object_ids=["object-a"],
        compilation_passed=False,
        artifacts=ReconstructionArtifactSet(**_urls()),
    )
    succeeded = ReconstructionJobRecord.model_validate(
        _base_record(
            job_version=3,
            status="succeeded",
            stage="complete",
            progress_percent=100,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            result=result,
        )
    )
    assert succeeded.result.compilation_passed is False

    failed = ReconstructionJobRecord.model_validate(
        _base_record(
            job_version=3,
            status="failed",
            stage="failed",
            progress_percent=100,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            error=ReconstructionJobError(code="compiler", message="failed", stage="compiling", retryable=True),
        )
    )
    assert failed.error.code == "compiler"

    with pytest.raises(ValidationError, match="succeeded jobs"):
        ReconstructionJobRecord.model_validate(_base_record(status="succeeded", stage="complete", progress_percent=100))
    with pytest.raises(ValidationError, match="filesystem"):
        ReconstructionArtifactSet(**{**_urls(), "moge_summary_url": r"C:\tmp\summary.json"})


def test_sanitize_user_message_redacts_paths(tmp_path: Path) -> None:
    message = reconstruction_engine.sanitize_user_message(
        f"failed at {tmp_path}\\stage and C:\\models\\checkpoint.pt",
        extra_roots=[tmp_path],
    )
    assert str(tmp_path) not in message
    assert "C:\\models" not in message
    assert "<redacted-path>" in message


def _moge_output(path: Path) -> None:
    path.mkdir()
    np.savez(
        path / "geometry.npz",
        points=np.zeros((2, 2, 3), np.float32),
        depth=np.ones((2, 2), np.float32),
        valid_mask=np.ones((2, 2), bool),
        intrinsics=np.eye(3, dtype=np.float32),
        normal=np.zeros((2, 2, 3), np.float32),
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "source_image": "C:/secret/source.png",
                "output_dir": "C:/secret/out",
                "output_paths": {"geometry_npz": "C:/secret/geometry.npz"},
                "python_executable": "C:/Python/python.exe",
                "repository": "C:/repo",
                "checkpoint": "C:/model.pt",
                "moge_import_path": "C:/repo/moge",
                "source_image_dimensions": {"width": 2, "height": 2},
                "array_shapes": {"points": [2, 2, 3]},
                "estimated_fov_x_degrees": 60.0,
                "runtime_seconds": 1.2,
                "device": "cuda",
                "model": "moge-2-vitl-normal",
            }
        )
    )


def test_moge_validation_and_sanitized_summary_exclude_paths(tmp_path: Path) -> None:
    moge = tmp_path / "moge"
    _moge_output(moge)
    reconstruction_engine.validate_moge_output(moge)
    summary = reconstruction_engine.sanitized_moge_summary(moge)
    serialized = json.dumps(summary)
    assert "source_image" not in summary
    assert "output_paths" not in summary
    assert "C:/secret" not in serialized
    assert summary["geometry_sha256"]


def test_compiler_command_and_summary_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.write_text("")
    sam = tmp_path / "sam"
    source = tmp_path / "source.png"
    moge = tmp_path / "moge"
    output = tmp_path / "scene"
    handoff = tmp_path / "handoff"
    for path in (sam, moge, output, handoff):
        path.mkdir()
    source.write_bytes(b"png")

    object_ids = ["object-a"]
    (output / "unified_scene_plan.json").write_text(json.dumps({"scene_id": "scene-1", "semantic_object_count": 1, "semantic_objects": [{"object_id": "object-a"}]}))
    (output / "compilation_report.json").write_text(json.dumps({"object_count": 1, "passed": False}))
    (handoff / "blender_one_batch_manifest.json").write_text(json.dumps({"semantic_primitive_count": 1, "semantic_primitives": [{"object_id": "object-a"}]}))

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"success":true,"scene_id":"scene-1","object_count":1,"compilation_passed":false}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = reconstruction_engine.run_compiler(
        sam_dir=sam,
        source_image=source,
        moge_dir=moge,
        output_dir=output,
        scene_id="scene-1",
        handoff_dir=handoff,
        expected_object_ids=object_ids,
        python=python,
    )

    command, kwargs = calls[0]
    assert command[:3] == [str(python), "-m", "experiments.moge_scene_graph_spike.src.compile_unified_v3_scene"]
    assert "--sam-dir" in command and str(sam) in command
    assert kwargs["shell"] is False
    assert result.compilation_passed is False


def test_compiler_rejects_object_id_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "scene"
    handoff = tmp_path / "handoff"
    output.mkdir()
    handoff.mkdir()
    (output / "unified_scene_plan.json").write_text(json.dumps({"scene_id": "scene-1", "semantic_object_count": 1, "semantic_objects": [{"object_id": "other"}]}))
    (output / "compilation_report.json").write_text(json.dumps({"object_count": 1}))
    (handoff / "blender_one_batch_manifest.json").write_text(json.dumps({"semantic_primitive_count": 1, "semantic_primitives": [{"object_id": "other"}]}))

    with pytest.raises(reconstruction_engine.ReconstructionEngineError, match="object IDs"):
        reconstruction_engine.validate_compiler_outputs(
            output_dir=output,
            handoff_dir=handoff,
            scene_id="scene-1",
            expected_object_ids=["object-a"],
        )
