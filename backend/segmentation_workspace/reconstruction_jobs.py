"""Persistent managed reconstruction jobs anchored to workspace exports."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from backend import moge_engine, reconstruction_engine

from .models import SegmentationExportRecord
from .reconstruction_models import (
    PROGRESS_BY_STAGE,
    RECONSTRUCTION_ARTIFACT_PREFIX,
    ReconstructionArtifactSet,
    ReconstructionJobError,
    ReconstructionJobRecord,
    ReconstructionJobRequest,
    ReconstructionJobResult,
)
from .store import (
    SegmentationWorkspaceStore,
    SegmentationWorkspaceStoreError,
    atomic_bytes,
    atomic_json,
    sha256_file,
    valid_id,
)


RECONSTRUCTION_KIND = "reconstruction-results"
DEFAULT_MAX_QUEUED_JOBS = 8
_QUEUE_LOCKS: dict[Path, threading.RLock] = {}
_QUEUE_LOCKS_GUARD = threading.Lock()


class ReconstructionJobStoreError(SegmentationWorkspaceStoreError):
    pass


@dataclass(frozen=True)
class ReconstructionInputs:
    workspace_id: str
    export_id: str
    export_dir: Path
    source_image_path: Path
    source_image_sha256: str
    object_ids: list[str]


def max_queued_jobs() -> int:
    raw = os.getenv("RECONSTRUCTION_MAX_QUEUED_JOBS", "").strip()
    if not raw:
        return DEFAULT_MAX_QUEUED_JOBS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_QUEUED_JOBS


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _artifact_key(url: str) -> str:
    prefix = "/api/segmentation-artifacts/"
    if not url.startswith(prefix):
        raise ReconstructionJobStoreError("artifact URL is not application-controlled", 500)
    return url[len(prefix) :]


def _url(workspace_id: str, job_id: str, suffix: str) -> str:
    return f"{RECONSTRUCTION_ARTIFACT_PREFIX}{workspace_id}/{job_id}/{suffix}"


def _entry(path: Path, relative: str, media_type: str) -> dict[str, str]:
    return {"path": relative.replace("\\", "/"), "media_type": media_type, "sha256": sha256_file(path)}


def _queue_lock_for(root: Path) -> threading.RLock:
    resolved = root.resolve()
    with _QUEUE_LOCKS_GUARD:
        return _QUEUE_LOCKS.setdefault(resolved, threading.RLock())


class ReconstructionJobStore:
    def __init__(
        self,
        workspace_store: SegmentationWorkspaceStore,
        *,
        clock: Callable[[], datetime] = _now,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self.workspace_store = workspace_store
        self.clock = clock
        self.id_factory = id_factory
        # Lock order for queue-wide operations is always queue lock, then
        # workspace locks in sorted workspace-id order.
        self._queue_lock = _queue_lock_for(workspace_store.root)
        self.reconcile_unfinished()

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self.workspace_store._workspace_dir(workspace_id)

    def _jobs_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "reconstruction-jobs"

    def _job_path(self, workspace_id: str, job_id: str) -> Path:
        if not valid_id(job_id):
            raise ReconstructionJobStoreError("unsafe reconstruction job ID")
        return self._jobs_dir(workspace_id) / f"{job_id}.json"

    def _reconstructions_dir(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "reconstructions"

    def _load_job_path(self, path: Path) -> ReconstructionJobRecord:
        try:
            return ReconstructionJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ReconstructionJobStoreError("stored reconstruction job is invalid", 500) from exc

    def _save_job(self, job: ReconstructionJobRecord) -> None:
        path = self._job_path(job.workspace_id, job.job_id)
        atomic_bytes(path, (job.model_dump_json(indent=2) + "\n").encode("utf-8"))

    def _all_jobs_unlocked(self, workspace_id: str) -> list[ReconstructionJobRecord]:
        jobs_dir = self._jobs_dir(workspace_id)
        if not jobs_dir.is_dir():
            return []
        return [self._load_job_path(path) for path in jobs_dir.glob("*.json")]

    def _all_workspace_ids(self) -> list[str]:
        if not self.workspace_store.root.is_dir():
            return []
        return sorted(path.name for path in self.workspace_store.root.iterdir() if path.is_dir() and valid_id(path.name))

    def _active_jobs_unlocked(self, workspace_id: str) -> list[ReconstructionJobRecord]:
        return [job for job in self._all_jobs_unlocked(workspace_id) if job.status in {"queued", "running"}]

    def _active_counts_under_queue_lock(self) -> dict[str, int]:
        counts = {"queued": 0, "running": 0}
        for workspace_id in self._all_workspace_ids():
            with self.workspace_store._lock(workspace_id):
                for job in self._active_jobs_unlocked(workspace_id):
                    counts[str(job.status)] += 1
        return counts

    def active_count(self) -> int:
        with self._queue_lock:
            counts = self._active_counts_under_queue_lock()
            return counts["queued"] + counts["running"]

    def active_count_by_status(self) -> dict[str, int]:
        with self._queue_lock:
            return self._active_counts_under_queue_lock()

    def workspace_has_active_jobs(self, workspace_id: str) -> bool:
        with self.workspace_store._lock(workspace_id):
            return bool(self._active_jobs_unlocked(workspace_id))

    def list_jobs(self, workspace_id: str) -> list[ReconstructionJobRecord]:
        self.workspace_store.get_workspace(workspace_id)
        jobs = self._all_jobs_unlocked(workspace_id)
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def get_job(self, workspace_id: str, job_id: str) -> ReconstructionJobRecord:
        self.workspace_store.get_workspace(workspace_id)
        path = self._job_path(workspace_id, job_id)
        if not path.is_file():
            raise ReconstructionJobStoreError("reconstruction job not found", 404)
        return self._load_job_path(path)

    def create_job(
        self,
        workspace_id: str,
        export_id: str,
        request: ReconstructionJobRequest,
    ) -> ReconstructionJobRecord:
        with self._queue_lock:
            counts = self._active_counts_under_queue_lock()
            if counts["queued"] + counts["running"] >= max_queued_jobs():
                raise ReconstructionJobStoreError("reconstruction queue limit reached", 409)
            with self.workspace_store._lock(workspace_id):
                workspace = self.workspace_store._load(workspace_id)
                if workspace.workspace_revision != request.expected_workspace_revision:
                    raise ReconstructionJobStoreError(
                        f"workspace revision conflict: expected {request.expected_workspace_revision}, current {workspace.workspace_revision}",
                        409,
                    )
                export = next((item for item in workspace.exports if item.export_id == export_id), None)
                if export is None:
                    raise ReconstructionJobStoreError("export not found", 404)
                if export.archive_sha256 != request.expected_export_archive_sha256:
                    raise ReconstructionJobStoreError("export archive hash conflict", 409)
                for job in self._active_jobs_unlocked(workspace_id):
                    if job.export_id == export_id:
                        raise ReconstructionJobStoreError("a reconstruction job is already active for this export", 409)
                inputs = self._verify_inputs_locked(workspace_id, export)
                stale = self.workspace_store.export_is_stale(workspace_id, export_id)
                job_id = self.id_factory()
                while (self._job_path(workspace_id, job_id)).exists():
                    job_id = self.id_factory()
                scene_id = f"seg-{workspace_id[:8]}-{export_id[:8]}-{job_id[:8]}"
                now = self.clock()
                job = ReconstructionJobRecord(
                    job_id=job_id,
                    job_version=1,
                    workspace_id=workspace_id,
                    export_id=export_id,
                    export_archive_sha256=export.archive_sha256,
                    source_image_id=workspace.source_image.image_id,
                    status="queued",
                    stage="queued",
                    progress_percent=PROGRESS_BY_STAGE["queued"],
                    created_at=now,
                    updated_at=now,
                    request=request,
                    export_was_stale_at_start=stale,
                    semantic_object_count=len(inputs.object_ids),
                    object_ids=inputs.object_ids,
                    scene_id=scene_id,
                )
                self._jobs_dir(workspace_id).mkdir(parents=True, exist_ok=True)
                self._save_job(job)
                return job

    def _verify_inputs_locked(self, workspace_id: str, export: SegmentationExportRecord) -> ReconstructionInputs:
        keys = [
            _artifact_key(export.archive_url),
            _artifact_key(export.metadata_url),
            _artifact_key(export.quality_report_url),
            _artifact_key(export.quality_markdown_url),
            _artifact_key(export.combined_preview_url),
        ]
        for mask in export.masks:
            keys.append(_artifact_key(mask.mask_url))
            keys.append(_artifact_key(mask.preview_url))
        resolved: dict[str, Path] = {}
        for key in keys:
            path, _media_type = self.workspace_store._resolve_workspace_artifact(workspace_id, key)
            resolved[key] = path
        archive_key = _artifact_key(export.archive_url)
        if sha256_file(resolved[archive_key]) != export.archive_sha256:
            raise ReconstructionJobStoreError("export archive integrity check failed", 500)
        metadata_path = resolved[_artifact_key(export.metadata_url)]
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReconstructionJobStoreError("export metadata could not be verified", 500) from exc
        metadata_ids = [item.get("mask_id") for item in metadata.get("masks", []) if isinstance(item, dict)]
        object_ids = [mask.object_id for mask in export.masks]
        if set(metadata_ids) != set(object_ids) or len(metadata_ids) != len(object_ids):
            raise ReconstructionJobStoreError("export metadata object IDs do not match the export record", 500)
        workspace = self.workspace_store._load(workspace_id)
        source_key = _artifact_key(workspace.source_image.url)
        source_path, _source_media_type = self.workspace_store._resolve_workspace_artifact(workspace_id, source_key)
        if sha256_file(source_path) != workspace.source_image.sha256:
            raise ReconstructionJobStoreError("source image integrity check failed", 500)
        return ReconstructionInputs(
            workspace_id=workspace_id,
            export_id=export.export_id,
            export_dir=resolved[archive_key].parent,
            source_image_path=source_path,
            source_image_sha256=workspace.source_image.sha256,
            object_ids=object_ids,
        )

    def prepare_execution_inputs(self, job: ReconstructionJobRecord) -> ReconstructionInputs:
        with self.workspace_store._lock(job.workspace_id):
            workspace = self.workspace_store._load(job.workspace_id)
            export = next((item for item in workspace.exports if item.export_id == job.export_id), None)
            if export is None:
                raise ReconstructionJobStoreError("export not found", 404)
            if export.archive_sha256 != job.export_archive_sha256:
                raise ReconstructionJobStoreError("export archive hash conflict", 409)
            inputs = self._verify_inputs_locked(job.workspace_id, export)
            if set(inputs.object_ids) != set(job.object_ids):
                raise ReconstructionJobStoreError("export object IDs no longer match the reconstruction job", 409)
            return inputs

    def transition_running(self, job: ReconstructionJobRecord, stage: str) -> ReconstructionJobRecord:
        with self.workspace_store._lock(job.workspace_id):
            current = self.get_job(job.workspace_id, job.job_id)
            if current.status not in {"queued", "running"}:
                raise ReconstructionJobStoreError("terminal reconstruction job cannot transition")
            if stage not in {"validating", "moge", "compiling", "publishing"}:
                raise ReconstructionJobStoreError("invalid reconstruction running stage")
            now = self.clock()
            document = current.model_dump(mode="json")
            document.update(
                {
                    "status": "running",
                    "stage": stage,
                    "progress_percent": PROGRESS_BY_STAGE[stage],
                    "job_version": current.job_version + 1,
                    "updated_at": now.isoformat(),
                    "started_at": (current.started_at or now).isoformat(),
                }
            )
            updated = ReconstructionJobRecord.model_validate(document)
            self._save_job(updated)
            return updated

    def fail_job(self, job: ReconstructionJobRecord, *, code: str, message: str, stage: str, retryable: bool = True) -> ReconstructionJobRecord:
        with self.workspace_store._lock(job.workspace_id):
            current = self.get_job(job.workspace_id, job.job_id)
            if current.status in {"succeeded", "failed", "interrupted"}:
                return current
            now = self.clock()
            document = current.model_dump(mode="json")
            document.update(
                {
                    "status": "failed",
                    "stage": "failed",
                    "progress_percent": PROGRESS_BY_STAGE["failed"],
                    "job_version": current.job_version + 1,
                    "updated_at": now.isoformat(),
                    "finished_at": now.isoformat(),
                    "error": ReconstructionJobError(code=code, message=message, stage=stage, retryable=retryable).model_dump(mode="json"),
                }
            )
            updated = ReconstructionJobRecord.model_validate(document)
            self._save_job(updated)
            return updated

    def interrupt_job(self, job: ReconstructionJobRecord, message: str = "The application stopped before reconstruction completed.") -> ReconstructionJobRecord:
        with self.workspace_store._lock(job.workspace_id):
            current = self.get_job(job.workspace_id, job.job_id)
            if current.status not in {"queued", "running"}:
                return current
            now = self.clock()
            document = current.model_dump(mode="json")
            document.update(
                {
                    "status": "interrupted",
                    "stage": "interrupted",
                    "progress_percent": PROGRESS_BY_STAGE["interrupted"],
                    "job_version": current.job_version + 1,
                    "updated_at": now.isoformat(),
                    "finished_at": now.isoformat(),
                    "error": ReconstructionJobError(
                        code="application_interrupted",
                        message=message,
                        stage="interrupted",
                        retryable=True,
                    ).model_dump(mode="json"),
                }
            )
            updated = ReconstructionJobRecord.model_validate(document)
            self._save_job(updated)
            return updated

    def publish_success(
        self,
        job: ReconstructionJobRecord,
        *,
        staging_dir: Path,
        compiler_result: reconstruction_engine.CompilerResult,
        source_image_sha256: str,
    ) -> ReconstructionJobRecord:
        with self.workspace_store._lock(job.workspace_id):
            current = self.get_job(job.workspace_id, job.job_id)
            if current.status != "running" or current.stage != "publishing":
                raise ReconstructionJobStoreError("reconstruction job is not ready to publish", 500)
            self._write_sanitized_moge_summary(staging_dir / "moge")
            artifacts, entries = self._build_result_artifacts(current, staging_dir, compiler_result, source_image_sha256)
            result = ReconstructionJobResult(
                scene_id=current.scene_id,
                semantic_object_count=current.semantic_object_count,
                object_ids=current.object_ids,
                compilation_passed=compiler_result.compilation_passed,
                artifacts=artifacts,
            )
            manifest = self._result_manifest(current, result, source_image_sha256, staging_dir)
            manifest_path = staging_dir / "reconstruction-result-manifest.json"
            atomic_json(manifest_path, manifest)
            entries[f"{RECONSTRUCTION_KIND}/{current.workspace_id}/{current.job_id}/result-manifest"] = _entry(
                manifest_path,
                f"reconstructions/{current.job_id}/reconstruction-result-manifest.json",
                "application/json",
            )
            final_dir = self._reconstructions_dir(current.workspace_id) / current.job_id
            if final_dir.exists():
                raise ReconstructionJobStoreError("reconstruction result already exists", 409)
            original_index = self.workspace_store._index(current.workspace_id)
            index = {key: dict(value) for key, value in original_index.items()}
            index.update(entries)
            target_published = False
            index_written = False
            job_written = False
            try:
                os.replace(staging_dir, final_dir)
                target_published = True
                self.workspace_store._write_index(current.workspace_id, index)
                index_written = True
                now = self.clock()
                document = current.model_dump(mode="json")
                document.update(
                    {
                        "status": "succeeded",
                        "stage": "complete",
                        "progress_percent": PROGRESS_BY_STAGE["complete"],
                        "job_version": current.job_version + 1,
                        "updated_at": now.isoformat(),
                        "finished_at": now.isoformat(),
                        "result": result.model_dump(mode="json"),
                    }
                )
                updated = ReconstructionJobRecord.model_validate(document)
                self._save_job(updated)
                job_written = True
                return updated
            except Exception:
                if index_written and not job_written:
                    try:
                        self.workspace_store._write_index(current.workspace_id, original_index)
                    except Exception:
                        pass
                if target_published:
                    shutil.rmtree(final_dir, ignore_errors=True)
                raise

    def _write_sanitized_moge_summary(self, moge_dir: Path) -> None:
        summary = reconstruction_engine.sanitized_moge_summary(moge_dir)
        atomic_json(moge_dir / "moge-summary.json", summary)

    def _build_result_artifacts(
        self,
        job: ReconstructionJobRecord,
        staging_dir: Path,
        compiler_result: reconstruction_engine.CompilerResult,
        source_image_sha256: str,
    ) -> tuple[ReconstructionArtifactSet, dict[str, dict[str, str]]]:
        del compiler_result, source_image_sha256
        scene = staging_dir / "scene"
        moge = staging_dir / "moge"
        handoff = staging_dir / "handoff"
        required = {
            "scene-plan": (scene / "unified_scene_plan.json", "scene/unified_scene_plan.json", "application/json"),
            "compilation-report": (scene / "compilation_report.json", "scene/compilation_report.json", "application/json"),
            "compilation-markdown": (scene / "compilation_report.md", "scene/compilation_report.md", "text/markdown; charset=utf-8"),
            "room-plan": (scene / "room_plan.json", "scene/room_plan.json", "application/json"),
            "camera-candidates": (scene / "camera_candidates.json", "scene/camera_candidates.json", "application/json"),
            "object-pose-report": (scene / "object_pose_report.json", "scene/object_pose_report.json", "application/json"),
            "placement-report": (scene / "placement_report.json", "scene/placement_report.json", "application/json"),
            "collision-report": (scene / "collision_report.json", "scene/collision_report.json", "application/json"),
            "confidence-report": (scene / "confidence_report.json", "scene/confidence_report.json", "application/json"),
            "support-graph": (scene / "support_graph.json", "scene/support_graph.json", "application/json"),
            "blender-manifest": (handoff / "blender_one_batch_manifest.json", "handoff/blender_one_batch_manifest.json", "application/json"),
            "moge-geometry": (moge / "geometry.npz", "moge/geometry.npz", "application/octet-stream"),
            "moge-summary": (moge / "moge-summary.json", "moge/moge-summary.json", "application/json"),
        }
        optional = {
            "overview": (scene / "clean_scene_plan_overview.png", "scene/clean_scene_plan_overview.png", "image/png"),
            "projected-primitives": (scene / "projected_primitives_overlay.png", "scene/projected_primitives_overlay.png", "image/png"),
            "room-camera": (scene / "room_and_camera_overlay.png", "scene/room_and_camera_overlay.png", "image/png"),
            "confidence-overview": (scene / "confidence_overview.png", "scene/confidence_overview.png", "image/png"),
            "ambiguity-overview": (scene / "ambiguity_overview.png", "scene/ambiguity_overview.png", "image/png"),
            "depth-preview": (moge / "depth_preview.png", "moge/depth_preview.png", "image/png"),
            "normal-preview": (moge / "normal_preview.png", "moge/normal_preview.png", "image/png"),
            "valid-mask-preview": (moge / "valid_mask_preview.png", "moge/valid_mask_preview.png", "image/png"),
        }
        entries: dict[str, dict[str, str]] = {}
        urls: dict[str, str | None] = {}
        for suffix, (path, relative, media_type) in required.items():
            if not path.is_file():
                raise ReconstructionJobStoreError(f"required reconstruction artifact is missing: {suffix}", 500)
            key = f"{RECONSTRUCTION_KIND}/{job.workspace_id}/{job.job_id}/{suffix}"
            entries[key] = _entry(path, f"reconstructions/{job.job_id}/{relative}", media_type)
            urls[suffix] = _url(job.workspace_id, job.job_id, suffix)
        for suffix, (path, relative, media_type) in optional.items():
            if path.is_file():
                key = f"{RECONSTRUCTION_KIND}/{job.workspace_id}/{job.job_id}/{suffix}"
                entries[key] = _entry(path, f"reconstructions/{job.job_id}/{relative}", media_type)
                urls[suffix] = _url(job.workspace_id, job.job_id, suffix)
            else:
                urls[suffix] = None
        return (
            ReconstructionArtifactSet(
                result_manifest_url=_url(job.workspace_id, job.job_id, "result-manifest"),
                unified_scene_plan_url=urls["scene-plan"],
                compilation_report_url=urls["compilation-report"],
                compilation_markdown_url=urls["compilation-markdown"],
                room_plan_url=urls["room-plan"],
                camera_candidates_url=urls["camera-candidates"],
                object_pose_report_url=urls["object-pose-report"],
                placement_report_url=urls["placement-report"],
                collision_report_url=urls["collision-report"],
                confidence_report_url=urls["confidence-report"],
                support_graph_url=urls["support-graph"],
                overview_url=urls["overview"],
                projected_primitives_url=urls["projected-primitives"],
                room_camera_url=urls["room-camera"],
                confidence_overview_url=urls["confidence-overview"],
                ambiguity_overview_url=urls["ambiguity-overview"],
                blender_manifest_url=urls["blender-manifest"],
                moge_geometry_url=urls["moge-geometry"],
                moge_summary_url=urls["moge-summary"],
                depth_preview_url=urls["depth-preview"],
                normal_preview_url=urls["normal-preview"],
                valid_mask_preview_url=urls["valid-mask-preview"],
            ),
            entries,
        )

    def _result_manifest(
        self,
        job: ReconstructionJobRecord,
        result: ReconstructionJobResult,
        source_image_sha256: str,
        staging_dir: Path,
    ) -> dict[str, Any]:
        artifacts = result.artifacts.model_dump(mode="json")
        return {
            "schema_version": "1.0.0",
            "job_id": job.job_id,
            "workspace_id": job.workspace_id,
            "export_id": job.export_id,
            "export_archive_sha256": job.export_archive_sha256,
            "scene_id": job.scene_id,
            "source_image_id": job.source_image_id,
            "source_image_sha256": source_image_sha256,
            "semantic_object_count": job.semantic_object_count,
            "object_ids": job.object_ids,
            "moge_geometry_sha256": sha256_file(staging_dir / "moge" / "geometry.npz"),
            "unified_scene_plan_sha256": sha256_file(staging_dir / "scene" / "unified_scene_plan.json"),
            "compilation_report_sha256": sha256_file(staging_dir / "scene" / "compilation_report.json"),
            "blender_manifest_sha256": sha256_file(staging_dir / "handoff" / "blender_one_batch_manifest.json"),
            "compilation_passed": result.compilation_passed,
            "created_at": self.clock().isoformat(),
            "artifacts": artifacts,
        }

    def reconcile_unfinished(self) -> None:
        with self._queue_lock:
            for workspace_id in self._all_workspace_ids():
                with self.workspace_store._lock(workspace_id):
                    jobs = self._all_jobs_unlocked(workspace_id)
                    for job in jobs:
                        if job.status in {"queued", "running"}:
                            self.interrupt_job(job)
                    reconstructions = self._reconstructions_dir(workspace_id)
                    if reconstructions.is_dir():
                        for child in reconstructions.iterdir():
                            if child.name.startswith(".") and child.is_dir():
                                shutil.rmtree(child, ignore_errors=True)


class ReconstructionJobManager:
    def __init__(
        self,
        store: ReconstructionJobStore,
        *,
        executor: Any | None = None,
        moge_runner: Callable[..., dict[str, Any]] | None = None,
        compiler_runner: Callable[..., reconstruction_engine.CompilerResult] | None = None,
    ) -> None:
        self.store = store
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="seg-reconstruction")
        self.moge_runner = moge_runner or moge_engine.run_inference_from_path
        self.compiler_runner = compiler_runner or reconstruction_engine.run_compiler
        atexit.register(self.shutdown)

    def submit(self, job: ReconstructionJobRecord) -> None:
        try:
            future = self.executor.submit(self._run_job, job.job_id, job.workspace_id)
        except Exception as exc:
            self.store.fail_job(
                job,
                code="queue_submission_failed",
                message="The reconstruction job could not be submitted to the worker queue.",
                stage="queued",
                retryable=True,
            )
            raise ReconstructionJobStoreError(
                "The reconstruction job could not be submitted to the worker queue.",
                503,
            ) from exc
        add_done_callback = getattr(future, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(lambda completed, submitted=job: self._handle_cancelled_before_run(submitted, completed))

    def _handle_cancelled_before_run(self, job: ReconstructionJobRecord, future: Any) -> None:
        try:
            if not future.cancelled():
                return
            current = self.store.get_job(job.workspace_id, job.job_id)
            if current.status == "queued":
                self.store.interrupt_job(current, "The reconstruction job was cancelled before it started.")
        except Exception:
            return

    def shutdown(self) -> None:
        shutdown = getattr(self.executor, "shutdown", None)
        if callable(shutdown):
            shutdown(wait=False, cancel_futures=True)

    def _run_job(self, job_id: str, workspace_id: str) -> None:
        staging_dir: Path | None = None
        stage = "validating"
        try:
            job = self.store.get_job(workspace_id, job_id)
            job = self.store.transition_running(job, "validating")
            inputs = self.store.prepare_execution_inputs(job)
            reconstructions_root = self.store._reconstructions_dir(workspace_id)
            reconstructions_root.mkdir(parents=True, exist_ok=True)
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{job_id}.", dir=reconstructions_root))
            moge_dir = staging_dir / "moge"
            scene_dir = staging_dir / "scene"
            handoff_dir = staging_dir / "handoff"
            stage = "moge"
            job = self.store.transition_running(job, "moge")
            self.moge_runner(
                image_path=inputs.source_image_path,
                output_dir=moge_dir,
                requested_outputs=reconstruction_engine.REQUIRED_MOGE_OUTPUTS,
                resolution_level=job.request.resolution_level,
                num_tokens=job.request.num_tokens,
            )
            reconstruction_engine.validate_moge_output(moge_dir)
            stage = "compiling"
            job = self.store.transition_running(job, "compiling")
            compiler_result = self.compiler_runner(
                sam_dir=inputs.export_dir,
                source_image=inputs.source_image_path,
                moge_dir=moge_dir,
                output_dir=scene_dir,
                scene_id=job.scene_id,
                handoff_dir=handoff_dir,
                expected_object_ids=job.object_ids,
            )
            stage = "publishing"
            job = self.store.transition_running(job, "publishing")
            self.store.publish_success(
                job,
                staging_dir=staging_dir,
                compiler_result=compiler_result,
                source_image_sha256=inputs.source_image_sha256,
            )
            staging_dir = None
        except reconstruction_engine.ReconstructionEngineError as exc:
            message = reconstruction_engine.sanitize_user_message(exc)
            self.store.fail_job(self.store.get_job(workspace_id, job_id), code=exc.code, message=message, stage=stage, retryable=exc.retryable)
        except moge_engine.MogeEngineError as exc:
            message = reconstruction_engine.sanitize_user_message(exc, extra_roots=[self.store.workspace_store.root])
            self.store.fail_job(self.store.get_job(workspace_id, job_id), code="moge_failed", message=message, stage=stage, retryable=True)
        except Exception as exc:
            message = reconstruction_engine.sanitize_user_message(exc, extra_roots=[self.store.workspace_store.root])
            self.store.fail_job(self.store.get_job(workspace_id, job_id), code="reconstruction_failed", message=message, stage=stage, retryable=True)
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
