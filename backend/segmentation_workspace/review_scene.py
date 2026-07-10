"""Trusted bridge from managed reconstruction jobs into scene review packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field

from backend.scene_package.store import SceneInputArtifact, SceneStore, SceneStoreError, sha256_file

from .models import ContractModel
from .reconstruction_jobs import ReconstructionJobStore
from .reconstruction_models import ReconstructionJobRecord
from .store import SegmentationWorkspaceStore, SegmentationWorkspaceStoreError


SEGMENTATION_ARTIFACT_PREFIX = "/api/segmentation-artifacts/"
REVIEW_REQUIRED_MESSAGE = (
    "This reconstruction completed, but compiler quality gates require review. "
    "Confirm the review-required import before creating the scene."
)


class CreateReviewSceneRequest(ContractModel):
    expected_job_version: int = Field(ge=1, strict=True)
    acknowledge_review_required: bool = False


class ReviewSceneResult(ContractModel):
    imported: bool
    created: bool
    workspace_id: str
    export_id: str
    job_id: str
    scene_id: str
    scene_url: str | None
    package_revision: int | None
    compilation_passed: bool
    semantic_object_count: int
    object_ids: list[str]


@dataclass(frozen=True)
class ReviewSceneInputs:
    workspace_id: str
    export_id: str
    job_id: str
    scene_id: str
    sam_export_dir: Path
    sam_metadata_path: Path
    source_image_path: Path
    unified_scene_plan_path: Path
    reconstruction_result_manifest_path: Path
    compilation_report_path: Path
    compilation_markdown_path: Path
    moge_geometry_path: Path
    moge_summary_path: Path
    object_ids: list[str]
    export_archive_sha256: str
    source_image_id: str
    result_manifest_sha256: str


class ReviewSceneBridgeError(SegmentationWorkspaceStoreError):
    pass


def _artifact_key(url: str) -> str:
    if not url.startswith(SEGMENTATION_ARTIFACT_PREFIX):
        raise ReviewSceneBridgeError("artifact URL is not application-controlled", 500)
    return url[len(SEGMENTATION_ARTIFACT_PREFIX) :]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewSceneBridgeError(f"{label} could not be verified", 500) from exc
    if not isinstance(value, dict):
        raise ReviewSceneBridgeError(f"{label} must be a JSON object", 500)
    return value


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except Exception as exc:
        raise ReviewSceneBridgeError("source image could not be verified", 500) from exc


def _object_ids_from_unified(plan: dict[str, Any]) -> list[str]:
    objects = plan.get("semantic_objects")
    if not isinstance(objects, list):
        raise ReviewSceneBridgeError("Unified scene plan object list is invalid", 500)
    ids = [str(item.get("object_id", "")) for item in objects if isinstance(item, dict)]
    if len(ids) != len(objects):
        raise ReviewSceneBridgeError("Unified scene plan object list is invalid", 500)
    return ids


class ReconstructionReviewSceneBridge:
    def __init__(
        self,
        workspace_store: SegmentationWorkspaceStore,
        job_store: ReconstructionJobStore,
        scene_store: SceneStore,
    ) -> None:
        self.workspace_store = workspace_store
        self.job_store = job_store
        self.scene_store = scene_store

    def status(self, workspace_id: str, job_id: str) -> ReviewSceneResult:
        job = self._eligible_job(workspace_id, job_id, expected_job_version=None)
        return self._status_for_job(job, created=False)

    def create(self, workspace_id: str, job_id: str, request: CreateReviewSceneRequest) -> ReviewSceneResult:
        job = self._eligible_job(workspace_id, job_id, expected_job_version=request.expected_job_version)
        if job.result is None:
            raise ReviewSceneBridgeError("reconstruction result is missing", 409)
        if not job.result.compilation_passed and not request.acknowledge_review_required:
            raise ReviewSceneBridgeError(REVIEW_REQUIRED_MESSAGE, 409)
        inputs = self._resolve_and_verify_inputs(job)
        with self.scene_store._lock(job.scene_id):
            existing = self._existing_scene_result(job)
            if existing:
                return existing
            provenance = self._provenance(job, inputs)
            extras = self._extra_scene_inputs(job.scene_id, inputs)
            try:
                scene = self.scene_store.import_scene(
                    inputs.sam_export_dir,
                    inputs.unified_scene_plan_path,
                    inputs.source_image_path,
                    provenance=provenance,
                    additional_scene_inputs=extras,
                )
            except SceneStoreError as exc:
                raise ReviewSceneBridgeError(str(exc), exc.status_code) from exc
            if not self._scene_matches_job(scene, job):
                raise ReviewSceneBridgeError("created scene failed provenance validation", 500)
            return self._result(job, imported=True, created=True, package_revision=scene.package_revision)

    def _eligible_job(
        self,
        workspace_id: str,
        job_id: str,
        *,
        expected_job_version: int | None,
    ) -> ReconstructionJobRecord:
        try:
            job = self.job_store.get_job(workspace_id, job_id)
        except SegmentationWorkspaceStoreError:
            raise
        if job.workspace_id != workspace_id:
            raise ReviewSceneBridgeError("reconstruction job not found", 404)
        if expected_job_version is not None and job.job_version != expected_job_version:
            raise ReviewSceneBridgeError(
                f"reconstruction job version conflict: expected {expected_job_version}, current {job.job_version}",
                409,
            )
        if job.status != "succeeded" or job.stage != "complete":
            raise ReviewSceneBridgeError("only successful completed reconstruction jobs can create review scenes", 409)
        if job.result is None or not job.scene_id or not job.result.scene_id:
            raise ReviewSceneBridgeError("reconstruction result state is incomplete", 409)
        if job.result.scene_id != job.scene_id:
            raise ReviewSceneBridgeError("reconstruction result scene ID does not match the job", 409)
        if job.result.object_ids != job.object_ids or job.result.semantic_object_count != job.semantic_object_count:
            raise ReviewSceneBridgeError("reconstruction result object IDs do not match the job", 409)
        return job

    def _resolve(self, workspace_id: str, url: str) -> Path:
        try:
            path, _media_type = self.workspace_store._resolve_workspace_artifact(workspace_id, _artifact_key(url))
        except SegmentationWorkspaceStoreError as exc:
            raise ReviewSceneBridgeError(str(exc), exc.status_code) from exc
        return path

    def _resolve_and_verify_inputs(self, job: ReconstructionJobRecord) -> ReviewSceneInputs:
        workspace = self.workspace_store.get_workspace(job.workspace_id)
        export = next((item for item in workspace.exports if item.export_id == job.export_id), None)
        if export is None:
            raise ReviewSceneBridgeError("export not found", 404)
        if export.archive_sha256 != job.export_archive_sha256:
            raise ReviewSceneBridgeError("export archive hash conflict", 409)
        archive_path = self._resolve(job.workspace_id, export.archive_url)
        if sha256_file(archive_path) != job.export_archive_sha256:
            raise ReviewSceneBridgeError("export archive integrity check failed", 500)
        metadata_path = self._resolve(job.workspace_id, export.metadata_url)
        metadata = _read_json(metadata_path, "SAM export metadata")
        metadata_masks = metadata.get("masks")
        if not isinstance(metadata_masks, list):
            raise ReviewSceneBridgeError("SAM export metadata mask list is invalid", 500)
        metadata_ids = [str(item.get("mask_id", "")) for item in metadata_masks if isinstance(item, dict)]
        if metadata_ids != job.object_ids or [mask.object_id for mask in export.masks] != job.object_ids:
            raise ReviewSceneBridgeError("export object IDs do not match the reconstruction job", 409)
        metadata_by_id = {str(item.get("mask_id")): item for item in metadata_masks if isinstance(item, dict)}
        if len(metadata_by_id) != len(metadata_masks):
            raise ReviewSceneBridgeError("SAM export metadata has duplicate object IDs", 500)
        export_dir = metadata_path.parent
        for mask in export.masks:
            raw = metadata_by_id.get(mask.object_id)
            if raw is None:
                raise ReviewSceneBridgeError("SAM export metadata is missing a final mask", 500)
            mask_path = self._resolve(job.workspace_id, mask.mask_url)
            if sha256_file(mask_path) != mask.mask_sha256:
                raise ReviewSceneBridgeError("final mask integrity check failed", 500)
            filename = str(raw.get("filename", ""))
            if (export_dir / filename).resolve() != mask_path.resolve():
                raise ReviewSceneBridgeError("SAM export metadata mask path does not match the export record", 500)
            preview_path = self._resolve(job.workspace_id, mask.preview_url)
            preview_name = str(raw.get("preview_path", ""))
            if not preview_name or (export_dir / preview_name).resolve() != preview_path.resolve():
                raise ReviewSceneBridgeError("SAM export metadata overlay path does not match the export record", 500)
        source_path = self._resolve(job.workspace_id, workspace.source_image.url)
        if sha256_file(source_path) != workspace.source_image.sha256:
            raise ReviewSceneBridgeError("source image integrity check failed", 500)
        if _image_size(source_path) != (int(metadata.get("width", 0)), int(metadata.get("height", 0))):
            raise ReviewSceneBridgeError("source image dimensions do not match SAM export metadata", 500)

        if job.result is None:
            raise ReviewSceneBridgeError("reconstruction result is missing", 409)
        artifacts = job.result.artifacts
        result_manifest_path = self._resolve(job.workspace_id, artifacts.result_manifest_url)
        unified_path = self._resolve(job.workspace_id, artifacts.unified_scene_plan_url)
        compilation_report_path = self._resolve(job.workspace_id, artifacts.compilation_report_url)
        compilation_markdown_path = self._resolve(job.workspace_id, artifacts.compilation_markdown_url)
        moge_geometry_path = self._resolve(job.workspace_id, artifacts.moge_geometry_url)
        moge_summary_path = self._resolve(job.workspace_id, artifacts.moge_summary_url)

        result_manifest = _read_json(result_manifest_path, "reconstruction result manifest")
        if result_manifest.get("job_id") != job.job_id or result_manifest.get("workspace_id") != job.workspace_id or result_manifest.get("export_id") != job.export_id:
            raise ReviewSceneBridgeError("reconstruction result manifest identity does not match the job", 500)
        if result_manifest.get("scene_id") != job.scene_id or result_manifest.get("object_ids") != job.object_ids:
            raise ReviewSceneBridgeError("reconstruction result manifest scene objects do not match the job", 500)
        if result_manifest.get("semantic_object_count") != job.semantic_object_count:
            raise ReviewSceneBridgeError("reconstruction result manifest object count does not match the job", 500)
        if result_manifest.get("compilation_passed") != job.result.compilation_passed:
            raise ReviewSceneBridgeError("reconstruction result manifest quality status does not match the job", 500)
        if result_manifest.get("export_archive_sha256") != job.export_archive_sha256:
            raise ReviewSceneBridgeError("reconstruction result manifest export SHA does not match the job", 500)
        if result_manifest.get("source_image_sha256") != workspace.source_image.sha256:
            raise ReviewSceneBridgeError("reconstruction result manifest source image SHA does not match the workspace", 500)
        unified = _read_json(unified_path, "Unified scene plan")
        if unified.get("scene_id") != job.scene_id or _object_ids_from_unified(unified) != job.object_ids:
            raise ReviewSceneBridgeError("Unified scene plan identity does not match the job", 500)
        if len(job.object_ids) != len(set(job.object_ids)):
            raise ReviewSceneBridgeError("reconstruction job has duplicate object IDs", 500)
        compilation_report = _read_json(compilation_report_path, "compilation report")
        if compilation_report.get("passed") != job.result.compilation_passed:
            raise ReviewSceneBridgeError("compilation report quality status does not match the job", 500)
        return ReviewSceneInputs(
            workspace_id=job.workspace_id,
            export_id=job.export_id,
            job_id=job.job_id,
            scene_id=job.scene_id,
            sam_export_dir=export_dir,
            sam_metadata_path=metadata_path,
            source_image_path=source_path,
            unified_scene_plan_path=unified_path,
            reconstruction_result_manifest_path=result_manifest_path,
            compilation_report_path=compilation_report_path,
            compilation_markdown_path=compilation_markdown_path,
            moge_geometry_path=moge_geometry_path,
            moge_summary_path=moge_summary_path,
            object_ids=list(job.object_ids),
            export_archive_sha256=job.export_archive_sha256,
            source_image_id=workspace.source_image.image_id,
            result_manifest_sha256=sha256_file(result_manifest_path),
        )

    def _provenance(self, job: ReconstructionJobRecord, inputs: ReviewSceneInputs) -> dict[str, Any]:
        return {
            "import_origin": "segmentation_reconstruction_job",
            "segmentation_workspace_id": job.workspace_id,
            "segmentation_export_id": job.export_id,
            "segmentation_export_archive_sha256": job.export_archive_sha256,
            "reconstruction_job_id": job.job_id,
            "reconstruction_job_version": job.job_version,
            "reconstruction_scene_id": job.scene_id,
            "reconstruction_compilation_passed": job.result.compilation_passed if job.result else False,
            "reconstruction_result_manifest_sha256": inputs.result_manifest_sha256,
            "source_image_id": inputs.source_image_id,
        }

    def _extra_scene_inputs(self, scene_id: str, inputs: ReviewSceneInputs) -> list[SceneInputArtifact]:
        def artifact(key: str, artifact_id: str, identifier: str, source: Path, relative: str, media_type: str) -> SceneInputArtifact:
            return SceneInputArtifact(
                artifact_url_key=key,
                artifact_id=f"{artifact_id}:{scene_id}",
                identifier=f"{scene_id}/{identifier}",
                source_path=source,
                relative_path=relative,
                media_type=media_type,
            )

        return [
            artifact("reconstruction_result_manifest", "reconstruction-result-manifest", "reconstruction-result-manifest", inputs.reconstruction_result_manifest_path, "inputs/reconstruction-result-manifest.json", "application/json"),
            artifact("compilation_report", "compilation-report", "compilation-report", inputs.compilation_report_path, "inputs/compilation_report.json", "application/json"),
            artifact("compilation_report_markdown", "compilation-markdown", "compilation-markdown", inputs.compilation_markdown_path, "inputs/compilation_report.md", "text/markdown; charset=utf-8"),
            artifact("moge_geometry", "moge-geometry", "moge-geometry", inputs.moge_geometry_path, "inputs/moge/geometry.npz", "application/octet-stream"),
            artifact("moge_summary", "moge-summary", "moge-summary", inputs.moge_summary_path, "inputs/moge/moge-summary.json", "application/json"),
        ]

    def _scene_matches_job(self, scene: Any, job: ReconstructionJobRecord) -> bool:
        metadata = scene.metadata or {}
        return (
            scene.scene_id == job.scene_id
            and metadata.get("import_origin") == "segmentation_reconstruction_job"
            and metadata.get("segmentation_workspace_id") == job.workspace_id
            and metadata.get("segmentation_export_id") == job.export_id
            and metadata.get("segmentation_export_archive_sha256") == job.export_archive_sha256
            and metadata.get("reconstruction_job_id") == job.job_id
            and [item.object_id for item in scene.semantic_objects] == job.object_ids
        )

    def _existing_scene_result(self, job: ReconstructionJobRecord) -> ReviewSceneResult | None:
        try:
            scene = self.scene_store.get(job.scene_id)
        except SceneStoreError as exc:
            if exc.status_code == 404:
                return None
            raise ReviewSceneBridgeError(str(exc), exc.status_code) from exc
        if not self._scene_matches_job(scene, job):
            raise ReviewSceneBridgeError("scene ID already exists with unrelated provenance", 409)
        return self._result(job, imported=True, created=False, package_revision=scene.package_revision)

    def _status_for_job(self, job: ReconstructionJobRecord, *, created: bool) -> ReviewSceneResult:
        existing = self._existing_scene_result(job)
        if existing:
            return existing
        return self._result(job, imported=False, created=created, package_revision=None)

    def _result(
        self,
        job: ReconstructionJobRecord,
        *,
        imported: bool,
        created: bool,
        package_revision: int | None,
    ) -> ReviewSceneResult:
        compilation_passed = bool(job.result.compilation_passed) if job.result else False
        return ReviewSceneResult(
            imported=imported,
            created=created,
            workspace_id=job.workspace_id,
            export_id=job.export_id,
            job_id=job.job_id,
            scene_id=job.scene_id,
            scene_url=f"/?scene={job.scene_id}" if imported else None,
            package_revision=package_revision,
            compilation_passed=compilation_passed,
            semantic_object_count=job.semantic_object_count,
            object_ids=list(job.object_ids),
        )
