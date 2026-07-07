"""Atomic, filesystem-backed persistence for browser-safe scene packages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid5

from PIL import Image, ImageChops

from .adapter import REVISION_NAMESPACE, SceneAdapterError, adapt_scene_package
from .models import ArtifactReference, MaskQualityReport, MaskRevision, Scene, Transform
from .serialization import RevisionConflictError, dump_scene, load_scene, validate_optimistic_lock


ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
MAX_IMAGE_PIXELS = 100_000_000


class SceneStoreError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def valid_id(value: str) -> bool:
    return 1 <= len(value) <= 192 and value[0].isalnum() and all(c in ID_CHARS for c in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def inspect_image(path: Path, *, binary_mask: bool = False) -> tuple[int, int, str, MaskQualityReport | None]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            media_type = IMAGE_FORMATS.get(image.format or "")
            if media_type is None or width * height > MAX_IMAGE_PIXELS:
                raise SceneStoreError("unsupported image format or dimensions")
            report = None
            if binary_mask:
                if image.format != "PNG" or image.mode not in {"1", "L", "P"}:
                    raise SceneStoreError("binary masks must be single-channel PNG images")
                grayscale = image.convert("L")
                extrema = grayscale.getextrema()
                histogram = grayscale.histogram()
                if any(histogram[1:255]):
                    raise SceneStoreError("mask pixels must contain only 0 and 255")
                bbox = grayscale.getbbox()
                area = histogram[255]
                report = MaskQualityReport(
                    binary=True,
                    non_empty=area > 0,
                    area_pixels=area,
                    bbox_xyxy=list(bbox) if bbox else None,
                    warnings=[] if area else ["empty_mask"],
                    metrics={"pixel_min": extrema[0], "pixel_max": extrema[1]},
                )
            return width, height, media_type, report
    except SceneStoreError:
        raise
    except Exception as exc:
        raise SceneStoreError("invalid image artifact") from exc


def normalize_uploaded_binary_mask(path: Path) -> None:
    """Validate browser canvas PNG output and rewrite it as canonical grayscale.

    Canvas ``toBlob('image/png')`` produces RGBA even when every displayed
    pixel is grayscale.  Revision artifacts remain strict single-channel PNGs;
    this function only accepts opaque RGB/RGBA input whose color channels are
    identical and binary, then normalizes it before hashing and persistence.
    """
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise SceneStoreError("binary masks must be PNG images")
            if image.mode in {"1", "L", "P"}:
                return
            if image.mode not in {"RGB", "RGBA"}:
                raise SceneStoreError("binary masks must be grayscale or opaque grayscale RGBA PNG images")
            rgba = image.convert("RGBA")
            red, green, blue, alpha = rgba.split()
            if alpha.getextrema() != (255, 255):
                raise SceneStoreError("binary mask alpha must be fully opaque")
            if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
                raise SceneStoreError("binary mask RGB channels must be identical")
            histogram = red.histogram()
            if any(histogram[1:255]):
                raise SceneStoreError("mask pixels must contain only 0 and 255")
            canonical = red.copy()
        canonical.save(path, format="PNG", optimize=False)
    except SceneStoreError:
        raise
    except Exception as exc:
        raise SceneStoreError("invalid image artifact") from exc


class SceneStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, scene_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(scene_id, threading.RLock())

    def _scene_dir(self, scene_id: str) -> Path:
        if not valid_id(scene_id):
            raise SceneStoreError("invalid scene ID")
        candidate = (self.root / scene_id).resolve()
        if candidate.parent != self.root:
            raise SceneStoreError("scene path escapes artifact root")
        return candidate

    def get(self, scene_id: str) -> Scene:
        path = self._scene_dir(scene_id) / "scene.json"
        if not path.is_file():
            raise SceneStoreError("scene not found", 404)
        try:
            return load_scene(path)
        except Exception as exc:
            raise SceneStoreError("stored scene package is invalid", 500) from exc

    def _index(self, scene_id: str) -> dict[str, dict[str, str]]:
        path = self._scene_dir(scene_id) / "artifact-index.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SceneStoreError("artifact index not found", 404) from exc
        if not isinstance(value, dict):
            raise SceneStoreError("invalid artifact index", 500)
        return value

    def resolve_artifact(self, kind: str, identifier: str) -> tuple[Path, str]:
        if kind not in {"source-images", "masks", "mask-overlays", "scene-inputs", "revision-deltas"}:
            raise SceneStoreError("unknown artifact kind", 404)
        if not identifier or ".." in identifier.split("/") or "\\" in identifier:
            raise SceneStoreError("invalid artifact identifier")
        # Artifact IDs are globally located by a bounded scan of registered indexes;
        # paths supplied by callers are never resolved directly on disk.
        key = f"{kind}/{identifier}"
        for scene_dir in self.root.iterdir():
            if not scene_dir.is_dir():
                continue
            try:
                entry = self._index(scene_dir.name).get(key)
            except SceneStoreError:
                continue
            if not entry:
                continue
            base = scene_dir.resolve()
            path = (base / entry["path"]).resolve()
            if base not in path.parents or not path.is_file():
                raise SceneStoreError("artifact index escaped its root", 500)
            if sha256_file(path) != entry["sha256"]:
                raise SceneStoreError("artifact integrity check failed", 500)
            return path, entry["media_type"]
        raise SceneStoreError("artifact not found", 404)

    def import_scene(self, sam_dir: Path, unified_manifest: Path, source_image: Path) -> Scene:
        metadata_path = sam_dir / "metadata.json"
        if not metadata_path.is_file():
            raise SceneStoreError("SAM archive must contain metadata.json")
        try:
            untrusted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_masks = untrusted_metadata["masks"]
            if not isinstance(raw_masks, list):
                raise TypeError
            for raw in raw_masks:
                if not isinstance(raw, dict):
                    raise TypeError
                for field in ("filename", "preview_path"):
                    if field not in raw or raw[field] is None:
                        continue
                    candidate = (sam_dir / str(raw[field])).resolve()
                    if sam_dir.resolve() not in candidate.parents:
                        raise SceneStoreError(f"{field} escapes the SAM archive")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SceneStoreError("invalid SAM metadata.json") from exc
        source_width, source_height, source_mime, _ = inspect_image(source_image)
        try:
            scene = adapt_scene_package(metadata_path, unified_manifest, source_image_path=source_image)
        except (SceneAdapterError, ValueError, KeyError) as exc:
            raise SceneStoreError(str(exc)) from exc
        if (source_width, source_height) != (scene.image_dimensions.width, scene.image_dimensions.height):
            raise SceneStoreError("source image dimensions do not match SAM metadata")

        target = self._scene_dir(scene.scene_id)
        with self._lock(scene.scene_id):
            if target.exists():
                raise SceneStoreError("scene already exists", 409)
            staging = Path(tempfile.mkdtemp(prefix=f".{scene.scene_id}.", dir=self.root))
            try:
                artifacts = staging / "artifacts"
                index: dict[str, dict[str, str]] = {}

                def register(key: str, source: Path, relative: str, media_type: str) -> None:
                    destination = artifacts / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    index[key] = {
                        "path": str(destination.relative_to(staging)).replace("\\", "/"),
                        "media_type": media_type,
                        "sha256": sha256_file(destination),
                    }

                source_ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[source_mime]
                source_id = scene.source_image.artifact_id.split(":", 1)[-1]
                register(f"source-images/{source_id}", source_image, f"source/source{source_ext}", source_mime)
                register(f"scene-inputs/{scene.scene_id}/sam-manifest", metadata_path, "inputs/sam-metadata.json", "application/json")
                register(f"scene-inputs/{scene.scene_id}/unified-v3-1-1", unified_manifest, "inputs/unified-scene-plan.json", "application/json")
                # The adapter's MoGe entry is descriptive when no standalone
                # geometry file was supplied; do not publish a broken URL.
                scene.artifact_urls.pop("moge_geometry", None)

                raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                masks_by_id = {str(item["mask_id"]): item for item in raw_metadata["masks"]}
                for item in scene.semantic_objects:
                    raw = masks_by_id[item.object_id]
                    mask_source = (sam_dir / str(raw["filename"])).resolve()
                    if sam_dir.resolve() not in mask_source.parents:
                        raise SceneStoreError("mask filename escapes SAM archive")
                    width, height, media, report = inspect_image(mask_source, binary_mask=True)
                    if media != "image/png" or (width, height) != (source_width, source_height):
                        raise SceneStoreError(f"mask dimensions mismatch for {item.object_id}")
                    if sha256_file(mask_source) != item.mask_hash:
                        raise SceneStoreError("mask hash changed during import")
                    register(f"masks/{item.object_id}/{item.mask_revision}", mask_source, f"revisions/{item.object_id}/{item.mask_revision}/mask.png", "image/png")
                    preview_name = raw.get("preview_path")
                    if preview_name:
                        preview = (sam_dir / str(preview_name)).resolve()
                        if sam_dir.resolve() not in preview.parents or not preview.is_file():
                            raise SceneStoreError(f"overlay missing for {item.object_id}")
                        pw, ph, preview_mime, _ = inspect_image(preview)
                        if (pw, ph) != (source_width, source_height):
                            raise SceneStoreError(f"overlay dimensions mismatch for {item.object_id}")
                        register(f"mask-overlays/{item.object_id}/{item.mask_revision}", preview, f"revisions/{item.object_id}/{item.mask_revision}/overlay.png", preview_mime)
                    else:
                        item.overlay_url = None
                    # Trust pixels, not claimed area/bbox, for the persisted quality report.
                    revision = next(r for r in scene.mask_revisions if r.revision_id == item.mask_revision)
                    revision.quality_report = report  # validated assignment

                quality = sam_dir / "mask_quality_report.json"
                if quality.is_file():
                    json.loads(quality.read_text(encoding="utf-8"))
                    register(f"scene-inputs/{scene.scene_id}/mask-quality-report", quality, "inputs/mask-quality-report.json", "application/json")
                    scene.artifact_urls["mask_quality_report"] = ArtifactReference(
                        artifact_id=f"mask-quality:{scene.scene_id}",
                        url=f"/artifacts/scene-inputs/{scene.scene_id}/mask-quality-report",
                        media_type="application/json",
                        sha256=sha256_file(quality),
                    )
                atomic_bytes(staging / "scene.json", dump_scene(scene).encode())
                atomic_json(staging / "artifact-index.json", index)
                os.replace(staging, target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return scene

    def update_object(self, scene_id: str, object_id: str, expected_package_revision: int, expected_object_version: int, changes: dict[str, Any]) -> Scene:
        with self._lock(scene_id):
            scene = self.get(scene_id)
            try:
                validate_optimistic_lock(scene, expected_package_revision=expected_package_revision, expected_object_versions={object_id: expected_object_version})
            except RevisionConflictError as exc:
                raise SceneStoreError(str(exc), 409) from exc
            document = scene.model_dump(mode="json")
            item = next((obj for obj in document["semantic_objects"] if obj["object_id"] == object_id), None)
            if item is None:
                raise SceneStoreError("object not found", 404)
            item.update(changes)
            item["version"] += 1
            document["package_revision"] += 1
            updated = Scene.model_validate(document)
            atomic_bytes(self._scene_dir(scene_id) / "scene.json", dump_scene(updated).encode())
            return updated

    def create_revision(self, scene_id: str, object_id: str, mask_path: Path, delta: dict[str, Any], *, expected_package_revision: int, expected_object_version: int, expected_mask_revision: str, author: str, operation: str) -> Scene:
        with self._lock(scene_id):
            scene = self.get(scene_id)
            try:
                validate_optimistic_lock(scene, expected_package_revision=expected_package_revision, expected_object_versions={object_id: expected_object_version}, expected_mask_revisions={object_id: expected_mask_revision})
            except RevisionConflictError as exc:
                raise SceneStoreError(str(exc), 409) from exc
            current = next((obj for obj in scene.semantic_objects if obj.object_id == object_id), None)
            if current is None:
                raise SceneStoreError("object not found", 404)
            normalize_uploaded_binary_mask(mask_path)
            width, height, media, report = inspect_image(mask_path, binary_mask=True)
            if media != "image/png" or (width, height) != (scene.image_dimensions.width, scene.image_dimensions.height):
                raise SceneStoreError("mask dimensions do not match the source image")
            mask_hash = sha256_file(mask_path)
            if mask_hash == current.mask_hash:
                raise SceneStoreError("resulting mask is unchanged", 409)
            numbers = [r.revision_number for r in scene.mask_revisions if r.object_id == object_id]
            number = max(numbers, default=0) + 1
            revision_id = uuid5(REVISION_NAMESPACE, f"{object_id}:{mask_hash}:{number}").hex
            index = self._index(scene_id)
            if any(r.revision_id == revision_id for r in scene.mask_revisions):
                raise SceneStoreError("revision already exists", 409)
            base = self._scene_dir(scene_id)
            relative_mask = f"artifacts/revisions/{object_id}/{revision_id}/mask.png"
            relative_delta = f"artifacts/revisions/{object_id}/{revision_id}/delta.json"
            destination = base / relative_mask
            if destination.exists():
                raise SceneStoreError("revision artifact already exists", 409)
            destination.parent.mkdir(parents=True, exist_ok=False)
            try:
                atomic_bytes(destination, mask_path.read_bytes())
                atomic_json(base / relative_delta, delta)
                document = scene.model_dump(mode="json")
                document["mask_revisions"].append(MaskRevision(
                    revision_id=revision_id, object_id=object_id, revision_number=number,
                    parent_revision=current.mask_revision, binary_mask_hash=mask_hash,
                    edit_operation=operation, author=author, source="frontend",
                    timestamp=datetime.now(timezone.utc), quality_report=report,
                    geometry_invalidation_status="recompute_required",
                ).model_dump(mode="json"))
                item = next(obj for obj in document["semantic_objects"] if obj["object_id"] == object_id)
                item.update({"mask_revision": revision_id, "mask_hash": mask_hash, "mask_filename": "mask.png", "mask_url": f"/artifacts/masks/{object_id}/{revision_id}", "overlay_url": None, "approval_status": "needs_revision"})
                item["version"] += 1
                document["package_revision"] += 1
                document["generation_status"]["stages"]["primitive_transforms"] = "stale"
                document["generation_status"]["stages"]["blender_objects"] = "stale"
                document["generation_status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                updated = Scene.model_validate(document)
                index[f"masks/{object_id}/{revision_id}"] = {"path": relative_mask, "media_type": "image/png", "sha256": mask_hash}
                index[f"revision-deltas/{object_id}/{revision_id}"] = {"path": relative_delta, "media_type": "application/json", "sha256": sha256_file(base / relative_delta)}
                atomic_json(base / "artifact-index.json", index)
                atomic_bytes(base / "scene.json", dump_scene(updated).encode())
                return updated
            except Exception:
                shutil.rmtree(destination.parent, ignore_errors=True)
                raise

    def import_transforms(self, scene_id: str, expected_package_revision: int, transforms: list[dict[str, Any]]) -> Scene:
        with self._lock(scene_id):
            scene = self.get(scene_id)
            try:
                validate_optimistic_lock(scene, expected_package_revision=expected_package_revision)
            except RevisionConflictError as exc:
                raise SceneStoreError(str(exc), 409) from exc
            document = scene.model_dump(mode="json")
            by_id = {obj["object_id"]: obj for obj in document["semantic_objects"]}
            seen: set[str] = set()
            for update in transforms:
                object_id = str(update.get("object_id", ""))
                if object_id in seen or object_id not in by_id:
                    raise SceneStoreError(f"invalid or duplicate object ID {object_id!r}")
                seen.add(object_id)
                obj = by_id[object_id]
                expected_version = update.get("expected_object_version")
                if expected_version != obj["version"]:
                    raise SceneStoreError(f"object {object_id} version conflict", 409)
                expected_mask = update.get("expected_mask_revision")
                if expected_mask != obj["mask_revision"]:
                    raise SceneStoreError(f"object {object_id} mask revision conflict", 409)
                transform = Transform.model_validate(update.get("transform"))
                obj.update(transform.model_dump(mode="json"))
                if "blender_object_identifier" in update:
                    obj["blender_object_identifier"] = update["blender_object_identifier"]
                obj["version"] += 1
                revision = next(r for r in document["mask_revisions"] if r["revision_id"] == obj["mask_revision"])
                revision["geometry_invalidation_status"] = "valid"
            document["package_revision"] += 1
            document["generation_status"]["stages"]["primitive_transforms"] = "completed"
            document["generation_status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = Scene.model_validate(document)
            atomic_bytes(self._scene_dir(scene_id) / "scene.json", dump_scene(updated).encode())
            return updated
