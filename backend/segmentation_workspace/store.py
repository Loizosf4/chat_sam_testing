"""Atomic filesystem-backed persistence for segmentation workspaces."""

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
from uuid import uuid4

from PIL import Image

from .models import (
    CURRENT_SCHEMA_VERSION,
    ImageDimensions,
    ManualMaskState,
    SamDraftCandidate,
    SamDraftState,
    SamPromptPoint,
    SegmentationObject,
    SegmentationWorkspace,
    SourceImage,
)


IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
IMAGE_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
MAX_IMAGE_PIXELS = 100_000_000
ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")


class SegmentationWorkspaceStoreError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def valid_id(value: str) -> bool:
    return 1 <= len(value) <= 64 and all(c in ID_CHARS for c in value)


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
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def dump_workspace(workspace: SegmentationWorkspace) -> str:
    return workspace.model_dump_json(indent=2, exclude_none=False) + "\n"


def inspect_source_image(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            media_type = IMAGE_FORMATS.get(image.format or "")
            if media_type is None:
                raise SegmentationWorkspaceStoreError("unsupported image type", 415)
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise SegmentationWorkspaceStoreError("unsupported image dimensions", 413)
            return width, height, media_type
    except SegmentationWorkspaceStoreError:
        raise
    except Exception as exc:
        raise SegmentationWorkspaceStoreError("invalid image file", 400) from exc


def save_binary_mask(path: Path, mask: Any) -> None:
    mask_array, _area, _bbox = validate_candidate_mask(mask, expected_height=None, expected_width=None)
    _save_binary_mask_array(path, mask_array)


def validate_candidate_mask(
    mask: Any,
    *,
    expected_height: int | None,
    expected_width: int | None,
) -> tuple[Any, int, list[int]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise SegmentationWorkspaceStoreError("candidate mask must be a two-dimensional array")
    if expected_height is not None and expected_width is not None and raw.shape != (expected_height, expected_width):
        raise SegmentationWorkspaceStoreError(
            "candidate mask dimensions do not match the source image"
        )
    mask_array = raw.astype(bool)
    y_values, x_values = np.where(mask_array)
    area = int(mask_array.sum())
    if area == 0:
        bbox = [0, 0, 0, 0]
    else:
        bbox = [
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max()),
            int(y_values.max()),
        ]
    return mask_array, area, bbox


def _save_binary_mask_array(path: Path, mask_array: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    image = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def load_binary_mask_png(
    path: Path,
    *,
    expected_height: int,
    expected_width: int,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode not in {"1", "L"}:
                raise SegmentationWorkspaceStoreError("mask artifact is not grayscale", 500)
            if image.size != (expected_width, expected_height):
                raise SegmentationWorkspaceStoreError("mask artifact dimensions do not match the source image", 500)
            pixels = np.asarray(image)
    except SegmentationWorkspaceStoreError:
        raise
    except Exception as exc:
        raise SegmentationWorkspaceStoreError("mask artifact could not be decoded", 500) from exc
    if pixels.ndim != 2:
        raise SegmentationWorkspaceStoreError("mask artifact must be two-dimensional", 500)
    if pixels.dtype == np.bool_:
        return pixels.astype(bool)
    unique = set(np.unique(pixels).tolist())
    if not unique.issubset({0, 255}):
        raise SegmentationWorkspaceStoreError("mask artifact is not binary", 500)
    return pixels == 255


def derive_manual_layers(base_mask: Any, edited_mask: Any) -> tuple[Any, Any, Any, int, list[int]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    base = np.asarray(base_mask).astype(bool)
    edited = np.asarray(edited_mask).astype(bool)
    if base.shape != edited.shape:
        raise SegmentationWorkspaceStoreError("edited mask dimensions do not match the selected candidate")
    manual_add = np.logical_and(edited, np.logical_not(base))
    manual_remove = np.logical_and(base, np.logical_not(edited))
    composite = np.logical_and(np.logical_or(base, manual_add), np.logical_not(manual_remove))
    if not np.array_equal(composite, edited):
        raise SegmentationWorkspaceStoreError("manual mask composition failed", 500)
    _mask, area, bbox = validate_candidate_mask(
        composite,
        expected_height=edited.shape[0],
        expected_width=edited.shape[1],
    )
    return manual_add, manual_remove, composite, area, bbox


def empty_manual_mask_state() -> dict[str, Any]:
    return ManualMaskState().model_dump(mode="json")


def _require_expected(current: int, expected: int, label: str) -> None:
    if current != expected:
        raise SegmentationWorkspaceStoreError(
            f"{label} conflict: expected {expected}, current {current}",
            409,
        )


def _clean_text(value: str, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise SegmentationWorkspaceStoreError(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise SegmentationWorkspaceStoreError(f"{label} cannot be empty")
    if len(stripped) > max_length:
        raise SegmentationWorkspaceStoreError(f"{label} is too long")
    return stripped


class SegmentationWorkspaceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, workspace_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(workspace_id, threading.RLock())

    def _workspace_dir(self, workspace_id: str) -> Path:
        if not valid_id(workspace_id):
            raise SegmentationWorkspaceStoreError("unsafe workspace ID")
        candidate = (self.root / workspace_id).resolve()
        if candidate.parent != self.root:
            raise SegmentationWorkspaceStoreError("workspace path escapes storage root")
        return candidate

    def _load(self, workspace_id: str) -> SegmentationWorkspace:
        path = self._workspace_dir(workspace_id) / "workspace.json"
        if not path.is_file():
            raise SegmentationWorkspaceStoreError("workspace not found", 404)
        try:
            return SegmentationWorkspace.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SegmentationWorkspaceStoreError("stored workspace is invalid", 500) from exc

    def _save(self, workspace: SegmentationWorkspace) -> None:
        atomic_bytes(self._workspace_dir(workspace.workspace_id) / "workspace.json", dump_workspace(workspace).encode("utf-8"))

    def _index(self, workspace_id: str) -> dict[str, dict[str, str]]:
        path = self._workspace_dir(workspace_id) / "artifact-index.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SegmentationWorkspaceStoreError("artifact index not found", 404) from exc
        if not isinstance(value, dict):
            raise SegmentationWorkspaceStoreError("invalid artifact index", 500)
        return value

    def _write_index(self, workspace_id: str, index: dict[str, dict[str, str]]) -> None:
        atomic_json(self._workspace_dir(workspace_id) / "artifact-index.json", index)

    def _find_object_document(self, document: dict[str, Any], object_id: str) -> dict[str, Any]:
        item = next((obj for obj in document["objects"] if obj["object_id"] == object_id), None)
        if item is None:
            raise SegmentationWorkspaceStoreError("object not found", 404)
        return item

    def _resolve_workspace_artifact(self, workspace_id: str, key: str) -> tuple[Path, str]:
        index = self._index(workspace_id)
        entry = index.get(key)
        if not entry:
            raise SegmentationWorkspaceStoreError("artifact not found", 404)
        base = self._workspace_dir(workspace_id).resolve()
        path = (base / entry["path"]).resolve()
        if base not in path.parents or not path.is_file():
            raise SegmentationWorkspaceStoreError("artifact index escaped workspace root", 500)
        if sha256_file(path) != entry["sha256"]:
            raise SegmentationWorkspaceStoreError("artifact integrity check failed", 500)
        return path, entry["media_type"]

    def _manual_prefix(self, workspace_id: str, object_id: str) -> str:
        return f"manual-masks/{workspace_id}/{object_id}/"

    def _manual_root(self, workspace_id: str, object_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "objects" / object_id / "manual"

    def _remove_old_manual_dirs(self, workspace_id: str, object_id: str, keep_revision: str | None = None) -> None:
        manual_root = self._manual_root(workspace_id, object_id)
        if not manual_root.is_dir():
            return
        for child in manual_root.iterdir():
            if child.name.startswith("."):
                shutil.rmtree(child, ignore_errors=True)
            elif keep_revision is None or child.name != keep_revision:
                shutil.rmtree(child, ignore_errors=True)

    def _selected_candidate_document(self, item: dict[str, Any]) -> dict[str, Any]:
        draft = item.get("sam_draft") or {}
        candidates = draft.get("candidates") or []
        if not candidates:
            raise SegmentationWorkspaceStoreError("current SAM candidates are required", 409)
        selected_index = draft.get("selected_candidate_index")
        if selected_index is None:
            raise SegmentationWorkspaceStoreError("a selected SAM candidate is required", 409)
        candidate = next((candidate for candidate in candidates if candidate["candidate_index"] == selected_index), None)
        if candidate is None:
            raise SegmentationWorkspaceStoreError("selected SAM candidate is missing", 409)
        return candidate

    def _manual_active(self, item: dict[str, Any]) -> bool:
        return int((item.get("manual_mask") or {}).get("manual_revision", 0)) > 0

    def create_workspace(
        self,
        source_image_path: Path,
        original_filename: str,
        *,
        workspace_id: str | None = None,
    ) -> SegmentationWorkspace:
        workspace_id = workspace_id or uuid4().hex
        if "/" in original_filename or "\\" in original_filename or original_filename in {"", ".", ".."}:
            raise SegmentationWorkspaceStoreError("original filename must be a basename")
        width, height, media_type = inspect_source_image(source_image_path)
        extension = IMAGE_EXTENSIONS[media_type]
        target = self._workspace_dir(workspace_id)
        with self._lock(workspace_id):
            if target.exists():
                raise SegmentationWorkspaceStoreError("workspace already exists", 409)
            staging = Path(tempfile.mkdtemp(prefix=f".{workspace_id}.", dir=self.root))
            try:
                source_dir = staging / "source"
                source_dir.mkdir(parents=True, exist_ok=True)
                source_target = source_dir / f"source{extension}"
                shutil.copyfile(source_image_path, source_target)
                stored_hash = sha256_file(source_target)
                image_id = uuid4().hex
                now = datetime.now(timezone.utc)
                workspace = SegmentationWorkspace(
                    schema_version=CURRENT_SCHEMA_VERSION,
                    workspace_id=workspace_id,
                    workspace_revision=1,
                    created_at=now,
                    updated_at=now,
                    source_image=SourceImage(
                        image_id=image_id,
                        url=f"/api/segmentation-artifacts/source-images/{image_id}",
                        original_filename=original_filename,
                        media_type=media_type,
                        sha256=stored_hash,
                    ),
                    image_dimensions=ImageDimensions(width=width, height=height),
                    status="draft",
                    objects=[],
                )
                index = {
                    f"source-images/{image_id}": {
                        "path": f"source/source{extension}",
                        "media_type": media_type,
                        "sha256": stored_hash,
                    }
                }
                atomic_bytes(staging / "workspace.json", dump_workspace(workspace).encode("utf-8"))
                atomic_json(staging / "artifact-index.json", index)
                if target.exists():
                    raise SegmentationWorkspaceStoreError("workspace already exists", 409)
                os.replace(staging, target)
                return workspace
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise

    def get_workspace(self, workspace_id: str) -> SegmentationWorkspace:
        return self._load(workspace_id)

    def delete_workspace(self, workspace_id: str, expected_workspace_revision: int) -> None:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(
                workspace.workspace_revision,
                expected_workspace_revision,
                "workspace revision",
            )
            target = self._workspace_dir(workspace_id)
            shutil.rmtree(target)

    def create_object(
        self,
        workspace_id: str,
        semantic_label: str,
        display_name: str,
        expected_workspace_revision: int,
    ) -> SegmentationWorkspace:
        semantic_label = _clean_text(semantic_label, "semantic label", 128)
        display_name = _clean_text(display_name, "display name", 192)
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            now = datetime.now(timezone.utc)
            object_ids = {item.object_id for item in workspace.objects}
            object_id = uuid4().hex
            while object_id in object_ids:
                object_id = uuid4().hex
            document = workspace.model_dump(mode="json")
            document["objects"].append(
                SegmentationObject(
                    object_id=object_id,
                    object_version=1,
                    semantic_label=semantic_label,
                    display_name=display_name,
                    status="draft",
                    created_at=now,
                    updated_at=now,
                    sam_draft=SamDraftState(),
                    manual_mask=ManualMaskState(),
                ).model_dump(mode="json")
            )
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            self._save(updated)
            return updated

    def update_object(
        self,
        workspace_id: str,
        object_id: str,
        semantic_label: str,
        display_name: str,
        expected_workspace_revision: int,
        expected_object_version: int,
    ) -> SegmentationWorkspace:
        semantic_label = _clean_text(semantic_label, "semantic label", 128)
        display_name = _clean_text(display_name, "display name", 192)
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = next((obj for obj in document["objects"] if obj["object_id"] == object_id), None)
            if item is None:
                raise SegmentationWorkspaceStoreError("object not found", 404)
            _require_expected(item["object_version"], expected_object_version, "object version")
            now = datetime.now(timezone.utc)
            item.update(
                {
                    "semantic_label": semantic_label,
                    "display_name": display_name,
                    "object_version": item["object_version"] + 1,
                    "updated_at": now.isoformat(),
                }
            )
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            self._save(updated)
            return updated

    def delete_object(
        self,
        workspace_id: str,
        object_id: str,
        expected_workspace_revision: int,
        expected_object_version: int,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = next((obj for obj in document["objects"] if obj["object_id"] == object_id), None)
            if item is None:
                raise SegmentationWorkspaceStoreError("object not found", 404)
            _require_expected(item["object_version"], expected_object_version, "object version")
            document["objects"] = [obj for obj in document["objects"] if obj["object_id"] != object_id]
            document["workspace_revision"] += 1
            document["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            original_index = self._index(workspace_id)
            index = {
                key: value
                for key, value in original_index.items()
                if not (
                    key.startswith(f"sam-candidates/{workspace_id}/{object_id}/")
                    or key.startswith(self._manual_prefix(workspace_id, object_id))
                )
            }
            index_written = False
            workspace_written = False
            try:
                self._write_index(workspace_id, index)
                index_written = True
                self._save(updated)
                workspace_written = True
            except Exception:
                if index_written and not workspace_written:
                    try:
                        self._write_index(workspace_id, original_index)
                    except Exception:
                        pass
                raise
            self._remove_object_artifacts(workspace_id, object_id)
            return updated

    def persist_sam_prediction(
        self,
        workspace_id: str,
        object_id: str,
        *,
        prompt_revision: int,
        points: list[dict[str, Any]],
        box: list[float] | None,
        candidate_masks: list[Any],
        candidate_scores: list[float],
        candidate_areas: list[int],
        candidate_bboxes: list[list[int]],
        selected_candidate_index: int,
        prepared_image_key: str,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            if workspace.status == "finalized":
                raise SegmentationWorkspaceStoreError("workspace is finalized", 409)
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            if item["status"] == "finalized":
                raise SegmentationWorkspaceStoreError("object is finalized", 409)
            if self._manual_active(item):
                raise SegmentationWorkspaceStoreError(
                    "Clear manual mask corrections before changing the SAM prompt.",
                    409,
                )
            current_revision = int(item.get("sam_draft", {}).get("prompt_revision", 0))
            if prompt_revision <= current_revision:
                raise SegmentationWorkspaceStoreError(
                    f"stale prompt revision: incoming {prompt_revision}, current {current_revision}",
                    409,
                )

            if not candidate_masks:
                raise SegmentationWorkspaceStoreError("SAM returned no candidates", 500)
            if len(candidate_masks) != len(candidate_scores):
                raise SegmentationWorkspaceStoreError("candidate metadata mismatch", 500)

            base = self._workspace_dir(workspace_id)
            sam_root = base / "objects" / object_id / "sam"
            sam_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{prompt_revision}.", dir=sam_root))
            target = sam_root / str(prompt_revision)
            if target.exists():
                shutil.rmtree(temporary, ignore_errors=True)
                raise SegmentationWorkspaceStoreError("candidate revision artifacts already exist", 409)
            candidates: list[dict[str, Any]] = []
            original_index = self._index(workspace_id)
            index = {key: dict(value) for key, value in original_index.items()}
            prefix = f"sam-candidates/{workspace_id}/{object_id}/"
            target_published = False
            index_written = False
            workspace_written = False
            try:
                for index_value, mask in enumerate(candidate_masks):
                    filename = f"candidate-{index_value}.png"
                    destination = temporary / filename
                    mask_array, area_pixels, bbox_xyxy = validate_candidate_mask(
                        mask,
                        expected_height=workspace.image_dimensions.height,
                        expected_width=workspace.image_dimensions.width,
                    )
                    _save_binary_mask_array(destination, mask_array)
                    artifact_id = f"{prefix}{prompt_revision}/{index_value}"
                    mask_hash = sha256_file(destination)
                    index[artifact_id] = {
                        "path": f"objects/{object_id}/sam/{prompt_revision}/{filename}",
                        "media_type": "image/png",
                        "sha256": mask_hash,
                    }
                    candidates.append(
                        SamDraftCandidate(
                            candidate_index=index_value,
                            score=float(candidate_scores[index_value]),
                            area_pixels=area_pixels,
                            bbox_xyxy=bbox_xyxy,
                            mask_url=f"/api/segmentation-artifacts/{artifact_id}",
                        ).model_dump(mode="json")
                    )

                if selected_candidate_index not in {candidate["candidate_index"] for candidate in candidates}:
                    raise SegmentationWorkspaceStoreError("selected candidate does not exist", 500)
                os.replace(temporary, target)
                target_published = True
                index = {
                    key: value
                    for key, value in index.items()
                    if not (key.startswith(prefix) and not key.startswith(f"{prefix}{prompt_revision}/"))
                }
                now = datetime.now(timezone.utc)
                item["sam_draft"] = SamDraftState(
                    prompt_revision=prompt_revision,
                    points=[SamPromptPoint.model_validate(point).model_dump(mode="json") for point in points],
                    box=box,
                    candidates=candidates,
                    selected_candidate_index=selected_candidate_index,
                    prepared_image_key=prepared_image_key,
                    updated_at=now,
                ).model_dump(mode="json")
                item["object_version"] += 1
                item["updated_at"] = now.isoformat()
                document["workspace_revision"] += 1
                document["updated_at"] = now.isoformat()
                updated = SegmentationWorkspace.model_validate(document)
                self._write_index(workspace_id, index)
                index_written = True
                self._save(updated)
                workspace_written = True
                self._remove_old_sam_dirs(base, object_id, str(prompt_revision))
                return updated
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                if target_published:
                    shutil.rmtree(target, ignore_errors=True)
                if index_written and not workspace_written:
                    try:
                        self._write_index(workspace_id, original_index)
                    except Exception:
                        pass
                raise

    def select_sam_candidate(
        self,
        workspace_id: str,
        object_id: str,
        *,
        prompt_revision: int,
        candidate_index: int,
        expected_workspace_revision: int,
        expected_object_version: int,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            if workspace.status == "finalized":
                raise SegmentationWorkspaceStoreError("workspace is finalized", 409)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            if item["status"] == "finalized":
                raise SegmentationWorkspaceStoreError("object is finalized", 409)
            _require_expected(item["object_version"], expected_object_version, "object version")
            draft = item.get("sam_draft") or {}
            if draft.get("prompt_revision", 0) != prompt_revision:
                raise SegmentationWorkspaceStoreError("prompt revision does not match current candidates", 409)
            if candidate_index not in {candidate["candidate_index"] for candidate in draft.get("candidates", [])}:
                raise SegmentationWorkspaceStoreError("candidate not found", 404)
            if self._manual_active(item) and candidate_index != draft.get("selected_candidate_index"):
                raise SegmentationWorkspaceStoreError(
                    "Clear manual mask corrections before selecting a different SAM candidate.",
                    409,
                )
            now = datetime.now(timezone.utc)
            draft["selected_candidate_index"] = candidate_index
            draft["updated_at"] = now.isoformat()
            item["sam_draft"] = draft
            item["object_version"] += 1
            item["updated_at"] = now.isoformat()
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            self._save(updated)
            return updated

    def clear_sam_draft(
        self,
        workspace_id: str,
        object_id: str,
        *,
        expected_workspace_revision: int,
        expected_object_version: int,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            _require_expected(item["object_version"], expected_object_version, "object version")
            now = datetime.now(timezone.utc)
            item["sam_draft"] = SamDraftState(prompt_revision=0, updated_at=now).model_dump(mode="json")
            item["manual_mask"] = empty_manual_mask_state()
            item["object_version"] += 1
            item["updated_at"] = now.isoformat()
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            original_index = self._index(workspace_id)
            index = {
                key: value
                for key, value in original_index.items()
                if not (
                    key.startswith(f"sam-candidates/{workspace_id}/{object_id}/")
                    or key.startswith(self._manual_prefix(workspace_id, object_id))
                )
            }
            index_written = False
            workspace_written = False
            try:
                self._write_index(workspace_id, index)
                index_written = True
                self._save(updated)
                workspace_written = True
            except Exception:
                if index_written and not workspace_written:
                    try:
                        self._write_index(workspace_id, original_index)
                    except Exception:
                        pass
                raise
            self._remove_object_artifacts(workspace_id, object_id)
            return updated

    def save_manual_mask(
        self,
        workspace_id: str,
        object_id: str,
        *,
        edited_mask: Any,
        base_prompt_revision: int,
        base_candidate_index: int,
        expected_workspace_revision: int,
        expected_object_version: int,
        expected_manual_revision: int,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            if workspace.status == "finalized":
                raise SegmentationWorkspaceStoreError("workspace is finalized", 409)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            if item["status"] == "finalized":
                raise SegmentationWorkspaceStoreError("object is finalized", 409)
            _require_expected(item["object_version"], expected_object_version, "object version")
            current_manual = item.get("manual_mask") or empty_manual_mask_state()
            _require_expected(
                int(current_manual.get("manual_revision", 0)),
                expected_manual_revision,
                "manual revision",
            )
            draft = item.get("sam_draft") or {}
            if int(draft.get("prompt_revision", 0)) != base_prompt_revision:
                raise SegmentationWorkspaceStoreError(
                    "manual mask was edited against an obsolete SAM prompt revision",
                    409,
                )
            selected_index = draft.get("selected_candidate_index")
            if selected_index is None:
                raise SegmentationWorkspaceStoreError("a selected SAM candidate is required", 409)
            if selected_index != base_candidate_index:
                raise SegmentationWorkspaceStoreError(
                    "manual mask was edited against an obsolete SAM candidate",
                    409,
                )
            candidate = self._selected_candidate_document(item)
            if candidate["candidate_index"] != base_candidate_index:
                raise SegmentationWorkspaceStoreError("selected SAM candidate is inconsistent", 409)
            mask_url = candidate["mask_url"]
            prefix = "/api/segmentation-artifacts/"
            if not mask_url.startswith(prefix):
                raise SegmentationWorkspaceStoreError("selected candidate URL is invalid", 500)
            artifact_key = mask_url[len(prefix) :]
            candidate_path, media_type = self._resolve_workspace_artifact(workspace_id, artifact_key)
            if media_type != "image/png":
                raise SegmentationWorkspaceStoreError("selected candidate is not a PNG mask", 500)
            base_mask = load_binary_mask_png(
                candidate_path,
                expected_height=workspace.image_dimensions.height,
                expected_width=workspace.image_dimensions.width,
            )
            manual_add, manual_remove, composite, area_pixels, bbox_xyxy = derive_manual_layers(base_mask, edited_mask)

            manual_revision = expected_manual_revision + 1
            manual_root = self._manual_root(workspace_id, object_id)
            manual_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{manual_revision}.", dir=manual_root))
            target = manual_root / str(manual_revision)
            if target.exists():
                shutil.rmtree(temporary, ignore_errors=True)
                raise SegmentationWorkspaceStoreError("manual revision artifacts already exist", 409)
            original_index = self._index(workspace_id)
            index = {key: dict(value) for key, value in original_index.items()}
            artifact_prefix = self._manual_prefix(workspace_id, object_id)
            target_published = False
            index_written = False
            workspace_written = False
            try:
                artifacts = {
                    "add": manual_add,
                    "remove": manual_remove,
                    "composite": composite,
                }
                urls: dict[str, str] = {}
                for name, mask in artifacts.items():
                    destination = temporary / f"{name}.png"
                    _save_binary_mask_array(destination, mask)
                    artifact_id = f"{artifact_prefix}{manual_revision}/{name}"
                    index[artifact_id] = {
                        "path": f"objects/{object_id}/manual/{manual_revision}/{name}.png",
                        "media_type": "image/png",
                        "sha256": sha256_file(destination),
                    }
                    urls[name] = f"/api/segmentation-artifacts/{artifact_id}"
                os.replace(temporary, target)
                target_published = True
                index = {
                    key: value
                    for key, value in index.items()
                    if not (key.startswith(artifact_prefix) and not key.startswith(f"{artifact_prefix}{manual_revision}/"))
                }
                now = datetime.now(timezone.utc)
                item["manual_mask"] = ManualMaskState(
                    manual_revision=manual_revision,
                    base_prompt_revision=base_prompt_revision,
                    base_candidate_index=base_candidate_index,
                    add_mask_url=urls["add"],
                    remove_mask_url=urls["remove"],
                    composite_mask_url=urls["composite"],
                    area_pixels=area_pixels,
                    bbox_xyxy=bbox_xyxy,
                    updated_at=now,
                ).model_dump(mode="json")
                item["object_version"] += 1
                item["updated_at"] = now.isoformat()
                document["workspace_revision"] += 1
                document["updated_at"] = now.isoformat()
                updated = SegmentationWorkspace.model_validate(document)
                self._write_index(workspace_id, index)
                index_written = True
                self._save(updated)
                workspace_written = True
                self._remove_old_manual_dirs(workspace_id, object_id, str(manual_revision))
                return updated
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                if target_published:
                    shutil.rmtree(target, ignore_errors=True)
                if index_written and not workspace_written:
                    try:
                        self._write_index(workspace_id, original_index)
                    except Exception:
                        pass
                raise

    def clear_manual_mask(
        self,
        workspace_id: str,
        object_id: str,
        *,
        expected_workspace_revision: int,
        expected_object_version: int,
        expected_manual_revision: int,
    ) -> SegmentationWorkspace:
        with self._lock(workspace_id):
            workspace = self._load(workspace_id)
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            _require_expected(item["object_version"], expected_object_version, "object version")
            current_manual = item.get("manual_mask") or empty_manual_mask_state()
            _require_expected(
                int(current_manual.get("manual_revision", 0)),
                expected_manual_revision,
                "manual revision",
            )
            if expected_manual_revision == 0:
                return workspace
            original_index = self._index(workspace_id)
            index = {
                key: value
                for key, value in original_index.items()
                if not key.startswith(self._manual_prefix(workspace_id, object_id))
            }
            now = datetime.now(timezone.utc)
            item["manual_mask"] = empty_manual_mask_state()
            item["object_version"] += 1
            item["updated_at"] = now.isoformat()
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            index_written = False
            workspace_written = False
            try:
                self._write_index(workspace_id, index)
                index_written = True
                self._save(updated)
                workspace_written = True
            except Exception:
                if index_written and not workspace_written:
                    try:
                        self._write_index(workspace_id, original_index)
                    except Exception:
                        pass
                raise
            self._remove_old_manual_dirs(workspace_id, object_id)
            return updated

    def _remove_object_artifacts(self, workspace_id: str, object_id: str) -> None:
        base = self._workspace_dir(workspace_id)
        shutil.rmtree(base / "objects" / object_id, ignore_errors=True)

    def _remove_old_sam_dirs(self, base: Path, object_id: str, keep_revision: str) -> None:
        sam_root = base / "objects" / object_id / "sam"
        if not sam_root.is_dir():
            return
        for child in sam_root.iterdir():
            if child.name != keep_revision:
                shutil.rmtree(child, ignore_errors=True)

    def resolve_artifact(self, kind: str, identifier: str) -> tuple[Path, str]:
        if kind not in {"source-images", "sam-candidates", "manual-masks"}:
            raise SegmentationWorkspaceStoreError("artifact not found", 404)
        if kind == "source-images" and (not valid_id(identifier) or "/" in identifier or "\\" in identifier):
            raise SegmentationWorkspaceStoreError("invalid artifact identifier")
        if kind in {"sam-candidates", "manual-masks"} and (
            not identifier
            or "\\" in identifier
            or any(part in {"", ".", ".."} for part in identifier.split("/"))
        ):
            raise SegmentationWorkspaceStoreError("invalid artifact identifier")
        key = f"{kind}/{identifier}"
        for workspace_dir in self.root.iterdir():
            if not workspace_dir.is_dir():
                continue
            try:
                entry = self._index(workspace_dir.name).get(key)
            except SegmentationWorkspaceStoreError:
                continue
            if not entry:
                continue
            base = workspace_dir.resolve()
            path = (base / entry["path"]).resolve()
            if base not in path.parents or not path.is_file():
                raise SegmentationWorkspaceStoreError("artifact index escaped workspace root", 500)
            if sha256_file(path) != entry["sha256"]:
                raise SegmentationWorkspaceStoreError("artifact integrity check failed", 500)
            return path, entry["media_type"]
        raise SegmentationWorkspaceStoreError("artifact not found", 404)
