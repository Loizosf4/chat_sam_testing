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
    try:
        import numpy as np
    except ImportError as exc:
        raise SegmentationWorkspaceStoreError("numpy is not installed", 500) from exc

    mask_array = np.asarray(mask).astype(bool)
    image = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


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
            index = {
                key: value
                for key, value in self._index(workspace_id).items()
                if not key.startswith(f"sam-candidates/{workspace_id}/{object_id}/")
            }
            self._write_index(workspace_id, index)
            self._remove_object_artifacts(workspace_id, object_id)
            self._save(updated)
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
            current_revision = int(item.get("sam_draft", {}).get("prompt_revision", 0))
            if prompt_revision <= current_revision:
                raise SegmentationWorkspaceStoreError(
                    f"stale prompt revision: incoming {prompt_revision}, current {current_revision}",
                    409,
                )

            if not candidate_masks:
                raise SegmentationWorkspaceStoreError("SAM returned no candidates", 500)
            if not (
                len(candidate_masks)
                == len(candidate_scores)
                == len(candidate_areas)
                == len(candidate_bboxes)
            ):
                raise SegmentationWorkspaceStoreError("candidate metadata mismatch", 500)

            base = self._workspace_dir(workspace_id)
            sam_root = base / "objects" / object_id / "sam"
            sam_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{prompt_revision}.", dir=sam_root))
            target = sam_root / str(prompt_revision)
            candidates: list[dict[str, Any]] = []
            index = self._index(workspace_id)
            prefix = f"sam-candidates/{workspace_id}/{object_id}/"
            try:
                for index_value, mask in enumerate(candidate_masks):
                    filename = f"candidate-{index_value}.png"
                    destination = temporary / filename
                    save_binary_mask(destination, mask)
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
                            area_pixels=int(candidate_areas[index_value]),
                            bbox_xyxy=candidate_bboxes[index_value],
                            mask_url=f"/api/segmentation-artifacts/{artifact_id}",
                        ).model_dump(mode="json")
                    )

                if selected_candidate_index not in {candidate["candidate_index"] for candidate in candidates}:
                    raise SegmentationWorkspaceStoreError("selected candidate does not exist", 500)
                if target.exists():
                    shutil.rmtree(target)
                os.replace(temporary, target)
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
                self._save(updated)
                self._remove_old_sam_dirs(base, object_id, str(prompt_revision))
                return updated
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
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
            _require_expected(workspace.workspace_revision, expected_workspace_revision, "workspace revision")
            document = workspace.model_dump(mode="json")
            item = self._find_object_document(document, object_id)
            _require_expected(item["object_version"], expected_object_version, "object version")
            draft = item.get("sam_draft") or {}
            if draft.get("prompt_revision", 0) != prompt_revision:
                raise SegmentationWorkspaceStoreError("prompt revision does not match current candidates", 409)
            if candidate_index not in {candidate["candidate_index"] for candidate in draft.get("candidates", [])}:
                raise SegmentationWorkspaceStoreError("candidate not found", 404)
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
            item["object_version"] += 1
            item["updated_at"] = now.isoformat()
            document["workspace_revision"] += 1
            document["updated_at"] = now.isoformat()
            updated = SegmentationWorkspace.model_validate(document)
            index = {
                key: value
                for key, value in self._index(workspace_id).items()
                if not key.startswith(f"sam-candidates/{workspace_id}/{object_id}/")
            }
            self._write_index(workspace_id, index)
            self._remove_object_artifacts(workspace_id, object_id)
            self._save(updated)
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
        if kind not in {"source-images", "sam-candidates"}:
            raise SegmentationWorkspaceStoreError("artifact not found", 404)
        if kind == "source-images" and (not valid_id(identifier) or "/" in identifier or "\\" in identifier):
            raise SegmentationWorkspaceStoreError("invalid artifact identifier")
        if kind == "sam-candidates" and (
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
