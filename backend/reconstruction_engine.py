"""Managed reconstruction runner helpers for MoGe + Unified V3 compilation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.segmentation_workspace.store import sha256_file


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_COMPILER_TIMEOUT_SECONDS = 1800
REQUIRED_MOGE_OUTPUTS = ["geometry_npz", "points", "depth", "normal", "mask", "intrinsics", "previews"]
REQUIRED_MOGE_ARRAYS = {"points", "depth", "valid_mask", "intrinsics", "normal"}


class ReconstructionEngineError(RuntimeError):
    def __init__(self, message: str, *, code: str = "reconstruction_failed", retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class CompilerResult:
    scene_id: str
    object_count: int
    compilation_passed: bool
    summary: dict[str, Any]


def sanitize_user_message(message: object, *, extra_roots: list[Path] | None = None) -> str:
    text = str(message or "Reconstruction failed.")
    roots = [ROOT_DIR, *(extra_roots or [])]
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved:
            text = text.replace(resolved, "<redacted-path>")
            text = text.replace(resolved.replace("\\", "/"), "<redacted-path>")
    text = re.sub(r"[A-Za-z]:[\\/][^\s\"']+", "<redacted-path>", text)
    text = re.sub(r"(?<![A-Za-z0-9_])/(?:tmp|var|home|Users|mnt|opt|workspace|repo|data)/[^\s\"']+", "<redacted-path>", text)
    text = text.replace("\\\\", "\\")
    return text[:500] or "Reconstruction failed."


def compiler_timeout_seconds() -> int:
    load_dotenv(ROOT_DIR / ".env")
    raw = os.getenv("RECONSTRUCTION_COMPILER_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_COMPILER_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_COMPILER_TIMEOUT_SECONDS
    return max(1, value)


def reconstruction_python(validate_exists: bool = True) -> Path:
    load_dotenv(ROOT_DIR / ".env")
    raw = os.getenv("RECONSTRUCTION_PYTHON", "").strip()
    if not raw:
        raise ReconstructionEngineError(
            "Reconstruction compiler Python is not configured.",
            code="compiler_unconfigured",
            retryable=False,
        )
    python = Path(raw).expanduser()
    if validate_exists and not python.is_file():
        raise ReconstructionEngineError(
            "Reconstruction compiler Python is not available.",
            code="compiler_python_missing",
            retryable=False,
        )
    return python


def compiler_health() -> dict[str, Any]:
    try:
        python = reconstruction_python(validate_exists=True)
    except ReconstructionEngineError as exc:
        return {"configured": False, "compiler_configured": False, "error": str(exc)}
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import numpy, PIL, scipy, pydantic, jsonschema; "
                    "import experiments.moge_scene_graph_spike.src.compile_unified_v3_scene; "
                    "print('ok')"
                ),
            ],
            cwd=str(ROOT_DIR),
            shell=False,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"configured": False, "compiler_configured": True, "error": "Reconstruction compiler dependency check timed out."}
    except OSError:
        return {"configured": False, "compiler_configured": True, "error": "Reconstruction compiler Python could not be executed."}
    if completed.returncode != 0:
        return {"configured": False, "compiler_configured": True, "error": "Reconstruction compiler dependencies are unavailable."}
    return {"configured": True, "compiler_configured": True, "error": None}


def run_compiler(
    *,
    sam_dir: Path,
    source_image: Path,
    moge_dir: Path,
    output_dir: Path,
    scene_id: str,
    handoff_dir: Path,
    expected_object_ids: list[str],
    python: Path | None = None,
    timeout_seconds: int | None = None,
) -> CompilerResult:
    python = python or reconstruction_python(validate_exists=True)
    timeout_seconds = timeout_seconds or compiler_timeout_seconds()
    command = [
        str(python),
        "-m",
        "experiments.moge_scene_graph_spike.src.compile_unified_v3_scene",
        "--mode",
        "clean_reconstruction",
        "--sam-dir",
        str(sam_dir),
        "--source-image",
        str(source_image),
        "--moge-dir",
        str(moge_dir),
        "--output-dir",
        str(output_dir),
        "--scene-id",
        scene_id,
        "--handoff-dir",
        str(handoff_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReconstructionEngineError(
            "The reconstruction compiler timed out.",
            code="compiler_timeout",
            retryable=True,
        ) from exc
    except OSError as exc:
        raise ReconstructionEngineError(
            "The reconstruction compiler could not be started.",
            code="compiler_start_failed",
            retryable=True,
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.strip().splitlines()[-1:] or ["The reconstruction compiler failed."]
        raise ReconstructionEngineError(
            sanitize_user_message(message[0], extra_roots=[sam_dir, source_image, moge_dir, output_dir, handoff_dir]),
            code="compiler_nonzero_exit",
            retryable=True,
        )
    summaries = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            summaries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ReconstructionEngineError("The reconstruction compiler returned malformed output.", code="compiler_malformed_output") from exc
    if len(summaries) != 1 or not isinstance(summaries[0], dict):
        raise ReconstructionEngineError("The reconstruction compiler did not return exactly one JSON summary.", code="compiler_malformed_output")
    summary = summaries[0]
    if summary.get("success") is not True:
        raise ReconstructionEngineError("The reconstruction compiler did not report success.", code="compiler_rejected")
    if summary.get("scene_id") != scene_id:
        raise ReconstructionEngineError("The compiler summary scene ID did not match the requested scene.", code="compiler_scene_id_mismatch")
    if int(summary.get("object_count", -1)) != len(expected_object_ids):
        raise ReconstructionEngineError("The compiler object count did not match the selected export.", code="compiler_object_count_mismatch")
    validate_compiler_outputs(output_dir=output_dir, handoff_dir=handoff_dir, scene_id=scene_id, expected_object_ids=expected_object_ids)
    return CompilerResult(
        scene_id=scene_id,
        object_count=len(expected_object_ids),
        compilation_passed=bool(summary.get("compilation_passed", False)),
        summary=summary,
    )


def validate_moge_output(moge_dir: Path) -> None:
    geometry = moge_dir / "geometry.npz"
    metadata = moge_dir / "metadata.json"
    if not geometry.is_file() or not metadata.is_file():
        raise ReconstructionEngineError("MoGe output is missing required geometry or metadata.", code="moge_output_missing")
    try:
        import numpy as np

        with np.load(geometry, allow_pickle=False) as archive:
            missing = REQUIRED_MOGE_ARRAYS - set(archive.files)
            if missing:
                raise ReconstructionEngineError(
                    f"MoGe geometry is missing arrays: {', '.join(sorted(missing))}.",
                    code="moge_output_invalid",
                )
    except ReconstructionEngineError:
        raise
    except Exception as exc:
        raise ReconstructionEngineError("MoGe geometry could not be validated.", code="moge_output_invalid") from exc


def validate_compiler_outputs(
    *,
    output_dir: Path,
    handoff_dir: Path,
    scene_id: str,
    expected_object_ids: list[str],
) -> None:
    required = [
        output_dir / "unified_scene_plan.json",
        output_dir / "compilation_report.json",
        handoff_dir / "blender_one_batch_manifest.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ReconstructionEngineError(f"Compiler output is missing required files: {', '.join(missing)}.", code="compiler_output_missing")
    try:
        scene = json.loads((output_dir / "unified_scene_plan.json").read_text(encoding="utf-8"))
        report = json.loads((output_dir / "compilation_report.json").read_text(encoding="utf-8"))
        manifest = json.loads((handoff_dir / "blender_one_batch_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReconstructionEngineError("Compiler output JSON could not be parsed.", code="compiler_output_invalid") from exc
    if scene.get("scene_id") != scene_id:
        raise ReconstructionEngineError("The compiler scene plan ID did not match the requested scene.", code="compiler_scene_id_mismatch")
    compiled_ids = [str(item.get("object_id", "")) for item in scene.get("semantic_objects", []) if isinstance(item, dict)]
    if len(compiled_ids) != len(set(compiled_ids)):
        raise ReconstructionEngineError("The compiler output contained duplicate object IDs.", code="compiler_object_id_mismatch")
    if set(compiled_ids) != set(expected_object_ids):
        raise ReconstructionEngineError("The compiler output object IDs did not match the selected export.", code="compiler_object_id_mismatch")
    if int(scene.get("semantic_object_count", -1)) != len(expected_object_ids):
        raise ReconstructionEngineError("The compiler scene object count did not match the selected export.", code="compiler_object_count_mismatch")
    if int(report.get("object_count", -1)) != len(expected_object_ids):
        raise ReconstructionEngineError("The compilation report object count did not match the selected export.", code="compiler_object_count_mismatch")
    if int(manifest.get("semantic_primitive_count", -1)) != len(expected_object_ids):
        raise ReconstructionEngineError("The handoff manifest object count did not match the selected export.", code="compiler_object_count_mismatch")
    for path in [output_dir, handoff_dir]:
        resolved = path.resolve()
        if output_dir.resolve() not in resolved.parents and resolved != output_dir.resolve() and handoff_dir.resolve() not in resolved.parents and resolved != handoff_dir.resolve():
            raise ReconstructionEngineError("Compiler output path escaped the reconstruction staging directory.", code="compiler_output_escape")


def sanitized_moge_summary(moge_dir: Path) -> dict[str, Any]:
    metadata_path = moge_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReconstructionEngineError("MoGe metadata could not be sanitized.", code="moge_metadata_invalid") from exc
    blocked = {
        "source_image",
        "output_dir",
        "output_paths",
        "python_executable",
        "repository",
        "checkpoint",
        "moge_import_path",
    }
    summary = {key: value for key, value in metadata.items() if key not in blocked}
    summary["schema_version"] = "1.0"
    summary["geometry_sha256"] = sha256_file(moge_dir / "geometry.npz")
    return summary
