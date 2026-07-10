"""JSON MoGe worker executed by the configured external MoGe Python."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SUPPORTED_OUTPUTS = {
    "geometry_npz",
    "points",
    "depth",
    "normal",
    "mask",
    "intrinsics",
    "previews",
}

_MODEL: Any | None = None
_MODEL_KEY: tuple[str, str] | None = None


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, default=_json_default))


def _emit_line(value: dict[str, Any]) -> None:
    print(json.dumps(value, default=_json_default), flush=True)


def _fail(exc: BaseException) -> int:
    if isinstance(exc, WorkerError):
        _emit({"success": False, "error": {"code": exc.code, "message": str(exc)}})
    else:
        _emit(
            {
                "success": False,
                "error": {
                    "code": "unexpected_error",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            }
        )
    return 1


def _resolve_file(path: str, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise WorkerError(f"{name}_missing", f"{name} does not point to a file: {resolved}")
    return resolved


def _resolve_dir(path: str, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise WorkerError(f"{name}_missing", f"{name} does not point to a directory: {resolved}")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _check_expected_python(expected_python: str | None) -> None:
    if not expected_python:
        return
    expected = Path(expected_python).expanduser().resolve()
    actual = Path(sys.executable).resolve()
    if actual != expected:
        raise WorkerError(
            "wrong_python_executable",
            f"MoGe worker is running under {actual}, expected {expected}",
        )


def _check_compatibility_fix(repository: Path) -> None:
    utils_path = repository / "moge" / "model" / "utils.py"
    if not utils_path.is_file():
        raise WorkerError(
            "compatibility_file_missing",
            f"MoGe compatibility target does not exist: {utils_path}",
        )
    first_line = utils_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != "from __future__ import annotations":
        raise WorkerError(
            "windows_amd_compatibility_fix_missing",
            "MoGe repository is missing 'from __future__ import annotations' "
            "at the beginning of moge/model/utils.py.",
        )


def _prepare_imports(repository: Path) -> dict[str, Any]:
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    try:
        import torch
    except Exception as exc:
        raise WorkerError("torch_import_failed", f"PyTorch import failed: {exc}") from exc

    try:
        import torchvision
    except Exception as exc:
        raise WorkerError("torchvision_import_failed", f"TorchVision import failed: {exc}") from exc

    try:
        import moge
        from moge.model.v2 import MoGeModel
    except Exception as exc:
        raise WorkerError("moge_import_failed", f"MoGe import failed: {exc}") from exc

    moge_path = Path(moge.__file__).resolve()
    if not _is_relative_to(moge_path, repository):
        raise WorkerError(
            "wrong_moge_repository",
            f"moge imported from {moge_path}, expected it under {repository}",
        )

    return {
        "torch": torch,
        "torchvision": torchvision,
        "MoGeModel": MoGeModel,
        "moge_import_path": str(moge_path),
    }


def _check_device(torch: Any, device_name: str, allow_cpu: bool) -> Any:
    device_name = (device_name or "").strip().lower()
    if device_name == "cpu" and not allow_cpu:
        raise WorkerError(
            "cpu_fallback_not_enabled",
            "MOGE_DEVICE=cpu was requested, but CPU execution is disabled. "
            "Set MOGE_ALLOW_CPU=true to make the slow CPU path explicit.",
        )
    if device_name == "cpu":
        return torch.device("cpu")
    if not (device_name == "cuda" or device_name.startswith("cuda:")):
        raise WorkerError(
            "invalid_device",
            f"Invalid MoGe device '{device_name}'. Use cuda, cuda:<index>, or cpu.",
        )
    if not torch.cuda.is_available():
        raise WorkerError(
            "gpu_unavailable",
            "MoGe device is CUDA-facing, but torch.cuda.is_available() is false. "
            "ROCm PyTorch should still report the AMD GPU through torch.cuda.",
        )
    device = torch.device(device_name)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise WorkerError(
            "device_unavailable",
            f"Device {device_name} is unavailable; PyTorch reports "
            f"{torch.cuda.device_count()} CUDA device(s).",
        )
    return device


def _model(MoGeModel: Any, checkpoint: Path, device: Any) -> Any:
    global _MODEL, _MODEL_KEY
    key = (str(checkpoint), str(device))
    if _MODEL is None or _MODEL_KEY != key:
        try:
            _MODEL = MoGeModel.from_pretrained(str(checkpoint)).to(device).eval()
        except Exception as exc:
            raise WorkerError("model_load_failed", f"Failed to load MoGe model: {exc}") from exc
        _MODEL_KEY = key
    return _MODEL


def _base_diagnostics(args: argparse.Namespace, *, load_model: bool) -> dict[str, Any]:
    _check_expected_python(args.expected_python)
    repository = _resolve_dir(args.repository, "moge_repository")
    checkpoint = _resolve_file(args.checkpoint, "moge_checkpoint")
    _check_compatibility_fix(repository)
    imports = _prepare_imports(repository)
    torch = imports["torch"]
    device = _check_device(torch, args.device, args.allow_cpu)

    loaded = False
    if load_model:
        _model(imports["MoGeModel"], checkpoint, device)
        loaded = True

    return {
        "success": True,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "repository": str(repository),
        "checkpoint": str(checkpoint),
        "moge_version": args.version,
        "model": args.model,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": imports["torchvision"].__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compatibility_fix_present": True,
        "moge_import_path": imports["moge_import_path"],
        "model_loaded": loaded,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    return _base_diagnostics(args, load_model=args.load_model)


def _horizontal_fov_degrees(intrinsics: Any) -> float:
    fx = float(intrinsics[0, 0])
    if not math.isfinite(fx) or fx <= 0:
        raise WorkerError("invalid_intrinsics", f"Invalid normalized focal length fx={fx}")
    return math.degrees(2.0 * math.atan(0.5 / fx))


def _as_numpy(value: Any) -> Any:
    return value.detach().cpu().numpy()


def _save_previews(output_dir: Path, depth: Any, normal: Any | None, valid: Any) -> dict[str, str]:
    import numpy as np
    from PIL import Image

    paths: dict[str, str] = {}
    valid_depth = depth[valid]
    if valid_depth.size:
        low, high = np.percentile(valid_depth, [2, 98])
        span = max(float(high - low), np.finfo(np.float32).eps)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        preview[valid] = np.clip((depth[valid] - low) / span * 255, 0, 255).astype(np.uint8)
        path = output_dir / "depth_preview.png"
        Image.fromarray(preview, mode="L").save(path)
        paths["depth_preview"] = str(path)
    valid_path = output_dir / "valid_mask_preview.png"
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(valid_path)
    paths["valid_mask_preview"] = str(valid_path)
    if normal is not None:
        normal_preview = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
        normal_preview[~valid] = 0
        normal_path = output_dir / "normal_preview.png"
        Image.fromarray(normal_preview, mode="RGB").save(normal_path)
        paths["normal_preview"] = str(normal_path)
    return paths


def infer(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    diagnostics = _base_diagnostics(args, load_model=False)
    repository = Path(diagnostics["repository"])
    checkpoint = Path(diagnostics["checkpoint"])
    imports = _prepare_imports(repository)
    torch = imports["torch"]
    device = _check_device(torch, args.device, args.allow_cpu)

    requested = set(args.output_type or [])
    if not requested:
        requested = {"geometry_npz", "points", "depth", "normal", "mask", "intrinsics", "previews"}
    unsupported = sorted(requested - SUPPORTED_OUTPUTS)
    if unsupported:
        raise WorkerError(
            "unsupported_output_type",
            f"Unsupported MoGe output type(s): {', '.join(unsupported)}",
        )

    input_path = _resolve_file(args.input, "input_image")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(input_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise WorkerError("image_load_failed", f"Failed to load input image: {exc}") from exc

    height, width = rgb.shape[:2]
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    model = _model(imports["MoGeModel"], checkpoint, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.infer(
            image_tensor,
            resolution_level=args.resolution_level,
            num_tokens=args.num_tokens,
            use_fp16=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    required = {"points", "depth", "mask", "intrinsics"}
    missing = sorted(required - set(output))
    if missing:
        raise WorkerError("missing_output", f"MoGe output is missing required keys: {missing}")

    arrays: dict[str, Any] = {
        "points": _as_numpy(output["points"]).astype(np.float32, copy=False),
        "depth": _as_numpy(output["depth"]).astype(np.float32, copy=False),
        "valid_mask": _as_numpy(output["mask"]).astype(bool, copy=False),
        "intrinsics": _as_numpy(output["intrinsics"]).astype(np.float32, copy=False),
    }
    if "normal" in output:
        arrays["normal"] = _as_numpy(output["normal"]).astype(np.float32, copy=False)

    output_paths: dict[str, str] = {}
    file_map = {
        "points": "points.npy",
        "depth": "depth.npy",
        "normal": "normal.npy",
        "mask": "valid_mask.npy",
        "intrinsics": "intrinsics.npy",
    }
    for output_type, filename in file_map.items():
        array_name = "valid_mask" if output_type == "mask" else output_type
        if output_type in requested and array_name in arrays:
            path = output_dir / filename
            np.save(path, arrays[array_name], allow_pickle=False)
            output_paths[output_type] = str(path)
    if "geometry_npz" in requested:
        geometry_path = output_dir / "geometry.npz"
        np.savez_compressed(geometry_path, **arrays)
        output_paths["geometry_npz"] = str(geometry_path)
    if "previews" in requested:
        output_paths.update(
            _save_previews(output_dir, arrays["depth"], arrays.get("normal"), arrays["valid_mask"])
        )

    metadata = {
        "schema_version": 1,
        "source_image": str(input_path),
        "source_image_dimensions": {"width": width, "height": height},
        "output_dir": str(output_dir),
        "output_paths": output_paths,
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "estimated_fov_x_degrees": _horizontal_fov_degrees(arrays["intrinsics"]),
        "normalized_intrinsics": arrays["intrinsics"].tolist(),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "fp16": False,
        **{key: diagnostics[key] for key in (
            "python_executable",
            "python_version",
            "repository",
            "checkpoint",
            "moge_version",
            "model",
            "device",
            "torch_version",
            "torchvision_version",
            "hip_version",
            "cuda_available",
            "gpu_name",
            "moge_import_path",
        )},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    output_paths["metadata"] = str(metadata_path)

    return {"success": True, **metadata}


def serve(args: argparse.Namespace) -> dict[str, Any]:
    base = vars(args).copy()
    base.pop("command", None)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = request.get("command")
            payload = base.copy()
            if command == "validate":
                payload["load_model"] = bool(request.get("load_model", True))
                response = validate(SimpleNamespace(**payload))
            elif command == "infer":
                payload.update(
                    {
                        "input": request.get("input"),
                        "output_dir": request.get("output_dir"),
                        "output_type": request.get("output_type"),
                        "resolution_level": int(request.get("resolution_level", 9)),
                        "num_tokens": request.get("num_tokens"),
                    }
                )
                response = infer(SimpleNamespace(**payload))
            else:
                raise WorkerError("invalid_command", f"Unsupported command: {command}")
            response["id"] = request_id
            _emit_line(response)
        except BaseException as exc:
            if isinstance(exc, WorkerError):
                response = {"success": False, "error": {"code": exc.code, "message": str(exc)}}
            else:
                response = {
                    "success": False,
                    "error": {"code": "unexpected_error", "message": f"{type(exc).__name__}: {exc}"},
                }
            response["id"] = request_id
            _emit_line(response)
    return {"success": True}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-python")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", default="v2", choices=["v2"])
    parser.add_argument("--model", default="moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    _add_common(validate_parser)
    validate_parser.add_argument("--load-model", action="store_true")

    infer_parser = subparsers.add_parser("infer")
    _add_common(infer_parser)
    infer_parser.add_argument("--input", required=True)
    infer_parser.add_argument("--output-dir", required=True)
    infer_parser.add_argument("--output-type", action="append", choices=sorted(SUPPORTED_OUTPUTS))
    infer_parser.add_argument("--resolution-level", type=int, default=9)
    infer_parser.add_argument("--num-tokens", type=int)

    serve_parser = subparsers.add_parser("serve")
    _add_common(serve_parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "validate":
            _emit(validate(args))
        elif args.command == "infer":
            _emit(infer(args))
        elif args.command == "serve":
            serve(args)
        else:
            raise WorkerError("invalid_command", f"Unsupported command: {args.command}")
        return 0
    except BaseException as exc:
        return _fail(exc)


if __name__ == "__main__":
    raise SystemExit(main())
