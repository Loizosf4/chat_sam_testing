"""Main-process MoGe integration via an isolated external Python worker."""

from __future__ import annotations

import json
import os
import atexit
import itertools
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
MOGE_OUTPUT_DIR = ROOT_DIR / "data" / "exports" / "moge"
WORKER_PATH = Path(__file__).resolve().parent / "moge_worker.py"
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_OUTPUT_TYPES = ["geometry_npz", "points", "depth", "normal", "mask", "intrinsics", "previews"]
WORKER_REQUEST_TIMEOUT_SECONDS = 900


class MogeEngineError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class MogeConfig:
    python: Path
    repository: Path
    checkpoint: Path
    version: str
    device: str
    model: str
    allow_cpu: bool


@dataclass
class _WorkerService:
    process: subprocess.Popen[str]
    config_key: tuple[str, str, str, str, str, str, bool]
    responses: "queue.Queue[dict[str, Any]]"


_service: _WorkerService | None = None
_service_lock = threading.RLock()
_request_ids = itertools.count(1)


def get_status() -> dict[str, Any]:
    try:
        config = _read_config(validate_paths=False)
    except MogeEngineError as exc:
        return {
            "ok": False,
            "configured": False,
            "error": str(exc),
            "model_loaded": False,
        }

    return {
        "ok": True,
        "configured": True,
        "model_loaded": False,
        "python": str(config.python),
        "repository": str(config.repository),
        "checkpoint": str(config.checkpoint),
        "version": config.version,
        "device": config.device,
        "model": config.model,
        "allow_cpu": config.allow_cpu,
        "worker": str(WORKER_PATH),
        "worker_running": _service is not None and _service.process.poll() is None,
        "supported_outputs": DEFAULT_OUTPUT_TYPES,
    }


def validate_model(load_model: bool = True) -> dict[str, Any]:
    config = _read_config()
    return _request_service(config, {"command": "validate", "load_model": load_model}, timeout=300)


def run_inference(
    image_id: str,
    output_dir: str | None = None,
    requested_outputs: list[str] | None = None,
    resolution_level: int = 9,
    num_tokens: int | None = None,
) -> dict[str, Any]:
    config = _read_config()
    image_path = _find_image_path(image_id)
    destination = _resolve_output_dir(output_dir, image_id)
    outputs = requested_outputs or DEFAULT_OUTPUT_TYPES
    request: dict[str, Any] = {
        "command": "infer",
        "input": str(image_path),
        "output_dir": str(destination),
        "resolution_level": resolution_level,
        "output_type": outputs,
    }
    if num_tokens is not None:
        if num_tokens < 1:
            raise MogeEngineError("num_tokens must be a positive integer.")
        request["num_tokens"] = num_tokens
    return _request_service(config, request, timeout=WORKER_REQUEST_TIMEOUT_SECONDS)


def run_inference_from_path(
    *,
    image_path: Path,
    output_dir: Path,
    requested_outputs: list[str] | None = None,
    resolution_level: int = 9,
    num_tokens: int | None = None,
) -> dict[str, Any]:
    """Run MoGe against an internally resolved, server-controlled image path."""
    config = _read_config()
    source = Path(image_path).resolve()
    if not source.is_file() or source.suffix.lower() not in VALID_IMAGE_SUFFIXES:
        raise MogeEngineError("source image is not a supported image", 400)
    try:
        with Image.open(source) as image:
            image.verify()
    except Exception as exc:
        raise MogeEngineError("source image is invalid", 400) from exc
    destination = Path(output_dir).resolve()
    outputs = requested_outputs or DEFAULT_OUTPUT_TYPES
    unsupported = sorted(set(outputs) - set(DEFAULT_OUTPUT_TYPES))
    if unsupported:
        raise MogeEngineError(f"unsupported MoGe output type(s): {', '.join(unsupported)}")
    if resolution_level < 1 or resolution_level > 9:
        raise MogeEngineError("resolution_level must be between 1 and 9")
    request: dict[str, Any] = {
        "command": "infer",
        "input": str(source),
        "output_dir": str(destination),
        "resolution_level": resolution_level,
        "output_type": outputs,
    }
    if num_tokens is not None:
        if num_tokens < 1:
            raise MogeEngineError("num_tokens must be a positive integer.")
        request["num_tokens"] = num_tokens
    return _request_service(config, request, timeout=WORKER_REQUEST_TIMEOUT_SECONDS)


def validate_model_config() -> dict[str, str | bool]:
    config = _read_config()
    return {
        "python": str(config.python),
        "repository": str(config.repository),
        "checkpoint": str(config.checkpoint),
        "version": config.version,
        "device": config.device,
        "model": config.model,
        "allow_cpu": config.allow_cpu,
        "current_python_executable": sys.executable,
        "worker": str(WORKER_PATH),
    }


def _read_config(validate_paths: bool = True) -> MogeConfig:
    load_dotenv(ROOT_DIR / ".env")

    python_raw = _required_env("MOGE_PYTHON")
    repository_raw = _required_env("MOGE_REPOSITORY")
    checkpoint_raw = _required_env("MOGE_CHECKPOINT")
    version = os.getenv("MOGE_VERSION", "v2").strip() or "v2"
    device = os.getenv("MOGE_DEVICE", "cuda").strip() or "cuda"
    model = os.getenv("MOGE_MODEL", "moge-2-vitl-normal").strip() or "moge-2-vitl-normal"
    allow_cpu = os.getenv("MOGE_ALLOW_CPU", "").strip().lower() in {"1", "true", "yes", "on"}

    if version != "v2":
        raise MogeEngineError("Only MOGE_VERSION=v2 is supported by this integration.")
    if device.strip().lower() == "cpu" and not allow_cpu:
        raise MogeEngineError(
            "MOGE_DEVICE=cpu requires MOGE_ALLOW_CPU=true so CPU fallback is explicit."
        )

    config = MogeConfig(
        python=Path(python_raw).expanduser(),
        repository=Path(repository_raw).expanduser(),
        checkpoint=Path(checkpoint_raw).expanduser(),
        version=version,
        device=device,
        model=model,
        allow_cpu=allow_cpu,
    )
    if validate_paths:
        _validate_paths(config)
    return config


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MogeEngineError(f"{name} is missing. Set it in .env or the process environment.")
    return value


def _validate_paths(config: MogeConfig) -> None:
    if not config.python.is_file():
        raise MogeEngineError(f"MOGE_PYTHON does not exist: {config.python}")
    if not config.repository.is_dir():
        raise MogeEngineError(f"MOGE_REPOSITORY does not exist: {config.repository}")
    if not config.checkpoint.is_file():
        raise MogeEngineError(f"MOGE_CHECKPOINT does not exist: {config.checkpoint}")
    if not WORKER_PATH.is_file():
        raise MogeEngineError(f"MoGe worker script is missing: {WORKER_PATH}", status_code=500)


def _worker_base_args(config: MogeConfig, command: str) -> list[str]:
    args = [
        str(WORKER_PATH),
        command,
        "--expected-python",
        str(config.python.resolve()),
        "--repository",
        str(config.repository.resolve()),
        "--checkpoint",
        str(config.checkpoint.resolve()),
        "--version",
        config.version,
        "--model",
        config.model,
        "--device",
        config.device,
    ]
    if config.allow_cpu:
        args.append("--allow-cpu")
    return args


def _config_key(config: MogeConfig) -> tuple[str, str, str, str, str, str, bool]:
    return (
        str(config.python.resolve()),
        str(config.repository.resolve()),
        str(config.checkpoint.resolve()),
        config.version,
        config.device,
        config.model,
        config.allow_cpu,
    )


def _request_service(config: MogeConfig, request: dict[str, Any], timeout: int) -> dict[str, Any]:
    with _service_lock:
        service = _ensure_service(config)
        request = dict(request)
        request["id"] = next(_request_ids)
        assert service.process.stdin is not None
        try:
            service.process.stdin.write(json.dumps(request) + "\n")
            service.process.stdin.flush()
        except OSError as exc:
            _stop_service()
            raise MogeEngineError(f"Failed to write to MoGe worker: {exc}", 500) from exc

        try:
            response = service.responses.get(timeout=timeout)
        except queue.Empty as exc:
            _stop_service()
            raise MogeEngineError(f"MoGe worker timed out after {timeout} seconds.", 504) from exc

        if response.get("id") != request["id"]:
            _stop_service()
            raise MogeEngineError("MoGe worker returned an out-of-order response.", 500)
        if not response.get("success"):
            error = response.get("error") if isinstance(response, dict) else None
            message = (
                str(error.get("message") or error.get("code"))
                if isinstance(error, dict)
                else "MoGe worker failed."
            )
            raise MogeEngineError(message, 500)
        return response


def _ensure_service(config: MogeConfig) -> _WorkerService:
    global _service
    key = _config_key(config)
    if _service is not None and _service.config_key == key and _service.process.poll() is None:
        return _service
    _stop_service()

    env = os.environ.copy()
    repository = str(config.repository.resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repository if not existing_pythonpath else repository + os.pathsep + existing_pythonpath
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            [str(config.python), *_worker_base_args(config, "serve")],
            cwd=str(ROOT_DIR),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise MogeEngineError(f"Failed to start MoGe worker: {exc}", 500) from exc

    responses: "queue.Queue[dict[str, Any]]" = queue.Queue()
    service = _WorkerService(process=process, config_key=key, responses=responses)

    def read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                responses.put(json.loads(line))
            except json.JSONDecodeError:
                responses.put(
                    {
                        "success": False,
                        "error": {
                            "code": "worker_non_json",
                            "message": f"MoGe worker returned non-JSON output: {line.strip()}",
                        },
                    }
                )

    threading.Thread(target=read_stdout, daemon=True).start()
    _service = service
    return service


def _stop_service() -> None:
    global _service
    service = _service
    _service = None
    if service is None:
        return
    process = service.process
    if process.poll() is not None:
        return
    try:
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        process.kill()


def _find_image_path(image_id: str) -> Path:
    clean_image_id = (image_id or "").strip()
    if not clean_image_id:
        raise MogeEngineError("image_id is required.")

    candidates = [
        path
        for path in IMAGE_DIR.glob(f"{clean_image_id}.*")
        if path.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]
    if not candidates:
        raise MogeEngineError(f"No uploaded image found for image_id '{clean_image_id}'.", 404)
    return candidates[0]


def _resolve_output_dir(output_dir: str | None, image_id: str) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser()
        return path if path.is_absolute() else (ROOT_DIR / path).resolve()
    return MOGE_OUTPUT_DIR / image_id


atexit.register(_stop_service)
