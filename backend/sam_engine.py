import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
DEFAULT_MODEL_TYPE = "vit_h"
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SamEngineError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SamState:
    model: Any = None
    predictor: Any = None
    model_type: str | None = None
    device: str | None = None
    checkpoint_path: Path | None = None
    image_id: str | None = None
    image_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None


_state = SamState()
_lock = Lock()


def load_model() -> dict[str, str | bool]:
    with _lock:
        if _state.predictor is not None:
            return _model_status()

        config = _read_config()

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise SamEngineError(
                "segment_anything or one of its dependencies is not installed. "
                "Install project requirements and a PyTorch build compatible with your device.",
                status_code=500,
            ) from exc

        try:
            model = sam_model_registry[config["model_type"]](
                checkpoint=str(config["checkpoint_path"])
            )
        except KeyError as exc:
            supported = ", ".join(sorted(sam_model_registry.keys()))
            raise SamEngineError(
                f"Unsupported SAM_MODEL_TYPE '{config['model_type']}'. Supported values: {supported}",
                status_code=400,
            ) from exc
        except Exception as exc:
            raise SamEngineError(f"Failed to load SAM checkpoint: {exc}", status_code=500) from exc

        try:
            model.to(device=config["device"])
            predictor = SamPredictor(model)
        except Exception as exc:
            raise SamEngineError(f"Failed to initialize SAM predictor: {exc}", status_code=500) from exc

        _state.model = model
        _state.predictor = predictor
        _state.model_type = config["model_type"]
        _state.device = config["device"]
        _state.checkpoint_path = config["checkpoint_path"]

        return _model_status()


def set_image(image_id: str) -> dict[str, str | int | bool]:
    with _lock:
        if _state.predictor is None:
            raise SamEngineError("SAM is not loaded. Call /load_model first.", status_code=400)

        image_path = _find_image_path(image_id)

        try:
            import numpy as np
        except ImportError as exc:
            raise SamEngineError(
                "numpy is not installed. Install project requirements before setting an image.",
                status_code=500,
            ) from exc

        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                image_array = np.array(rgb_image)
        except Exception as exc:
            raise SamEngineError(f"Failed to load image '{image_id}': {exc}", status_code=400) from exc

        try:
            _state.predictor.set_image(image_array)
        except Exception as exc:
            raise SamEngineError(f"Failed to set image in SAM predictor: {exc}", status_code=500) from exc

        _state.image_id = image_id
        _state.image_path = image_path
        _state.image_width = width
        _state.image_height = height

        return {
            "image_set": True,
            "image_id": image_id,
            "filename": image_path.name,
            "width": width,
            "height": height,
            "model_loaded": True,
        }


def _read_config() -> dict[str, str | Path]:
    load_dotenv(ROOT_DIR / ".env")

    checkpoint_path_raw = os.getenv("SAM_CHECKPOINT_PATH", "").strip()
    if not checkpoint_path_raw:
        raise SamEngineError(
            "SAM_CHECKPOINT_PATH is missing. Set it in .env or the environment.",
            status_code=400,
        )

    checkpoint_path = Path(checkpoint_path_raw).expanduser()
    if not checkpoint_path.is_file():
        raise SamEngineError(
            f"SAM_CHECKPOINT_PATH does not point to a valid file: {checkpoint_path}",
            status_code=400,
        )

    model_type = os.getenv("SAM_MODEL_TYPE", DEFAULT_MODEL_TYPE).strip() or DEFAULT_MODEL_TYPE
    device = os.getenv("SAM_DEVICE", "").strip() or _default_device()

    return {
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "device": device,
    }


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_image_path(image_id: str) -> Path:
    clean_image_id = image_id.strip()
    if not clean_image_id:
        raise SamEngineError("image_id is required.", status_code=400)

    candidates = [
        path
        for path in IMAGE_DIR.glob(f"{clean_image_id}.*")
        if path.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]

    if not candidates:
        raise SamEngineError(f"No uploaded image found for image_id '{clean_image_id}'.", status_code=404)

    return candidates[0]


def _model_status() -> dict[str, str | bool]:
    return {
        "loaded": True,
        "model_type": _state.model_type or "",
        "device": _state.device or "",
        "checkpoint_path": str(_state.checkpoint_path or ""),
    }
