import base64
import io
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4
from time import perf_counter

from dotenv import load_dotenv
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "data" / "images"
MASK_DIR = ROOT_DIR / "data" / "masks"
DEFAULT_MODEL_TYPE = "vit_h"
SUPPORTED_MODEL_TYPES = {"default", "vit_b", "vit_l", "vit_h"}
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


@dataclass
class CandidatePrediction:
    image_key: str
    width: int
    height: int
    masks: Any
    scores: list[float]
    areas: list[int]
    bboxes: list[list[int]]
    logits: Any
    elapsed_ms: float


_state = SamState()
_lock = RLock()
_mask_metadata: dict[str, dict[str, Any]] = {}


def get_status() -> dict[str, str | bool | None]:
    with _lock:
        return {
            "ok": True,
            "model_loaded": _state.predictor is not None,
            "device": _state.device,
            "checkpoint_path": str(_state.checkpoint_path) if _state.checkpoint_path else None,
            "model_type": _state.model_type,
            "active_image_id": _state.image_id,
        }


def load_model(
    checkpoint_path: str | None = None,
    model_type: str | None = None,
    device: str | None = None,
) -> dict[str, str | bool]:
    with _lock:
        if _state.predictor is not None:
            return _model_status()

        config = _read_config(
            checkpoint_path_override=checkpoint_path,
            model_type_override=model_type,
            device_override=device,
        )

        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise SamEngineError(
                "segment_anything, torch, or another SAM dependency is not installed in the "
                f"current Python environment ({sys.executable}). Install project requirements "
                "and a PyTorch build compatible with your selected SAM_DEVICE, or start the "
                "project from the environment that contains both the app dependencies and SAM.",
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

        return _set_image_unlocked(image_id)


def prepare_image_from_path(image_key: str, image_path: Path) -> dict[str, str | int | bool]:
    """Prepare a trusted server-resolved image path for SAM prediction."""
    with _lock:
        if _state.predictor is None:
            raise SamEngineError("SAM is not loaded. Call /load_model first.", status_code=400)
        width, height = _prepare_image_path_unlocked(image_key, image_path)
        return {
            "image_set": True,
            "image_id": image_key,
            "filename": image_path.name,
            "width": width,
            "height": height,
            "model_loaded": True,
        }


def predict_candidates(
    image_key: str,
    *,
    image_path: Path | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    box: list[float] | None = None,
    mask_input: Any = None,
    multimask_output: bool = True,
) -> CandidatePrediction:
    with _lock:
        clean_image_key = _validate_image_id(image_key)
        normalized_points, normalized_labels, normalized_box = _validate_prompt(
            points=points,
            point_labels=point_labels,
            box=box,
        )

        if _state.predictor is None:
            raise SamEngineError("SAM is not loaded. Call /load_model first.", status_code=400)

        if image_path is not None and _state.image_id != clean_image_key:
            _prepare_image_path_unlocked(clean_image_key, image_path)
        elif _state.image_id != clean_image_key:
            raise SamEngineError("SAM image is not prepared for this prediction.", status_code=400)

        try:
            import numpy as np
        except ImportError as exc:
            raise SamEngineError(
                "numpy is not installed. Install project requirements before predicting.",
                status_code=500,
            ) from exc

        point_coords_array = (
            np.array(normalized_points, dtype=np.float32) if normalized_points else None
        )
        point_labels_array = (
            np.array(normalized_labels, dtype=np.int32) if normalized_labels else None
        )
        box_array = np.array(normalized_box, dtype=np.float32) if normalized_box else None

        started = perf_counter()
        try:
            masks, scores, logits = _state.predictor.predict(
                point_coords=point_coords_array,
                point_labels=point_labels_array,
                box=box_array,
                mask_input=mask_input,
                multimask_output=multimask_output,
            )
        except TypeError:
            if mask_input is not None:
                raise
            try:
                masks, scores, logits = _state.predictor.predict(
                    point_coords=point_coords_array,
                    point_labels=point_labels_array,
                    box=box_array,
                    multimask_output=multimask_output,
                )
            except Exception as exc:
                raise SamEngineError(f"SAM prediction failed: {exc}", status_code=500) from exc
        except Exception as exc:
            raise SamEngineError(f"SAM prediction failed: {exc}", status_code=500) from exc
        elapsed_ms = (perf_counter() - started) * 1000

        expected_shape = (_state.image_height, _state.image_width)
        areas: list[int] = []
        bboxes: list[list[int]] = []
        normalized_masks = []
        for mask in masks:
            mask_bool = mask.astype(bool)
            if mask_bool.shape != expected_shape:
                raise SamEngineError(
                    "SAM returned a mask with unexpected size: "
                    f"{mask_bool.shape}, expected {expected_shape}",
                    status_code=500,
                )
            normalized_masks.append(mask_bool)
            areas.append(int(mask_bool.sum()))
            bboxes.append(_mask_bbox(mask_bool))

        return CandidatePrediction(
            image_key=clean_image_key,
            width=int(_state.image_width or 0),
            height=int(_state.image_height or 0),
            masks=np.array(normalized_masks, dtype=bool),
            scores=[float(score) for score in scores],
            areas=areas,
            bboxes=bboxes,
            logits=logits,
            elapsed_ms=elapsed_ms,
        )


def predict(
    image_id: str,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    box: list[float] | None = None,
    multimask_output: bool = True,
) -> dict[str, Any]:
    clean_image_id = _validate_image_id(image_id)
    image_path = _find_image_path(clean_image_id)
    prediction = predict_candidates(
        clean_image_id,
        image_path=image_path,
        points=points,
        point_labels=point_labels,
        box=box,
        multimask_output=multimask_output,
    )

    with _lock:
        try:
            import numpy as np
        except ImportError as exc:
            raise SamEngineError(
                "numpy is not installed. Install project requirements before predicting.",
                status_code=500,
            ) from exc
        MASK_DIR.mkdir(parents=True, exist_ok=True)
        mask_results = []

        for index, mask in enumerate(prediction.masks):
            mask_bool = mask.astype(bool)
            mask_id = uuid4().hex
            mask_path = MASK_DIR / f"{mask_id}.png"
            mask_png = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
            mask_png.save(mask_path)

            png_buffer = io.BytesIO()
            mask_png.save(png_buffer, format="PNG")
            png_base64 = base64.b64encode(png_buffer.getvalue()).decode("ascii")

            metadata = {
                "mask_id": mask_id,
                "image_id": clean_image_id,
                "path": mask_path,
                "score": prediction.scores[index],
                "area": prediction.areas[index],
                "bbox": prediction.bboxes[index],
                "width": prediction.width,
                "height": prediction.height,
            }
            _mask_metadata[mask_id] = metadata

            mask_results.append(
                {
                    "mask_id": mask_id,
                    "score": prediction.scores[index],
                    "area": prediction.areas[index],
                    "bbox": prediction.bboxes[index],
                    "png_base64": png_base64,
                }
            )

        return {
            "image_id": clean_image_id,
            "masks": mask_results,
        }


def get_mask_path(mask_id: str) -> Path:
    clean_mask_id = mask_id.strip()
    if not clean_mask_id:
        raise SamEngineError("mask_id is required.", status_code=400)

    mask_path = MASK_DIR / f"{clean_mask_id}.png"
    if not mask_path.is_file():
        raise SamEngineError(f"No mask found for mask_id '{clean_mask_id}'.", status_code=404)

    return mask_path


def validate_model_config(
    checkpoint_path: str | None = None,
    model_type: str | None = None,
    device: str | None = None,
) -> dict[str, str]:
    config = _read_config(
        checkpoint_path_override=checkpoint_path,
        model_type_override=model_type,
        device_override=device,
    )
    return {
        "checkpoint_path": str(config["checkpoint_path"]),
        "model_type": str(config["model_type"]),
        "device": str(config["device"]),
        "python_executable": sys.executable,
    }


def _read_config(
    checkpoint_path_override: str | None = None,
    model_type_override: str | None = None,
    device_override: str | None = None,
) -> dict[str, str | Path]:
    explicit_checkpoint = os.environ.get("SAM_CHECKPOINT")
    explicit_legacy_checkpoint = os.environ.get("SAM_CHECKPOINT_PATH")
    explicit_model_type = os.environ.get("SAM_MODEL_TYPE")
    explicit_device = os.environ.get("SAM_DEVICE")
    load_dotenv(ROOT_DIR / ".env")

    checkpoint_path_raw = _first_config_value(
        checkpoint_path_override,
        explicit_checkpoint,
        explicit_legacy_checkpoint,
        os.getenv("SAM_CHECKPOINT"),
        os.getenv("SAM_CHECKPOINT_PATH"),
    )
    if not checkpoint_path_raw:
        raise SamEngineError(
            "SAM_CHECKPOINT is missing. Set SAM_CHECKPOINT in .env or the environment. "
            "SAM_CHECKPOINT_PATH is also accepted for backward compatibility.",
            status_code=400,
        )

    checkpoint_path = Path(checkpoint_path_raw).expanduser()
    if not checkpoint_path.is_file():
        raise SamEngineError(
            f"SAM_CHECKPOINT does not point to a valid checkpoint file: {checkpoint_path}",
            status_code=400,
        )

    model_type = (
        _first_config_value(model_type_override, explicit_model_type, os.getenv("SAM_MODEL_TYPE"))
        or DEFAULT_MODEL_TYPE
    )
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        raise SamEngineError(
            f"Unsupported SAM_MODEL_TYPE '{model_type}'. Expected one of: {supported}. "
            "For the local sam_vit_b_01ec64.pth checkpoint, set SAM_MODEL_TYPE=vit_b.",
            status_code=400,
        )

    device = _first_config_value(device_override, explicit_device, os.getenv("SAM_DEVICE")) or _default_device()
    device = _validate_device(device)

    return {
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "device": device,
    }


def _first_config_value(*values: str | None) -> str:
    for value in values:
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped

    return ""


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"

    if normalized == "cuda" or normalized.startswith("cuda:"):
        try:
            import torch
        except ImportError as exc:
            raise SamEngineError(
                "SAM_DEVICE was set to CUDA, but torch is not installed in the current "
                f"Python environment ({sys.executable}). Activate the correct environment "
                "or install a PyTorch build compatible with CUDA.",
                status_code=500,
            ) from exc

        if not torch.cuda.is_available():
            raise SamEngineError(
                "SAM_DEVICE was set to CUDA, but CUDA is not available to the installed "
                f"PyTorch build in {sys.executable}. Set SAM_DEVICE=cpu for a CPU-only setup.",
                status_code=400,
            )

        if ":" in normalized:
            try:
                index = int(normalized.split(":", 1)[1])
            except ValueError as exc:
                raise SamEngineError(
                    f"Invalid SAM_DEVICE '{device}'. Use cpu, cuda, or cuda:<index>.",
                    status_code=400,
                ) from exc

            if index < 0 or index >= torch.cuda.device_count():
                raise SamEngineError(
                    f"SAM_DEVICE '{device}' is not available. PyTorch reports "
                    f"{torch.cuda.device_count()} CUDA device(s).",
                    status_code=400,
                )

        return normalized

    raise SamEngineError(
        f"Invalid SAM_DEVICE '{device}'. Use cpu, cuda, or cuda:<index>.",
        status_code=400,
    )


def _set_image_unlocked(image_id: str) -> dict[str, str | int | bool]:
    clean_image_id = _validate_image_id(image_id)
    image_path = _find_image_path(clean_image_id)
    width, height = _prepare_image_path_unlocked(clean_image_id, image_path)

    return {
        "image_set": True,
        "image_id": clean_image_id,
        "filename": image_path.name,
        "width": width,
        "height": height,
        "model_loaded": True,
    }


def _prepare_image_path_unlocked(image_key: str, image_path: Path) -> tuple[int, int]:
    clean_image_key = _validate_image_id(image_key)
    resolved = image_path.resolve()

    if _state.image_id == clean_image_key and _state.image_path == resolved:
        return int(_state.image_width or 0), int(_state.image_height or 0)

    try:
        import numpy as np
    except ImportError as exc:
        raise SamEngineError(
            "numpy is not installed. Install project requirements before setting an image.",
            status_code=500,
        ) from exc

    try:
        with Image.open(resolved) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            image_array = np.array(rgb_image)
    except Exception as exc:
        raise SamEngineError(f"Failed to load image '{clean_image_key}': {exc}", status_code=400) from exc

    try:
        _state.predictor.set_image(image_array)
    except Exception as exc:
        raise SamEngineError(f"Failed to set image in SAM predictor: {exc}", status_code=500) from exc

    _state.image_id = clean_image_key
    _state.image_path = resolved
    _state.image_width = width
    _state.image_height = height

    return width, height


def _find_image_path(image_id: str) -> Path:
    clean_image_id = _validate_image_id(image_id)

    candidates = [
        path
        for path in IMAGE_DIR.glob(f"{clean_image_id}.*")
        if path.suffix.lower() in VALID_IMAGE_SUFFIXES
    ]

    if not candidates:
        raise SamEngineError(f"No uploaded image found for image_id '{clean_image_id}'.", status_code=400)

    return candidates[0]


def _validate_image_id(image_id: str) -> str:
    clean_image_id = (image_id or "").strip()
    if not clean_image_id:
        raise SamEngineError("image_id is required.", status_code=400)

    return clean_image_id


def _validate_prompt(
    points: list[list[float]] | None,
    point_labels: list[int] | None,
    box: list[float] | None,
) -> tuple[list[list[float]], list[int], list[float] | None]:
    normalized_points = points or []
    normalized_labels = point_labels or []

    if len(normalized_points) != len(normalized_labels):
        raise SamEngineError("points and point_labels must have the same length.", status_code=400)

    parsed_points = []
    for point in normalized_points:
        if len(point) != 2:
            raise SamEngineError("Each point must be [x, y].", status_code=400)
        parsed_points.append([
            _as_float(point[0], "point x"),
            _as_float(point[1], "point y"),
        ])

    for label in normalized_labels:
        if label not in (0, 1):
            raise SamEngineError("point_labels must contain only 1 or 0.", status_code=400)

    normalized_box = None
    if box is not None:
        if len(box) != 4:
            raise SamEngineError("box must be [x1, y1, x2, y2].", status_code=400)
        normalized_box = [_as_float(value, "box coordinate") for value in box]

    if not normalized_points and normalized_box is None:
        raise SamEngineError("Provide at least one point or a box before predicting.", status_code=400)

    return parsed_points, normalized_labels, normalized_box


def _as_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SamEngineError(f"Invalid {label}: {value}", status_code=400) from exc
    if not math.isfinite(parsed):
        raise SamEngineError(f"Invalid {label}: value must be finite", status_code=400)
    return parsed


def _mask_bbox(mask_bool: Any) -> list[int]:
    import numpy as np

    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]

    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _model_status() -> dict[str, str | bool]:
    return {
        "loaded": True,
        "model_type": _state.model_type or "",
        "device": _state.device or "",
        "checkpoint_path": str(_state.checkpoint_path or ""),
    }
