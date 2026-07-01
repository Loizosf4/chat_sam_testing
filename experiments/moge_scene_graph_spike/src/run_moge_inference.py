"""Run isolated, single-image MoGe-2 inference and save numerical maps."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(EXPERIMENT_ROOT / ".cache" / "huggingface"))

import numpy as np
import torch
from PIL import Image
from moge.model.v2 import MoGeModel


DEFAULT_MODEL = "Ruicheng/moge-2-vits-normal"
DEFAULT_MODEL_REVISION = "679230677b4d282c6f304189a93e98e14f085902"


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _horizontal_fov_degrees(intrinsics: np.ndarray) -> float:
    fx = float(intrinsics[0, 0])
    if not math.isfinite(fx) or fx <= 0:
        raise ValueError(f"Invalid normalized focal length fx={fx}")
    return math.degrees(2.0 * math.atan(0.5 / fx))


def _save_previews(output_dir: Path, depth: np.ndarray, normal: np.ndarray | None, valid: np.ndarray) -> None:
    valid_depth = depth[valid]
    if valid_depth.size:
        low, high = np.percentile(valid_depth, [2, 98])
        span = max(float(high - low), np.finfo(np.float32).eps)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        preview[valid] = np.clip((depth[valid] - low) / span * 255, 0, 255).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(output_dir / "depth_preview.png")
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(output_dir / "valid_mask_preview.png")
    if normal is not None:
        normal_preview = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
        normal_preview[~valid] = 0
        Image.fromarray(normal_preview, mode="RGB").save(output_dir / "normal_preview.png")


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1)

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model = MoGeModel.from_pretrained(args.model, revision=args.model_revision).to(device).eval()
    image_tensor = image_tensor.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.infer(
        image_tensor,
        num_tokens=args.num_tokens,
        apply_mask=True,
        use_fp16=args.fp16 and device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    required = {"points", "depth", "mask", "intrinsics"}
    missing = sorted(required - output.keys())
    if missing:
        raise RuntimeError(f"MoGe output is missing required keys: {missing}")

    points = _as_numpy(output["points"]).astype(np.float32, copy=False)
    depth = _as_numpy(output["depth"]).astype(np.float32, copy=False)
    valid = _as_numpy(output["mask"]).astype(bool, copy=False)
    intrinsics = _as_numpy(output["intrinsics"]).astype(np.float32, copy=False)
    normal = _as_numpy(output["normal"]).astype(np.float32, copy=False) if "normal" in output else None
    fov_x_degrees = _horizontal_fov_degrees(intrinsics)

    arrays: dict[str, np.ndarray] = {
        "points": points,
        "depth": depth,
        "valid_mask": valid,
        "intrinsics": intrinsics,
    }
    if normal is not None:
        arrays["normal"] = normal

    for name, array in arrays.items():
        np.save(output_dir / f"{name}.npy", array, allow_pickle=False)
    np.savez_compressed(output_dir / "geometry.npz", **arrays)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_name": args.model,
        "model_revision": args.model_revision,
        "moge_version": "2",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "runtime_seconds": runtime_seconds,
        "source_image": str(input_path.relative_to(EXPERIMENT_ROOT)),
        "source_image_dimensions": {"width": width, "height": height},
        "num_tokens": args.num_tokens,
        "fp16": args.fp16 and device.type == "cuda",
        "estimated_fov_x_degrees": fov_x_degrees,
        "normalized_intrinsics": intrinsics.tolist(),
        "output_keys": sorted(arrays),
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "python_version": platform.python_version(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.previews:
        _save_previews(output_dir, depth, normal, valid)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXPERIMENT_ROOT / "inputs" / "office_test" / "image.png")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "outputs" / "office_test" / "moge")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific torch device")
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--previews", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
