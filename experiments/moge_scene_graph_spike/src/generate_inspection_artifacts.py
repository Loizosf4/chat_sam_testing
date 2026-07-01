"""Generate inspection artifacts and geometry diagnostics from persisted MoGe output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from PIL import Image


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOGE_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "moge"
DEFAULT_IMAGE = EXPERIMENT_ROOT / "inputs" / "office_test" / "image.png"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    if colors is None:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    else:
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    properties = ["property float x", "property float y", "property float z"]
    if colors is not None:
        vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
        properties.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}", *properties, "end_header", ""]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def _fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold_m: float = 0.05,
    iterations: int = 400,
    seed: int = 24457,
) -> dict[str, Any]:
    if len(points) < 3:
        return {"plausible": False, "reason": "fewer than three candidate points"}
    rng = np.random.default_rng(seed)
    sample = points
    if len(sample) > 12000:
        sample = sample[rng.choice(len(sample), 12000, replace=False)]
    best_inliers = np.zeros(len(sample), dtype=bool)
    best_count = 0
    for _ in range(iterations):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ tri[0])
        inliers = np.abs(sample @ normal + offset) <= threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_count < 3:
        return {"plausible": False, "reason": "RANSAC found no plane"}

    centroid = sample[best_inliers].mean(axis=0)
    _, _, vh = np.linalg.svd(sample[best_inliers] - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    residuals = np.abs(sample @ normal + offset)
    inliers = residuals <= threshold_m
    inlier_residuals = residuals[inliers]
    inlier_ratio = float(inliers.mean())
    p95 = float(np.percentile(inlier_residuals, 95))
    plausible = inlier_ratio >= 0.55 and p95 <= threshold_m
    return {
        "plausible": plausible,
        "candidate_point_count": int(len(points)),
        "sampled_point_count": int(len(sample)),
        "distance_threshold_m": threshold_m,
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": inlier_ratio,
        "plane_normal_camera_xyz": normal.tolist(),
        "plane_offset_m": offset,
        "inlier_residual_m": {
            "median": float(np.median(inlier_residuals)),
            "p95": p95,
            "max": float(inlier_residuals.max()),
        },
    }


def _region_plane_report(
    name: str,
    bounds: tuple[float, float, float, float],
    points: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    height, width = valid.shape
    x0, y0, x1, y1 = bounds
    xs = slice(round(x0 * width), round(x1 * width))
    ys = slice(round(y0 * height), round(y1 * height))
    region_valid = valid[ys, xs]
    region_points = points[ys, xs][region_valid]
    region_normals = normals[ys, xs][region_valid]
    result: dict[str, Any] = {
        "name": name,
        "normalized_bounds_xyxy": list(bounds),
        "pixel_bounds_xyxy": [xs.start, ys.start, xs.stop, ys.stop],
        "valid_pixel_count": int(region_valid.sum()),
        "valid_pixel_percentage": float(region_valid.mean() * 100.0),
    }
    if not len(region_points):
        result["plane_fit"] = {"plausible": False, "reason": "no valid pixels"}
        return result

    reference_normal = np.median(region_normals, axis=0)
    reference_normal /= np.linalg.norm(reference_normal)
    angular_selection = region_normals @ reference_normal >= math.cos(math.radians(25.0))
    selected_points = region_points[angular_selection]
    result["median_predicted_normal_camera_xyz"] = reference_normal.tolist()
    result["normal_consistent_pixel_percentage"] = float(angular_selection.mean() * 100.0)
    result["plane_fit"] = _fit_plane_ransac(selected_points)
    return result


def _plane_angle_degrees(a: list[float], b: list[float]) -> float:
    dot = abs(float(np.dot(np.asarray(a), np.asarray(b))))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def _save_previews(
    inspection_dir: Path,
    depth: np.ndarray,
    normal: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    valid_depth = depth[valid & np.isfinite(depth)]
    clip_min, clip_max = np.percentile(valid_depth, [2, 98])
    scaled = np.zeros(depth.shape, dtype=np.float32)
    scaled[valid] = np.clip((depth[valid] - clip_min) / (clip_max - clip_min), 0.0, 1.0)
    depth_rgb = (colormaps["turbo"](scaled)[..., :3] * 255).astype(np.uint8)
    depth_rgb[~valid] = 0
    Image.fromarray(depth_rgb).save(inspection_dir / "depth_color.png")

    normal_rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)
    normal_rgb[~valid] = 0
    Image.fromarray(normal_rgb).save(inspection_dir / "normal_map.png")

    validity = valid.astype(np.uint8) * 255
    Image.fromarray(validity).save(inspection_dir / "validity_mask.png")
    return {
        "depth_preview": {
            "path": "depth_color.png",
            "colormap": "turbo",
            "clip_percentiles": [2, 98],
            "clip_depth_m": [float(clip_min), float(clip_max)],
            "invalid_color_rgb": [0, 0, 0],
            "note": "Visualization clipping only; persisted depth values were not changed.",
        },
        "normal_preview": {"path": "normal_map.png", "mapping": "[-1, 1] to [0, 255] RGB", "invalid_color_rgb": [0, 0, 0]},
        "validity_preview": {"path": "validity_mask.png", "valid_value": 255, "invalid_value": 0},
    }


def generate(moge_dir: Path, source_image: Path) -> dict[str, Any]:
    moge_dir = moge_dir.resolve()
    inspection_dir = moge_dir / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((moge_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(moge_dir / "geometry.npz", allow_pickle=False) as archive:
        points = archive["points"]
        depth = archive["depth"]
        normal = archive["normal"]
        valid = archive["valid_mask"].astype(bool)
        intrinsics = archive["intrinsics"]
    with Image.open(source_image) as image:
        colors = np.asarray(image.convert("RGB"), dtype=np.uint8)

    height, width = valid.shape
    if colors.shape[:2] != (height, width):
        raise ValueError(f"source image shape {colors.shape[:2]} does not match geometry {(height, width)}")

    previews = _save_previews(inspection_dir, depth, normal, valid)
    finite_vertex = valid & np.isfinite(depth) & np.isfinite(points).all(axis=-1) & np.isfinite(normal).all(axis=-1)
    vertices = points[finite_vertex].astype(np.float32, copy=False)
    vertex_colors = colors[finite_vertex]
    _write_binary_ply(inspection_dir / "point_cloud.ply", vertices)
    _write_binary_ply(inspection_dir / "point_cloud_colored.ply", vertices, vertex_colors)

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    fov_x = math.degrees(2 * math.atan(0.5 / fx))
    fov_y = math.degrees(2 * math.atan(0.5 / fy))
    intrinsics_summary = {
        "coordinate_convention": "OpenCV camera coordinates: +x right, +y down, +z forward",
        "normalized_intrinsics": intrinsics.tolist(),
        "pixel_intrinsics": {
            "fx": fx * width,
            "fy": fy * height,
            "cx": float(intrinsics[0, 2]) * width,
            "cy": float(intrinsics[1, 2]) * height,
        },
        "estimated_fov_degrees": {"horizontal": fov_x, "vertical": fov_y},
        "source_image_dimensions": {"width": width, "height": height},
        "model_name": metadata["model_name"],
        "model_revision": metadata.get("model_revision"),
    }
    (inspection_dir / "camera_summary.json").write_text(json.dumps(intrinsics_summary, indent=2) + "\n", encoding="utf-8")
    camera_md = (
        "# Camera summary\n\n"
        f"- Image: {width} × {height}\n"
        f"- Estimated horizontal FOV: {fov_x:.3f}°\n"
        f"- Estimated vertical FOV: {fov_y:.3f}°\n"
        f"- Normalized focal lengths: fx={fx:.6f}, fy={fy:.6f}\n"
        f"- Normalized principal point: cx={intrinsics[0, 2]:.6f}, cy={intrinsics[1, 2]:.6f}\n"
        f"- Pixel focal lengths: fx={fx * width:.3f}, fy={fy * height:.3f}\n"
        "- Coordinate convention: OpenCV (+x right, +y down, +z forward)\n"
    )
    (inspection_dir / "camera_summary.md").write_text(camera_md, encoding="utf-8")

    total = valid.size
    valid_count = int(valid.sum())
    invalid_count = total - valid_count
    finite_depth = np.isfinite(depth)
    finite_points = np.isfinite(points).all(axis=-1)
    finite_normals = np.isfinite(normal).all(axis=-1)
    valid_depth = depth[valid & finite_depth]
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]

    pair_valid_h = valid[:, 1:] & valid[:, :-1] & finite_depth[:, 1:] & finite_depth[:, :-1]
    pair_valid_v = valid[1:, :] & valid[:-1, :] & finite_depth[1:, :] & finite_depth[:-1, :]
    jumps_h = np.abs(depth[:, 1:][pair_valid_h] - depth[:, :-1][pair_valid_h])
    jumps_v = np.abs(depth[1:, :][pair_valid_v] - depth[:-1, :][pair_valid_v])
    jumps = np.concatenate([jumps_h, jumps_v])
    discontinuity_threshold_m = 0.5
    suspicious_jumps = jumps > discontinuity_threshold_m

    region_specs = {
        "floor": (0.08, 0.55, 0.92, 0.94),
        "left_wall": (0.05, 0.18, 0.49, 0.58),
        "right_wall": (0.49, 0.16, 0.93, 0.58),
    }
    regions = {name: _region_plane_report(name, bounds, points, normal, valid) for name, bounds in region_specs.items()}
    plane_normals = {
        name: result["plane_fit"].get("plane_normal_camera_xyz")
        for name, result in regions.items()
        if result["plane_fit"].get("plane_normal_camera_xyz") is not None
    }
    angles: dict[str, float] = {}
    for a, b in [("floor", "left_wall"), ("floor", "right_wall"), ("left_wall", "right_wall")]:
        if a in plane_normals and b in plane_normals:
            angles[f"{a}_to_{b}"] = _plane_angle_degrees(plane_normals[a], plane_normals[b])
    regions_plausible = all(result["plane_fit"].get("plausible", False) for result in regions.values())
    angles_plausible = len(angles) == 3 and all(70.0 <= angle <= 110.0 for angle in angles.values())
    coherent = regions_plausible and angles_plausible

    concerns: list[str] = []
    valid_nonfinite = int((valid & ~(finite_depth & finite_points & finite_normals)).sum())
    if valid_nonfinite:
        concerns.append(f"{valid_nonfinite} valid pixels contain non-finite geometry.")
    valid_percentage = valid_count / total * 100.0
    if valid_percentage < 80.0:
        concerns.append(f"Only {valid_percentage:.2f}% of image pixels are valid; the black exterior/background is excluded.")
    discontinuity_percentage = float(suspicious_jumps.mean() * 100.0) if jumps.size else 0.0
    if discontinuity_percentage > 2.0:
        concerns.append(
            f"{discontinuity_percentage:.2f}% of adjacent valid pixel pairs exceed the fixed {discontinuity_threshold_m:.2f} m depth-jump threshold."
        )
    if fov_x < 30.0 or fov_x > 120.0:
        concerns.append(f"Estimated horizontal FOV is {fov_x:.2f}°, outside the report's conventional 30°–120° range.")
    if not coherent:
        concerns.append("One or more floor/wall plane fits or their pairwise orientations failed the coherence thresholds.")

    usable = valid_nonfinite == 0 and valid_percentage >= 60.0 and coherent
    report: dict[str, Any] = {
        "schema_version": 1,
        "assessment": {
            "usable_for_inspection": usable,
            "summary": "Usable with documented caveats." if usable and concerns else ("Usable." if usable else "Not currently usable."),
            "concerns": concerns,
        },
        "limitations": [
            "No ground-truth camera or geometry is available, so absolute metric scale cannot be independently verified.",
            "Planarity checks use fixed image-space regions rather than semantic floor/wall masks.",
        ],
        "source": {
            "geometry_archive": str((moge_dir / "geometry.npz").relative_to(EXPERIMENT_ROOT)),
            "image": str(source_image.resolve().relative_to(EXPERIMENT_ROOT)),
            "model_name": metadata["model_name"],
            "model_revision": metadata.get("model_revision"),
        },
        "array_shapes": {"points": list(points.shape), "depth": list(depth.shape), "normal": list(normal.shape), "valid_mask": list(valid.shape)},
        "validity": {
            "total_pixels": total,
            "valid_pixels": valid_count,
            "invalid_pixels": invalid_count,
            "valid_percentage": valid_percentage,
        },
        "non_finite_values": {
            "valid_depth_non_finite": int((valid & ~finite_depth).sum()),
            "valid_points_non_finite": int((valid & ~finite_points).sum()),
            "valid_normals_non_finite": int((valid & ~finite_normals).sum()),
            "masked_out_depth_non_finite": int((~valid & ~finite_depth).sum()),
            "masked_out_points_non_finite": int((~valid & ~finite_points).sum()),
            "note": "Masked-out point/depth infinities are retained from raw MoGe output and are excluded from the PLY files; they were not normalized or repaired.",
        },
        "depth_m": {
            "minimum": float(valid_depth.min()),
            "maximum": float(valid_depth.max()),
            "mean": float(valid_depth.mean()),
            "percentiles": {str(p): float(v) for p, v in zip(percentiles, np.percentile(valid_depth, percentiles))},
        },
        "depth_discontinuities": {
            "adjacent_valid_pair_count": int(jumps.size),
            "absolute_jump_threshold_m": discontinuity_threshold_m,
            "suspicious_pair_count": int(suspicious_jumps.sum()),
            "suspicious_pair_percentage": discontinuity_percentage,
            "absolute_jump_percentiles_m": {
                str(p): float(v) for p, v in zip([50, 90, 95, 99, 100], np.percentile(jumps, [50, 90, 95, 99, 100]))
            },
            "note": "Large jumps can be real object boundaries; this metric flags candidates rather than smoothing them.",
        },
        "camera": intrinsics_summary,
        "planar_regions": {
            "method": "Fixed image-space ROIs; pixels within 25° of each ROI's median predicted normal; deterministic RANSAC with a fixed 0.05 m residual threshold.",
            "regions": regions,
            "pairwise_plane_normal_angles_degrees": angles,
            "major_regions_plausible": regions_plausible,
            "floor_and_walls_geometrically_coherent": coherent,
            "coherence_thresholds": {"plane_inlier_ratio_minimum": 0.55, "pairwise_normal_angle_degrees": [70.0, 110.0]},
            "limitation": "This is an image-space diagnostic, not semantic segmentation; furniture and boxes inside an ROI may reduce its plane score.",
        },
        "inspection_artifacts": {
            **previews,
            "point_cloud": {"path": "point_cloud.ply", "vertex_count": int(len(vertices)), "colors": False},
            "source_colored_point_cloud": {"path": "point_cloud_colored.ply", "vertex_count": int(len(vertices)), "colors": True},
            "camera_summary_json": "camera_summary.json",
            "camera_summary_markdown": "camera_summary.md",
        },
    }
    report = json.loads(json.dumps(report, default=_json_value, allow_nan=False))
    (inspection_dir / "geometry_quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    region_lines = []
    for name, region in regions.items():
        fit = region["plane_fit"]
        region_lines.append(
            f"| {name} | {region['valid_pixel_percentage']:.2f}% | {region.get('normal_consistent_pixel_percentage', 0):.2f}% | "
            f"{fit.get('inlier_ratio', 0) * 100:.2f}% | {'yes' if fit.get('plausible') else 'no'} |"
        )
    concern_md = "\n".join(f"- {item}" for item in concerns) or "- None detected by the configured checks."
    report_md = f"""# MoGe geometry quality report

## Assessment

- Usable for inspection: **{'yes' if usable else 'no'}**
- Floor and walls geometrically coherent: **{'yes' if coherent else 'no'}**
- Valid pixels: **{valid_percentage:.2f}%** ({valid_count:,}/{total:,})
- Estimated horizontal FOV: **{fov_x:.3f}°**

### Specific concerns

{concern_md}

## Raw geometry checks

- Non-finite values inside valid regions: {valid_nonfinite}
- Masked-out non-finite depth pixels retained from MoGe: {report['non_finite_values']['masked_out_depth_non_finite']:,}
- Valid depth range: {valid_depth.min():.4f}–{valid_depth.max():.4f} m
- Depth p1 / p50 / p99: {np.percentile(valid_depth, 1):.4f} / {np.percentile(valid_depth, 50):.4f} / {np.percentile(valid_depth, 99):.4f} m
- Adjacent valid pairs over {discontinuity_threshold_m:.2f} m jump: {discontinuity_percentage:.3f}%

Preview clipping is visualization-only: depth is mapped with the turbo colormap between p2={previews['depth_preview']['clip_depth_m'][0]:.4f} m and p98={previews['depth_preview']['clip_depth_m'][1]:.4f} m. Raw geometry is unchanged.

## Planar-region diagnostics

| Region | Valid pixels | Normal-consistent pixels | Plane inliers | Plausible |
|---|---:|---:|---:|---|
{chr(10).join(region_lines)}

Pairwise fitted-plane normal angles: {', '.join(f'{key}={value:.2f}°' for key, value in angles.items())}.

The diagnostic uses fixed image-space regions, normal filtering, and RANSAC. It is not semantic segmentation, and foreground objects can lower plane-fit scores.

## Artifacts

- Colorized depth: `depth_color.png`
- Normal map: `normal_map.png`
- Validity mask: `validity_mask.png`
- Geometry-only point cloud: `point_cloud.ply`
- Source-colored point cloud: `point_cloud_colored.ply`
- Camera summary: `camera_summary.json`, `camera_summary.md`
"""
    (inspection_dir / "geometry_quality_report.md").write_text(report_md, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moge-dir", type=Path, default=DEFAULT_MOGE_DIR)
    parser.add_argument("--source-image", type=Path, default=DEFAULT_IMAGE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = generate(args.moge_dir, args.source_image)
    print(json.dumps(result["assessment"], indent=2))
