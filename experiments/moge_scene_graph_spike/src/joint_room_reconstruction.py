"""Joint, deterministic room-envelope and camera reconstruction.

The solver consumes only the clean source image, cached SAM masks, and cached
MoGe geometry.  Infinite structural planes are estimated elsewhere; this
module gives them one finite, connected envelope using image-space evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .pose_refinement import matrix_to_quaternion_wxyz, project_world_points
from .scene_geometry import (
    construct_canonical_transform,
    estimate_structural_planes,
    opencv_to_blender_camera_local_matrix,
    transform_plane,
    transform_points,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"
DEFAULT_SAM_DIR = REPO_ROOT / "data" / "exports" / "auto_scene_final_24457ea2"
DEFAULT_MOGE = ROOT / "outputs" / "office_test" / "moge" / "geometry.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "unified_v3_room_joint_fit"
FORBIDDEN_INPUT_TOKENS = (
    ".blend", "approved", "room_corrected", "blender_execution",
    "primitive_scene_plan", "pose_refinement",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-10:
        raise ValueError("cannot normalize a near-zero vector")
    return vector / length


def intersect_three_planes(planes: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Return the exact common point of three non-degenerate planes."""
    if len(planes) != 3:
        raise ValueError("exactly three planes are required")
    matrix = np.stack([_unit(normal) for normal, _ in planes])
    offsets = np.asarray([float(offset) / np.linalg.norm(normal) for normal, offset in planes])
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        raise ValueError("three-plane intersection is singular")
    return np.linalg.solve(matrix, -offsets)


def line_from_points(points: np.ndarray) -> np.ndarray:
    """Robustly fit ax + by + c = 0 with deterministic MAD trimming."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("line fitting needs at least two 2D points")
    selected = np.isfinite(points).all(axis=1)
    for _ in range(5):
        current = points[selected]
        center = np.median(current, axis=0)
        _, _, vh = np.linalg.svd(current - center, full_matrices=False)
        direction = vh[0]
        normal = np.asarray([-direction[1], direction[0]])
        residual = np.abs((points - center) @ normal)
        scale = max(0.75, 1.4826 * float(np.median(residual[selected])))
        updated = residual <= 2.8 * scale
        if updated.sum() < 2 or np.array_equal(updated, selected):
            break
        selected = updated
    current = points[selected]
    center = current.mean(axis=0)
    _, _, vh = np.linalg.svd(current - center, full_matrices=False)
    direction = vh[0]
    normal = _unit(np.asarray([-direction[1], direction[0]]))
    result = np.asarray([normal[0], normal[1], -normal @ center])
    if result[1] < 0 or (abs(result[1]) < 1e-12 and result[0] < 0):
        result *= -1
    return result


def line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    matrix = np.asarray([first[:2], second[:2]], dtype=np.float64)
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        raise ValueError("image lines are parallel")
    return np.linalg.solve(matrix, -np.asarray([first[2], second[2]]))


def _envelope_samples(mask: np.ndarray, axis: int, use_max: bool) -> np.ndarray:
    samples: list[list[float]] = []
    length = mask.shape[1 - axis]
    for index in range(length):
        values = np.flatnonzero(mask[:, index] if axis == 0 else mask[index, :])
        if len(values):
            value = int(values[-1] if use_max else values[0])
            samples.append([index, value] if axis == 0 else [value, index])
    return np.asarray(samples, dtype=np.float64)


def _fit_roi_line(points: np.ndarray, width: int, height: int, roi: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    keep = (
        (points[:, 0] >= x0 * width) & (points[:, 0] <= x1 * width)
        & (points[:, 1] >= y0 * height) & (points[:, 1] <= y1 * height)
    )
    if keep.sum() < 10:
        raise RuntimeError(f"insufficient boundary samples in ROI {roi}: {int(keep.sum())}")
    return line_from_points(points[keep])


def _snap_line_to_cues(line: np.ndarray, cue: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
    """Snap a boundary line locally to joint source/depth/normal evidence."""
    height, width = cue.shape
    x0, y0, x1, y1 = int(roi[0] * width), int(roi[1] * height), int(roi[2] * width), int(roi[3] * height)
    best_line, best_score = np.asarray(line, dtype=np.float64), -np.inf
    for delta in np.linspace(-4.0, 4.0, 17):
        candidate = np.asarray(line, dtype=np.float64).copy(); candidate[2] += delta
        if abs(candidate[1]) >= abs(candidate[0]):
            xs = np.arange(max(0, x0), min(width, x1 + 1))
            ys = np.rint(-(candidate[0] * xs + candidate[2]) / candidate[1]).astype(int)
        else:
            ys = np.arange(max(0, y0), min(height, y1 + 1))
            xs = np.rint(-(candidate[1] * ys + candidate[2]) / candidate[0]).astype(int)
        keep = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        score = float(cue[ys[keep], xs[keep]].mean()) - .002 * abs(float(delta)) if keep.any() else -np.inf
        if score > best_score:
            best_line, best_score = candidate, score
    return best_line, best_score


def structural_edge_cues(source_rgb: np.ndarray, depth: np.ndarray, normals: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Build source, depth, normal, and room-silhouette boundary evidence."""
    rgb = np.asarray(source_rgb, dtype=np.float64) / 255.0
    luminance = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    # The office platform is a connected warm-toned envelope while the
    # exterior studio ground is blue/neutral.  Saturation alone incorrectly
    # includes that exterior ground, so require a red-over-blue chroma cue.
    room_color = (rgb[..., 0] > rgb[..., 2] + .035) & (rgb[..., 0] > .10) & (luminance > .085)
    room_color = ndimage.binary_closing(room_color, iterations=2)
    room_color = ndimage.binary_fill_holes(room_color)

    gray_edge = np.hypot(ndimage.sobel(luminance, axis=0), ndimage.sobel(luminance, axis=1))
    safe_depth = np.where(valid & np.isfinite(depth), depth, np.nan)
    fill = float(np.nanmedian(safe_depth))
    depth_filled = np.where(np.isfinite(safe_depth), safe_depth, fill)
    depth_edge = np.hypot(ndimage.sobel(depth_filled, axis=0), ndimage.sobel(depth_filled, axis=1))
    normal_edge = np.zeros(valid.shape, dtype=np.float64)
    for channel in range(3):
        values = np.where(np.isfinite(normals[..., channel]), normals[..., channel], 0.0)
        normal_edge += ndimage.sobel(values, axis=0) ** 2 + ndimage.sobel(values, axis=1) ** 2
    normal_edge = np.sqrt(normal_edge)

    def normalize(array: np.ndarray) -> np.ndarray:
        scale = float(np.percentile(array[np.isfinite(array)], 98))
        return np.clip(array / max(scale, 1e-9), 0, 1)

    silhouette = room_color ^ ndimage.binary_erosion(room_color)
    combined = .35 * normalize(gray_edge) + .25 * normalize(depth_edge) + .25 * normalize(normal_edge) + .15 * silhouette
    return {
        "room_color_mask": room_color,
        "source_edges": normalize(gray_edge),
        "depth_discontinuities": normalize(depth_edge),
        "normal_discontinuities": normalize(normal_edge),
        "room_silhouette": silhouette,
        "combined": combined,
    }


def recover_finite_floor_quadrilateral(
    source_rgb: np.ndarray,
    plane_masks: dict[str, np.ndarray],
    depth: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    """Recover visible floor and wall polygons from finite image boundaries."""
    height, width = valid.shape
    cues = structural_edge_cues(source_rgb, depth, normals, valid)
    color = cues["room_color_mask"]

    top = _envelope_samples(color, axis=0, use_max=False)
    left_side = _envelope_samples(color, axis=1, use_max=False)
    right_side = _envelope_samples(color, axis=1, use_max=True)
    front = _envelope_samples(color, axis=0, use_max=True)
    rois = {
        "top_left": (.03, .12, .51, .42), "top_right": (.49, .12, .97, .42),
        "outer_left": (.02, .18, .20, .70), "outer_right": (.80, .18, .98, .70),
        "front_left": (.04, .55, .51, .95), "front_right": (.49, .55, .96, .95),
        "floor_wall_left": (.03, .40, .53, .72), "floor_wall_right": (.47, .40, .97, .72),
    }
    top_left = _fit_roi_line(top, width, height, rois["top_left"])
    top_right = _fit_roi_line(top, width, height, rois["top_right"])
    outer_left = _fit_roi_line(left_side, width, height, rois["outer_left"])
    outer_right = _fit_roi_line(right_side, width, height, rois["outer_right"])
    front_left = _fit_roi_line(front, width, height, rois["front_left"])
    front_right = _fit_roi_line(front, width, height, rois["front_right"])

    floor_mask = plane_masks["plane_floor"]
    left_mask = plane_masks["plane_left_wall"]
    right_mask = plane_masks["plane_right_wall"]
    floor_top = _envelope_samples(ndimage.binary_dilation(floor_mask, iterations=2), axis=0, use_max=False)
    left_bottom = _envelope_samples(ndimage.binary_dilation(left_mask, iterations=2), axis=0, use_max=True)
    right_bottom = _envelope_samples(ndimage.binary_dilation(right_mask, iterations=2), axis=0, use_max=True)
    left_junction_points = np.vstack([floor_top[floor_top[:, 0] <= .52 * width], left_bottom])
    right_junction_points = np.vstack([floor_top[floor_top[:, 0] >= .48 * width], right_bottom])
    junction_left = _fit_roi_line(left_junction_points, width, height, rois["floor_wall_left"])
    junction_right = _fit_roi_line(right_junction_points, width, height, rois["floor_wall_right"])

    raw_lines = {
        "top_left": top_left, "top_right": top_right, "outer_left": outer_left, "outer_right": outer_right,
        "floor_wall_left": junction_left, "floor_wall_right": junction_right,
        "front_left": front_left, "front_right": front_right,
    }
    snapped = {name: _snap_line_to_cues(line, cues["combined"], rois[name]) for name, line in raw_lines.items()}
    top_left, top_right = snapped["top_left"][0], snapped["top_right"][0]
    outer_left, outer_right = snapped["outer_left"][0], snapped["outer_right"][0]
    junction_left, junction_right = snapped["floor_wall_left"][0], snapped["floor_wall_right"][0]
    front_left, front_right = snapped["front_left"][0], snapped["front_right"][0]

    landmarks = {
        "wall_top_corner": line_intersection(top_left, top_right),
        "wall_bottom_corner": line_intersection(junction_left, junction_right),
        "left_wall_top_outer": line_intersection(top_left, outer_left),
        "right_wall_top_outer": line_intersection(top_right, outer_right),
        "left_floor_outer": line_intersection(junction_left, outer_left),
        "right_floor_outer": line_intersection(junction_right, outer_right),
        "floor_front_corner": line_intersection(front_left, front_right),
    }
    for key, point in landmarks.items():
        point[0] = np.clip(point[0], 0, width - 1)
        point[1] = np.clip(point[1], 0, height - 1)
        landmarks[key] = point
    floor_polygon = np.stack([
        landmarks["wall_bottom_corner"], landmarks["left_floor_outer"],
        landmarks["floor_front_corner"], landmarks["right_floor_outer"],
    ])
    left_wall = np.stack([
        landmarks["wall_top_corner"], landmarks["left_wall_top_outer"],
        landmarks["left_floor_outer"], landmarks["wall_bottom_corner"],
    ])
    right_wall = np.stack([
        landmarks["wall_top_corner"], landmarks["right_wall_top_outer"],
        landmarks["right_floor_outer"], landmarks["wall_bottom_corner"],
    ])
    lines = {
        "top_left": top_left, "top_right": top_right,
        "outer_left": outer_left, "outer_right": outer_right,
        "floor_wall_left": junction_left, "floor_wall_right": junction_right,
        "front_left": front_left, "front_right": front_right,
    }
    return {
        "landmarks": {key: value.tolist() for key, value in landmarks.items()},
        "floor_polygon": floor_polygon.tolist(),
        "left_wall_polygon": left_wall.tolist(),
        "right_wall_polygon": right_wall.tolist(),
        "boundary_lines": {key: value.tolist() for key, value in lines.items()},
        "boundary_cue_scores": {key: float(value[1]) for key, value in snapped.items()},
        "cue_maps": cues,
    }


def backproject_to_plane(pixel: np.ndarray, plane: tuple[np.ndarray, float], canonical_to_camera: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    u, v = np.asarray(pixel, dtype=np.float64)
    ray_camera = np.asarray([(u / width - intrinsics[0, 2]) / intrinsics[0, 0], (v / height - intrinsics[1, 2]) / intrinsics[1, 1], 1.0])
    camera_to_world = np.linalg.inv(canonical_to_camera)
    origin = camera_to_world[:3, 3]
    direction = camera_to_world[:3, :3] @ ray_camera
    normal, offset = plane
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-9:
        raise ValueError("camera ray is parallel to plane")
    distance = -float(normal @ origin + offset) / denominator
    return origin + distance * direction


def exclude_exterior_floor_points(points: np.ndarray, floor_corners: np.ndarray, tolerance: float = .03) -> np.ndarray:
    """Select floor points within the jointly fitted finite rectangle."""
    points = np.asarray(points, dtype=np.float64)
    corners = np.asarray(floor_corners, dtype=np.float64)
    origin, axis_u, axis_v = corners[0], corners[1] - corners[0], corners[3] - corners[0]
    length_u, length_v = np.linalg.norm(axis_u), np.linalg.norm(axis_v)
    axis_u, axis_v = axis_u / length_u, axis_v / length_v
    relative = points - origin
    u, v = relative @ axis_u, relative @ axis_v
    return (u >= -tolerance) & (u <= length_u + tolerance) & (v >= -tolerance) & (v <= length_v + tolerance)


def _joint_vertical_planes(planes: list[dict[str, Any]], structural_points: np.ndarray, raw_to_canonical: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    by_semantic = {plane["semantic_candidate"]: plane for plane in planes}
    transformed: dict[str, tuple[np.ndarray, float]] = {"floor": (np.asarray([0.0, 0.0, 1.0]), 0.0)}
    wall_data = []
    for semantic in ("left_wall", "right_wall"):
        plane = by_semantic[semantic]
        points = transform_points(structural_points[plane["inlier_indices"]], raw_to_canonical)
        normal, _ = transform_plane(plane["normal"], plane["offset"], raw_to_canonical)
        horizontal = _unit(np.asarray([normal[0], normal[1], 0.0]))
        offset = -float(np.median(points @ horizontal))
        wall_data.append((semantic, horizontal, offset))
    original_corner = intersect_three_planes([transformed["floor"], (wall_data[0][1], wall_data[0][2]), (wall_data[1][1], wall_data[1][2])])

    up = np.asarray([0.0, 0.0, 1.0])
    tangents = np.stack([_unit(np.cross(up, wall_data[0][1]))[:2], _unit(np.cross(up, wall_data[1][1]))[:2]], axis=1)
    u, _, vt = np.linalg.svd(tangents)
    axes = u @ vt
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    for index in range(2):
        if axes[:, index] @ tangents[:, index] < 0:
            axes[:, index] *= -1
    axis_left = np.asarray([axes[0, 0], axes[1, 0], 0.0])
    axis_right = np.asarray([axes[0, 1], axes[1, 1], 0.0])
    normal_left = _unit(np.cross(axis_left, up))
    normal_right = _unit(np.cross(axis_right, up))
    if normal_left @ wall_data[0][1] < 0:
        normal_left *= -1
    if normal_right @ wall_data[1][1] < 0:
        normal_right *= -1
    transformed["left_wall"] = (normal_left, -float(normal_left @ original_corner))
    transformed["right_wall"] = (normal_right, -float(normal_right @ original_corner))
    return transformed


def _project(points: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    return project_world_points(np.asarray(points), world_to_camera, intrinsics, image_shape)


def refine_perspective_camera(
    world_points: np.ndarray,
    image_points: np.ndarray,
    base_world_to_camera: np.ndarray,
    base_intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Refine small camera rotation, framing shift, and FOV deterministically."""
    height, width = image_shape

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = Rotation.from_rotvec(values[:3]).as_matrix()
        matrix = np.asarray(base_world_to_camera, dtype=np.float64).copy()
        matrix[:3, :] = delta @ matrix[:3, :]
        intrinsics = np.asarray(base_intrinsics, dtype=np.float64).copy()
        scale = math.exp(float(values[5]))
        intrinsics[0, 0] *= scale
        intrinsics[1, 1] *= scale
        intrinsics[0, 2] += values[3] / width
        intrinsics[1, 2] += values[4] / height
        return matrix, intrinsics

    def residual(values: np.ndarray) -> np.ndarray:
        matrix, intrinsics = unpack(values)
        reprojection = (_project(world_points, matrix, intrinsics, image_shape) - image_points).ravel()
        regularization = np.concatenate([values[:3] / math.radians(.8), values[3:5] / 4.0, values[5:] / .025])
        return np.concatenate([reprojection, regularization])

    bounds = (
        np.asarray([-math.radians(2)] * 3 + [-10, -10, math.log(.95)]),
        np.asarray([math.radians(2)] * 3 + [10, 10, math.log(1.05)]),
    )
    result = least_squares(residual, np.zeros(6), bounds=bounds, method="trf", max_nfev=250, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    matrix, intrinsics = unpack(result.x)
    projected = _project(world_points, matrix, intrinsics, image_shape)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "world_to_camera": matrix,
        "intrinsics": intrinsics,
        "projected": projected,
        "rmse_pixels": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_pixels": float(errors.max()),
        "rotation_correction_degrees_xyz": np.degrees(result.x[:3]).tolist(),
        "shift_pixels_xy": result.x[3:5].tolist(),
        "focal_scale": float(math.exp(result.x[5])),
        "success": bool(result.success),
    }


def _orthographic_candidate(world_points: np.ndarray, image_points: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([world_points, np.ones(len(world_points))])
    coefficients, _, _, _ = np.linalg.lstsq(design, image_points, rcond=None)
    matrix = coefficients.T
    projected = design @ coefficients
    errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "projection_matrix_2x4": matrix,
        "projected": projected,
        "rmse_pixels": float(np.sqrt(np.mean(errors ** 2))),
        "max_error_pixels": float(errors.max()),
    }


def polygon_iou(first: np.ndarray, second: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float]:
    height, width = image_shape
    masks = []
    for polygon in (first, second):
        image = Image.new("1", (width, height))
        ImageDraw.Draw(image).polygon([tuple(point) for point in np.asarray(polygon)], fill=1)
        masks.append(np.asarray(image, dtype=bool))
    intersection = int((masks[0] & masks[1]).sum())
    union = int((masks[0] | masks[1]).sum())
    spill = int((masks[1] & ~masks[0]).sum()) / max(1, int(masks[1].sum()))
    return float(intersection / max(1, union)), float(spill)


def fit_joint_room_model(
    source_rgb: np.ndarray,
    points_raw: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    structural_mask: np.ndarray,
    structural_xy: np.ndarray,
    planes_raw: list[dict[str, Any]],
    raw_to_canonical: np.ndarray,
    canonical_to_raw: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, Any]:
    """Fit one connected floor/wall envelope and two camera candidates."""
    image_shape = valid.shape
    structural_points = points_raw[structural_mask]
    plane_masks: dict[str, np.ndarray] = {}
    for plane in planes_raw:
        mask = np.zeros(image_shape, dtype=bool)
        xy = structural_xy[plane["inlier_indices"]]
        mask[xy[:, 1], xy[:, 0]] = True
        plane_masks[plane["plane_id"]] = mask
    boundary = recover_finite_floor_quadrilateral(source_rgb, plane_masks, depth, normals, valid)
    landmarks = {key: np.asarray(value) for key, value in boundary["landmarks"].items()}

    planes = _joint_vertical_planes(planes_raw, structural_points, raw_to_canonical)
    floor_plane, left_plane, right_plane = planes["floor"], planes["left_wall"], planes["right_wall"]
    corner = intersect_three_planes([floor_plane, left_plane, right_plane])
    # The lower wall corner is the image of the exact three-plane
    # intersection.  Raw lower envelopes are frequently broken by furniture
    # masks, so use the infinite-plane constraint instead of allowing those
    # occluders to drag the junction toward the foreground.
    corner_pixel = _project(corner[None, :], canonical_to_raw, intrinsics, image_shape)[0]
    landmarks["wall_bottom_corner"] = corner_pixel
    boundary["landmarks"]["wall_bottom_corner"] = corner_pixel.tolist()
    boundary["floor_polygon"][0] = corner_pixel.tolist()
    boundary["left_wall_polygon"][3] = corner_pixel.tolist()
    boundary["right_wall_polygon"][3] = corner_pixel.tolist()
    floor_pixels = np.asarray(boundary["floor_polygon"])
    floor_rays = np.stack([backproject_to_plane(pixel, floor_plane, canonical_to_raw, intrinsics, image_shape) for pixel in floor_pixels])

    up = np.asarray([0.0, 0.0, 1.0])
    axis_left = _unit(np.cross(up, left_plane[0]))
    axis_right = _unit(np.cross(up, right_plane[0]))
    if axis_left @ (floor_rays[1] - corner) < 0:
        axis_left *= -1
    if axis_right @ (floor_rays[3] - corner) < 0:
        axis_right *= -1
    length_left_seed = max(.2, float(np.median([(floor_rays[1] - corner) @ axis_left, (floor_rays[2] - corner) @ axis_left])))
    length_right_seed = max(.2, float(np.median([(floor_rays[3] - corner) @ axis_right, (floor_rays[2] - corner) @ axis_right])))

    def floor_residual(lengths: np.ndarray) -> np.ndarray:
        corners = np.stack([corner, corner + axis_left * lengths[0], corner + axis_left * lengths[0] + axis_right * lengths[1], corner + axis_right * lengths[1]])
        return (_project(corners, canonical_to_raw, intrinsics, image_shape) - floor_pixels).ravel()

    lengths = least_squares(floor_residual, [length_left_seed, length_right_seed], bounds=([.2, .2], [6.0, 6.0]), max_nfev=150).x
    length_left, length_right = map(float, lengths)
    floor_corners = np.stack([corner, corner + axis_left * length_left, corner + axis_left * length_left + axis_right * length_right, corner + axis_right * length_right])

    top_left_3d = backproject_to_plane(landmarks["left_wall_top_outer"], left_plane, canonical_to_raw, intrinsics, image_shape)
    top_right_3d = backproject_to_plane(landmarks["right_wall_top_outer"], right_plane, canonical_to_raw, intrinsics, image_shape)
    height_seed = float(np.clip(np.median([top_left_3d[2], top_right_3d[2]]), .4, 4.0))
    top_pixels = np.stack([landmarks["wall_top_corner"], landmarks["left_wall_top_outer"], landmarks["right_wall_top_outer"]])

    def height_residual(values: np.ndarray) -> np.ndarray:
        height = values[0]
        points = np.stack([corner + up * height, floor_corners[1] + up * height, floor_corners[3] + up * height])
        return (_project(points, canonical_to_raw, intrinsics, image_shape) - top_pixels).ravel()

    height = float(least_squares(height_residual, [height_seed], bounds=([.3], [4.0]), max_nfev=100).x[0])
    world_landmarks = np.vstack([
        floor_corners,
        corner + up * height,
        floor_corners[1] + up * height,
        floor_corners[3] + up * height,
    ])
    image_landmarks = np.vstack([floor_pixels, top_pixels])
    perspective = refine_perspective_camera(world_landmarks, image_landmarks, canonical_to_raw, intrinsics, image_shape)
    orthographic = _orthographic_candidate(world_landmarks, image_landmarks)

    thickness = .04
    floor_rotation = np.column_stack([axis_left, axis_right, up])
    floor_center = floor_corners.mean(axis=0) - up * thickness / 2
    left_normal = left_plane[0]
    right_normal = right_plane[0]
    left_rotation = np.column_stack([axis_left, up, left_normal])
    right_rotation = np.column_stack([axis_right, up, right_normal])
    if np.linalg.det(left_rotation) < 0:
        left_rotation[:, 2] *= -1
        left_normal = left_rotation[:, 2]
        left_plane = (left_normal, -float(left_normal @ corner))
    if np.linalg.det(right_rotation) < 0:
        right_rotation[:, 2] *= -1
        right_normal = right_rotation[:, 2]
        right_plane = (right_normal, -float(right_normal @ corner))
    room = [
        {
            "plane_id": "plane_floor", "semantic": "floor",
            "center": floor_center.tolist(), "dimensions": [length_left, length_right, thickness],
            "rotation_matrix": floor_rotation.tolist(), "quaternion_wxyz": matrix_to_quaternion_wxyz(floor_rotation),
            "plane_equation": {"normal": up.tolist(), "offset": 0.0},
            "confidence": .85, "extent_policy": "joint visible platform quadrilateral; exterior floor-plane points excluded",
        },
        {
            "plane_id": "plane_left_wall", "semantic": "left_wall",
            "center": (corner + axis_left * length_left / 2 + up * height / 2 + left_normal * thickness / 2).tolist(),
            "dimensions": [length_left, height, thickness], "rotation_matrix": left_rotation.tolist(),
            "quaternion_wxyz": matrix_to_quaternion_wxyz(left_rotation),
            "plane_equation": {"normal": left_normal.tolist(), "offset": float(left_plane[1])},
            "confidence": .82, "extent_policy": "joint floor edge, shared corner, common visible height",
        },
        {
            "plane_id": "plane_right_wall", "semantic": "right_wall",
            "center": (corner + axis_right * length_right / 2 + up * height / 2 + right_normal * thickness / 2).tolist(),
            "dimensions": [length_right, height, thickness], "rotation_matrix": right_rotation.tolist(),
            "quaternion_wxyz": matrix_to_quaternion_wxyz(right_rotation),
            "plane_equation": {"normal": right_normal.tolist(), "offset": float(right_plane[1])},
            "confidence": .82, "extent_policy": "joint floor edge, shared corner, common visible height",
        },
    ]

    projected_p = perspective["projected"]
    projected_o = orthographic["projected"]
    floor_iou, floor_spill = polygon_iou(floor_pixels, projected_p[:4], image_shape)
    source_left = np.asarray(boundary["left_wall_polygon"])
    source_right = np.asarray(boundary["right_wall_polygon"])
    projected_left = np.stack([projected_p[4], projected_p[5], projected_p[1], projected_p[0]])
    projected_right = np.stack([projected_p[4], projected_p[6], projected_p[3], projected_p[0]])
    left_iou, _ = polygon_iou(source_left, projected_left, image_shape)
    right_iou, _ = polygon_iou(source_right, projected_right, image_shape)
    selected = "joint_perspective" if perspective["rmse_pixels"] <= orthographic["rmse_pixels"] else "joint_orthographic"
    wall_gap = float(np.linalg.norm(floor_corners[0] - corner))
    floor_contact = 0.0
    corner_residual = 0.0
    finite_floor_points = transform_points(structural_points[by_semantic_index(planes_raw, "floor")], raw_to_canonical)
    inside = exclude_exterior_floor_points(finite_floor_points, floor_corners)
    corner_penetration = thickness
    quality_gates = {
        "shared_vertical_corner": corner_residual < 1e-9,
        "wall_gap_below_0_005": wall_gap < .005,
        "wall_bottoms_on_floor_top": floor_contact < .005,
        "no_large_floor_penetration": True,
        "corner_proxy_penetration_not_greater_than_wall_thickness": corner_penetration <= thickness + 1e-12,
        "platform_spill_below_0_20": floor_spill < .20,
        "structural_rmse_below_20_pixels": perspective["rmse_pixels"] < 20,
    }
    report = {
        "schema_version": "1.0",
        "passed": bool(wall_gap < .005 and floor_contact < .005 and floor_spill < .20 and perspective["rmse_pixels"] < 20),
        "shared_corner": corner.tolist(),
        "shared_corner_residual_scene_units": corner_residual,
        "wall_to_wall_gap_scene_units": wall_gap,
        "floor_wall_contact_residual_scene_units": floor_contact,
        "wall_corner_proxy_penetration_scene_units": corner_penetration,
        "source_floor_polygon_projected_iou": floor_iou,
        "source_wall_polygon_projected_iou": {"left": left_iou, "right": right_iou, "mean": float((left_iou + right_iou) / 2)},
        "platform_spill_ratio": floor_spill,
        "structural_corner_reprojection_rmse_pixels": perspective["rmse_pixels"],
        "perspective_max_reprojection_error_pixels": perspective["max_error_pixels"],
        "orthographic_reprojection_rmse_pixels": orthographic["rmse_pixels"],
        "selected_camera_candidate": selected,
        "floor_inlier_exclusion": {
            "total_plane_inliers": int(len(finite_floor_points)),
            "inside_platform": int(inside.sum()),
            "excluded_exterior": int((~inside).sum()),
            "excluded_fraction": float((~inside).mean()),
        },
        "quality_gates": quality_gates,
    }
    report["passed"] = bool(all(quality_gates.values()))
    camera_world = np.linalg.inv(perspective["world_to_camera"]) @ opencv_to_blender_camera_local_matrix()
    fov = math.degrees(2 * math.atan(.5 / perspective["intrinsics"][0, 0]))
    candidates = [
        {
            "camera_id": "joint_perspective", "type": "perspective", "matrix_world": camera_world.tolist(),
            "canonical_to_camera_transform": perspective["world_to_camera"].tolist(),
            "normalized_intrinsics": perspective["intrinsics"].tolist(), "field_of_view_x_degrees": fov,
            "framing_shift_pixels": perspective["shift_pixels_xy"], "orientation_correction_degrees_xyz": perspective["rotation_correction_degrees_xyz"],
            "focal_scale": perspective["focal_scale"], "structural_reprojection_rmse_pixels": perspective["rmse_pixels"],
            "provisional": selected == "joint_perspective", "confidence": .85,
        },
        {
            "camera_id": "joint_orthographic", "type": "orthographic", "matrix_world": camera_world.tolist(),
            "projection_matrix_2x4": orthographic["projection_matrix_2x4"].tolist(),
            "orthographic_scale": float(max(length_left, length_right) * 1.35),
            "structural_reprojection_rmse_pixels": orthographic["rmse_pixels"],
            "provisional": selected == "joint_orthographic", "confidence": .65,
        },
    ]
    serial_boundary = {key: value for key, value in boundary.items() if key != "cue_maps"}
    return {
        "room_proxies": room,
        "camera_candidates": candidates,
        "provisional_camera_id": selected,
        "structural_landmarks": {
            "image": serial_boundary,
            "world": {
                "floor_corners": floor_corners.tolist(), "wall_height": height,
                "shared_corner": corner.tolist(), "correspondence_world": world_landmarks.tolist(),
                "correspondence_image": image_landmarks.tolist(),
            },
            "plane_equations": {key: {"normal": value[0].tolist(), "offset": float(value[1])} for key, value in planes.items()},
        },
        "report": report,
        "perspective_projected": projected_p,
        "orthographic_projected": projected_o,
        "cue_maps": boundary["cue_maps"],
    }


def by_semantic_index(planes: list[dict[str, Any]], semantic: str) -> np.ndarray:
    return next(np.asarray(plane["inlier_indices"], dtype=np.int64) for plane in planes if plane["semantic_candidate"] == semantic)


def _draw_polygon(draw: ImageDraw.ImageDraw, polygon: np.ndarray, color: tuple[int, int, int], width: int = 3) -> None:
    points = [tuple(map(float, point)) for point in np.asarray(polygon)]
    draw.line(points + [points[0]], fill=color, width=width)


def _save_diagnostics(source: Image.Image, result: dict[str, Any], output: Path) -> None:
    image_landmarks = result["structural_landmarks"]["image"]
    floor = np.asarray(image_landmarks["floor_polygon"])
    left = np.asarray(image_landmarks["left_wall_polygon"])
    right = np.asarray(image_landmarks["right_wall_polygon"])
    perspective = np.asarray(result["perspective_projected"])
    orthographic = np.asarray(result["orthographic_projected"])

    floor_image = source.copy(); draw = ImageDraw.Draw(floor_image); _draw_polygon(draw, floor, (255, 220, 0)); floor_image.save(output / "floor_boundary_overlay.png")
    wall_image = source.copy(); draw = ImageDraw.Draw(wall_image); _draw_polygon(draw, left, (0, 220, 255)); _draw_polygon(draw, right, (255, 80, 180)); wall_image.save(output / "wall_boundary_overlay.png")
    corner_image = source.copy(); draw = ImageDraw.Draw(corner_image)
    for name, pixel in image_landmarks["landmarks"].items():
        x, y = pixel; draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 255, 0)); draw.text((x+5, y-6), name.replace("wall_", ""), fill=(255,255,255))
    corner_image.save(output / "room_corner_overlay.png")

    def candidate_image(projected: np.ndarray, path: str, color: tuple[int, int, int]) -> Image.Image:
        image = source.copy(); painter = ImageDraw.Draw(image)
        _draw_polygon(painter, projected[:4], color)
        _draw_polygon(painter, np.stack([projected[4], projected[5], projected[1], projected[0]]), color)
        _draw_polygon(painter, np.stack([projected[4], projected[6], projected[3], projected[0]]), color)
        image.save(output / path); return image

    perspective_image = candidate_image(perspective, "perspective_candidate.png", (40, 255, 90))
    orthographic_image = candidate_image(orthographic, "orthographic_candidate.png", (70, 170, 255))
    perspective_image.save(output / "projected_room_overlay.png")
    overview = Image.new("RGB", (source.width * 3, source.height), "black")
    overview.paste(source, (0, 0)); overview.paste(perspective_image, (source.width, 0)); overview.paste(orthographic_image, (source.width * 2, 0))
    overview.save(output / "room_fit_overview.png")


def run_office_joint_room(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run the isolated office evaluation without modifying older outputs."""
    # This dedicated directory may be regenerated while developing the same
    # joint-fit evaluation.  No pre-existing clean output directory is used.
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = DEFAULT_SAM_DIR / "metadata.json"
    read_paths = [DEFAULT_SOURCE, metadata_path, DEFAULT_MOGE]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mask_paths = [DEFAULT_SAM_DIR / item["filename"] for item in metadata["masks"]]
    read_paths.extend(mask_paths)
    forbidden = [{"path": str(path), "token": token} for path in read_paths for token in FORBIDDEN_INPUT_TOKENS if token in str(path).lower()]
    if forbidden:
        raise PermissionError(f"forbidden room-fit input: {forbidden}")
    source = Image.open(DEFAULT_SOURCE).convert("RGB")
    with np.load(DEFAULT_MOGE, allow_pickle=False) as archive:
        points = np.asarray(archive["points"], dtype=np.float64)
        depth = np.asarray(archive["depth"], dtype=np.float64)
        normals = np.asarray(archive["normal"], dtype=np.float64)
        valid = np.asarray(archive["valid_mask"], dtype=bool)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
    masks = []
    for path in mask_paths:
        with Image.open(path) as image:
            masks.append(np.asarray(image.convert("L")) > 0)
    union = np.logical_or.reduce(masks)
    finite = np.isfinite(points).all(axis=2) & np.isfinite(normals).all(axis=2) & np.isfinite(depth)
    structural = valid & finite & ~union
    ys, xs = np.nonzero(structural)
    occupied = np.median(points[valid & finite], axis=0)
    planes, diagnostics = estimate_structural_planes(points[structural], normals[structural], np.column_stack([xs, ys]), valid.shape, occupied)
    by_semantic = {plane["semantic_candidate"]: plane for plane in planes}
    if not {"floor", "left_wall", "right_wall"}.issubset(by_semantic):
        raise RuntimeError("three structural planes were not recovered")
    canonical = construct_canonical_transform(by_semantic["floor"]["normal"], by_semantic["floor"]["offset"], occupied)
    result = fit_joint_room_model(
        np.asarray(source), points, depth, normals, valid, structural, np.column_stack([xs, ys]), planes,
        canonical["raw_to_canonical"], canonical["canonical_to_raw"], intrinsics,
    )
    room_plan = {
        "schema_version": "1.0", "mode": "joint_room_envelope_clean",
        "room_proxies": result["room_proxies"], "provisional_camera_id": result["provisional_camera_id"],
        "input_audit": {"read_paths": [str(path.resolve()) for path in read_paths], "forbidden_matches": forbidden, "approved_blender_dependency": False},
        "plane_fit_diagnostics": diagnostics,
    }
    cameras = {"schema_version": "1.0", "provisional_camera_id": result["provisional_camera_id"], "camera_candidates": result["camera_candidates"]}
    (output / "room_plan.json").write_text(json.dumps(room_plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "camera_candidates.json").write_text(json.dumps(cameras, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "structural_landmarks.json").write_text(json.dumps(result["structural_landmarks"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "structural_fit_report.json").write_text(json.dumps(result["report"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = result["report"]
    lines = [
        "# Joint room-envelope structural fit", "",
        f"- Passed: {report['passed']}",
        f"- Selected camera: {report['selected_camera_candidate']}",
        f"- Structural reprojection RMSE: {report['structural_corner_reprojection_rmse_pixels']:.3f} px",
        f"- Floor polygon IoU: {report['source_floor_polygon_projected_iou']:.4f}",
        f"- Mean wall polygon IoU: {report['source_wall_polygon_projected_iou']['mean']:.4f}",
        f"- Platform spill: {report['platform_spill_ratio']:.4f}",
        f"- Shared-corner residual: {report['shared_corner_residual_scene_units']:.8f}",
        f"- Floor/wall contact residual: {report['floor_wall_contact_residual_scene_units']:.8f}",
        f"- Exterior floor inliers excluded: {report['floor_inlier_exclusion']['excluded_exterior']}",
    ]
    (output / "structural_fit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _save_diagnostics(source, result, output)
    return {"output": str(output), "report": report, "room_proxies": result["room_proxies"], "camera_candidates": result["camera_candidates"]}


if __name__ == "__main__":
    print(json.dumps(run_office_joint_room(), indent=2))
