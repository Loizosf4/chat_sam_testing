"""Build deterministic room-plane, coordinate, support, and spatial evidence."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src.extract_object_geometry import _write_ply
from src.scene_geometry import (
    construct_canonical_transform,
    detect_upward_support_patch,
    estimate_structural_planes,
    opencv_to_blender_camera_local_matrix,
    plane_intersection,
    support_distance_evidence,
    transform_normals,
    transform_plane,
    transform_points,
    unknown_support_candidate,
    validate_relationship_targets,
    wall_attachment_evidence,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "inputs" / "office_test" / "manifest.json"
DEFAULT_MOGE_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "moge"
DEFAULT_GEOMETRY_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry"
PLANE_COLORS = {
    "floor": np.asarray([45, 120, 255], dtype=np.uint8),
    "left_wall": np.asarray([255, 80, 70], dtype=np.uint8),
    "right_wall": np.asarray([60, 210, 100], dtype=np.uint8),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) == 255


def _safe_record(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _relationship(
    subject: str,
    predicate: str,
    target: str | None,
    confidence: float,
    evidence: dict[str, Any],
    thresholds: dict[str, Any],
    sources: list[str],
    *,
    contradictions: list[str] | None = None,
    uncertainty: str = "",
) -> dict[str, Any]:
    return {
        "subject_object_id": subject,
        "predicate": predicate,
        "target_id": target,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "evidence": evidence,
        "thresholds_used": thresholds,
        "evidence_source": sources,
        "contradictions": contradictions or [],
        "uncertainty": uncertainty,
        "requires_vlm_review": True,
        "requires_user_review": bool(confidence < 0.6 or contradictions or uncertainty),
    }


def _serialize_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in patch.items()
        if key not in {"point_selection_mask", "patch_candidate_inlier_mask", "image_xy", "canonical_points"}
    }


def _adjacency_and_occlusion(
    object_a: dict[str, Any],
    object_b: dict[str, Any],
    masks: dict[str, np.ndarray],
    depth: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    id_a, id_b = object_a["object_id"], object_b["object_id"]
    mask_a, mask_b = masks[id_a], masks[id_b]
    overlap = int((mask_a & mask_b).sum())
    adjacency = float(ndimage.distance_transform_edt(~mask_a)[mask_b].min()) if mask_b.any() else math.inf
    dilation_radius = 4
    near_a = mask_a & ndimage.binary_dilation(mask_b, iterations=dilation_radius) & valid
    near_b = mask_b & ndimage.binary_dilation(mask_a, iterations=dilation_radius) & valid
    median_a = float(np.median(depth[near_a])) if near_a.any() else None
    median_b = float(np.median(depth[near_b])) if near_b.any() else None
    evidence = {
        "mask_overlap_pixels": overlap,
        "minimum_mask_distance_pixels": adjacency,
        "boundary_band_radius_pixels": dilation_radius,
        "subject_boundary_depth_median": median_a,
        "target_boundary_depth_median": median_b,
        "actual_overlap": overlap > 0,
        "image_space_adjacent": adjacency <= dilation_radius,
    }
    relationships: list[dict[str, Any]] = []
    if adjacency <= 3.0 and median_a is not None and median_b is not None:
        delta = median_b - median_a
        if abs(delta) >= 0.03:
            front, rear = (id_a, id_b) if delta > 0 else (id_b, id_a)
            confidence = min(0.95, 0.45 + abs(delta) / 0.35)
            common = {
                "boundary_depth_delta": abs(delta),
                "minimum_mask_distance_pixels": adjacency,
                "actual_mask_overlap_pixels": overlap,
                "interpretation": "image-space boundary depth only; hidden volume is not inferred",
            }
            relationships.append(
                _relationship(front, "image_occludes", rear, confidence, common, {"maximum_adjacency_pixels": 3, "minimum_depth_delta": 0.03}, ["mask_adjacency", "local_moge_depth"])
            )
            relationships.append(
                _relationship(rear, "image_occluded_by", front, confidence, common, {"maximum_adjacency_pixels": 3, "minimum_depth_delta": 0.03}, ["mask_adjacency", "local_moge_depth"])
            )
    return evidence, relationships


def _pairwise_spatial_evidence(
    objects: list[dict[str, Any]],
    object_data: dict[str, dict[str, Any]],
    masks: dict[str, np.ndarray],
    depth: np.ndarray,
    valid: np.ndarray,
    image_shape: tuple[int, int],
    raw_to_normalized: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = image_shape
    pair_records: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for index, a in enumerate(objects):
        for b in objects[index + 1 :]:
            id_a, id_b = a["object_id"], b["object_id"]
            pa, pb = object_data[id_a]["canonical_points"], object_data[id_b]["canonical_points"]
            centroid_a, centroid_b = np.median(pa, axis=0), np.median(pb, axis=0)
            raw_a, raw_b = object_data[id_a]["raw_points"], object_data[id_b]["raw_points"]
            normalized_a = transform_points(np.median(raw_a, axis=0, keepdims=True), raw_to_normalized)[0]
            normalized_b = transform_points(np.median(raw_b, axis=0, keepdims=True), raw_to_normalized)[0]
            normalized_distance = float(np.linalg.norm(normalized_b - normalized_a))
            image_a = np.asarray(a["mask_centroid_2d_xy"])
            image_b = np.asarray(b["mask_centroid_2d_xy"])
            image_delta = image_b - image_a
            canonical_delta = centroid_b - centroid_a
            depth_a = float(np.median(raw_a[:, 2]))
            depth_b = float(np.median(raw_b[:, 2]))
            adjacency, occlusion = _adjacency_and_occlusion(a, b, masks, depth, valid)
            relationships.extend(occlusion)
            pair_record = {
                    "object_a": id_a,
                    "object_b": id_b,
                    "normalized_centroid_distance": normalized_distance,
                    "canonical_centroid_delta_b_minus_a": canonical_delta.tolist(),
                    "image_centroid_delta_b_minus_a": image_delta.tolist(),
                    "raw_camera_depth_medians": {id_a: depth_a, id_b: depth_b},
                    "image_adjacency_and_occlusion": adjacency,
                }
            semantic_pair = {a["semantic_label"], b["semantic_label"]}
            if semantic_pair == {"desk", "desk_chair"}:
                pair_record["visibility_note"] = (
                    "The chair mask contains only the visible backrest. It is 7.21 pixels from the desk mask in this fixture; "
                    "physical chair support and desk/chair occlusion cannot be confirmed from these masks."
                )
            pair_records.append(pair_record)

            dx = float(image_delta[0] / width)
            if abs(dx) >= 0.04:
                left, right = (id_a, id_b) if dx > 0 else (id_b, id_a)
                confidence = min(0.95, abs(dx) / 0.35)
                ev = {"normalized_image_centroid_delta_x": abs(dx), "canonical_delta_x": abs(float(canonical_delta[0]))}
                relationships.append(_relationship(left, "left_of", right, confidence, ev, {"minimum_normalized_image_delta": 0.04}, ["2d_mask_centroid", "canonical_centroid"] ))
                relationships.append(_relationship(right, "right_of", left, confidence, ev, {"minimum_normalized_image_delta": 0.04}, ["2d_mask_centroid", "canonical_centroid"] ))

            dz = float(canonical_delta[2])
            if abs(dz) >= 0.06:
                below, above = (id_a, id_b) if dz > 0 else (id_b, id_a)
                confidence = min(0.95, abs(dz) / 0.6)
                ev = {"canonical_height_difference": abs(dz)}
                relationships.append(_relationship(above, "above", below, confidence, ev, {"minimum_height_difference": 0.06}, ["canonical_visible_centroid"] ))
                relationships.append(_relationship(below, "below", above, confidence, ev, {"minimum_height_difference": 0.06}, ["canonical_visible_centroid"] ))

            depth_delta = depth_b - depth_a
            if abs(depth_delta) >= 0.10:
                front, behind = (id_a, id_b) if depth_delta > 0 else (id_b, id_a)
                confidence = min(0.95, abs(depth_delta) / 1.0)
                ev = {"raw_camera_median_depth_difference": abs(depth_delta)}
                relationships.append(_relationship(front, "in_front_of", behind, confidence, ev, {"minimum_depth_difference": 0.10}, ["raw_moge_camera_depth"] ))
                relationships.append(_relationship(behind, "behind", front, confidence, ev, {"minimum_depth_difference": 0.10}, ["raw_moge_camera_depth"] ))

            near_confidence = max(0.0, 1.0 - normalized_distance / 0.35)
            if adjacency["minimum_mask_distance_pixels"] <= 10:
                near_confidence = max(near_confidence, 1.0 - adjacency["minimum_mask_distance_pixels"] / 12.0)
            if near_confidence >= 0.25:
                ev = {"normalized_centroid_distance": normalized_distance, "minimum_mask_distance_pixels": adjacency["minimum_mask_distance_pixels"]}
                relationships.append(_relationship(id_a, "near", id_b, near_confidence, ev, {"normalized_distance_scale": 0.35, "image_distance_scale_pixels": 12}, ["scene_normalized_centroids", "mask_adjacency"] ))
                relationships.append(_relationship(id_b, "near", id_a, near_confidence, ev, {"normalized_distance_scale": 0.35, "image_distance_scale_pixels": 12}, ["scene_normalized_centroids", "mask_adjacency"] ))
    return pair_records, relationships


def _overlay(source: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]], path: Path) -> None:
    output = source.astype(np.float32).copy()
    for mask, color in masks:
        color_array = np.asarray(color, dtype=np.float32)
        output[mask] = 0.35 * output[mask] + 0.65 * color_array
    Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).save(path)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    plane_lines = "\n".join(
        f"| {p['semantic_candidate']} | {p['confidence']:.3f} | {p['inlier_count']:,} | {p['residual_statistics']['median']:.4f} | {p['residual_statistics']['p95']:.4f} |"
        for p in result["structural_planes"]
    )
    support_lines = "\n".join(
        f"| {r['subject_object_id'][:8]} | {r['predicate']} | {(r['target_id'] or 'null')[:18]} | {r['confidence']:.3f} | {r['uncertainty'] or '-'} |"
        for r in result["relationship_candidates"]
        if r["predicate"] in {"supported_by", "attached_to", "unknown_support", "image_occludes", "image_occluded_by"}
    )
    warnings = "\n".join(f"- {warning}" for warning in result["warnings"])
    text = f"""# Deterministic scene evidence

This is evidence, not a final scene graph. It contains no inferred hidden geometry.

## Structural planes

| Plane | Confidence | Inliers | Median residual | P95 residual |
|---|---:|---:|---:|---:|
{plane_lines}

## Canonical coordinates

- Right-handed: {result['coordinate_systems']['canonical_scene_world']['right_handed']}
- Canonical +Z: floor normal toward occupied room volume
- Raw/canonical round-trip maximum error: {result['quality_checks']['transform_round_trip_max_error']:.3e}
- Blender camera-local conversion: OpenCV `(X right, Y down, Z forward)` → Blender `(X right, Y up, -Z forward)`

## Support, attachment, and occlusion candidates

| Subject | Predicate | Target | Confidence | Uncertainty |
|---|---|---|---:|---|
{support_lines}

## Warnings

{warnings}

## Readiness

**{result['next_phase_readiness']['status']}** — {result['next_phase_readiness']['explanation']}
"""
    path.write_text(text, encoding="utf-8")


def build(manifest_path: Path, moge_dir: Path, geometry_dir: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    object_geometry = _load_json(geometry_dir / "object_geometry.json")
    moge_metadata = _load_json(moge_dir / "metadata.json")
    with np.load(moge_dir / "geometry.npz", allow_pickle=False) as archive:
        points, depth = archive["points"], archive["depth"]
        normals, valid = archive["normal"], archive["valid_mask"].astype(bool)
        intrinsics = archive["intrinsics"]
    with Image.open(manifest_path.parent / manifest["image"]["filename"]) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = valid.shape
    object_ids = {obj["object_id"] for obj in manifest["objects"]}
    masks = {obj["object_id"]: _mask(manifest_path.parent / obj["mask_filename"]) for obj in manifest["objects"]}
    union_mask = np.logical_or.reduce(list(masks.values()))
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & np.isfinite(normals).all(axis=-1)
    structural_mask = valid & finite & ~union_mask
    ys, xs = np.nonzero(structural_mask)
    structural_xy = np.column_stack([xs, ys])
    occupied_point = np.median(points[valid & finite], axis=0)
    planes_internal, plane_diagnostics = estimate_structural_planes(
        points[structural_mask], normals[structural_mask], structural_xy, (height, width), occupied_point
    )
    by_semantic = {plane["semantic_candidate"]: plane for plane in planes_internal}
    required_planes = {"floor", "left_wall", "right_wall"}
    if not required_planes.issubset(by_semantic):
        raise RuntimeError(f"structural plane preflight failed; found {sorted(by_semantic)}")

    floor = by_semantic["floor"]
    canonical = construct_canonical_transform(floor["normal"], floor["offset"], occupied_point)
    raw_to_canonical = canonical["raw_to_canonical"]
    canonical_to_raw = canonical["canonical_to_raw"]
    raw_to_normalized = np.asarray(object_geometry["raw_to_normalized_transformation"]["raw_to_normalized_matrix_4x4"], dtype=np.float64)
    normalized_to_raw = np.asarray(object_geometry["raw_to_normalized_transformation"]["normalized_to_raw_matrix_4x4"], dtype=np.float64)
    cv_to_blender = opencv_to_blender_camera_local_matrix()

    sample_points = points[valid & finite][:: max(1, int(valid.sum() / 10000))]
    round_trip = transform_points(transform_points(sample_points, raw_to_canonical), canonical_to_raw)
    round_trip_error = float(np.max(np.linalg.norm(round_trip - sample_points, axis=1)))
    canonical_sample = transform_points(sample_points, raw_to_canonical)
    normalized_sample = transform_points(sample_points, raw_to_normalized)
    raw_dist = np.linalg.norm(sample_points[1:] - sample_points[:-1], axis=1)
    canonical_dist = np.linalg.norm(canonical_sample[1:] - canonical_sample[:-1], axis=1)
    normalized_dist = np.linalg.norm(normalized_sample[1:] - normalized_sample[:-1], axis=1)
    distance_error = float(np.max(np.abs(raw_dist - canonical_dist)))
    normalization_scale = float(object_geometry["raw_to_normalized_transformation"]["scale_factor"])
    normalized_distance_scale_error = float(np.max(np.abs(normalized_dist - raw_dist * normalization_scale)))

    plane_image_masks: dict[str, np.ndarray] = {}
    structural_planes: list[dict[str, Any]] = []
    for plane in planes_internal:
        plane_mask = np.zeros((height, width), dtype=bool)
        selected_xy = structural_xy[plane["inlier_indices"]]
        plane_mask[selected_xy[:, 1], selected_xy[:, 0]] = True
        plane_image_masks[plane["semantic_candidate"]] = plane_mask
        normalized_n, normalized_d = transform_plane(plane["normal"], plane["offset"], raw_to_normalized)
        canonical_n, canonical_d = transform_plane(plane["normal"], plane["offset"], raw_to_canonical)
        structural_planes.append(
            {
                "plane_id": plane["plane_id"],
                "semantic_candidate": plane["semantic_candidate"],
                "raw_moge_plane_equation": {"normal": plane["normal"].tolist(), "offset": plane["offset"]},
                "scene_normalized_plane_equation": {"normal": normalized_n.tolist(), "offset": normalized_d},
                "canonical_plane_equation": {"normal": canonical_n.tolist(), "offset": canonical_d},
                "oriented_unit_normal_raw_moge": plane["normal"].tolist(),
                "inlier_count": plane["inlier_count"],
                "candidate_cluster_point_count": plane["cluster_point_count"],
                "inlier_ratio": plane["inlier_ratio"],
                "residual_statistics": plane["residual_statistics"],
                "image_extent": {"bbox_xyxy": plane["image_bbox_xyxy"], "bbox_area_ratio": plane["image_bbox_area_ratio"], "coverage_ratio": plane["image_coverage_ratio"]},
                "extent_3d_raw_moge": plane["extent_3d"],
                "confidence": plane["confidence"],
                "classification_evidence": plane["classification_evidence"],
                "warnings": plane["warnings"],
            }
        )

    plane_intersections: list[dict[str, Any]] = []
    for name, a, b in [
        ("wall_wall", "left_wall", "right_wall"),
        ("left_wall_floor", "left_wall", "floor"),
        ("right_wall_floor", "right_wall", "floor"),
    ]:
        pa, pb = by_semantic[a], by_semantic[b]
        raw_line = plane_intersection((pa["normal"], pa["offset"]), (pb["normal"], pb["offset"]))
        record: dict[str, Any] = {
            "intersection_id": f"intersection_{name}", "plane_ids": [pa["plane_id"], pb["plane_id"]],
            "stable": raw_line["stable"], "condition_sine": raw_line["condition_sine"], "angle_degrees": raw_line.get("angle_degrees"),
        }
        if raw_line["stable"]:
            canonical_point = transform_points(raw_line["point"][None, :], raw_to_canonical)[0]
            canonical_direction = raw_line["direction"] @ raw_to_canonical[:3, :3].T
            record.update(
                {
                    "raw_moge": {"point": raw_line["point"].tolist(), "direction": raw_line["direction"].tolist()},
                    "canonical_scene_world": {"point": canonical_point.tolist(), "direction": canonical_direction.tolist()},
                    "confidence": float(min(pa["confidence"], pb["confidence"]) * raw_line["condition_sine"]),
                }
            )
        plane_intersections.append(record)

    geometry_records = {obj["object_id"]: obj for obj in object_geometry["objects"]}
    manifest_records = {obj["object_id"]: obj for obj in manifest["objects"]}
    object_data: dict[str, dict[str, Any]] = {}
    support_patches: list[dict[str, Any]] = []
    patch_internal: dict[str, dict[str, Any]] = {}
    for manifest_object in manifest["objects"]:
        object_id = manifest_object["object_id"]
        diagnostic_dir = geometry_dir / "diagnostics" / object_id
        with np.load(diagnostic_dir / "filtered_visible_geometry.npz", allow_pickle=False) as archive:
            raw_object_points = archive["points"].astype(np.float64)
            raw_object_normals = archive["normal"].astype(np.float64)
            image_xy = archive["image_xy"].astype(np.int32)
        canonical_points = transform_points(raw_object_points, raw_to_canonical)
        canonical_normals = transform_normals(raw_object_normals, raw_to_canonical)
        patch = detect_upward_support_patch(canonical_points, canonical_normals, image_xy)
        if "point_selection_mask" in patch:
            candidate_indices = np.flatnonzero(patch["point_selection_mask"])
            final_indices = candidate_indices[patch["patch_candidate_inlier_mask"]]
            patch["image_xy"] = image_xy[final_indices]
            patch["canonical_points"] = canonical_points[final_indices]
        patch_internal[object_id] = patch
        support_patches.append(
            {
                "object_id": object_id,
                "semantic_label": manifest_records[object_id]["semantic_label"],
                "semantic_role_evidence": manifest_records[object_id].get("selection_role", []),
                "visible_support_patch": _serialize_patch(patch),
                "warning": None if patch.get("detected") else "no reliable upward-facing support patch detected",
            }
        )
        object_data[object_id] = {
            "raw_points": raw_object_points, "raw_normals": raw_object_normals, "image_xy": image_xy,
            "canonical_points": canonical_points, "canonical_normals": canonical_normals,
        }

    relationships: list[dict[str, Any]] = []
    support_quality: dict[str, Any] = {}
    floor_id = by_semantic["floor"]["plane_id"]
    assigned_support: set[str] = set()
    for obj in object_geometry["objects"]:
        object_id = obj["object_id"]
        evidence = support_distance_evidence(object_data[object_id]["canonical_points"], np.asarray([0.0, 0.0, 1.0]), 0.0, None, distance_threshold=0.10)
        support_quality[object_id] = {"floor": evidence}
        partial = "partially_occluded" in manifest_records[object_id].get("selection_role", [])
        if partial:
            relationships.append(
                unknown_support_candidate(
                    object_id,
                    "visible mask contains the occluded chair backrest but no visible floor-contact/base geometry",
                    {"floor_distance_evidence": evidence, "visible_components": obj["connected_component_count"]},
                )
            )
            assigned_support.add(object_id)
        elif evidence["confidence"] >= 0.30:
            uncertainty = "" if evidence["supported"] else "borderline visible floor-contact evidence; retained as a candidate, not an assertion"
            relationships.append(
                _relationship(
                    object_id, "supported_by", floor_id, evidence["confidence"], evidence,
                    {"distance_threshold": 0.10, "candidate_confidence_minimum": 0.30, "assertion_confidence_minimum": 0.45},
                    ["canonical_lower_visible_points", "structural_floor_plane", "semantic_role_reported_separately"], uncertainty=uncertainty,
                )
            )
            assigned_support.add(object_id)

    desk_id = next(obj["object_id"] for obj in manifest["objects"] if obj["semantic_label"] == "desk")
    box_id = next(obj["object_id"] for obj in manifest["objects"] if obj["semantic_label"] == "desktop_box")
    desk_patch = patch_internal[desk_id]
    if desk_patch.get("detected"):
        box_desk = support_distance_evidence(
            object_data[box_id]["canonical_points"], np.asarray(desk_patch["plane_canonical"]["normal"]), float(desk_patch["plane_canonical"]["offset"]),
            desk_patch.get("boundary_canonical_xy"), distance_threshold=0.10,
        )
        support_quality[box_id]["desk_tabletop"] = box_desk
        if box_desk["confidence"] >= 0.30:
            relationships.append(
                _relationship(
                    box_id, "supported_by", desk_id, box_desk["confidence"],
                    {**box_desk, "desktop_box_depth_policy": "lower local visible points only; raw full-Z extent excluded"},
                    {"distance_threshold": 0.10, "minimum_horizontal_overlap": 0.10},
                    ["desktop_box_lower_visible_points", "desk_tabletop_patch", "local_plane_distance", "horizontal_patch_overlap"],
                    contradictions=[] if box_desk["supported"] else ["expected desk support was not fully confirmed"],
                    uncertainty="MoGe depth spread across the desktop box is excessive; full raw Z extent is not used",
                )
            )
            assigned_support.add(box_id)

    wall_light_id = next(obj["object_id"] for obj in manifest["objects"] if obj["semantic_label"] == "wall_light_fixture")
    wall_evidence: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for semantic in ["left_wall", "right_wall"]:
        plane = by_semantic[semantic]
        evidence = wall_attachment_evidence(object_data[wall_light_id]["raw_points"], plane["normal"], plane["offset"], distance_threshold=0.12)
        wall_evidence.append((evidence["confidence"], plane, evidence))
    _, closest_wall, attachment = max(wall_evidence, key=lambda item: item[0])
    support_quality[wall_light_id]["wall_attachment"] = attachment
    if attachment["confidence"] >= 0.30:
        relationships.append(
            _relationship(
                wall_light_id, "attached_to", closest_wall["plane_id"], attachment["confidence"], attachment,
                {"distance_threshold": 0.12, "candidate_confidence_minimum": 0.30},
                ["visible_points_to_structural_wall_distance", "structural_plane_confidence", "semantic_label_reported_separately"],
                uncertainty="attachment is visible-surface proximity evidence; hidden fasteners are not inferred",
            )
        )
        assigned_support.add(wall_light_id)

    for object_id in sorted(object_ids - assigned_support):
        relationships.append(unknown_support_candidate(object_id, "no support or attachment candidate met the geometric evidence threshold", support_quality[object_id]))

    pairwise, pair_relationships = _pairwise_spatial_evidence(
        object_geometry["objects"], object_data, masks, depth, valid, (height, width), raw_to_normalized
    )
    relationships.extend(pair_relationships)
    plane_ids = {plane["plane_id"] for plane in planes_internal}
    validate_relationship_targets(relationships, object_ids, plane_ids)

    diagnostics_dir = geometry_dir / "scene_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plane_overlay_items = [(plane_image_masks[name], tuple(PLANE_COLORS[name])) for name in ["floor", "left_wall", "right_wall"]]
    _overlay(source, plane_overlay_items, diagnostics_dir / "structural_plane_segmentation.png")
    _overlay(source, [(plane_image_masks["floor"], tuple(PLANE_COLORS["floor"]))], diagnostics_dir / "floor_inliers.png")
    _overlay(source, [(plane_image_masks["left_wall"], tuple(PLANE_COLORS["left_wall"]))], diagnostics_dir / "left_wall_inliers.png")
    _overlay(source, [(plane_image_masks["right_wall"], tuple(PLANE_COLORS["right_wall"]))], diagnostics_dir / "right_wall_inliers.png")
    all_plane_mask = np.logical_or.reduce(list(plane_image_masks.values()))
    rejected_structural = structural_mask & ~all_plane_mask
    _overlay(source, [(rejected_structural, (255, 40, 180)), (union_mask, (90, 90, 90))], diagnostics_dir / "rejected_plane_outliers.png")

    scene_valid = valid & finite
    scene_y, scene_x = np.nonzero(scene_valid)
    scene_points = points[scene_valid]
    scene_colors = source[scene_valid].copy()
    plane_membership = {name: plane_image_masks[name][scene_valid] for name in plane_image_masks}
    for name, member in plane_membership.items():
        scene_colors[member] = PLANE_COLORS[name]
    step = max(1, len(scene_points) // 70000)
    _write_ply(diagnostics_dir / "source_colored_structural_planes.ply", scene_points[::step], scene_colors[::step])
    canonical_scene = transform_points(scene_points, raw_to_canonical)
    _write_ply(diagnostics_dir / "canonical_scene_point_cloud.ply", canonical_scene[::step], source[scene_valid][::step])
    axis_length = float(np.percentile(np.linalg.norm(canonical_scene[:, :2], axis=1), 80) * 0.35)
    axis_points = np.asarray([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]], dtype=np.float32)
    axis_edges = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    cloud_step = max(1, len(canonical_scene) // 25000)
    gray = np.tile(np.asarray([145, 145, 145], dtype=np.uint8), (len(canonical_scene[::cloud_step]), 1))
    _write_ply(
        diagnostics_dir / "canonical_axes.ply", canonical_scene[::cloud_step], gray,
        marker_points=axis_points, marker_color=(255, 220, 20), edges=axis_edges,
    )

    desk_patch_mask = np.zeros((height, width), dtype=bool)
    if desk_patch.get("detected"):
        desk_patch_xy = desk_patch["image_xy"]
        desk_patch_mask[desk_patch_xy[:, 1], desk_patch_xy[:, 0]] = True
    _overlay(source, [(desk_patch_mask, (40, 255, 100))], diagnostics_dir / "desk_tabletop_support_patch.png")
    box_lower_mask = np.zeros((height, width), dtype=bool)
    box_points = object_data[box_id]["canonical_points"]
    lower = box_points[:, 2] <= np.quantile(box_points[:, 2], 0.08)
    box_xy = object_data[box_id]["image_xy"][lower]
    box_lower_mask[box_xy[:, 1], box_xy[:, 0]] = True
    _overlay(source, [(desk_patch_mask, (40, 220, 80)), (box_lower_mask, (255, 40, 220))], diagnostics_dir / "desktop_box_to_desk_contact.png")

    floor_contact_items: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    palette = [(255, 80, 40), (255, 210, 40), (40, 240, 220), (180, 80, 255), (80, 220, 70), (255, 80, 170)]
    for index, obj in enumerate(object_geometry["objects"]):
        object_id = obj["object_id"]
        if support_quality[object_id]["floor"]["confidence"] < 0.25:
            continue
        canonical_points = object_data[object_id]["canonical_points"]
        lower_selection = canonical_points[:, 2] <= np.quantile(canonical_points[:, 2], 0.08)
        xy = object_data[object_id]["image_xy"][lower_selection]
        contact_mask = np.zeros((height, width), dtype=bool)
        contact_mask[xy[:, 1], xy[:, 0]] = True
        floor_contact_items.append((contact_mask, palette[index % len(palette)]))
    _overlay(source, floor_contact_items, diagnostics_dir / "floor_contact_candidates.png")

    right_wall_mask = plane_image_masks["right_wall"]
    walllight_near = masks[wall_light_id] & valid
    _overlay(source, [(right_wall_mask, (50, 210, 110)), (walllight_near, (255, 40, 220))], diagnostics_dir / "wall_light_to_wall_proximity.png")
    _relationship_preview(source, object_geometry["objects"], relationships, diagnostics_dir / "relationship_graph_preview.png")
    _combined_overview(source, structural_planes, relationships, diagnostics_dir / "combined_confidence_warning_overview.png")

    plane_angles: dict[str, float] = {}
    for a, b in [("floor", "left_wall"), ("floor", "right_wall"), ("left_wall", "right_wall")]:
        dot = abs(float(np.dot(by_semantic[a]["normal"], by_semantic[b]["normal"])))
        plane_angles[f"{a}_to_{b}"] = math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))
    confidences = np.asarray([relationship["confidence"] for relationship in relationships])
    contradictions: list[str] = []
    filing_id = next(obj["object_id"] for obj in manifest["objects"] if obj["semantic_label"] == "left_tall_filing_cabinet")
    filing_floor = support_quality[filing_id]["floor"]
    if not filing_floor["supported"]:
        contradictions.append("filing cabinet is large furniture but visible floor-contact evidence is borderline; candidate retained without assertion")
    if not support_quality[box_id].get("desk_tabletop", {}).get("supported", False):
        contradictions.append("desktop box expected desk support is not confirmed by local geometry")

    camera_uncertainty = object_geometry["camera_model_uncertainty"]
    warnings = list(object_geometry["warnings"])
    warnings.extend(
        [
            "structural fitting excludes six known masks but unmasked furniture remains as robust-fit outliers",
            "canonical coordinates use MoGe scale without claiming that absolute metric scale is verified",
            "canonical transform estimates orientation and origin from planes but does not estimate camera extrinsics",
            "relationship records are deterministic candidates, not a final scene graph",
        ]
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "scene_id": manifest["scene_id"],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "fixture_manifest": str(manifest_path.relative_to(EXPERIMENT_ROOT)),
            "moge_geometry": str((moge_dir / "geometry.npz").relative_to(EXPERIMENT_ROOT)),
            "object_geometry": str((geometry_dir / "object_geometry.json").relative_to(EXPERIMENT_ROOT)),
            "image_dimensions": {"width": width, "height": height},
            "known_object_mask_count": len(masks),
        },
        "camera_evidence": {
            "normalized_intrinsics": intrinsics.tolist(),
            "estimated_horizontal_fov_degrees": moge_metadata["estimated_fov_x_degrees"],
            "model_name": moge_metadata["model_name"],
            "model_revision": moge_metadata.get("model_revision"),
            "extrinsics_estimated": False,
        },
        "camera_model_uncertainty": camera_uncertainty,
        "coordinate_systems": {
            "raw_moge_camera": object_geometry["coordinate_systems"]["raw_moge"],
            "scene_normalized_camera": object_geometry["coordinate_systems"]["scene_normalized"],
            "canonical_scene_world": {
                "right_handed": True, "+X": "camera-right projected onto floor", "+Y": "cross(+Z,+X)", "+Z": "floor normal toward occupied room volume",
                "origin": "scene-wide valid-point median projected orthogonally onto the fitted floor",
                "scale": "same unverified MoGe metric scale; rigid transform only",
            },
            "blender_camera_local": {
                "definition": "Blender camera local X right, Y up, camera looks along -Z",
                "opencv_to_blender_camera_local_matrix": cv_to_blender.tolist(),
                "inverse_matrix": np.linalg.inv(cv_to_blender).tolist(),
                "extrinsics_claimed": False,
            },
        },
        "transforms": {
            "raw_moge_to_scene_normalized": raw_to_normalized.tolist(),
            "scene_normalized_to_raw_moge": normalized_to_raw.tolist(),
            "raw_moge_to_canonical_scene_world": raw_to_canonical.tolist(),
            "canonical_scene_world_to_raw_moge": canonical_to_raw.tolist(),
            "uniform_scene_normalization_scale": object_geometry["raw_to_normalized_transformation"]["scale_factor"],
            "canonical_rigid_transform_scale": 1.0,
            "canonical_origin_raw_moge": canonical["origin_raw"].tolist(),
            "canonical_axes_in_raw_moge": {key: value.tolist() for key, value in canonical["axes_raw"].items()},
            "canonical_rotation_determinant": canonical["determinant"],
        },
        "structural_plane_parameters": {
            "known_object_masks_excluded": True, "normal_cluster_count": 8, "ransac_distance_threshold": 0.05,
            "minimum_cluster_points": 1000, "minimum_plane_inliers": 1800,
            "black_background_excluded_by_moge_validity": True,
        },
        "structural_planes": structural_planes,
        "plane_intersections": plane_intersections,
        "canonical_gravity_up_direction": [0.0, 0.0, 1.0],
        "object_geometry_references": [
            {
                "object_id": obj["object_id"], "semantic_label": obj["semantic_label"], "geometry_confidence": obj["geometry_confidence"],
                "object_geometry_record": f"object_geometry.json#/objects/{index}",
            }
            for index, obj in enumerate(object_geometry["objects"])
        ],
        "object_support_surface_candidates": support_patches,
        "pairwise_spatial_evidence": pairwise,
        "relationship_candidates": relationships,
        "thresholds": {
            "floor_contact_distance": 0.10, "support_candidate_confidence": 0.30, "support_assertion_confidence": 0.45,
            "wall_attachment_distance": 0.12, "occlusion_adjacency_pixels": 3, "occlusion_depth_delta": 0.03,
            "relative_height_difference": 0.06, "relative_depth_difference": 0.10,
        },
        "quality_checks": {
            "plane_normal_angles_degrees": plane_angles,
            "plane_residual_statistics": {plane["plane_id"]: plane["residual_statistics"] for plane in structural_planes},
            "plane_intersection_stability": {item["intersection_id"]: {"stable": item["stable"], "condition_sine": item["condition_sine"], "confidence": item.get("confidence")} for item in plane_intersections},
            "transform_round_trip_max_error": round_trip_error,
            "canonical_distance_preservation_max_error": distance_error,
            "scene_normalized_distance_scale_max_error": normalized_distance_scale_error,
            "canonical_rotation_determinant": canonical["determinant"],
            "canonical_floor_equation": next(p["canonical_plane_equation"] for p in structural_planes if p["semantic_candidate"] == "floor"),
            "occupied_point_canonical_height": float(transform_points(occupied_point[None, :], raw_to_canonical)[0, 2]),
            "object_support_distances": support_quality,
            "support_patch_coverage": {item["object_id"]: item["visible_support_patch"].get("coverage_ratio_of_visible_object", 0.0) for item in support_patches},
            "relationship_confidence_distribution": {
                "count": int(len(confidences)), "minimum": float(confidences.min()), "median": float(np.median(confidences)),
                "maximum": float(confidences.max()), "below_0_5": int((confidences < 0.5).sum()), "at_or_above_0_75": int((confidences >= 0.75).sum()),
            },
            "semantic_role_contradictions": contradictions,
        },
        "warnings": warnings,
        "next_phase_readiness": {
            "status": "conditionally_ready_for_vlm_review",
            "sufficiently_reliable": True,
            "explanation": "three coherent structural planes, reversible canonical coordinates, and expected desk/box, desk/floor, coat-rack/floor, and wall-light/wall candidates are present; unresolved evidence remains explicit",
            "blocking_uncertainties": [
                "absolute metric scale is unverified", "perspective versus orthographic/isometric camera remains unresolved",
                "chair physical support is unknown because only the backrest is visible", "filing-cabinet floor contact is borderline",
                "desktop-box full depth extent is unreliable and excluded from support evidence",
            ],
        },
        "diagnostic_artifacts": {
            "structural_plane_segmentation": "scene_diagnostics/structural_plane_segmentation.png",
            "floor_inliers": "scene_diagnostics/floor_inliers.png", "left_wall_inliers": "scene_diagnostics/left_wall_inliers.png",
            "right_wall_inliers": "scene_diagnostics/right_wall_inliers.png", "rejected_plane_outliers": "scene_diagnostics/rejected_plane_outliers.png",
            "source_colored_structural_planes": "scene_diagnostics/source_colored_structural_planes.ply",
            "canonical_scene_point_cloud": "scene_diagnostics/canonical_scene_point_cloud.ply", "canonical_axes": "scene_diagnostics/canonical_axes.ply",
            "desk_tabletop_support_patch": "scene_diagnostics/desk_tabletop_support_patch.png",
            "desktop_box_to_desk_contact": "scene_diagnostics/desktop_box_to_desk_contact.png",
            "floor_contact_candidates": "scene_diagnostics/floor_contact_candidates.png",
            "wall_light_to_wall_proximity": "scene_diagnostics/wall_light_to_wall_proximity.png",
            "relationship_graph_preview": "scene_diagnostics/relationship_graph_preview.png",
            "combined_confidence_warning_overview": "scene_diagnostics/combined_confidence_warning_overview.png",
        },
        "plane_fit_diagnostics": plane_diagnostics,
    }
    result = json.loads(json.dumps(result, default=_safe_record, allow_nan=False))
    output_json = geometry_dir / "scene_evidence.json"
    output_md = geometry_dir / "scene_evidence.md"
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_md, result)
    return result


def _relationship_preview(source: np.ndarray, objects: list[dict[str, Any]], relationships: list[dict[str, Any]], path: Path) -> None:
    image = Image.fromarray(source.copy())
    draw = ImageDraw.Draw(image)
    centroids = {obj["object_id"]: tuple(obj["mask_centroid_2d_xy"]) for obj in objects}
    shown = [r for r in relationships if r["target_id"] in centroids and r["confidence"] >= 0.60 and r["predicate"] in {"supported_by", "near", "image_occludes"}]
    colors = {"supported_by": (40, 220, 80), "near": (255, 190, 40), "image_occludes": (255, 50, 190)}
    for relationship in shown[:18]:
        start, end = centroids[relationship["subject_object_id"]], centroids[relationship["target_id"]]
        color = colors[relationship["predicate"]]
        draw.line((*start, *end), fill=color, width=2)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        draw.text(midpoint, relationship["predicate"], fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(path)


def _combined_overview(source: np.ndarray, planes: list[dict[str, Any]], relationships: list[dict[str, Any]], path: Path) -> None:
    panel_width = 430
    image = Image.new("RGB", (source.shape[1] + panel_width, source.shape[0]), (20, 20, 24))
    image.paste(Image.fromarray(source), (0, 0))
    draw = ImageDraw.Draw(image)
    x = source.shape[1] + 12
    draw.text((x, 10), "Structural planes", fill=(240, 240, 240))
    y = 28
    for plane in planes:
        color = tuple(PLANE_COLORS[plane["semantic_candidate"]])
        draw.text((x, y), f"{plane['semantic_candidate']}: {plane['confidence']:.3f}, p95 {plane['residual_statistics']['p95']:.4f}", fill=color)
        y += 15
    y += 8
    draw.text((x, y), "Support / attachment / occlusion evidence", fill=(240, 240, 240))
    y += 18
    selected = [r for r in relationships if r["predicate"] in {"supported_by", "attached_to", "unknown_support", "image_occludes"}]
    for relationship in selected:
        text = f"{relationship['subject_object_id'][:8]} {relationship['predicate']} {(relationship['target_id'] or 'null')[:12]} {relationship['confidence']:.3f}"
        color = (60, 220, 100) if relationship["confidence"] >= 0.7 else (255, 190, 40) if relationship["confidence"] >= 0.45 else (230, 70, 70)
        draw.text((x, y), text, fill=color)
        y += 14
        if relationship["uncertainty"]:
            for line in textwrap.wrap(relationship["uncertainty"], width=66)[:2]:
                draw.text((x + 8, y), line, fill=(165, 165, 170))
                y += 11
        if y > image.height - 30:
            break
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--moge-dir", type=Path, default=DEFAULT_MOGE_DIR)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build(args.manifest.resolve(), args.moge_dir.resolve(), args.geometry_dir.resolve())
    print(json.dumps({"planes": [p["semantic_candidate"] for p in result["structural_planes"]], "relationships": len(result["relationship_candidates"]), "readiness": result["next_phase_readiness"]["status"]}, indent=2))
