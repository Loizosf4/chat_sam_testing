"""Build the deterministic, compact VLM review input package."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "inputs" / "office_test" / "manifest.json"
DEFAULT_GEOMETRY_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "geometry"
DEFAULT_VLM_DIR = EXPERIMENT_ROOT / "outputs" / "office_test" / "vlm"
PROMPT_VERSION = "scene_graph_review_v1"
DEFAULT_MODEL = "gpt-5.5"
OBJECT_COLORS = np.asarray(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]],
    dtype=np.uint8,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_candidate_id(candidate: dict[str, Any]) -> str:
    canonical = {
        "subject_object_id": candidate["subject_object_id"],
        "predicate": candidate["predicate"],
        "target_id": candidate.get("target_id"),
        "deterministic_confidence": candidate["confidence"],
        "evidence": candidate["evidence"],
        "thresholds_used": candidate["thresholds_used"],
        "evidence_source": candidate["evidence_source"],
        "contradictions": candidate["contradictions"],
        "uncertainty": candidate["uncertainty"],
    }
    return "rel_" + hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()[:24]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numbered_overlay(source: Image.Image, manifest: dict[str, Any], fixture_dir: Path) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    height, width = rgb.shape[:2]
    panel_width = 470
    canvas = Image.new("RGB", (width + panel_width, height), (20, 20, 24))
    for index, obj in enumerate(manifest["objects"], start=1):
        mask = np.asarray(Image.open(fixture_dir / obj["mask_filename"]).convert("L")) == 255
        color = OBJECT_COLORS[index - 1].astype(np.float32)
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
        boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        rgb[boundary] = OBJECT_COLORS[index - 1]
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((width + 12, 10), "Approved semantic objects", fill=(245, 245, 245))
    for index, obj in enumerate(manifest["objects"], start=1):
        mask = np.asarray(Image.open(fixture_dir / obj["mask_filename"]).convert("L")) == 255
        ys, xs = np.nonzero(mask)
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        color = tuple(int(v) for v in OBJECT_COLORS[index - 1])
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(0, 0, 0), outline=color, width=2)
        draw.text((cx - 3, cy - 6), str(index), fill=color)
        y = 37 + (index - 1) * 65
        draw.rectangle((width + 12, y, width + 25, y + 13), fill=color)
        draw.text((width + 31, y), f"{index}. {obj['semantic_label']}", fill=color)
        draw.text((width + 31, y + 15), obj["object_id"], fill=(215, 215, 220))
        draw.text((width + 31, y + 29), f"roles: {', '.join(obj.get('selection_role', []))}", fill=(155, 155, 165))
    return canvas


def _labeled_mosaic(items: list[tuple[str, Path]]) -> Image.Image:
    tiles: list[Image.Image] = []
    for label, path in items:
        image = Image.open(path).convert("RGB")
        tile = ImageOps.expand(image, border=(0, 24, 0, 0), fill=(15, 15, 18))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 6), label, fill=(245, 245, 245))
        tiles.append(tile)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (width * 2, height * 2), (15, 15, 18))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 2) * width, (index // 2) * height))
    return canvas


def _compact_vlm_input(
    manifest: dict[str, Any],
    object_geometry: dict[str, Any],
    scene_evidence: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    object_geometry_by_id = {obj["object_id"]: obj for obj in object_geometry["objects"]}
    allowed_object_ids = [obj["object_id"] for obj in manifest["objects"]]
    allowed_plane_ids = [plane["plane_id"] for plane in scene_evidence["structural_planes"]]
    relationships: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for candidate in scene_evidence["relationship_candidates"]:
        candidate_id = stable_candidate_id(candidate)
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate deterministic candidate identity: {candidate_id}")
        candidate_ids.add(candidate_id)
        relationships.append(
            {
                "candidate_id": candidate_id,
                "subject_object_id": candidate["subject_object_id"],
                "predicate": candidate["predicate"],
                "target_id": candidate.get("target_id"),
                "deterministic_confidence": candidate["confidence"],
                "evidence": candidate["evidence"],
                "thresholds_used": candidate["thresholds_used"],
                "evidence_source": candidate["evidence_source"],
                "contradictions": candidate["contradictions"],
                "uncertainty": candidate["uncertainty"],
                "requires_vlm_review": candidate["requires_vlm_review"],
                "requires_user_review": candidate["requires_user_review"],
            }
        )
    return {
        "schema_version": "1.0",
        "scene_id": manifest["scene_id"],
        "requested_model": requested_model,
        "prompt_version": PROMPT_VERSION,
        "allowed_object_ids": allowed_object_ids,
        "allowed_plane_ids": allowed_plane_ids,
        "objects": [
            {
                "object_index": index,
                "object_id": manifest_obj["object_id"],
                "semantic_label": manifest_obj["semantic_label"],
                "selection_roles": manifest_obj.get("selection_role", []),
                "mask_bbox_xyxy_inclusive": object_geometry_by_id[manifest_obj["object_id"]]["bbox_2d_xyxy_inclusive"],
                "geometry_confidence": object_geometry_by_id[manifest_obj["object_id"]]["geometry_confidence"],
                "visible_surface_warnings": object_geometry_by_id[manifest_obj["object_id"]]["warnings"],
                "connected_component_count": object_geometry_by_id[manifest_obj["object_id"]]["connected_component_count"],
                "oriented_bounds": {
                    "estimated": object_geometry_by_id[manifest_obj["object_id"]]["oriented_bounding_box"]["estimated"],
                    "confidence": object_geometry_by_id[manifest_obj["object_id"]]["oriented_bounding_box"]["confidence"],
                },
                "geometry_source_reference": f"object_geometry.json#/objects/{index - 1}",
            }
            for index, manifest_obj in enumerate(manifest["objects"], start=1)
        ],
        "structural_plane_evidence": [
            {
                "plane_id": plane["plane_id"],
                "semantic_candidate": plane["semantic_candidate"],
                "equation_raw_moge": plane["raw_moge_plane_equation"],
                "equation_canonical": plane["canonical_plane_equation"],
                "confidence": plane["confidence"],
                "inlier_count": plane["inlier_count"],
                "residual_statistics": plane["residual_statistics"],
                "image_extent": plane["image_extent"],
                "warnings": plane["warnings"],
            }
            for plane in scene_evidence["structural_planes"]
        ],
        "camera_evidence": scene_evidence["camera_evidence"],
        "camera_model_uncertainty": scene_evidence["camera_model_uncertainty"],
        "support_surface_candidates": scene_evidence["object_support_surface_candidates"],
        "relationship_candidates": relationships,
        "deterministic_contradictions": scene_evidence["quality_checks"]["semantic_role_contradictions"],
        "required_review_flags": {
            "floor_extent_includes_exterior_background": True,
            "wall_extents_include_decor_or_unmasked_pixels": True,
            "plane_equations_stronger_than_visible_extents": True,
            "absolute_metric_scale_unverified": True,
            "camera_model_unresolved": True,
            "chair_support_not_visible": True,
            "filing_cabinet_floor_support_borderline": True,
            "desktop_box_depth_extent_unreliable": True,
            "desktop_box_mask_leakage_not_established": True,
            "desktop_box_chair_occlusion_requires_skeptical_review": True,
            "desktop_box_chair_overlap_inspection_note": "Inspection described approximately two boundary/overlap pixels; use the candidate's deterministic evidence values and do not auto-accept.",
        },
        "review_contract": {
            "exact_object_count": len(allowed_object_ids),
            "exact_relationship_review_count": len(relationships),
            "no_new_ids": True,
            "no_numeric_geometry": True,
            "hypotheses_unverified_only": True,
        },
    }


def build_input_package(
    manifest_path: Path = DEFAULT_MANIFEST,
    geometry_dir: Path = DEFAULT_GEOMETRY_DIR,
    vlm_dir: Path = DEFAULT_VLM_DIR,
    requested_model: str | None = None,
) -> dict[str, Any]:
    requested_model = requested_model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    if not requested_model:
        raise ValueError("OPENAI_MODEL must not be empty")
    input_dir = vlm_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(manifest_path)
    object_geometry = _load_json(geometry_dir / "object_geometry.json")
    scene_evidence = _load_json(geometry_dir / "scene_evidence.json")
    source_path = manifest_path.parent / manifest["image"]["filename"]
    with Image.open(source_path) as source:
        source_rgb = source.convert("RGB")
        source_rgb.save(input_dir / "source_image.png")
        _numbered_overlay(source_rgb, manifest, manifest_path.parent).save(input_dir / "numbered_object_overlay.png")
    Image.open(geometry_dir / "scene_diagnostics" / "structural_plane_segmentation.png").convert("RGB").save(input_dir / "structural_planes_overlay.png")
    _labeled_mosaic(
        [
            ("Desk tabletop patch", geometry_dir / "scene_diagnostics" / "desk_tabletop_support_patch.png"),
            ("Desktop box / desk contact", geometry_dir / "scene_diagnostics" / "desktop_box_to_desk_contact.png"),
            ("Floor-contact candidates", geometry_dir / "scene_diagnostics" / "floor_contact_candidates.png"),
            ("Wall-light / wall proximity", geometry_dir / "scene_diagnostics" / "wall_light_to_wall_proximity.png"),
        ]
    ).save(input_dir / "support_evidence_overlay.png")
    Image.open(geometry_dir / "scene_diagnostics" / "relationship_graph_preview.png").convert("RGB").save(input_dir / "relationship_evidence_overlay.png")

    vlm_input = _compact_vlm_input(manifest, object_geometry, scene_evidence, requested_model)
    vlm_input_path = input_dir / "vlm_input.json"
    vlm_input_path.write_text(json.dumps(vlm_input, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prompt_path = EXPERIMENT_ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
    schema_path = EXPERIMENT_ROOT / "schemas" / "vlm_scene_graph.schema.json"
    image_names = [
        "source_image.png",
        "numbered_object_overlay.png",
        "structural_planes_overlay.png",
        "support_evidence_overlay.png",
        "relationship_evidence_overlay.png",
    ]
    files = [
        {"filename": name, "sha256": sha256_file(input_dir / name), "bytes": (input_dir / name).stat().st_size, "media_type": "image/png"}
        for name in image_names
    ]
    files.append({"filename": "vlm_input.json", "sha256": sha256_file(vlm_input_path), "bytes": vlm_input_path.stat().st_size, "media_type": "application/json"})
    request_manifest = {
        "schema_version": "1.0",
        "api": "Responses API",
        "sdk": {"package": "openai", "version": importlib.metadata.version("openai")},
        "requested_model": requested_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_file(prompt_path),
        "structured_output_schema": str(schema_path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"),
        "structured_output_schema_sha256": sha256_file(schema_path),
        "input_order": ["source_image.png", "numbered_object_overlay.png", "structural_planes_overlay.png", "support_evidence_overlay.png", "relationship_evidence_overlay.png", "vlm_input.json"],
        "files": files,
        "candidate_count": len(vlm_input["relationship_candidates"]),
        "object_count": len(vlm_input["allowed_object_ids"]),
        "plane_count": len(vlm_input["allowed_plane_ids"]),
        "credentials_included": False,
        "request_snapshot_stored": False,
    }
    request_manifest["request_fingerprint"] = hashlib.sha256(canonical_json(request_manifest).encode("utf-8")).hexdigest()
    (input_dir / "request_manifest.json").write_text(json.dumps(request_manifest, indent=2) + "\n", encoding="utf-8")
    return {"vlm_input": vlm_input, "request_manifest": request_manifest, "input_dir": input_dir}


def validate_request_manifest(input_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest = _load_json(input_dir / "request_manifest.json")
    fingerprint = manifest.pop("request_fingerprint", None)
    expected = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
    if fingerprint != expected:
        errors.append("request fingerprint mismatch")
    for item in manifest["files"]:
        path = input_dir / item["filename"]
        if not path.is_file():
            errors.append(f"missing request input: {item['filename']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['filename']}")
    if manifest.get("credentials_included") is not False:
        errors.append("request manifest must state credentials_included=false")
    return errors


if __name__ == "__main__":
    package = build_input_package()
    errors = validate_request_manifest(package["input_dir"])
    if errors:
        raise SystemExit("; ".join(errors))
    print(json.dumps({"input_dir": str(package["input_dir"]), "objects": package["request_manifest"]["object_count"], "candidates": package["request_manifest"]["candidate_count"], "model": package["request_manifest"]["requested_model"]}, indent=2))
