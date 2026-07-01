"""Independent semantic and identity validation for structured VLM reviews."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.vlm_models import SemanticSceneGraph


SECRET_PATTERN = re.compile(r"\b(?:sk|sess|key)-[A-Za-z0-9_-]{8,}\b")
FORBIDDEN_GEOMETRY_KEYS = {
    "position",
    "rotation",
    "scale",
    "transform",
    "matrix",
    "quaternion",
    "coordinates",
    "dimensions",
    "translation",
    "euler",
}


def redact_secrets(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {key: redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return SECRET_PATTERN.sub("[REDACTED]", redacted)
    return value


def _find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_GEOMETRY_KEYS:
                errors.append(f"invented numeric geometry field is forbidden: {path}.{key}")
            errors.extend(_find_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return errors


def validate_semantic_scene_graph(
    raw_result: dict[str, Any] | SemanticSceneGraph,
    vlm_input: dict[str, Any],
    *,
    requested_model: str,
    returned_model: str,
) -> tuple[SemanticSceneGraph | None, dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_dict = raw_result.model_dump(mode="json") if isinstance(raw_result, SemanticSceneGraph) else raw_result
    errors.extend(_find_forbidden_keys(raw_dict))
    try:
        result = raw_result if isinstance(raw_result, SemanticSceneGraph) else SemanticSceneGraph.model_validate(raw_result)
    except ValidationError as exc:
        result = None
        errors.append(f"schema validation failed: {exc}")
        return None, {"valid": False, "errors": errors, "warnings": warnings}

    if result.scene_id != vlm_input["scene_id"]:
        errors.append("scene ID differs from deterministic input")
    if result.generator.requested_model != requested_model:
        errors.append("generator.requested_model differs from configured model")
    if result.generator.returned_model != returned_model:
        errors.append("generator.returned_model differs from API response model")

    allowed_objects = vlm_input["allowed_object_ids"]
    returned_objects = [obj.object_id for obj in result.objects]
    if len(returned_objects) != len(set(returned_objects)):
        errors.append("duplicate object ID in VLM output")
    missing_objects = sorted(set(allowed_objects) - set(returned_objects))
    unknown_objects = sorted(set(returned_objects) - set(allowed_objects))
    if missing_objects:
        errors.append(f"missing object IDs: {missing_objects}")
    if unknown_objects:
        errors.append(f"unknown object IDs: {unknown_objects}")
    input_objects = {obj["object_id"]: obj for obj in vlm_input["objects"]}
    for obj in result.objects:
        if obj.object_id in input_objects and obj.original_label != input_objects[obj.object_id]["semantic_label"]:
            errors.append(f"original label changed for object {obj.object_id}")

    allowed_planes = set(vlm_input["allowed_plane_ids"])
    returned_planes = [plane.plane_id for plane in result.structural_plane_reviews]
    if len(returned_planes) != len(set(returned_planes)):
        errors.append("duplicate structural plane review")
    if set(returned_planes) != allowed_planes:
        errors.append(f"structural plane reviews must exactly cover {sorted(allowed_planes)}")
    for plane in result.structural_plane_reviews:
        if plane.plane_id == "plane_floor" and plane.extent_policy == "visible_image_extent":
            errors.append("floor visible_image_extent is forbidden because inliers include exterior background")

    candidates = {candidate["candidate_id"]: candidate for candidate in vlm_input["relationship_candidates"]}
    review_ids = [review.candidate_id for review in result.relationship_reviews]
    if len(review_ids) != len(set(review_ids)):
        errors.append("a deterministic candidate received multiple reviews")
    missing_reviews = sorted(set(candidates) - set(review_ids))
    unknown_reviews = sorted(set(review_ids) - set(candidates))
    if missing_reviews:
        errors.append(f"deterministic candidates missing reviews: {missing_reviews}")
    if unknown_reviews:
        errors.append(f"unknown relationship candidate IDs: {unknown_reviews}")
    for review in result.relationship_reviews:
        candidate = candidates.get(review.candidate_id)
        if candidate is None:
            continue
        identity = (review.subject_object_id, review.predicate, review.target_id)
        expected = (candidate["subject_object_id"], candidate["predicate"], candidate["target_id"])
        if identity != expected:
            errors.append(f"candidate identity changed for {review.candidate_id}")
        if abs(review.deterministic_confidence - candidate["deterministic_confidence"]) > 1e-12:
            errors.append(f"deterministic confidence changed for {review.candidate_id}")
        if review.decision == "reject" and not review.contradiction_summary.strip():
            errors.append(f"rejected candidate lacks rejection explanation: {review.candidate_id}")
        if candidate["uncertainty"] and review.decision == "accept" and not review.requires_user_review:
            errors.append(f"required uncertainty silently removed for {review.candidate_id}")

    accepted = [review for review in result.relationship_reviews if review.decision == "accept"]
    supports: dict[str, set[str]] = {}
    unknown_support: set[str] = set()
    left_relations: set[tuple[str, str]] = set()
    above_relations: set[tuple[str, str]] = set()
    occludes: set[tuple[str, str]] = set()
    for review in accepted:
        if review.predicate == "supported_by" and review.target_id is not None:
            supports.setdefault(review.subject_object_id, set()).add(review.target_id)
        elif review.predicate == "unknown_support":
            unknown_support.add(review.subject_object_id)
        elif review.predicate == "left_of" and review.target_id is not None:
            left_relations.add((review.subject_object_id, review.target_id))
        elif review.predicate == "right_of" and review.target_id is not None:
            left_relations.add((review.target_id, review.subject_object_id))
        elif review.predicate == "above" and review.target_id is not None:
            above_relations.add((review.subject_object_id, review.target_id))
        elif review.predicate == "below" and review.target_id is not None:
            above_relations.add((review.target_id, review.subject_object_id))
        elif review.predicate == "image_occludes" and review.target_id is not None:
            occludes.add((review.subject_object_id, review.target_id))
        elif review.predicate == "image_occluded_by" and review.target_id is not None:
            occludes.add((review.target_id, review.subject_object_id))
    for subject, targets in supports.items():
        if len(targets) > 1:
            errors.append(f"object has incompatible definitive support targets: {subject} -> {sorted(targets)}")
        if subject in unknown_support:
            errors.append(f"unknown_support coexists with definitive support: {subject}")
    for relation, label in [(left_relations, "left/right"), (above_relations, "above/below"), (occludes, "occlusion")]:
        for subject, target in relation:
            if (target, subject) in relation:
                errors.append(f"contradictory accepted {label} relationships: {subject}, {target}")
                break

    valid_hypothesis_targets = set(allowed_objects) | allowed_planes
    hypothesis_ids: set[str] = set()
    for hypothesis in result.proposed_hypotheses:
        if hypothesis.hypothesis_id in hypothesis_ids:
            errors.append(f"duplicate hypothesis ID: {hypothesis.hypothesis_id}")
        hypothesis_ids.add(hypothesis.hypothesis_id)
        if hypothesis.subject_object_id not in set(allowed_objects):
            errors.append(f"hypothesis subject is unknown: {hypothesis.subject_object_id}")
        if hypothesis.target_id is not None and hypothesis.target_id not in valid_hypothesis_targets:
            errors.append(f"hypothesis target is unknown: {hypothesis.target_id}")

    if vlm_input["required_review_flags"]["camera_model_unresolved"]:
        if not result.camera_review.test_both_in_blender or not result.camera_review.requires_user_review:
            errors.append("camera-model uncertainty was not preserved")

    if not result.global_uncertainties:
        warnings.append("global_uncertainties is empty despite deterministic caveats")
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "objects_expected": len(allowed_objects),
            "objects_returned": len(result.objects),
            "planes_expected": len(allowed_planes),
            "planes_returned": len(result.structural_plane_reviews),
            "candidates_expected": len(candidates),
            "candidate_reviews_returned": len(result.relationship_reviews),
            "hypotheses_returned": len(result.proposed_hypotheses),
        },
    }
    return result if not errors else None, report


def write_validation_report(report: dict[str, Any], json_path: Path, markdown_path: Path, *, context: str) -> None:
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    errors = "\n".join(f"- {item}" for item in report.get("errors", [])) or "- None"
    warnings = "\n".join(f"- {item}" for item in report.get("warnings", [])) or "- None"
    counts = "\n".join(f"- {key}: {value}" for key, value in report.get("counts", {}).items()) or "- Not applicable"
    markdown_path.write_text(
        f"# VLM validation report\n\n- Context: {context}\n- Valid: **{report.get('valid', False)}**\n\n## Errors\n\n{errors}\n\n## Warnings\n\n{warnings}\n\n## Counts\n\n{counts}\n",
        encoding="utf-8",
    )
