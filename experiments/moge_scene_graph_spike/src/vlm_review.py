"""Construct, dry-run, call, and validate the OpenAI semantic review request."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import ValidationError
from PIL import Image, ImageDraw

from src.vlm_input_package import (
    DEFAULT_MODEL,
    DEFAULT_VLM_DIR,
    EXPERIMENT_ROOT,
    PROMPT_VERSION,
    build_input_package,
    sha256_file,
    validate_request_manifest,
)
from src.vlm_models import GeneratorInfo, SemanticSceneGraph
from src.vlm_validation import redact_secrets, validate_semantic_scene_graph, write_validation_report


IMAGE_ORDER = [
    ("Source image", "source_image.png"),
    ("Numbered approved-object overlay", "numbered_object_overlay.png"),
    ("Structural-plane evidence overlay", "structural_planes_overlay.png"),
    ("Support evidence overlay", "support_evidence_overlay.png"),
    ("Relationship evidence overlay", "relationship_evidence_overlay.png"),
]


class VLMRefusalError(RuntimeError):
    pass


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_request_payload(input_dir: Path, requested_model: str) -> dict[str, Any]:
    prompt = (EXPERIMENT_ROOT / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    vlm_input_text = (input_dir / "vlm_input.json").read_text(encoding="utf-8")
    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "Review this scene using the five labeled image inputs and deterministic JSON. Image labels precede each image. Return only the strict structured result.",
        }
    ]
    for label, filename in IMAGE_ORDER:
        user_content.append({"type": "input_text", "text": label})
        user_content.append({"type": "input_image", "image_url": _data_url(input_dir / filename), "detail": "original"})
    user_content.append({"type": "input_text", "text": "Deterministic VLM input JSON:\n" + vlm_input_text})
    return {
        "model": requested_model,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": prompt}]},
            {"role": "user", "content": user_content},
        ],
        "text_format": SemanticSceneGraph,
        "reasoning": {"effort": "medium"},
        "max_output_tokens": 30000,
        "store": False,
    }


def validate_dry_run_payload(payload: dict[str, Any], input_dir: Path) -> list[str]:
    errors = validate_request_manifest(input_dir)
    if payload.get("model") != json.loads((input_dir / "request_manifest.json").read_text(encoding="utf-8"))["requested_model"]:
        errors.append("request model differs from request manifest")
    if payload.get("text_format") is not SemanticSceneGraph:
        errors.append("strict typed text_format is missing")
    messages = payload.get("input", [])
    if [message.get("role") for message in messages] != ["developer", "user"]:
        errors.append("request must contain developer then user input")
        return errors
    images = [item for item in messages[1]["content"] if item.get("type") == "input_image"]
    if len(images) != len(IMAGE_ORDER):
        errors.append(f"expected {len(IMAGE_ORDER)} image inputs, found {len(images)}")
    for item, (_, filename) in zip(images, IMAGE_ORDER):
        url = item.get("image_url", "")
        prefix = "data:image/png;base64,"
        if not url.startswith(prefix):
            errors.append(f"invalid image data URL for {filename}")
            continue
        try:
            decoded = base64.b64decode(url[len(prefix) :], validate=True)
        except Exception:
            errors.append(f"invalid base64 image for {filename}")
            continue
        if decoded != (input_dir / filename).read_bytes():
            errors.append(f"data URL bytes differ from {filename}")
    request_text = json.dumps({key: value for key, value in payload.items() if key != "text_format"}, default=str)
    if "OPENAI_API_KEY" in request_text or "sk-" in request_text:
        errors.append("request payload appears to contain credentials")
    return errors


def _usage_record(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return json.loads(json.dumps(usage, default=str))


def _retryable(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(error, APIStatusError) and error.status_code >= 500


def call_responses_api(client: Any, payload: dict[str, Any]) -> tuple[Any, int]:
    attempts = 0
    while True:
        attempts += 1
        try:
            response = client.responses.parse(**payload)
            if getattr(response, "output_parsed", None) is None:
                raise VLMRefusalError("Responses API returned no parsed structured output; refusal or incomplete response")
            return response, attempts
        except (ValidationError, VLMRefusalError) as exc:
            if attempts < 2:
                continue
            raise exc
        except Exception as exc:
            if attempts < 2 and _retryable(exc):
                continue
            raise


def _write_review_overview(scene_graph: SemanticSceneGraph, vlm_input: dict[str, Any], output_path: Path, input_dir: Path) -> None:
    source = Image.open(input_dir / "source_image.png").convert("RGB")
    panel_width = 570
    canvas = Image.new("RGB", (source.width + panel_width, source.height), (20, 20, 24))
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    input_objects = {obj["object_id"]: obj for obj in vlm_input["objects"]}
    reviewed = {obj.object_id: obj for obj in scene_graph.objects}
    centroids: dict[str, tuple[float, float]] = {}
    for obj in scene_graph.objects:
        x0, y0, x1, y1 = input_objects[obj.object_id]["mask_bbox_xyxy_inclusive"]
        centroids[obj.object_id] = ((x0 + x1) / 2, (y0 + y1) / 2)
        color = (255, 190, 40) if obj.requires_user_review else (50, 220, 100)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.text((x0, max(0, y0 - 11)), obj.reviewed_label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
    relation_colors = {"accept": (40, 220, 90), "reject": (230, 70, 70), "uncertain": (255, 190, 40)}
    for review in scene_graph.relationship_reviews:
        if review.target_id in centroids and review.predicate in {"supported_by", "attached_to", "image_occludes", "unknown_support"}:
            draw.line((*centroids[review.subject_object_id], *centroids[review.target_id]), fill=relation_colors[review.decision], width=2)

    x, y = source.width + 12, 10
    draw.text((x, y), f"Camera: {scene_graph.camera_review.recommended_model} ({scene_graph.camera_review.confidence:.2f})", fill=(240, 240, 240))
    y += 18
    for plane in scene_graph.structural_plane_reviews:
        draw.text((x, y), f"{plane.plane_id}: {plane.decision}, {plane.extent_policy}, {plane.confidence:.2f}", fill=(180, 210, 255))
        y += 15
    y += 8
    draw.text((x, y), "Objects", fill=(240, 240, 240))
    y += 16
    for obj in scene_graph.objects:
        color = (255, 190, 40) if obj.requires_user_review else (50, 220, 100)
        draw.text((x, y), f"{obj.reviewed_label}: {obj.geometry_strategy} ({obj.confidence:.2f})", fill=color)
        y += 14
    y += 8
    counts = {decision: sum(review.decision == decision for review in scene_graph.relationship_reviews) for decision in ["accept", "reject", "uncertain"]}
    draw.text((x, y), f"Relationships: accepted {counts['accept']}, rejected {counts['reject']}, uncertain {counts['uncertain']}", fill=(240, 240, 240))
    y += 16
    draw.text((x, y), f"Hypotheses: {len(scene_graph.proposed_hypotheses)}", fill=(240, 240, 240))
    canvas.save(output_path)


def _dry_run(vlm_dir: Path, requested_model: str) -> int:
    package = build_input_package(vlm_dir=vlm_dir, requested_model=requested_model)
    payload = build_request_payload(package["input_dir"], requested_model)
    errors = validate_dry_run_payload(payload, package["input_dir"])
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    report = {
        "valid": not errors,
        "mode": "dry_run",
        "errors": errors,
        "warnings": (["OPENAI_API_KEY is absent; live request intentionally not attempted"] if not key_present else ["dry-run requested; live API was not called"]),
        "counts": {
            "objects": package["request_manifest"]["object_count"],
            "planes": package["request_manifest"]["plane_count"],
            "relationship_candidates": package["request_manifest"]["candidate_count"],
            "image_inputs": len(IMAGE_ORDER),
        },
    }
    write_validation_report(report, vlm_dir / "validation_report.json", vlm_dir / "validation_report.md", context="dry-run request construction")
    metadata = {
        "schema_version": "1.0",
        "mode": "dry_run",
        "api_call_status": "blocked_no_api_key" if not key_present else "not_called_dry_run",
        "requested_model": requested_model,
        "returned_model": None,
        "response_id": None,
        "token_usage": None,
        "api_duration_seconds": None,
        "prompt_version": PROMPT_VERSION,
        "input_hashes": {item["filename"]: item["sha256"] for item in package["request_manifest"]["files"]},
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "credentials_stored": False,
        "blockers": ([] if key_present else ["OPENAI_API_KEY is not set"]),
    }
    (vlm_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dry_run_valid": not errors, "api_key_present": key_present, "api_call_status": metadata["api_call_status"], "requested_model": requested_model}, indent=2))
    return 0 if not errors else 1


def _live(vlm_dir: Path, requested_model: str, client: Any | None = None) -> int:
    if not os.environ.get("OPENAI_API_KEY") and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set; run with --dry-run")
    package = build_input_package(vlm_dir=vlm_dir, requested_model=requested_model)
    payload = build_request_payload(package["input_dir"], requested_model)
    dry_errors = validate_dry_run_payload(payload, package["input_dir"])
    if dry_errors:
        raise RuntimeError("request validation failed: " + "; ".join(dry_errors))
    client = client or OpenAI()
    started = time.perf_counter()
    try:
        response, attempts = call_responses_api(client, payload)
        duration = time.perf_counter() - started
        returned_model = str(response.model)
        parsed: SemanticSceneGraph = response.output_parsed
        parsed = parsed.model_copy(
            update={
                "generator": GeneratorInfo(
                    provider="openai",
                    requested_model=requested_model,
                    returned_model=returned_model,
                    prompt_version=PROMPT_VERSION,
                )
            }
        )
        sanitized_response = {
            "response_id": response.id,
            "status": getattr(response, "status", None),
            "requested_model": requested_model,
            "returned_model": returned_model,
            "usage": _usage_record(response),
            "structured_result": parsed.model_dump(mode="json"),
            "sanitized": True,
            "hidden_reasoning_stored": False,
        }
        sanitized_response = redact_secrets(sanitized_response)
        (vlm_dir / "raw_response.json").write_text(json.dumps(sanitized_response, indent=2) + "\n", encoding="utf-8")
        validated, report = validate_semantic_scene_graph(
            parsed,
            package["vlm_input"],
            requested_model=requested_model,
            returned_model=returned_model,
        )
        write_validation_report(report, vlm_dir / "validation_report.json", vlm_dir / "validation_report.md", context="live structured response")
        if validated is not None:
            (vlm_dir / "semantic_scene_graph.json").write_text(validated.model_dump_json(indent=2) + "\n", encoding="utf-8")
            _write_review_overview(validated, package["vlm_input"], vlm_dir / "vlm_review_overview.png", package["input_dir"])
        metadata = {
            "schema_version": "1.0",
            "mode": "live",
            "api_call_status": "validated" if validated is not None else "post_validation_failed",
            "response_id": response.id,
            "requested_model": requested_model,
            "returned_model": returned_model,
            "token_usage": _usage_record(response),
            "api_duration_seconds": duration,
            "attempt_count": attempts,
            "prompt_version": PROMPT_VERSION,
            "input_hashes": {item["filename"]: item["sha256"] for item in package["request_manifest"]["files"]},
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "credentials_stored": False,
        }
        (vlm_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return 0 if validated is not None else 1
    except Exception as exc:
        duration = time.perf_counter() - started
        sanitized_error = redact_secrets(str(exc))
        report = {"valid": False, "errors": [sanitized_error], "warnings": [], "counts": {}}
        write_validation_report(report, vlm_dir / "validation_report.json", vlm_dir / "validation_report.md", context="live API error")
        metadata = {
            "schema_version": "1.0",
            "mode": "live",
            "api_call_status": "error",
            "requested_model": requested_model,
            "returned_model": None,
            "response_id": None,
            "token_usage": None,
            "api_duration_seconds": duration,
            "prompt_version": PROMPT_VERSION,
            "error": sanitized_error,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "credentials_stored": False,
        }
        (vlm_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="build and validate the full request without calling OpenAI")
    parser.add_argument("--vlm-dir", type=Path, default=DEFAULT_VLM_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    if not model:
        raise SystemExit("OPENAI_MODEL must not be empty")
    if args.dry_run:
        raise SystemExit(_dry_run(args.vlm_dir.resolve(), model))
    raise SystemExit(_live(args.vlm_dir.resolve(), model))
