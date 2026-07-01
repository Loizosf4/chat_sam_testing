from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.vlm_input_package import build_input_package, stable_candidate_id, validate_request_manifest
from src.vlm_models import SemanticSceneGraph
from src.vlm_review import VLMRefusalError, _live, build_request_payload, call_responses_api, validate_dry_run_payload
from src.vlm_validation import redact_secrets, validate_semantic_scene_graph


REQUESTED_MODEL = "gpt-5.5"
RETURNED_MODEL = "gpt-5.5-2026-04-23"


def _valid_result(vlm_input: dict) -> dict:
    strategies = {
        "left_tall_filing_cabinet": "retained_obb",
        "desk": "support_aligned_proxy",
        "desk_chair": "visible_surface_proxy",
        "desktop_box": "support_aligned_proxy",
        "coat_rack": "visible_surface_proxy",
        "wall_light_fixture": "wall_attached_thin_proxy",
    }
    return {
        "schema_version": "1.0",
        "scene_id": vlm_input["scene_id"],
        "generator": {
            "provider": "openai",
            "requested_model": REQUESTED_MODEL,
            "returned_model": RETURNED_MODEL,
            "prompt_version": "scene_graph_review_v1",
        },
        "camera_review": {
            "recommended_model": "unresolved",
            "confidence": 0.8,
            "evidence": ["Rendered isometric appearance"],
            "contradictory_evidence": ["MoGe estimated a perspective FOV"],
            "test_both_in_blender": True,
            "requires_user_review": True,
        },
        "structural_plane_reviews": [
            {
                "plane_id": plane["plane_id"],
                "semantic_label": plane["semantic_candidate"],
                "decision": "accept_equation",
                "confidence": plane["confidence"],
                "extent_policy": "equation_only" if plane["plane_id"] == "plane_floor" else "requires_reconstruction_review",
                "reasoning_summary": "Equation is useful; visible extent remains uncertain.",
                "warnings": plane["warnings"],
                "requires_user_review": True,
            }
            for plane in vlm_input["structural_plane_evidence"]
        ],
        "objects": [
            {
                "object_id": obj["object_id"],
                "original_label": obj["semantic_label"],
                "reviewed_label": obj["semantic_label"],
                "label_decision": "accept",
                "semantic_category": "scene_object",
                "primitive_policy": "one_cube",
                "geometry_strategy": strategies[obj["semantic_label"]],
                "geometry_source_reference": obj["geometry_source_reference"],
                "support_summary": "Use deterministic support evidence only.",
                "attachment_summary": "No additional attachment is invented.",
                "occlusion_summary": "Visible masks only; hidden geometry is not inferred.",
                "confidence": 0.8,
                "uncertainty": obj["visible_surface_warnings"],
                "requires_user_review": bool(obj["visible_surface_warnings"]),
            }
            for obj in vlm_input["objects"]
        ],
        "relationship_reviews": [
            {
                "candidate_id": candidate["candidate_id"],
                "subject_object_id": candidate["subject_object_id"],
                "predicate": candidate["predicate"],
                "target_id": candidate["target_id"],
                "deterministic_confidence": candidate["deterministic_confidence"],
                "decision": "uncertain",
                "reviewed_confidence": candidate["deterministic_confidence"],
                "evidence_summary": "Deterministic evidence retained for review.",
                "contradiction_summary": candidate["uncertainty"],
                "requires_user_review": True,
            }
            for candidate in vlm_input["relationship_candidates"]
        ],
        "proposed_hypotheses": [],
        "global_uncertainties": ["Absolute scale and camera model remain unresolved."],
        "requires_user_review": True,
        "review_summary": "Evidence is structurally complete and uncertainty is preserved.",
    }


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return build_input_package(vlm_dir=tmp_path_factory.mktemp("vlm-package"), requested_model=REQUESTED_MODEL)


def _validate(data: dict, vlm_input: dict):
    return validate_semantic_scene_graph(data, vlm_input, requested_model=REQUESTED_MODEL, returned_model=RETURNED_MODEL)


def test_candidate_ids_are_deterministic_and_evidence_sensitive(package: dict) -> None:
    source_candidate = package["vlm_input"]["relationship_candidates"][0]
    raw = {
        "subject_object_id": source_candidate["subject_object_id"],
        "predicate": source_candidate["predicate"],
        "target_id": source_candidate["target_id"],
        "confidence": source_candidate["deterministic_confidence"],
        "evidence": source_candidate["evidence"],
        "thresholds_used": source_candidate["thresholds_used"],
        "evidence_source": source_candidate["evidence_source"],
        "contradictions": source_candidate["contradictions"],
        "uncertainty": source_candidate["uncertainty"],
    }
    assert stable_candidate_id(raw) == stable_candidate_id(copy.deepcopy(raw))
    changed = copy.deepcopy(raw)
    changed["evidence"] = {**changed["evidence"], "test_delta": 1}
    assert stable_candidate_id(raw) != stable_candidate_id(changed)


def test_request_manifest_hashes_are_stable(tmp_path: Path) -> None:
    first = build_input_package(vlm_dir=tmp_path / "vlm", requested_model=REQUESTED_MODEL)
    second = build_input_package(vlm_dir=tmp_path / "vlm", requested_model=REQUESTED_MODEL)
    assert first["request_manifest"] == second["request_manifest"]
    assert validate_request_manifest(first["input_dir"]) == []


def test_valid_result_preserves_exact_objects_and_candidate_coverage(package: dict) -> None:
    validated, report = _validate(_valid_result(package["vlm_input"]), package["vlm_input"])
    assert report["valid"]
    assert validated is not None
    assert {obj.object_id for obj in validated.objects} == set(package["vlm_input"]["allowed_object_ids"])
    assert {review.candidate_id for review in validated.relationship_reviews} == {candidate["candidate_id"] for candidate in package["vlm_input"]["relationship_candidates"]}


def test_unknown_object_id_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["objects"][0]["object_id"] = "unknown_object"
    _, report = _validate(data, package["vlm_input"])
    assert not report["valid"]
    assert any("unknown object" in error for error in report["errors"])


def test_duplicate_object_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["objects"][1] = copy.deepcopy(data["objects"][0])
    _, report = _validate(data, package["vlm_input"])
    assert any("duplicate object" in error for error in report["errors"])


def test_missing_object_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["objects"].pop()
    _, report = _validate(data, package["vlm_input"])
    assert any("missing object" in error for error in report["errors"])


def test_unknown_candidate_review_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["relationship_reviews"][0]["candidate_id"] = "rel_unknown"
    _, report = _validate(data, package["vlm_input"])
    assert any("unknown relationship candidate" in error for error in report["errors"])


def test_invented_transform_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["objects"][0]["position"] = [1.0, 2.0, 3.0]
    _, report = _validate(data, package["vlm_input"])
    assert any("invented numeric geometry" in error for error in report["errors"])


def test_contradictory_relationships_are_rejected(package: dict) -> None:
    vlm_input = copy.deepcopy(package["vlm_input"])
    a, b = vlm_input["allowed_object_ids"][:2]
    custom = []
    for candidate_id, predicate in [("rel_test_left", "left_of"), ("rel_test_right", "right_of")]:
        custom.append(
            {
                "candidate_id": candidate_id,
                "subject_object_id": a,
                "predicate": predicate,
                "target_id": b,
                "deterministic_confidence": 0.8,
                "evidence": {},
                "thresholds_used": {},
                "evidence_source": ["test"],
                "contradictions": [],
                "uncertainty": "",
                "requires_vlm_review": True,
                "requires_user_review": False,
            }
        )
    vlm_input["relationship_candidates"] = custom
    data = _valid_result(vlm_input)
    for review in data["relationship_reviews"]:
        review["decision"] = "accept"
        review["requires_user_review"] = False
    _, report = _validate(data, vlm_input)
    assert any("contradictory accepted left/right" in error for error in report["errors"])


def test_invalid_confidence_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["objects"][0]["confidence"] = 1.1
    _, report = _validate(data, package["vlm_input"])
    assert any("schema validation failed" in error for error in report["errors"])


def test_hypothesis_unknown_target_is_rejected(package: dict) -> None:
    data = _valid_result(package["vlm_input"])
    data["proposed_hypotheses"] = [
        {
            "hypothesis_id": "hyp_1",
            "subject_object_id": package["vlm_input"]["allowed_object_ids"][0],
            "predicate": "near",
            "target_id": "not_in_fixture",
            "status": "unverified",
            "evidence_type": "semantic",
            "evidence_summary": "Test hypothesis.",
            "confidence": 0.2,
            "requires_user_review": True,
        }
    ]
    _, report = _validate(data, package["vlm_input"])
    assert any("hypothesis target is unknown" in error for error in report["errors"])


def test_api_key_redaction() -> None:
    secret = "sk-testSecretCredential123456"
    sanitized = redact_secrets({"error": f"request failed for {secret}", "nested": [secret]}, (secret,))
    serialized = str(sanitized)
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_dry_run_request_is_complete_and_has_no_credentials(package: dict) -> None:
    payload = build_request_payload(package["input_dir"], REQUESTED_MODEL)
    assert validate_dry_run_payload(payload, package["input_dir"]) == []
    assert len([item for item in payload["input"][1]["content"] if item["type"] == "input_image"]) == 5


class _MockUsage:
    def model_dump(self, mode: str = "json"):
        return {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}


class _MockResponse:
    def __init__(self, parsed: SemanticSceneGraph | None):
        self.id = "resp_mock"
        self.model = RETURNED_MODEL
        self.status = "completed"
        self.usage = _MockUsage()
        self.output_parsed = parsed


class _MockResponses:
    def __init__(self, response: _MockResponse):
        self.response = response
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        return self.response


class _MockClient:
    def __init__(self, response: _MockResponse):
        self.responses = _MockResponses(response)


def test_mocked_structured_api_response_parsing(package: dict) -> None:
    parsed = SemanticSceneGraph.model_validate(_valid_result(package["vlm_input"]))
    client = _MockClient(_MockResponse(parsed))
    response, attempts = call_responses_api(client, build_request_payload(package["input_dir"], REQUESTED_MODEL))
    assert response.output_parsed == parsed
    assert attempts == 1
    assert client.responses.calls == 1


def test_mocked_live_run_writes_validated_sanitized_outputs(tmp_path: Path) -> None:
    package = build_input_package(vlm_dir=tmp_path / "vlm", requested_model=REQUESTED_MODEL)
    parsed = SemanticSceneGraph.model_validate(_valid_result(package["vlm_input"]))
    client = _MockClient(_MockResponse(parsed))
    assert _live(tmp_path / "vlm", REQUESTED_MODEL, client=client) == 0
    assert (tmp_path / "vlm" / "semantic_scene_graph.json").is_file()
    raw = (tmp_path / "vlm" / "raw_response.json").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in raw
    assert "hidden_reasoning_stored\": false" in raw


def test_refusal_is_retried_once_then_reported(package: dict) -> None:
    client = _MockClient(_MockResponse(None))
    with pytest.raises(VLMRefusalError):
        call_responses_api(client, build_request_payload(package["input_dir"], REQUESTED_MODEL))
    assert client.responses.calls == 2
