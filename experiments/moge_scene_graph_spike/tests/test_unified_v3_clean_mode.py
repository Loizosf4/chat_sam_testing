import json
from pathlib import Path

import pytest

from src.unified_v3_scene_compiler import CleanReadGuard, ROOT
from tests.test_unified_v3_scene_compiler import load, unified_pair


def test_clean_guard_rejects_forbidden_or_unlisted_artifact(tmp_path):
    allowed=tmp_path/"allowed.json";allowed.write_text("{}",encoding="utf-8")
    forbidden=tmp_path/"approved_scene.blend";forbidden.write_bytes(b"BLENDER")
    guard=CleanReadGuard([allowed])
    assert guard.json(allowed)=={}
    with pytest.raises(PermissionError):guard.json(forbidden)
    assert str(forbidden.resolve()) in guard.denied_log


def test_clean_input_audit_contains_only_raw_inputs(unified_pair):
    first,_=unified_pair;audit=load(first/"clean_input_audit.json");forbidden=load(first/"forbidden_input_audit.json")
    assert audit["all_reads_allowed"] and not audit["denied_paths"]
    assert forbidden["passed"] and forbidden["violations"]==[]
    lowered="\n".join(audit["read_paths"]).lower()
    for token in (".blend","blender_execution","room_corrected","primitive_plan","approved"):
        assert token not in lowered


def test_compiler_source_has_no_approved_transform_constants():
    source=(ROOT/"src"/"unified_v3_scene_compiler.py").read_text(encoding="utf-8").lower()
    for forbidden in ("0.113511429","0.4257657802","-48.327316","candidate c","approved candidate"):
        assert forbidden not in source


def test_clean_outputs_are_deterministic(unified_pair):
    first,second=unified_pair
    names=("unified_scene_plan.json","room_plan.json","camera_candidates.json","object_pose_report.json","collision_report.json","confidence_report.json","ambiguity_report.md","blender_one_batch_manifest.json")
    for name in names:
        assert (first/name).read_bytes()==(second/name).read_bytes()


def test_required_clean_outputs_and_per_object_diagnostics(unified_pair):
    first,_=unified_pair
    required={"unified_scene_plan.json","unified_scene_plan.md","room_plan.json","camera_candidates.json","object_pose_report.json","object_pose_report.md","collision_report.json","confidence_report.json","ambiguity_report.md","clean_input_audit.json","forbidden_input_audit.json","compilation_report.json","compilation_report.md","blender_one_batch_manifest.json","numbered_object_overlay.png","normal_frames_overlay.png","projected_primitives_overlay.png","room_and_camera_overlay.png","confidence_overview.png","ambiguity_overview.png","clean_scene_plan_overview.png"}
    assert required<={x.name for x in first.iterdir()}
    plan=load(first/"unified_scene_plan.json")
    per_required={"sam_overlay.png","normal_clusters.png","horizontal_normal_frame.png","edge_family_visualization.png","projected_primitive.png","candidate_comparison.png","metrics.json"}
    for item in plan["semantic_objects"]:
        assert per_required<={x.name for x in (first/"per_object"/item["object_id"]).iterdir()}

