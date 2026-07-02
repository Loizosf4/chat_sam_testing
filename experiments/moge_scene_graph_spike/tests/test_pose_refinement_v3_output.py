import json
from pathlib import Path

import jsonschema
import pytest

from src.evaluate_pose_refinement_v3 import ROOT, evaluate


REQUIRED_GLOBAL={
    "evaluation.json","evaluation.md","candidate_update_set.json","candidate_update_set.md",
    "regression_audit.json","regression_audit.md","normal_frame_summary.json",
    "normal_frame_summary.png","orientation_failure_analysis.md","candidate_contact_sheet.png",
    "recommended_candidates_overlay.png",
}
REQUIRED_PER_OBJECT={
    "source_mask_overlay.png","masked_normal_map.png","normal_cluster_visualization.png",
    "horizontal_normal_azimuth_plot.png","image_edge_families.png","candidate_comparison.png",
    "candidate_metrics.json","recommendation.md",
}


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root=tmp_path_factory.mktemp("pose_refinement_v3")
    first,second=root/"first",root/"second"
    evaluate(first); evaluate(second)
    return first,second


def test_v3_schema_and_exact_object_scope(generated):
    first,_=generated
    evaluation=load(first/"evaluation.json")
    jsonschema.validate(evaluation,load(ROOT/"schemas"/"pose_refinement_v3.schema.json"))
    assert len({x["object_id"] for x in evaluation["objects"]})==8
    assert all(2<=len(x["candidates"])<=4 for x in evaluation["objects"])


def test_v3_outputs_are_deterministic(generated):
    first,second=generated
    for name in ("evaluation.json","candidate_update_set.json","regression_audit.json","normal_frame_summary.json"):
        assert load(first/name)==load(second/name)


def test_v3_never_forces_candidate_bbox_fit_and_updates_are_gated(generated):
    first,_=generated
    evaluation=load(first/"evaluation.json"); updates=load(first/"candidate_update_set.json")
    assert all(x["no_per_candidate_bbox_fit"] for x in evaluation["objects"])
    allowed={x["object_id"] for x in evaluation["objects"] if x["classification"]=="apply_normal_refined_transform"}
    assert {x["object_id"] for x in updates["updates"]}==allowed
    for item in evaluation["objects"]:
        current=next(x for x in item["candidates"] if x["candidate_id"]=="current_blender")
        assert current["transform"]["center"]==item["current_blender_transform"]["center"]


def test_v3_preserves_protected_inputs_and_regressions(generated):
    first,_=generated
    evaluation=load(first/"evaluation.json"); regression=load(first/"regression_audit.json")
    assert evaluation["protected_inputs_unchanged"] and not evaluation["blender_modified"]
    assert regression["protected_inputs_unchanged"] and regression["all_approved_transforms_preserved"]
    assert {x["semantic_label"] for x in regression["objects"]}=={"desk","desk_chair","left_tall_filing_cabinet","desktop_box"}
    assert all(x["action"]=="audit_only_no_update" for x in regression["objects"])


def test_v3_required_visual_diagnostics_exist(generated):
    first,_=generated
    assert REQUIRED_GLOBAL<={x.name for x in first.iterdir()}
    evaluation=load(first/"evaluation.json")
    for item in evaluation["objects"]:
        folder=first/"per_object"/item["object_id"]
        assert REQUIRED_PER_OBJECT<={x.name for x in folder.iterdir()}
