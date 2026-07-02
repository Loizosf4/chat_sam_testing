from src.unified_v3_scene_compiler import run_regression_audit, sha256
from tests.test_unified_v3_scene_compiler import load, unified_pair


def test_regression_is_separate_and_never_mutates_clean_result(unified_pair,tmp_path):
    first,_=unified_pair;before=sha256(first/"unified_scene_plan.json")
    report=run_regression_audit(first,tmp_path/"audit")
    assert report["mode"]=="regression_audit"
    assert report["clean_result_mutated"] is False
    assert report["clean_plan_unchanged"] is True
    assert sha256(first/"unified_scene_plan.json")==before
    assert len(report["differences"])==20
    assert {"regression_audit.json","regression_audit.md","approved_vs_clean_overlay.png","transform_difference_report.json"}<={x.name for x in (tmp_path/"audit").iterdir()}


def test_clean_plan_contains_no_regression_fields(unified_pair):
    first,_=unified_pair;plan=load(first/"unified_scene_plan.json")
    text=(first/"unified_scene_plan.json").read_text(encoding="utf-8").lower()
    assert plan["mode"]=="clean_reconstruction"
    assert "approved_transform" not in text
    assert "regression_audit" not in text
