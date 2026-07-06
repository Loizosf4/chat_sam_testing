from tests.test_unified_v3_scene_compiler import load, unified_pair
from src.support_contact_propagation import support_gap
import numpy as np
import pytest


def test_office_desktop_box_support_and_placement_regression(unified_pair):
    first,_=unified_pair;plan=load(first/"unified_scene_plan.json");objects={x["semantic_label"]:x for x in plan["semantic_objects"]};box=objects["desktop_box"];desk=objects["desk"]
    assert box["support_type"]=="object"
    assert box["support_target"]==desk["object_id"]
    assignment=plan["support_graph"]["assignments"][box["object_id"]]
    assert support_gap(desk,box,np.asarray(assignment["support_normal"]))==pytest.approx(0,abs=1e-6)
    assert box["transform"]["center"][2]==pytest.approx(.575981,abs=2e-4)
    assert box["validation_metrics"]["bbox_iou"]>.5
    assert box["validation_metrics"]["centroid_error_pixels"]<8
    assert box["placement_classification"]=="placement_high_confidence"
    assert all(c["placement"]["rotation_unchanged"] for c in box["rotation_candidates"])


def test_office_chair_is_occlusion_aware_visible_proxy(unified_pair):
    first,_=unified_pair;objects={x["semantic_label"]:x for x in load(first/"unified_scene_plan.json")["semantic_objects"]};chair=objects["desk_chair"]
    assert chair["occlusion"]["partially_occluded"]
    assert chair["support_type"]=="unknown"
    assert chair["placement_classification"]=="placement_with_occlusion"
    assert chair["validation_metrics"]["bbox_iou"]>.5
    assert chair["validation_metrics"]["centroid_error_pixels"]<8
    assert chair["transform"]["dimensions"][1]<.1
    assert all(c["placement"]["rotation_unchanged"] for c in chair["rotation_candidates"])
