import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src.unified_v3_scene_compiler import ROOT, compile_clean


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def unified_pair(tmp_path_factory):
    root=tmp_path_factory.mktemp("unified_v3")
    first,second=root/"first",root/"second"
    compile_clean(first,handoff=root/"handoff_first")
    compile_clean(second,handoff=root/"handoff_second")
    return first,second


def test_universal_v3_invocation_and_exact_scope(unified_pair):
    first,_=unified_pair; plan=load(first/"unified_scene_plan.json")
    assert plan["semantic_object_count"]==20
    assert len({x["object_id"] for x in plan["semantic_objects"]})==20
    assert all(x["orientation_method"]=="normal_first_v3_universal" for x in plan["semantic_objects"])
    assert all(x["normal_frame"]["numeric_normal_count"]>0 for x in plan["semantic_objects"])
    assert plan["room_surface_semantic_count"]==0


def test_unified_plan_schema_and_transform_quality(unified_pair):
    first,_=unified_pair; plan=load(first/"unified_scene_plan.json")
    jsonschema.validate(plan,load(ROOT/"schemas"/"unified_v3_scene_plan.schema.json"))
    for item in plan["semantic_objects"]:
        transform=item["transform"]; rotation=np.asarray(transform["rotation_matrix"],float)
        assert np.isfinite(np.asarray(transform["center"])).all()
        assert (np.asarray(transform["dimensions"])>0).all()
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
        assert np.linalg.det(rotation)==pytest.approx(1,abs=1e-6)


def test_generic_floor_and_wall_constraints(unified_pair):
    first,_=unified_pair; plan=load(first/"unified_scene_plan.json"); planes={x["plane_id"]:x for x in plan["room_proxies"]}
    for item in plan["semantic_objects"]:
        center=np.asarray(item["transform"]["center"]);dims=np.asarray(item["transform"]["dimensions"]);rotation=np.asarray(item["transform"]["rotation_matrix"])
        if item["support_type"]=="floor":
            assert center[2]-dims[2]/2==pytest.approx(0,abs=1e-6)
        if item["support_type"]=="wall":
            plane=planes[item["support_target"]];normal=np.asarray(plane["plane_equation"]["normal"]);offset=plane["plane_equation"]["offset"]
            assert normal@center+offset==pytest.approx(dims[2]/2,abs=1e-6)
            assert abs(normal@rotation[:,2])==pytest.approx(1,abs=1e-6)


def test_natural_projection_is_not_forced_to_perfect_bbox(unified_pair):
    first,_=unified_pair; objects=load(first/"unified_scene_plan.json")["semantic_objects"]
    ious=[x["validation_metrics"]["bbox_iou"] for x in objects]
    assert sum(value>.99 for value in ious)<5
    assert any(value<.5 for value in ious)
    assert all("scale_fit" not in json.dumps(x).lower() for x in objects)


def test_blender_manifest_is_complete_and_approval_free(unified_pair):
    first,_=unified_pair; manifest=load(first/"blender_one_batch_manifest.json")
    assert manifest["semantic_primitive_count"]==20
    assert len(manifest["semantic_primitives"])==20
    assert len(manifest["room_proxies"])==3
    assert manifest["approved_artifact_references"]==[]
    assert all(x["object_id"] and x["semantic_label"] for x in manifest["semantic_primitives"])

