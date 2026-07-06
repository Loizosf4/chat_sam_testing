import copy

import numpy as np
import pytest

from src.support_contact_propagation import (
    oriented_box_extent,
    propagate_support_contacts,
    support_face_normal,
    support_gap,
    topological_support_order,
)


def transform(center, dimensions, rotation=None):
    return {"center": list(center), "dimensions": list(dimensions), "rotation_matrix": np.eye(3).tolist() if rotation is None else np.asarray(rotation).tolist()}


def obj(object_id, center, dimensions, rotation=None):
    return {"object_id": object_id, "transform": transform(center, dimensions, rotation)}


def test_parent_resized_after_child_placement_is_propagated():
    parent=obj("parent",[0,0,.75],[2,2,1.5]);child=obj("child",[0,0,1.25],[.4,.4,.5])
    assignments={"parent":{"type":"floor","target":"plane_floor"},"child":{"type":"object","target":"parent","support_normal":[0,0,1]}}
    before=copy.deepcopy(child["transform"]);report=propagate_support_contacts([parent,child],assignments)
    assert report["records"][0]["gap_before"]==pytest.approx(-.5)
    assert support_gap(parent,child,np.asarray([0,0,1]))==pytest.approx(0,abs=1e-12)
    assert child["transform"]["dimensions"]==before["dimensions"]
    assert child["transform"]["rotation_matrix"]==before["rotation_matrix"]


def test_rotated_support_uses_general_obb_extent_formula():
    angle=np.deg2rad(31);rotation=np.asarray([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    normal=np.asarray([1.0,0.0,1.0]);normal/=np.linalg.norm(normal)
    parent=obj("parent",[.2,-.1,.4],[1.4,.8,.6],rotation);child=obj("child",[1.3,-.1,1.5],[.3,.5,.4],rotation)
    expected=sum(abs(normal@rotation[:,i])*parent["transform"]["dimensions"][i]/2 for i in range(3))
    assert oriented_box_extent(parent["transform"],normal)==pytest.approx(expected)
    face_normal=rotation[:,2]
    np.testing.assert_allclose(support_face_normal(parent["transform"],face_normal+.01*rotation[:,0]),face_normal,atol=1e-12)
    assignments={"parent":{"type":"unknown","target":None},"child":{"type":"object","target":"parent","support_normal":face_normal.tolist()}}
    propagate_support_contacts([parent,child],assignments)
    assert support_gap(parent,child,face_normal)==pytest.approx(0,abs=1e-12)


def test_multi_level_chain_updates_descendants_in_topological_order():
    parent=obj("a",[0,0,.5],[1,1,1]);middle=obj("b",[0,0,1.2],[.6,.6,.4]);child=obj("c",[0,0,1.7],[.2,.2,.2])
    assignments={"c":{"type":"object","target":"b","support_normal":[0,0,1]},"b":{"type":"object","target":"a","support_normal":[0,0,1]},"a":{"type":"floor","target":"plane_floor"}}
    assert topological_support_order(["c","b","a"],assignments)==["a","b","c"]
    report=propagate_support_contacts([child,middle,parent],assignments)
    assert [record["child_object_id"] for record in report["records"]]==["b","c"]
    assert support_gap(parent,middle,np.asarray([0,0,1]))==pytest.approx(0,abs=1e-12)
    assert support_gap(middle,child,np.asarray([0,0,1]))==pytest.approx(0,abs=1e-12)


def test_propagation_is_deterministic_and_preserves_unsupported_objects():
    objects=[obj("parent",[0,0,.5],[1,1,1]),obj("child",[0,0,1.4],[.3,.3,.3]),obj("free",[4,5,6],[.7,.8,.9])]
    assignments={"parent":{"type":"floor","target":"plane_floor"},"child":{"type":"object","target":"parent","support_normal":[0,0,1]},"free":{"type":"unknown","target":None}}
    first=copy.deepcopy(objects);second=copy.deepcopy(objects);free_before=copy.deepcopy(objects[2]["transform"])
    first_report=propagate_support_contacts(first,copy.deepcopy(assignments));second_report=propagate_support_contacts(second,copy.deepcopy(assignments))
    assert first==second
    assert first_report==second_report
    assert first[2]["transform"]==free_before
    assert first_report["maximum_final_gap"]<1e-6
