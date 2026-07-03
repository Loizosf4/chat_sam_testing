import numpy as np
import pytest

from src.occlusion_aware_fitting import (
    detect_occlusion, fit_position_dimensions_fixed_rotation,
    placement_hard_gates, projection_masks,
)
from src.pose_refinement import cuboid_corners, project_world_points
from src.pose_refinement_v3 import yaw_rotation


CAMERA={"canonical_to_camera":np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,5],[0,0,0,1]],float),"intrinsics":np.asarray([[1,0,.5],[0,1,.5],[0,0,1]],float),"image_shape":(100,100)}


def cuboid_surface(center,dims,rotation,count=9):
    local=np.random.default_rng(2).uniform(-.5,.5,(count**2,3))*dims
    return local@rotation.T+center


def mask_for(center,dims,rotation):
    projected=project_world_points(cuboid_corners(np.asarray(center),np.asarray(dims),rotation),CAMERA["canonical_to_camera"],CAMERA["intrinsics"],CAMERA["image_shape"])
    return projection_masks(projected,(100,100))[0]


def test_fixed_rotation_mask_constrained_scale_recovery():
    rotation=yaw_rotation(31);center=np.asarray([0.2,-.1,.35]);dims=np.asarray([.5,.3,.7]);points=cuboid_surface(center,dims,rotation,20);mask=mask_for(center,dims,rotation)
    result=fit_position_dimensions_fixed_rotation(points,rotation,mask,CAMERA,{"type":"floor","target":"plane_floor","confidence":1})
    np.testing.assert_array_equal(result["rotation"],rotation)
    assert result["rotation_unchanged"]
    assert result["metrics"]["bbox_iou"]>.5
    assert abs(result["center"][2]-result["dimensions"][2]/2)<1e-9


def test_partially_occluded_proxy_uses_visible_projection():
    subject_mask=np.zeros((60,60),bool);subject_mask[10:35,25:35]=1;foreground=np.zeros_like(subject_mask);foreground[30:50,10:50]=1
    points=np.tile([0,0,6.0],(100,1));normals=np.tile([1,0,0],(100,1))
    subject={"object_id":"chair","semantic_label":"chair","mask":subject_mask,"depth":np.full(100,6.0),"points":points,"normals":normals}
    other={"object_id":"desk","semantic_label":"desk","mask":foreground,"depth":np.full(100,5.0),"points":points,"normals":normals}
    evidence=detect_occlusion(subject,[subject,other])
    assert evidence["partially_occluded"]
    assert evidence["foreground_occluders"][0]["object_id"]=="desk"
    projected=np.asarray([[20,10],[40,10],[40,45],[20,45]],float)
    full,visible=projection_masks(projected,subject_mask.shape,[foreground])
    assert visible.sum()<full.sum()
    assert np.any(full & foreground)


def test_unsupported_object_is_not_floor_snapped():
    rotation=np.eye(3);center=np.asarray([0,0,1.2]);dims=np.asarray([.3,.2,.2]);points=cuboid_surface(center,dims,rotation,20);mask=mask_for(center,dims,rotation)
    result=fit_position_dimensions_fixed_rotation(points,rotation,mask,CAMERA,{"type":"unknown","target":None,"confidence":.2})
    assert result["center"][2]-result["dimensions"][2]/2>.5


def test_wall_supported_object_keeps_wall_contact_during_mask_fit():
    rotation=np.asarray([[1,0,0],[0,0,1],[0,-1,0]],float);center=np.asarray([0,.12,.6]);dims=np.asarray([.5,.12,.7]);points=cuboid_surface(center,dims,rotation,20);mask=mask_for(center,dims,rotation)
    wall={"plane_equation":{"normal":[0,1,0],"offset":0.0}}
    result=fit_position_dimensions_fixed_rotation(points,rotation,mask,CAMERA,{"type":"wall","target":"plane_wall","confidence":1,"wall":wall})
    np.testing.assert_array_equal(result["rotation"],rotation)
    assert np.asarray(wall["plane_equation"]["normal"])@result["center"]==pytest.approx(result["dimensions"][2]/2,abs=1e-8)
    assert result["support_contact_error"]==0


def test_high_iou_but_incorrect_scale_is_rejected():
    metrics={"bbox_iou":.95,"centroid_error_pixels":1.0}
    gates=placement_hard_gates(metrics,np.asarray([1.0,1.8,1.0]))
    assert "severe_scale_inconsistency" in gates
