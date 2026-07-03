"""Fixed-rotation mask/support fitting and occlusion-aware validation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image,ImageDraw
from scipy.optimize import minimize

from .mask_constrained_geometry import robust_extents_in_rotation
from .pose_refinement import cuboid_corners,project_world_points
from .support_graph import mask_adjacency


def detect_occlusion(subject: dict[str,Any], others: list[dict[str,Any]], adjacency_limit: float = 10.0) -> dict[str,Any]:
    candidates=[];subject_depth=float(np.median(subject["depth"]))
    for other in others:
        if other is subject:continue
        adjacency=mask_adjacency(subject["mask"],other["mask"]);other_depth=float(np.median(other["depth"]));depth_delta=subject_depth-other_depth
        if adjacency<=adjacency_limit and depth_delta>.05:candidates.append({"object_id":other["object_id"],"semantic_label":other["semantic_label"],"adjacency_pixels":adjacency,"foreground_depth_delta":depth_delta,"mask":other["mask"]})
    candidates.sort(key=lambda x:(x["adjacency_pixels"],-x["foreground_depth_delta"]));return {"partially_occluded":bool(candidates),"foreground_occluders":[{k:v for k,v in x.items() if k!="mask"} for x in candidates],"occluder_masks":[x["mask"] for x in candidates],"hidden_geometry_policy":"visible_surface_proxy; hidden lower volume remains unknown" if candidates else "none"}


def _hull(points:np.ndarray)->np.ndarray:
    pts=sorted(set(map(tuple,np.asarray(points,float))))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1])


def projection_masks(projected: np.ndarray, shape: tuple[int,int], occluder_masks: list[np.ndarray] | None = None) -> tuple[np.ndarray,np.ndarray]:
    hull=_hull(projected);image=Image.new("1",(shape[1],shape[0]));ImageDraw.Draw(image).polygon([tuple(x) for x in hull],fill=1);full=np.asarray(image,bool);visible=full.copy()
    for mask in occluder_masks or []:visible&=~np.asarray(mask,bool)
    return full,visible


def projection_metrics(projected: np.ndarray, mask: np.ndarray, occluder_masks: list[np.ndarray] | None = None) -> dict[str,Any]:
    full,visible=projection_masks(projected,mask.shape,occluder_masks);y,x=np.nonzero(mask);target=np.asarray([x.min(),y.min(),x.max()+1,y.max()+1],float);box=np.asarray([projected[:,0].min(),projected[:,1].min(),projected[:,0].max(),projected[:,1].max()]);iw=max(0,min(box[2],target[2])-max(box[0],target[0]));ih=max(0,min(box[3],target[3])-max(box[1],target[1]));inter=iw*ih;area=(box[2]-box[0])*(box[3]-box[1]);target_area=(target[2]-target[0])*(target[3]-target[1])
    def iou(pred):return float((pred&mask).sum()/max(1,(pred|mask).sum()))
    return {"bbox_iou":float(inter/max(1e-9,area+target_area-inter)),"full_hull_iou":iou(full),"visible_hull_iou":iou(visible),"centroid_error_pixels":float(np.linalg.norm((box[:2]+box[2:]-target[:2]-target[2:])/2)),"projected_bbox":box.tolist(),"predicted_full_pixels":int(full.sum()),"predicted_visible_pixels":int(visible.sum()),"occluded_projected_pixels":int((full&~visible).sum())}


def placement_hard_gates(metrics: dict[str,Any], scale_ratio: np.ndarray, unsupported_floating: bool = False, below_support: bool = False) -> list[str]:
    gates=[]
    if metrics["bbox_iou"]<.30:gates.append("bbox_iou_below_0.30")
    if metrics["centroid_error_pixels"]>15:gates.append("centroid_error_above_15px")
    if (np.asarray(scale_ratio)<.6).any() or (np.asarray(scale_ratio)>1.4).any():gates.append("severe_scale_inconsistency")
    if unsupported_floating:gates.append("unsupported_floating_object")
    if below_support:gates.append("below_support_surface")
    return gates


def fit_position_dimensions_fixed_rotation(points_world: np.ndarray, rotation: np.ndarray, mask: np.ndarray, camera: dict[str,Any], support: dict[str,Any], occluder_masks: list[np.ndarray] | None = None) -> dict[str,Any]:
    rotation=np.asarray(rotation,float).copy();rotation_before=rotation.copy();center0,dims0,_=robust_extents_in_rotation(points_world,rotation,5,95);support_type=support.get("type","unknown")
    if support_type=="floor":center0[2]=dims0[2]/2
    elif support_type=="object":center0[2]=float(support["support_top_z"])+dims0[2]/2
    local=np.asarray(points_world)@rotation
    y,x=np.nonzero(mask);target_center=np.asarray([x.mean(),y.mean()]);target_size=np.asarray([x.max()-x.min()+1,y.max()-y.min()+1],float)
    def project(values):
        center=values[:3];dims=np.maximum(values[3:],.015);return project_world_points(cuboid_corners(center,dims,rotation),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"])
    def objective(values):
        center=values[:3];dims=np.maximum(values[3:],.015);projected=project(values);box=np.asarray([projected[:,0].min(),projected[:,1].min(),projected[:,0].max(),projected[:,1].max()]);pc=(box[:2]+box[2:])/2;ps=box[2:]-box[:2];center_loss=np.sum(((pc-target_center)/20)**2);size_loss=np.sum(np.log(np.maximum(ps,1)/target_size)**2);local_center=center@rotation;lower=local_center-dims/2;upper=local_center+dims/2;outside=np.maximum(lower-local,0)+np.maximum(local-upper,0);point_loss=float(np.mean((outside/np.maximum(dims,.02))**2));regularization=float(np.sum(np.log(dims/dims0)**2));support_loss=0.0
        if support_type=="floor":support_loss=((center[2]-dims[2]/2)/.04)**2
        elif support_type=="object":support_loss=((center[2]-dims[2]/2-float(support["support_top_z"]))/ .04)**2
        return .28*point_loss+.25*center_loss+.18*size_loss+.14*support_loss+.15*regularization
    initial=np.r_[center0,dims0];bounds=[(center0[i]-.25,center0[i]+.25) for i in range(3)]+[(max(.015,dims0[i]*.55),dims0[i]*1.45) for i in range(3)];result=minimize(objective,initial,method="L-BFGS-B",bounds=bounds,options={"maxiter":80,"ftol":1e-10})
    values=result.x;center=values[:3];dims=np.maximum(values[3:],.015)
    if support_type=="floor":center[2]=dims[2]/2
    elif support_type=="object":center[2]=float(support["support_top_z"])+dims[2]/2
    metrics=projection_metrics(project(np.r_[center,dims]),mask,occluder_masks)
    scale_ratio=dims/np.maximum(dims0,1e-9);floating=support_type=="unknown" and not occluder_masks and center[2]-dims[2]/2>.15;below=support_type=="object" and center[2]-dims[2]/2<float(support["support_top_z"])-1e-6;gates=placement_hard_gates(metrics,scale_ratio,floating,below)
    placement="placement_invalid" if gates else "placement_with_occlusion" if occluder_masks else "placement_high_confidence" if metrics["bbox_iou"]>=.55 and metrics["centroid_error_pixels"]<=8 else "placement_review_required"
    return {"center":center,"dimensions":dims,"rotation":rotation,"rotation_unchanged":bool(np.array_equal(rotation,rotation_before)),"initial_center":center0,"initial_dimensions":dims0,"optimizer":{"success":bool(result.success),"iterations":int(result.nit),"objective":float(result.fun)},"metrics":metrics,"scale_ratio":scale_ratio.tolist(),"placement_classification":placement,"hard_gates":gates,"support_contact_error":0.0 if support_type in ("floor","object") else None}
