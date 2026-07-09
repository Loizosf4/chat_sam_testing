"""Generic support graph inferred from cleaned geometry and masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _bounds(points: np.ndarray) -> tuple[np.ndarray,np.ndarray]: return np.percentile(points,[5,95],axis=0)


def _horizontal_overlap(subject: np.ndarray, support: np.ndarray) -> float:
    slo,shi=_bounds(subject);olo,ohi=_bounds(support);intersection=max(0,min(shi[0],ohi[0])-max(slo[0],olo[0]))*max(0,min(shi[1],ohi[1])-max(slo[1],olo[1]));area=max(1e-9,(shi[0]-slo[0])*(shi[1]-slo[1]));return float(intersection/area)


def mask_adjacency(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if (mask_a&mask_b).any(): return 0.0
    return float(ndimage.distance_transform_edt(~np.asarray(mask_a,bool))[np.asarray(mask_b,bool)].min())


def estimate_top_surface(item: dict[str,Any]) -> dict[str,Any]:
    points=np.asarray(item["points"]);normals=np.asarray(item["normals"]);up=normals[:,2]>=0.78
    if up.sum()>=20:
        candidates=points[up];candidate_normals=normals[up];cut=np.quantile(candidates[:,2],0.85);selected=candidates[:,2]>=cut;top_points=candidates[selected];top_normals=candidate_normals[selected];z=float(np.median(top_points[:,2]));confidence=float(min(1,up.mean()*2.5))
    else:
        z=float(np.percentile(points[:,2],92));selected=points[:,2]>=np.percentile(points[:,2],85);top_points=points[selected];top_normals=normals[selected];confidence=.3
    normal=np.median(top_normals,axis=0);normal=normal/max(1e-12,np.linalg.norm(normal))
    if normal[2]<0:normal*=-1
    lo,hi=_bounds(top_points)
    return {"z":z,"normal":normal.tolist(),"xy_bounds":[lo[:2].tolist(),hi[:2].tolist()],"confidence":confidence,"point_count":int(len(top_points))}


def semantic_support_compatibility(subject_label: str, support_label: str) -> float:
    support_tokens=("desk","table","cabinet","storage","shelf","box")
    subject_small=any(token in subject_label.lower() for token in ("box","light","object"))
    return .15 if subject_small and any(token in support_label.lower() for token in support_tokens) else 0.0


def infer_support_graph(objects: list[dict[str,Any]], wall_assignments: dict[str,dict[str,Any]] | None = None) -> dict[str,Any]:
    wall_assignments=wall_assignments or {};tops={x["object_id"]:estimate_top_surface(x) for x in objects};relationships=[];assignments={}
    for subject in objects:
        sid=subject["object_id"]
        if sid in wall_assignments and wall_assignments[sid]["type"]=="wall":assignments[sid]=wall_assignments[sid];continue
        slo,shi=_bounds(subject["points"]);bottom=float(slo[2]);candidates=[]
        for support in objects:
            if support is subject:continue
            top=tops[support["object_id"]];gap=bottom-top["z"];overlap=_horizontal_overlap(subject["points"],support["points"]);adjacency=mask_adjacency(subject["mask"],support["mask"]);depth_order=float(np.median(subject["depth"])-np.median(support["depth"]));height_score=max(0,1-abs(gap)/.18);overlap_score=min(1,overlap/.45);adjacency_score=max(0,1-adjacency/18);compat=semantic_support_compatibility(subject["semantic_label"],support["semantic_label"]);score=.38*height_score+.30*overlap_score+.12*adjacency_score+.12*top["confidence"]+.08*max(0,1-abs(depth_order)/.5)+compat
            candidates.append({"subject_object_id":sid,"support_object_id":support["object_id"],"support_label":support["semantic_label"],"score":float(min(1,score)),"bottom_to_top_distance":gap,"horizontal_overlap_ratio":overlap,"mask_adjacency_pixels":adjacency,"relative_depth":depth_order,"support_top":top})
        best=max(candidates,key=lambda x:x["score"]) if candidates else None
        if best is not None and best["score"]>=.62 and best["horizontal_overlap_ratio"]>=.08 and abs(best["bottom_to_top_distance"])<=.22:
            assignments[sid]={"target":best["support_object_id"],"type":"object","confidence":best["score"],"support_top_z":best["support_top"]["z"],"support_normal":best["support_top"]["normal"]};best["selected"]=True;relationships.append(best)
        elif bottom<=.10:
            assignments[sid]={"target":"plane_floor","type":"floor","confidence":float(max(.35,1-abs(bottom)/.12))}
        else:
            assignments[sid]={"target":None,"type":"unknown","confidence":.25}
    return {"assignments":assignments,"relationships":relationships,"top_surfaces":tops}
