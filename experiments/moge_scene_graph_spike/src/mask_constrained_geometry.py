"""Mask-interior and depth-coherent geometry extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def erode_object_mask(mask: np.ndarray, width: int = 2, minimum_ratio: float = 0.35) -> tuple[np.ndarray, dict[str, Any]]:
    mask=np.asarray(mask,bool)
    eroded=ndimage.binary_erosion(mask,iterations=width,border_value=0)
    fallback=not eroded.any() or eroded.sum()/max(1,mask.sum())<minimum_ratio
    selected=mask if fallback else eroded
    return selected,{"boundary_width":width,"original_pixels":int(mask.sum()),"interior_pixels":int(selected.sum()),"retained_ratio":float(selected.sum()/max(1,mask.sum())),"fallback_to_full_mask":fallback}


def depth_clusters(depths: np.ndarray, maximum_gap: float | None = None) -> list[np.ndarray]:
    depths=np.asarray(depths,float)
    order=np.argsort(depths);sorted_depth=depths[order]
    mad=float(np.median(np.abs(sorted_depth-np.median(sorted_depth))))
    threshold=maximum_gap if maximum_gap is not None else float(np.clip(0.45*1.4826*mad,0.025,0.075))
    splits=np.flatnonzero(np.diff(sorted_depth)>threshold)+1
    return [chunk for chunk in np.split(order,splits) if len(chunk)]


def clean_masked_geometry(mask: np.ndarray, points: np.ndarray, depth: np.ndarray, normals: np.ndarray, valid: np.ndarray, boundary_width: int = 2) -> dict[str, Any]:
    interior,erosion=erode_object_mask(mask,boundary_width)
    selection=interior&np.asarray(valid,bool)&np.isfinite(points).all(axis=2)&np.isfinite(depth)&np.isfinite(normals).all(axis=2)
    y,x=np.nonzero(selection);raw_points=np.asarray(points)[selection];raw_depth=np.asarray(depth)[selection];raw_normals=np.asarray(normals)[selection]
    if len(raw_points)<12:
        return {"success":False,"reason":"fewer than 12 valid interior samples","selection_mask":selection,"erosion":erosion}
    clusters=depth_clusters(raw_depth)
    median=float(np.median(raw_depth));ranked=sorted(clusters,key=lambda idx:(-len(idx),abs(float(np.median(raw_depth[idx]))-median)))
    chosen=ranked[0];cluster_depth=raw_depth[chosen];cluster_median=float(np.median(cluster_depth));mad=float(np.median(np.abs(cluster_depth-cluster_median)));limit=max(0.025,4.5*1.4826*mad)
    keep_local=np.abs(cluster_depth-cluster_median)<=limit;indices=chosen[keep_local]
    cleaned_mask=np.zeros(mask.shape,bool);cleaned_mask[y[indices],x[indices]]=True
    return {"success":True,"points":raw_points[indices],"depth":raw_depth[indices],"normals":raw_normals[indices],"pixel_yx":np.column_stack([y[indices],x[indices]]),"selection_mask":cleaned_mask,"erosion":erosion,"depth_cluster_count":len(clusters),"selected_cluster_samples":int(len(chosen)),"cleaned_samples":int(len(indices)),"rejected_samples":int(len(raw_points)-len(indices)),"depth_median":cluster_median,"depth_mad":mad,"depth_limit":limit}


def robust_extents_in_rotation(points_world: np.ndarray, rotation: np.ndarray, lower: float = 5, upper: float = 95) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    local=np.asarray(points_world,float)@np.asarray(rotation,float)
    lo,hi=np.percentile(local,[lower,upper],axis=0);dimensions=np.maximum(hi-lo,0.02);center_local=(lo+hi)/2;center=center_local@np.asarray(rotation,float).T
    return center,dimensions,np.column_stack([lo,hi])
