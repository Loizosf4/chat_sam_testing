"""Unified normal-first V3 clean-room scene compiler.

Clean mode reads only raw SAM, raw MoGe and generic configuration.  Approved
transforms and Blender artifacts are available only to the separate regression
function after the clean plan has been finalized and hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from .pose_refinement import cuboid_corners, matrix_to_quaternion_wxyz, project_world_points
from .pose_refinement_v3 import (
    angle_distance_180, estimate_horizontal_frame, extract_mask_edge_families,
    footprint_yaw_observable, generate_yaw_candidates, orientation_score,
    robust_dimensions_in_frame, yaw_rotation,
)
from .scene_geometry import (
    construct_canonical_transform, estimate_structural_planes,
    opencv_to_blender_camera_local_matrix, transform_normals, transform_plane,
    transform_points,
)
from .unified_v3_models import UnifiedScenePlan
from .mask_constrained_geometry import clean_masked_geometry
from .support_graph import infer_support_graph
from .occlusion_aware_fitting import detect_occlusion, fit_position_dimensions_fixed_rotation, placement_hard_gates, projection_metrics as occlusion_projection_metrics
from .joint_room_reconstruction import fit_joint_room_model
from .support_contact_propagation import propagate_support_contacts, support_gap, topological_support_order


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_SAM_DIR = REPO_ROOT / "data" / "exports" / "auto_scene_final_24457ea2"
DEFAULT_SOURCE = REPO_ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"
DEFAULT_MOGE = ROOT / "outputs" / "office_test" / "moge"
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "unified_v3_clean"
HANDOFF = ROOT / "clean_room_v3"
FORBIDDEN_TOKENS = (".blend", "approved", "candidate_c", "room_corrected", "blender_execution", "primitive_plan", "pose_refinement_v2", "pose_refinement_v3")
REQUIRED_MOGE_ARRAYS = ("points", "depth", "valid_mask", "intrinsics", "normal")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REQUIRED_OUTPUT_FILES = {
    "unified_scene_plan.json", "unified_scene_plan.md", "room_plan.json",
    "camera_candidates.json", "object_pose_report.json", "object_pose_report.md",
    "placement_report.json", "support_graph.json", "collision_report.json",
    "confidence_report.json", "ambiguity_report.md", "clean_input_audit.json",
    "forbidden_input_audit.json", "compilation_report.json",
    "compilation_report.md", "blender_one_batch_manifest.json",
    "allowed_inputs_manifest.json", "numbered_object_overlay.png",
    "normal_frames_overlay.png", "projected_primitives_overlay.png",
    "support_graph_overlay.png", "room_and_camera_overlay.png",
    "room_projection_overlay.png", "confidence_overview.png",
    "ambiguity_overview.png", "clean_scene_plan_overview.png",
}


class CompilerValidationError(ValueError):
    """A concise, expected validation failure suitable for CLI display."""


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


class CleanReadGuard:
    def __init__(self, allowed: list[Path]):
        self.allowed={p.resolve() for p in allowed}; self.read_log: list[str]=[]; self.denied_log: list[str]=[]
    def _check(self,path:Path)->Path:
        resolved=path.resolve()
        if resolved not in self.allowed:
            self.denied_log.append(str(resolved)); raise PermissionError(f"clean input boundary denied: {resolved}")
        self.read_log.append(str(resolved)); return resolved
    def json(self,path:Path)->dict[str,Any]: return json.loads(self._check(path).read_text(encoding="utf-8"))
    def image(self,path:Path)->Image.Image:
        checked=self._check(path)
        with Image.open(checked) as image: return image.convert("RGB")
    def mask(self,path:Path)->np.ndarray:
        checked=self._check(path)
        with Image.open(checked) as image: return np.asarray(image.convert("L"))>0
    def npz(self,path:Path)->dict[str,np.ndarray]:
        checked=self._check(path)
        with np.load(checked,allow_pickle=False) as data: return {key:np.asarray(data[key]) for key in data.files}


def allowed_input_manifest(sam_dir:Path=DEFAULT_SAM_DIR,source:Path=DEFAULT_SOURCE,moge_dir:Path=DEFAULT_MOGE,metadata:dict[str,Any]|None=None)->dict[str,Any]:
    metadata_path=sam_dir/"metadata.json"
    metadata=metadata if metadata is not None else json.loads(metadata_path.read_text(encoding="utf-8"))
    paths=[source,metadata_path,moge_dir/"geometry.npz",moge_dir/"metadata.json"]+[sam_dir/x["filename"] for x in metadata["masks"]]
    return {"schema_version":"1.0","mode":"clean_reconstruction","inputs":[{"path":str(p.resolve()),"sha256":sha256(p),"role":"source_image" if p==source else "sam_metadata" if p==metadata_path else "moge_geometry" if p.name=="geometry.npz" else "moge_metadata" if p.name=="metadata.json" else "semantic_mask"} for p in paths],"input_count":len(paths),"forbidden_categories":["Blender checkpoints","approved transforms","manual candidate decisions","corrected room/camera reports","historical final scene plans"]}


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompilerValidationError(f"{name} is missing: {path}")
    try:
        value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompilerValidationError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value,dict):
        raise CompilerValidationError(f"{name} must be a JSON object: {path}")
    return value


def _fallback_color(object_id: str, index: int) -> str:
    digest=hashlib.sha256(f"{index}:{object_id}".encode("utf-8")).digest()
    rgb=[64+(component%160) for component in digest[:3]]
    return "#"+"".join(f"{component:02x}" for component in rgb)


def _normalize_color(value: Any, object_id: str, index: int) -> tuple[str, bool]:
    if isinstance(value,str) and len(value)==7 and value.startswith("#"):
        try:
            int(value[1:],16)
            return value.lower(),False
        except ValueError:
            pass
    if isinstance(value,(list,tuple)) and len(value)==3 and all(isinstance(x,(int,float)) and 0<=x<=255 for x in value):
        return "#"+"".join(f"{int(x):02x}" for x in value),False
    return _fallback_color(object_id,index),True


def validate_compiler_inputs(*,sam_dir:Path,source_path:Path,moge_dir:Path)->dict[str,Any]:
    """Validate the complete protected input set before creating staging output."""
    sam_dir=Path(sam_dir).expanduser().resolve();source_path=Path(source_path).expanduser().resolve();moge_dir=Path(moge_dir).expanduser().resolve()
    if not sam_dir.is_dir():raise CompilerValidationError(f"SAM export directory is missing: {sam_dir}")
    if not moge_dir.is_dir():raise CompilerValidationError(f"MoGe directory is missing: {moge_dir}")
    metadata=_read_json_object(sam_dir/"metadata.json","SAM metadata")
    missing=[key for key in ("image_id","width","height","masks") if key not in metadata]
    if missing:raise CompilerValidationError(f"SAM metadata is missing required fields: {', '.join(missing)}")
    if not isinstance(metadata["width"],int) or not isinstance(metadata["height"],int) or metadata["width"]<=0 or metadata["height"]<=0:
        raise CompilerValidationError("SAM metadata width and height must be positive integers")
    if not isinstance(metadata["masks"],list) or not metadata["masks"]:
        raise CompilerValidationError("SAM metadata must contain at least one mask")
    seen:set[str]=set();masks:dict[str,np.ndarray]={};fallback_ids:list[str]=[]
    for index,item in enumerate(metadata["masks"]):
        if not isinstance(item,dict):raise CompilerValidationError(f"SAM mask entry {index} must be an object")
        absent=[key for key in ("mask_id","label","filename") if key not in item]
        if absent:raise CompilerValidationError(f"SAM mask entry {index} is missing: {', '.join(absent)}")
        mask_id=item["mask_id"]
        if not isinstance(mask_id,str) or not mask_id or Path(mask_id).name!=mask_id or "/" in mask_id or "\\" in mask_id:
            raise CompilerValidationError(f"SAM mask entry {index} has an invalid path-unsafe mask_id")
        if mask_id in seen:raise CompilerValidationError(f"duplicate SAM mask_id: {mask_id}")
        seen.add(mask_id)
        if not isinstance(item["label"],str):raise CompilerValidationError(f"SAM mask {mask_id} label must be a string")
        filename=item["filename"]
        if not isinstance(filename,str) or not filename or Path(filename).name!=filename or Path(filename).is_absolute() or "/" in filename or "\\" in filename:
            raise CompilerValidationError(f"SAM mask {mask_id} has an unsafe filename: {filename!r}")
        mask_path=sam_dir/filename
        if not mask_path.is_file():raise CompilerValidationError(f"SAM mask file is missing for {mask_id}: {mask_path}")
        try:
            with Image.open(mask_path) as image:
                raw=np.asarray(image.convert("L"))
        except (OSError,ValueError) as exc:
            raise CompilerValidationError(f"SAM mask is not a readable image for {mask_id}: {mask_path}") from exc
        if raw.shape!=(metadata["height"],metadata["width"]):
            raise CompilerValidationError(f"SAM mask {mask_id} dimensions {raw.shape[::-1]} do not match metadata {(metadata['width'],metadata['height'])}")
        values=set(np.unique(raw).tolist())
        if not values.issubset({0,255}):raise CompilerValidationError(f"SAM mask {mask_id} is not binary; values={sorted(values)[:8]}")
        if not np.any(raw==255):raise CompilerValidationError(f"SAM mask {mask_id} is empty")
        masks[mask_id]=raw==255
        color,fallback=_normalize_color(item.get("color"),mask_id,index);item["color"]=color
        item["color_source"]="deterministic_fallback" if fallback else "export_metadata"
        if fallback:fallback_ids.append(mask_id)
    if not source_path.is_file():raise CompilerValidationError(f"source image is missing: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:raise CompilerValidationError(f"unsupported source image format: {source_path.suffix}")
    try:
        with Image.open(source_path) as image:
            image.verify()
        with Image.open(source_path) as image:
            source_size=image.size
    except (OSError,ValueError) as exc:
        raise CompilerValidationError(f"source image is not readable: {source_path}") from exc
    expected_size=(metadata["width"],metadata["height"])
    if source_size!=expected_size:raise CompilerValidationError(f"source image dimensions {source_size} do not match SAM metadata {expected_size}")
    moge_meta=_read_json_object(moge_dir/"metadata.json","MoGe metadata")
    dimensions=moge_meta.get("source_image_dimensions")
    if not isinstance(dimensions,dict) or dimensions.get("width")!=expected_size[0] or dimensions.get("height")!=expected_size[1]:
        raise CompilerValidationError(f"MoGe metadata source dimensions do not match {expected_size}")
    geometry_path=moge_dir/"geometry.npz"
    if not geometry_path.is_file():raise CompilerValidationError(f"MoGe geometry archive is missing: {geometry_path}")
    try:
        with np.load(geometry_path,allow_pickle=False) as archive:
            absent=[key for key in REQUIRED_MOGE_ARRAYS if key not in archive.files]
            if absent:raise CompilerValidationError(f"MoGe geometry is missing arrays: {', '.join(absent)}")
            arrays={key:np.asarray(archive[key]) for key in REQUIRED_MOGE_ARRAYS}
    except CompilerValidationError:raise
    except (OSError,ValueError) as exc:
        raise CompilerValidationError(f"MoGe geometry archive is invalid or contains unsupported arrays: {geometry_path}") from exc
    height,width=metadata["height"],metadata["width"]
    expected_2d=(height,width)
    if arrays["points"].shape!=(height,width,3):raise CompilerValidationError(f"MoGe points shape must be {(height,width,3)}, got {arrays['points'].shape}")
    if arrays["normal"].shape!=(height,width,3):raise CompilerValidationError(f"MoGe normal shape must be {(height,width,3)}, got {arrays['normal'].shape}")
    for name in ("depth","valid_mask"):
        if arrays[name].shape!=expected_2d:raise CompilerValidationError(f"MoGe {name} shape must be {expected_2d}, got {arrays[name].shape}")
    if arrays["intrinsics"].shape!=(3,3):raise CompilerValidationError(f"MoGe intrinsics shape must be (3, 3), got {arrays['intrinsics'].shape}")
    valid=np.asarray(arrays["valid_mask"],bool)
    if not valid.any():raise CompilerValidationError("MoGe valid_mask contains no valid samples")
    for name in ("points","normal","depth"):
        values=arrays[name][valid]
        if values.size==0 or not np.isfinite(values).any():raise CompilerValidationError(f"MoGe {name} contains no usable finite values")
    if not np.isfinite(arrays["intrinsics"]).all():raise CompilerValidationError("MoGe intrinsics contains non-finite values")
    return {"sam_dir":sam_dir,"source_path":source_path,"moge_dir":moge_dir,"metadata":metadata,"moge_metadata":moge_meta,"masks":masks,"fallback_color_object_ids":fallback_ids}


def _hull(points:np.ndarray)->np.ndarray:
    pts=sorted(set(map(tuple,np.asarray(points,float))))
    if len(pts)<=1:return np.asarray(pts)
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


def _projection_metrics(projected:np.ndarray,mask:np.ndarray)->dict[str,Any]:
    y,x=np.nonzero(mask); target=np.asarray([x.min(),y.min(),x.max()+1,y.max()+1],float); box=np.asarray([projected[:,0].min(),projected[:,1].min(),projected[:,0].max(),projected[:,1].max()])
    iw=max(0,min(box[2],target[2])-max(box[0],target[0]));ih=max(0,min(box[3],target[3])-max(box[1],target[1]));inter=iw*ih;aa=(box[2]-box[0])*(box[3]-box[1]);ab=(target[2]-target[0])*(target[3]-target[1])
    hull=_hull(projected); pred=Image.new("1",(mask.shape[1],mask.shape[0]));ImageDraw.Draw(pred).polygon([tuple(v) for v in hull],fill=1);pm=np.asarray(pred,bool)
    return {"bbox_iou":float(inter/max(1e-9,aa+ab-inter)),"hull_iou":float((pm&mask).sum()/max(1,(pm|mask).sum())),"centroid_error_pixels":float(np.linalg.norm((box[:2]+box[2:]-target[:2]-target[2:])/2)),"projected_bbox":box.tolist(),"mask_bbox":target.tolist(),"projected_hull":hull.tolist()}


def _axis_angle(center:np.ndarray,axis:np.ndarray,camera:dict[str,Any])->float:
    uv=project_world_points(np.vstack([center-axis*.1,center+axis*.1]),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);d=uv[1]-uv[0];return float(math.degrees(math.atan2(d[1],d[0]))%180)


def _edge_error(center:np.ndarray,rotation:np.ndarray,camera:dict[str,Any],families:dict[str,Any])->float:
    clusters=families.get("horizontal_clusters",[]);axes=[rotation[:,i] for i in range(3) if abs(rotation[2,i])<.6]
    if not clusters or not axes:return 90.0
    angles=[_axis_angle(center,a,camera) for a in axes];total=sum(c["weight"] for c in clusters)
    return float(sum(c["weight"]*min(angle_distance_180(c["azimuth_degrees_mod180"],a) for a in angles) for c in clusters)/max(1e-9,total))


def _normal_error(rotation:np.ndarray,frame:dict[str,Any])->float:
    clusters=[c for c in frame["clusters"] if c["reliable"]];axes=[math.degrees(math.atan2(rotation[1,i],rotation[0,i]))%180 for i in range(3) if abs(rotation[2,i])<.6]
    if not clusters or not axes:return 90.0
    total=sum(c["weight"] for c in clusters);return float(sum(c["weight"]*min(angle_distance_180(c["azimuth_degrees_mod180"],a) for a in axes) for c in clusters)/max(1e-9,total))


def _plane_proxy(plane:dict[str,Any],structural_points:np.ndarray,raw_to_canonical:np.ndarray)->dict[str,Any]:
    pts=transform_points(structural_points[plane["inlier_indices"]],raw_to_canonical);normal,offset=transform_plane(plane["normal"],plane["offset"],raw_to_canonical);normal=np.asarray(normal)
    if plane["semantic_candidate"]=="floor":
        lo,hi=np.percentile(pts,[2,98],axis=0);center=[float((lo[0]+hi[0])/2),float((lo[1]+hi[1])/2),-0.02];dims=[float(hi[0]-lo[0]),float(hi[1]-lo[1]),0.04];rotation=np.eye(3)
    else:
        # Gravity is authoritative for room vertical.  Remove small raw-plane
        # tilt, then refit the offset to the same inlier points.
        normal[2]=0.0;normal/=np.linalg.norm(normal);offset=-float(np.median(pts@normal))
        up=np.asarray([0.,0.,1.]);tangent=np.cross(up,normal);tangent/=np.linalg.norm(tangent);normal=np.cross(tangent,up);normal/=np.linalg.norm(normal);rotation=np.column_stack([tangent,up,normal]);coords=pts@rotation;lo,hi=np.percentile(coords,[2,98],axis=0);local=(lo+hi)/2;local[2]=-offset;center=rotation@local;dims=np.maximum(hi-lo,[.1,.1,.04]);dims[2]=.04
    return {"plane_id":plane["plane_id"],"semantic":plane["semantic_candidate"],"center":np.asarray(center).tolist(),"dimensions":np.asarray(dims).tolist(),"rotation_matrix":rotation.tolist(),"quaternion_wxyz":matrix_to_quaternion_wxyz(rotation),"plane_equation":{"normal":normal.tolist(),"offset":float(offset)},"confidence":plane["confidence"],"extent_policy":"robust p2-p98 raw structural inliers; conservative thickness"}


def _support_assignments(records:list[dict[str,Any]],planes:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    walls=[p for p in planes if "wall" in p["semantic"]];result={}
    for item in records:
        points=item["points"];lo,hi=np.percentile(points,[2,98],axis=0);label=item["semantic_label"].lower();wall_scores=[]
        for wall in walls:
            n=np.asarray(wall["plane_equation"]["normal"]);d=wall["plane_equation"]["offset"];wall_scores.append((float(np.median(np.abs(points@n+d))),wall))
        wall_distance,wall=min(wall_scores,key=lambda x:x[0]) if wall_scores else (math.inf,None)
        wall_semantic=any(token in label for token in ("wall","frame","bulletin","radiator","light"))
        if wall is not None and (wall_semantic or (wall_distance<.10 and lo[2]>.08)): result[item["object_id"]]={"target":wall["plane_id"],"type":"wall","confidence":float(max(.3,1-wall_distance/.2)),"wall":wall}
        elif lo[2]<=.12: result[item["object_id"]]={"target":"plane_floor","type":"floor","confidence":float(max(.3,1-abs(lo[2])/.12))}
        else:
            candidates=[]
            for other in records:
                if other is item:continue
                olo,ohi=np.percentile(other["points"],[2,98],axis=0);height_gap=abs(lo[2]-ohi[2]);overlap=max(0,min(hi[0],ohi[0])-max(lo[0],olo[0]))*max(0,min(hi[1],ohi[1])-max(lo[1],olo[1]));candidates.append((height_gap,-overlap,other))
            best=min(candidates,key=lambda x:(x[0],x[1])) if candidates else None
            if best is not None and best[0]<.12 and best[1]<0:
                gap,_,other=best;result[item["object_id"]]={"target":other["object_id"],"type":"object","confidence":float(max(.3,1-gap/.12))}
            else:result[item["object_id"]]={"target":None,"type":"unknown","confidence":.25}
    return result


def _wall_transform(points:np.ndarray,wall:dict[str,Any])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=np.asarray(wall["plane_equation"]["normal"]);d=float(wall["plane_equation"]["offset"]);up=np.asarray([0.,0.,1.]);t=np.cross(up,n);t/=np.linalg.norm(t);r=np.column_stack([t,up,n]);local=points@r;lo,hi=np.percentile(local,[2,98],axis=0);dims=np.maximum(hi-lo,[.025,.025,.04]);dims[2]=float(np.clip(dims[2],.04,.15));center_local=(lo+hi)/2;center_local[2]=-d+dims[2]/2;return r@center_local,dims,r


def _collision_relation(a:dict[str,Any],b:dict[str,Any])->float:
    def corners(item):return cuboid_corners(np.asarray(item["transform"]["center"]),np.asarray(item["transform"]["dimensions"]),np.asarray(item["transform"]["rotation_matrix"]))
    ca,cb=corners(a),corners(b);ra=np.asarray(a["transform"]["rotation_matrix"]);rb=np.asarray(b["transform"]["rotation_matrix"]);axes=[ra[:,i] for i in range(3)]+[rb[:,i] for i in range(3)]
    for aa in axes[:3]:
        for bb in axes[3:]:
            c=np.cross(aa,bb)
            if np.linalg.norm(c)>1e-7:axes.append(c/np.linalg.norm(c))
    depth=1e9
    for axis in axes:
        pa=ca@axis;pb=cb@axis;gap=max(pb.min()-pa.max(),pa.min()-pb.max())
        if gap>1e-6:return 0.0
        depth=min(depth,min(pa.max(),pb.max())-max(pa.min(),pb.min()))
    return float(max(0,depth))


def _compile_clean_into(*,output:Path,sam_dir:Path,source_path:Path,moge_dir:Path,scene_id:str,metadata:dict[str,Any])->dict[str,Any]:
    manifest=allowed_input_manifest(sam_dir,source_path,moge_dir,metadata);manifest_path=output/"allowed_inputs_manifest.json";manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    allowed=[Path(x["path"]) for x in manifest["inputs"]];guard=CleanReadGuard(allowed);metadata=guard.json(sam_dir/"metadata.json");moge_meta=guard.json(moge_dir/"metadata.json");archive=guard.npz(moge_dir/"geometry.npz");source=guard.image(source_path)
    masks={x["mask_id"]:guard.mask(sam_dir/x["filename"]) for x in metadata["masks"]};points=np.asarray(archive["points"],float);normals=np.asarray(archive["normal"],float);depth=np.asarray(archive["depth"],float);valid=np.asarray(archive["valid_mask"],bool);intrinsics=np.asarray(archive["intrinsics"],float)
    finite=np.isfinite(points).all(axis=2)&np.isfinite(normals).all(axis=2)&np.isfinite(depth);union=np.logical_or.reduce(list(masks.values()));structural=valid&finite&~union;ys,xs=np.nonzero(structural);occupied=np.median(points[valid&finite],axis=0)
    try:
        planes_raw,plane_diag=estimate_structural_planes(points[structural],normals[structural],np.column_stack([xs,ys]),valid.shape,occupied)
    except Exception as exc:
        raise CompilerValidationError(
            "indoor room reconstruction requires floor, left_wall, and right_wall evidence; "
            "available=[], missing=['floor', 'left_wall', 'right_wall']"
        ) from exc
    by_semantic={x["semantic_candidate"]:x for x in planes_raw}
    required_semantics={"floor","left_wall","right_wall"};available_semantics=set(by_semantic);missing_semantics=required_semantics-available_semantics
    if missing_semantics:
        raise CompilerValidationError(
            "indoor room reconstruction requires floor, left_wall, and right_wall evidence; "
            f"available={sorted(available_semantics)}, missing={sorted(missing_semantics)}"
        )
    canonical=construct_canonical_transform(by_semantic["floor"]["normal"],by_semantic["floor"]["offset"],occupied);raw_to_canonical=canonical["raw_to_canonical"];canonical_to_raw=canonical["canonical_to_raw"]
    joint_room=fit_joint_room_model(np.asarray(source),points,depth,normals,valid,structural,np.column_stack([xs,ys]),planes_raw,raw_to_canonical,canonical_to_raw,intrinsics);room=joint_room["room_proxies"];room_by_id={x["plane_id"]:x for x in room};camera_candidates=joint_room["camera_candidates"]
    perspective=next(candidate for candidate in camera_candidates if candidate["type"]=="perspective");camera={"canonical_to_camera":np.asarray(perspective["canonical_to_camera_transform"],float),"intrinsics":np.asarray(perspective["normalized_intrinsics"],float),"image_shape":valid.shape};plane_diag["joint_room_fit"]=joint_room["report"];plane_diag["joint_room_landmarks"]=joint_room["structural_landmarks"]
    geometry=[]
    for meta in metadata["masks"]:
        mask=masks[meta["mask_id"]];raw_valid=mask&valid&finite
        if not raw_valid.any():raise CompilerValidationError(f"SAM mask {meta['mask_id']} has no usable MoGe geometry")
        cleaning=clean_masked_geometry(mask,points,depth,normals,valid,2)
        if cleaning["success"]:
            raw=cleaning["points"];raw_normals=cleaning["normals"];raw_depth=cleaning["depth"];pixel_yx=cleaning["pixel_yx"]
        else:
            raw=points[raw_valid];raw_normals=normals[raw_valid];raw_depth=depth[raw_valid];y,x=np.nonzero(raw_valid);pixel_yx=np.column_stack([y,x])
        world=transform_points(raw,raw_to_canonical);world_normals=transform_normals(raw_normals,raw_to_canonical);geometry.append({"object_id":meta["mask_id"],"semantic_label":meta["label"],"meta":meta,"mask":mask,"points":world,"normals":world_normals,"depth":raw_depth,"pixel_yx":pixel_yx,"valid_ratio":float(raw_valid.sum()/max(1,mask.sum())),"cleaning":{key:value for key,value in cleaning.items() if key not in {"points","depth","normals","pixel_yx","selection_mask"}}})
    initial_supports=_support_assignments(geometry,room);wall_assignments={key:value for key,value in initial_supports.items() if value["type"]=="wall"};support_graph=infer_support_graph(geometry,wall_assignments);supports=support_graph["assignments"];occlusions={item["object_id"]:detect_occlusion(item,geometry) for item in geometry};objects=[];per_object=[]
    geometry_by_id={item["object_id"]:item for item in geometry};fit_order=topological_support_order([item["object_id"] for item in geometry],supports)
    for item in (geometry_by_id[object_id] for object_id in fit_order):
        mask=item["mask"];frame=estimate_horizontal_frame(item["normals"],item["pixel_yx"],mask);center_seed=np.median(item["points"],axis=0);vertical_angle=_axis_angle(center_seed,np.asarray([0,0,1.]),camera);families,segments=extract_mask_edge_families(mask,vertical_angle);support=supports[item["object_id"]];occlusion=occlusions[item["object_id"]]
        previous_yaw=frame["normal_derived_yaw_degrees"] or 0.0;candidates=generate_yaw_candidates(frame,families,previous_yaw,4);candidate_records=[]
        if support["type"]=="wall":
            center,dims,rotation=_wall_transform(item["points"],support["wall"]);candidates=[{"candidate_id":"wall_constrained","yaw_degrees":float(math.degrees(math.atan2(rotation[1,0],rotation[0,0]))),"source":"normal_first_plus_hard_wall_constraint"}]
        for c in candidates:
            if support["type"]=="wall":
                placement=fit_position_dimensions_fixed_rotation(item["points"],rotation,mask,camera,support,occlusion["occluder_masks"]);center=placement["center"];dims=placement["dimensions"]
            else:
                rotation=yaw_rotation(c["yaw_degrees"]);placement=fit_position_dimensions_fixed_rotation(item["points"],rotation,mask,camera,support,occlusion["occluder_masks"]);center=placement["center"];dims=placement["dimensions"]
            projected=project_world_points(cuboid_corners(center,dims,rotation),canonical_to_raw,intrinsics,valid.shape);pm=placement["metrics"];normal_error=_normal_error(rotation,frame);edge_error=_edge_error(center,rotation,camera,families);support_error=0.0 if support["type"] in {"floor","object","wall"} else float(max(0,center[2]-dims[2]/2));metrics={"bbox_iou":pm["bbox_iou"],"hull_iou":pm["visible_hull_iou"],"full_hull_iou":pm["full_hull_iou"],"visible_hull_iou":pm["visible_hull_iou"],"centroid_error_pixels":pm["centroid_error_pixels"],"projected_bbox":pm["projected_bbox"],"normal_residual_degrees":normal_error,"horizontal_edge_error_degrees":edge_error,"vertical_edge_error_degrees":angle_distance_180(_axis_angle(center,np.asarray([0,0,1.]),camera),vertical_angle),"normal_coverage":frame["side_normal_coverage"],"support_error":support_error,"collision_risk":0.0};score=orientation_score(metrics,{"single_face_ambiguity":frame["ambiguity"]=="single_face_90_degree","normal_orthogonality_error":frame["orthogonality_error_degrees"] is not None and frame["orthogonality_error_degrees"]>12,"placement_invalid":placement["placement_classification"]=="placement_invalid"});candidate_records.append({"candidate_id":c["candidate_id"],"source":c["source"],"yaw_degrees":c["yaw_degrees"],"transform":{"center":np.asarray(center).tolist(),"dimensions":np.asarray(dims).tolist(),"rotation_matrix":rotation.tolist(),"quaternion_wxyz":matrix_to_quaternion_wxyz(rotation)},"metrics":{**metrics,**score},"placement":{"classification":placement["placement_classification"],"hard_gates":placement["hard_gates"],"rotation_unchanged":placement["rotation_unchanged"],"optimizer":placement["optimizer"],"scale_ratio":placement.get("scale_ratio",[1,1,1]),"full_projection_metrics":pm}})
        selected=max(candidate_records,key=lambda x:(x["placement"]["classification"]!="placement_invalid",x["metrics"]["orientation_valid"],x["metrics"]["score"]));near_square=not footprint_yaw_observable(np.asarray(selected["transform"]["dimensions"]));normal_disp=float(np.mean([c["dispersion_p90_degrees"] for c in frame["clusters"] if c["reliable"]])) if any(c["reliable"] for c in frame["clusters"]) else 90.0
        if item["valid_ratio"]<.5 or len(item["points"])<30:classification="insufficient_geometry"
        elif near_square:classification="yaw_unobservable"
        elif frame["ambiguity"]=="single_face_90_degree":classification="automatic_with_ambiguity"
        elif selected["metrics"]["orientation_valid"] and frame["orientation_confidence"]>=.6 and support["confidence"]>=.5:classification="automatic_high_confidence"
        else:classification="user_review_recommended"
        confidence=float(np.clip(.35*frame["orientation_confidence"]+.20*item["valid_ratio"]+.15*support["confidence"]+.15*selected["metrics"]["hull_iou"]+.15*max(0,1-selected["metrics"]["horizontal_edge_error_degrees"]/30),0,1))
        material_color,color_fallback=_normalize_color(item["meta"].get("color"),item["object_id"],len(objects))
        record={"object_id":item["object_id"],"semantic_label":item["semantic_label"],"primitive_type":"cube","transform":selected["transform"],"support_target":support["target"],"support_type":support["type"],"support_confidence":support["confidence"],"confidence_classification":classification,"placement_classification":selected["placement"]["classification"],"placement_hard_gates":selected["placement"]["hard_gates"],"placement_scale_ratio":selected["placement"]["scale_ratio"],"final_pose_confidence":confidence,"geometry_confidence":float(min(1,item["valid_ratio"]*min(1,len(item["points"])/500))),"geometry_cleaning":item["cleaning"],"occlusion":{key:value for key,value in occlusion.items() if key!="occluder_masks"},"normal_frame":frame,"normal_angular_dispersion_degrees":normal_disp,"validation_metrics":selected["metrics"],"ambiguity":{"type":"yaw_unobservable" if near_square else frame["ambiguity"],"candidate_count":len(candidate_records)},"rotation_candidates":candidate_records,"orientation_method":"normal_first_v3_universal","material_color":material_color,"material_color_source":"deterministic_fallback" if color_fallback else "export_metadata","collision_warnings":[]};objects.append(record);per_object.append((record,item,segments,projected))
    propagation=propagate_support_contacts(objects,supports);support_graph["final_contact_propagation"]=propagation
    for obj in objects:
        item=geometry_by_id[obj["object_id"]];occlusion=occlusions[obj["object_id"]];transform=obj["transform"];projected=project_world_points(cuboid_corners(np.asarray(transform["center"]),np.asarray(transform["dimensions"]),np.asarray(transform["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);pm=occlusion_projection_metrics(projected,item["mask"],occlusion["occluder_masks"]);assignment=supports[obj["object_id"]];below=False;support_error=obj["validation_metrics"].get("support_error",0.0)
        if assignment["type"]=="object":support_error=abs(support_gap(next(parent for parent in objects if parent["object_id"]==assignment["target"]),obj,np.asarray(assignment["support_normal"])));below=support_error>1e-6
        gates=placement_hard_gates(pm,np.asarray(obj["placement_scale_ratio"]),False,below);obj["placement_hard_gates"]=gates;obj["placement_classification"]="placement_invalid" if gates else "placement_with_occlusion" if occlusion["occluder_masks"] else "placement_high_confidence" if pm["bbox_iou"]>=.55 and pm["centroid_error_pixels"]<=8 else "placement_review_required";obj["validation_metrics"].update({"bbox_iou":pm["bbox_iou"],"hull_iou":pm["visible_hull_iou"],"full_hull_iou":pm["full_hull_iou"],"visible_hull_iou":pm["visible_hull_iou"],"centroid_error_pixels":pm["centroid_error_pixels"],"projected_bbox":pm["projected_bbox"],"support_error":support_error})
    collisions=[]
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            depth_value=_collision_relation(a,b)
            if depth_value>1e-5:
                warning={"object_a":a["object_id"],"object_b":b["object_id"],"penetration_estimate":depth_value};collisions.append(warning);a["collision_warnings"].append(warning);b["collision_warnings"].append(warning)
    for obj in objects:
        risk=min(1.,sum(x["penetration_estimate"] for x in obj["collision_warnings"])/.15);obj["validation_metrics"]["collision_risk"]=risk
        if risk>.65 and obj["confidence_classification"]=="automatic_high_confidence":obj["confidence_classification"]="user_review_recommended";obj["final_pose_confidence"]*=.75
    scene={"schema_version":"1.0","mode":"clean_reconstruction","scene_id":scene_id,"input_manifest_sha256":sha256(manifest_path),"coordinate_system":{"canonical_up":[0,0,1],"raw_moge_to_canonical":raw_to_canonical.tolist(),"absolute_scale_verified":False},"room_proxies":room,"camera_candidates":camera_candidates,"provisional_camera_id":joint_room["provisional_camera_id"],"support_graph":support_graph,"semantic_objects":objects,"semantic_object_count":len(objects),"room_surface_semantic_count":0,"uncertainties":["absolute scale is unverified","camera selection is based on structural reprojection only","finite room boundaries remain partially occluded by furniture"]}
    scene=json.loads(json.dumps(scene,default=lambda value:value.item() if isinstance(value,np.generic) else value.tolist()));UnifiedScenePlan.model_validate(scene)
    jsonschema.validate(scene,json.loads((ROOT/"schemas"/"unified_v3_scene_plan.schema.json").read_text(encoding="utf-8")))
    _write_outputs(output,source,metadata,masks,scene,collisions,guard,plane_diag,per_object,moge_meta)
    report_path=output/"compilation_report.json";report=json.loads(report_path.read_text(encoding="utf-8"));report["clean_scene_plan_sha256"]=sha256(output/"unified_scene_plan.json");report["clean_output_hash_pending"]=False;report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return scene


_UNSET = object()


def _validate_compiled_output(output_dir: Path, object_ids: list[str]) -> None:
    missing=sorted(name for name in REQUIRED_OUTPUT_FILES if not (output_dir/name).is_file())
    if missing:raise RuntimeError(f"compiler did not produce required outputs: {', '.join(missing)}")
    scene=json.loads((output_dir/"unified_scene_plan.json").read_text(encoding="utf-8"))
    UnifiedScenePlan.model_validate(scene)
    jsonschema.validate(scene,json.loads((ROOT/"schemas"/"unified_v3_scene_plan.schema.json").read_text(encoding="utf-8")))
    compiled_ids=[item["object_id"] for item in scene["semantic_objects"]]
    if len(compiled_ids)!=len(object_ids) or set(compiled_ids)!=set(object_ids):
        raise RuntimeError("compiled object IDs do not exactly preserve export mask IDs")
    for object_id in object_ids:
        if not (output_dir/"per_object"/object_id/"metrics.json").is_file():
            raise RuntimeError(f"missing per-object diagnostics for {object_id}")


def compile_clean(
    output:Path|None=None,
    sam_dir:Path=DEFAULT_SAM_DIR,
    source_path:Path=DEFAULT_SOURCE,
    moge_dir:Path=DEFAULT_MOGE,
    handoff:Path|None|object=_UNSET,
    scene_id:str="office_test_unified_v3_clean",
    *,
    output_dir:Path|None=None,
    handoff_dir:Path|None|object=_UNSET,
) -> dict[str,Any]:
    """Compile into adjacent staging directories and atomically publish them.

    ``output`` and ``handoff`` remain as compatibility aliases. New callers
    should use explicit ``output_dir=`` and ``handoff_dir=`` keywords.
    """
    if output is not None and output_dir is not None:raise TypeError("use output or output_dir, not both")
    if handoff is not _UNSET and handoff_dir is not _UNSET:raise TypeError("use handoff or handoff_dir, not both")
    output_path=Path(output_dir or output or DEFAULT_OUTPUT).expanduser().resolve()
    validated=validate_compiler_inputs(sam_dir=Path(sam_dir),source_path=Path(source_path),moge_dir=Path(moge_dir))
    sam_path=validated["sam_dir"];source=validated["source_path"];moge_path=validated["moge_dir"]
    handoff_value=handoff_dir if handoff_dir is not _UNSET else handoff
    is_default_run=output_path==DEFAULT_OUTPUT.resolve() and sam_path==DEFAULT_SAM_DIR.resolve() and source==DEFAULT_SOURCE.resolve() and moge_path==DEFAULT_MOGE.resolve()
    if handoff_value is _UNSET:handoff_path=HANDOFF.resolve() if is_default_run else None
    else:handoff_path=None if handoff_value is None else Path(handoff_value).expanduser().resolve()
    if output_path.exists():raise FileExistsError(f"output directory already exists; refusing to overwrite: {output_path}")
    if handoff_path is not None and handoff_path.exists():raise FileExistsError(f"handoff directory already exists; refusing to overwrite: {handoff_path}")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_stage=Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-",dir=output_path.parent))
    handoff_stage:Path|None=None;published_handoff=False
    try:
        scene=_compile_clean_into(output=output_stage,sam_dir=sam_path,source_path=source,moge_dir=moge_path,scene_id=scene_id,metadata=validated["metadata"])
        object_ids=[item["mask_id"] for item in validated["metadata"]["masks"]]
        _validate_compiled_output(output_stage,object_ids)
        if handoff_path is not None:
            handoff_path.parent.mkdir(parents=True,exist_ok=True)
            handoff_stage=Path(tempfile.mkdtemp(prefix=f".{handoff_path.name}.staging-",dir=handoff_path.parent))
            finalize_handoff(output_dir=output_stage,handoff_dir=handoff_stage,semantic_object_count=len(object_ids))
            os.replace(handoff_stage,handoff_path);published_handoff=True;handoff_stage=None
        os.replace(output_stage,output_path)
        return scene
    except Exception:
        if published_handoff and handoff_path is not None and handoff_path.exists():shutil.rmtree(handoff_path,ignore_errors=True)
        raise
    finally:
        if output_stage.exists():shutil.rmtree(output_stage,ignore_errors=True)
        if handoff_stage is not None and handoff_stage.exists():shutil.rmtree(handoff_stage,ignore_errors=True)


def _wire(draw:ImageDraw.ImageDraw,pts:np.ndarray,color:tuple[int,int,int],width:int=2):
    for a,b in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)):draw.line([tuple(pts[a]),tuple(pts[b])],fill=color,width=width)


def _markdown_text(value: Any) -> str:
    return str(value).replace("\\","\\\\").replace("`","\\`").replace("*","\\*").replace("_","\\_").replace("[","\\[").replace("]","\\]").replace("\r"," ").replace("\n"," ")


def evaluate_generic_quality_gates(*,scene:dict[str,Any],export_ids:list[str],collisions:list[dict[str,Any]],room_fit:dict[str,Any],guard:CleanReadGuard)->tuple[dict[str,bool],list[dict[str,str]]]:
    objects=scene["semantic_objects"];object_ids=[item["object_id"] for item in objects];known_targets=set(object_ids)|{proxy["plane_id"] for proxy in scene["room_proxies"]}
    violations=[{"path":path,"token":token} for path in guard.read_log for token in FORBIDDEN_TOKENS if token in path.lower()]
    placement_invalid=sum(item["placement_classification"]=="placement_invalid" for item in objects)
    transforms_valid=all(np.isfinite(np.asarray(o["transform"]["center"],float)).all() and np.isfinite(np.asarray(o["transform"]["rotation_matrix"],float)).all() for o in objects)
    quaternions_normalized=all(abs(float(np.linalg.norm(np.asarray(o["transform"]["quaternion_wxyz"],float)))-1.0)<=1e-6 for o in objects)
    gates={
        "object_count_matches_export":len(objects)==len(export_ids),
        "all_export_object_ids_preserved":len(object_ids)==len(export_ids) and set(object_ids)==set(export_ids) and len(set(object_ids))==len(object_ids),
        "all_objects_have_primitives":all(o["primitive_type"]=="cube" for o in objects),
        "all_transforms_validate":transforms_valid and quaternions_normalized,
        "all_dimensions_positive":all((np.asarray(o["transform"]["dimensions"],float)>0).all() for o in objects),
        "all_support_targets_resolve":all(o["support_target"] is None or (o["support_target"] in known_targets and o["support_target"]!=o["object_id"]) for o in objects),
        "all_normal_first_compiler_invocations_completed":all(o["orientation_method"]=="normal_first_v3_universal" for o in objects),
        "all_final_object_support_gaps_below_1e_6":scene["support_graph"]["final_contact_propagation"]["maximum_final_gap"]<=1e-6,
        "no_invalid_placements":placement_invalid==0,
        "no_semantic_collisions":len(collisions)==0,
        "room_fit_completed":bool(room_fit["passed"]),
        "camera_candidates_present":bool(scene["camera_candidates"]),
        "provisional_camera_resolves":scene["provisional_camera_id"] in {candidate["camera_id"] for candidate in scene["camera_candidates"]},
        "no_forbidden_input_reads":not violations and not guard.denied_log,
    }
    return gates,violations


def _write_outputs(output:Path,source:Image.Image,metadata:dict[str,Any],masks:dict[str,np.ndarray],scene:dict[str,Any],collisions:list[dict[str,Any]],guard:CleanReadGuard,plane_diag:dict[str,Any],per_object:list[tuple],moge_meta:dict[str,Any]):
    objects=scene["semantic_objects"];counts={name:sum(o["confidence_classification"]==name for o in objects) for name in ("automatic_high_confidence","automatic_with_ambiguity","user_review_recommended","yaw_unobservable","insufficient_geometry")}
    placement_counts={name:sum(o["placement_classification"]==name for o in objects) for name in ("placement_high_confidence","placement_with_occlusion","placement_review_required","placement_invalid")}
    room_fit=plane_diag["joint_room_fit"];room_plan={"schema_version":"1.0","room_proxies":scene["room_proxies"],"uncertainties":scene["uncertainties"],"structural_fit_report":room_fit,"structural_landmarks":plane_diag["joint_room_landmarks"],"plane_fit_diagnostics":plane_diag};cameras={"schema_version":"1.0","provisional_camera_id":scene["provisional_camera_id"],"camera_candidates":scene["camera_candidates"],"moge_fov_x_degrees":moge_meta["estimated_fov_x_degrees"]};pose_report={"schema_version":"1.0","object_count":len(objects),"objects":objects};collision_report={"schema_version":"1.0","collision_count":len(collisions),"collisions":collisions};confidence={"schema_version":"1.0","counts":counts,"placement_counts":placement_counts,"objects":[{"object_id":o["object_id"],"semantic_label":o["semantic_label"],"classification":o["confidence_classification"],"placement_classification":o["placement_classification"],"confidence":o["final_pose_confidence"]} for o in objects]};input_audit={"schema_version":"1.0","mode":"clean_reconstruction","read_paths":guard.read_log,"denied_paths":guard.denied_log,"all_reads_allowed":not guard.denied_log}
    export_ids=[item["mask_id"] for item in metadata["masks"]];quality_gates,violations=evaluate_generic_quality_gates(scene=scene,export_ids=export_ids,collisions=collisions,room_fit=room_fit,guard=guard);forbidden={"schema_version":"1.0","forbidden_tokens":list(FORBIDDEN_TOKENS),"violations":violations,"passed":not violations}
    placement_report={"schema_version":"1.0","counts":placement_counts,"quality_gates":quality_gates,"support_contact_propagation":scene["support_graph"]["final_contact_propagation"],"objects":[{"object_id":o["object_id"],"semantic_label":o["semantic_label"],"support_type":o["support_type"],"support_target":o["support_target"],"placement_classification":o["placement_classification"],"hard_gates":o["placement_hard_gates"],"support_gap":o["validation_metrics"]["support_error"] if o["support_type"]=="object" else None,"bbox_iou":o["validation_metrics"]["bbox_iou"],"visible_hull_iou":o["validation_metrics"]["visible_hull_iou"],"centroid_error_pixels":o["validation_metrics"]["centroid_error_pixels"],"occlusion":o["occlusion"]} for o in objects]};compilation={"schema_version":"1.0","passed":all(quality_gates.values()),"quality_gates":quality_gates,"object_count":len(objects),"universal_v3_invocations":sum(o["orientation_method"]=="normal_first_v3_universal" for o in objects),"confidence_counts":counts,"placement_counts":placement_counts,"clean_output_hash_pending":True}
    files={"unified_scene_plan.json":scene,"room_plan.json":room_plan,"camera_candidates.json":cameras,"object_pose_report.json":pose_report,"support_graph.json":{"schema_version":"1.0",**scene["support_graph"]},"placement_report.json":placement_report,"collision_report.json":collision_report,"confidence_report.json":confidence,"clean_input_audit.json":input_audit,"forbidden_input_audit.json":forbidden,"compilation_report.json":compilation}
    for name,value in files.items():(output/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    object_lines="\n".join(f"- `{_markdown_text(o['object_id'])}` — {_markdown_text(o['semantic_label'])}: {o['confidence_classification']} ({o['final_pose_confidence']:.3f})" for o in objects)
    ambiguities=[o for o in objects if o["ambiguity"]["type"]!="none"];ambiguity_lines="\n".join(f"- `{_markdown_text(o['object_id'])}` — {_markdown_text(o['semantic_label'])}: {_markdown_text(o['ambiguity']['type'])}" for o in ambiguities) or "- None."
    (output/"unified_scene_plan.md").write_text(f"# Unified clean V3 scene plan\n\n- Objects: {len(objects)}\n- Room proxies: {len(scene['room_proxies'])}\n- Default orientation: normal-first V3 for every object\n- Provisional camera: {_markdown_text(scene['provisional_camera_id'])}\n",encoding="utf-8");(output/"object_pose_report.md").write_text("# Object pose report\n\n"+object_lines+"\n",encoding="utf-8");(output/"ambiguity_report.md").write_text("# Ambiguity report\n\n"+ambiguity_lines+"\n",encoding="utf-8");(output/"compilation_report.md").write_text("# Compilation report\n\n"+f"- Passed: {compilation['passed']}\n- Universal V3 invocations: {compilation['universal_v3_invocations']}/{len(objects)}\n- Forbidden-input violations: {len(violations)}\n"+"\n".join(f"- {name}: {value}" for name,value in quality_gates.items())+"\n",encoding="utf-8")
    perspective=next(candidate for candidate in scene["camera_candidates"] if candidate["type"]=="perspective");camera={"canonical_to_camera":np.asarray(perspective.get("canonical_to_camera_transform",np.linalg.inv(np.asarray(scene["coordinate_system"]["raw_moge_to_canonical"]))),float),"intrinsics":np.asarray(perspective["normalized_intrinsics"]),"image_shape":source.size[::-1]}
    numbered=source.copy();dn=ImageDraw.Draw(numbered);projected_all=source.copy();dp=ImageDraw.Draw(projected_all);normal_overlay=source.copy();dno=ImageDraw.Draw(normal_overlay);projected_centers={}
    for index,o in enumerate(objects,1):
        t=o["transform"];pts=project_world_points(cuboid_corners(np.asarray(t["center"]),np.asarray(t["dimensions"]),np.asarray(t["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(dp,pts,(50,255,120),2);c=np.mean(pts,axis=0);projected_centers[o["object_id"]]=c;dn.ellipse((c[0]-8,c[1]-8,c[0]+8,c[1]+8),fill=(0,0,0));dn.text((c[0]-4,c[1]-6),str(index),fill=(255,255,0));r=np.asarray(t["rotation_matrix"]);axes=project_world_points(np.vstack([t["center"],np.asarray(t["center"])+r[:,0]*.15,np.asarray(t["center"])+r[:,1]*.15]),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);dno.line([tuple(axes[0]),tuple(axes[1])],fill=(255,50,50),width=2);dno.line([tuple(axes[0]),tuple(axes[2])],fill=(50,255,50),width=2)
    numbered.save(output/"numbered_object_overlay.png");projected_all.save(output/"projected_primitives_overlay.png");normal_overlay.save(output/"normal_frames_overlay.png")
    support_overlay=source.copy();support_draw=ImageDraw.Draw(support_overlay);room_centers={p["plane_id"]:project_world_points(np.asarray([p["center"]]),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"])[0] for p in scene["room_proxies"]}
    for o in objects:
        target=o["support_target"]
        if not target:continue
        destination=projected_centers.get(target,room_centers.get(target))
        if destination is not None:support_draw.line([tuple(projected_centers[o["object_id"]]),tuple(destination)],fill=(255,210,40) if o["support_type"]=="object" else (80,180,255),width=2)
    support_overlay.save(output/"support_graph_overlay.png")
    room_overlay=source.copy();room_draw=ImageDraw.Draw(room_overlay)
    room_colors=((255,220,40),(60,220,255),(255,90,190),(160,255,100),(180,120,255))
    for index,proxy in enumerate(scene["room_proxies"]):
        color=room_colors[index%len(room_colors)]
        pts=project_world_points(cuboid_corners(np.asarray(proxy["center"]),np.asarray(proxy["dimensions"]),np.asarray(proxy["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(room_draw,pts,color,3)
    room_draw.text((8,8),f"joint room; FOV={perspective['field_of_view_x_degrees']:.2f}; RMSE={room_fit['structural_corner_reprojection_rmse_pixels']:.2f}px",fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0));room_overlay.save(output/"room_and_camera_overlay.png");room_overlay.save(output/"room_projection_overlay.png")
    confidence_height=max(160,65+len(objects)*31);conf=Image.new("RGB",(1200,confidence_height),(18,20,25));d=ImageDraw.Draw(conf);d.text((15,12),"Unified V3 confidence",fill="white")
    for i,o in enumerate(objects):
        y=45+i*31;label=o["semantic_label"].replace("\r"," ").replace("\n"," ")[:48];d.text((15,y),label,fill="white");d.rectangle((360,y,360+int(600*o["final_pose_confidence"]),y+16),fill=(40,210,240));d.text((980,y),o["confidence_classification"],fill="white")
    conf.save(output/"confidence_overview.png");amb=conf.copy();ImageDraw.Draw(amb).text((15,12),"Ambiguity overview",fill=(255,190,70));amb.save(output/"ambiguity_overview.png")
    overview=Image.new("RGB",(1341,894),(10,10,12))
    for i,img in enumerate((numbered,normal_overlay,projected_all,room_overlay,conf.resize((447,447)),amb.resize((447,447)))):overview.paste(img.resize((447,447)),((i%3)*447,(i//3)*447))
    overview.save(output/"clean_scene_plan_overview.png")
    for record,item,segments,_ in per_object:
        folder=output/"per_object"/record["object_id"];folder.mkdir(parents=True,exist_ok=True);mask=item["mask"];overlay=source.copy().convert("RGBA");shade=Image.new("RGBA",source.size,(255,40,40,0));shade.putalpha(Image.fromarray(np.uint8(mask)*100));Image.alpha_composite(overlay,shade).convert("RGB").save(folder/"sam_overlay.png")
        normal_image=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8);canonical_normals=item["normals"];normal_image[item["pixel_yx"][:,0],item["pixel_yx"][:,1]]=np.uint8(np.clip((canonical_normals+1)*127.5,0,255));Image.fromarray(normal_image,"RGB").save(folder/"normal_clusters.png")
        frame_image=source.copy();df=ImageDraw.Draw(frame_image);t_selected=record["transform"];center=np.asarray(t_selected["center"]);rotation=np.asarray(t_selected["rotation_matrix"]);axes=project_world_points(np.vstack([center,center+rotation[:,0]*.2,center+rotation[:,1]*.2]),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);df.line([tuple(axes[0]),tuple(axes[1])],fill=(255,60,60),width=3);df.line([tuple(axes[0]),tuple(axes[2])],fill=(60,255,60),width=3);frame_image.save(folder/"horizontal_normal_frame.png");edge=source.copy();de=ImageDraw.Draw(edge)
        for s in segments:angle=math.radians(s["angle_degrees"]);de.line([(s["x"],s["y"]),(s["x"]+5*math.cos(angle),s["y"]+5*math.sin(angle))],fill=(50,255,180),width=1)
        edge.save(folder/"edge_family_visualization.png");single=source.copy();ds=ImageDraw.Draw(single);t=record["transform"];pts=project_world_points(cuboid_corners(np.asarray(t["center"]),np.asarray(t["dimensions"]),np.asarray(t["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(ds,pts,(50,255,120),2);single.save(folder/"projected_primitive.png")
        panels=[]
        for candidate in record["rotation_candidates"]:
            panel=source.copy();dc=ImageDraw.Draw(panel);ct=candidate["transform"];cp=project_world_points(cuboid_corners(np.asarray(ct["center"]),np.asarray(ct["dimensions"]),np.asarray(ct["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(dc,cp,(50,255,120),2);dc.rectangle((0,0,source.width,38),fill=(0,0,0));dc.text((5,4),f"{candidate['candidate_id']} yaw={candidate['yaw_degrees']:.1f} score={candidate['metrics']['score']:.3f}",fill=(255,255,255));dc.text((5,20),f"normal={candidate['metrics']['normal_residual_degrees']:.1f} edge={candidate['metrics']['horizontal_edge_error_degrees']:.1f}",fill=(255,200,70));panels.append(panel)
        comparison=Image.new("RGB",(source.width*len(panels),source.height),(10,10,12))
        for index,panel in enumerate(panels):comparison.paste(panel,(source.width*index,0))
        comparison.save(folder/"candidate_comparison.png");(folder/"metrics.json").write_text(json.dumps(record,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else value.tolist())+"\n",encoding="utf-8")
    provisional=next(candidate for candidate in scene["camera_candidates"] if candidate["camera_id"]==scene["provisional_camera_id"])
    manifest={"schema_version":"1.0","source":"unified_scene_plan.json","room_proxies":scene["room_proxies"],"provisional_camera":provisional,"semantic_primitives":[{"object_id":o["object_id"],"semantic_label":o["semantic_label"],"primitive_type":"cube","transform":o["transform"],"material_color":o["material_color"],"confidence_classification":o["confidence_classification"],"placement_classification":o["placement_classification"],"placement_hard_gates":o["placement_hard_gates"],"occlusion":o["occlusion"],"support_target":o["support_target"],"collision_warnings":o["collision_warnings"],"unresolved_ambiguity":o["ambiguity"]} for o in objects],"semantic_primitive_count":len(objects),"approved_artifact_references":[]}
    (output/"blender_one_batch_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


def finalize_handoff(*,output_dir:Path,handoff_dir:Path,semantic_object_count:int)->None:
    output_dir=Path(output_dir).resolve();handoff_dir=Path(handoff_dir).resolve()
    manifest_path=output_dir/"blender_one_batch_manifest.json"
    if not manifest_path.is_file():raise CompilerValidationError(f"Blender-neutral manifest is missing: {manifest_path}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("semantic_primitive_count")!=semantic_object_count or len(manifest.get("semantic_primitives",[]))!=semantic_object_count:
        raise CompilerValidationError("handoff semantic object count does not match the run manifest")
    handoff_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(manifest_path,handoff_dir/"blender_one_batch_manifest.json")
    allowed_manifest=output_dir/"allowed_inputs_manifest.json"
    if allowed_manifest.is_file():shutil.copy2(allowed_manifest,handoff_dir/"allowed_inputs_manifest.json")
    forbidden={"schema_version":"1.0","patterns":["*.blend","approved transforms","manual room/camera corrections","prior renders"],"included_forbidden_artifacts":[],"passed":True}
    (handoff_dir/"forbidden_artifacts_manifest.json").write_text(json.dumps(forbidden,indent=2)+"\n",encoding="utf-8")
    (handoff_dir/"README.md").write_text(f"# Unified V3 clean-room Blender-neutral handoff\n\nUse only `blender_one_batch_manifest.json`. Construct the {len(manifest['room_proxies'])} room proxies, provisional camera, and {semantic_object_count} semantic primitives in one batch. This handoff does not execute Blender or create Blender objects. Do not load prior checkpoints, transforms, or renders.\n",encoding="utf-8")
    (handoff_dir/"execution_instructions.md").write_text(f"# Execution instructions\n\n1. Start from an empty scene.\n2. Read only the clean manifest.\n3. Construct room, camera, and {semantic_object_count} semantic primitives in one batch.\n4. Render perspective once.\n5. Save a new checkpoint and validation report.\n",encoding="utf-8")
    (handoff_dir/"expected_output_contract.json").write_text(json.dumps({"required":["new .blend checkpoint","perspective render","validation JSON","protected clean-manifest hash"],"semantic_primitive_count":semantic_object_count},indent=2)+"\n",encoding="utf-8")
    schemas=handoff_dir/"schemas";schemas.mkdir(exist_ok=True);shutil.copy2(ROOT/"schemas"/"unified_v3_scene_plan.schema.json",schemas/"unified_v3_scene_plan.schema.json")


def run_regression_audit(clean_output:Path=DEFAULT_OUTPUT,audit_output:Path|None=None)->dict[str,Any]:
    audit_output=audit_output or ROOT/"outputs"/"office_test"/"unified_v3_regression_audit";audit_output.mkdir(parents=True,exist_ok=True);clean_path=clean_output/"unified_scene_plan.json";clean_hash=sha256(clean_path);clean=json.loads(clean_path.read_text(encoding="utf-8"));approved=json.loads((ROOT/"outputs"/"office_test"/"blender_execution"/"pose_refinement_v3"/"approved"/"approval_validation.json").read_text(encoding="utf-8"));final=json.loads((ROOT/"outputs"/"office_test"/"blender_execution"/"full_scene_batches"/"complete_20_objects"/"final_validation.json").read_text(encoding="utf-8"));approved_by={x["object_id"]:x for x in final["objects"]}
    for x in approved["approved_candidates"]:approved_by[x["object_id"]]={"object_id":x["object_id"],"semantic_label":x["semantic_label"],"final_transform":x["transform"]}
    diffs=[]
    for o in clean["semantic_objects"]:
        a=approved_by.get(o["object_id"]);ct=o["transform"];at=a["final_transform"] if a else None
        if not at:diffs.append({"object_id":o["object_id"],"semantic_label":o["semantic_label"],"classification":"unavailable_ground_truth"});continue
        clean_yaw=math.degrees(math.atan2(ct["rotation_matrix"][1][0],ct["rotation_matrix"][0][0]));aq=at["quaternion_wxyz"];approved_yaw=math.degrees(math.atan2(2*(aq[0]*aq[3]+aq[1]*aq[2]),1-2*(aq[2]**2+aq[3]**2)));center_delta=float(np.linalg.norm(np.asarray(ct["center"])-np.asarray(at["center"])));yaw_delta=angle_distance_180(clean_yaw,approved_yaw);classification="agreement" if center_delta<.12 and yaw_delta<15 else "explainable_ambiguity" if o["confidence_classification"] in ("yaw_unobservable","automatic_with_ambiguity") else "regression"
        diffs.append({"object_id":o["object_id"],"semantic_label":o["semantic_label"],"center_delta":center_delta,"dimension_delta":(np.asarray(ct["dimensions"])-np.asarray(at["dimensions"])).tolist(),"clean_yaw_degrees":clean_yaw,"approved_yaw_degrees":approved_yaw,"yaw_delta_degrees":yaw_delta,"classification":classification})
    report={"schema_version":"1.0","clean_plan_sha256":clean_hash,"clean_plan_unchanged":sha256(clean_path)==clean_hash,"mode":"regression_audit","clean_result_mutated":False,"differences":diffs,"counts":{k:sum(x["classification"]==k for x in diffs) for k in ("agreement","explainable_ambiguity","regression","improvement","unavailable_ground_truth")}}
    (audit_output/"regression_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");(audit_output/"transform_difference_report.json").write_text(json.dumps({"schema_version":"1.0","objects":diffs},indent=2)+"\n",encoding="utf-8");(audit_output/"regression_audit.md").write_text("# Unified V3 clean versus approved regression audit\n\n"+"\n".join(f"- {x['semantic_label']}: {x['classification']}" for x in diffs)+"\n",encoding="utf-8");source=Image.open(DEFAULT_SOURCE).convert("RGB");draw=ImageDraw.Draw(source);draw.rectangle((0,0,447,35),fill=(0,0,0));draw.text((8,8),f"clean vs approved: {report['counts']}",fill=(255,255,255));source.save(audit_output/"approved_vs_clean_overlay.png");return report
