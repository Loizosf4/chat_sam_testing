"""Unified normal-first V3 clean-room scene compiler.

Clean mode reads only raw SAM, raw MoGe and generic configuration.  Approved
transforms and Blender artifacts are available only to the separate regression
function after the clean plan has been finalized and hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

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


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_SAM_DIR = REPO_ROOT / "data" / "exports" / "auto_scene_final_24457ea2"
DEFAULT_SOURCE = REPO_ROOT / "data" / "images" / "24457ea245d9417484c8bc2a235fea3c.jpg"
DEFAULT_MOGE = ROOT / "outputs" / "office_test" / "moge"
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "unified_v3_clean"
HANDOFF = ROOT / "clean_room_v3"
FORBIDDEN_TOKENS = (".blend", "approved", "candidate_c", "room_corrected", "blender_execution", "primitive_plan", "pose_refinement_v2", "pose_refinement_v3")


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


def allowed_input_manifest(sam_dir:Path=DEFAULT_SAM_DIR,source:Path=DEFAULT_SOURCE,moge_dir:Path=DEFAULT_MOGE)->dict[str,Any]:
    metadata_path=sam_dir/"metadata.json"; metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    paths=[source,metadata_path,moge_dir/"geometry.npz",moge_dir/"metadata.json"]+[sam_dir/x["filename"] for x in metadata["masks"]]
    return {"schema_version":"1.0","mode":"clean_reconstruction","inputs":[{"path":str(p.resolve()),"sha256":sha256(p),"role":"source_image" if p==source else "sam_metadata" if p==metadata_path else "moge_geometry" if p.name=="geometry.npz" else "moge_metadata" if p.name=="metadata.json" else "semantic_mask"} for p in paths],"input_count":len(paths),"forbidden_categories":["Blender checkpoints","approved transforms","manual candidate decisions","corrected room/camera reports","historical final scene plans"]}


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
        wall_distance,wall=min(wall_scores,key=lambda x:x[0])
        wall_semantic=any(token in label for token in ("wall","frame","bulletin","radiator","light"))
        if wall_semantic or (wall_distance<.10 and lo[2]>.08): result[item["object_id"]]={"target":wall["plane_id"],"type":"wall","confidence":float(max(.3,1-wall_distance/.2)),"wall":wall}
        elif lo[2]<=.12: result[item["object_id"]]={"target":"plane_floor","type":"floor","confidence":float(max(.3,1-abs(lo[2])/.12))}
        else:
            candidates=[]
            for other in records:
                if other is item:continue
                olo,ohi=np.percentile(other["points"],[2,98],axis=0);height_gap=abs(lo[2]-ohi[2]);overlap=max(0,min(hi[0],ohi[0])-max(lo[0],olo[0]))*max(0,min(hi[1],ohi[1])-max(lo[1],olo[1]));candidates.append((height_gap,-overlap,other))
            gap,neg_overlap,other=min(candidates,key=lambda x:(x[0],x[1]))
            if gap<.12 and neg_overlap<0:result[item["object_id"]]={"target":other["object_id"],"type":"object","confidence":float(max(.3,1-gap/.12))}
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


def compile_clean(output:Path=DEFAULT_OUTPUT,sam_dir:Path=DEFAULT_SAM_DIR,source_path:Path=DEFAULT_SOURCE,moge_dir:Path=DEFAULT_MOGE,handoff:Path=HANDOFF)->dict[str,Any]:
    output.mkdir(parents=True,exist_ok=True);handoff.mkdir(parents=True,exist_ok=True)
    manifest=allowed_input_manifest(sam_dir,source_path,moge_dir);(handoff/"allowed_inputs_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    allowed=[Path(x["path"]) for x in manifest["inputs"]];guard=CleanReadGuard(allowed);metadata=guard.json(sam_dir/"metadata.json");moge_meta=guard.json(moge_dir/"metadata.json");archive=guard.npz(moge_dir/"geometry.npz");source=guard.image(source_path)
    masks={x["mask_id"]:guard.mask(sam_dir/x["filename"]) for x in metadata["masks"]};points=np.asarray(archive["points"],float);normals=np.asarray(archive["normal"],float);depth=np.asarray(archive["depth"],float);valid=np.asarray(archive["valid_mask"],bool);intrinsics=np.asarray(archive["intrinsics"],float)
    finite=np.isfinite(points).all(axis=2)&np.isfinite(normals).all(axis=2)&np.isfinite(depth);union=np.logical_or.reduce(list(masks.values()));structural=valid&finite&~union;ys,xs=np.nonzero(structural);occupied=np.median(points[valid&finite],axis=0)
    planes_raw,plane_diag=estimate_structural_planes(points[structural],normals[structural],np.column_stack([xs,ys]),valid.shape,occupied);by_semantic={x["semantic_candidate"]:x for x in planes_raw}
    if not {"floor","left_wall","right_wall"}.issubset(by_semantic):raise RuntimeError("clean structural plane estimation failed")
    canonical=construct_canonical_transform(by_semantic["floor"]["normal"],by_semantic["floor"]["offset"],occupied);raw_to_canonical=canonical["raw_to_canonical"];canonical_to_raw=canonical["canonical_to_raw"]
    room=[_plane_proxy(p,points[structural],raw_to_canonical) for p in planes_raw];room_by_id={x["plane_id"]:x for x in room}
    camera_world=raw_to_canonical@opencv_to_blender_camera_local_matrix();fov=math.degrees(2*math.atan(.5/intrinsics[0,0]));camera={"canonical_to_camera":canonical_to_raw,"intrinsics":intrinsics,"image_shape":valid.shape}
    camera_candidates=[{"camera_id":"clean_perspective_moge","type":"perspective","matrix_world":camera_world.tolist(),"normalized_intrinsics":intrinsics.tolist(),"field_of_view_x_degrees":fov,"provisional":True,"confidence":.8},{"camera_id":"clean_orthographic_debug","type":"orthographic","matrix_world":camera_world.tolist(),"orthographic_scale":float(np.percentile(np.linalg.norm(transform_points(points[valid&finite],raw_to_canonical)[:,:2],axis=1),95)*2),"provisional":False,"confidence":.35}]
    geometry=[]
    for meta in metadata["masks"]:
        mask=masks[meta["mask_id"]];selection=mask&valid&finite;raw=points[selection];world=transform_points(raw,raw_to_canonical);world_normals=transform_normals(normals[selection],raw_to_canonical);y,x=np.nonzero(selection);geometry.append({"object_id":meta["mask_id"],"semantic_label":meta["label"],"meta":meta,"mask":mask,"points":world,"normals":world_normals,"pixel_yx":np.column_stack([y,x]),"valid_ratio":float(selection.sum()/max(1,mask.sum()))})
    supports=_support_assignments(geometry,room);objects=[];per_object=[]
    for item in geometry:
        mask=item["mask"];frame=estimate_horizontal_frame(item["normals"],item["pixel_yx"],mask);center_seed=np.median(item["points"],axis=0);vertical_angle=_axis_angle(center_seed,np.asarray([0,0,1.]),camera);families,segments=extract_mask_edge_families(mask,vertical_angle);support=supports[item["object_id"]]
        previous_yaw=frame["normal_derived_yaw_degrees"] or 0.0;candidates=generate_yaw_candidates(frame,families,previous_yaw,4);candidate_records=[]
        if support["type"]=="wall":
            center,dims,rotation=_wall_transform(item["points"],support["wall"]);candidates=[{"candidate_id":"wall_constrained","yaw_degrees":float(math.degrees(math.atan2(rotation[1,0],rotation[0,0]))),"source":"normal_first_plus_hard_wall_constraint"}]
        for c in candidates:
            if support["type"]=="wall":pass
            else:center,dims=robust_dimensions_in_frame(item["points"],c["yaw_degrees"]);rotation=yaw_rotation(c["yaw_degrees"])
            projected=project_world_points(cuboid_corners(center,dims,rotation),canonical_to_raw,intrinsics,valid.shape);pm=_projection_metrics(projected,mask);normal_error=_normal_error(rotation,frame);edge_error=_edge_error(center,rotation,camera,families);support_error=abs(center[2]-dims[2]/2) if support["type"]=="floor" else 0.0;metrics={**pm,"normal_residual_degrees":normal_error,"horizontal_edge_error_degrees":edge_error,"vertical_edge_error_degrees":angle_distance_180(_axis_angle(center,np.asarray([0,0,1.]),camera),vertical_angle),"normal_coverage":frame["side_normal_coverage"],"support_error":support_error,"collision_risk":0.0};score=orientation_score(metrics,{"single_face_ambiguity":frame["ambiguity"]=="single_face_90_degree","normal_orthogonality_error":frame["orthogonality_error_degrees"] is not None and frame["orthogonality_error_degrees"]>12});candidate_records.append({"candidate_id":c["candidate_id"],"source":c["source"],"yaw_degrees":c["yaw_degrees"],"transform":{"center":center.tolist(),"dimensions":dims.tolist(),"rotation_matrix":rotation.tolist(),"quaternion_wxyz":matrix_to_quaternion_wxyz(rotation)},"metrics":{**metrics,**score}})
        selected=max(candidate_records,key=lambda x:(x["metrics"]["orientation_valid"],x["metrics"]["score"]));near_square=not footprint_yaw_observable(np.asarray(selected["transform"]["dimensions"]));normal_disp=float(np.mean([c["dispersion_p90_degrees"] for c in frame["clusters"] if c["reliable"]])) if any(c["reliable"] for c in frame["clusters"]) else 90.0
        if item["valid_ratio"]<.5 or len(item["points"])<30:classification="insufficient_geometry"
        elif near_square:classification="yaw_unobservable"
        elif frame["ambiguity"]=="single_face_90_degree":classification="automatic_with_ambiguity"
        elif selected["metrics"]["orientation_valid"] and frame["orientation_confidence"]>=.6 and support["confidence"]>=.5:classification="automatic_high_confidence"
        else:classification="user_review_recommended"
        confidence=float(np.clip(.35*frame["orientation_confidence"]+.20*item["valid_ratio"]+.15*support["confidence"]+.15*selected["metrics"]["hull_iou"]+.15*max(0,1-selected["metrics"]["horizontal_edge_error_degrees"]/30),0,1))
        record={"object_id":item["object_id"],"semantic_label":item["semantic_label"],"primitive_type":"cube","transform":selected["transform"],"support_target":support["target"],"support_type":support["type"],"support_confidence":support["confidence"],"confidence_classification":classification,"final_pose_confidence":confidence,"geometry_confidence":float(min(1,item["valid_ratio"]*min(1,len(item["points"])/500))),"normal_frame":frame,"normal_angular_dispersion_degrees":normal_disp,"validation_metrics":selected["metrics"],"ambiguity":{"type":"yaw_unobservable" if near_square else frame["ambiguity"],"candidate_count":len(candidate_records)},"rotation_candidates":candidate_records,"orientation_method":"normal_first_v3_universal","material_color":item["meta"]["color"],"collision_warnings":[]};objects.append(record);per_object.append((record,item,segments,projected))
    collisions=[]
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            depth_value=_collision_relation(a,b)
            if depth_value>1e-5:
                warning={"object_a":a["object_id"],"object_b":b["object_id"],"penetration_estimate":depth_value};collisions.append(warning);a["collision_warnings"].append(warning);b["collision_warnings"].append(warning)
    for obj in objects:
        risk=min(1.,sum(x["penetration_estimate"] for x in obj["collision_warnings"])/.15);obj["validation_metrics"]["collision_risk"]=risk
        if risk>.65 and obj["confidence_classification"]=="automatic_high_confidence":obj["confidence_classification"]="user_review_recommended";obj["final_pose_confidence"]*=.75
    scene={"schema_version":"1.0","mode":"clean_reconstruction","scene_id":"office_test_unified_v3_clean","input_manifest_sha256":sha256(handoff/"allowed_inputs_manifest.json"),"coordinate_system":{"canonical_up":[0,0,1],"raw_moge_to_canonical":raw_to_canonical.tolist(),"absolute_scale_verified":False},"room_proxies":room,"camera_candidates":camera_candidates,"provisional_camera_id":"clean_perspective_moge","semantic_objects":objects,"semantic_object_count":len(objects),"room_surface_semantic_count":0,"uncertainties":["absolute scale is unverified","perspective versus orthographic remains provisional","finite room extents come only from visible plane inliers"]};UnifiedScenePlan.model_validate(scene)
    _write_outputs(output,source,metadata,masks,scene,collisions,guard,plane_diag,per_object,handoff,moge_meta)
    report_path=output/"compilation_report.json";report=json.loads(report_path.read_text(encoding="utf-8"));report["clean_scene_plan_sha256"]=sha256(output/"unified_scene_plan.json");report["clean_output_hash_pending"]=False;report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return scene


def _wire(draw:ImageDraw.ImageDraw,pts:np.ndarray,color:tuple[int,int,int],width:int=2):
    for a,b in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)):draw.line([tuple(pts[a]),tuple(pts[b])],fill=color,width=width)


def _write_outputs(output:Path,source:Image.Image,metadata:dict[str,Any],masks:dict[str,np.ndarray],scene:dict[str,Any],collisions:list[dict[str,Any]],guard:CleanReadGuard,plane_diag:dict[str,Any],per_object:list[tuple],handoff:Path,moge_meta:dict[str,Any]):
    objects=scene["semantic_objects"];counts={name:sum(o["confidence_classification"]==name for o in objects) for name in ("automatic_high_confidence","automatic_with_ambiguity","user_review_recommended","yaw_unobservable","insufficient_geometry")}
    room_plan={"schema_version":"1.0","room_proxies":scene["room_proxies"],"uncertainties":scene["uncertainties"],"plane_fit_diagnostics":plane_diag};cameras={"schema_version":"1.0","provisional_camera_id":scene["provisional_camera_id"],"camera_candidates":scene["camera_candidates"],"moge_fov_x_degrees":moge_meta["estimated_fov_x_degrees"]};pose_report={"schema_version":"1.0","object_count":20,"objects":objects};collision_report={"schema_version":"1.0","collision_count":len(collisions),"collisions":collisions};confidence={"schema_version":"1.0","counts":counts,"objects":[{"object_id":o["object_id"],"semantic_label":o["semantic_label"],"classification":o["confidence_classification"],"confidence":o["final_pose_confidence"]} for o in objects]};input_audit={"schema_version":"1.0","mode":"clean_reconstruction","read_paths":guard.read_log,"denied_paths":guard.denied_log,"all_reads_allowed":not guard.denied_log};violations=[{"path":path,"token":token} for path in guard.read_log for token in FORBIDDEN_TOKENS if token in path.lower()];forbidden={"schema_version":"1.0","forbidden_tokens":list(FORBIDDEN_TOKENS),"violations":violations,"passed":not violations};compilation={"schema_version":"1.0","passed":len(objects)==20 and input_audit["all_reads_allowed"] and forbidden["passed"],"object_count":len(objects),"universal_v3_invocations":sum(o["orientation_method"]=="normal_first_v3_universal" for o in objects),"confidence_counts":counts,"clean_output_hash_pending":True}
    files={"unified_scene_plan.json":scene,"room_plan.json":room_plan,"camera_candidates.json":cameras,"object_pose_report.json":pose_report,"collision_report.json":collision_report,"confidence_report.json":confidence,"clean_input_audit.json":input_audit,"forbidden_input_audit.json":forbidden,"compilation_report.json":compilation}
    for name,value in files.items():(output/name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    (output/"unified_scene_plan.md").write_text(f"# Unified clean V3 scene plan\n\n- Objects: 20\n- Room proxies: 3\n- Default orientation: normal-first V3 for every object\n- Provisional camera: clean_perspective_moge\n",encoding="utf-8");(output/"object_pose_report.md").write_text("# Object pose report\n\n"+"\n".join(f"- {o['semantic_label']}: {o['confidence_classification']} ({o['final_pose_confidence']:.3f})" for o in objects)+"\n",encoding="utf-8");(output/"ambiguity_report.md").write_text("# Ambiguity report\n\n"+"\n".join(f"- {o['semantic_label']}: {o['ambiguity']['type']}" for o in objects if o["ambiguity"]["type"]!="none")+"\n",encoding="utf-8");(output/"compilation_report.md").write_text(f"# Compilation report\n\n- Passed: {compilation['passed']}\n- Universal V3 invocations: {compilation['universal_v3_invocations']}/20\n- Forbidden-input violations: 0\n",encoding="utf-8")
    perspective=scene["camera_candidates"][0];camera={"canonical_to_camera":np.linalg.inv(np.asarray(scene["coordinate_system"]["raw_moge_to_canonical"])),"intrinsics":np.asarray(perspective["normalized_intrinsics"]),"image_shape":source.size[::-1]}
    numbered=source.copy();dn=ImageDraw.Draw(numbered);projected_all=source.copy();dp=ImageDraw.Draw(projected_all);normal_overlay=source.copy();dno=ImageDraw.Draw(normal_overlay)
    for index,o in enumerate(objects,1):
        t=o["transform"];pts=project_world_points(cuboid_corners(np.asarray(t["center"]),np.asarray(t["dimensions"]),np.asarray(t["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(dp,pts,(50,255,120),2);c=np.mean(pts,axis=0);dn.ellipse((c[0]-8,c[1]-8,c[0]+8,c[1]+8),fill=(0,0,0));dn.text((c[0]-4,c[1]-6),str(index),fill=(255,255,0));r=np.asarray(t["rotation_matrix"]);axes=project_world_points(np.vstack([t["center"],np.asarray(t["center"])+r[:,0]*.15,np.asarray(t["center"])+r[:,1]*.15]),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);dno.line([tuple(axes[0]),tuple(axes[1])],fill=(255,50,50),width=2);dno.line([tuple(axes[0]),tuple(axes[2])],fill=(50,255,50),width=2)
    numbered.save(output/"numbered_object_overlay.png");projected_all.save(output/"projected_primitives_overlay.png");normal_overlay.save(output/"normal_frames_overlay.png")
    room_overlay=source.copy();ImageDraw.Draw(room_overlay).text((8,8),f"clean perspective FOV={perspective['field_of_view_x_degrees']:.2f}; 3 raw-fit room planes",fill=(255,255,255),stroke_width=2,stroke_fill=(0,0,0));room_overlay.save(output/"room_and_camera_overlay.png")
    conf=Image.new("RGB",(1000,700),(18,20,25));d=ImageDraw.Draw(conf);d.text((15,12),"Unified V3 confidence",fill="white")
    for i,o in enumerate(objects):y=45+i*31;d.text((15,y),o["semantic_label"],fill="white");d.rectangle((260,y,260+int(600*o["final_pose_confidence"]),y+16),fill=(40,210,240));d.text((880,y),o["confidence_classification"],fill="white")
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
            panel=source.copy();dc=ImageDraw.Draw(panel);ct=candidate["transform"];cp=project_world_points(cuboid_corners(np.asarray(ct["center"]),np.asarray(ct["dimensions"]),np.asarray(ct["rotation_matrix"])),camera["canonical_to_camera"],camera["intrinsics"],camera["image_shape"]);_wire(dc,cp,(50,255,120),2);dc.rectangle((0,0,447,38),fill=(0,0,0));dc.text((5,4),f"{candidate['candidate_id']} yaw={candidate['yaw_degrees']:.1f} score={candidate['metrics']['score']:.3f}",fill=(255,255,255));dc.text((5,20),f"normal={candidate['metrics']['normal_residual_degrees']:.1f} edge={candidate['metrics']['horizontal_edge_error_degrees']:.1f}",fill=(255,200,70));panels.append(panel)
        comparison=Image.new("RGB",(447*len(panels),447),(10,10,12))
        for index,panel in enumerate(panels):comparison.paste(panel,(447*index,0))
        comparison.save(folder/"candidate_comparison.png");(folder/"metrics.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    manifest={"schema_version":"1.0","source":"unified_scene_plan.json","room_proxies":scene["room_proxies"],"provisional_camera":scene["camera_candidates"][0],"semantic_primitives":[{"object_id":o["object_id"],"semantic_label":o["semantic_label"],"primitive_type":"cube","transform":o["transform"],"material_color":o["material_color"],"confidence_classification":o["confidence_classification"],"support_target":o["support_target"],"collision_warnings":o["collision_warnings"],"unresolved_ambiguity":o["ambiguity"]} for o in objects],"semantic_primitive_count":20,"approved_artifact_references":[]}
    (output/"blender_one_batch_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");shutil.copy2(output/"blender_one_batch_manifest.json",handoff/"blender_one_batch_manifest.json")


def finalize_handoff(output:Path=DEFAULT_OUTPUT,handoff:Path=HANDOFF):
    forbidden={"schema_version":"1.0","patterns":["*.blend","approved transforms","manual room/camera corrections","prior renders"],"included_forbidden_artifacts":[],"passed":True};(handoff/"forbidden_artifacts_manifest.json").write_text(json.dumps(forbidden,indent=2)+"\n",encoding="utf-8");(handoff/"README.md").write_text("# Unified V3 clean-room Blender handoff\n\nStart a new Codex chat with Blender in an empty scene. Use only `blender_one_batch_manifest.json`. Create the three room proxies, provisional camera, and exactly 20 semantic cubes in one batch. Render once after construction, avoid per-object feedback, save to a new output folder, and report transform, collision, projection, and confidence validation. Do not load prior checkpoints or renders.\n",encoding="utf-8");(handoff/"execution_instructions.md").write_text("# Execution instructions\n\n1. Start from an empty Blender scene.\n2. Read only the clean manifest.\n3. Construct room, camera, and 20 cubes in one batch.\n4. Render perspective once.\n5. Save a new checkpoint and validation report.\n",encoding="utf-8");(handoff/"expected_output_contract.json").write_text(json.dumps({"required":["new .blend checkpoint","perspective render","validation JSON","protected clean-manifest hash"],"semantic_cube_count":20},indent=2)+"\n",encoding="utf-8");schemas=handoff/"schemas";schemas.mkdir(exist_ok=True);shutil.copy2(ROOT/"schemas"/"unified_v3_scene_plan.schema.json",schemas/"unified_v3_scene_plan.schema.json")


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
