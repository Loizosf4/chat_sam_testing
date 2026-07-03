"""Read-only comparison of the original clean V3 plan and support-fit plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT=Path(__file__).resolve().parents[1]


def load(path:Path)->dict[str,Any]:return json.loads(path.read_text(encoding="utf-8"))


def evaluate(before_path:Path,after_path:Path,output:Path)->dict[str,Any]:
    before=load(before_path);after=load(after_path);b={x["object_id"]:x for x in before["semantic_objects"]};a={x["object_id"]:x for x in after["semantic_objects"]};objects=[]
    for oid,item in a.items():
        old=b[oid];objects.append({"object_id":oid,"semantic_label":item["semantic_label"],"orientation_rotation_unchanged_during_fitting":all(c["placement"]["rotation_unchanged"] for c in item["rotation_candidates"]),"before":{"support_type":old["support_type"],"support_target":old["support_target"],"bbox_iou":old["validation_metrics"]["bbox_iou"],"centroid_error_pixels":old["validation_metrics"]["centroid_error_pixels"],"dimensions":old["transform"]["dimensions"]},"after":{"support_type":item["support_type"],"support_target":item["support_target"],"placement_classification":item["placement_classification"],"bbox_iou":item["validation_metrics"]["bbox_iou"],"full_hull_iou":item["validation_metrics"]["full_hull_iou"],"visible_hull_iou":item["validation_metrics"]["visible_hull_iou"],"centroid_error_pixels":item["validation_metrics"]["centroid_error_pixels"],"dimensions":item["transform"]["dimensions"],"occlusion":item["occlusion"],"hard_gates":item["placement_hard_gates"]}})
    result={"schema_version":"1.0","before_plan":str(before_path),"after_plan":str(after_path),"object_count":len(objects),"objects":objects,"placement_counts":load(after_path.parent/"confidence_report.json")["placement_counts"],"passed":len(objects)==20 and all(x["orientation_rotation_unchanged_during_fitting"] for x in objects)};output.mkdir(parents=True,exist_ok=True);(output/"placement_evaluation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");focus=[x for x in objects if x["semantic_label"] in {"desktop_box","desk_chair"}];lines=["# Support-aware placement evaluation","",f"- Passed: {result['passed']}","","| Object | Support before/after | Bbox IoU | Centroid px | Placement |","|---|---|---|---|---|"]
    for x in focus:lines.append(f"| {x['semantic_label']} | {x['before']['support_type']} -> {x['after']['support_type']} | {x['before']['bbox_iou']:.3f} -> {x['after']['bbox_iou']:.3f} | {x['before']['centroid_error_pixels']:.2f} -> {x['after']['centroid_error_pixels']:.2f} | {x['after']['placement_classification']} |")
    (output/"placement_evaluation.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return result


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--before",type=Path,default=ROOT/"outputs"/"office_test"/"unified_v3_clean"/"unified_scene_plan.json");parser.add_argument("--after",type=Path,default=ROOT/"outputs"/"office_test"/"unified_v3_clean_support_fit"/"unified_scene_plan.json");parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"office_test"/"unified_v3_support_fit_evaluation");args=parser.parse_args();result=evaluate(args.before.resolve(),args.after.resolve(),args.output.resolve());print(json.dumps({"output":str(args.output),"passed":result["passed"],"placement_counts":result["placement_counts"]},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
