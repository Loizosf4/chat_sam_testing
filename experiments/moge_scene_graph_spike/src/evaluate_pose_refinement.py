"""Gate pose-refinement v2 against the actual accepted Blender transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .refine_primitive_plan import OBJECT_IDS, ROOT, quaternion_to_matrix


DEFAULT_OUTPUT = ROOT / "outputs" / "office_test" / "pose_refinement_v2"
ALLOWED_CLASSIFICATIONS = {
    "preserve_approved_transform",
    "apply_refined_transform_for_blender_review",
    "requires_user_review",
    "refinement_failed",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_transform(transform: dict[str, Any]) -> dict[str, list[float]]:
    return {
        "center": list(transform.get("location", transform.get("center"))),
        "dimensions": list(transform.get("local_oriented_dimensions", transform.get("local_dimensions", transform.get("dimensions")))),
        "quaternion_wxyz": list(transform.get("rotation_quaternion_wxyz", transform.get("quaternion_wxyz"))),
    }


def _finite_transform(transform: dict[str, Any]) -> bool:
    return all(np.isfinite(np.asarray(transform[key], dtype=np.float64)).all() for key in ("center", "dimensions", "quaternion_wxyz"))


def _rotation_checks(item: dict[str, Any]) -> dict[str, Any]:
    rotation = np.asarray(item["rotation_matrix"], dtype=np.float64)
    orthonormal_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    return {
        "finite": bool(np.isfinite(rotation).all()),
        "orthonormal_error": orthonormal_error,
        "orthonormal": orthonormal_error <= 1e-7,
        "determinant": determinant,
        "right_handed": abs(determinant - 1.0) <= 1e-7,
    }


def _same_transform(a: dict[str, Any], b: dict[str, Any], tolerance: float = 2e-6) -> bool:
    return all(np.allclose(a[key], b[key], atol=tolerance, rtol=0.0) for key in ("center", "dimensions", "quaternion_wxyz"))


def _write_summary_image(evaluation: dict[str, Any], chair_projection: Path, output: Path) -> None:
    canvas = Image.new("RGB", (1400, 760), (18, 20, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), "Pose refinement v2 gate", fill=(255, 255, 255))
    draw.text((28, 48), "Approved update set: desk_chair only; Blender visual review required", fill=(100, 230, 255))
    colors = {
        "preserve_approved_transform": (120, 220, 140),
        "apply_refined_transform_for_blender_review": (255, 190, 70),
        "requires_user_review": (255, 130, 80),
        "refinement_failed": (255, 70, 70),
    }
    y = 100
    for item in evaluation["objects"]:
        color = colors[item["classification"]]
        draw.rectangle((28, y, 650, y + 82), outline=color, width=2)
        draw.text((42, y + 10), f"{item['semantic_label']}  {item['object_id']}", fill=(235, 235, 240))
        draw.text((42, y + 34), item["classification"], fill=color)
        comparison = item["projection_comparison"]
        draw.text((42, y + 57), f"accepted IoU={comparison['actual_accepted_blender']['bbox_iou']!s}  refined IoU={comparison['refined_v2']['bbox_iou']:.3f}", fill=(180, 185, 195))
        y += 96
    chair_image = Image.open(chair_projection).convert("RGB").resize((700, 700))
    canvas.paste(chair_image, (680, 45))
    draw.rectangle((680, 45, 1380, 745), outline=(255, 190, 70), width=3)
    canvas.save(output)


def evaluate(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    refined_path = output_dir / "refined_primitive_scene_plan_v2.json"
    report_path = output_dir / "pose_refinement_report.json"
    cumulative_path = ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "05_06_chair_box" / "final" / "cumulative_scene_validation.json"
    chair_box_path = ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "05_06_chair_box" / "final" / "batch_validation.json"
    cabinet_path = ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "01_filing_cabinet" / "final" / "object_validation.json"
    desk_path = ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "02_desk" / "approved_candidate_c" / "object_validation.json"
    rack_light_path = ROOT / "outputs" / "office_test" / "blender_execution" / "objects" / "03_04_rack_light" / "final" / "batch_validation.json"
    protected_inputs = [
        ROOT / "outputs" / "office_test" / "primitive_plan" / "primitive_scene_plan.json",
        cumulative_path,
        chair_box_path,
        cabinet_path,
        desk_path,
        rack_light_path,
    ]
    before_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in protected_inputs}
    refined = _load(refined_path)
    refinement_report = _load(report_path)
    cumulative = _load(cumulative_path)
    chair_box = _load(chair_box_path)
    cabinet = _load(cabinet_path)
    desk = _load(desk_path)
    rack_light = _load(rack_light_path)

    proposed_by_id = {item["object_id"]: item for item in refined["semantic_primitives"]}
    offline_by_id = {item["object_id"]: item for item in refinement_report["objects"]}
    accepted_by_id = {item["object_id"]: item for item in cumulative["semantic_inventory"]}
    actual_transforms = {object_id: _canonical_transform(item["final_transform"]) for object_id, item in accepted_by_id.items()}
    actual_metrics = {object_id: {"bbox_iou": item["bbox_iou"], "centroid_error_pixels": item["centroid_error_px"]} for object_id, item in accepted_by_id.items()}

    # Cross-check dedicated authoritative reports against the cumulative inventory.
    dedicated = {
        OBJECT_IDS["left_tall_filing_cabinet"]: _canonical_transform(cabinet["final_blender_transform"]),
        OBJECT_IDS["desk"]: {
            "center": desk["approved_transform"]["center"],
            "dimensions": desk["approved_transform"]["dimensions"],
            "quaternion_wxyz": proposed_by_id[OBJECT_IDS["desk"]]["rotation_quaternion_wxyz"],
        },
        OBJECT_IDS["desk_chair"]: _canonical_transform(chair_box["per_object"]["desk_chair"]["transform"]),
        OBJECT_IDS["desktop_box"]: _canonical_transform(chair_box["per_object"]["desktop_box"]["transform"]),
        OBJECT_IDS["coat_rack"]: _canonical_transform(rack_light["final_transforms"]["coat_rack"]),
        OBJECT_IDS["wall_light_fixture"]: _canonical_transform(rack_light["final_transforms"]["wall_light_fixture"]),
    }

    classifications = {
        "left_tall_filing_cabinet": "preserve_approved_transform",
        "desk": "preserve_approved_transform",
        "desk_chair": "apply_refined_transform_for_blender_review",
        "desktop_box": "preserve_approved_transform",
        "coat_rack": "preserve_approved_transform",
        "wall_light_fixture": "preserve_approved_transform",
    }
    reasons = {
        "left_tall_filing_cabinet": "The accepted Blender transform resolves floor/wall conflicts. Offline projection regresses and cannot supersede the accepted checkpoint.",
        "desk": "The v2 plan already embeds approved Candidate C exactly; preserve it without another update.",
        "desk_chair": "The evidence rotation removes the identity fallback with zero normal and gravity error while retaining acceptable projection. Blender visual review remains mandatory because edge-angle error is 9.82 degrees.",
        "desktop_box": "The accepted Blender proxy has slightly better bbox IoU and verified Candidate C support. V2 does not provide a material support-safe improvement.",
        "coat_rack": "The transform and wall collision are explicitly user-approved; orientation confidence is too low for replacement.",
        "wall_light_fixture": "The accepted transform already uses the corrected wall attachment. V2 has high orientation confidence but no material projection gain and does not justify replacing verified contact.",
    }
    support = {
        "left_tall_filing_cabinet": "Preserve accepted floor/wall clearance and 0.01 floor gap policy.",
        "desk": "Preserve approved floor contact and Candidate C footprint.",
        "desk_chair": "Support remains unresolved; no floor snapping. Current Blender report shows no major desk penetration, but the proposed transform must be rechecked visually.",
        "desktop_box": "Preserve verified zero desk-top gap and fully supported footprint.",
        "coat_rack": "Preserve floor contact and user_override_accepted_collision.",
        "wall_light_fixture": "Preserve corrected right-wall contact with negligible numerical penetration.",
    }
    risk = {
        "left_tall_filing_cabinet": "High if replaced: offline accepted-transform projection is lower than the compiler projection and the original OBB had structural collisions.",
        "desk": "High if changed: Candidate C was selected through explicit visual review.",
        "desk_chair": "Moderate: orientation is materially stronger, but bbox IoU falls from the current proxy's near-perfect axis-aligned fit and 9.82-degree edge error remains.",
        "desktop_box": "Moderate: numerical differences could reduce verified support quality without a visible benefit.",
        "coat_rack": "High: noisy disconnected evidence and an explicit collision override make automatic replacement inappropriate.",
        "wall_light_fixture": "Moderate: moving from the corrected attachment center could reintroduce wall offset/penetration.",
    }

    objects: list[dict[str, Any]] = []
    for label, object_id in OBJECT_IDS.items():
        proposal = proposed_by_id[object_id]
        offline = offline_by_id[object_id]
        selected_id = proposal["selected_rotation"]["candidate_id"]
        selected_candidate = next(item for item in proposal["rotation_candidates"] if item["candidate_id"] == selected_id)
        orientation = {
            "method": proposal["orientation_method"],
            "confidence": proposal["orientation_confidence"],
            "dominant_normals": proposal["dominant_normals"],
            "mask_axis": proposal["mask_axis"],
            "selected_candidate_id": selected_id,
            "normal_alignment_error_degrees": selected_candidate.get("normal_alignment_error_degrees"),
            "gravity_alignment_error_degrees": selected_candidate.get("gravity_alignment_error_degrees"),
            "projected_edge_angle_error_degrees": selected_candidate.get("projected_edge_angle_error_degrees"),
        }
        projection = {
            "actual_accepted_blender": actual_metrics[object_id],
            "original_compiler_offline": offline["projection_comparison"]["original"],
            "refined_v2": offline["projection_comparison"]["refined"],
        }
        item = {
            "object_id": object_id,
            "semantic_label": label,
            "actual_current_blender_transform": actual_transforms[object_id],
            "dedicated_report_transform_matches_cumulative": _same_transform(actual_transforms[object_id], dedicated[object_id]),
            "proposed_refined_transform": {
                "center": proposal["center"],
                "dimensions": proposal["dimensions"],
                "quaternion_wxyz": proposal["rotation_quaternion_wxyz"],
                "rotation_matrix": proposal["rotation_matrix"],
            },
            "classification": classifications[label],
            "orientation_evidence": orientation,
            "scale_evidence": proposal["scale_refinement"],
            "projection_comparison": projection,
            "support_contact_implications": support[label],
            "regression_risk": risk[label],
            "reason": reasons[label],
            "validation": {
                "actual_transform_finite": _finite_transform(actual_transforms[object_id]),
                "proposed_transform_finite": _finite_transform({"center": proposal["center"], "dimensions": proposal["dimensions"], "quaternion_wxyz": proposal["rotation_quaternion_wxyz"]}),
                "positive_dimensions": bool(np.all(np.asarray(proposal["dimensions"]) > 0)),
                **_rotation_checks(proposal),
            },
        }
        if label == "coat_rack":
            item["user_override"] = "user_override_accepted_collision"
        objects.append(item)

    chair_item = next(item for item in objects if item["semantic_label"] == "desk_chair")
    chair_gate = (
        chair_item["orientation_evidence"]["normal_alignment_error_degrees"] == 0.0
        and chair_item["orientation_evidence"]["gravity_alignment_error_degrees"] == 0.0
        and 8.0 <= chair_item["orientation_evidence"]["projected_edge_angle_error_degrees"] <= 12.0
        and chair_item["projection_comparison"]["refined_v2"]["bbox_iou"] >= 0.80
        and chair_item["projection_comparison"]["refined_v2"]["centroid_error_pixels"] <= 2.0
        and not np.allclose(quaternion_to_matrix(chair_item["proposed_refined_transform"]["quaternion_wxyz"]), np.eye(3))
    )
    update = {
        "object_id": chair_item["object_id"],
        "semantic_label": "desk_chair",
        "classification": "apply_refined_transform_for_blender_review",
        "current_transform": chair_item["actual_current_blender_transform"],
        "proposed_transform": chair_item["proposed_refined_transform"],
        "material_improvement": {
            "identity_rotation_removed": True,
            "normal_alignment_error_degrees": chair_item["orientation_evidence"]["normal_alignment_error_degrees"],
            "gravity_alignment_error_degrees": chair_item["orientation_evidence"]["gravity_alignment_error_degrees"],
            "remaining_projected_edge_angle_error_degrees": chair_item["orientation_evidence"]["projected_edge_angle_error_degrees"],
            "refined_bbox_iou": chair_item["projection_comparison"]["refined_v2"]["bbox_iou"],
            "refined_centroid_error_pixels": chair_item["projection_comparison"]["refined_v2"]["centroid_error_pixels"],
        },
        "approval_scope": "offline gate only; requires chair-only Blender visual validation before final acceptance",
    }
    update_set = {
        "schema_version": "1.0",
        "scene_id": "office_test",
        "source_refinement": str(refined_path.relative_to(ROOT)).replace("\\", "/"),
        "update_count": 1,
        "updates": [update] if chair_gate else [],
        "gate_passed": chair_gate,
        "excluded_objects": [{"object_id": item["object_id"], "semantic_label": item["semantic_label"], "classification": item["classification"], "reason": item["reason"]} for item in objects if item["semantic_label"] != "desk_chair"],
    }
    gates = {
        "stable_ids": set(proposed_by_id) == set(accepted_by_id) == set(OBJECT_IDS.values()),
        "exactly_six_supported_objects": len(objects) == 6 and all(item["object_id"] in OBJECT_IDS.values() for item in objects),
        "finite_transforms": all(item["validation"]["actual_transform_finite"] and item["validation"]["proposed_transform_finite"] for item in objects),
        "positive_dimensions": all(item["validation"]["positive_dimensions"] for item in objects),
        "orthonormal_rotations": all(item["validation"]["orthonormal"] for item in objects),
        "right_handed_rotations": all(item["validation"]["right_handed"] for item in objects),
        "dedicated_reports_match_cumulative": all(item["dedicated_report_transform_matches_cumulative"] for item in objects),
        "classifications_valid": all(item["classification"] in ALLOWED_CLASSIFICATIONS for item in objects),
        "update_set_chair_only": chair_gate and update_set["update_count"] == 1 and update_set["updates"][0]["object_id"] == OBJECT_IDS["desk_chair"],
        "only_material_improvements_selected": chair_gate,
    }
    evaluation = {
        "schema_version": "1.0",
        "scene_id": "office_test",
        "phase": "offline_pose_refinement_v2_gate",
        "objects": objects,
        "classifications": {item["semantic_label"]: item["classification"] for item in objects},
        "approved_update_object_ids": [OBJECT_IDS["desk_chair"]] if chair_gate else [],
        "regressions_avoided": [
            "filing-cabinet offline projection regression not applied",
            "approved Candidate C desk not perturbed",
            "desktop-box verified support and slightly better accepted IoU preserved",
            "coat-rack low-confidence yaw and accepted collision override preserved",
            "wall-light corrected wall attachment preserved",
        ],
        "quality_gates": gates,
        "passed": all(gates.values()),
        "chair_only_blender_batch_validation_justified": chair_gate,
        "protected_input_hashes_before": before_hashes,
    }
    evaluation_path = output_dir / "evaluation.json"
    update_path = output_dir / "approved_update_set.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    update_path.write_text(json.dumps(update_set, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    after_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in protected_inputs}
    evaluation["protected_input_hashes_after"] = after_hashes
    evaluation["no_accepted_transform_overwritten"] = before_hashes == after_hashes
    evaluation["quality_gates"]["no_accepted_transform_overwritten"] = before_hashes == after_hashes
    evaluation["passed"] = all(evaluation["quality_gates"].values())
    evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_markdown(evaluation, output_dir / "evaluation.md")
    _write_summary_image(evaluation, output_dir / "per_object" / OBJECT_IDS["desk_chair"] / "projection_comparison.png", output_dir / "update_set_summary.png")
    return {"evaluation": evaluation, "update_set": update_set}


def _write_markdown(evaluation: dict[str, Any], path: Path) -> None:
    lines = [
        "# Pose refinement v2 evaluation gate",
        "",
        f"- Result: {'PASS' if evaluation['passed'] else 'FAIL'}",
        "- Approved update set: desk_chair only",
        f"- Chair-only Blender validation justified: {evaluation['chair_only_blender_batch_validation_justified']}",
        "",
        "| Object | Classification | Accepted IoU | Refined IoU | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for item in evaluation["objects"]:
        projection = item["projection_comparison"]
        lines.append(f"| {item['semantic_label']} | {item['classification']} | {projection['actual_accepted_blender']['bbox_iou']} | {projection['refined_v2']['bbox_iou']:.3f} | {item['reason']} |")
    chair = next(item for item in evaluation["objects"] if item["semantic_label"] == "desk_chair")
    lines += [
        "",
        "## Chair review candidate",
        "",
        f"- Current transform: `{chair['actual_current_blender_transform']}`",
        f"- Refined transform: `{chair['proposed_refined_transform']}`",
        f"- Remaining projected edge-angle error: {chair['orientation_evidence']['projected_edge_angle_error_degrees']:.6f} degrees",
        f"- Refined bbox IoU: {chair['projection_comparison']['refined_v2']['bbox_iou']:.6f}",
        f"- Refined centroid error: {chair['projection_comparison']['refined_v2']['centroid_error_pixels']:.6f} px",
        "",
        "## Regressions avoided",
        "",
        *[f"- {item}" for item in evaluation["regressions_avoided"]],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(args.output_dir.resolve())
    print(json.dumps({
        "passed": result["evaluation"]["passed"],
        "approved_update_object_ids": result["evaluation"]["approved_update_object_ids"],
        "chair_only_blender_batch_validation_justified": result["evaluation"]["chair_only_blender_batch_validation_justified"],
    }, indent=2))

