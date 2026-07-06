"""Final oriented-box support contact propagation.

Support evidence is inferred from visible geometry, but final contact must be
computed from the final parent and child primitives.  This module performs
that last deterministic pass without changing rotations or dimensions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("support normal is near zero")
    return vector / length


def oriented_box_extent(transform: dict[str, Any], normal: np.ndarray) -> float:
    """Half extent of an OBB along an arbitrary unit direction."""
    normal = _unit(normal)
    rotation = np.asarray(transform["rotation_matrix"], dtype=np.float64)
    dimensions = np.asarray(transform["dimensions"], dtype=np.float64)
    return float(sum(abs(float(normal @ rotation[:, index])) * dimensions[index] / 2.0 for index in range(3)))


def support_face_normal(parent_transform: dict[str, Any], evidence_normal: np.ndarray) -> np.ndarray:
    """Select the final parent OBB face most aligned with support evidence."""
    evidence = _unit(evidence_normal)
    rotation = np.asarray(parent_transform["rotation_matrix"], dtype=np.float64)
    dots = np.asarray([float(evidence @ rotation[:, index]) for index in range(3)])
    index = int(np.argmax(np.abs(dots)))
    return rotation[:, index] * (1.0 if dots[index] >= 0 else -1.0)


def support_gap(parent: dict[str, Any], child: dict[str, Any], normal: np.ndarray) -> float:
    """Signed child-bottom minus parent-top separation along ``normal``."""
    normal = _unit(normal)
    parent_transform = parent["transform"]
    child_transform = child["transform"]
    parent_top = float(normal @ np.asarray(parent_transform["center"]) + oriented_box_extent(parent_transform, normal))
    child_bottom = float(normal @ np.asarray(child_transform["center"]) - oriented_box_extent(child_transform, normal))
    return child_bottom - parent_top


def topological_support_order(object_ids: list[str], assignments: dict[str, dict[str, Any]]) -> list[str]:
    """Order supporters before descendants, rejecting support cycles."""
    known = set(object_ids)
    children = {object_id: [] for object_id in object_ids}
    indegree = {object_id: 0 for object_id in object_ids}
    for child_id, assignment in assignments.items():
        parent_id = assignment.get("target") if assignment.get("type") == "object" else None
        if child_id in known and parent_id in known:
            children[parent_id].append(child_id)
            indegree[child_id] += 1
    order_index = {object_id: index for index, object_id in enumerate(object_ids)}
    ready = sorted((object_id for object_id in object_ids if indegree[object_id] == 0), key=order_index.get)
    result: list[str] = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for child in sorted(children[current], key=order_index.get):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=order_index.get)
    if len(result) != len(object_ids):
        cycle = sorted(object_id for object_id, degree in indegree.items() if degree)
        raise ValueError(f"object support graph contains a cycle: {cycle}")
    return result


def propagate_support_contacts(objects: list[dict[str, Any]], assignments: dict[str, dict[str, Any]], tolerance: float = 1e-6) -> dict[str, Any]:
    """Translate supported descendants along only their final support normals."""
    by_id = {item["object_id"]: item for item in objects}
    order = topological_support_order(list(by_id), assignments)
    records: list[dict[str, Any]] = []
    supported_children: set[str] = set()
    for child_id in order:
        assignment = assignments.get(child_id, {})
        if assignment.get("type") != "object" or assignment.get("target") not in by_id:
            continue
        parent_id = assignment["target"]
        parent, child = by_id[parent_id], by_id[child_id]
        evidence_normal = _unit(np.asarray(assignment.get("support_normal", [0.0, 0.0, 1.0]), dtype=np.float64))
        normal = support_face_normal(parent["transform"], evidence_normal)
        assignment["support_normal_evidence"] = evidence_normal.tolist()
        assignment["support_normal"] = normal.tolist()
        before_center = np.asarray(child["transform"]["center"], dtype=np.float64)
        before_dimensions = np.asarray(child["transform"]["dimensions"], dtype=np.float64).copy()
        before_rotation = np.asarray(child["transform"]["rotation_matrix"], dtype=np.float64).copy()
        gap_before = support_gap(parent, child, normal)
        translation = -gap_before * normal
        child["transform"]["center"] = (before_center + translation).tolist()
        gap_after = support_gap(parent, child, normal)
        if abs(gap_after) > tolerance:
            raise RuntimeError(f"support propagation failed for {child_id}: gap={gap_after}")
        if not np.array_equal(before_dimensions, np.asarray(child["transform"]["dimensions"])):
            raise RuntimeError("support propagation changed child dimensions")
        if not np.array_equal(before_rotation, np.asarray(child["transform"]["rotation_matrix"])):
            raise RuntimeError("support propagation changed child rotation")
        record = {
            "parent_object_id": parent_id,
            "child_object_id": child_id,
            "support_normal": normal.tolist(),
            "support_normal_evidence": evidence_normal.tolist(),
            "gap_before": float(gap_before),
            "gap_after": float(gap_after),
            "translation": translation.tolist(),
            "translation_magnitude": float(np.linalg.norm(translation)),
            "child_center_before": before_center.tolist(),
            "child_center_after": child["transform"]["center"],
            "rotation_unchanged": True,
            "dimensions_unchanged": True,
        }
        child["support_contact_propagation"] = record
        assignment["final_support_gap"] = float(gap_after)
        assignment["final_parent_support_plane"] = float(normal @ np.asarray(parent["transform"]["center"]) + oriented_box_extent(parent["transform"], normal))
        records.append(record)
        supported_children.add(child_id)
    return {
        "topological_order": order,
        "records": records,
        "supported_child_count": len(records),
        "maximum_final_gap": float(max((abs(record["gap_after"]) for record in records), default=0.0)),
        "unsupported_object_ids": [object_id for object_id in order if object_id not in supported_children],
    }
