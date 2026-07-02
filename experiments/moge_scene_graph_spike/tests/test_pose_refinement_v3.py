import numpy as np
import pytest

from src.pose_refinement_v3 import (
    angle_distance_180,
    classify_image_edge_families,
    cluster_azimuths_mod180,
    estimate_horizontal_frame,
    footprint_yaw_observable,
    orientation_score,
)


def normals_at(angles, count=80, z=0.0):
    rows=[]
    for angle in angles:
        radians=np.radians(angle)
        rows.extend([[np.cos(radians),np.sin(radians),z]]*count)
    return np.asarray(rows,float)


def test_sign_invariant_normal_clustering():
    normals=np.vstack([normals_at([32],60),normals_at([212],60)])
    frame=estimate_horizontal_frame(normals)
    assert frame["reliable_cluster_count"]==1
    assert angle_distance_180(frame["clusters"][0]["azimuth_degrees_mod180"],32)<1e-6


def test_azimuth_clustering_wraps_modulo_180():
    clusters=cluster_azimuths_mod180(np.asarray([179,1,0,180,359]),threshold_degrees=5)
    assert len(clusters)==1
    assert angle_distance_180(clusters[0]["azimuth_degrees_mod180"],0)<1.1


def test_two_perpendicular_faces_recover_manhattan_frame():
    frame=estimate_horizontal_frame(normals_at([27,117],100))
    assert frame["supported_visible_faces"]==2
    assert frame["orthogonality_error_degrees"]==pytest.approx(0,abs=1e-6)
    assert frame["ambiguity"]=="none"
    assert min(angle_distance_180(frame["normal_derived_yaw_degrees"],27),angle_distance_180(frame["normal_derived_yaw_degrees"],117))<1e-6


def test_one_face_preserves_90_degree_ambiguity():
    frame=estimate_horizontal_frame(normals_at([41],120))
    assert frame["supported_visible_faces"]==1
    assert frame["ambiguity"]=="single_face_90_degree"
    assert frame["orientation_confidence"]<0.7


def test_square_footprint_yaw_is_unobservable():
    assert not footprint_yaw_observable(np.asarray([0.30,0.31,0.5]))
    assert footprint_yaw_observable(np.asarray([0.30,0.50,0.5]))


def test_projected_vertical_edges_are_excluded_from_yaw_families():
    families=classify_image_edge_families(np.asarray([88,91,25,27]),np.asarray([10,12,20,18]),90)
    assert families["vertical_segment_count"]==2
    assert families["horizontal_segment_count"]==2
    assert len(families["horizontal_clusters"])==1
    assert angle_distance_180(families["horizontal_clusters"][0]["azimuth_degrees_mod180"],26)<2


def test_high_iou_wrong_yaw_fails_orientation_score():
    result=orientation_score({"normal_residual_degrees":35,"horizontal_edge_error_degrees":32,"hull_iou":0.99,"centroid_error_pixels":0.2,"support_error":0,"collision_risk":0,"normal_coverage":0.8})
    assert result["score"]<0.6
    assert not result["orientation_valid"]
    assert "horizontal_edge_error_above_15_degrees" in result["hard_review_gates"]


def test_open_box_noisy_normals_reduce_confidence():
    rng=np.random.default_rng(7)
    angles=rng.uniform(0,180,400)
    frame=estimate_horizontal_frame(normals_at(angles,1))
    assert frame["orientation_confidence"]<0.5
