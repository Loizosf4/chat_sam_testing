import numpy as np

from src.mask_constrained_geometry import clean_masked_geometry, erode_object_mask


def test_mask_erosion_removes_boundary_leakage():
    mask=np.ones((20,20),bool);interior,report=erode_object_mask(mask,2)
    assert interior.sum()==16*16
    assert report["retained_ratio"]<1


def test_depth_separated_support_pixels_are_rejected():
    mask=np.ones((24,24),bool);valid=np.ones_like(mask);depth=np.full(mask.shape,5.0);depth[:2]=10;depth[-2:]=10;depth[:,:2]=10;depth[:,-2:]=10
    yy,xx=np.mgrid[:24,:24];points=np.stack([xx/100,yy/100,depth],axis=2);normals=np.zeros_like(points);normals[...,2]=1
    result=clean_masked_geometry(mask,points,depth,normals,valid,2)
    assert result["success"]
    assert np.median(result["depth"])==5
    assert result["depth"].max()<6
    assert result["rejected_samples"]==0

