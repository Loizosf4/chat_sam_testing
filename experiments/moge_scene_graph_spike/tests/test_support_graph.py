import numpy as np

from src.support_graph import infer_support_graph


def grid_box(x0,x1,y0,y1,z0,z1,count=10):
    x,y,z=np.meshgrid(np.linspace(x0,x1,count),np.linspace(y0,y1,count),np.linspace(z0,z1,3),indexing="ij")
    return np.column_stack([x.ravel(),y.ravel(),z.ravel()])


def item(object_id,label,points,mask,depth):
    normals=np.zeros_like(points);normals[:,2]=1
    return {"object_id":object_id,"semantic_label":label,"points":points,"normals":normals,"mask":mask,"depth":np.full(len(points),depth)}


def test_small_box_is_supported_by_desk():
    desk_mask=np.zeros((60,60),bool);desk_mask[25:48,10:50]=1;box_mask=np.zeros_like(desk_mask);box_mask[15:25,27:38]=1
    desk=item("desk","work_desk",grid_box(-1,1,-.6,.6,0,.75),desk_mask,5.0)
    box=item("box","small_box",grid_box(-.2,.2,-.2,.2,.77,1.05),box_mask,4.9)
    graph=infer_support_graph([desk,box])
    assert graph["assignments"]["box"]["type"]=="object"
    assert graph["assignments"]["box"]["target"]=="desk"
    assert graph["assignments"]["box"]["support_top_z"]>0.7


def test_support_is_not_inferred_without_overlap():
    a_mask=np.zeros((30,30),bool);a_mask[2:8,2:8]=1;b_mask=np.zeros_like(a_mask);b_mask[20:28,20:28]=1
    support=item("a","desk",grid_box(-1,-.5,-1,-.5,0,.7),a_mask,5)
    subject=item("b","small_box",grid_box(.5,1,.5,1,.72,1),b_mask,5)
    graph=infer_support_graph([support,subject])
    assert graph["assignments"]["b"]["type"]!="object"

