"""标记行候选的结构/退化边界合同；真实完整阶段另作benchmark。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.mris_register_nonlinear import face_area_normals, ordered_neighbors_from_faces
from fnit.recon_all.mris_register_overlap import remove_overlap_sphere
from fnit.recon_all.mris_register_overlap_marked import (
    OverlapTopology, negative_sphere_faces, remove_overlap_sphere_marked)
from fnit.recon_all.sphere_standard_finish import finish_standard_sphere


def octahedron(device):
    xyz=torch.tensor([[100.,0,0],[-100.,0,0],[0,100.,0],[0,-100.,0],
                      [0,0,100.],[0,0,-100.],[18.1,31.5,42.3]],device=device)
    faces=torch.tensor([[0,2,4],[2,1,4],[1,3,4],[3,0,4],
                        [2,0,5],[1,2,5],[3,1,5],[0,3,5]],device=device)
    faces[0]=faces[0].flip(0)
    return xyz,faces


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_negative_faces_zero_underflow_and_finite(device):
    if device.startswith("cuda") and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    torch.manual_seed(17)
    xyz=torch.randn(128,3,device=device)*100
    faces=torch.randint(0,128,(300,3),device=device)
    expected=face_area_normals(xyz,faces,signed_sphere=True)[0]<0
    assert torch.equal(negative_sphere_faces(xyz,faces),expected)
    boundary=torch.tensor([[0,0,100],[1e-12,0,100],[0,-1e-12,100],
                           [1,1,1],[1,1,1],[1,1,1]],dtype=torch.float32,device=device)
    triangles=torch.tensor([[0,1,2],[3,4,5]],device=device)
    expected=face_area_normals(boundary,triangles,signed_sphere=True)[0]<0
    assert not expected.any()  # FP32平方下溢后的signed area为-0。
    assert torch.equal(negative_sphere_faces(boundary,triangles),expected)


def test_negative_faces_dtype_is_explicit():
    with pytest.raises(ValueError,match="float32"):
        negative_sphere_faces(torch.zeros(3,3,dtype=torch.float64),torch.tensor([[0,1,2]]))


def test_topology_full_order_includes_isolated_and_duplicate_faces():
    xyz,faces=octahedron("cpu")
    faces=torch.cat((faces,faces[:1]))
    context=OverlapTopology(triangles=faces,nvertices=len(xyz))
    neighbors,degrees=ordered_neighbors_from_faces(faces,len(xyz))
    assert torch.equal(context.neighbors,neighbors)
    assert torch.equal(context.degrees,degrees)
    assert degrees[-1]==0


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_complete_rules_marked_rows_keep_full_radial_projection(device):
    if device.startswith("cuda") and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    xyz,faces=octahedron(device);copy=xyz.clone()
    expected,history=remove_overlap_sphere(xyz,faces,start_iteration=13,max_iterations=40)
    context=OverlapTopology(triangles=faces,nvertices=len(xyz))
    actual,actual_history=remove_overlap_sphere_marked(
        xyz,faces,start_iteration=13,max_iterations=40,topology=context)
    assert torch.equal(actual,expected)
    assert actual_history==history and len(history)>0
    assert torch.equal(xyz,copy)
    # 孤立未标顶点也必须继续原全体projection，不能保留其初始半径。
    assert abs(float(torch.linalg.vector_norm(actual[-1]))-100)<1e-4


def test_cache_identity_rejected_even_if_no_negative_faces():
    xyz,faces=octahedron("cpu")
    context=OverlapTopology(triangles=faces.flip(0),nvertices=len(xyz))
    with pytest.raises(ValueError,match="different ordered"):
        remove_overlap_sphere_marked(xyz,faces,topology=context)


def test_finish_backend_explicit_and_default_preserved():
    xyz,faces=octahedron("cpu")
    with pytest.raises(ValueError,match="overlap_backend"):
        finish_standard_sphere(np.zeros((3,3),np.float32),np.array([[0,1,2]]),
                               start_iteration=0,overlap_backend="approximate")
    # 无负面早退用正常octahedron；默认与显式dense逐位相同。
    faces[0]=faces[0].flip(0)
    default=finish_standard_sphere(xyz.numpy(),faces.numpy(),start_iteration=0)
    dense=finish_standard_sphere(xyz.numpy(),faces.numpy(),start_iteration=0,overlap_backend="dense")
    assert np.array_equal(default[0],dense[0]) and default[1]==dense[1]
