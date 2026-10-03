"""独立行并行保持原有float32算术；共享缓存按实际坐标重算。"""
import numpy as np
import torch
from numba import njit
from fnit.recon_all.place_surface_normals import _normals, ordered_face_csr
from fnit.recon_all.sphere_standard_unfold import _distance_force, _face_geometry
from fnit.recon_all.mris_register_objective import (_rotation_geometry, _project_rigid_positions,
    _rigid_sse_projected, rigid_sse)


def test_parallel_geometry_matches_serial_compiled_rows():
    rng=np.random.default_rng(113)
    xyz=rng.normal(size=(129,3)).astype(np.float32)
    xyz*=100/np.linalg.norm(xyz,axis=1)[:,None]
    faces=rng.integers(0,len(xyz),size=(180,3),dtype=np.int64)
    off,fid,corner=ordered_face_csr(faces,len(xyz))
    actual=_normals(xyz,faces,fid,corner,off)
    expected=njit(_normals.py_func)(xyz,faces,fid,corner,off)
    np.testing.assert_array_equal(actual,expected)
    for a,b in zip(_face_geometry(xyz,faces),njit(_face_geometry.py_func)(xyz,faces)):
        np.testing.assert_array_equal(a,b)
    neighbors=rng.integers(0,len(xyz),size=len(xyz)*12,dtype=np.int32)
    offsets=np.arange(len(xyz)+1,dtype=np.int64)*12
    distances=rng.uniform(.1,4,size=len(neighbors)).astype(np.float32)
    args=(xyz,actual,offsets,neighbors,distances,np.float32(.7),np.float32(.1))
    np.testing.assert_array_equal(_distance_force(*args),njit(_distance_force.py_func)(*args))


def test_rigid_sampling_geometry_is_local_to_beta_gamma():
    rng=torch.Generator().manual_seed(44)
    vertices=torch.randn((1009,3),generator=rng)*100
    curvature=torch.randn((1009,),generator=rng)
    mean=torch.randn((32,16),generator=rng)
    variance=torch.rand((32,16),generator=rng)+.1
    projected=_project_rigid_positions(vertices,100.)
    for beta,gamma in [(.15,-.3),(0.,0.),(-.2,.1)]:
        geometry=_rotation_geometry(projected,mean.shape,beta,gamma,100.)
        for alpha in [-.41,0.,.17]:
            expected=rigid_sse(vertices,curvature,mean,variance,(alpha,beta,gamma))
            actual=_rigid_sse_projected(projected,curvature,mean,variance,(alpha,beta,gamma),geometry=geometry)
            assert actual==expected
