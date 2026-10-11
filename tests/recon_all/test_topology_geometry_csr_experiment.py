"""整数CSR合同；人工网格只用于次序/失败/状态隔离，不作为benchmark。"""
import numpy as np
import pytest
from benchmark_recon_topology_geometry_csr import counted_face_csr
from fnit.recon_all.place_surface_normals import ordered_face_csr


@pytest.mark.parametrize('faces,nvertices',[
 (np.array([[2,0,2],[1,0,2],[2,0,2],[0,0,0]],np.int32),5),
 (np.empty((0,3),np.int64),0),
 (np.empty((0,3),np.int32),4),
 (np.array([[3,2,1],[0,1,2],[1,1,1]],np.int64)[::-1],5),
])
def test_complete_source_order_duplicates_corners_and_empty_rows(faces,nvertices):
    original=faces.copy();expected=ordered_face_csr(faces,nvertices)
    actual=counted_face_csr(faces,np.int64(nvertices))
    for a,b in zip(actual,expected):
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    np.testing.assert_array_equal(faces,original)
    if len(actual[1]):
        actual[1][0]=-9
        again=counted_face_csr(faces,nvertices)
        np.testing.assert_array_equal(again[1],expected[1])


def test_changed_candidate_faces_never_reuses_old_topology():
    faces=np.array([[0,1,2]],np.int32);old=counted_face_csr(faces,4)
    faces[0,2]=3;new=counted_face_csr(faces,4)
    assert old[0].tobytes()!=new[0].tobytes()
    for a,b in zip(new,ordered_face_csr(faces,4)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('faces,nvertices',[
 (np.array([[0,1,2]],np.float32),3),
 (np.array([0,1,2]),3),
 (np.array([[-1,1,2]]),3),
 (np.array([[0,1,3]]),3),
 (np.empty((0,3),np.int32),True),
 (np.empty((0,3),np.int32),3.0),
 (np.empty((0,3),np.int32),-1),
])
def test_invalid_input_fails_before_compiled_allocation(faces,nvertices,monkeypatch):
    import benchmark_recon_topology_geometry_csr as module
    def forbidden(*args):pytest.fail('invalid input reached allocating kernel')
    monkeypatch.setattr(module,'_counted_csr',forbidden)
    with pytest.raises(ValueError):module.counted_face_csr(faces,nvertices)
