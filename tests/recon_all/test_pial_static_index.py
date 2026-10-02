import numpy as np
import pytest
from fnit.recon_all.place_surface_normals import FaceNormalTopology
from fnit.recon_all.place_surface_repulsion import OriginalVertexBuckets, original_vertex_normals, vertex_buckets

def test_full_bucket_order_ripping_and_changed_current_coordinates():
    original=np.array([[.2,.2,.2],[.1,.1,.1],[1.2,.2,.2],[.3,.3,.3]],np.float32)
    ripped=np.array([False,False,False,True])
    index=OriginalVertexBuckets(original,ripped)
    offsets,ids=index.query(original)
    np.testing.assert_array_equal(offsets,[0,2,4,5,5])
    np.testing.assert_array_equal(ids,[0,1,0,1,2])
    current=original.copy();current[0]=[1.5,.2,.2]
    offsets,ids=index.query(current)
    np.testing.assert_array_equal(offsets,[0,1,3,4,4])
    np.testing.assert_array_equal(ids,[2,0,1,2])
    original[:] = 100  # 缓存拥有固定原始键，不读取外部变化坐标。
    ripped[:] = True
    np.testing.assert_array_equal(index.query(current)[1],[2,0,1,2])

def test_hash_key_keeps_float32_addition_and_truncation_toward_zero():
    xyz=np.array([[-1000.25,0,0],[-999.75,0,0],[-1001.25,0,0]],np.float32)
    offsets,ids=vertex_buckets(xyz,xyz,np.zeros(3,bool))
    np.testing.assert_array_equal(offsets,[0,2,4,5])
    np.testing.assert_array_equal(ids,[0,1,0,1,2])

def test_original_normals_cache_preserves_own_definition_and_recomputes_geometry():
    xyz=np.array([[0,0,0],[1,0,0],[.4,1,0],[0,0,2]],np.float32)
    faces=np.array([[0,1,2],[0,2,3],[0,3,1]],np.int32)
    topology=FaceNormalTopology(faces,len(xyz))
    old=original_vertex_normals(xyz,faces)
    np.testing.assert_array_equal(original_vertex_normals(xyz,faces,topology=topology),old)
    shifted=xyz.copy();shifted[3]=[.6,.6,2]
    current=original_vertex_normals(shifted,faces,topology=topology)
    np.testing.assert_array_equal(current,original_vertex_normals(shifted,faces))
    assert not np.array_equal(current,old)
    with pytest.raises(ValueError):original_vertex_normals(xyz,faces[:,::-1],topology=topology)

def test_empty_buckets_and_shape_validation():
    xyz=np.ones((3,3),np.float32)
    index=OriginalVertexBuckets(xyz,np.ones(3,bool))
    np.testing.assert_array_equal(index.query(xyz)[0],np.zeros(4,np.int32))
    assert len(index.query(xyz)[1])==0
    with pytest.raises(ValueError):index.query(xyz[:2])
