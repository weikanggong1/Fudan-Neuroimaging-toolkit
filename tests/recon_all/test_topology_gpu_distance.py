"""保守GPU块索引必须覆盖每个体素的全部原有序面，保留重复行。"""
import numpy as np
import pytest
from fnit.recon_all.topology_gpu_distance import _block_rows


@pytest.mark.parametrize('block',[32,64,128])
def test_block_rows_covers_wrapped_blocks_and_keeps_source_order(block):
    shape=np.array([3,4,11],np.int64)
    bounds=np.array([[8,0,1,0,2,2,7],[8,0,1,0,2,2,7],[1,2,2,3,3,10,10],[4,1,2,1,3,0,10]],np.int64)
    offsets,rows=_block_rows(bounds,shape,block)
    for index in range(int(np.prod(shape))):
        i=index//(shape[1]*shape[2]);j=index//shape[2]%shape[1];k=index%shape[2]
        expected=[r for r,b in enumerate(bounds)if b[1]<=i<=b[2]and b[3]<=j<=b[4]and b[5]<=k<=b[6]]
        group=rows[offsets[index//block]:offsets[index//block+1]]
        assert np.all(np.diff(group)>0)
        assert all(r in group for r in expected)
        actual=[r for r in group if bounds[r,1]<=i<=bounds[r,2]and bounds[r,3]<=j<=bounds[r,4]and bounds[r,5]<=k<=bounds[r,6]]
        assert actual==expected


def test_empty_face_blocks_are_explicit():
    offsets,rows=_block_rows(np.empty((0,7),np.int64),np.array([2,3,4],np.int64),32)
    np.testing.assert_array_equal(offsets,[0,0]);assert rows.dtype==np.int64 and rows.size==0


def test_large_grid_rejected_before_cuda_allocation():
    from fnit.recon_all.topology_gpu_distance import gpu_defect_signed_distance
    with pytest.raises(ValueError,match='signed 32-bit linear indexing'):
        gpu_defect_signed_distance(original_vertices=np.empty((0,3),np.float32),faces=np.empty((0,3),np.int64),face_normals=np.empty((0,3),np.float32),incident_offsets=np.array([0],np.int64),incident_faces=np.empty(0,np.int64),incident_corners=np.empty(0,np.int64),volume_shape=np.array([2**31,1,1],np.int64),volume_origin_mm=np.zeros(3,np.float32),volume_index_scale_per_mm=np.ones(3,np.float32),device='cuda:0')


@pytest.mark.parametrize('block',[64.0,np.float64(64),True,np.bool_(False),'64',None,0,63])
def test_invalid_block_rejected_before_device_or_allocations(block,monkeypatch):
    import fnit.recon_all.topology_gpu_distance as module
    def forbidden(*args,**kwargs):
        pytest.fail('invalid block reached device setup or CUDA allocation')
    monkeypatch.setattr(module.torch,'device',forbidden)
    monkeypatch.setattr(module.torch,'tensor',forbidden)
    monkeypatch.setattr(module.torch,'empty',forbidden)
    with pytest.raises(ValueError,match='integer 32,64 or 128'):
        module.gpu_defect_signed_distance(original_vertices=None,faces=None,face_normals=None,incident_offsets=None,incident_faces=None,incident_corners=None,volume_shape=None,volume_origin_mm=None,volume_index_scale_per_mm=None,device='cuda:0',block_voxels=block)


@pytest.mark.parametrize('block',[np.int32(32),np.int64(128)])
def test_numpy_integer_block_passes_type_validation(block):
    from fnit.recon_all.topology_gpu_distance import gpu_defect_signed_distance
    with pytest.raises(ValueError,match='signed 32-bit linear indexing'):
        gpu_defect_signed_distance(original_vertices=np.empty((0,3),np.float32),faces=np.empty((0,3),np.int64),face_normals=np.empty((0,3),np.float32),incident_offsets=np.array([0],np.int64),incident_faces=np.empty(0,np.int64),incident_corners=np.empty(0,np.int64),volume_shape=np.array([2**31,1,1],np.int64),volume_origin_mm=np.zeros(3,np.float32),volume_index_scale_per_mm=np.ones(3,np.float32),device='cuda:0',block_voxels=block)
