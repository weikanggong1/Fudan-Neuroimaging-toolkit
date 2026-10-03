"""GCAM solver regression: edge replication, frozen shells and stopping rules."""
import numpy as np
import pytest
import torch
from fnit.recon_all.ca_register_inverse_fill import voronoi_fill, soap_bubble_float
from fnit.recon_all.mni_warp_inverse import fill_inverse_fields


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA regression")
@pytest.mark.parametrize("edge", [False, True])
def test_gpu_fill_retains_complete_cpu_algorithm(edge):
    shape = (11, 9, 8)
    mask = np.zeros(shape, bool)
    if edge:
        mask[0, :3, 0] = True
        mask[-2, -2, -1] = True
    else:
        mask[4:7, 3:6, 2:5] = True
    rng = np.random.default_rng(102)
    seeds = np.where(mask[None], rng.uniform(0, 200, (3, *shape)).astype(np.float32), 0)
    expected, iterations, shells = [], [], []
    for seed in seeds:
        expanded, shell = voronoi_fill(seed, mask)
        result, count = soap_bubble_float(expanded, mask)
        expected.append(result)
        iterations.append(count)
        shells.append(shell)
    actual, report = fill_inverse_fields(seeds, mask, device="cuda:0")
    np.testing.assert_array_equal(actual, np.stack(expected))
    assert report["soap_iterations_xyz"] == iterations
    assert report["voronoi_iterations"] == shells[0]


def test_gpu_fill_never_silently_falls_back_to_cpu():
    with pytest.raises(RuntimeError, match="CUDA"):
        fill_inverse_fields(np.zeros((3,2,2,2),np.float32), np.ones((2,2,2),bool), device="cpu")


def test_forward_geometry_extension_round_trip(tmp_path):
    import nibabel as nib
    from fnit.recon_all.mni_warp_io import write_forward_warp, native_geometry
    from fnit.recon_all.ca_register_inverse import read_warp_geometries
    source=nib.Nifti1Image(np.zeros((9,8,7),np.uint8),np.diag([-1.,1.,1.,1.]))
    target=nib.Nifti1Image(np.zeros((7,6,5),np.uint8),np.eye(4))
    values=np.zeros((*target.shape,3),np.float32)
    path=tmp_path/'warp.nii.gz'
    write_forward_warp(values,source,target,path)
    image=nib.load(path)
    a,b,shape=read_warp_geometries(image)
    np.testing.assert_array_equal(a,native_geometry(source))
    np.testing.assert_array_equal(b,native_geometry(target))
    assert shape==source.shape
    assert image.shape==(*target.shape,1,3)
    assert image.header.get_intent()[0]=='displacement vector'


def test_absolute_sampler_handles_last_cell_and_outside_without_border_fill():
    from fnit.recon_all.mni_warp_sampling import _linear_absolute
    field=torch.arange(3*2*2*2,dtype=torch.float32).reshape(3,2,2,2)
    q=torch.tensor([[1.5,2.,-.1],[1.,1.,1.],[1.,1.,1.]]).reshape(3,3,1,1)
    result,valid=_linear_absolute(field,q)
    assert valid[:,0,0].tolist()==[True,False,False]
    torch.testing.assert_close(result[:,0,0,0],field[:,1,1,1])
    assert not result[:,1:].any()


def test_native_nearest_promotes_half_addition_and_matches_border_rint():
    from fnit.recon_all.mni_warp_sampling import _native_nearest_plan
    q=torch.tensor([[127.49999237060547,-.5,2.5],[0,0,0],[0,0,0]],dtype=torch.float32)
    idx,valid=_native_nearest_plan(q,(256,2,2))
    assert idx[0].tolist()==[127,0,3]
    assert valid.tolist()==[True,True,True]
    idx,valid=_native_nearest_plan(q,(3,2,2))
    assert idx[0][2].item()==2 and valid[2].item()


def test_native_splat_preserves_last_half_voxel_rejection():
    from fnit.recon_all.ca_register_inverse import splat_inverse_counts, splat_inverse_coordinate_sums
    for x,expected in [(2.49,1.),(2.5,1.),(2.51,0.),(3.1,1.)]:
        positions=np.array([[[[x,1.,1.]]]],np.float32)
        counts=splat_inverse_counts(positions,(3,3,3))
        assert float(counts.sum())==expected
        sums=splat_inverse_coordinate_sums(positions,(3,3,3))
        assert all(not item.any() for item in sums)  # node coordinates are zero



def _small_warp(tmp_path, source):
    import nibabel as nib
    from fnit.recon_all.mni_warp_io import write_forward_warp
    path=tmp_path/'field.nii.gz'
    write_forward_warp(np.zeros((*source.shape,3),np.float32),source,source,path)
    return path


def test_check_rejects_same_shape_wrong_scanner_affine(tmp_path):
    import nibabel as nib
    from fnit.recon_all.mni_warp_sampling import resample_mni_check
    source=nib.Nifti1Image(np.ones((4,4,4),np.uint8),np.eye(4))
    warp=_small_warp(tmp_path,source)
    shifted=np.eye(4);shifted[0,3]=4
    path=tmp_path/'wrong.nii.gz';nib.save(nib.Nifti1Image(np.ones(source.shape,np.uint8),shifted),path)
    with pytest.raises(ValueError,match='scanner-RAS'):
        resample_mni_check(path,warp,tmp_path/'output.nii.gz',device='cpu')


def test_check_accepts_big_endian_mgz_multibyte_voxels(tmp_path):
    import nibabel as nib
    from fnit.recon_all.mni_warp_sampling import resample_mni_check
    values=np.arange(64,dtype=np.int16).reshape(4,4,4)
    source=nib.MGHImage(values,np.eye(4));original=tmp_path/'orig.mgz';nib.save(source,original)
    loaded=nib.load(original)
    assert np.asarray(loaded.dataobj).dtype.byteorder=='>'
    warp=_small_warp(tmp_path,loaded);output=tmp_path/'output.nii.gz'
    resample_mni_check(original,warp,output,device='cpu')
    result=np.asarray(nib.load(output).dataobj)
    expected=values.copy();expected[:,:,0]=0  # native GCAM source-domain check
    np.testing.assert_array_equal(result,expected)
    assert result.dtype.kind=='i' and result.dtype.itemsize==2


@pytest.mark.parametrize('encoding,spacing',[(0,1),(1,1),(2,1),(3,2)])
def test_warp_reader_rejects_wrong_vector_encoding(tmp_path,encoding,spacing):
    import nibabel as nib,struct
    from fnit.recon_all.ca_register_inverse import read_warp_geometries
    source=nib.Nifti1Image(np.zeros((4,4,4),np.uint8),np.eye(4))
    image=nib.load(_small_warp(tmp_path,source))
    payload=bytearray(image.header.extensions[0].get_content());cursor=4
    while cursor+12<=len(payload):
        tag,length=struct.unpack_from('>iq',payload,cursor)
        if tag==13:
            struct.pack_into('>ii',payload,cursor+12,encoding,spacing);break
        cursor+=12+length
    image.header.extensions.clear();image.header.extensions.append(nib.nifti1.Nifti1Extension(14,bytes(payload)))
    with pytest.raises(ValueError,match='DISP_RAS'):
        read_warp_geometries(image)


def test_warp_reader_rejects_multiple_vector_frames(tmp_path):
    import nibabel as nib
    from fnit.recon_all.ca_register_inverse import read_warp_geometries
    source=nib.Nifti1Image(np.zeros((4,4,4),np.uint8),np.eye(4));single=nib.load(_small_warp(tmp_path,source))
    multiple=nib.Nifti1Image(np.zeros((4,4,4,2,3),np.float32),single.affine,single.header)
    with pytest.raises(ValueError,match='complete shape'):
        read_warp_geometries(multiple)
