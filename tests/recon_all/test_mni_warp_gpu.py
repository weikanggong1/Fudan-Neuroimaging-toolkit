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
