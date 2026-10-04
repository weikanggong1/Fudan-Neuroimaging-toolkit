"""Periodic coefficients retain the documented finite source FOV contract."""
import nibabel as nib
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from fnit._world_resampling import WorldTransformChain, resample_world_image
from fnit.applywarp import TorchApplyWarp
from fnit.synthmorph import apply_transform


@pytest.mark.parametrize('entry', ['synthmorph', 'applywarp', 'shared'])
@pytest.mark.parametrize('method,order', [('linear', 1), ('nearest', 0), ('spline', 3)])
def test_periodic_coefficients_do_not_wrap_queries_outside_source_fov(entry, method, order):
    values = np.arange(60, dtype=np.float32).reshape(5, 4, 3) + 10
    source = nib.Nifti1Image(values, np.eye(4))
    source.header.set_xyzt_units('mm')
    affine = np.eye(4)
    affine[:3, 3] = [-0.25, 0.25, 0.25]
    target = nib.Nifti1Image(np.zeros((7, 6, 5), np.float32), affine)
    chain = WorldTransformChain(target, np.eye(4))
    if entry == 'synthmorph':
        result = apply_transform(source, chain, method=method, boundary='periodic', device='cpu')
    elif entry == 'applywarp':
        result = TorchApplyWarp(device='cpu').apply_world(
            source, chain, interpolation=method, boundary='periodic')
    else:
        result = resample_world_image(source, target, np.eye(4),
                                      interpolation=method, boundary='periodic', device='cpu')
    query = nib.affines.apply_affine(affine, np.indices(target.shape).reshape(3, -1).T).T
    inside = ((query >= 0) & (query <= np.array(source.shape)[:, None] - 1)).all(0)
    oracle = map_coordinates(values, query, order=order, mode='grid-wrap')
    oracle[~inside] = 0
    actual = np.asarray(result.dataobj).ravel()
    # All three public routes keep finite-FOV zeros even though a genuine
    # circular oracle has nonzero values at these outside query points.
    assert np.count_nonzero(~inside) > 0
    np.testing.assert_array_equal(actual[~inside], np.zeros(np.count_nonzero(~inside)))
    np.testing.assert_allclose(actual, oracle, rtol=0, atol=2e-5)


@pytest.mark.parametrize('offset', [-5e-7, -2e-6])
@pytest.mark.parametrize('method', ['linear', 'nearest', 'spline'])
def test_periodic_only_clamps_roundoff_within_declared_edge_tolerance(offset, method):
    source = nib.Nifti1Image(np.full((4, 3, 2), 10, np.float32), np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = offset
    target = nib.Nifti1Image(np.zeros((1, 1, 1), np.float32), affine)
    result = apply_transform(source, WorldTransformChain(target, np.eye(4)),
                             method=method, boundary='periodic', device='cpu')
    if offset >= -1e-6:
        identity_target = nib.Nifti1Image(np.zeros((1, 1, 1), np.float32), np.eye(4))
        identity = apply_transform(source, WorldTransformChain(identity_target, np.eye(4)),
                                   method=method, boundary='periodic', device='cpu')
        np.testing.assert_array_equal(np.asarray(result.dataobj), np.asarray(identity.dataobj))
    else:
        assert np.asarray(result.dataobj).item() == 0
