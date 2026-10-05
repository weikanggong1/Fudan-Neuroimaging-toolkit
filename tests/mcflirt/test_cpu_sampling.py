"""CPU sampler numerical contracts; tiny inputs are unit tests, not benchmarks."""
import numpy as np
import pytest
import torch

from fnit.flirt.core import _edge_background, _manual_trilinear
from fnit.mcflirt.sampling import _coordinates, _cubic_coefficients, _sample_cubic
from fnit.mcflirt._sampling_cpu import sample


@pytest.mark.parametrize('layout',('C','F'))
@pytest.mark.parametrize('interpolation',('linear','spline'))
@pytest.mark.parametrize('shape',((7,6,5),(2,2,2)))
def test_cpu_full_frame_preserves_tensor_arithmetic(layout,interpolation,shape):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
        values=np.array(values*17.5+101.,order=layout,dtype=np.float32)
        pull=np.array([[1.01,.13,.07,-.5],[-.09,.98,.03,.25],[.02,.08,1.03,-1.1]],dtype=np.float64)
        tensor=torch.from_numpy(values)
        coordinates=_coordinates(pull,shape,'cpu')
        lower=coordinates.floor();valid=torch.ones(shape,dtype=torch.bool)
        for axis,size in enumerate(shape):valid&=(lower[axis]>=-1)&(lower[axis]<size)
        background=_edge_background(tensor)
        if interpolation=='spline':expected=_sample_cubic(_cubic_coefficients(tensor),coordinates)
        else:
            bounded=torch.stack([coordinates[axis].clamp(0,size-1) for axis,size in enumerate(shape)]).reshape(3,-1)
            expected=_manual_trilinear(tensor,bounded).reshape(shape)
        expected=torch.where(valid,expected,background)
        actual=sample(values,pull,shape,float(background),interpolation)
        np.testing.assert_array_equal(actual,expected.numpy())
        assert actual.dtype==np.float32
    finally:torch.set_num_threads(previous)


@pytest.mark.parametrize('shape',((1,6,5),(7,1,5),(7,6,1)))
def test_spline_thin_axis_matches_existing_tensor_path(shape):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+31
        pull=np.eye(4,dtype=np.float64)[:3];pull[:,3]=[-.25,.5,-1.1]
        tensor=torch.from_numpy(values);coordinates=_coordinates(pull,shape,'cpu');lower=coordinates.floor();valid=torch.ones(shape,dtype=torch.bool)
        for axis,size in enumerate(shape):valid&=(lower[axis]>=-1)&(lower[axis]<size)
        background=_edge_background(tensor)
        expected=torch.where(valid,_sample_cubic(_cubic_coefficients(tensor),coordinates),background)
        np.testing.assert_array_equal(sample(values,pull,shape,float(background),'spline'),expected.numpy())
    finally:torch.set_num_threads(previous)


def test_api_preserves_nonfinite_tensor_fallback(monkeypatch):
    import nibabel as nib
    import fnit.mcflirt._sampling_cpu as compiled
    from fnit.mcflirt.sampling import sample_motion_frame
    values=np.arange(120,dtype=np.float32).reshape(4,5,6)+1
    values[1,2,3]=np.nan
    image=nib.Nifti1Image(values,np.diag([-1.,1.,1.,1.]))
    def unexpected(*args):
        raise AssertionError('nonfinite image entered compiled CPU sampler')
    monkeypatch.setattr(compiled,'sample',unexpected)
    actual=sample_motion_frame(values,image,image,np.eye(4),device='cpu',interpolation='linear')
    assert actual.shape==image.shape
    assert torch.isnan(actual).any()
