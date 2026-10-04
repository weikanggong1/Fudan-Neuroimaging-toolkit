"""CPU image semantics and device isolation regressions."""
import numpy as np
import pytest
import torch

from fnit._nib import FNITNifti1Image
from fnit._transforms import AffineTransform, DenseWarp
from fnit.synthmorph import SynthMorph, apply_transform
from fnit.synthmorph import models, pipeline, spatial


@pytest.mark.parametrize('device,configure,expected', [
    ('cpu', True, False), ('cuda:0', False, False), ('cuda:0', True, True),
])
def test_constructor_precision_is_device_scoped(monkeypatch, device, configure, expected):
    # No model or GPU is needed to check this process-global constructor policy.
    monkeypatch.setattr(pipeline, 'resolve_weights', lambda *args: '/unused.h5')
    monkeypatch.setattr(models, 'SynthMorphNetwork', lambda **kwargs: object())
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        SynthMorph(device=device, model='affine', configure_precision=configure)
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.mark.parametrize('kind', ['affine', 'dense'])
@pytest.mark.parametrize('shift,expected', [(.25, 12.), (1., -7.)])
def test_cpu_final_linear_accepts_last_center_band(kind, shift, expected):
    image = FNITNifti1Image(np.arange(10, 13, dtype=np.float32).reshape(3, 1, 1), np.eye(4))
    if kind == 'affine':
        matrix = np.eye(4); matrix[0, 3] = -shift
        transformation = AffineTransform(matrix, source=image, target=image, space='world')
    else:
        data = np.zeros((3, 1, 1, 3), dtype=np.float32); data[..., 0] = shift
        transformation = DenseWarp(data, source=image, target=image)
    result = apply_transform(image, transformation, fill=-7, device='cpu')
    assert np.asarray(result.dataobj)[2, 0, 0] == expected


def test_network_sampler_retains_neurite_closed_center_domain():
    image = torch.arange(10, 13, dtype=torch.float32).reshape(1, 1, 3, 1, 1)
    pull = torch.eye(4); pull[0, 3] = .25
    result = spatial.transform(image, pull, fill_value=-7)
    assert result[0, 0, 2, 0, 0].item() == -7


@pytest.mark.parametrize('steps', [5, 7])
def test_cpu_integration_reuses_grid_and_is_bitwise_unchanged(monkeypatch, steps):
    generator = torch.Generator().manual_seed(401)
    velocity = torch.randn((1, 3, 13, 11, 9), generator=generator) * .2
    previous = velocity / 2 ** steps
    for _ in range(steps):
        previous = previous + spatial.transform(previous, previous, fill_value=None)
    original_grid = spatial.grid
    calls = []
    def count_grid(*args, **kwargs):
        calls.append(1)
        return original_grid(*args, **kwargs)
    monkeypatch.setattr(spatial, 'grid', count_grid)
    actual = spatial.integrate(velocity, steps)
    assert len(calls) == 1
    assert torch.equal(actual.view(torch.int32), previous.view(torch.int32))
