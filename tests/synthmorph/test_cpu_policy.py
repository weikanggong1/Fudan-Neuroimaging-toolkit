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


@pytest.mark.parametrize('center', [0.5, 127.5])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_cpu_surfa_nearest_rounds_float_coordinate_before_half_tie(center, direction):
    from fnit.synthmorph.spatial import surfa_nearest
    coordinate = torch.tensor(center, dtype=torch.float32)
    if direction:
        coordinate = torch.nextafter(coordinate, torch.tensor(float('inf')*direction))
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[0, 3] = coordinate
    image = torch.arange(260, dtype=torch.float32).reshape(1, 1, 260, 1, 1)
    result = surfa_nearest(image, matrix, shape=(1, 1, 1))
    # C libc round receives the exact float32 value promoted to double.
    expected = np.floor(np.float64(coordinate.item()) + .5)
    assert result.item() == expected


def test_legacy_prepared_nearest_policy_remains_explicit_for_cuda_plans():
    from fnit.synthmorph.spatial import _prepare_transform, _sample_prepared
    coordinate = torch.nextafter(torch.tensor(.5), torch.tensor(-float('inf')))
    matrix = torch.eye(4);matrix[0, 3] = coordinate
    image = torch.arange(2, dtype=torch.float32).reshape(1, 1, 2, 1, 1)
    options = dict(device='cpu', dtype=torch.float32, shape=(1, 1, 1),
                   method='nearest', surfa_nearest_rule=True)
    legacy = _prepare_transform(matrix, image.shape[2:], **options)
    strict = _prepare_transform(matrix, image.shape[2:], **options,
                                surfa_nearest_half_up=True)
    assert _sample_prepared(image, legacy).item() == 1
    assert _sample_prepared(image, strict).item() == 0


def test_cpu_linear_affine_coordinates_do_not_cancel_at_fill_boundary():
    image = FNITNifti1Image(np.arange(101, dtype=np.float32).reshape(101, 1, 1), np.eye(4))
    pull = np.eye(4, dtype=np.float32)
    pull[0, 0] = .1
    pull[0, 3] = np.float32(-10.000001)
    result = pipeline._resampled_image(image, pull, image, 'cpu', fill=-7)
    assert np.asarray(result.dataobj)[100, 0, 0] == -7
    # Network sampling keeps the old affine -> displacement -> coordinates.
    tensor = torch.from_numpy(np.asarray(image.dataobj))[None, None]
    previous = spatial.transform(tensor, torch.from_numpy(pull), fill_value=-7)
    assert previous[0, 0, 100, 0, 0].item() == 0
