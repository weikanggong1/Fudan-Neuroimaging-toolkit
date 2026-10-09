"""Call-scoped FOD sampler preserves coordinate arithmetic and view semantics."""
import pickle
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from fnit.connectome.tracking import _VolumeSampler


def _uncached_sample(volume, points, inverse):
    voxel = points.double() @ inverse[:3, :3].double().T + inverse[:3, 3].double()
    scale = voxel.new_tensor([max(size - 1, 1) for size in volume.shape[:3]])
    grid = (2 * voxel / scale - 1).float().reshape(1, 1, 1, -1, 3)
    return F.grid_sample(volume.permute(3, 2, 1, 0)[None], grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True).reshape(volume.shape[-1], -1).T


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(1, 4, 5, 45), (6, 4, 5, 1)])
def test_cached_sampler_is_bitwise_equal_at_boundary_and_after_volume_update(device, dtype, shape):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    generator = torch.Generator().manual_seed(22)
    # Noncontiguous layout and oblique, anisotropic affine exercise prepared views.
    volume = torch.randn((shape[2], shape[1], shape[0], shape[3]), generator=generator).to(device).permute(2,1,0,3)
    inverse = torch.tensor([[.5, -.1, 0, 1], [.1, 1.2, .02, -2], [0, .04, -.7, 3], [0,0,0,1]], dtype=dtype, device=device)
    points = torch.tensor([[0,0,0], [2,3,4], [-100,0,0], [0,100,0], [0,0,100], [0.0005,0,0], [-0.0005,0,0]], dtype=torch.float32, device=device)
    sampler = _VolumeSampler(volume, inverse)
    for _ in range(2):
        assert torch.equal(sampler(points), _uncached_sample(volume, points, inverse))
        volume.add_(.1)
    assert sampler.image.untyped_storage().data_ptr() == volume.untyped_storage().data_ptr()


def _ordered_5tt(volume, points, inverse):
    # Frozen eight-corner rule, including zero nearest voxel and tiny weights.
    voxel = points.double() @ inverse[:3, :3].double().T + inverse[:3, 3].double()
    shape = volume.shape[:3]
    inside = ((voxel > -.5) & (voxel < voxel.new_tensor(shape) - .5)).all(-1)
    nearest = torch.floor(voxel + .5).long()
    nearest = torch.stack(tuple(nearest[:, axis].clamp(0, shape[axis] - 1) for axis in range(3)), -1)
    nonzero = (volume[nearest[:,0], nearest[:,1], nearest[:,2]] != 0).any(-1)
    lower = torch.floor(voxel).long()
    fraction = (voxel - lower).float()
    result = volume.new_zeros((len(points),5))
    for dz in (0,1):
        iz = (lower[:,2] + dz).clamp(0,shape[2]-1)
        wz = fraction[:,2] if dz else 1-fraction[:,2]
        for dy in (0,1):
            iy = (lower[:,1] + dy).clamp(0,shape[1]-1)
            wy = fraction[:,1] if dy else 1-fraction[:,1]
            partial = wy*wz
            for dx in (0,1):
                ix = (lower[:,0] + dx).clamp(0,shape[0]-1)
                wx = fraction[:,0] if dx else 1-fraction[:,0]
                weight = wx*partial
                result += volume[ix,iy,iz]*torch.where(weight < 1e-6,0.,weight)[:,None]
    return torch.where((inside & nonzero)[:,None],result.clamp(0,1),0.)


def _raw_bytes(value):
    return value.detach().cpu().contiguous().numpy().reshape(-1).view(np.uint8).tobytes()


def _input_state(values):
    return tuple((value.dtype, tuple(value.shape), tuple(value.stride()), _raw_bytes(value))
                 for value in values)


def _rng_state(device, generator):
    return (_raw_bytes(torch.get_rng_state()), _raw_bytes(generator.get_state()),
            _raw_bytes(torch.cuda.get_rng_state(device)) if device == 'cuda' else None,
            pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))


def _assert_5tt_exact_and_unchanged(volume, points, inverse, device, generator):
    from fnit.connectome.tracking import _five_tissue_mrtrix
    inputs = (volume, points, inverse)
    input_before = _input_state(inputs)
    rng_before = _rng_state(device, generator)
    expected = _ordered_5tt(*inputs)
    assert _input_state(inputs) == input_before
    assert _rng_state(device, generator) == rng_before
    actual = _five_tissue_mrtrix(*inputs)
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape == (len(points), 5)
    assert actual.device == expected.device == volume.device
    assert torch.equal(actual, expected)
    assert _raw_bytes(actual) == _raw_bytes(expected)
    assert _input_state(inputs) == input_before
    assert _rng_state(device, generator) == rng_before
    return actual


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('layout', ['contiguous', 'spatial_permuted', 'sliced', 'channel_first'])
@pytest.mark.parametrize('oblique', [False, True])
@pytest.mark.parametrize('point_dtype', [torch.float32, torch.float64])
def test_axis_reuse_preserves_masked_ordered_5tt_interpolation(device, layout, oblique, point_dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(31)
    if layout == 'contiguous':
        volume = torch.rand((4, 5, 6, 5), generator=generator).to(device)
    elif layout == 'spatial_permuted':
        volume = torch.rand((6, 5, 4, 5), generator=generator).to(device).permute(2, 1, 0, 3)
    elif layout == 'sliced':
        # Transfer the backing storage first to preserve the sliced CUDA strides.
        volume = torch.rand((8, 10, 12, 10), generator=generator).to(device)[::2, ::2, ::2, ::2]
    else:
        volume = torch.rand((5, 4, 5, 6), generator=generator).to(device).permute(1, 2, 3, 0)
    assert volume.is_contiguous() == (layout == 'contiguous')
    volume[1,1,1] = 0
    inverse_cpu = (torch.tensor([[.91, .07, -.12, 13.25], [-.04, .73, .19, -7.5],
                                 [.16, -.09, 1.27, 4.75], [0., 0., 0., 1.]], dtype=torch.float64)
                   if oblique else torch.eye(4, dtype=torch.float64))
    voxels = torch.cat((torch.rand((257, 3), generator=generator, dtype=torch.float64)
                        * torch.tensor([6., 7., 8.]) - 1,
                        torch.tensor([[-.5, 0, 0], [-.49999, 0, 0], [3.5, 0, 0], [3.49999, 0, 0],
                                      [1, 1, 1], [.000001, 2, 3], [.0000001, 2, 3]], dtype=torch.float64)))
    world = ((voxels - inverse_cpu[:3, 3]) @ torch.linalg.inv(inverse_cpu[:3, :3]).T).to(point_dtype)
    point_storage = torch.empty((len(world), 6), dtype=point_dtype, device=device)
    points = point_storage[:, ::2]
    points.copy_(world)
    assert not points.is_contiguous()
    inverse = inverse_cpu.to(device)
    for queries in (points, points[:0]):
        _assert_5tt_exact_and_unchanged(volume, queries, inverse, device, generator)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('point_dtype', [torch.float32, torch.float64])
def test_ordered_5tt_tiny_weight_threshold_is_strict(device, point_dtype):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(47)
    # The nearest voxel stays nonzero; channel 0 isolates one corner's weight.
    volume = torch.zeros((2, 2, 2, 5), dtype=torch.float32, device=device)
    volume[0, 0, 0, 4] = 1
    for axis in range(3):
        corner = [0, 0, 0]
        corner[axis] = 1
        volume[tuple(corner) + (0,)] = 1
    threshold = torch.tensor(1e-6, dtype=torch.float32)
    weights = torch.stack((torch.nextafter(threshold, torch.tensor(0.)), threshold,
                           torch.nextafter(threshold, torch.tensor(float('inf')))))
    points = torch.zeros((9, 3), dtype=point_dtype, device=device)
    for axis in range(3):
        points[3 * axis:3 * axis + 3, axis] = weights.to(device=device, dtype=point_dtype)
    result = _assert_5tt_exact_and_unchanged(
        volume, points, torch.eye(4, dtype=torch.float64, device=device), device, generator)
    # Equality is retained: replacing < by <= would zero the middle sample.
    retained = torch.tensor([0., weights[1].item(), weights[2].item()], dtype=torch.float32, device=device)
    assert _raw_bytes(result[:, 0]) == _raw_bytes(retained.repeat(3))
