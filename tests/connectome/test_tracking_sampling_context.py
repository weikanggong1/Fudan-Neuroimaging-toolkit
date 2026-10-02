"""Call-scoped FOD sampler preserves coordinate arithmetic and view semantics."""
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


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_axis_reuse_preserves_masked_ordered_5tt_interpolation(device):
    from fnit.connectome.tracking import _five_tissue_mrtrix
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(31)
    volume = torch.rand((4,5,6,5), generator=generator).to(device)
    volume[1,1,1] = 0
    points = torch.cat((torch.rand((257,3),generator=generator)*6-1,
                        torch.tensor([[-.5,0,0],[-.49999,0,0],[3.5,0,0],[3.49999,0,0],
                                      [1,1,1],[.000001,2,3],[.0000001,2,3]]))).to(device)
    inverse = torch.eye(4,dtype=torch.float64,device=device)
    assert torch.equal(_five_tissue_mrtrix(volume,points,inverse),_ordered_5tt(volume,points,inverse))
