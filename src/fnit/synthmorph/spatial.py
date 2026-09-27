"""Voxel-index pull transforms, matching VoxelMorph's ij convention.

Tensor spatial axes are i,j,k; vector channels are di,dj,dk. PyTorch sampling
grids reverse this order. Out-of-domain image samples are filled completely,
whereas displacement fields use border extension, as in Neurite interpn.
"""
import torch
import torch.nn.functional as F


def grid(shape, device, dtype=torch.float32):
    return torch.stack(torch.meshgrid(*[
        torch.arange(n, device=device, dtype=dtype) for n in shape
    ], indexing='ij'), dim=0)[None]


def square(matrix):
    if matrix.shape == (3, 4):
        matrix = torch.cat((matrix, matrix.new_tensor([[0, 0, 0, 1]])))
    return matrix


def dense(matrix, shape, warp_right=None):
    coords = grid(shape, matrix.device, matrix.dtype)
    loc = coords if warp_right is None else coords + warp_right
    return torch.einsum('ij,bjxyz->bixyz', matrix[:3, :3], loc) + matrix[:3, 3][None, :, None, None, None] - coords


def _dense_affine_nearest(matrix, shape):
    """Build an affine shift without TF32 changing half-voxel ties."""
    coords = grid(shape, matrix.device, matrix.dtype)
    mapped = torch.stack([
        matrix[row, 0] * coords[:, 0]
        + matrix[row, 1] * coords[:, 1]
        + matrix[row, 2] * coords[:, 2]
        for row in range(3)
    ], dim=1)
    mapped = mapped + matrix[:3, 3][None, :, None, None, None]
    return mapped - coords


def surfa_nearest(volume, trans, shape=None, fill_value=0):
    """Resample with the nearest-neighbour rules used by Surfa 0.6.3."""
    trans = torch.as_tensor(trans, dtype=torch.float32, device=volume.device)
    if trans.ndim == 2:
        shape = volume.shape[2:] if shape is None else tuple(shape)
        coords = grid(shape, volume.device, torch.float32)
        loc = torch.stack([
            trans[row, 0] * coords[:, 0]
            + trans[row, 1] * coords[:, 1]
            + trans[row, 2] * coords[:, 2]
            + trans[row, 3]
            for row in range(3)
        ], dim=1)
    else:
        loc = grid(trans.shape[2:], volume.device, torch.float32) + trans

    valid = torch.ones_like(loc[:, :1], dtype=torch.bool)
    indices = []
    for dimension, size in enumerate(volume.shape[2:]):
        valid &= (loc[:, dimension:dimension + 1] >= 0)
        valid &= (loc[:, dimension:dimension + 1] < size)
        index = torch.floor(loc[:, dimension] + 0.5).long()
        indices.append(index.clamp(0, size - 1))

    flat = (
        indices[0] * volume.shape[3] + indices[1]
    ) * volume.shape[4] + indices[2]
    out = torch.gather(
        volume.flatten(2),
        2,
        flat.flatten(1)[:, None].expand(
            volume.shape[0], volume.shape[1], -1
        ),
    )
    out = out.reshape(volume.shape[0], volume.shape[1], *loc.shape[2:])
    if fill_value is not None:
        fill = torch.as_tensor(fill_value, dtype=out.dtype, device=out.device)
        out = torch.where(valid, out, fill)
    return out


def transform(volume, trans, shape=None, fill_value=0, method='linear'):
    """Resample N,C,I,J,K data with a matrix or N,3,I,J,K displacement."""
    trans = torch.as_tensor(trans, dtype=volume.dtype, device=volume.device)
    if method not in ('linear', 'nearest'):
        raise ValueError('method must be linear or nearest')
    if trans.ndim == 2:
        shape = volume.shape[2:] if shape is None else tuple(shape)
        # Match Neurite's affine -> displacement -> coordinates path. The
        # nearest branch avoids TF32 GEMM so exact half-voxel ties remain ties.
        shift = (
            _dense_affine_nearest(trans, shape)
            if method == 'nearest'
            else dense(trans, shape)
        )
        loc = grid(shape, volume.device, volume.dtype) + shift
    else:
        loc = grid(trans.shape[2:], volume.device, volume.dtype) + trans
    if method == 'nearest':
        idx = [loc[:, d].round().long().clamp(0, n - 1) for d, n in enumerate(volume.shape[2:])]
        flat = (idx[0] * volume.shape[3] + idx[1]) * volume.shape[4] + idx[2]
        out = torch.gather(volume.flatten(2), 2, flat.flatten(1)[:, None].expand(volume.shape[0], volume.shape[1], -1))
        out = out.reshape(volume.shape[0], volume.shape[1], *loc.shape[2:])
    else:
        norm = [loc[:, d] * (2 / (n - 1)) - 1 if n > 1 else torch.zeros_like(loc[:, d])
                for d, n in enumerate(volume.shape[2:])]
        sample_grid = torch.stack(norm[::-1], dim=-1).expand(volume.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(volume, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    if fill_value is not None:
        valid = torch.ones_like(loc[:, :1], dtype=torch.bool)
        for d, n in enumerate(volume.shape[2:]):
            valid &= (loc[:, d:d+1] >= 0) & (loc[:, d:d+1] <= n - 1)
        out = torch.where(valid, out, torch.as_tensor(fill_value, dtype=out.dtype, device=out.device))
    return out


def compose(transforms, shape=None):
    """Compose pull maps A(B(C(x))) from [A,B,C], with border extension."""
    if not transforms:
        raise ValueError('transforms must not be empty')
    ref = next((x for x in transforms if isinstance(x, torch.Tensor)), None)
    device = ref.device if ref is not None else 'cpu'
    curr = None
    for nxt in reversed(transforms):
        nxt = torch.as_tensor(nxt, dtype=torch.float32, device=device)
        if curr is None:
            curr = nxt
        elif nxt.ndim != 2:
            if curr.ndim == 2:
                curr = dense(curr, nxt.shape[2:] if shape is None else shape)
            curr = curr + transform(nxt, curr, fill_value=None)
        elif curr.ndim != 2:
            curr = dense(nxt, curr.shape[2:], warp_right=curr)
        else:
            curr = square(nxt) @ square(curr)
    return curr


def integrate(vec, steps=7):
    out = vec / (2 ** steps)
    for _ in range(steps):
        out = out + transform(out, out, fill_value=None)
    return out


affine_to_dense = dense
