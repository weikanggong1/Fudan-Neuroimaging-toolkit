"""Voxel-index pull transforms, matching VoxelMorph's ij convention.

Tensor spatial axes are i,j,k; vector channels are di,dj,dk. PyTorch sampling
grids reverse this order. Out-of-domain image samples are filled completely,
whereas displacement fields use border extension, as in Neurite interpn.
"""
from dataclasses import dataclass

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
    return _dense_from_grid(matrix, coords, warp_right)


def _dense_from_grid(matrix, coords, warp_right=None):
    loc = coords if warp_right is None else coords + warp_right
    return torch.einsum('ij,bjxyz->bixyz', matrix[:3, :3], loc) + matrix[:3, 3][None, :, None, None, None] - coords


def _dense_affine_nearest(matrix, shape, coords=None):
    """Build an affine shift without TF32 changing half-voxel ties."""
    if coords is None:
        coords = grid(shape, matrix.device, matrix.dtype)
    mapped = torch.stack([
        matrix[row, 0] * coords[:, 0]
        + matrix[row, 1] * coords[:, 1]
        + matrix[row, 2] * coords[:, 2]
        for row in range(3)
    ], dim=1)
    mapped = mapped + matrix[:3, 3][None, :, None, None, None]
    return mapped - coords


@dataclass(frozen=True)
class _SamplingPlan:
    """Coordinates shared by every frame of one image pull operation."""

    shape: tuple
    sample_grid: torch.Tensor | None
    flat_indices: torch.Tensor | None
    valid: torch.Tensor | None

    def to(self, device):
        """Move prepared coordinates without recomputing their arithmetic."""
        return _SamplingPlan(
            self.shape,
            None if self.sample_grid is None else self.sample_grid.to(device),
            None if self.flat_indices is None else self.flat_indices.to(device),
            None if self.valid is None else self.valid.to(device),
        )


def _prepare_transform(
    trans, source_shape, *, device, dtype, shape=None, method='linear',
    fill_value=0, surfa_nearest_rule=False, surfa_linear_rule=False,
    surfa_nearest_half_up=False,
    base_grid=None,
):
    """Prepare the existing sampler's coordinates without sampling frames.

    Keep the affine/displacement arithmetic and nearest tie rules unchanged.
    A plan can be reused for channel chunks without recreating voxel grids.
    """
    if method not in ('linear', 'nearest'):
        raise ValueError('method must be linear or nearest')
    use_surfa = method == 'nearest' and surfa_nearest_rule
    use_surfa_domain = use_surfa or (method == 'linear' and surfa_linear_rule)
    trans = torch.as_tensor(
        trans, dtype=torch.float32 if use_surfa_domain else dtype, device=device
    )
    if trans.ndim == 2:
        shape = source_shape if shape is None else tuple(shape)
        if use_surfa_domain:
            # The original final image sampler evaluates the affine directly.
            # Its float32 coordinates can change at the fill boundary if an
            # intermediate displacement is subtracted and added again.
            coords = grid(shape, device, torch.float32)
            loc = torch.stack([
                trans[row, 0] * coords[:, 0]
                + trans[row, 1] * coords[:, 1]
                + trans[row, 2] * coords[:, 2]
                + trans[row, 3]
                for row in range(3)
            ], dim=1)
        else:
            # Match Neurite's affine -> displacement -> coordinates path.
            # Avoid TF32 GEMM for exact nearest half-voxel ties.
            coords = grid(shape, device, dtype)
            shift = (
                _dense_affine_nearest(trans, shape, coords=coords)
                if method == 'nearest'
                else _dense_from_grid(trans, coords)
            )
            loc = coords + shift
    else:
        coords = (base_grid if base_grid is not None else grid(
            trans.shape[2:], device, torch.float32 if use_surfa else dtype
        ))
        loc = coords + trans
    valid = None
    if fill_value is not None:
        valid = torch.ones_like(loc[:, :1], dtype=torch.bool)
        for dimension, size in enumerate(source_shape):
            valid &= loc[:, dimension:dimension + 1] >= 0
            valid &= (
                loc[:, dimension:dimension + 1] < size
                if use_surfa_domain else loc[:, dimension:dimension + 1] <= size - 1
            )
    if method == 'nearest':
        idx = []
        for dimension, size in enumerate(source_shape):
            coordinate = loc[:, dimension]
            if use_surfa and surfa_nearest_half_up:
                # libc round promotes the float coordinate before rounding.
                # Adding .5 in float32 can round a value just below a tie up
                # to the next integer. Compare its fractional part directly.
                lower = coordinate.floor()
                rounded = lower + ((coordinate - lower) >= 0.5)
            else:
                rounded = (torch.floor(coordinate + 0.5)
                           if use_surfa else coordinate.round())
            idx.append(rounded.long().clamp(0, size - 1))
        flat = (idx[0] * source_shape[1] + idx[1]) * source_shape[2] + idx[2]
        return _SamplingPlan(tuple(loc.shape[2:]), None, flat, valid)
    norm = [loc[:, d] * (2 / (n - 1)) - 1 if n > 1 else torch.zeros_like(loc[:, d])
            for d, n in enumerate(source_shape)]
    sample_grid = torch.stack(norm[::-1], dim=-1)
    return _SamplingPlan(tuple(loc.shape[2:]), sample_grid, None, valid)


def _sample_prepared(volume, plan, fill_value=0):
    """Sample a channel chunk using a precomputed pull grid or indices."""
    if plan.flat_indices is not None:
        out = torch.gather(
            volume.flatten(2), 2,
            plan.flat_indices.flatten(1)[:, None].expand(
                volume.shape[0], volume.shape[1], -1
            ),
        )
        out = out.reshape(volume.shape[0], volume.shape[1], *plan.shape)
    else:
        sample_grid = plan.sample_grid.expand(volume.shape[0], -1, -1, -1, -1)
        out = F.grid_sample(
            volume, sample_grid, mode='bilinear', padding_mode='border',
            align_corners=True,
        )
    if fill_value is not None:
        out = torch.where(
            plan.valid, out,
            torch.as_tensor(fill_value, dtype=out.dtype, device=out.device),
        )
    return out


def surfa_nearest(volume, trans, shape=None, fill_value=0):
    """Resample with the nearest-neighbour rules used by Surfa 0.6.3."""
    plan = _prepare_transform(
        trans, volume.shape[2:], device=volume.device, dtype=volume.dtype,
        shape=shape, method='nearest', fill_value=fill_value,
        surfa_nearest_rule=True,
        surfa_nearest_half_up=volume.device.type == 'cpu',
    )
    return _sample_prepared(volume, plan, fill_value)


def transform(volume, trans, shape=None, fill_value=0, method='linear'):
    """Resample N,C,I,J,K data with a matrix or N,3,I,J,K displacement."""
    plan = _prepare_transform(
        trans, volume.shape[2:], device=volume.device, dtype=volume.dtype,
        shape=shape, method=method, fill_value=fill_value,
    )
    return _sample_prepared(volume, plan, fill_value)


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
    if vec.device.type == 'cpu':
        # Scaling-and-squaring repeatedly queries the same voxel grid. Reuse
        # its values on CPU; keep the Neurite border-extension rule and every
        # arithmetic operation. CUDA continues through its original path.
        coords = grid(vec.shape[2:], vec.device, vec.dtype)
        for _ in range(steps):
            plan = _prepare_transform(
                out, out.shape[2:], device=out.device, dtype=out.dtype,
                fill_value=None, base_grid=coords,
            )
            out = out + _sample_prepared(out, plan, None)
        return out
    for _ in range(steps):
        out = out + transform(out, out, fill_value=None)
    return out


affine_to_dense = dense
