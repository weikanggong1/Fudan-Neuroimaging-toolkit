"""CPU joint inference preprocessing in raw voxel coordinates."""
import itertools

import torch

from .spatial import _dense_from_grid, grid, transform


def network_transform(volume, matrix, shape=None, fill_value=0):
    """Use ordered trilinear corners without normalized-grid roundoff.

    Coordinates retain the established affine-to-shift-to-location formula.
    The source center domain is closed, with border extension for corner
    indices and explicit fill only outside that domain, as in the network
    sampler. Differentiable callers retain the established Torch sampler.
    """
    matrix = torch.as_tensor(matrix, dtype=volume.dtype, device=volume.device)
    if (volume.device.type != 'cpu' or volume.dtype != torch.float32
            or torch.is_grad_enabled() or volume.requires_grad or matrix.requires_grad
            or matrix.ndim != 2):
        return transform(volume, matrix, shape=shape, fill_value=fill_value)
    shape = volume.shape[2:] if shape is None else tuple(shape)
    coords = grid(shape, volume.device, volume.dtype)
    locations = coords + _dense_from_grid(matrix, coords)
    source_shape = volume.shape[2:]
    lower = [locations[:, axis].floor().clamp(0, size - 1)
             for axis, size in enumerate(source_shape)]
    upper = [(coordinate + 1).clamp(0, size - 1)
             for coordinate, size in zip(lower, source_shape)]
    clipped = [locations[:, axis].clamp(0, size - 1)
               for axis, size in enumerate(source_shape)]
    lower_weights = [high - coordinate for high, coordinate in zip(upper, clipped)]
    weights = [lower_weights, [1 - weight for weight in lower_weights]]
    indices = [[coordinate.long() for coordinate in lower],
               [coordinate.long() for coordinate in upper]]
    result = volume.new_zeros((volume.shape[0], volume.shape[1], *shape))
    flat = volume.flatten(2)
    for corner in itertools.product((0, 1), repeat=3):
        i, j, k = [indices[choice][axis] for axis, choice in enumerate(corner)]
        index = ((i * source_shape[1] + j) * source_shape[2] + k).flatten(1)
        value = torch.gather(flat, 2, index[:, None].expand(volume.shape[0], volume.shape[1], -1)).reshape_as(result)
        wi, wj, wk = [weights[choice][axis] for axis, choice in enumerate(corner)]
        result = result + ((wi * wj) * wk)[:, None] * value
    if fill_value is not None:
        valid = torch.ones_like(locations[:, :1], dtype=torch.bool)
        for axis, size in enumerate(source_shape):
            coordinate = locations[:, axis:axis+1]
            valid &= (coordinate >= 0) & (coordinate <= size - 1)
        result = torch.where(valid, result,
                             torch.as_tensor(fill_value, dtype=volume.dtype, device=volume.device))
    return result
