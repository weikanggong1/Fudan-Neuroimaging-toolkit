"""Validation-only fusion of the v29 raw FP32 trilinear sampling arithmetic.

Coordinates remain the original Torch grid + dense(einsum) + grid. No CUDA,
world-space or final-image sampler is changed. NumBa has fastmath disabled.
"""
import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads, config
import torch


@njit(parallel=True, fastmath=False, cache=True)
def _sample(volume, locations, use_fill, fill):
    batch, channels, ni, nj, nk = volume.shape
    _, _, oi, oj, ok = locations.shape
    result = np.empty((batch, channels, oi, oj, ok), dtype=np.float32)
    zero, one = np.float32(0), np.float32(1)
    for offset in prange(oi * oj * ok):
        k = offset % ok
        j = (offset // ok) % oj
        i = offset // (oj * ok)
        ci, cj, ck = locations[0, 0, i, j, k], locations[0, 1, i, j, k], locations[0, 2, i, j, k]
        invalid = ci < 0 or ci > ni - 1 or cj < 0 or cj > nj - 1 or ck < 0 or ck > nk - 1
        if use_fill and invalid:
            for b in range(batch):
                for c in range(channels): result[b, c, i, j, k] = fill
            continue
        # Floor before clamp; upper is based on the already clamped lower.
        li = np.float32(min(max(np.float32(np.floor(ci)), zero), np.float32(ni - 1)))
        lj = np.float32(min(max(np.float32(np.floor(cj)), zero), np.float32(nj - 1)))
        lk = np.float32(min(max(np.float32(np.floor(ck)), zero), np.float32(nk - 1)))
        hi = np.float32(min(np.float32(li + one), np.float32(ni - 1)))
        hj = np.float32(min(np.float32(lj + one), np.float32(nj - 1)))
        hk = np.float32(min(np.float32(lk + one), np.float32(nk - 1)))
        ci = np.float32(min(max(ci, zero), np.float32(ni - 1)))
        cj = np.float32(min(max(cj, zero), np.float32(nj - 1)))
        ck = np.float32(min(max(ck, zero), np.float32(nk - 1)))
        wi, wj, wk = np.float32(hi - ci), np.float32(hj - cj), np.float32(hk - ck)
        ui, uj, uk = np.float32(one - wi), np.float32(one - wj), np.float32(one - wk)
        for b in range(batch):
            for c in range(channels):
                value = zero
                for ai in range(2):
                    xi = wi if ai == 0 else ui
                    ii = int(li) if ai == 0 else int(hi)
                    for aj in range(2):
                        xj = wj if aj == 0 else uj
                        jj = int(lj) if aj == 0 else int(hj)
                        for ak in range(2):
                            xk = wk if ak == 0 else uk
                            kk = int(lk) if ak == 0 else int(hk)
                            weight = np.float32(np.float32(xi * xj) * xk)
                            term = np.float32(weight * volume[b, c, ii, jj, kk])
                            value = np.float32(value + term)
                result[b, c, i, j, k] = value
    return result


def sample_at_locations(volume, locations, fill_value=0):
    """Private no-grad FP32 finite coordinate prototype; preserve input data."""
    if (volume.device.type != 'cpu' or volume.dtype != torch.float32
            or locations.device.type != 'cpu' or locations.dtype != torch.float32
            or locations.shape[:2] != (1, 3) or not torch.isfinite(locations).all()):
        raise ValueError('prototype requires finite single-grid CPU FP32 inputs')
    previous = get_num_threads()
    try:
        set_num_threads(min(torch.get_num_threads(), config.NUMBA_NUM_THREADS))
        output = _sample(volume.numpy(), locations.numpy(), fill_value is not None,
                         np.float32(0 if fill_value is None else fill_value))
    finally:
        set_num_threads(previous)
    return torch.from_numpy(output)


def network_transform(volume, matrix, shape=None, fill_value=0):
    from fnit.synthmorph.spatial import grid, _dense_from_grid
    matrix = torch.as_tensor(matrix, dtype=volume.dtype, device=volume.device)
    shape = volume.shape[2:] if shape is None else tuple(shape)
    coords = grid(shape, volume.device, volume.dtype)
    locations = coords + _dense_from_grid(matrix, coords)
    return sample_at_locations(volume, locations, fill_value)
