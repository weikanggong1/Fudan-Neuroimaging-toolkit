"""CPU-only optional raw-coordinate trilinear sampler.

Self-written ordered FP32 loop; importing this helper is restricted to the
existing CPU inference route. Unsupported/failed attempts retain Torch.
"""
from copy import deepcopy
import os
import platform
import threading

import numpy as np
import torch

_INFO = threading.local()
_KERNEL = None
_KERNEL_LOCK = threading.RLock()
_FAILURE = None
_MIN_VOXELS = 32768
try:
    import numba
    _prange = numba.prange
except Exception as error:
    numba = None
    _prange = range
    _FAILURE = 'NumBa unavailable: ' + str(error)


def backend_info():
    """Private copy of the latest attempt on this Python thread."""
    return deepcopy(getattr(_INFO, 'value', {'backend': None, 'reason': 'not called'}))


def _fallback(reason, **details):
    _INFO.value = {'backend': 'torch', 'reason': str(reason), **details}
    return None


def _autocast_cpu():
    try:
        return torch.is_autocast_enabled('cpu')
    except TypeError:
        return torch.is_autocast_cpu_enabled()


def _sample_impl(volume, locations, use_fill, fill):
    batch, channels, ni, nj, nk = volume.shape
    _, _, oi, oj, ok = locations.shape
    result = np.empty((batch, channels, oi, oj, ok), dtype=np.float32)
    zero, one = np.float32(0), np.float32(1)
    for offset in _prange(oi * oj * ok):
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


def _kernel():
    global _KERNEL
    with _KERNEL_LOCK:
        if _KERNEL is None:
            _KERNEL = numba.njit(parallel=True, fastmath=False, cache=True)(_sample_impl)
        return _KERNEL


def try_sample(volume, locations, fill_value=0):
    """New FP32 CPU result, or None for the original ordered Torch sampler.

    The caller keeps grid/einsum locations and the CPUjoint eval guard. This
    helper changes no global floating/OMP policy and restores the NumBa mask.
    """
    global _FAILURE
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        return _fallback('unsupported CPU platform')
    if (not isinstance(volume, torch.Tensor) or not isinstance(locations, torch.Tensor)
            or volume.device.type != 'cpu' or locations.device.type != 'cpu'
            or volume.dtype != torch.float32 or locations.dtype != torch.float32
            or volume.ndim != 5 or locations.ndim != 5 or locations.shape[:2] != (1, 3)
            or min(volume.shape) < 1 or min(locations.shape) < 1):
        return _fallback('unsupported CPU FP32 tensor contract')
    if (torch.is_grad_enabled() or volume.requires_grad or locations.requires_grad or _autocast_cpu()):
        return _fallback('autograd or autocast retains Torch')
    if locations[0, 0].numel() < _MIN_VOXELS:
        return _fallback('small sampling grid retains Torch')
    if os.environ.get('FNIT_SYNTHMORPH_CPU_RAW_NUMBA', '1') == '0':
        return _fallback('CPU raw NumBa sampler disabled')
    if _FAILURE is not None:
        return _fallback(_FAILURE)
    if numba is None or numba.config.DISABLE_JIT:
        return _fallback('NumBa unavailable or JIT disabled')
    if not torch.isfinite(volume).all() or not torch.isfinite(locations).all():
        return _fallback('nonfinite tensor retains Torch')
    try:
        fill = np.float32(0 if fill_value is None else fill_value)
        if fill.ndim != 0 or not np.isfinite(fill):
            return _fallback('nonfinite or nonscalar fill retains Torch')
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return _fallback('unsupported fill retains Torch')
    try:
        previous = numba.get_num_threads()
        requested = min(previous, torch.get_num_threads())
    except Exception as error:
        _FAILURE = 'NumBa thread initialization failed: ' + str(error)
        return _fallback(_FAILURE)
    try:
        numba.set_num_threads(requested)
        output = _kernel()(volume.numpy(), locations.numpy(), fill_value is not None, fill)
        result = torch.from_numpy(output)
        _INFO.value = {'backend': 'numba', 'reason': 'ordered FP32 raw sampling',
                       'requested_threads': requested, 'previous_threads': previous,
                       'numba_version': numba.__version__}
        return result
    except Exception as error:
        _FAILURE = 'NumBa sampling failed: ' + str(error)
        return _fallback(_FAILURE, requested_threads=requested, previous_threads=previous)
    finally:
        numba.set_num_threads(previous)
