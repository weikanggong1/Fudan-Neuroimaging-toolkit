"""CPU tetrahedron lookup; interpolation and its autograd stay in PyTorch."""

import numpy as np
from llvmlite import ir
from numba import config, get_num_threads, njit, prange, set_num_threads
from numba import types
from numba.extending import intrinsic
import torch


@intrinsic
def _fma32(typing_context, first, second, third):
    if first != types.float32 or second != first or third != first:
        return None
    signature = first(first, second, third)
    def codegen(context, builder, sig, args):
        floating = ir.FloatType()
        function = builder.module.declare_intrinsic(
            "llvm.fma", [floating], fnty=ir.FunctionType(floating, [floating] * 3))
        return builder.call(function, args)
    return signature, codegen


@njit(cache=True, parallel=True, fastmath=False)
def _lookup(points, ids, included, origins, inverses, singular, rows, tolerance):
    count = rows.size
    selected = np.empty(count, dtype=np.int64)
    covered = np.zeros(count, dtype=np.bool_)
    width = points.shape[1]
    for row in prange(count):
        block = rows[row] // width
        point = rows[row] % width
        best = -np.inf
        owner = ids[block, 0]
        for column in range(ids.shape[1]):
            cell = ids[block, column]
            if not included[block, column] or singular[cell]:
                continue
            dx = points[block, point, 0] - origins[cell, 0]
            dy = points[block, point, 1] - origins[cell, 1]
            dz = points[block, point, 2] - origins[cell, 2]
            w1 = _fma32(inverses[cell, 0, 2], dz, _fma32(
                inverses[cell, 0, 1], dy, inverses[cell, 0, 0] * dx))
            w2 = _fma32(inverses[cell, 1, 2], dz, _fma32(
                inverses[cell, 1, 1], dy, inverses[cell, 1, 0] * dx))
            w3 = _fma32(inverses[cell, 2, 2], dz, _fma32(
                inverses[cell, 2, 1], dy, inverses[cell, 2, 0] * dx))
            w0 = np.float32(1) - ((w1 + w2) + w3)
            score = min(w0, w1, w2, w3)
            if np.isnan(w0) or np.isnan(w1) or np.isnan(w2) or np.isnan(w3):
                # torch.amin propagates NaN and torch.max chooses its first
                # candidate. The associated voxel remains uncovered.
                owner = cell
                best = np.nan
                break
            # The original max returns the first candidate on a tie.
            if score > best:
                best = score
                owner = cell
        selected[row] = owner
        covered[row] = best >= -tolerance
    return selected, covered


def lookup_candidates_cpu(points, ids, candidate_mask, all_v0, all_inv,
                          all_singular, point_rows, *, tolerance=2e-5,
                          previous_selected=None, hint_tolerance=2e-4,
                          return_hint_hits=False):
    """Select real rows on CPU without allocating point-by-candidate tensors.

    FP32 keeps ordered products and explicit FMA, with fastmath disabled.
    Numba temporarily uses at most the current Torch intraop budget and its
    initialized pool capacity. CUDA tensors return None and never enter here.
    Owner hints are not used; the full ordered candidate list is evaluated.
    """
    tensors = (points, ids, candidate_mask, all_v0, all_inv, all_singular, point_rows)
    if any(value.device.type != "cpu" for value in tensors):
        return None
    if points.dtype != torch.float32:
        return None
    try:
        autocast_enabled = torch.is_autocast_enabled("cpu")
    except TypeError:
        autocast_enabled = torch.is_autocast_cpu_enabled()
    if autocast_enabled:
        return None
    if all_v0.dtype != points.dtype or all_inv.dtype != points.dtype:
        return None
    if not torch.backends.mkl.is_available():
        return None
    if ids.dtype != torch.long or point_rows.dtype != torch.long:
        return None
    if candidate_mask.dtype != torch.bool or all_singular.dtype != torch.bool:
        return None
    if points.ndim != 3 or points.shape[-1] != 3 or ids.ndim != 2:
        return None
    if ids.shape[0] != points.shape[0] or candidate_mask.shape != ids.shape:
        return None
    if all_v0.shape != (len(all_singular), 3) or all_inv.shape != (len(all_singular), 3, 3):
        return None
    if point_rows.ndim != 1 or not ids.shape[1] or not points.shape[1] or not len(all_singular):
        return None
    if (bool((ids < 0).any()) or bool((ids >= len(all_singular)).any())
            or bool((point_rows < 0).any())
            or bool((point_rows >= points.shape[0] * points.shape[1]).any())):
        return None
    previous_threads = get_num_threads()
    desired = min(torch.get_num_threads(), int(config.NUMBA_NUM_THREADS))
    try:
        if previous_threads != desired:
            set_num_threads(desired)
        # Torch compares its FP32 scores against a scalar rounded to FP32.
        result = _lookup(*(value.detach().numpy() for value in tensors), np.float32(tolerance))
    finally:
        if previous_threads != desired:
            set_num_threads(previous_threads)
    selected, covered = torch.from_numpy(result[0]), torch.from_numpy(result[1])
    return (selected, covered, None) if return_hint_hits else (selected, covered)
