"""CPU-only FP64 pointwise normal products; CUDA retains the tensor path.

Numba is imported lazily by the CPU optimized linearization. No thread mask
or PyTorch thread setting is changed here. Multiplication, addition and final
division follow the existing unfused tensor operations; fastmath is disabled.
"""
from __future__ import annotations

from numba import config, get_num_threads, njit, prange
import numpy as np
import torch
from ._normal_simd_cpu import normal_block8


@njit(inline="always", fastmath=False, error_model="numpy")
def _normal_voxel(field, weights, cross, scale, fit_scale, count, output, x, y, z):
    d0, d1, d2 = field[0, x, y, z], field[1, x, y, z], field[2, x, y, z]
    v0 = 0.0 + weights[0][x, y, z] * d0
    v0 = v0 + weights[1][x, y, z] * d1
    v0 = v0 + weights[2][x, y, z] * d2
    v1 = 0.0 + weights[3][x, y, z] * d0
    v1 = v1 + weights[4][x, y, z] * d1
    v1 = v1 + weights[5][x, y, z] * d2
    v2 = 0.0 + weights[6][x, y, z] * d0
    v2 = v2 + weights[7][x, y, z] * d1
    v2 = v2 + weights[8][x, y, z] * d2
    if fit_scale:
        v0 = v0 + cross[0][x, y, z] * scale
        v1 = v1 + cross[1][x, y, z] * scale
        v2 = v2 + cross[2][x, y, z] * scale
    output[0, x, y, z] = v0 / count
    output[1, x, y, z] = v1 / count
    output[2, x, y, z] = v2 / count


def _weighted_normal(field, weights, cross, scale, fit_scale, count, output, fastest):
    # Separable einsum returns a Y-contiguous field. Preserve that layout for
    # the adjoint and visit its contiguous dimension in the innermost loop.
    # Independent voxels may be visited in any order without changing sums.
    if fastest == 1:
        for z in prange(field.shape[3]):
            for x in range(field.shape[1]):
                for y in range(field.shape[2]):
                    _normal_voxel(field, weights, cross, scale, fit_scale, count, output, x, y, z)
    elif fastest == 0:
        for z in prange(field.shape[3]):
            for y in range(field.shape[2]):
                for x in range(field.shape[1]):
                    _normal_voxel(field, weights, cross, scale, fit_scale, count, output, x, y, z)
    else:
        for x in prange(field.shape[1]):
            for y in range(field.shape[2]):
                for z in range(field.shape[3]):
                    _normal_voxel(field, weights, cross, scale, fit_scale, count, output, x, y, z)


_serial = njit(cache=True, fastmath=False, error_model="numpy")(_weighted_normal)
_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_weighted_normal)


@njit(inline="always", fastmath=False, error_model="numpy")
def _flat_voxel(field, weights, cross, scale, fit_scale, count, output, source, voxel, inner):
    # Read all channels before writing: field may be this operator's scratch.
    d0, d1, d2 = field[source], field[source + inner], field[source + 2 * inner]
    v0 = 0.0 + weights[0][voxel] * d0
    v0 = v0 + weights[1][voxel] * d1
    v0 = v0 + weights[2][voxel] * d2
    v1 = 0.0 + weights[3][voxel] * d0
    v1 = v1 + weights[4][voxel] * d1
    v1 = v1 + weights[5][voxel] * d2
    v2 = 0.0 + weights[6][voxel] * d0
    v2 = v2 + weights[7][voxel] * d1
    v2 = v2 + weights[8][voxel] * d2
    if fit_scale:
        v0 = v0 + cross[0][voxel] * scale
        v1 = v1 + cross[1][voxel] * scale
        v2 = v2 + cross[2][voxel] * scale
    output[source] = v0 / count
    output[source + inner] = v1 / count
    output[source + 2 * inner] = v2 / count


def _weighted_normal_flat(field, weights, cross, scale, fit_scale, count, output, inner):
    groups = field.size // (3 * inner)
    for group in prange(groups):
        source = group * 3 * inner
        voxel = group * inner
        position = 0
        while position + 8 <= inner:
            used = normal_block8(field, weights, cross, output, source + position,
                                 voxel + position, inner, scale, fit_scale, count)
            if not used:
                for lane in range(8):
                    _flat_voxel(field, weights, cross, scale, fit_scale, count, output,
                                source + position + lane, voxel + position + lane, inner)
            position += 8
        for position in range(position, inner):
            _flat_voxel(field, weights, cross, scale, fit_scale, count, output,
                        source + position, voxel + position, inner)


_flat_serial = njit(cache=True, fastmath=False, error_model="numpy")(_weighted_normal_flat)
_flat_parallel = njit(cache=True, fastmath=False, error_model="numpy", parallel=True)(_weighted_normal_flat)


class SpatialNormalCPU:
    """One fixed-weight layout and one scratch for a single linearization.

    Weights are constant throughout that linearization's PCG solve. Packing
    only copies FP64 bits; each newly constructed operator owns its cache.
    """

    def __init__(self, weights, cross_weights, count):
        if len(weights) != 3 or any(len(row) != 3 for row in weights):
            raise ValueError("CPU fused normal requires a 3-by-3 weight matrix")
        if cross_weights is not None and len(cross_weights) != 3:
            raise ValueError("CPU fused normal requires three cross weights")
        tensors = [value for row in weights for value in row]
        if cross_weights is not None:
            tensors.extend(cross_weights)
        if any(value.device.type != "cpu" or value.dtype != torch.float64 or value.requires_grad for value in tensors):
            raise ValueError("CPU fused normal requires non-differentiable FP64 CPU weights")
        self._geometry = tuple(tensors[0].shape)
        if len(self._geometry) != 3 or any(tuple(value.shape) != self._geometry for value in tensors):
            raise ValueError("Linearization weight geometry must be one matching 3D grid")
        self.weights = tuple(value.numpy() for row in weights for value in row)
        self.cross = tuple(value.numpy() for value in cross_weights) if cross_weights is not None else self.weights[:3]
        self._source_weights = self.weights
        self._source_cross = self.cross if cross_weights is not None else None
        self._layout = None
        self.flat_weights = None
        self.flat_cross = None
        self.layout_copy_bytes = 0
        self.count = float(count)
        self.scratch = None

    def _prepare_layout(self, field):
        # Match the dense field's spatial traversal. Normally the expanded
        # field is Y-contiguous while image-derived weights are Z-contiguous.
        axes = tuple(sorted(range(3), key=lambda axis: field.stride(axis + 1), reverse=True))
        if axes == self._layout:
            return
        inverse_axes = tuple(np.argsort(axes))
        copied = 0

        def pack(value):
            nonlocal copied
            if value.shape != tuple(field.shape[1:]):
                raise ValueError("Linearization weight geometry changed")
            ordered = value.transpose(axes)
            packed = np.ascontiguousarray(ordered)
            if not np.shares_memory(packed, value):
                copied += packed.nbytes
            return packed.transpose(inverse_axes)

        self.weights = tuple(pack(value) for value in self._source_weights)
        self.cross = tuple(pack(value) for value in self._source_cross) if self._source_cross is not None else self.weights[:3]
        self._layout = axes
        self.layout_copy_bytes = copied
        # These are views of the already packed arrays, not additional PCG
        # copies. Their C layout gives the compiler a contiguous inner loop.
        self.flat_weights = tuple(value.transpose(axes).reshape(-1) for value in self.weights)
        self.flat_cross = tuple(value.transpose(axes).reshape(-1) for value in self.cross)
        if any(not np.shares_memory(flat, value) for flat, value in zip(self.flat_weights, self.weights)):
            raise RuntimeError("Packed normal weights require physical views")
        if any(not np.shares_memory(flat, value) for flat, value in zip(self.flat_cross, self.cross)):
            raise RuntimeError("Packed cross weights require physical views")

    def __call__(self, field, scale):
        if field.device.type != "cpu" or field.dtype != torch.float64 or field.requires_grad:
            raise ValueError("CPU fused normal requires a non-differentiable FP64 CPU field")
        if tuple(field.shape) != (3, *self._geometry):
            raise ValueError("Linearization field geometry changed")
        self._prepare_layout(field)
        if self.scratch is None or self.scratch.stride() != field.stride():
            self.scratch = torch.empty_like(field)
        if self.scratch.shape != field.shape:
            raise ValueError("Linearization field geometry changed")
        # Respect both callers' budgets, without changing either global setting.
        # A larger pre-existing Numba pool uses the serial kernel, never all CPUs.
        budget = torch.get_num_threads()
        # get_num_threads() initializes Numba's pool. Avoid creating an
        # oversized default pool even when the serial branch will be used.
        threads = get_num_threads() if config.NUMBA_NUM_THREADS <= budget else 1
        scalar = 0.0 if scale is None else float(scale)
        parallel = 1 < threads <= budget
        # Permuting to stride order is a contiguity check only. Ravel K must
        # expose the same physical buffer; no dense field is copied per PCG.
        order = tuple(sorted(range(4), key=field.stride, reverse=True))
        inner = field.stride(0)
        if (inner > 0 and field.permute(order).is_contiguous()
                and self.scratch.permute(order).is_contiguous()
                and (not parallel or field.numel() // (3 * inner) >= threads)):
            field_array, output_array = field.numpy(), self.scratch.numpy()
            field_flat, output_flat = field_array.ravel(order="K"), output_array.ravel(order="K")
            if np.shares_memory(field_flat, field_array) and np.shares_memory(output_flat, output_array):
                kernel = _flat_parallel if parallel else _flat_serial
                kernel(field_flat, self.flat_weights, self.flat_cross, scalar, scale is not None,
                       self.count, output_flat, inner)
                return self.scratch
        kernel = _parallel if parallel else _serial
        fastest = min(range(3), key=lambda axis: field.stride(axis + 1))
        kernel(field.numpy(), self.weights, self.cross, scalar, scale is not None, self.count, self.scratch.numpy(), fastest)
        return self.scratch
