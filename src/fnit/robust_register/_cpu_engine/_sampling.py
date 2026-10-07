"""Source-defined interpolation and pyramid for the standalone CPU robust engine.

Modified PyTorch adaptation of FreeSurfer d932c45 MRI routines; retain
licenses/FreeSurfer.txt. Not an official FreeSurfer release. The general
FNIT samplers keep their original boundary and precision policies.

Original author: Martin Reuter. Copyright (c) 2021 The General Hospital
Corporation (Boston, MA), "MGH". B-spline algorithms: Thevenaz, Blu and
Unser (2000). FNIT modification: batched PyTorch computation, 2026.
"""
from __future__ import annotations

import math

import numpy as np
import torch


PYRAMID_KERNEL = (.0625, .25, .375, .25, .0625)
PREFILTER = (.03504, .24878, .43234, .24878, .03504)
DERIVATIVE = (-.10689, -.28461, 0., .28461, .10689)
CENTERED_SPLINE = (
    .708792, .328616, -.165157, -.114448, .0944036, .0543881,
    -.05193, -.0284868, .0281854, .0152877, -.0152508,
    -.00825077, .00824629, .00445865, -.0044582, -.00241009,
    .00241022, .00130278, -.00130313, -.000704109, .000704784,
)

def filter_axis(values, kernel, axis):
    """Ordered FP32 taps with MRI xi/yi/zi edge replication."""
    if values.shape[axis] == 1:
        return values.clone()
    index = torch.arange(values.shape[axis], device=values.device)
    result = torch.zeros_like(values)
    taps = torch.tensor(kernel, dtype=torch.float32, device=values.device)
    for tap in range(len(kernel)):
        selected = (index + tap - len(kernel) // 2).clamp(0, len(index) - 1)
        result = result + values.index_select(axis, selected) * taps[tap]
    return result


def blur(values, kernel=PREFILTER):
    for axis in range(3):
        values = filter_axis(values, kernel, axis)
    return values


def partials(values):
    """Native ordered x/y/z derivative passes plus smoothed intensity."""
    dx = filter_axis(values, DERIVATIVE, 0)
    dx = filter_axis(filter_axis(dx, PREFILTER, 1), PREFILTER, 2)
    bx = filter_axis(values, PREFILTER, 0)
    dy = filter_axis(filter_axis(bx, DERIVATIVE, 1), PREFILTER, 2)
    bxy = filter_axis(bx, PREFILTER, 1)
    dz = filter_axis(bxy, DERIVATIVE, 2)
    return dx, dy, dz, filter_axis(bxy, PREFILTER, 2)


def reduce_axis(values, axis):
    """Centered spline order3, endpoint-repeated mirror and Haar at .5.

    Odd lines are truncated to 2*floor(n/2) BEFORE the mirror operation.
    Double accumulation is rounded to FP32 after each axis, as MRI_FLOAT.
    The 21-tap loop is over coefficients, not voxels.
    """
    length = values.shape[axis]
    if length == 1:
        return values.clone()
    length = (length // 2) * 2
    values = values.narrow(axis, 0, length).double()
    index = torch.arange(length, device=values.device)

    def reflect(q):
        q = q.remainder(2 * length)
        return torch.where(q >= length, 2 * length - q - 1, q)

    result = values * CENTERED_SPLINE[0]
    for offset, weight in enumerate(CENTERED_SPLINE[1:], 1):
        result = result + weight * (
            values.index_select(axis, reflect(index - offset))
            + values.index_select(axis, reflect(index + offset))
        )
    even = result.index_select(axis, index[::2])
    odd = result.index_select(axis, index[1::2])
    return ((even + odd) / 2).float()


def downsample(values):
    values = blur(values, PYRAMID_KERNEL)
    for axis in range(3):
        values = reduce_axis(values, axis)
    return values


def cubic_coefficients(values):
    """Double recursive cubic pole, with FP32 MRI storage after each axis.

    All lines of an axis are processed together. This is intentionally
    separate from FNIT's mature FFT spline helper, which keeps Double
    coefficients across axes and has a different rounding boundary.
    """
    pole = math.sqrt(3.) - 2.
    horizon = math.ceil(math.log(np.finfo(float).eps) / math.log(abs(pole)))
    gain = (1 - pole) * (1 - 1 / pole)
    for axis in range(3):
        n = values.shape[axis]
        if n == 1:
            continue
        lines = values.movedim(axis, 0).double().clone() * gain
        if horizon < n:
            initial = lines[0].clone()
            zn = pole
            for i in range(1, horizon):
                initial = initial + zn * lines[i]
                zn *= pole
        else:
            zn = pole
            z2n = pole ** (n - 1)
            initial = lines[0] + z2n * lines[-1]
            z2n *= z2n / pole
            for i in range(1, n - 1):
                initial = initial + (zn + z2n) * lines[i]
                zn *= pole
                z2n /= pole
            initial = initial / (1 - zn * zn)
        lines[0] = initial
        for i in range(1, n):
            lines[i] = lines[i] + pole * lines[i - 1]
        lines[-1] = (pole / (pole * pole - 1)) * (pole * lines[-2] + lines[-1])
        for i in range(n - 2, -1, -1):
            lines[i] = pole * (lines[i + 1] - lines[i])
        values = lines.float().movedim(0, axis).contiguous()
    return values


def _inside_round_support(coordinates, shape):
    # MRIindexNotInVolume uses C rint (ties to even), then Float storage.
    rounded = coordinates.round()
    valid = torch.ones_like(coordinates[0], dtype=torch.bool)
    for axis, size in enumerate(shape):
        valid &= (rounded[axis] >= 0) & (rounded[axis] < size)
    return valid


def _linear(values, coordinates):
    q = coordinates.double()
    valid = _inside_round_support(q, values.shape)
    lo, hi, up = [], [], []
    for axis, n in enumerate(values.shape):
        c = q[axis].clamp(0, n - 1)
        l = c.long()
        lo.append(l)
        hi.append((l + 1).clamp_max(n - 1))
        up.append(c - l.double())
    result = torch.zeros_like(q[0])
    # Same x-major eight-term order as MRIsampleVolumeFrame.
    for x in range(2):
        for y in range(2):
            for z in range(2):
                bits = (x, y, z)
                weight = [(up[a] if bits[a] else 1 - up[a]) for a in range(3)]
                index = [(hi[a] if bits[a] else lo[a]) for a in range(3)]
                result = result + (weight[0] * weight[1] * weight[2]) * values[tuple(index)].double()
    # Native FEQUAL integer shortcut, FLT_EPSILON from float.h.
    integer = torch.trunc(q)
    near = ((q - integer).abs() < np.finfo(np.float32).eps).all(dim=0)
    nearest = [q[a].round().long().clamp(0, n - 1) for a, n in enumerate(values.shape)]
    result = torch.where(near, values[tuple(nearest)].double(), result)
    return torch.where(valid, result, 0).float()


def _cubic(coefficients, coordinates, *, nonnegative):
    q = coordinates.double()
    valid = _inside_round_support(q, coefficients.shape)
    ids, weights = [], []
    for axis, n in enumerate(coefficients.shape):
        if n == 1:
            ids.append([torch.zeros_like(q[axis], dtype=torch.long)])
            weights.append([torch.ones_like(q[axis])])
            continue
        start = torch.floor(q[axis]).long() - 1
        fraction = q[axis] - (start + 1).double()
        w3 = fraction ** 3 / 6
        w0 = 1 / 6 + .5 * fraction * (fraction - 1) - w3
        w2 = fraction + w0 - 2 * w3
        w1 = 1 - w0 - w2 - w3
        weights.append([w0, w1, w2, w3])
        indices = []
        for i in range(4):
            index = (start + i).abs().remainder(2 * n - 2)
            indices.append(torch.where(index >= n, 2 * n - 2 - index, index))
        ids.append(indices)
    result = torch.zeros_like(q[0])
    # Original nested double x -> y -> z accumulations.
    for z in range(len(ids[2])):
        row = torch.zeros_like(result)
        for y in range(len(ids[1])):
            line = torch.zeros_like(result)
            for x in range(len(ids[0])):
                line = line + weights[0][x] * coefficients[ids[0][x], ids[1][y], ids[2][z]].double()
            row = row + weights[1][y] * line
        result = result + weights[2][z] * row
    if nonnegative:
        result = result.clamp_min(0)
    return torch.where(valid, result, 0).float()


def native_matmul(left, right):
    """MATRIX_REAL: four ordered FP32 multiplies/adds, independent of TF32."""
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    result = np.zeros((left.shape[0], right.shape[1]), np.float32)
    for k in range(left.shape[1]):
        result = np.float32(result + np.float32(left[:, k, None] * right[None, k, :]))
    return result


def native_inverse(matrix):
    # Isolated VNL source-order inverse; mature FNIT implementation unchanged.
    from .inverse import inverse_float_source_order
    result = inverse_float_source_order(np.asarray(matrix, np.float32))["inverse"]
    if not np.isfinite(result).all():
        raise ValueError("sampling transform is singular")
    return result


def resample(values, target_shape, pull, *, cubic=False, chunk_size=131072):
    """Pull in source voxels; Float MATRIX multiply, Double interpolation."""
    shape = tuple(int(v) for v in target_shape)
    output = torch.empty(math.prod(shape), device=values.device, dtype=torch.float32)
    matrix = torch.as_tensor(np.asarray(pull, np.float32), device=values.device)
    coefficients = cubic_coefficients(values) if cubic else values
    nonnegative = bool((values >= 0).all()) if cubic else False
    for start in range(0, output.numel(), chunk_size):
        end = min(start + chunk_size, output.numel())
        index = torch.arange(start, end, device=values.device)
        grid = (torch.div(index, shape[1] * shape[2], rounding_mode="floor").float(),
                torch.div(index, shape[2], rounding_mode="floor").remainder(shape[1]).float(),
                index.remainder(shape[2]).float())
        coordinates = []
        for a in range(3):
            c = torch.zeros_like(grid[0])
            for k in range(3):
                c = c + matrix[a, k] * grid[k]
            coordinates.append(c + matrix[a, 3])
        q = torch.stack(coordinates)
        output[start:end] = (_cubic(coefficients, q, nonnegative=nonnegative)
                             if cubic else _linear(values, q))
    return output.reshape(shape)
