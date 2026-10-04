"""Fused float32 sampling for CPU FLIRT, without changing cost reductions.

Voxel arithmetic and ordered bin accumulation are compiled. Candidate order,
the Brent search, image filtering and final bin-level cost reduction stay in
:mod:`core`. ``fastmath=False`` keeps each float32 multiply/add separate.
CUDA never imports or executes this module.
"""

import numpy as np
from numba import njit, prange
from fnit.flirt._cpu_simd import sample_block8, sample_block8_weighted


@njit(cache=True, fastmath=False, inline="always")
def _coordinate(coefficients, row, x, y, z):
    value = np.float32(np.float32(y * coefficients[row, 1])
                       + np.float32(z * coefficients[row, 2]))
    value = np.float32(value + coefficients[row, 3])
    return np.float32(value + np.float32(x * coefficients[row, 0]))


# Keep these array-taking helper boundaries in Numba IR. Forcing IR inlining
# creates array aliases with NRT retain/release inside every voxel iteration.
# LLVM can still inline their scalar machine code without that atomic overhead.
@njit(cache=True, fastmath=False, inline="never")
def _trilinear_at(data, x, y, z, ix, iy, iz):
    dx, dy, dz = (np.float32(x - np.float32(ix)),
                  np.float32(y - np.float32(iy)),
                  np.float32(z - np.float32(iz)))
    temp1 = np.float32(np.float32(np.float32(data[ix + 1, iy, iz] - data[ix, iy, iz]) * dx) + data[ix, iy, iz])
    temp2 = np.float32(np.float32(np.float32(data[ix + 1, iy, iz + 1] - data[ix, iy, iz + 1]) * dx) + data[ix, iy, iz + 1])
    temp3 = np.float32(np.float32(np.float32(data[ix + 1, iy + 1, iz] - data[ix, iy + 1, iz]) * dx) + data[ix, iy + 1, iz])
    temp4 = np.float32(np.float32(np.float32(data[ix + 1, iy + 1, iz + 1] - data[ix, iy + 1, iz + 1]) * dx) + data[ix, iy + 1, iz + 1])
    temp5 = np.float32(np.float32(np.float32(temp3 - temp1) * dy) + temp1)
    temp6 = np.float32(np.float32(np.float32(temp4 - temp2) * dy) + temp2)
    return np.float32(np.float32(np.float32(temp6 - temp5) * dz) + temp5)


@njit(cache=True, fastmath=False, inline="never")
def _trilinear(data, x, y, z):
    ix = min(max(int(np.floor(x)), 0), data.shape[0] - 2)
    iy = min(max(int(np.floor(y)), 0), data.shape[1] - 2)
    iz = min(max(int(np.floor(z)), 0), data.shape[2] - 2)
    return _trilinear_at(data, x, y, z, ix, iy, iz)


# This helper is called only after the whole-row proof establishes
# nonnegative coordinates. The caller clips floor indices to size-2 for
# rounded large-axis endpoints before unsigned +1 neighbours. Unsigned loads avoid
# repeated Python-style negative-index corrections; boundary rows keep
# the original signed helper, including thin-axis behaviour.
@njit(cache=True, fastmath=False, inline='never')
def _trilinear_at_inside(data, x, y, z, ix, iy, iz):
    (dx, dy, dz) = (np.float32(x - np.float32(ix)), np.float32(y - np.float32(iy)), np.float32(z - np.float32(iz)))
    (ux, uy, uz) = (np.uint64(ix), np.uint64(iy), np.uint64(iz))
    (ux1, uy1, uz1) = (ux + np.uint64(1), uy + np.uint64(1), uz + np.uint64(1))
    temp1 = np.float32(np.float32(np.float32(data[ux1, uy, uz] - data[ux, uy, uz]) * dx) + data[ux, uy, uz])
    temp2 = np.float32(np.float32(np.float32(data[ux1, uy, uz1] - data[ux, uy, uz1]) * dx) + data[ux, uy, uz1])
    temp3 = np.float32(np.float32(np.float32(data[ux1, uy1, uz] - data[ux, uy1, uz]) * dx) + data[ux, uy1, uz])
    temp4 = np.float32(np.float32(np.float32(data[ux1, uy1, uz1] - data[ux, uy1, uz1]) * dx) + data[ux, uy1, uz1])
    temp5 = np.float32(np.float32(np.float32(temp3 - temp1) * dy) + temp1)
    temp6 = np.float32(np.float32(np.float32(temp4 - temp2) * dy) + temp2)
    return np.float32(np.float32(np.float32(temp6 - temp5) * dz) + temp5)


@njit(cache=True, fastmath=False, inline="never")
def _trilinear_inside(data, x, y, z):
    return _trilinear_at_inside(data, x, y, z,
                         min(int(np.floor(x)), data.shape[0] - 2),
                         min(int(np.floor(y)), data.shape[1] - 2),
                         min(int(np.floor(z)), data.shape[2] - 2))


@njit(cache=True, fastmath=False, error_model="numpy", inline="always")
def _taper(coordinate, upper, smooth):
    far = np.float32(upper - coordinate)
    if coordinate < smooth:
        return np.float32(coordinate / smooth)
    if far < smooth:
        return np.float32(far / smooth)
    return np.float32(1)


@njit(cache=True, fastmath=False, error_model="numpy")
def sample_cost(moving, coefficients, reference_shape, upper, smooth,
                taper, moving_weight, reference_weight, weighted):
    """Sample in the reference cost's x-fastest order; return float32 arrays."""
    nx, ny, nz = reference_shape
    size = nx * ny * nz
    values = np.empty(size, dtype=np.float32)
    weights = np.empty(size, dtype=np.float32)
    any_valid = False
    index = 0
    for z_index in range(nz):
        z = np.float32(z_index)
        for y_index in range(ny):
            y = np.float32(y_index)
            base_x = np.float32(np.float32(np.float32(y * coefficients[0, 1])
                                 + np.float32(z * coefficients[0, 2])) + coefficients[0, 3])
            base_y = np.float32(np.float32(np.float32(y * coefficients[1, 1])
                                 + np.float32(z * coefficients[1, 2])) + coefficients[1, 3])
            base_z = np.float32(np.float32(np.float32(y * coefficients[2, 1])
                                 + np.float32(z * coefficients[2, 2])) + coefficients[2, 3])
            last_x = np.float32(nx - 1)
            end_x = np.float32(base_x + np.float32(last_x * coefficients[0, 0]))
            end_y = np.float32(base_y + np.float32(last_x * coefficients[1, 0]))
            end_z = np.float32(base_z + np.float32(last_x * coefficients[2, 0]))
            # Each float32 affine coordinate is monotone in x. If both
            # endpoints are inside the strict cost field, every lower index
            # is nonnegative. Validity tests are redundant; the helper still clips
            # the floor to size-2 when a large-axis upper rounds to size-1.
            row_inside = (np.isfinite(end_x) and np.isfinite(end_y) and np.isfinite(end_z)
                          and min(base_x, end_x) >= 0 and max(base_x, end_x) <= upper[0]
                          and min(base_y, end_y) >= 0 and max(base_y, end_y) <= upper[1]
                          and min(base_z, end_z) >= 0 and max(base_z, end_z) <= upper[2])
            for x_index in range(nx):
                x = np.float32(x_index)
                cx = np.float32(base_x + np.float32(x * coefficients[0, 0]))
                cy = np.float32(base_y + np.float32(x * coefficients[1, 0]))
                cz = np.float32(base_z + np.float32(x * coefficients[2, 0]))
                if row_inside:
                    valid = True
                    sx, sy, sz = cx, cy, cz
                else:
                    valid = (cx >= 0 and cx <= upper[0] and cy >= 0
                             and cy <= upper[1] and cz >= 0 and cz <= upper[2])
                    sx, sy, sz = (min(max(cx, np.float32(0)), upper[0]),
                                  min(max(cy, np.float32(0)), upper[1]),
                                  min(max(cz, np.float32(0)), upper[2]))
                any_valid = any_valid or valid
                values[index] = (_trilinear_inside(moving, sx, sy, sz) if row_inside
                                  else _trilinear(moving, sx, sy, sz))
                weight = np.float32(1)
                if taper:
                    wx = _taper(cx, upper[0], smooth[0])
                    wy = _taper(cy, upper[1], smooth[1])
                    wz = _taper(cz, upper[2], smooth[2])
                    weight = max(np.float32(np.float32(wx * wy) * wz), np.float32(0))
                if weighted:
                    sampled_weight = (_trilinear_inside(moving_weight, sx, sy, sz)
                                      if row_inside else _trilinear(moving_weight, sx, sy, sz))
                    weight = np.float32(weight * sampled_weight)
                    weight = max(np.float32(weight * reference_weight[index]), np.float32(0))
                weights[index] = np.float32(weight * np.float32(valid))
                index += 1
    return values, weights, any_valid


@njit(cache=True, fastmath=False)
def sample_output(moving, coefficients, output_shape, background):
    """Sample final output in C order, retaining FLIRT's edge background."""
    nx, ny, nz = output_shape
    output = np.empty(output_shape, dtype=np.float32)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                cx = _coordinate(coefficients, 0, np.float32(x), np.float32(y), np.float32(z))
                cy = _coordinate(coefficients, 1, np.float32(x), np.float32(y), np.float32(z))
                cz = _coordinate(coefficients, 2, np.float32(x), np.float32(y), np.float32(z))
                if (cx >= 0 and cx <= moving.shape[0] - 1
                        and cy >= 0 and cy <= moving.shape[1] - 1
                        and cz >= 0 and cz <= moving.shape[2] - 1):
                    output[x, y, z] = _trilinear(moving, cx, cy, cz)
                else:
                    output[x, y, z] = background
    return output


@njit(cache=True, fastmath=False)
def histogram_nmi(values, weights, reference_bins, bins, factor, offset):
    """Preserve the three original float64 scatter passes and float32 products."""
    stride = bins + 1
    joint = np.zeros((stride, stride), dtype=np.float64)
    for pass_index in range(3):
        for index in range(values.size):
            bin_float = np.float32(np.float32(values[index] * factor) + offset)
            # The tensor reference rejects scatter indices outside the joint
            # histogram. Reject before converting to an integer: NaN/overflow
            # would otherwise become an unchecked native array index.
            if (not np.isfinite(bin_float) or bin_float <= -2
                    or bin_float >= bins + 2):
                raise ValueError("NMI sampled intensity maps outside the finite histogram")
            truncated = np.float32(np.trunc(bin_float))
            raw_centre = int(truncated)
            fractional = np.float32(abs(np.float32(bin_float - truncated)))
            if fractional < np.float32(0.5):
                centre_weight = np.float32(np.float32(0.5) + fractional)
            elif fractional > np.float32(0.5):
                centre_weight = np.float32(np.float32(1.5) - fractional)
            else:
                centre_weight = np.float32(1)
            centre_weight = min(max(centre_weight, np.float32(0)), np.float32(1))
            if pass_index == 0:
                target = min(max(raw_centre, 0), bins - 1)
                bin_weight = centre_weight
            elif pass_index == 1:
                target = max(raw_centre - 1, 0)
                bin_weight = (np.float32(np.float32(1) - centre_weight)
                              if fractional < np.float32(0.5) else np.float32(0))
            else:
                target = min(raw_centre + 1, bins - 1)
                bin_weight = (np.float32(np.float32(1) - centre_weight)
                              if fractional > np.float32(0.5) else np.float32(0))
            joint[reference_bins[index], target] += np.float64(np.float32(weights[index] * bin_weight))
    return joint


@njit(cache=True, fastmath=False, error_model="numpy")
def sample_corratio(moving, coefficients, reference_shape, upper, smooth,
                taper, moving_weight, reference_weight, weighted, bin_index, bins):
    """Sample one row at a time and preserve each bin's float32 sum order."""
    nx, ny, nz = reference_shape
    values = np.empty(nx, dtype=np.float32)
    weights = np.empty(nx, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.float32)
    sums = np.zeros(bins, dtype=np.float32)
    sums2 = np.zeros(bins, dtype=np.float32)
    any_valid = False
    index = 0
    for z_index in range(nz):
        z = np.float32(z_index)
        for y_index in range(ny):
            y = np.float32(y_index)
            base_x = np.float32(np.float32(np.float32(y * coefficients[0, 1])
                                 + np.float32(z * coefficients[0, 2])) + coefficients[0, 3])
            base_y = np.float32(np.float32(np.float32(y * coefficients[1, 1])
                                 + np.float32(z * coefficients[1, 2])) + coefficients[1, 3])
            base_z = np.float32(np.float32(np.float32(y * coefficients[2, 1])
                                 + np.float32(z * coefficients[2, 2])) + coefficients[2, 3])
            row_start = index
            last_x = np.float32(nx - 1)
            end_x = np.float32(base_x + np.float32(last_x * coefficients[0, 0]))
            end_y = np.float32(base_y + np.float32(last_x * coefficients[1, 0]))
            end_z = np.float32(base_z + np.float32(last_x * coefficients[2, 0]))
            # Each float32 affine coordinate is monotone in x. If both
            # endpoints are inside the strict cost field, every lower index
            # is nonnegative. Validity tests are redundant; the helper still clips
            # the floor to size-2 when a large-axis upper rounds to size-1.
            row_inside = (np.isfinite(end_x) and np.isfinite(end_y) and np.isfinite(end_z)
                          and min(base_x, end_x) >= 0 and max(base_x, end_x) <= upper[0]
                          and min(base_y, end_y) >= 0 and max(base_y, end_y) <= upper[1]
                          and min(base_z, end_z) >= 0 and max(base_z, end_z) <= upper[2])
            for x_index in range(nx):
                x = np.float32(x_index)
                cx = np.float32(base_x + np.float32(x * coefficients[0, 0]))
                cy = np.float32(base_y + np.float32(x * coefficients[1, 0]))
                cz = np.float32(base_z + np.float32(x * coefficients[2, 0]))
                if row_inside:
                    valid = True
                    sx, sy, sz = cx, cy, cz
                else:
                    valid = (cx >= 0 and cx <= upper[0] and cy >= 0
                             and cy <= upper[1] and cz >= 0 and cz <= upper[2])
                    sx, sy, sz = (min(max(cx, np.float32(0)), upper[0]),
                                  min(max(cy, np.float32(0)), upper[1]),
                                  min(max(cz, np.float32(0)), upper[2]))
                any_valid = any_valid or valid
                values[x_index] = (_trilinear_inside(moving, sx, sy, sz) if row_inside
                                  else _trilinear(moving, sx, sy, sz))
                weight = np.float32(1)
                if taper:
                    wx = _taper(cx, upper[0], smooth[0])
                    wy = _taper(cy, upper[1], smooth[1])
                    wz = _taper(cz, upper[2], smooth[2])
                    weight = max(np.float32(np.float32(wx * wy) * wz), np.float32(0))
                if weighted:
                    sampled_weight = (_trilinear_inside(moving_weight, sx, sy, sz)
                                      if row_inside else _trilinear(moving_weight, sx, sy, sz))
                    weight = np.float32(weight * sampled_weight)
                    weight = max(np.float32(weight * reference_weight[index]), np.float32(0))
                weights[x_index] = np.float32(weight * np.float32(valid))
                index += 1
            # Row buffers fit in cache. Each bin still receives samples in
            # exactly the original x-fastest order, including zero weights.
            for x_index in range(nx):
                target = bin_index[row_start + x_index]
                weight = weights[x_index]
                value = values[x_index]
                product = np.float32(weight * value)
                counts[target] = np.float32(counts[target] + weight)
                sums[target] = np.float32(sums[target] + product)
                sums2[target] = np.float32(sums2[target] + np.float32(product * value))
    return counts, sums, sums2, any_valid


# Independent reference rows write into disjoint, x-fastest buffers.
# Only sampling is parallel; ordered bin accumulation stays serial.
@njit(cache=True, fastmath=False, error_model='numpy', parallel=True)
def _sample_cost_parallel(moving, coefficients, reference_shape, upper, smooth, taper, moving_weight, reference_weight, weighted):
    """Sample in the reference cost's x-fastest order; return float32 arrays."""
    (nx, ny, nz) = reference_shape
    size = nx * ny * nz
    values = np.empty(size, dtype=np.float32)
    weights = np.empty(size, dtype=np.float32)
    any_valid = False
    index = 0
    valid_rows = np.zeros(ny * nz, dtype=np.bool_)
    for row_index in prange(ny * nz):
        z = np.float32(row_index // ny)
        y_index = row_index % ny
        index = row_index * nx
        any_valid = False
        y = np.float32(y_index)
        base_x = np.float32(np.float32(np.float32(y * coefficients[0, 1]) + np.float32(z * coefficients[0, 2])) + coefficients[0, 3])
        base_y = np.float32(np.float32(np.float32(y * coefficients[1, 1]) + np.float32(z * coefficients[1, 2])) + coefficients[1, 3])
        base_z = np.float32(np.float32(np.float32(y * coefficients[2, 1]) + np.float32(z * coefficients[2, 2])) + coefficients[2, 3])
        last_x = np.float32(nx - 1)
        end_x = np.float32(base_x + np.float32(last_x * coefficients[0, 0]))
        end_y = np.float32(base_y + np.float32(last_x * coefficients[1, 0]))
        end_z = np.float32(base_z + np.float32(last_x * coefficients[2, 0]))
        row_inside = np.isfinite(end_x) and np.isfinite(end_y) and np.isfinite(end_z) and (min(base_x, end_x) >= 0) and (max(base_x, end_x) <= upper[0]) and (min(base_y, end_y) >= 0) and (max(base_y, end_y) <= upper[1]) and (min(base_z, end_z) >= 0) and (max(base_z, end_z) <= upper[2])
        for x_index in range(nx):
            x = np.float32(x_index)
            cx = np.float32(base_x + np.float32(x * coefficients[0, 0]))
            cy = np.float32(base_y + np.float32(x * coefficients[1, 0]))
            cz = np.float32(base_z + np.float32(x * coefficients[2, 0]))
            if row_inside:
                valid = True
                (sx, sy, sz) = (cx, cy, cz)
            else:
                valid = cx >= 0 and cx <= upper[0] and (cy >= 0) and (cy <= upper[1]) and (cz >= 0) and (cz <= upper[2])
                (sx, sy, sz) = (min(max(cx, np.float32(0)), upper[0]), min(max(cy, np.float32(0)), upper[1]), min(max(cz, np.float32(0)), upper[2]))
            any_valid = any_valid or valid
            values[index] = _trilinear_inside(moving, sx, sy, sz) if row_inside else _trilinear(moving, sx, sy, sz)
            weight = np.float32(1)
            if taper:
                wx = _taper(cx, upper[0], smooth[0])
                wy = _taper(cy, upper[1], smooth[1])
                wz = _taper(cz, upper[2], smooth[2])
                weight = max(np.float32(np.float32(wx * wy) * wz), np.float32(0))
            if weighted:
                sampled_weight = _trilinear_inside(moving_weight, sx, sy, sz) if row_inside else _trilinear(moving_weight, sx, sy, sz)
                weight = np.float32(weight * sampled_weight)
                weight = max(np.float32(weight * reference_weight[index]), np.float32(0))
            weights[index] = np.float32(weight * np.float32(valid))
            index += 1
        valid_rows[row_index] = any_valid
    return (values, weights, valid_rows.any())


@njit(cache=True, fastmath=False)
def reduce_corratio(values, weights, bin_index, bins):
    """Accumulate each bin in its original voxel order, using float32 sums."""
    counts = np.zeros(bins, dtype=np.float32)
    sums = np.zeros(bins, dtype=np.float32)
    sums2 = np.zeros(bins, dtype=np.float32)
    for index in range(values.size):
        target = bin_index[index]
        weighted = np.float32(weights[index] * values[index])
        counts[target] = np.float32(counts[target] + weights[index])
        sums[target] = np.float32(sums[target] + weighted)
        sums2[target] = np.float32(sums2[target] + np.float32(weighted * values[index]))
    return (counts, sums, sums2)


_sample_cost_serial = sample_cost
_sample_corratio_serial = sample_corratio

def _corratio_parallel(moving, coefficients, reference_shape, upper, smooth, taper, moving_weight, reference_weight, weighted, bin_index, bins):
    (values, weights, valid) = _sample_cost_parallel(moving, coefficients, reference_shape, upper, smooth, taper, moving_weight, reference_weight, weighted)
    (counts, sums, sums2) = reduce_corratio(values, weights, bin_index, bins)
    return (counts, sums, sums2, valid)


# Respect the caller's PyTorch CPU budget, including CLI --threads=1.
# Numba's mask is local to this calling thread and restored on exceptions.
def _dispatch_with_cpu_budget(parallel_function, serial_function, arguments):
    import torch
    from numba import get_num_threads, set_num_threads
    previous = get_num_threads()
    budget = min(previous, torch.get_num_threads())
    if budget < 2:
        return serial_function(*arguments)
    changed = budget != previous
    if changed:
        set_num_threads(budget)
    try:
        return parallel_function(*arguments)
    finally:
        if changed:
            set_num_threads(previous)


def sample_cost(*arguments):
    return _dispatch_with_cpu_budget(_sample_cost_parallel, _sample_cost_serial, arguments)


def sample_corratio(*arguments):
    return _dispatch_with_cpu_budget(_corratio_parallel, _sample_corratio_serial, arguments)



# Eight independent float32 samples retain their scalar operation order.
# A nonfinite result, thin/large axis or partial block keeps scalar sampling.
# Weighted sampling applies moving and reference weights in the scalar order.
# The ordered bin loop below is unchanged.
@njit(cache=True, fastmath=False, error_model='numpy')
def _sample_corratio_simd_taper(moving, coefficients, reference_shape, upper, smooth, taper, moving_weight, reference_weight, weighted, bin_index, bins):
    """Sample one row at a time and preserve each bin's float32 sum order."""
    (nx, ny, nz) = reference_shape
    # float32(size - 1.0001) can round to size - 1 on large axes.
    # This proof is also checked here for direct internal callers.
    simd_safe = taper and min(moving.shape) >= 2 and upper[0] < moving.shape[0] - 1 and upper[1] < moving.shape[1] - 1 and upper[2] < moving.shape[2] - 1
    values = np.empty(nx, dtype=np.float32)
    weights = np.empty(nx, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.float32)
    sums = np.zeros(bins, dtype=np.float32)
    sums2 = np.zeros(bins, dtype=np.float32)
    any_valid = False
    index = 0
    for z_index in range(nz):
        z = np.float32(z_index)
        for y_index in range(ny):
            y = np.float32(y_index)
            base_x = np.float32(np.float32(np.float32(y * coefficients[0, 1]) + np.float32(z * coefficients[0, 2])) + coefficients[0, 3])
            base_y = np.float32(np.float32(np.float32(y * coefficients[1, 1]) + np.float32(z * coefficients[1, 2])) + coefficients[1, 3])
            base_z = np.float32(np.float32(np.float32(y * coefficients[2, 1]) + np.float32(z * coefficients[2, 2])) + coefficients[2, 3])
            row_start = index
            last_x = np.float32(nx - 1)
            end_x = np.float32(base_x + np.float32(last_x * coefficients[0, 0]))
            end_y = np.float32(base_y + np.float32(last_x * coefficients[1, 0]))
            end_z = np.float32(base_z + np.float32(last_x * coefficients[2, 0]))
            row_inside = np.isfinite(end_x) and np.isfinite(end_y) and np.isfinite(end_z) and (min(base_x, end_x) >= 0) and (max(base_x, end_x) <= upper[0]) and (min(base_y, end_y) >= 0) and (max(base_y, end_y) <= upper[1]) and (min(base_z, end_z) >= 0) and (max(base_z, end_z) <= upper[2])
            for block_start in range(0, nx, 8):
                block_x = np.float32(block_start)
                block_last_x = np.float32(block_start + 7)
                begin_x = np.float32(base_x + np.float32(block_x * coefficients[0, 0]))
                begin_y = np.float32(base_y + np.float32(block_x * coefficients[1, 0]))
                begin_z = np.float32(base_z + np.float32(block_x * coefficients[2, 0]))
                finish_x = np.float32(base_x + np.float32(block_last_x * coefficients[0, 0]))
                finish_y = np.float32(base_y + np.float32(block_last_x * coefficients[1, 0]))
                finish_z = np.float32(base_z + np.float32(block_last_x * coefficients[2, 0]))
                block_inside = block_start + 8 <= nx and simd_safe and np.isfinite(begin_x) and np.isfinite(begin_y) and np.isfinite(begin_z) and np.isfinite(finish_x) and np.isfinite(finish_y) and np.isfinite(finish_z)
                block_sampled = False
                if block_inside:
                    if weighted:
                        if moving_weight.shape == moving.shape:
                            (block_sampled, block_any_valid) = sample_block8_weighted(moving, values, weights, block_start, base_x, base_y, base_z, coefficients[0, 0], coefficients[1, 0], coefficients[2, 0], upper, smooth, moving_weight, reference_weight, index)
                    else:
                        (block_sampled, block_any_valid) = sample_block8(moving, values, weights, block_start, base_x, base_y, base_z, coefficients[0, 0], coefficients[1, 0], coefficients[2, 0], upper, smooth)

                if block_sampled:
                    any_valid = any_valid or block_any_valid
                    index += 8
                else:
                    for x_index in range(block_start, min(block_start + 8, nx)):
                        x = np.float32(x_index)
                        cx = np.float32(base_x + np.float32(x * coefficients[0, 0]))
                        cy = np.float32(base_y + np.float32(x * coefficients[1, 0]))
                        cz = np.float32(base_z + np.float32(x * coefficients[2, 0]))
                        if row_inside:
                            valid = True
                            (sx, sy, sz) = (cx, cy, cz)
                        else:
                            valid = cx >= 0 and cx <= upper[0] and (cy >= 0) and (cy <= upper[1]) and (cz >= 0) and (cz <= upper[2])
                            (sx, sy, sz) = (min(max(cx, np.float32(0)), upper[0]), min(max(cy, np.float32(0)), upper[1]), min(max(cz, np.float32(0)), upper[2]))
                        any_valid = any_valid or valid
                        values[x_index] = _trilinear_inside(moving, sx, sy, sz) if row_inside else _trilinear(moving, sx, sy, sz)
                        weight = np.float32(1)
                        if taper:
                            wx = _taper(cx, upper[0], smooth[0])
                            wy = _taper(cy, upper[1], smooth[1])
                            wz = _taper(cz, upper[2], smooth[2])
                            weight = max(np.float32(np.float32(wx * wy) * wz), np.float32(0))
                        if weighted:
                            sampled_weight = _trilinear_inside(moving_weight, sx, sy, sz) if row_inside else _trilinear(moving_weight, sx, sy, sz)
                            weight = np.float32(weight * sampled_weight)
                            weight = max(np.float32(weight * reference_weight[index]), np.float32(0))
                        weights[x_index] = np.float32(weight * np.float32(valid))
                        index += 1
            for x_index in range(nx):
                target = bin_index[row_start + x_index]
                weight = weights[x_index]
                value = values[x_index]
                product = np.float32(weight * value)
                counts[target] = np.float32(counts[target] + weight)
                sums[target] = np.float32(sums[target] + product)
                sums2[target] = np.float32(sums2[target] + np.float32(product * value))
    return (counts, sums, sums2, any_valid)


_production_serial_corratio = _sample_corratio_serial

def _simd_serial_corratio(*arguments):
    if arguments[5] and np.all(arguments[3] < np.asarray(arguments[0].shape, dtype=np.float64) - 1):
        return _sample_corratio_simd_taper(*arguments)
    return _production_serial_corratio(*arguments)

_sample_corratio_serial = _simd_serial_corratio
