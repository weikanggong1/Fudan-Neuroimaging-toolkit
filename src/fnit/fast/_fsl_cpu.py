"""Compiled CPU kernels for the source-ordered FAST path.

These kernels keep float32 storage and the explicitly double intermediates
used by FAST. They do not use fastmath or change CUDA execution. Spatial
updates retain z/y/x Gauss-Seidel dependencies; independent blur/evidence
voxels can use the caller's Numba thread budget.
"""

import math
from contextlib import contextmanager

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads


@contextmanager
def thread_budget(threads):
    """Limit this call's parallel mask and restore the caller on every exit."""
    previous = get_num_threads()
    target = min(int(threads), config.NUMBA_NUM_THREADS)
    try:
        set_num_threads(max(1, target))
        yield
    finally:
        set_num_threads(previous)


@njit(cache=True, fastmath=False)
def random_raw(state, position, count):
    output = np.empty(count, dtype=np.uint32)
    for index in range(count):
        value = (np.uint64(state[position])
                 + np.uint64(state[(position + 28) % 31])) & np.uint64(0xFFFFFFFF)
        state[position] = np.uint32(value)
        output[index] = np.uint32(value >> np.uint64(1))
        position = (position + 1) % 31
    return output, position


@njit(cache=True, fastmath=False)
def tanaka_zyx(probabilities, energy, included, neighbours, beta, iterations):
    """One contiguous X row at a time, with the original ordered updates."""
    nz, ny, nx = included.shape
    for _ in range(iterations):
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if not included[z, y, x]:
                        continue
                    support0 = np.float64(0)
                    support1 = np.float64(0)
                    support2 = np.float64(0)
                    for neighbour in range(neighbours.shape[0]):
                        dx = int(neighbours[neighbour, 0])
                        dy = int(neighbours[neighbour, 1])
                        dz = int(neighbours[neighbour, 2])
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                            weight = neighbours[neighbour, 3]
                            support0 += np.float64(probabilities[0, zz, yy, xx]) * weight
                            support1 += np.float64(probabilities[1, zz, yy, xx]) * weight
                            support2 += np.float64(probabilities[2, zz, yy, xx]) * weight
                    p0 = np.float32(math.exp(beta * support0 - np.float64(energy[0, z, y, x])))
                    p1 = np.float32(math.exp(beta * support1 - np.float64(energy[1, z, y, x])))
                    p2 = np.float32(math.exp(beta * support2 - np.float64(energy[2, z, y, x])))
                    total = (np.float64(p0) + np.float64(p1)) + np.float64(p2)
                    if total > 0:
                        probabilities[0, z, y, x] = np.float32(np.float64(p0) / total)
                        probabilities[1, z, y, x] = np.float32(np.float64(p1) / total)
                        probabilities[2, z, y, x] = np.float32(np.float64(p2) / total)
                    else:
                        probabilities[0, z, y, x] = np.float32(0)
                        probabilities[1, z, y, x] = np.float32(0)
                        probabilities[2, z, y, x] = np.float32(0)


@njit(cache=True, fastmath=False)
def icm_zyx(labels, probabilities, included, neighbours, pairwise, beta):
    nz, ny, nx = included.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if not included[z, y, x]:
                    continue
                best_score = np.float32(-np.inf)
                best_label = 0
                for candidate in range(6):
                    clique = np.float32(0)
                    for neighbour in range(neighbours.shape[0]):
                        dx = int(neighbours[neighbour, 0])
                        dy = int(neighbours[neighbour, 1])
                        dz = int(neighbours[neighbour, 2])
                        xx, yy, zz = x + dx, y + dy, z + dz
                        if (0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz
                                and included[zz, yy, xx]):
                            product = np.float32(pairwise[candidate, labels[zz, yy, xx]]
                                                 * np.float32(neighbours[neighbour, 3]))
                            clique = np.float32(clique + product)
                    exponent = np.float32(beta * clique)
                    score = np.float32(probabilities[candidate, z, y, x]
                                       * np.exp(exponent))
                    if score > best_score:
                        best_score = score
                        best_label = candidate
                labels[z, y, x] = best_label


@njit(cache=True, parallel=True, fastmath=False)
def blur_axis(volume, kernel, axis):
    output = np.empty_like(volume)
    nx, ny, nz = volume.shape
    radius = kernel.size // 2
    for index in prange(volume.size):
        x = index // (ny * nz)
        y = (index // nz) % ny
        z = index % nz
        value = np.float32(0)
        for tap in range(kernel.size):
            offset = tap - radius
            xx = x + offset if axis == 0 else x
            yy = y + offset if axis == 1 else y
            zz = z + offset if axis == 2 else z
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                value = np.float32(np.float64(value)
                                   + np.float64(volume[xx, yy, zz]) * kernel[tap])
        output[x, y, z] = value
    return output


@njit(cache=True, parallel=True, fastmath=False)
def mixel_evidence(values, included, means, variances, scales):
    """Three pairs, preserving each 101-step float32 accumulator."""
    output = np.zeros((3, values.size), dtype=np.float32)
    for index in prange(values.size):
        if included[index]:
            value = values[index]
            for pair in range(3):
                probability = np.float32(0)
                for fraction in range(means.shape[1]):
                    delta = np.float32(value - means[pair, fraction])
                    quadratic = np.float32(0.5 * np.float64(delta) * np.float64(delta)
                                           / np.float64(variances[pair, fraction]))
                    energy = np.float32(scales[pair, fraction] + quadratic)
                    probability = np.float32(np.float64(probability)
                                             + math.exp(-np.float64(energy)) * 0.01)
                output[pair, index] = probability
    return output


@njit(cache=True, parallel=True, fastmath=False)
def partial_volumes(values, included, mixel, fractions, means, variances, log_variances):
    output = np.zeros((3, values.size), dtype=np.float32)
    first_tissue = (0, 0, 1)
    second_tissue = (1, 2, 2)
    for index in prange(values.size):
        if not included[index]:
            continue
        code = mixel[index]
        if code < 3:
            output[code, index] = np.float32(1)
            continue
        pair = code - 3
        minimum = np.float32(1e13)
        best = np.float32(0)
        assigned = False
        for candidate in range(fractions.size):
            delta = np.float32(values[index] - means[pair, candidate])
            square = np.float32(delta * delta)
            ratio = np.float32(square / variances[pair, candidate])
            energy = np.float32(np.float32(ratio + log_variances[pair, candidate]) / np.float32(2))
            if energy < minimum:
                minimum = energy
                best = fractions[candidate]
                assigned = True
        if assigned:
            output[first_tissue[pair], index] = best
            output[second_tissue[pair], index] = np.float32(np.float32(1) - best)
    return output


def neighbours_array(scan):
    return np.asarray(scan.neighbours, dtype=np.float64)


def included_zyx(scan):
    included = np.zeros(scan.shape, dtype=np.bool_)
    included.reshape(-1)[scan.source_order.numpy()] = True
    return np.ascontiguousarray(included.transpose(2, 1, 0))


__all__ = []
