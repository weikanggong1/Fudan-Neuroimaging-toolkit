"""Single-thread float32 adaptive non-local means for the fixed recon-all profile.

Adapted from ITK/ANTs itkAdaptiveNonLocalMeansDenoisingImageFilter.hxx
(Copyright Insight Software Consortium; Apache License 2.0) as bundled
with FreeSurfer source commit d932c45. This is a modified Numba translation.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _local_stats(image):
    sx, sy, sz = image.shape
    mean = np.empty(image.shape, dtype=np.float32)
    variance = np.empty(image.shape, dtype=np.float32)
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                total = np.float32(0)
                squares = np.float32(0)
                for dz in range(-1, 2):
                    qz = min(max(z + dz, 0), sz - 1)
                    for dy in range(-1, 2):
                        qy = min(max(y + dy, 0), sy - 1)
                        for dx in range(-1, 2):
                            qx = min(max(x + dx, 0), sx - 1)
                            v = np.float32(image[qx, qy, qz])
                            total += v
                            squares += v * v
                mean[x, y, z] = np.float32(np.float64(total) / 27.0)
                variance[x, y, z] = np.float32((np.float64(squares) - np.float64(total * total) / 27.0) / 26.0)
    return mean, variance


@njit(cache=True)
def _similar(mean0, var0, mean1, var1, maximum):
    if mean1 <= np.float32(1e-5) or var1 <= np.float32(1e-5):
        return False
    mean_ratio = np.float32(mean0 / mean1)
    if maximum == mean1:
        mean_inverse = np.float32(np.inf)
    else:
        mean_inverse = np.float32(np.float32(maximum - mean0) / np.float32(maximum - mean1))
    variance_ratio = np.float32(var0 / var1)
    return (((mean_ratio > np.float32(0.95) and mean_ratio < np.float32(1.0 / np.float32(0.95))) or
             (mean_inverse > np.float32(0.95) and mean_inverse < np.float32(1.0 / np.float32(0.95)))) and
            variance_ratio > np.float32(0.5) and variance_ratio < np.float32(2.0))


@njit(cache=True)
def _denoise(image, mean, variance):
    sx, sy, sz = image.shape
    output = np.zeros(image.shape, dtype=np.float32)
    count_image = np.zeros(image.shape, dtype=np.float32)
    maximum = np.float32(np.max(image))
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                center = image[x, y, z]
                mc = mean[x, y, z]
                vc = variance[x, y, z]
                weights = np.zeros(27, dtype=np.float32)
                max_weight = np.float32(0)
                sum_weights = np.float32(0)
                if center > 0 and mc > np.float32(1e-5) and vc > np.float32(1e-5):
                    minimum = np.float32(np.finfo(np.float32).max)
                    for dz in range(-2, 3):
                        qz = z + dz
                        if qz < 0 or qz >= sz:
                            continue
                        for dy in range(-2, 3):
                            qy = y + dy
                            if qy < 0 or qy >= sy:
                                continue
                            for dx in range(-2, 3):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                qx = x + dx
                                if qx < 0 or qx >= sx or image[qx, qy, qz] == 0:
                                    continue
                                if not _similar(mc, vc, mean[qx, qy, qz], variance[qx, qy, qz], maximum):
                                    continue
                                d = np.float32(0)
                                n = np.float32(0)
                                for pz in range(-1, 2):
                                    rz = qz + pz
                                    if rz < 0 or rz >= sz:
                                        continue
                                    for py in range(-1, 2):
                                        ry = qy + py
                                        if ry < 0 or ry >= sy:
                                            continue
                                        for px in range(-1, 2):
                                            rx = qx + px
                                            if rx < 0 or rx >= sx:
                                                continue
                                            residual = np.float32(np.float32(image[rx, ry, rz]) - mean[rx, ry, rz])
                                            d += np.float32(residual * residual)
                                            n += np.float32(1)
                                d = np.float32(d / n)
                                minimum = min(minimum, d)
                    if minimum == np.float32(0):
                        minimum = np.float32(1)
                    for dz in range(-2, 3):
                        qz = z + dz
                        if qz < 0 or qz >= sz:
                            continue
                        for dy in range(-2, 3):
                            qy = y + dy
                            if qy < 0 or qy >= sy:
                                continue
                            for dx in range(-2, 3):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                qx = x + dx
                                if qx < 0 or qx >= sx or image[qx, qy, qz] == 0:
                                    continue
                                if not _similar(mc, vc, mean[qx, qy, qz], variance[qx, qy, qz], maximum):
                                    continue
                                d = np.float32(0)
                                n = np.float32(0)
                                for pz in range(-1, 2):
                                    rz = qz + pz
                                    cz = z + pz
                                    if rz < 0 or rz >= sz or cz < 0 or cz >= sz:
                                        continue
                                    for py in range(-1, 2):
                                        ry = qy + py
                                        cy = y + py
                                        if ry < 0 or ry >= sy or cy < 0 or cy >= sy:
                                            continue
                                        for px in range(-1, 2):
                                            rx = qx + px
                                            cx = x + px
                                            if rx < 0 or rx >= sx or cx < 0 or cx >= sx:
                                                continue
                                            a = np.float32(np.float32(image[rx, ry, rz]) - mean[rx, ry, rz])
                                            b = np.float32(np.float32(image[cx, cy, cz]) - mean[cx, cy, cz])
                                            diff = np.float32(a - b)
                                            d += np.float32(diff * diff)
                                            n += np.float32(1)
                                d = np.float32(d / n)
                                weight = np.float32(0)
                                if d <= np.float32(np.float32(3) * minimum):
                                    weight = np.float32(np.exp(np.float32(-d / minimum)))
                                if weight > max_weight:
                                    max_weight = weight
                                if weight > 0:
                                    for pz in range(-1, 2):
                                        rz = qz + pz
                                        if rz < 0 or rz >= sz:
                                            continue
                                        for py in range(-1, 2):
                                            ry = qy + py
                                            if ry < 0 or ry >= sy:
                                                continue
                                            for px in range(-1, 2):
                                                rx = qx + px
                                                if rx < 0 or rx >= sx:
                                                    continue
                                                pi = (pz + 1) * 9 + (py + 1) * 3 + (px + 1)
                                                weights[pi] += np.float32(weight * np.float32(image[rx, ry, rz]))
                                    sum_weights += weight
                    if max_weight == np.float32(0):
                        max_weight = np.float32(1)
                else:
                    max_weight = np.float32(1)
                for pz in range(-1, 2):
                    cz = z + pz
                    if cz < 0 or cz >= sz:
                        continue
                    for py in range(-1, 2):
                        cy = y + py
                        if cy < 0 or cy >= sy:
                            continue
                        for px in range(-1, 2):
                            cx = x + px
                            if cx < 0 or cx >= sx:
                                continue
                            pi = (pz + 1) * 9 + (py + 1) * 3 + (px + 1)
                            weights[pi] += np.float32(max_weight * np.float32(image[cx, cy, cz]))
                sum_weights += max_weight
                for pz in range(-1, 2):
                    cz = z + pz
                    if cz < 0 or cz >= sz:
                        continue
                    for py in range(-1, 2):
                        cy = y + py
                        if cy < 0 or cy >= sy:
                            continue
                        for px in range(-1, 2):
                            cx = x + px
                            if cx < 0 or cx >= sx:
                                continue
                            pi = (pz + 1) * 9 + (py + 1) * 3 + (px + 1)
                            output[cx, cy, cz] += np.float32(weights[pi] / sum_weights)
                            count_image[cx, cy, cz] += np.float32(1)
    return output / count_image


def denoise_array(image: np.ndarray) -> np.ndarray:
    source = np.asfortranarray(image, dtype=np.uint8)
    mean, variance = _local_stats(source)
    return _denoise(source, mean, variance)
