"""Ordered FAST updates implemented with independent CPU/GPU tensor kernels.

The original z/y/x sweep has only 18 effective neighbours. Ordering updates by
``x + 2*y + 3*z`` preserves every directed neighbour dependency, while allowing
independent voxels at the same level to run together. CUDA uses one Triton
kernel per level; CPU uses compiled source-ordered scans on contiguous rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # CPU-only installations do not need the CUDA compiler.
    triton = tl = libdevice = None


class GlibcRandom:
    """Local glibc additive-generator stream, without changing process RNGs."""

    def __init__(self, seed=-1, *, compiled=False):
        seed = (int(seed) & 0xFFFFFFFF) or 1
        state = [seed]
        previous = seed if seed < 0x80000000 else seed - 0x100000000
        for _ in range(1, 31):
            previous = (16807 * previous) % 2147483647
            state.append(previous)
        state.extend(state[:3])
        for index in range(34, 344):
            state.append((state[index - 31] + state[index - 3]) & 0xFFFFFFFF)
        self._state = state[-31:]
        self._position = 0
        self._compiled = compiled
        if compiled:
            self._state = np.asarray(self._state, dtype=np.uint32)

    def raw(self, count):
        if self._compiled:
            from ._fsl_cpu import random_raw

            output, self._position = random_raw(self._state, self._position, int(count))
            return output
        output = np.empty(int(count), dtype=np.uint32)
        state, position = self._state, self._position
        for index in range(len(output)):
            value = (state[position] + state[(position + 28) % 31]) & 0xFFFFFFFF
            state[position] = value
            output[index] = value >> 1
            position = (position + 1) % 31
        self._position = position
        return output


@dataclass
class ScanSchedule:
    shape: tuple[int, int, int]
    indices: torch.Tensor
    source_order: torch.Tensor
    spans: tuple[tuple[int, int], ...]
    neighbours: tuple[tuple[int, int, int, float], ...]
    weights: torch.Tensor


def schedule(mask, voxel_size):
    shape = tuple(mask.shape)
    zyx = np.argwhere(mask.detach().cpu().numpy().transpose(2, 1, 0))
    source_order = (zyx[:, 2] * shape[1] + zyx[:, 1]) * shape[2] + zyx[:, 0]
    level = zyx[:, 2] + 2 * zyx[:, 1] + 3 * zyx[:, 0]
    sorted_order = np.argsort(level, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, np.diff(level[sorted_order]) != 0, True])
    spans = tuple((int(start), int(stop - start)) for start, stop in zip(boundaries[:-1], boundaries[1:]))
    neighbours = []
    for z in (-1, 0, 1):
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                if sum(value != 0 for value in (x, y, z)) not in (1, 2):
                    continue
                distance = math.sqrt(sum((offset * spacing) ** 2 for offset, spacing in zip((x, y, z), voxel_size)))
                neighbours.append((x, y, z, float(np.float32(1 / distance))))
    device = mask.device
    return ScanSchedule(shape,
                        torch.as_tensor(source_order[sorted_order].copy(), device=device),
                        torch.as_tensor(source_order.copy(), device=device), spans,
                        tuple(neighbours),
                        torch.tensor([entry[3] for entry in neighbours], dtype=torch.float64, device=device))


def random_posteriors(scan, random):
    count = scan.source_order.numel()
    values = random.raw(3 * count).astype(np.float32).reshape(count, 3)
    values /= np.float32(2147483647)
    total = (values[:, 0] + values[:, 1]) + values[:, 2]
    np.divide(values, total[:, None], out=values, where=total[:, None] > 0)
    result = torch.zeros((3, int(np.prod(scan.shape))), dtype=torch.float32, device=scan.indices.device)
    result[:, scan.source_order] = torch.as_tensor(values.T.copy(), device=scan.indices.device)
    return result.reshape(3, *scan.shape)


def _neighbour_indices(ids, shape, neighbour):
    x = ids // (shape[1] * shape[2])
    y = (ids // shape[2]) % shape[1]
    z = ids % shape[2]
    x, y, z = x + neighbour[0], y + neighbour[1], z + neighbour[2]
    valid = (x >= 0) & (x < shape[0]) & (y >= 0) & (y < shape[1]) & (z >= 0) & (z < shape[2])
    neighbour_ids = (x.clamp(0, shape[0] - 1) * shape[1] + y.clamp(0, shape[1] - 1)) * shape[2] + z.clamp(0, shape[2] - 1)
    return neighbour_ids, valid


if triton is not None:
    @triton.jit
    def _tanaka_kernel(P, Energy, Ids, Weights, START, COUNT,
                       X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
                       BETA: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = lane < COUNT
        ids = tl.load(Ids + START + lane, active, 0)
        x, y, z = ids // (Y * Z), (ids // Z) % Y, ids % Z
        support0 = tl.full((BLOCK,), 0, tl.float64)
        support1 = tl.full((BLOCK,), 0, tl.float64)
        support2 = tl.full((BLOCK,), 0, tl.float64)
        neighbour = 0
        for dz in tl.static_range(-1, 2):
            for dy in tl.static_range(-1, 2):
                for dx in tl.static_range(-1, 2):
                    if (dx != 0) + (dy != 0) + (dz != 0) == 1 or (dx != 0) + (dy != 0) + (dz != 0) == 2:
                        valid = active & (x + dx >= 0) & (x + dx < X) & (y + dy >= 0) & (y + dy < Y) & (z + dz >= 0) & (z + dz < Z)
                        pos = ((x + dx) * Y + y + dy) * Z + z + dz
                        weight = tl.load(Weights + neighbour)
                        support0 += weight * tl.load(P + pos, valid, 0).to(tl.float64)
                        support1 += weight * tl.load(P + X * Y * Z + pos, valid, 0).to(tl.float64)
                        support2 += weight * tl.load(P + 2 * X * Y * Z + pos, valid, 0).to(tl.float64)
                        neighbour += 1
        e0 = tl.load(Energy + ids, active, 0).to(tl.float64)
        e1 = tl.load(Energy + X * Y * Z + ids, active, 0).to(tl.float64)
        e2 = tl.load(Energy + 2 * X * Y * Z + ids, active, 0).to(tl.float64)
        p0 = libdevice.exp(BETA * support0 - e0).to(tl.float32)
        p1 = libdevice.exp(BETA * support1 - e1).to(tl.float32)
        p2 = libdevice.exp(BETA * support2 - e2).to(tl.float32)
        total = (p0.to(tl.float64) + p1.to(tl.float64)) + p2.to(tl.float64)
        tl.store(P + ids, tl.where(total > 0, p0.to(tl.float64) / total, 0).to(tl.float32), active)
        tl.store(P + X * Y * Z + ids, tl.where(total > 0, p1.to(tl.float64) / total, 0).to(tl.float32), active)
        tl.store(P + 2 * X * Y * Z + ids, tl.where(total > 0, p2.to(tl.float64) / total, 0).to(tl.float32), active)

    @triton.jit
    def _compatibility(label, candidate: tl.constexpr):
        if candidate == 0:
            shared = (label == 3) | (label == 4)
        elif candidate == 1:
            shared = (label == 3) | (label == 5)
        elif candidate == 2:
            shared = (label == 4) | (label == 5)
        elif candidate == 3:
            shared = (label == 0) | (label == 1)
        elif candidate == 4:
            shared = (label == 0) | (label == 2)
        else:
            shared = (label == 1) | (label == 2)
        return tl.where(label == candidate, 2., tl.where(shared, 1., -1.))

    @triton.jit
    def _icm_kernel(Labels, Probability, Mask, Ids, Weights, START, COUNT,
                    X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
                    BETA: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = lane < COUNT
        ids = tl.load(Ids + START + lane, active, 0)
        x, y, z = ids // (Y * Z), (ids // Z) % Y, ids % Z
        clique0 = tl.full((BLOCK,), 0, tl.float32)
        clique1 = tl.full((BLOCK,), 0, tl.float32)
        clique2 = tl.full((BLOCK,), 0, tl.float32)
        clique3 = tl.full((BLOCK,), 0, tl.float32)
        clique4 = tl.full((BLOCK,), 0, tl.float32)
        clique5 = tl.full((BLOCK,), 0, tl.float32)
        neighbour = 0
        for dz in tl.static_range(-1, 2):
            for dy in tl.static_range(-1, 2):
                for dx in tl.static_range(-1, 2):
                    if (dx != 0) + (dy != 0) + (dz != 0) == 1 or (dx != 0) + (dy != 0) + (dz != 0) == 2:
                        valid = active & (x + dx >= 0) & (x + dx < X) & (y + dy >= 0) & (y + dy < Y) & (z + dz >= 0) & (z + dz < Z)
                        pos = ((x + dx) * Y + y + dy) * Z + z + dz
                        included = tl.load(Mask + pos, valid, 0) != 0
                        label = tl.load(Labels + pos, valid & included, 0)
                        weight = tl.where(included, tl.load(Weights + neighbour).to(tl.float32), 0.)
                        clique0 += weight * _compatibility(label, 0)
                        clique1 += weight * _compatibility(label, 1)
                        clique2 += weight * _compatibility(label, 2)
                        clique3 += weight * _compatibility(label, 3)
                        clique4 += weight * _compatibility(label, 4)
                        clique5 += weight * _compatibility(label, 5)
                        neighbour += 1
        best = tl.full((BLOCK,), -1, tl.float32)
        selected = tl.full((BLOCK,), 0, tl.int32)
        for candidate in tl.static_range(6):
            if candidate == 0:
                clique = clique0
            elif candidate == 1:
                clique = clique1
            elif candidate == 2:
                clique = clique2
            elif candidate == 3:
                clique = clique3
            elif candidate == 4:
                clique = clique4
            else:
                clique = clique5
            exponent = (BETA * clique).to(tl.float32)
            value = tl.load(Probability + candidate * X * Y * Z + ids, active, 0) * libdevice.exp(exponent)
            improve = value > best
            selected = tl.where(improve, candidate, selected)
            best = tl.where(improve, value, best)
        tl.store(Labels + ids, selected, active)

    @triton.jit
    def _blur_kernel(Input, Output, Kernel, SIZE: tl.constexpr,
                     X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
                     AXIS: tl.constexpr, RADIUS: tl.constexpr, BLOCK: tl.constexpr):
        ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = ids < SIZE
        if AXIS == 0:
            coordinate, stride, extent = ids // (Y * Z), Y * Z, X
        elif AXIS == 1:
            coordinate, stride, extent = (ids // Z) % Y, Z, Y
        else:
            coordinate, stride, extent = ids % Z, 1, Z
        value = tl.full((BLOCK,), 0, tl.float32)
        for tap in tl.static_range(2 * RADIUS + 1):
            offset = tap - RADIUS
            sample = tl.load(Input + ids + offset * stride,
                             active & (coordinate + offset >= 0) & (coordinate + offset < extent), 0)
            weight = tl.load(Kernel + tap)
            value = (value.to(tl.float64) + sample.to(tl.float64) * weight).to(tl.float32)
        tl.store(Output + ids, value, active)


def tanaka(probabilities, energy, scan, beta, iterations=5):
    probabilities = probabilities.contiguous()
    energy = energy.contiguous()
    beta = float(np.float32(beta))
    if probabilities.is_cuda:
        if triton is None:
            raise RuntimeError("FAST execution='fsl' on CUDA requires the project's Triton dependency")
        with torch.cuda.device(probabilities.device):
            for _ in range(iterations):
                for start, count in scan.spans:
                    _tanaka_kernel[(triton.cdiv(count, 128),)](
                        probabilities, energy, scan.indices, scan.weights, start, count,
                        *scan.shape, beta, 128, enable_fp_fusion=False,
                    )
    else:
        from ._fsl_cpu import included_zyx, neighbours_array, tanaka_zyx

        ordered = np.ascontiguousarray(probabilities.numpy().transpose(0, 3, 2, 1))
        ordered_energy = np.ascontiguousarray(energy.numpy().transpose(0, 3, 2, 1))
        tanaka_zyx(ordered, ordered_energy, included_zyx(scan), neighbours_array(scan), beta, iterations)
        probabilities.copy_(torch.from_numpy(ordered.transpose(0, 3, 2, 1)))
    return probabilities


def compatibility(device="cpu"):
    constituents = ({0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2})
    result = torch.full((6, 6), -1., device=device)
    for candidate in range(6):
        for neighbour in range(6):
            if candidate == neighbour:
                result[candidate, neighbour] = 2
            elif (len(constituents[candidate]) == 1) != (len(constituents[neighbour]) == 1) and constituents[candidate] & constituents[neighbour]:
                result[candidate, neighbour] = 1
    return result


def icm(probabilities, mask, scan, beta):
    labels = probabilities.argmax(dim=0).to(torch.int32).contiguous()
    labels[~mask] = 0
    beta = float(np.float32(beta))
    if labels.is_cuda:
        if triton is None:
            raise RuntimeError("FAST execution='fsl' on CUDA requires the project's Triton dependency")
        with torch.cuda.device(labels.device):
            for start, count in scan.spans:
                _icm_kernel[(triton.cdiv(count, 128),)](
                    labels, probabilities.contiguous(), mask.contiguous(), scan.indices, scan.weights, start, count,
                    *scan.shape, beta, 128, enable_fp_fusion=False,
                )
    else:
        from ._fsl_cpu import icm_zyx, neighbours_array

        ordered = np.ascontiguousarray(labels.numpy().transpose(2, 1, 0))
        ordered_probability = np.ascontiguousarray(probabilities.numpy().transpose(0, 3, 2, 1))
        included = np.ascontiguousarray(mask.numpy().transpose(2, 1, 0))
        icm_zyx(ordered, ordered_probability, included, neighbours_array(scan),
                compatibility().numpy(), np.float32(beta))
        labels = torch.from_numpy(ordered.transpose(2, 1, 0).copy())
    return labels.long()


def blur(volume, kernels):
    result = volume.contiguous()
    if not result.is_cuda:
        from ._fsl_cpu import blur_axis

        array = result.numpy()
        for axis, kernel in enumerate(kernels):
            array = blur_axis(array, kernel.numpy(), axis)
        return torch.from_numpy(array)
    for axis, kernel in enumerate(kernels):
        if result.is_cuda:
            if triton is None:
                raise RuntimeError("FAST execution='fsl' on CUDA requires the project's Triton dependency")
            output = torch.empty_like(result)
            with torch.cuda.device(result.device):
                _blur_kernel[(triton.cdiv(result.numel(), 256),)](
                    result, output, kernel.contiguous(), result.numel(), *result.shape,
                    axis, kernel.numel() // 2, 256, enable_fp_fusion=False,
                )
        result = output
    return result
