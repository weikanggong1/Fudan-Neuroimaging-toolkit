"""GPU probabilistic FOD tractography on a registered tissue grid."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .fod import real_sh


@dataclass
class Tractogram:
    """Accepted streamlines; coordinates and lengths use world millimetres."""

    paths: tuple[torch.Tensor, ...]
    endpoints: torch.Tensor
    lengths_mm: torch.Tensor
    mean_fa: torch.Tensor | None
    seeds_attempted: int


def _sample(volume: torch.Tensor, points: torch.Tensor, inverse_affine: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample [X,Y,Z,C] at world-mm [N,3] points."""
    shape = volume.shape[:3]
    voxel = points @ inverse_affine[:3, :3].T + inverse_affine[:3, 3]
    scale = points.new_tensor([max(size - 1, 1) for size in shape])
    grid = (2 * voxel / scale - 1).reshape(1, 1, 1, -1, 3)
    image = volume.permute(3, 2, 1, 0)[None]
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True).reshape(volume.shape[-1], -1).T


def _label(tissues: torch.Tensor, points: torch.Tensor, inverse_affine: torch.Tensor):
    voxel = (points @ inverse_affine[:3, :3].T + inverse_affine[:3, 3]).round().long()
    inside = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    for axis, size in enumerate(tissues.shape):
        inside &= (voxel[:, axis] >= 0) & (voxel[:, axis] < size)
    safe = torch.stack([voxel[:, axis].clamp(0, size - 1)
                        for axis, size in enumerate(tissues.shape)], dim=-1)
    labels = tissues[safe[:, 0], safe[:, 1], safe[:, 2]]
    return torch.where(inside, labels, torch.zeros_like(labels))


def _sphere(count: int, device: torch.device):
    index = torch.arange(count, device=device, dtype=torch.float32)
    z = 1 - 2 * (index + 0.5) / count
    angle = index * (math.pi * (3 - math.sqrt(5)))
    radial = (1 - z.square()).sqrt()
    return torch.stack((radial * angle.cos(), radial * angle.sin(), z), dim=-1)


def _grow(
    seeds: torch.Tensor,
    tangents: torch.Tensor,
    fod: torch.Tensor,
    tissues: torch.Tensor,
    inverse_affine: torch.Tensor,
    directions: torch.Tensor,
    basis: torch.Tensor,
    generator: torch.Generator,
    *,
    step_mm: float,
    max_steps: int,
    max_angle_degrees: float,
    cutoff: float,
    power: float,
):
    batch = seeds.shape[0]
    paths = seeds.new_zeros((batch, max_steps + 1, 3))
    paths[:, 0] = seeds
    counts = torch.ones(batch, dtype=torch.long, device=seeds.device)
    active = torch.ones(batch, dtype=torch.bool, device=seeds.device)
    ended_in_gm = torch.zeros_like(active)
    positions = seeds.clone()
    prior = tangents.clone()
    cosine_limit = math.cos(math.radians(max_angle_degrees))
    for step in range(max_steps):
        if not bool(active.any()):
            break
        coefficient = _sample(fod, positions, inverse_affine)
        amplitudes = (coefficient @ basis.T).clamp_min(0)
        permitted = ((prior @ directions.T) >= cosine_limit) & (amplitudes >= cutoff)
        scores = amplitudes.pow(power) * permitted
        moving = active & (scores.sum(-1) > 0)
        # Multinomial also receives inactive rows, so give them one legal item.
        scores = torch.where(moving[:, None], scores, torch.ones_like(scores))
        choice = torch.multinomial(scores, 1, generator=generator).squeeze(-1)
        direction = directions[choice]
        proposed = positions + step_mm * direction
        label = _label(tissues, proposed, inverse_affine)
        valid = moving & ((label == 1) | (label == 2))
        paths[valid, step + 1] = proposed[valid]
        counts[valid] += 1
        entered_gm = valid & (label == 1)
        ended_in_gm |= entered_gm
        active = valid & ~entered_gm
        positions = torch.where(valid[:, None], proposed, positions)
        prior = torch.where(valid[:, None], direction, prior)
    return paths, counts, ended_in_gm


@torch.inference_mode()
def probabilistic_tractography(
    wm_sh: torch.Tensor,
    affine: torch.Tensor,
    tissues: torch.Tensor,
    *,
    n_seeds: int,
    lmax: int = 4,
    fa: torch.Tensor | None = None,
    seed: int = 0,
    batch_size: int = 1024,
    sphere_samples: int = 128,
    max_length_mm: float = 250.,
    min_length_mm: float | None = None,
    step_mm: float | None = None,
    max_angle_degrees: float = 45.,
    cutoff: float = 0.1,
    power: float = 0.5,
) -> Tractogram:
    """Track both ways from GM/WM boundary seeds using the WM FOD.

    ``tissues`` is an integer DWI-grid volume: 0 outside/CSF, 1 GM,
    2 WM. This is a GPU FOD sampler with ACT-like tissue stopping; its
    sampling law differs from MRtrix iFOD2 and its outputs need paired
    empirical comparison before use as a reference replacement.
    """
    if wm_sh.ndim != 4 or tissues.shape != wm_sh.shape[:3] or affine.shape != (4, 4):
        raise ValueError('expected WM SH [X,Y,Z,C], tissues [X,Y,Z], affine [4,4]')
    if fa is not None and fa.shape != tissues.shape:
        raise ValueError('FA must match the DWI tissue grid')
    if n_seeds < 1 or batch_size < 1 or sphere_samples < 16:
        raise ValueError('n_seeds, batch_size and sphere_samples must be positive')
    device = wm_sh.device
    affine = affine.to(device=device, dtype=torch.float32)
    inverse = torch.linalg.inv(affine)
    wm_sh = wm_sh.to(torch.float32)
    tissues = tissues.to(device=device)
    voxel_mm = torch.linalg.vector_norm(affine[:3, :3], dim=0)
    step_mm = float(voxel_mm.min()) / 2 if step_mm is None else float(step_mm)
    if step_mm <= 0 or max_length_mm <= step_mm:
        raise ValueError('step and maximum length must be positive')
    # Either half may contain almost the full permitted streamline length.
    max_steps = math.ceil(max_length_mm / step_mm)
    min_length_mm = 2 * float(voxel_mm.min()) if min_length_mm is None else min_length_mm
    if min_length_mm < 0 or min_length_mm > max_length_mm:
        raise ValueError('minimum length must lie between zero and maximum length')
    # The original UKB command sets -cutoff 0.1 explicitly; ACT only halves
    # MRtrix's implicit default threshold, not this explicit value.
    wm = (tissues == 2).to(torch.float32)[None, None]
    gm = (tissues == 1).to(torch.float32)[None, None]
    adjacent_gm = F.max_pool3d(gm, 3, stride=1, padding=1)[0, 0]
    seed_voxels = torch.nonzero((wm[0, 0] > 0) & (adjacent_gm > 0), as_tuple=False)
    if seed_voxels.numel() == 0:
        raise ValueError('no white-matter voxels touch grey matter')
    directions = _sphere(sphere_samples, device)
    basis = real_sh(directions, lmax)
    if wm_sh.shape[-1] != basis.shape[-1]:
        raise ValueError('WM SH coefficient count does not match lmax')
    generator = torch.Generator(device=device).manual_seed(seed)
    collected = []
    endpoints = []
    lengths = []
    fa_means = []
    for first in range(0, n_seeds, batch_size):
        size = min(batch_size, n_seeds - first)
        choices = torch.randint(len(seed_voxels), (size,), device=device, generator=generator)
        voxels = seed_voxels[choices].to(torch.float32)
        voxels += (torch.rand((size, 3), device=device, generator=generator) - 0.5) * 0.4
        seeds = voxels @ affine[:3, :3].T + affine[:3, 3]
        amplitude = (_sample(wm_sh, seeds, inverse) @ basis.T).clamp_min(0).pow(power)
        amplitude = torch.where(amplitude.sum(-1, keepdim=True) > 0,
                                amplitude, torch.ones_like(amplitude))
        initial = directions[torch.multinomial(amplitude, 1, generator=generator).squeeze(-1)]
        forward, nf, gf = _grow(
            seeds, initial, wm_sh, tissues, inverse, directions, basis, generator,
            step_mm=step_mm, max_steps=max_steps,
            max_angle_degrees=max_angle_degrees, cutoff=cutoff, power=power,
        )
        backward, nb, gb = _grow(
            seeds, -initial, wm_sh, tissues, inverse, directions, basis, generator,
            step_mm=step_mm, max_steps=max_steps,
            max_angle_degrees=max_angle_degrees, cutoff=cutoff, power=power,
        )
        total = (nf + nb - 2) * step_mm
        keep = torch.nonzero(gf & gb & (total >= min_length_mm) &
                             (total <= max_length_mm), as_tuple=False).flatten()
        for index in keep.tolist():
            path = torch.cat((backward[index, :nb[index]].flip(0),
                              forward[index, 1:nf[index]]), dim=0)
            collected.append(path)
            endpoints.append(torch.stack((path[0], path[-1])))
            lengths.append(total[index])
            if fa is not None:
                fa_means.append(_sample(fa.to(device=device, dtype=torch.float32)[..., None],
                                        path, inverse).mean())
    empty_endpoints = wm_sh.new_empty((0, 2, 3))
    return Tractogram(
        paths=tuple(collected),
        endpoints=torch.stack(endpoints) if endpoints else empty_endpoints,
        lengths_mm=torch.stack(lengths) if lengths else wm_sh.new_empty(0),
        mean_fa=torch.stack(fa_means) if fa_means else (wm_sh.new_empty(0) if fa is not None else None),
        seeds_attempted=n_seeds,
    )
