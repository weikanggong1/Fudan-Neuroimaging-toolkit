"""Orientation-aware, SIFT2-inspired streamline weighting in PyTorch.

This is an approximation, not MRtrix ``tcksift2``. MRtrix segments FOD lobes
into fixels, uses an ACT processing mask, and optimizes exponential
streamline coefficients with along-track regularization.
Here, fixed hemisphere direction bins stand in for fixels, and nonnegative
least squares matches streamline length density to the WM FOD in visited bins.

Streamline points and the affine use world millimetres. WM FOD coefficients are
even real SH (the convention of :func:`fnit.connectome.fod.real_sh`) in units
of tissue fraction per steradian. The returned weights are nonnegative,
dimensionless per-streamline factors. A global proportionality coefficient
``mu`` (FOD bin fraction per unit normalized streamline length) is estimated
internally so weights remain around one; it is not an MRtrix-compatible mu.
"""

from __future__ import annotations

import math

import torch

from .fod import real_sh


def _hemisphere_directions(count: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(count, device=device, dtype=torch.float32)
    z = (index + 0.5) / count
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radial = torch.sqrt(1.0 - z.square())
    return torch.stack((radial * angle.cos(), radial * angle.sin(), z), dim=-1)


@torch.inference_mode()
def estimate_sift2_weights(
    paths: tuple[torch.Tensor, ...],
    wm_sh: torch.Tensor,
    affine: torch.Tensor,
    lmax: int = 4,
) -> torch.Tensor:
    """Estimate approximate SIFT2 weights for world-mm streamlines.

    ``wm_sh`` has shape ``[X,Y,Z,C]`` and float32 dtype. Every path has shape
    ``[P,3]`` and uses the same device. Returned weights have shape
    ``[len(paths)]`` in input order; paths with no in-grid segment receive 0.
    A segment is subdivided to at most half the smallest voxel edge before
    midpoint assignment to one voxel and the nearest antipodal direction bin.
    """
    if wm_sh.ndim != 4 or wm_sh.dtype != torch.float32 or affine.shape != (4, 4):
        raise ValueError("expected float32 WM SH [X,Y,Z,C] and affine [4,4]")
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a nonnegative even integer")
    device = wm_sh.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    n_tracks = len(paths)
    weights = torch.zeros(n_tracks, device=device, dtype=torch.float32)
    if not n_tracks:
        return weights
    affine = affine.to(device=device, dtype=torch.float32)
    inverse = torch.linalg.inv(affine)
    voxel_width = torch.linalg.vector_norm(affine[:3, :3], dim=0).min()
    if not bool(torch.isfinite(voxel_width)) or float(voxel_width) <= 0:
        raise ValueError("affine must have positive finite voxel edge lengths")
    bin_directions = _hemisphere_directions(32, device)
    bin_basis = real_sh(bin_directions, lmax)
    if wm_sh.shape[-1] != bin_basis.shape[-1]:
        raise ValueError("WM SH coefficient count does not match lmax")
    shape = wm_sh.shape[:3]
    n_bins = len(bin_directions)
    rows, tracks, lengths = [], [], []
    for track_index, path in enumerate(paths):
        if path.ndim != 2 or path.shape[-1] != 3 or path.dtype != torch.float32 or path.device != device:
            raise ValueError("each path must be float32 [P,3] on the WM FOD device")
        if len(path) < 2:
            continue
        start = path[:-1]
        delta = path[1:] - start
        segment_length = torch.linalg.vector_norm(delta, dim=1)
        nonzero = torch.isfinite(segment_length) & (segment_length > 0)
        start, delta, segment_length = start[nonzero], delta[nonzero], segment_length[nonzero]
        if not len(segment_length):
            continue
        repeats = torch.ceil(segment_length / (0.5 * voxel_width)).long().clamp_min(1)
        segment = torch.repeat_interleave(torch.arange(len(repeats), device=device), repeats)
        first = torch.repeat_interleave(torch.cumsum(repeats, 0) - repeats, repeats)
        fraction = (torch.arange(len(segment), device=device) - first + 0.5) / repeats[segment]
        midpoint = start[segment] + fraction[:, None] * delta[segment]
        voxel = (midpoint @ inverse[:3, :3].T + inverse[:3, 3]).round().long()
        inside = torch.ones(len(segment), device=device, dtype=torch.bool)
        for axis, size in enumerate(shape):
            inside &= (voxel[:, axis] >= 0) & (voxel[:, axis] < size)
        if not bool(inside.any()):
            continue
        voxel, segment = voxel[inside], segment[inside]
        tangent = delta[segment] / segment_length[segment, None]
        angular_bin = torch.abs(tangent @ bin_directions.T).argmax(dim=1)
        voxel_index = (voxel[:, 0] * shape[1] + voxel[:, 1]) * shape[2] + voxel[:, 2]
        rows.append(voxel_index * n_bins + angular_bin)
        tracks.append(torch.full((len(segment),), track_index, device=device, dtype=torch.long))
        lengths.append(segment_length[segment] / repeats[segment] / voxel_width)
    if not rows:
        return weights
    row = torch.cat(rows)
    track = torch.cat(tracks)
    contribution = torch.cat(lengths)
    unique_row, local_row = torch.unique(row, sorted=True, return_inverse=True)
    n_rows = len(unique_row)
    target = (
        wm_sh.reshape(-1, wm_sh.shape[-1])[unique_row // n_bins]
        * bin_basis[unique_row % n_bins]
    ).sum(dim=1).clamp_min(0) * (4.0 * math.pi / n_bins)
    row_density = torch.zeros(n_rows, device=device).index_add_(0, local_row, contribution)
    if not bool(target.sum() > 0):
        return weights
    mu = target.sum() / row_density.sum()
    desired_density = target / mu
    # FISTA for nonnegative length-density matching with a small prior on
    # dimensionless factors near one. Scatter operations keep the data sparse.
    regularizer = 0.01
    bound = torch.zeros(n_tracks, device=device).index_add_(
        0, track, contribution * row_density[local_row]
    ).max() + regularizer
    step = 1.0 / bound
    current = torch.ones(n_tracks, device=device)
    extrapolated = current.clone()
    acceleration = 1.0
    for _ in range(300):
        predicted = torch.zeros(n_rows, device=device).index_add_(
            0, local_row, contribution * extrapolated[track]
        )
        gradient = torch.zeros(n_tracks, device=device).index_add_(
            0, track, contribution * (predicted - desired_density)[local_row]
        ) + regularizer * (extrapolated - 1.0)
        updated = torch.clamp(extrapolated - step * gradient, min=0.0)
        next_acceleration = (1.0 + math.sqrt(1.0 + 4.0 * acceleration * acceleration)) / 2.0
        extrapolated = updated + ((acceleration - 1.0) / next_acceleration) * (updated - current)
        current, acceleration = updated, next_acceleration
    used = torch.zeros(n_tracks, device=device, dtype=torch.long).index_add_(
        0, track, torch.ones_like(track)
    ) > 0
    return torch.where(used, current, weights)
