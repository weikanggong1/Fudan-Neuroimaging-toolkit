"""MRtrix 026e850d FMLS positive FOD lobes and direction lookup in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
import math
from importlib.resources import files

import numpy as np
import torch

from .fod import real_sh


@dataclass(frozen=True)
class FixelSegmentation:
    """Sparse FMLS map; global fixel index zero is reserved as invalid."""

    voxel_ids: torch.Tensor  # C-order flat spatial indices, sorted
    first_fixel_index: torch.Tensor  # MRtrix internal one-based, x-fastest ordering
    count: torch.Tensor  # uint8 per listed voxel
    lookup_table: torch.Tensor  # uint8 [V,1281], local lobe ID
    fixel_integrals: torch.Tensor  # float64 [F+1], with dummy zero
    count_image: torch.Tensor  # uint8 [X,Y,Z]
    target_image: torch.Tensor  # float64 [X,Y,Z], NaN where no lobe


def _directions(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    asset = files("fnit.connectome").joinpath("data/mrtrix_sift2_1281.npz")
    with asset.open("rb") as stream, np.load(stream) as data:
        xyz = torch.as_tensor(data["directions"].copy(), device=device)
        indptr = data["adjacency_indptr"].copy()
        indices = data["adjacency_indices"].copy()
        weights = torch.as_tensor(data["integration_weights"].copy(), device=device)
    adjacent = torch.arange(1281, dtype=torch.long).repeat(6, 1).T
    for row in range(1281):
        neighbors = indices[indptr[row] : indptr[row + 1]]
        adjacent[row, : len(neighbors)] = torch.as_tensor(neighbors.astype(np.int64))
    return xyz, adjacent.to(device), weights


def _peak_al_table(lmax: int, device: torch.device) -> torch.Tensor:
    elevation = torch.linspace(0.0, math.pi, 2562, device=device, dtype=torch.float64)
    phi_zero = torch.stack((elevation.sin(), torch.zeros_like(elevation), elevation.cos()), dim=1)
    basis = real_sh(phi_zero, lmax)
    table = torch.empty((2562, (lmax // 2 + 1) ** 2), device=device, dtype=torch.float64)
    for degree in range(0, lmax + 1, 2):
        for order in range(degree + 1):
            table[:, degree * degree // 4 + order] = basis[:, degree * (degree + 1) // 2 + order] / (math.sqrt(2.0) if order else 1.0)
    return table


def _refine_peaks(
    sh: torch.Tensor, direction: torch.Tensor, lmax: int, table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MRtrix SH.h get_peak Newton updates, batched over borderline lobes."""
    unit = direction.clone()
    amplitude_out = torch.full((len(sh),), float("nan"), device=sh.device, dtype=torch.float64)
    finished = torch.zeros(len(sh), device=sh.device, dtype=torch.bool)
    failed = torch.zeros_like(finished)
    sqrt2 = math.sqrt(2.0)
    inc = math.pi / 2561.0
    for _ in range(50):
        az = torch.atan2(unit[:, 1], unit[:, 0])
        el = torch.acos(unit[:, 2].clamp(-1.0, 1.0))
        sine, cosine = el.sin(), el.cos()
        at_pole = sine < 1e-4
        pos = el / inc
        lower = pos.long().clamp(0, 2561)
        frac = torch.where(lower < 2561, pos - lower, 0.0)
        al = table[lower] * (1.0 - frac[:, None]) + table[(lower + 1).clamp_max(2561)] * frac[:, None]
        amp = sh[:, 0] * al[:, 0]
        d_el = torch.zeros_like(amp)
        d_az = torch.zeros_like(amp)
        d2_el = torch.zeros_like(amp)
        d2_el_az = torch.zeros_like(amp)
        d2_az = torch.zeros_like(amp)
        for degree in range(2, lmax + 1, 2):
            base = degree * degree // 4
            v = sh[:, degree * (degree + 1) // 2]
            amp = amp + v * al[:, base]
            d_el = d_el + v * math.sqrt(degree * (degree + 1)) * al[:, base + 1]
            d2_el = d2_el + v * (math.sqrt(degree * (degree + 1) * (degree - 1) * (degree + 2)) * al[:, base + 2] - degree * (degree + 1) * al[:, base]) / 2.0
        for order in range(1, lmax + 1):
            caz, saz = sqrt2 * torch.cos(order * az), sqrt2 * torch.sin(order * az)
            for degree in range(order + (order & 1), lmax + 1, 2):
                base = degree * degree // 4
                center = degree * (degree + 1) // 2
                vp, vm = sh[:, center + order], sh[:, center - order]
                term = vp * caz + vm * saz
                cross = vm * caz - vp * saz
                alm = al[:, base + order]
                amp = amp + term * alm
                tmp = math.sqrt((degree + order) * (degree - order + 1)) * al[:, base + order - 1]
                if degree > order:
                    tmp = tmp - math.sqrt((degree - order) * (degree + order + 1)) * al[:, base + order + 1]
                tmp = tmp / -2.0
                d_el = d_el + term * tmp
                tmp2 = -((degree + order) * (degree - order + 1) + (degree - order) * (degree + order + 1)) * alm
                if order == 1:
                    tmp2 = tmp2 - math.sqrt((degree + order) * (degree - order + 1) * (degree + order - 1) * (degree - order + 2)) * al[:, base + 1]
                else:
                    tmp2 = tmp2 + math.sqrt((degree + order) * (degree - order + 1) * (degree + order - 1) * (degree - order + 2)) * al[:, base + order - 2]
                if degree > order + 1:
                    tmp2 = tmp2 + math.sqrt((degree - order) * (degree + order + 1) * (degree - order - 1) * (degree + order + 2)) * al[:, base + order + 2]
                d2_el = d2_el + term * tmp2 / 4.0
                d_az = d_az + torch.where(at_pole, cross * tmp, order * cross * alm)
                d2_el_az = d2_el_az + order * cross * tmp
                d2_az = d2_az - term * order * order * alm
        safe_sine = torch.where(at_pole, 1.0, sine)
        d_az = torch.where(at_pole, d_az, d_az / safe_sine)
        d2_el_az = torch.where(at_pole, 0.0, d2_el_az / safe_sine)
        d2_az = torch.where(at_pole, 0.0, d2_az / safe_sine.square())
        grad_norm = torch.sqrt(d_el.square() + d_az.square())
        el_dir = torch.where(grad_norm > 0, d_el / grad_norm, 0.0)
        az_dir = torch.where(grad_norm > 0, d_az / grad_norm, 0.0)
        first = az_dir * d_az + el_dir * d_el
        second = az_dir.square() * d2_az + 2.0 * az_dir * el_dir * d2_el_az + el_dir.square() * d2_el
        step = torch.where(second != 0, -first / second, 0.0).abs().clamp_max(0.2)
        el_step, az_step = el_dir * step, az_dir * step
        update = torch.stack((
            unit[:, 0] + el_step * az.cos() * cosine - az_step * az.sin(),
            unit[:, 1] + el_step * az.sin() * cosine + az_step * az.cos(),
            unit[:, 2] - el_step * sine,
        ), dim=1)
        update = torch.nn.functional.normalize(update, dim=1)
        running = ~(finished | failed)
        converged = running & torch.isfinite(step) & (step < 1e-4)
        amplitude_out = torch.where(converged, amp, amplitude_out)
        failed = failed | (running & (~torch.isfinite(step) | ~torch.isfinite(update).all(dim=1)))
        unit = torch.where(running[:, None], update, unit)
        finished = finished | converged
        if not bool((~(finished | failed)).any()):
            break
    return amplitude_out, unit, finished


@torch.inference_mode()
def segment_fod_fixels(
    wm_sh: torch.Tensor,
    processing_mask: torch.Tensor,
    *,
    batch_size: int = 4096,
) -> FixelSegmentation:
    """Segment masked WM FOD voxels using MRtrix's default FMLS growth rule.

    FOD SH coefficients are float32 [X,Y,Z,C]. All per-voxel angular work is
    batched on ``wm_sh.device`` in float64, matching MRtrix's ``default_type``.
    The fixed 1281-direction table and integration weights are package data.
    """
    if wm_sh.ndim != 4 or wm_sh.dtype != torch.float32 or processing_mask.shape != wm_sh.shape[:3]:
        raise ValueError("expected float32 WM SH [X,Y,Z,C] and same-grid processing mask")
    if wm_sh.device != processing_mask.device or batch_size < 1:
        raise ValueError("FOD and processing mask must share a device; batch_size must be positive")
    device = wm_sh.device
    xsize, ysize, zsize, ncoeff = wm_sh.shape
    lmax = next((l for l in range(0, 34, 2) if (l + 1) * (l + 2) // 2 == ncoeff), None)
    if lmax is None:
        raise ValueError("WM SH coefficient count must correspond to an even order")
    xyz, adjacent, weights = _directions(device)
    basis = real_sh(xyz, lmax)
    peak_table = _peak_al_table(lmax, device)
    # PyTorch's C-order tensor layout differs from MRtrix's x-fastest Loop.
    f_order = torch.nonzero(processing_mask.permute(2, 1, 0).reshape(-1) != 0).flatten()
    xv = f_order % xsize
    yv = f_order.div(xsize, rounding_mode="floor") % ysize
    zv = f_order.div(xsize * ysize, rounding_mode="floor")
    active_c = (xv * ysize + yv) * zsize + zv
    all_count, all_target, all_lookup, all_integrals = [], [], [], []
    storage_dtype = torch.uint8
    for ids in active_c.split(batch_size):
        coefficients = wm_sh.reshape(-1, ncoeff)[ids].to(torch.float64)
        value = coefficients @ basis.T
        order = value.abs().argsort(dim=1, descending=True, stable=True)
        first_value = value.gather(1, order[:, :1]).flatten()
        eligible = (coefficients[:, 0] > 0) & torch.isfinite(coefficients[:, 0]) & (first_value > 0)
        rows = torch.arange(len(ids), device=device)
        labels = torch.full((len(ids), 1281), -1, device=device, dtype=torch.long)
        deferred = torch.full_like(labels, -1)
        integral = torch.zeros((len(ids), 1281), device=device, dtype=torch.float64)
        peak = torch.zeros_like(integral)
        seed_direction = torch.zeros((len(ids), 1281), device=device, dtype=torch.long)
        next_lobe = torch.zeros(len(ids), device=device, dtype=torch.long)
        for rank in range(1281):
            direction = order[:, rank]
            amplitude = value[rows, direction]
            neighbors = labels.gather(1, adjacent[direction])
            earliest = torch.where(neighbors >= 0, neighbors, 1281).min(dim=1).values
            has_neighbor = earliest < 1281
            multiple = ((neighbors >= 0) & (neighbors != earliest[:, None])).any(dim=1)
            positive = eligible & (amplitude > 0)
            new = positive & ~has_neighbor
            assigned = positive & ~multiple
            postponed = positive & multiple
            lobe = torch.where(has_neighbor, earliest, next_lobe)
            safe_lobe = torch.where(positive, lobe, 0)
            labels[rows, direction] = torch.where(assigned, safe_lobe, -1)
            deferred[rows, direction] = torch.where(postponed, safe_lobe, -1)
            old_peak = peak.gather(1, safe_lobe[:, None]).flatten()
            peak.scatter_(1, safe_lobe[:, None], torch.where(new, amplitude, old_peak)[:, None])
            old_seed = seed_direction.gather(1, safe_lobe[:, None]).flatten()
            seed_direction.scatter_(1, safe_lobe[:, None], torch.where(new, direction, old_seed)[:, None])
            integral.scatter_add_(1, safe_lobe[:, None],
                                  torch.where(assigned, amplitude * weights[direction], 0.0)[:, None])
            next_lobe += new.long()
        integral.scatter_add_(1, deferred.clamp_min(0),
                              torch.where(deferred >= 0, value * weights, 0.0))
        raw_label = torch.where(deferred >= 0, deferred, labels)
        borderline = (torch.arange(1281, device=device)[None, :] < next_lobe[:, None]) & (peak > 0) & (peak < 0.1)
        row_index, lobe_index = torch.where(borderline)
        if len(row_index):
            refined_amp, refined_dir, converged = _refine_peaks(
                coefficients[row_index], xyz[seed_direction[row_index, lobe_index]], lmax, peak_table,
            )
            nearest = (refined_dir @ xyz.T).abs().argmax(dim=1)
            accepted = converged & (raw_label[row_index, nearest] == lobe_index)
            peak[row_index, lobe_index] = torch.where(accepted, refined_amp, peak[row_index, lobe_index])
        keep = (torch.arange(1281, device=device)[None, :] < next_lobe[:, None]) & (peak >= 0.1)
        count = keep.sum(dim=1)
        remap = keep.long().cumsum(dim=1) - 1
        valid = (raw_label >= 0) & keep.gather(1, raw_label.clamp_min(0))
        lookup = torch.where(valid, remap.gather(1, raw_label.clamp_min(0)), count[:, None])
        kept_integral = torch.zeros_like(integral)
        kept_integral.scatter_add_(1, remap.clamp_min(0), torch.where(keep, integral, 0.0))
        while True:
            pending = (lookup >= count[:, None]) & (count[:, None] > 0)
            if not bool(pending.any()):
                break
            neighbor_label = lookup[:, adjacent]
            neighbor_valid = neighbor_label < count[:, None, None]
            neighbor_score = kept_integral.gather(1, neighbor_label.reshape(len(ids), -1)).reshape(len(ids), 1281, 6)
            neighbor_score = torch.where(neighbor_valid, neighbor_score, -float("inf"))
            choice = neighbor_score.argmax(dim=2, keepdim=True)
            proposed = neighbor_label.gather(2, choice).squeeze(2)
            lookup = torch.where(pending & neighbor_valid.any(dim=2), proposed, lookup)
        if storage_dtype == torch.uint8 and bool((count > 255).any()):
            storage_dtype = torch.int16
            all_count = [item.to(storage_dtype) for item in all_count]
            all_lookup = [item.to(storage_dtype) for item in all_lookup]
        all_count.append(count.to(storage_dtype))
        all_target.append(torch.where(keep, integral, 0.0).sum(dim=1))
        all_lookup.append(lookup.to(storage_dtype))
        all_integrals.append(integral[keep])
    count = torch.cat(all_count) if all_count else torch.empty(0, device=device, dtype=storage_dtype)
    target = torch.cat(all_target) if all_target else torch.empty(0, device=device, dtype=torch.float64)
    lookup = torch.cat(all_lookup) if all_lookup else torch.empty((0, 1281), device=device, dtype=storage_dtype)
    prefix = count.long().cumsum(dim=0) - count.long() + 1
    valid = count > 0
    voxel_ids = active_c[valid]
    c_order = voxel_ids.argsort()
    count_image = torch.zeros(wm_sh.shape[:3], device=device, dtype=storage_dtype)
    count_image.reshape(-1)[active_c] = count
    target_image = torch.full(wm_sh.shape[:3], float("nan"), device=device, dtype=torch.float64)
    target_image.reshape(-1)[active_c[valid]] = target[valid]
    fixel_integrals = torch.cat((torch.zeros(1, device=device, dtype=torch.float64), *all_integrals))
    return FixelSegmentation(
        voxel_ids[c_order], prefix[valid][c_order], count[valid][c_order], lookup[valid][c_order],
        fixel_integrals, count_image, target_image,
    )
