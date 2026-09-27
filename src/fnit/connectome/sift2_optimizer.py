"""GPU SIFT2 coefficient optimization for a fixed streamline-to-fixel map.

This implements the default nonlinear objective and per-streamline Newton
updates of MRtrix3 ``tcksift2`` (3.0.3-103-g026e850d). Fixel segmentation and
streamline mapping are separate inputs, so they can be validated independently.
MRtrix3 source: https://github.com/MRtrix3/mrtrix3/tree/026e850d
License: MPL-2.0; see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SIFT2Optimization:
    """Weights [T], scalar mu, fixel density [F], exclusion mask [F], iterations."""

    weights: torch.Tensor
    mu: float
    fixel_density: torch.Tensor
    excluded_fixels: torch.Tensor
    iterations: int


@torch.inference_mode()
def optimize_sift2_fixels(
    track_index: torch.Tensor,
    fixel_index: torch.Tensor,
    length_mm: torch.Tensor,
    target: torch.Tensor,
    processing_mask: torch.Tensor,
    n_tracks: int,
    *,
    min_td_fraction: float = 0.1,
    tv_lambda: float = 0.1,
    min_iterations: int = 10,
    max_iterations: int = 1000,
    min_cost_decrease: float = 2.5e-5,
) -> SIFT2Optimization:
    """Fit one nonnegative factor per track using the MRtrix SIFT2 objective.

    ``track_index`` and ``fixel_index`` are aligned int tensors [K] naming
    track-fixel pairs. ``length_mm`` is the corresponding positive length [K],
    after MRtrix's 8-bit per-fixel quantization. Each pair must occur once; a
    track may have several pairs with the same fixel when 8-bit storage would
    overflow. ``target`` is the FMLS FOD lobe integral [F], and
    ``processing_mask`` is the ACT/SIFT mask value [F] in [0,1]. All tensors
    must reside on one device; output ``weights`` is float64 [n_tracks] in
    input track order. ``fixel_density`` [F] is weighted track density in mm;
    ``mu`` converts that density to FOD integral units. The original command
    is ``tcksift2 tracks.tck wm_fod.mif weights.txt -act 5tt.mif``.

    This function consumes an already matched fixel model and track map. It
    does not create either, and therefore does not itself imply end-to-end
    SIFT2 parity.
    """
    if n_tracks < 1 or min_iterations < 1 or max_iterations < min_iterations:
        raise ValueError("invalid track count or iteration bounds")
    if not (0 <= min_td_fraction <= 1 and tv_lambda >= 0 and min_cost_decrease >= 0):
        raise ValueError("invalid SIFT2 coefficients")
    if any(x.ndim != 1 for x in (track_index, fixel_index, length_mm, target, processing_mask)):
        raise ValueError("all inputs must be 1D tensors")
    if not (len(track_index) == len(fixel_index) == len(length_mm)) or len(target) != len(processing_mask):
        raise ValueError("incompatible sparse map or fixel dimensions")
    device = target.device
    if any(x.device != device for x in (track_index, fixel_index, length_mm, processing_mask)):
        raise ValueError("all tensors must be on the same device")
    if track_index.dtype not in (torch.int32, torch.int64) or fixel_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("track and fixel indices must be integers")
    if not len(track_index):
        raise ValueError("no streamline contributes to a fixel")
    if bool(((track_index < 0) | (track_index >= n_tracks)).any()) or bool(
        ((fixel_index < 0) | (fixel_index >= len(target))).any()
    ):
        raise ValueError("track or fixel index out of range")
    if bool((length_mm <= 0).any()) or bool((~torch.isfinite(length_mm)).any()):
        raise ValueError("lengths must be positive and finite")
    if bool((target < 0).any()) or bool((~torch.isfinite(target)).any()) or bool(
        ((processing_mask < 0) | (processing_mask > 1) | ~torch.isfinite(processing_mask)).any()
    ):
        raise ValueError("invalid fixel target or processing mask")

    t = track_index.long()
    f = fixel_index.long()
    length = length_mm.to(torch.float64)
    fod = target.to(torch.float64)
    pm = processing_mask.to(torch.float64)
    n_fixels = len(fod)
    original_density = torch.zeros(n_fixels, device=device, dtype=torch.float64).index_add_(0, f, length)
    track_count = torch.zeros(n_fixels, device=device, dtype=torch.int32).index_add_(
        0, f, torch.ones_like(f, dtype=torch.int32)
    )
    weighted_density = (pm * original_density).sum()
    if not bool(weighted_density > 0):
        raise ValueError("processing mask contains no contributing streamline")
    mu = float((pm * fod).sum() / weighted_density)
    excluded = (original_density > 0) & (
        (mu * original_density < min_td_fraction * fod) | (track_count == 1)
    )
    active = ~excluded[f]
    effective_length = torch.zeros(n_tracks, device=device, dtype=torch.float64).index_add_(
        0, t, length * pm[f]
    ).clamp_min(1e-10)
    density_safe = original_density.clamp_min(1e-10)
    reg_scale = tv_lambda * float((pm * fod.square()).sum() / n_tracks)
    initial_cost = float((pm * (mu * original_density - fod).square()).sum())
    stop_decrease = min_cost_decrease * initial_cost
    coefficient = torch.zeros(n_tracks, device=device, dtype=torch.float64)
    previous_cost = initial_cost

    for iteration in range(1, max_iterations + 1):
        old_factor = coefficient.exp()
        density = torch.zeros(n_fixels, device=device, dtype=torch.float64).index_add_(
            0, f, length * old_factor[t]
        )
        mean_coefficient = torch.zeros(n_fixels, device=device, dtype=torch.float64).index_add_(
            0, f, length * coefficient[t]
        ) / density_safe
        mean_coefficient = torch.where(track_count > 1, mean_coefficient, 0.)
        step = torch.zeros_like(coefficient)

        for _ in range(100):
            new_coefficient = coefficient[t] + step[t]
            new_factor = new_coefficient.exp()
            old_contribution = length * old_factor[t]
            other_density = density[f] - old_contribution
            coupled_derivative = (original_density[f] - length) * old_factor[t]
            diff = mu * (other_density + length * new_factor + coupled_derivative * step[t]) - fod[f]
            derivative_density = mu * (length * new_factor + coupled_derivative)
            fraction = length / density_safe[f]
            gradient_pair = 2 * pm[f] * fraction * derivative_density * diff
            hessian_pair = 2 * pm[f] * fraction * (
                derivative_density.square() + mu * length * new_factor * diff
            )
            base = mean_coefficient[f]
            factor_base = base.exp()
            lower = new_coefficient <= base
            tv_gradient = torch.where(
                lower, 2 * (new_coefficient - base),
                2 * new_factor * (new_factor - factor_base),
            )
            tv_hessian = torch.where(
                lower, torch.full_like(new_coefficient, 2.),
                2 * new_factor * (2 * new_factor - factor_base),
            )
            tv_weight = reg_scale * pm[f] * length / effective_length[t]
            gradient_pair = torch.where(active, gradient_pair + tv_weight * tv_gradient, 0.)
            hessian_pair = torch.where(active, hessian_pair + tv_weight * tv_hessian, 0.)
            gradient = torch.zeros_like(coefficient).index_add_(0, t, gradient_pair)
            hessian = torch.zeros_like(coefficient).index_add_(0, t, hessian_pair)
            change = torch.where(hessian != 0, -gradient / hessian, 0.)
            change = torch.where(hessian < 0, -change, change)
            change = torch.where(torch.isfinite(change), change, 0.).clamp(-1., 1.)
            change = torch.where(
                ((step >= 1) & (change > 0)) | ((step <= -1) & (change < 0)),
                0., change,
            )
            step += change
            if float(change.abs().max()) <= 0.001:
                break

        coefficient += step.clamp(-1., 1.)
        factor = coefficient.exp()
        density = torch.zeros(n_fixels, device=device, dtype=torch.float64).index_add_(
            0, f, length * factor[t]
        )
        mean_coefficient = torch.zeros(n_fixels, device=device, dtype=torch.float64).index_add_(
            0, f, length * coefficient[t]
        ) / density_safe
        mean_coefficient = torch.where(track_count > 1, mean_coefficient, 0.)
        tv_delta = coefficient[t] - mean_coefficient[f]
        tv_delta = torch.where(tv_delta <= 0, tv_delta, factor[t] - mean_coefficient[f].exp())
        tv_cost = (pm[f] * length * tv_delta.square() / effective_length[t]).sum()
        cost = float((pm * (mu * density - fod).square()).sum() + reg_scale * tv_cost)
        if iteration >= min_iterations and previous_cost - cost < stop_decrease:
            break
        previous_cost = cost

    return SIFT2Optimization(factor, mu, density, excluded, iteration)
