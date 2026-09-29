from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class OutlierStats:
    outlier_map: torch.Tensor
    n_stdev: torch.Tensor
    n_sqr_stdev: torch.Tensor
    mean_diff: torch.Tensor
    mean_sqr_diff: torch.Tensor
    nvox: torch.Tensor


def _mean_and_std(mdiff, msqrd, nvox, old_outliers, minvox: int, error_type: int = 1):
    valid = (~old_outliers) & (nvox >= minvox)
    if not valid.any():
        z = torch.tensor(0.0, dtype=mdiff.dtype, device=mdiff.device)
        return z, z, torch.tensor(1.0, dtype=mdiff.dtype, device=mdiff.device), torch.tensor(1.0, dtype=mdiff.dtype, device=mdiff.device)
    if error_type == 1:
        ntot = nvox[valid].sum().to(mdiff.dtype)
        m1 = (nvox[valid] * mdiff[valid]).sum() / ntot
        m2 = (nvox[valid] * msqrd[valid]).sum() / ntot
        # FSL code divides weighted squared deviations by number of slice observations-1,
        # not total voxel count.
        denom = max(int(valid.sum()) - 1, 1)
        s1 = torch.sqrt((nvox[valid] * (mdiff[valid] - m1) ** 2).sum() / denom).clamp_min(1e-12)
        s2 = torch.sqrt((nvox[valid] * (msqrd[valid] - m2) ** 2).sum() / denom).clamp_min(1e-12)
    else:
        m1 = mdiff[valid].mean(); m2 = msqrd[valid].mean()
        s1 = mdiff[valid].std(unbiased=True).clamp_min(1e-12)
        s2 = msqrd[valid].std(unbiased=True).clamp_min(1e-12)
    return m1, m2, s1, s2


def detect_slice_outliers(original_obs: torch.Tensor, predicted_obs: torch.Tensor, valid_mask: torch.Tensor,
                          previous_outliers: torch.Tensor | None = None, nstd: float = 4.0,
                          minvox: int = 250, error_type: int = 1, consider_pos: bool = False,
                          consider_sqr: bool = False) -> OutlierStats:
    """Port of DiffStats + slicewise ReplacementManager::Update defaults.

    Arrays are N x X x Y x Z and detection is only invoked for DWI scans.
    """
    diff = original_obs - predicted_obs
    m = valid_mask.to(diff.dtype)
    nvox = m.sum((1, 2)).to(torch.long)
    denom = nvox.clamp_min(1).to(diff.dtype)
    mdiff = (diff * m).sum((1, 2)) / denom
    msqrd = ((diff * diff) * m).sum((1, 2)) / denom
    if previous_outliers is None:
        previous_outliers = torch.zeros_like(mdiff, dtype=torch.bool)
    pop_m, pop_ms, sd_m, sd_ms = _mean_and_std(mdiff, msqrd, nvox, previous_outliers, minvox, error_type)
    if error_type == 1:
        sf = torch.rsqrt(nvox.clamp_min(1).to(diff.dtype))
    else:
        sf = torch.ones_like(mdiff)
    nsv = (mdiff - pop_m) / (sf * sd_m)
    nsq = (msqrd - pop_ms) / (sf * sd_ms)
    enough = nvox >= minvox
    out = enough & (-nsv > nstd)
    if consider_pos:
        out |= enough & (nsv > nstd)
    if consider_sqr:
        out |= enough & (nsq > nstd)
    return OutlierStats(out, nsv, nsq, mdiff, msqrd, nvox)
