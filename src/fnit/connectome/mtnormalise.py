"""PyTorch implementation of MRtrix3 three-tissue ``mtnormalise`` defaults.

The polynomial, tissue balance and outlier updates follow MRtrix3
3.0.3-103-g026e850d ``cmd/mtnormalise.cpp`` (MPL-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MTNormaliseResult:
    """Normalised WM [X,Y,Z,C], GM/CSF [X,Y,Z], field/mask [X,Y,Z], factors [3]."""

    wm: torch.Tensor
    gm: torch.Tensor
    csf: torch.Tensor
    field: torch.Tensor
    accepted_mask: torch.Tensor
    balance_factors: torch.Tensor


def _basis(world: torch.Tensor) -> torch.Tensor:
    x, y, z = world.unbind(-1)
    return torch.stack((
        torch.ones_like(x), x, y, z,
        x*x, y*y, z*z, x*y, x*z, y*z,
        x*x*x, y*y*y, z*z*z, x*x*y, x*x*z, y*y*x,
        y*y*z, z*z*x, z*z*y, x*y*z,
    ), dim=-1)


def _mrtrix_quartile_indices(count: int) -> tuple[int, int]:
    """Zero-based nth_element indices using C++ positive half-up rounding."""
    return math.floor(count * .25 + .5), math.floor(count * .75 + .5)


@torch.inference_mode()
def normalise_mrtrix_three_tissue(
    wm_sh: torch.Tensor,
    gm: torch.Tensor,
    csf: torch.Tensor,
    mask: torch.Tensor,
    affine: torch.Tensor,
) -> MTNormaliseResult:
    """Normalise raw MSMT-CSD tissue images using default ``mtnormalise``.

    Inputs are float32 WM even SH ``[X,Y,Z,C]`` (channel 0 is the WM DC
    coefficient), float32 GM and CSF ``[X,Y,Z]``, a bool 3D processing mask,
    and float32/64 voxel-to-RAS ``[4,4]`` affine, all on one torch device.
    Outputs are float32 images on the original grid and a float64 ``[3]``
    diagnostic balance vector. The full bias field is float32 ``[X,Y,Z]``;
    the accepted mask is bool ``[X,Y,Z]`` after outlier rejection.

    Equivalent command: ``mtnormalise wm.mif wm_norm.mif gm.mif gm_norm.mif
    csf.mif csf_norm.mif -mask brain_mask.mif``. The default third-order
    polynomial, 15 main updates, up to 7 tissue balance updates, and reference
    0.28209479177 are fixed to match the original invocation.
    Outlier quartile indices use C++ positive half-up rounding; Python's
    ties-to-even ``round`` is not equivalent for mask sizes 2 modulo 4.
    """
    shape = wm_sh.shape[:3]
    if (wm_sh.ndim != 4 or wm_sh.shape[-1] < 1 or gm.shape != shape or
            csf.shape != shape or mask.shape != shape or affine.shape != (4, 4)):
        raise ValueError("expected WM [X,Y,Z,C], GM/CSF/mask [X,Y,Z], affine [4,4]")
    if any(x.device != wm_sh.device for x in (gm, csf, mask, affine)):
        raise ValueError("all inputs must share a device")
    if any(x.dtype != torch.float32 for x in (wm_sh, gm, csf)) or mask.dtype != torch.bool:
        raise ValueError("tissue images must be float32 and mask bool")
    if not bool(mask.any()):
        raise ValueError("normalisation mask is empty")
    if wm_sh.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    device = wm_sh.device
    voxel = torch.nonzero(mask, as_tuple=False).to(torch.float64)
    matrix = affine.to(torch.float64)
    world = voxel @ matrix[:3, :3].T + matrix[:3, 3]
    basis = _basis(world)
    data = torch.stack((wm_sh[..., 0][mask], gm[mask], csf[mask]), -1).to(torch.float64).clamp_min(0)
    if not bool(torch.isfinite(data).all()):
        raise ValueError("non-finite tissue data inside processing mask")
    weights = (data.sum(-1) > 0).to(torch.float64)
    field = torch.ones(len(data), device=device, dtype=torch.float64)
    factors = torch.ones(3, device=device, dtype=torch.float64)
    log_reference = math.log(0.28209479177)

    def reject_outliers(multiplier: float) -> bool:
        nonlocal weights
        logsum = ((data @ factors) / field).log()
        ordered = torch.where(torch.isnan(logsum), -float("inf"), logsum)
        n = len(ordered)
        lower_index, upper_index = _mrtrix_quartile_indices(n)
        lower = ordered.kthvalue(lower_index + 1).values
        upper = ordered.kthvalue(upper_index + 1).values
        width = upper - lower
        revised = (torch.isfinite(logsum) &
                   (logsum >= lower - multiplier * width) &
                   (logsum <= upper + multiplier * width)).to(torch.float64)
        changed = bool((revised != weights).any())
        weights = revised
        return changed

    reject_outliers(3.0)
    coefficients = torch.zeros(20, device=device, dtype=torch.float64)
    for _ in range(15):
        for _ in range(7):
            scaled = (data / field[:, None]) * weights[:, None]
            factors = torch.linalg.solve(scaled.T @ scaled, scaled.sum(0))
            if bool((factors <= 0).any()) or not bool(torch.isfinite(factors).all()):
                raise ValueError("non-positive tissue balance factor")
            factors /= factors.log().mean().exp()
            if not reject_outliers(1.5):
                break
        summed = data @ factors
        logsum = torch.where(summed > 0, weights * (summed.log() - log_reference), 0.)
        normal = basis.T @ (weights[:, None] * basis)
        coefficients = torch.linalg.solve(normal, basis.T @ logsum)
        field = (basis @ coefficients).exp()

    axes = [torch.arange(n, device=device, dtype=torch.float64) for n in shape]
    full_voxel = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    full_world = full_voxel @ matrix[:3, :3].T + matrix[:3, 3]
    full_field = (_basis(full_world) @ coefficients).exp().reshape(shape).to(torch.float32)
    accepted = torch.zeros(shape, dtype=torch.bool, device=device)
    accepted[mask] = weights.bool()
    return MTNormaliseResult(
        wm_sh / full_field[..., None], gm / full_field, csf / full_field,
        full_field, accepted, factors,
    )
