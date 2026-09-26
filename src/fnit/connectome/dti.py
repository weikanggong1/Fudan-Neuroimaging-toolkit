"""Float32 diffusion-tensor fitting for streamline FA sampling."""

from __future__ import annotations

import torch


def fit_tensor_fa(
    signal: torch.Tensor,
    bvals: torch.Tensor,
    bvecs: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    max_bval: float = 1500.0,
) -> torch.Tensor:
    """Fit a single tensor to low-shell DWI and return voxelwise FA.

    ``signal`` has shape ``[X, Y, Z, N]``. ``bvals`` are in s/mm² and
    ``bvecs`` has shape ``[N, 3]`` in the image's world-axis convention.
    This linear least-squares fit is only used for the FA edge contrast; it
    does not stand in for the multi-shell FOD model used for tractography.
    """
    if signal.ndim != 4 or bvals.shape != (signal.shape[-1],) or bvecs.shape != (signal.shape[-1], 3):
        raise ValueError("expected DWI [X,Y,Z,N], bvals [N], and bvecs [N,3]")
    if mask is not None and mask.shape != signal.shape[:3]:
        raise ValueError("mask must match the DWI spatial shape")
    device = signal.device
    data = signal.to(torch.float32)
    bvals = bvals.to(device=device, dtype=torch.float32)
    bvecs = bvecs.to(device=device, dtype=torch.float32)
    baseline = bvals < 50
    diffusion = (bvals >= 50) & (bvals <= max_bval)
    if not bool(baseline.any()) or int(diffusion.sum()) < 6:
        raise ValueError("DTI fit requires a b0 and at least six low-shell directions")
    directions = torch.nn.functional.normalize(bvecs[diffusion], dim=-1)
    gx, gy, gz = directions.unbind(-1)
    b = bvals[diffusion]
    design = -b[:, None] * torch.stack(
        (gx.square(), gy.square(), gz.square(), 2 * gx * gy, 2 * gx * gz, 2 * gy * gz),
        dim=-1,
    )
    flat = data.reshape(-1, data.shape[-1])
    selected = torch.ones(flat.shape[0], dtype=torch.bool, device=device) if mask is None else mask.reshape(-1).to(device=device, dtype=torch.bool)
    output = torch.zeros(flat.shape[0], dtype=torch.float32, device=device)
    if not bool(selected.any()):
        return output.reshape(signal.shape[:3])
    samples = flat[selected]
    s0 = samples[:, baseline].mean(dim=-1).clamp_min(1e-6)
    log_ratio = (samples[:, diffusion].clamp_min(1e-6) / s0[:, None]).clamp_min(1e-6).log()
    coefficients = log_ratio @ torch.linalg.pinv(design).T
    dxx, dyy, dzz, dxy, dxz, dyz = coefficients.unbind(-1)
    tensor = torch.stack(
        (dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz), dim=-1
    ).reshape(-1, 3, 3)
    eigenvalues = torch.linalg.eigvalsh(tensor).clamp_min(0)
    mean = eigenvalues.mean(dim=-1, keepdim=True)
    numerator = ((eigenvalues - mean).square().sum(dim=-1) * 1.5).sqrt()
    denominator = eigenvalues.square().sum(dim=-1).sqrt().clamp_min(1e-12)
    output[selected] = (numerator / denominator).clamp(0, 1)
    return output.reshape(signal.shape[:3])
