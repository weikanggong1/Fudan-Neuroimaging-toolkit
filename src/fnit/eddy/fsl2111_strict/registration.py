from __future__ import annotations

import math
import torch

from .geometry import identity_grid, quadratic_ec_basis
from .warp import model_to_scan, sample_linear_mask
from .spline import fsl_cubic_coefficients


# FSL derivative scales are implementation-specific; these starting steps are
# used only to numerically obtain the same local image derivatives on GPU.
# The final update itself is the 2111 normal equation and is not a gradient
# descent optimiser.
_DEFAULT_STEPS = torch.tensor([
    1e-2, 1e-2, 1e-2,        # movement translations, ScanMovementModel::GetDerivScale
    1e-5, 1e-5, 1e-5,        # movement rotations
    1e-3, 1e-3, 1e-3,        # quadratic EC linear components
    1e-5, 1e-5, 1e-5,        # quadratic pure-square components
    1e-5, 1e-5, 1e-5,        # quadratic mixed components
    1e-2,                     # EC field offset when TOPUP field is present
], dtype=torch.float32)


def _masked_gaussian_plan(fwhm_mm: float, voxel_sizes, mask: torch.Tensor):
    if fwhm_mm <= 0:
        return lambda x: x
    sigma = [fwhm_mm / math.sqrt(8 * math.log(2)) / float(v) for v in voxel_sizes]
    kernels=[]
    edges=[]
    m=mask.to(torch.float32)[None,None]
    for dim, s in enumerate(sigma):
        if s <= 1e-6:
            continue
        radius = max(1, int(6 * s + 0.5))
        p = torch.arange(-radius, radius + 1, device=mask.device, dtype=m.dtype)
        k = torch.exp(-0.5 * (p / s) ** 2); k = k / k.sum()
        if dim == 0:
            kernel = k[:, None, None].view(1, 1, -1, 1, 1); pad=(0,0,0,0,radius,radius)
        elif dim == 1:
            kernel = k[None, :, None].view(1, 1, 1, -1, 1); pad=(0,0,radius,radius,0,0)
        else:
            kernel = k[None, None, :].view(1, 1, 1, 1, -1); pad=(radius,radius,0,0,0,0)
        edge = torch.nn.functional.conv3d(torch.nn.functional.pad(torch.ones_like(m), pad), kernel)
        m = torch.nn.functional.conv3d(torch.nn.functional.pad(m, pad), kernel) / edge
        kernels.append((kernel,pad)); edges.append(edge)
    def smooth(x):
        y=(x*mask)[None,None]
        for (kernel,pad),edge in zip(kernels,edges):
            y=torch.nn.functional.conv3d(torch.nn.functional.pad(y,pad),kernel)/edge
        return torch.where(mask,y[0,0]/m[0,0].clamp_min(1e-6),0)
    return smooth


def gaussian_smooth_masked(x: torch.Tensor, fwhm_mm: float, voxel_sizes, mask: torch.Tensor):
    return _masked_gaussian_plan(fwhm_mm,voxel_sizes,mask)(x),mask


def parameter_update(pred_model: torch.Tensor, observed_scan: torch.Tensor,
                     params16: torch.Tensor, susceptibility: torch.Tensor,
                     pe_vector: torch.Tensor, readout: torch.Tensor,
                     voxel_sizes, fwhm_mm: float, base_mask: torch.Tensor,
                     precision: float = 1e-8, active_indices=None):
    """One FSL-2111-style all-parameter Gauss-Newton update for one volume.

    Image derivatives are central numerical derivatives of the exact same
    transform. This preserves the official normal equation/update-rejection
    semantics while keeping all large tensors on CUDA.
    """
    steps = _DEFAULT_STEPS.to(device=params16.device, dtype=params16.dtype)
    if active_indices is None:
        active_indices = list(range(16))
    active_indices = list(active_indices)
    grid=identity_grid(pred_model.shape,pred_model.device,pred_model.dtype)
    basis=quadratic_ec_basis(pred_model.shape,voxel_sizes,pred_model.device,pred_model.dtype)
    pred_coeff=fsl_cubic_coefficients(pred_model,precision)
    susc_coeff=fsl_cubic_coefficients(susceptibility,precision)
    def render(p, inverse_template=None, masked_jacobian=True):
        out, vm, _, coords, inverse, inverse_mask = model_to_scan(pred_model, p[:6], p[6:], susceptibility,
                                      pe_vector, readout, voxel_sizes, precision, True,
                                      pred_coeff, susc_coeff, inverse_template,
                                      return_inverse=True,
                                      masked_jacobian=masked_jacobian,grid=grid,basis=basis)
        return out, vm & sample_linear_mask(base_mask,coords), (inverse,inverse_mask)
    base, vm, inverse_template = render(params16)
    mask = vm
    smooth=_masked_gaussian_plan(fwhm_mm,voxel_sizes,mask)
    dima = base - observed_scan
    dima_s = smooth(dima)
    derivs = []
    # Reuse the unchanged voxel grid and EC basis across parameter derivatives.
    for j in active_indices:
        step = steps[j]
        pp = params16.clone(); pp[j] += step
        yp, _, _ = render(pp, inverse_template)
        # DerivativeCalculator 2111 uses a one-sided p+GetDerivScale finite difference.
        der = (yp - base) / step
        der = smooth(der)
        derivs.append(der)
    X = torch.stack(derivs, -1)[mask].to(torch.float64)
    y = dima_s[mask].to(torch.float64)
    n = max(int(X.shape[0]), 1)
    XtX = X.T @ X
    Xty = X.T @ y
    lam = 1.0 / n
    H = XtX / n + lam * torch.eye(len(active_indices), dtype=torch.float64, device=X.device)
    rhs = Xty / n
    try:
        update = -torch.linalg.solve(H, rhs)
    except RuntimeError:
        update = -torch.linalg.pinv(H) @ rhs
    candidate = params16.clone()
    candidate[torch.as_tensor(active_indices, device=params16.device)] += update.to(params16.dtype)
    old_cost = (y * y).mean()
    new_img, new_vm, _ = render(candidate, masked_jacobian=False)
    new_mask = new_vm
    nd = new_img - observed_scan
    nd = _masked_gaussian_plan(fwhm_mm,voxel_sizes,new_mask)(nd)
    new_cost = (nd[new_mask].to(torch.float64) ** 2).mean() if new_mask.any() else torch.tensor(float('inf'), device=X.device)
    accepted = bool(torch.isfinite(new_cost) and new_cost <= old_cost)
    if not accepted:
        candidate = params16
    diag = {
        "mss": float(old_cost.detach().cpu()),
        "new_mss": float(new_cost.detach().cpu()),
        "accepted": accepted,
        "update_norm": float(update.norm().detach().cpu()),
    }
    return candidate, diag
