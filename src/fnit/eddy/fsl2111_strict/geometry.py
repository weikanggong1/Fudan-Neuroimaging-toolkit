from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .spline import _periodic_prefilter_axis


def fsl_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """FSL MISCMATHS convention: R = Rx Ry Rz.

    angles: (..., 3) in radians. Returns (..., 3, 3).
    """
    rx, ry, rz = angles.unbind(-1)
    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)
    Rx = torch.stack((
        one, zero, zero,
        zero, torch.cos(rx), torch.sin(rx),
        zero, -torch.sin(rx), torch.cos(rx),
    ), dim=-1).reshape(*angles.shape[:-1], 3, 3)
    Ry = torch.stack((
        torch.cos(ry), zero, -torch.sin(ry),
        zero, one, zero,
        torch.sin(ry), zero, torch.cos(ry),
    ), dim=-1).reshape(*angles.shape[:-1], 3, 3)
    Rz = torch.stack((
        torch.cos(rz), torch.sin(rz), zero,
        -torch.sin(rz), torch.cos(rz), zero,
        zero, zero, one,
    ), dim=-1).reshape(*angles.shape[:-1], 3, 3)
    return Rx @ Ry @ Rz


def movepar_to_matrix(mp: torch.Tensor, shape, voxel_sizes) -> torch.Tensor:
    """Port of TOPUP::MovePar2Matrix for mp=(tx,ty,tz,rx,ry,rz)."""
    if mp.shape[-1] != 6:
        raise ValueError("movement parameters must end in 6 values")
    centre = torch.as_tensor(
        [((shape[i] - 1) * voxel_sizes[i]) / 2.0 for i in range(3)],
        dtype=mp.dtype, device=mp.device,
    )
    R = fsl_rotation_matrix(mp[..., 3:6])
    t = mp[..., :3]
    c = centre.expand(*mp.shape[:-1], 3)
    off = c - torch.matmul(R, c[..., None]).squeeze(-1) + t
    M = torch.zeros((*mp.shape[:-1], 4, 4), dtype=mp.dtype, device=mp.device)
    M[..., :3, :3] = R
    M[..., :3, 3] = off
    M[..., 3, 3] = 1
    return M


def matrix_to_movepar(M: torch.Tensor, shape, voxel_sizes) -> torch.Tensor:
    """Port of TOPUP::Matrix2MovePar for rigid matrices."""
    R = M[..., :3, :3]
    cy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 0, 1] ** 2)
    normal = cy >= 1e-4
    ax = torch.empty_like(cy)
    ay = torch.empty_like(cy)
    az = torch.empty_like(cy)
    ax[normal] = torch.atan2(R[..., 1, 2][normal] / cy[normal], R[..., 2, 2][normal] / cy[normal])
    ay[normal] = torch.atan2(-R[..., 0, 2][normal], cy[normal])
    az[normal] = torch.atan2(R[..., 0, 1][normal] / cy[normal], R[..., 0, 0][normal] / cy[normal])
    if (~normal).any():
        ax[~normal] = torch.atan2(-R[..., 2, 1][~normal], R[..., 1, 1][~normal])
        ay[~normal] = torch.atan2(-R[..., 0, 2][~normal], torch.zeros_like(cy[~normal]))
        az[~normal] = 0
    zero_t = torch.zeros((*M.shape[:-2], 6), dtype=M.dtype, device=M.device)
    zero_t[..., 3:] = torch.stack((ax, ay, az), -1)
    R0 = movepar_to_matrix(zero_t, shape, voxel_sizes)
    tr = M[..., :3, 3] - R0[..., :3, 3]
    return torch.cat((tr, torch.stack((ax, ay, az), -1)), -1)


def rereference_movement(movement: torch.Tensor, ref: int, shape, voxel_sizes) -> torch.Tensor:
    Ms = movepar_to_matrix(movement, shape, voxel_sizes)
    Mr_inv = torch.linalg.inv(Ms[ref])
    out = torch.matmul(Ms, Mr_inv)
    return matrix_to_movepar(out, shape, voxel_sizes)


def quadratic_ec_basis(shape, voxel_sizes, device, dtype=torch.float32):
    axes = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    centre = torch.as_tensor([(n - 1) / 2.0 for n in shape], device=device, dtype=dtype)[:, None, None, None]
    vox = torch.as_tensor(voxel_sizes, device=device, dtype=dtype)[:, None, None, None]
    x, y, z = (grid - centre) * vox
    return torch.stack((x, y, z, x*x, y*y, z*z, x*y, x*z, y*z, torch.ones_like(x)))


def ec_field(params: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...k,kxyz->...xyz", params, basis)


def hz_to_voxel_displacement(field_hz: torch.Tensor, pe_vector: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
    """FSL acquisition convention: displacement in voxel units along PE vector."""
    # pe_vector is (...,3), normally exactly one non-zero entry +/-1.
    return field_hz[..., None, :, :, :] * (pe_vector * readout[..., None])[..., :, None, None, None]


def voxel_to_mm_displacement(disp_vox: torch.Tensor, voxel_sizes) -> torch.Tensor:
    s = torch.as_tensor(voxel_sizes, dtype=disp_vox.dtype, device=disp_vox.device)
    return disp_vox * s.view(*([1] * (disp_vox.ndim - 4)), 3, 1, 1, 1)


def jacobian_from_pe_displacement(disp_vox: torch.Tensor, pe_axis: int) -> torch.Tensor:
    """FSL scan-to-model cubic-spline field derivative at voxel centres."""
    d = disp_vox[..., pe_axis, :, :, :]
    axis = d.ndim - 3 + pe_axis
    coeff = _periodic_prefilter_axis(d, axis)
    return 1.0 + 0.5 * (coeff.roll(-1, axis) - coeff.roll(1, axis))


def identity_grid(shape, device, dtype=torch.float32):
    axes = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def apply_affine_voxel(grid: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply one/batched homogeneous matrix to a voxel grid."""
    flat = grid.reshape(3, -1)
    ones = torch.ones((1, flat.shape[1]), dtype=grid.dtype, device=grid.device)
    h = torch.cat((flat, ones), 0)
    if M.ndim == 2:
        return (M @ h)[:3].reshape_as(grid)
    out = torch.matmul(M, h.expand(M.shape[0], -1, -1))[:, :3]
    return out.reshape(M.shape[0], 3, *grid.shape[1:])


def fsl_linear_design(bvals: torch.Tensor, bvecs_n3: torch.Tensor) -> torch.Tensor:
    return (bvals[:, None] / 1000.0) * bvecs_n3


def apply_slm_linear(ec_params: torch.Tensor, bvals: torch.Tensor, bvecs_n3: torch.Tensor) -> torch.Tensor:
    """Port ECScanManager::SetPredictedECParam(..., Linear_2nd_lvl_mdl)."""
    if ec_params.numel() == 0:
        return ec_params
    X = fsl_linear_design(bvals, bvecs_n3).to(dtype=torch.float64)
    H = X @ torch.linalg.pinv(X.T @ X) @ X.T
    out = ec_params.to(torch.float64).clone()
    if out.shape[1] > 1:
        out[:, :-1] = H @ out[:, :-1]
    return out.to(ec_params.dtype)


def separate_offset_from_movement(
    movement: torch.Tensor,
    ec_params: torch.Tensor,
    bvals: torch.Tensor,
    bvecs_n3: torch.Tensor,
    pe_vector: torch.Tensor,
    readout: torch.Tensor,
    voxel_sizes,
):
    """Linear offset model used by --sep_offs_move in eddy 2111.0.

    GetHz2mmVector is PE * readout * voxel-size in mm/Hz for the acquisition.
    """
    if ec_params.shape[1] == 0:
        return movement, ec_params
    hz2mm = pe_vector * readout[:, None] * torch.as_tensor(voxel_sizes, dtype=movement.dtype, device=movement.device)
    denom = (hz2mm * hz2mm).sum(1).clamp_min(1e-12)
    apparent_hz = ec_params[:, -1] + (movement[:, :3] * hz2mm).sum(1) / denom
    X = fsl_linear_design(bvals, bvecs_n3).to(torch.float64)
    X = X - X.mean(0, keepdim=True)
    H = X @ torch.linalg.pinv(X.T @ X) @ X.T
    hz_hat = (H @ apparent_hz.to(torch.float64)).to(movement.dtype)
    out_ec = ec_params.clone(); out_ec[:, -1] = hz_hat
    out_m = movement.clone()
    resid = apparent_hz - hz_hat
    active = hz2mm.abs() > 0
    out_m[:, :3] = torch.where(active, resid[:, None] * hz2mm, out_m[:, :3])
    return out_m, out_ec


def load_topup_movpar(prefix, n_acq_rows: int):
    if prefix is None:
        return np.zeros((n_acq_rows, 6), dtype=np.float64)
    p = Path(prefix)
    f = p.with_name(p.name + "_movpar.txt")
    if not f.exists():
        raise FileNotFoundError(f"FSL strict backend requires TOPUP movement file: {f}")
    a = np.loadtxt(f, dtype=np.float64, ndmin=2)
    if a.shape[1] != 6:
        raise ValueError(f"invalid TOPUP movpar shape {a.shape}")
    if a.shape[0] < n_acq_rows:
        raise ValueError("TOPUP movpar has fewer rows than acqparams")
    return a
