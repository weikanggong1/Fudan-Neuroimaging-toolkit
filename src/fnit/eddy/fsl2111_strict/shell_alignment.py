from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from .geometry import movepar_to_matrix, matrix_to_movepar
from .spline import fsl_cubic_coefficients, sample_cubic_periodic_fast, valid_mask
from .warp import _rigid_inverse_grid, _rigid_forward_grid


def _linear_sample(volume: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    shape=volume.shape
    grid=torch.stack(tuple(2*coords[j]/(shape[j]-1)-1 for j in (2,1,0)),-1)
    return F.grid_sample(volume[None,None],grid[None],mode='bilinear',
                         padding_mode='zeros',align_corners=True)[0,0]


def _soft_mi(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor,
             arange: tuple[float,float], brange: tuple[float,float], bins: int = 256) -> float:
    valid=weights>0
    if int(valid.sum())<1000:
        return 0.0
    x=a[valid]; y=b[valid]; w=weights[valid]
    def soft_index(v, bounds):
        t=(v-bounds[0])*(bins/(bounds[1]-bounds[0]))
        low=t<=0.5; high=t>=bins-0.5
        t=t.clamp(0.5,bins-0.5)-0.5
        i=torch.floor(t).long().clamp(max=bins-1)
        r=t-i
        return torch.where(low,0,torch.where(high,bins-1,i)),torch.where(low|high,0,r)
    ix,rx=soft_index(x,arange); iy,ry=soft_index(y,brange)
    hist=torch.zeros(bins*bins,device=a.device,dtype=torch.float64)
    for dx,wx in ((0,1-rx),(1,rx)):
        for dy,wy in ((0,1-ry),(1,ry)):
            jx=(ix+dx).clamp(max=bins-1); jy=(iy+dy).clamp(max=bins-1)
            hist.scatter_add_(0,jy*bins+jx,(w*wx*wy).to(torch.float64))
    joint=hist.reshape(bins,bins)/w.sum()
    px=joint.sum(0); py=joint.sum(1)
    def entropy(p):
        q=p[p>0]
        return -(q*torch.log(q)).sum()
    return float((entropy(px)+entropy(py)-entropy(joint)).item())


def register_shell_mean(ref: torch.Tensor, moving: torch.Tensor, mask: torch.Tensor, voxel_sizes):
    """Register one shell mean to the b0 mean with symmetric soft MI and 6 DOF."""
    from .geometry import identity_grid
    grid=identity_grid(ref.shape,ref.device,ref.dtype)
    ref_coeff=fsl_cubic_coefficients(ref)
    moving_coeff=fsl_cubic_coefficients(moving)
    soft_mask=mask.to(ref.dtype)
    # FSL supplies each image's robust intensity range to MutualInfoHelper.
    def robust_range(x):
        lo,hi=torch.quantile(x.flatten(),torch.tensor([0.02,0.98],device=x.device))
        return float(lo),float(hi)
    rr=robust_range(ref); mr=robust_range(moving)
    def cost(p):
        mp=torch.as_tensor(p,dtype=ref.dtype,device=ref.device)
        fwd_coords=_rigid_inverse_grid(grid,mp,voxel_sizes)
        fwd=sample_cubic_periodic_fast(moving_coeff[None],fwd_coords[None])[0]
        fwd_w=_linear_sample(soft_mask,fwd_coords)*valid_mask(fwd_coords[None],ref.shape)[0]
        back_coords=_rigid_forward_grid(grid,mp,voxel_sizes)
        back=sample_cubic_periodic_fast(ref_coeff[None],back_coords[None])[0]
        back_w=_linear_sample(soft_mask,back_coords)*valid_mask(back_coords[None],ref.shape)[0]
        return -0.5*(_soft_mi(ref,fwd,fwd_w,rr,mr)+_soft_mi(moving,back,back_w,mr,rr))
    x0=np.zeros(6,dtype=np.float64)
    simplex=np.vstack((x0,x0+np.diag([1.,1.,1.,math.pi/180,math.pi/180,math.pi/180])))
    opt=minimize(cost,x0,method='Nelder-Mead',options={'initial_simplex':simplex,'maxiter':500,
                                                       'xatol':1e-5,'fatol':1e-7})
    return torch.as_tensor(opt.x,dtype=ref.dtype,device=ref.device)


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    h, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = h / max(h.sum(), 1.0)
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    nz = pxy > 0
    denom = px @ py
    return float((pxy[nz] * np.log(pxy[nz] / denom[nz])).sum())


def register_shell_pe(ref: torch.Tensor, moving: torch.Tensor, mask: torch.Tensor, voxel_sizes, pe_axis: int):
    """Estimate the diagnostic PE-only shell translation."""
    from .geometry import identity_grid
    grid = identity_grid(moving.shape, moving.device, moving.dtype)
    ref_coeff=fsl_cubic_coefficients(ref)
    moving_coeff=fsl_cubic_coefficients(moving)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    ref_np = ref.detach().cpu().numpy()
    moving_np=moving.detach().cpu().numpy()
    def cost(p):
        mp=torch.zeros(6,device=moving.device,dtype=moving.dtype)
        mp[pe_axis]=float(p[0])
        coords = _rigid_inverse_grid(grid, mp, voxel_sizes)
        warped = sample_cubic_periodic_fast(moving_coeff[None], coords[None])[0]
        vm = valid_mask(coords[None], moving.shape)[0].detach().cpu().numpy()
        m = mask_np & vm
        if m.sum() < 1000:
            return 1e12
        backward=_rigid_inverse_grid(grid,-mp,voxel_sizes)
        reverse=sample_cubic_periodic_fast(ref_coeff[None],backward[None])[0]
        reverse_mask=valid_mask(backward[None],ref.shape)[0].detach().cpu().numpy()
        m2=mask_np & reverse_mask
        if m2.sum()<1000:
            return 1e12
        return -0.5*(_mutual_information(ref_np[m],warped.detach().cpu().numpy()[m])+
                     _mutual_information(moving_np[m2],reverse.detach().cpu().numpy()[m2]))
    opt=minimize(cost,np.zeros(1),method='Nelder-Mead',
                 options={'initial_simplex':np.array([[0.],[1.]]),'maxiter':500,
                          'xatol':1e-5,'fatol':1e-7})
    update=torch.zeros(6,dtype=moving.dtype,device=moving.device)
    update[pe_axis]=float(opt.x[0])
    return update


def update_shell_movements(movement: torch.Tensor, shell_indices: list[list[int]], shell_updates: list[torch.Tensor], shape, voxel_sizes):
    out=movement.clone()
    for inds, upd in zip(shell_indices, shell_updates):
        U=movepar_to_matrix(upd[None],shape,voxel_sizes)[0]
        for idx in inds:
            M=movepar_to_matrix(out[idx:idx+1],shape,voxel_sizes)[0]
            # PostEddy update composes a shell-level correction with each scan movement.
            out[idx]=matrix_to_movepar((M @ torch.linalg.inv(U))[None],shape,voxel_sizes)[0]
    return out
