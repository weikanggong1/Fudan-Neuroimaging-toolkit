from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import (
    ec_field,
    fsl_rotation_matrix,
    identity_grid,
    jacobian_from_pe_displacement,
    quadratic_ec_basis,
)
from .spline import fsl_cubic_coefficients, sample_cubic_periodic_fast, valid_mask


def _rigid_matmul(a, b):
    # A TF32-rounded coordinate grid swamps 1e-5 rad finite differences.
    tf32 = a.is_cuda and torch.backends.cuda.matmul.allow_tf32
    try:
        if tf32: torch.backends.cuda.matmul.allow_tf32 = False
        return a @ b
    finally:
        if tf32: torch.backends.cuda.matmul.allow_tf32 = True


def _rigid_inverse_grid(grid, mp, voxel_sizes):
    """Model-grid target -> source scan voxel coordinates for FSL rigid mp."""
    R = fsl_rotation_matrix(mp[None, 3:6])[0]
    vs = torch.as_tensor(voxel_sizes, dtype=grid.dtype, device=grid.device)
    centre = (torch.as_tensor(grid.shape[1:], dtype=grid.dtype, device=grid.device) - 1) * vs / 2
    off = centre - _rigid_matmul(R, centre) + mp[:3]
    target_mm = grid.reshape(3, -1) * vs[:, None]
    src_mm = _rigid_matmul(R.T, target_mm - off[:, None])
    return (src_mm / vs[:, None]).reshape_as(grid)


def _rigid_forward_grid(grid, mp, voxel_sizes):
    """Scan-grid target -> model voxel coordinates under forward movement."""
    R = fsl_rotation_matrix(mp[None, 3:6])[0]
    vs = torch.as_tensor(voxel_sizes, dtype=grid.dtype, device=grid.device)
    centre = (torch.as_tensor(grid.shape[1:], dtype=grid.dtype, device=grid.device) - 1) * vs / 2
    off = centre - _rigid_matmul(R, centre) + mp[:3]
    target_mm = grid.reshape(3, -1) * vs[:, None]
    model_mm = _rigid_matmul(R, target_mm) + off[:, None]
    return (model_mm / vs[:, None]).reshape_as(grid)


def _sample_scalar(volume, coords, precision=1e-8):
    coeff = fsl_cubic_coefficients(volume, precision=precision)
    return sample_cubic_periodic_fast(coeff, coords)[0]


def sample_linear_mask(mask: torch.Tensor, coords: torch.Tensor, threshold: float = 0.99) -> torch.Tensor:
    if coords.ndim == 4:
        coords=coords[None]
    shape=mask.shape
    normalized=[2*coords[:,axis]/(shape[axis]-1)-1 for axis in range(3)]
    grid=torch.stack((normalized[2],normalized[1],normalized[0]),-1)
    sampled=F.grid_sample(mask.to(coords.dtype)[None,None],grid,mode='bilinear',
                          padding_mode='zeros',align_corners=True)[0,0]
    return sampled>threshold


def _inverse_1d_displacement(disp: torch.Tensor, axis: int, inmask: torch.Tensor | None = None,
                             derivative_path: bool = True):
    """Invert each PE line by bracketing adjacent transformed voxel centres."""
    moved=disp.movedim(axis,-1)
    n=moved.shape[-1]
    lines=moved.reshape(-1,n).contiguous()
    target=torch.arange(n,device=disp.device,dtype=disp.dtype)[None].expand_as(lines).contiguous()
    mapped=lines+target
    upper=torch.searchsorted(mapped,target,side='left')
    # DerivativeCalculator searches from zero for each target; FieldGpuUtils
    # retains the preceding lower bracket. They differ on folded PE lines.
    folded=(mapped[:,1:]<mapped[:,:-1]).any(1).nonzero().flatten()
    for indices in folded.split(256):
        line=mapped[indices]
        positions=torch.arange(n,device=disp.device)[None]
        if derivative_path:
            crossing=line[:,:,None]>=target[indices,None,:]
            first=crossing.to(torch.int8).argmax(1)
            upper[indices]=torch.where(crossing.any(1),first,n)
        else:
            lower=torch.zeros(len(indices),device=disp.device,dtype=torch.long)
            columns=[]
            for i in range(n):
                crossing=(positions>=lower[:,None])&(line>=i)
                first=torch.where(crossing.any(1),crossing.to(torch.int8).argmax(1),n)
                columns.append(first)
                lower=(first-1).clamp_min(0)
            upper[indices]=torch.stack(columns,1)
    in_range=(upper>0)&(upper<n)
    lo=(upper-1).clamp(0,n-2); hi=(lo+1)
    mapped_lo=torch.gather(mapped,1,lo)
    mapped_hi=torch.gather(mapped,1,hi)
    fraction=(target-mapped_lo)/(mapped_hi-mapped_lo).clamp_min(1e-6)
    inverse=lo.to(disp.dtype)+fraction-target
    if inmask is None:
        valid=in_range
    else:
        source_mask=inmask.movedim(axis,-1).reshape(-1,n)
        valid=in_range & torch.gather(source_mask,1,lo)
        if derivative_path:
            valid=valid & torch.gather(source_mask,1,hi)
    if derivative_path:
        inverse=torch.where(valid,inverse,0)
        return inverse.reshape(moved.shape).movedim(-1,axis),valid.reshape(moved.shape).movedim(-1,axis)
    # The generic FieldGpuUtils kernel extends the first and last invertible
    # displacements to the ends of each PE line; its mask remains invalid there.
    any_range=in_range.any(1)
    first=in_range.to(torch.int8).argmax(1)
    last=n-1-in_range.flip(1).to(torch.int8).argmax(1)
    first_value=torch.gather(inverse,1,first[:,None])
    last_value=torch.gather(inverse,1,last[:,None])
    inverse=torch.where(target<first[:,None],first_value,inverse)
    inverse=torch.where(target>last[:,None],last_value,inverse)
    inverse=torch.where(any_range[:,None],inverse,0)
    return inverse.reshape(moved.shape).movedim(-1,axis),valid.reshape(moved.shape).movedim(-1,axis)


def _masked_inverse_jacobian(inverse: torch.Tensor, mask: torch.Tensor, axis: int):
    """EDDY's masked periodic PE derivative of the inverse field."""
    previous=inverse.roll(1,axis)
    following=inverse.roll(-1,axis)
    previous_valid=mask.roll(1,axis)
    following_valid=mask.roll(-1,axis)
    lower=torch.where(previous_valid,previous,inverse)
    upper=torch.where(following_valid,following,inverse)
    divisor=torch.where(previous_valid & following_valid,2.0,1.0)
    return torch.where(mask,1.0+(upper-lower)/divisor,1.0)


def _inverse_from_template(disp: torch.Tensor, axis: int,
                           template: torch.Tensor, mask: torch.Tensor):
    """EDDY derivative path: invert near the baseline inverse, or retain it."""
    moved=disp.movedim(axis,-1)
    n=moved.shape[-1]
    lines=moved.reshape(-1,n)
    previous=template.movedim(axis,-1).reshape(-1,n)
    valid=mask.movedim(axis,-1).reshape(-1,n)
    target=torch.arange(n,device=disp.device,dtype=disp.dtype)[None].expand_as(lines)
    upper=torch.trunc(previous+target+1).long()
    in_range=(upper>0)&(upper<n)&valid
    lo=(upper-1).clamp(0,n-2)
    hi=lo+1
    mapped_lo=lo.to(disp.dtype)+torch.gather(lines,1,lo)
    mapped_hi=hi.to(disp.dtype)+torch.gather(lines,1,hi)
    bracketed=in_range&(mapped_hi>target)&(mapped_lo<target)
    calculated=lo.to(disp.dtype)+(target-mapped_lo)/(mapped_hi-mapped_lo).clamp_min(1e-6)-target
    return torch.where(bracketed,calculated,previous).reshape(moved.shape).movedim(-1,axis)


def unwarp_scan_to_model(scan: torch.Tensor, mp: torch.Tensor, ec_params: torch.Tensor,
                         susceptibility: torch.Tensor, pe_vector: torch.Tensor,
                         readout: torch.Tensor, voxel_sizes, precision=1e-8,
                         jacobian_modulate: bool = True,
                         pe_extrapolation_valid: bool = False):
    """Volumetric eddy scan->model transform for the pinned 2111 path."""
    shape = scan.shape
    device, dtype = scan.device, scan.dtype
    grid = identity_grid(shape, device, dtype)
    # FSL general_transform(ima, InverseMovementMatrix, field, I) internally
    # applies (InverseMovementMatrix)^-1, i.e. the forward movement, to model-grid targets.
    rigid = _rigid_forward_grid(grid, mp, voxel_sizes)
    basis = quadratic_ec_basis(shape, voxel_sizes, device, dtype)
    ec_scan = ec_field(ec_params, basis)
    # FSL transforms EC field with inverse movement into model space, then adds susc.
    ec_model = _sample_scalar(ec_scan, rigid[None], precision)
    total = susceptibility + ec_model
    pe_axis = int(torch.nonzero(pe_vector.abs() > 1e-8, as_tuple=False)[0])
    disp = total * readout * pe_vector[pe_axis]
    coords = rigid.clone()
    coords[pe_axis] = coords[pe_axis] + disp
    coeff = fsl_cubic_coefficients(scan, precision)
    out = sample_cubic_periodic_fast(coeff[None], coords[None])[0]
    # FSL intersects image-sampling validity with the EC field's affine
    # sampling validity, even when all EC coefficients are zero.
    vm = valid_mask(coords[None], shape, pe_axis if pe_extrapolation_valid else None)[0] & valid_mask(rigid[None], shape)[0]
    # Jacobian in model space. For one-dimensional EPI field this is 1+dD/dPE.
    vec = torch.zeros((3, *shape), dtype=dtype, device=device); vec[pe_axis] = disp
    jac = jacobian_from_pe_displacement(vec, pe_axis)
    if jacobian_modulate:
        out = out * jac
    return out, vm, jac, coords


def model_to_scan(pred: torch.Tensor, mp: torch.Tensor, ec_params: torch.Tensor,
                  susceptibility: torch.Tensor, pe_vector: torch.Tensor,
                  readout: torch.Tensor, voxel_sizes, precision=1e-8,
                  jacobian_modulate: bool = True, pred_coeff=None, susc_coeff=None,
                  inverse_template=None, return_inverse=False,
                  masked_jacobian=False, grid=None, basis=None):
    """Volumetric model->scan transform used for parameter updates/outliers.

    Implements the 2111 sequence: EC in scan space + susceptibility transformed
    to scan space -> forward PE displacement -> inverse displacement -> rigid
    composition -> spline sampling -> inverse-field Jacobian modulation.
    """
    shape = pred.shape
    device, dtype = pred.device, pred.dtype
    if grid is None: grid = identity_grid(shape, device, dtype)
    if basis is None: basis = quadratic_ec_basis(shape, voxel_sizes, device, dtype)
    ec_scan = ec_field(ec_params, basis)
    # Transform model-space susceptibility to scan grid by inverse forward movement.
    susc_coords = _rigid_inverse_grid(grid, mp, voxel_sizes)
    if susc_coeff is None:
        susc_coeff = fsl_cubic_coefficients(susceptibility, precision)
    # EddyUtils::SetSplineInterp sets the TOPUP susceptibility volume to mirror
    # extrapolation; its CUDA coefficient prefilter remains periodic.
    susc_scan = sample_cubic_periodic_fast(susc_coeff, susc_coords[None],boundary='mirror')[0]
    total = ec_scan + susc_scan
    pe_axis = int(torch.nonzero(pe_vector.abs() > 1e-8, as_tuple=False)[0])
    d = total * readout * pe_vector[pe_axis]
    if inverse_template is None:
        invd, inverse_mask = _inverse_1d_displacement(d, pe_axis, valid_mask(susc_coords[None],shape)[0],
                                                      derivative_path=masked_jacobian)
    else:
        template, inverse_mask = inverse_template
        invd = _inverse_from_template(d, pe_axis, template, inverse_mask)
    # In FSL general_transform(pred, I, inverse_field_mm, R): target scan voxel is
    # displaced by inverse field in scanner/world coordinates and then R^-1 maps to model.
    displaced = grid.clone(); displaced[pe_axis] = displaced[pe_axis] + invd
    coords = _rigid_inverse_grid(displaced, mp, voxel_sizes)
    if pred_coeff is None:
        pred_coeff = fsl_cubic_coefficients(pred, precision)
    out = sample_cubic_periodic_fast(pred_coeff[None], coords[None])[0]
    vm = valid_mask(coords[None], shape)[0] & inverse_mask
    if masked_jacobian:
        jac = _masked_inverse_jacobian(invd, inverse_mask, pe_axis)
    else:
        jac = 1.0 + 0.5 * (invd.roll(-1, pe_axis) - invd.roll(1, pe_axis))
        first=[slice(None)]*3; first[pe_axis]=0
        second=[slice(None)]*3; second[pe_axis]=1
        last=[slice(None)]*3; last[pe_axis]=-1
        previous=[slice(None)]*3; previous[pe_axis]=-2
        jac[tuple(first)]=1.0+0.5*(invd[tuple(second)]-invd[tuple(first)])
        jac[tuple(last)]=1.0+0.5*(invd[tuple(last)]-invd[tuple(previous)])
    if jacobian_modulate:
        out = out * jac
    if return_inverse:
        return out, vm, jac, coords, invd, inverse_mask
    return out, vm, jac, coords
