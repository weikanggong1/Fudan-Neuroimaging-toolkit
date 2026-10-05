"""Private FP64 CPU PCG candidate; no reference inputs or installed FSL calls.

The operator callback is supplied by the caller. No matrix construction,
registration state, GPU setup or numerical tolerance change happens here.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
import own_reductions as arithmetic

@dataclass(frozen=True)
class PCGReport:
    iterations:int
    converged:bool
    relative_residual:float

def cpu_vector(value,name,*,check_finite=True):
    if not isinstance(value,torch.Tensor):raise TypeError(name+' must be a torch Tensor')
    if value.device.type!='cpu' or value.dtype!=torch.float64 or value.requires_grad:
        raise ValueError(name+' must be non-differentiable FP64 CPU')
    if value.ndim!=1:raise ValueError(name+' must be one-dimensional')
    # Arbitrary strided 1D input is supported; copying only preserves FP64 bits.
    result=np.ascontiguousarray(value.numpy())
    if check_finite and not np.isfinite(result).all():raise ValueError(name+' must be finite')
    return result

def preconditioned_conjugate_gradient_cpu(matvec,rhs,*,diagonal=None,
        tolerance=1e-3,max_iterations=500):
    if not math.isfinite(tolerance) or tolerance<=0 or max_iterations<1:
        raise ValueError('invalid stopping parameters')
    right=cpu_vector(rhs,'rhs')
    if diagonal is None:weights=np.ones(right.size,dtype=np.float64)
    else:
        weights=cpu_vector(diagonal,'diagonal')
        if weights.shape!=right.shape or np.any(weights<=0):raise ValueError('positive same-shape diagonal required')
    solution=np.zeros_like(right);residual=right.copy();right_norm=arithmetic.norm(right)
    if right_norm==0:return torch.from_numpy(solution),PCGReport(0,True,0.)
    z=residual/weights;direction=z.copy();rho=arithmetic.dot(residual,z);relative=1.
    for iteration in range(1,max_iterations+1):
        product=cpu_vector(matvec(torch.from_numpy(direction)),'matvec output',check_finite=False)
        if product.shape!=right.shape:raise ValueError('matvec changed vector shape')
        denominator=arithmetic.dot(direction,product)
        if not math.isfinite(denominator) or denominator<=0:
            return torch.from_numpy(solution),PCGReport(iteration-1,False,relative)
        alpha=rho/denominator
        solution=solution+alpha*direction
        residual=residual-alpha*product
        relative=arithmetic.norm(residual)/right_norm
        if relative<=tolerance:return torch.from_numpy(solution),PCGReport(iteration,True,relative)
        z=residual/weights;new_rho=arithmetic.dot(residual,z)
        direction=z+(new_rho/rho)*direction;rho=new_rho
    return torch.from_numpy(solution),PCGReport(max_iterations,False,relative)
