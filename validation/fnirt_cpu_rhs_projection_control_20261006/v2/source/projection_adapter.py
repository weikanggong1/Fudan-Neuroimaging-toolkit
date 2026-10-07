"""Private diagnostic: signed FP32 axis division on unchanged raw derivatives.

This is not a production backend. Geometry signs are supplied by a separately
validated header/FSL-grid contract; no reference gradient chooses them.
"""
from __future__ import annotations

import math


def axis_divisors(raw, moving_fsl2vox, voxel_sizes, axis_signs):
    import torch
    if (not isinstance(raw, torch.Tensor) or raw.device.type != "cpu"
            or raw.dtype != torch.float32 or raw.requires_grad
            or raw.layout != torch.strided or raw.ndim != 4 or raw.shape[0] != 3):
        raise ValueError("raw derivatives must be no-grad strided FP32 CPU [3,X,Y,Z]")
    if torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("diagnostic requires no-grad and no CPU autocast")
    if (not isinstance(moving_fsl2vox, torch.Tensor)
            or moving_fsl2vox.device.type != "cpu" or moving_fsl2vox.dtype != torch.float32
            or moving_fsl2vox.requires_grad or moving_fsl2vox.shape != (4, 4)):
        raise ValueError("declared FP32 CPU 4x4 voxel Jacobian required")
    if len(voxel_sizes) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in voxel_sizes):
        raise ValueError("three positive finite header pixdims required")
    if tuple(axis_signs) not in ((-1, 1, 1), (1, 1, 1)):
        raise ValueError("resolved radiological/original-storage axis signs required")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(moving_fsl2vox).all()):
        raise ValueError("nonfinite derivative or matrix")
    jacobian = moving_fsl2vox[:3, :3]
    diagonal = torch.diagonal(jacobian)
    if not torch.equal(jacobian, torch.diag(diagonal)):
        raise ValueError("this axis-division diagnostic excludes rotation and shear")
    # Current FNIT forms the diagonal inverse in FP64, then stores it in FP32.
    # This checks geometry only; the candidate below still divides by FP32 vxs.
    expected_reciprocal = raw.new_tensor(tuple(sign / float(v) for sign, v in zip(axis_signs, voxel_sizes)))
    if not torch.equal(diagonal.view(torch.int32), expected_reciprocal.view(torch.int32)):
        raise ValueError("stored reciprocal/sign disagrees with declared header geometry")
    divisors = raw.new_tensor(tuple(float(v) for v in voxel_sizes))
    if any(float(value) != float(source) for value, source in zip(divisors, voxel_sizes)):
        raise ValueError("pixdim does not retain its header FP32 value")
    if divisors.ndim != 1 or divisors.numel() != 3 or divisors.dtype != torch.float32:
        raise ValueError("non-scalar three-element FP32 divisor Tensor required")
    return divisors


def signed_axis_division(raw, moving_fsl2vox, voxel_sizes, axis_signs):
    import torch
    divisors = axis_divisors(raw, moving_fsl2vox, voxel_sizes, axis_signs)
    components = tuple(raw[axis].neg() if sign < 0 else raw[axis]
                       for axis, sign in enumerate(axis_signs))
    # Explicit Tensor overload plus a three-element non-0D operand. Do not
    # use a Python scalar or 0D Tensor, which can select scalar fastpaths.
    numerator = torch.stack(components)
    denominator = divisors.reshape(3, 1, 1, 1)
    result = torch.ops.aten.div.Tensor(numerator, denominator)
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise RuntimeError("FP32 finite division output required")
    return result
