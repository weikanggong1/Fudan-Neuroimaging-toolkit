"""CPU Float 加权误差验证候选；独立真实同输入与自然 IRLS 已通过。

Float32 平方根权重先平方，再按行累计 sw 与 swr，最后 Float32 除法。
这里只返回最终误差；没有将内部 sw/swr 声称为单独实测输出。
原实现来源、输入域与默认路径状态见本目录 README。
"""
from __future__ import annotations

import torch

_kernel = None


def try_serial_weighted_error(current_residual, sqrt_weights):
    """Return a Float32-rounded Python scalar, or None for the old path.

    Only ordinary contiguous CPU Float32 vectors with no AD wrapper qualify.
    CUDA and unsupported arguments return before NumPy or Numba is imported.
    The caller retains its existing all-zero-weight and finite-error checks.
    """
    values = (current_residual, sqrt_weights)
    if any(type(value) is not torch.Tensor for value in values):
        return None
    if any(value.device.type != "cpu" for value in values):
        return None
    for value in values:
        if (value.dtype != torch.float32 or value.layout != torch.strided
                or value.is_nested or value.ndim != 1
                or value.is_neg() or value.is_conj()
                or not value.is_contiguous() or value.requires_grad
                or torch._C._functorch.is_functorch_wrapped_tensor(value)
                or torch.autograd.forward_ad.unpack_dual(value).tangent is not None):
            return None
    if current_residual.shape != sqrt_weights.shape or not current_residual.numel():
        return None

    global _kernel
    if _kernel is None:
        import numpy as np
        from numba import njit

        @njit(fastmath=False, parallel=False, nogil=True, cache=False,
              error_model="numpy")
        def ordered_error(residual, weights):
            weighted_sum = np.float32(0)
            weight_sum = np.float32(0)
            for row in range(residual.size):
                weight_squared = np.float32(weights[row] * weights[row])
                residual_squared = np.float32(residual[row] * residual[row])
                weight_sum = np.float32(weight_sum + weight_squared)
                weighted_sum = np.float32(
                    weighted_sum + np.float32(weight_squared * residual_squared))
            return np.float32(weighted_sum / weight_sum)

        _kernel = ordered_error
    return float(_kernel(current_residual.numpy(), sqrt_weights.numpy()))
