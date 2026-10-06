"""CPU Float 残差验证候选；独立真实同输入对照与自然 IRLS 已通过。

每行按列顺序累计 Float32 A*p，再单独计算 Float32 b-A*p。
原实现来源和验收范围见本目录 README；默认配准与 GPU 路径未修改。
"""
from __future__ import annotations

import torch

_kernel = None


def try_cpu_ordered_residual(design_matrix, rhs, parameters):
    """Return a fresh CPU Float32 residual, or None for the caller's old path.

    This changes no weights, QR, error reduction or GPU operation. The caller
    supplies unweighted design and rhs in their original row/column order.
    """
    values = (design_matrix, rhs, parameters)
    if any(type(value) is not torch.Tensor for value in values):
        return None
    if any(value.device.type != "cpu" for value in values):
        return None
    for value in values:
        if (value.dtype != torch.float32 or value.layout != torch.strided
                or value.is_nested or value.is_neg() or value.is_conj()
                or not value.is_contiguous() or value.requires_grad
                or torch._C._functorch.is_functorch_wrapped_tensor(value)
                or torch.autograd.forward_ad.unpack_dual(value).tangent is not None):
            return None
    if (design_matrix.ndim != 2 or rhs.ndim != 1 or parameters.ndim != 1
            or design_matrix.shape[0] != rhs.shape[0]
            or design_matrix.shape[1] != parameters.shape[0]
            or not 1 <= parameters.numel() <= 12):
        return None

    global _kernel
    if _kernel is None:
        import numpy as np
        from numba import njit

        @njit(fastmath=False, parallel=False, cache=False, nogil=True,
              error_model="numpy")
        def ordered_residual(matrix, target, solution):
            residual = np.empty(target.size, dtype=np.float32)
            for row in range(target.size):
                prediction = np.float32(0)
                for column in range(solution.size):
                    product = np.float32(matrix[row, column] * solution[column])
                    prediction = np.float32(prediction + product)
                residual[row] = np.float32(target[row] - prediction)
            return residual

        _kernel = ordered_residual
    return torch.from_numpy(_kernel(design_matrix.numpy(), rhs.numpy(), parameters.numpy()))
