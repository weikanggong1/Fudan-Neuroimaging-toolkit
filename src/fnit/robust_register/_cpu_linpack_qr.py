"""CPU Float LINPACK QR candidate, verified on one saved real weighted design.

The first 9196-by-6 real system returned the same six Float32 words as the saved
native SDK QR result. The natural four-iteration IRLS on the same real six-column
system also matched all recorded boundaries. The explicit CPU API uses this
helper for eligible inputs; broader registration inputs remain unvalidated.
It does not replace the GPU solver.
Float work arrays and f2c Double scalar boundaries are explicit. Unsupported
Tensor inputs return None so the caller can retain its previous implementation.
"""
from __future__ import annotations

import torch

_cpu_kernel = None


def try_cpu_float_linpack(weighted_matrix, weighted_rhs):
    """Solve weighted Float CPU least squares, or return None for old path.

    Arguments already contain Float32 weighting, in original positional row
    order. This routine does not change MAD, weights, residual, IRLS or GPU.
    """
    values = (weighted_matrix, weighted_rhs)
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
    if (weighted_matrix.ndim != 2 or weighted_rhs.ndim != 1
            or weighted_matrix.shape[0] != weighted_rhs.shape[0]
            or not 1 <= weighted_matrix.shape[1] <= 12
            or weighted_matrix.shape[0] < weighted_matrix.shape[1]):
        return None

    global _cpu_kernel
    if _cpu_kernel is None:
        import numpy as np
        from numba import njit

        @njit(fastmath=False, parallel=False, cache=False, nogil=True,
              error_model="numpy")
        def dot_float(left, right, start, end):
            # SDOT accumulates Float in the same left-associative item order
            # as its cleanup/unroll5 source (without explicit block unrolling). The f2c function then widens the Float total on return.
            total = np.float32(0)
            for row in range(start, end):
                total = np.float32(total + np.float32(left[row] * right[row]))
            return np.float64(total)

        @njit(fastmath=False, parallel=False, cache=False, nogil=True,
              error_model="numpy")
        def norm_float(column, start, end):
            if end <= start:
                return np.float32(0)
            if end - start == 1:
                return np.float32(abs(column[start]))
            scale = np.float32(0)
            squared_sum = np.float32(1)
            for row in range(start, end):
                if column[row] != np.float32(0):
                    absolute = np.float32(abs(column[row]))
                    if scale < absolute:
                        ratio = np.float32(scale / absolute)
                        squared_sum = np.float32(np.float32(squared_sum *
                            np.float32(ratio * ratio)) + np.float32(1))
                        scale = absolute
                    else:
                        ratio = np.float32(absolute / scale)
                        squared_sum = np.float32(squared_sum + np.float32(ratio * ratio))
            # C sqrt promotes ssq to Double; multiply is Double before Float
            # assignment. Float sqrt then multiply would be a different path.
            return np.float32(np.float64(scale) * np.sqrt(np.float64(squared_sum)))

        @njit(fastmath=False, parallel=False, cache=False, nogil=True,
              error_model="numpy")
        def solve_float(columns, rhs):
            # columns is a private C array, each row is one Fortran column.
            parameter_count, row_count = columns.shape
            auxiliary = np.zeros(parameter_count, dtype=np.float32)
            for pivot in range(parameter_count):
                if pivot == row_count - 1:
                    continue
                norm = norm_float(columns[pivot], pivot, row_count)
                if norm == np.float32(0):
                    continue
                if columns[pivot, pivot] != np.float32(0):
                    if columns[pivot, pivot] < np.float32(0):
                        norm = np.float32(-norm)
                reciprocal = np.float32(np.float32(1) / norm)
                for row in range(pivot, row_count):
                    columns[pivot, row] = np.float32(columns[pivot, row] * reciprocal)
                columns[pivot, pivot] = np.float32(columns[pivot, pivot] + np.float32(1))
                for column in range(pivot + 1, parameter_count):
                    factor = np.float32(-dot_float(columns[pivot], columns[column],
                        pivot, row_count) / np.float64(columns[pivot, pivot]))
                    if factor != np.float32(0):
                        for row in range(pivot, row_count):
                            columns[column, row] = np.float32(columns[column, row] +
                                np.float32(factor * columns[pivot, row]))
                auxiliary[pivot] = columns[pivot, pivot]
                columns[pivot, pivot] = np.float32(-norm)
            # SQRSL JOB=100: apply Q^T to b without forming a dense Q.
            for pivot in range(min(parameter_count, row_count - 1)):
                if auxiliary[pivot] != np.float32(0):
                    diagonal = columns[pivot, pivot]
                    columns[pivot, pivot] = auxiliary[pivot]
                    factor = np.float32(-dot_float(columns[pivot], rhs,
                        pivot, row_count) / np.float64(columns[pivot, pivot]))
                    if factor != np.float32(0):
                        for row in range(pivot, row_count):
                            rhs[row] = np.float32(rhs[row] +
                                np.float32(factor * columns[pivot, row]))
                    columns[pivot, pivot] = diagonal
            solution = rhs[:parameter_count].copy()
            for column in range(parameter_count - 1, -1, -1):
                if columns[column, column] == np.float32(0):
                    return solution, column + 1
                solution[column] = np.float32(solution[column] / columns[column, column])
                factor = np.float32(-solution[column])
                if factor != np.float32(0):
                    for row in range(column):
                        solution[row] = np.float32(solution[row] +
                            np.float32(factor * columns[column, row]))
            return solution, 0

        _cpu_kernel = solve_float

    # Copies isolate f2c-style in-place Householder operations from input storage.
    columns = weighted_matrix.T.contiguous().numpy().copy()
    rhs = weighted_rhs.numpy().copy()
    result, info = _cpu_kernel(columns, rhs)
    if info:
        raise ValueError("Float LINPACK weighted design is rank deficient")
    import numpy as np
    if not bool(np.isfinite(result).all()):
        raise ValueError("Float LINPACK returned a nonfinite step")
    return torch.from_numpy(result)
