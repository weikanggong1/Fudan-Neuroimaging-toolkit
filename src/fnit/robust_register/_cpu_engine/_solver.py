"""Stable CPU dispatch for the validated arithmetic and original fallback.

No patch to existing default/GPU module globals; this module is CPU engine only.
"""
from ._solver_torch import *
from . import _solver_torch
from .._cpu_linpack_qr import try_cpu_float_linpack
from .._cpu_ordered_residual import try_cpu_ordered_residual
from .._cpu_weighted_error import try_serial_weighted_error
def weighted_qr(design, residual, sqrt_weights):
    if design.ndim != 2 or residual.shape != (design.shape[0],):
        raise ValueError('QR expects A=(rows,parameters), b=(rows,)')
    if design.dtype != torch.float32 or residual.dtype != torch.float32:
        raise ValueError('the source-default QR operates on Float A and b')
    matrix = design * sqrt_weights[:, None]
    rhs = residual * sqrt_weights
    if design.device.type == 'cpu':
        value = try_cpu_float_linpack(matrix, rhs)
        if value is not None:
            solution = value
            if not bool(torch.isfinite(solution).all()):
                raise ValueError('weighted QR returned a nonfinite step')
            return solution
    (q, r) = torch.linalg.qr(matrix, mode='reduced')
    if r.shape[0] != r.shape[1] or bool((r.diagonal() == 0).any()):
        raise ValueError('weighted design is rank deficient; no ridge or SVD fallback')
    solution = torch.linalg.solve_triangular(r, (q.T @ rhs)[:, None], upper=True)[:, 0]
    if not bool(torch.isfinite(solution).all()):
        raise ValueError('weighted QR returned a nonfinite step')
    return solution

def _robust_regression_cpu(design, residual, *, saturation=50.0):
    """MAD/Tukey IRLS, max20, absolute weighted error2e-12, rollback.

    Eligible CPU inputs use Float LINPACK-order QR, ordered residuals and
    weighted-error sums; unsupported inputs retain the original Torch path.
    The six-column real case passed the full natural IRLS comparison.
    An increasing error restores both previous parameters and weights.
    """
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError('saturation must be finite and positive')
    if design.shape[0] < design.shape[1]:
        raise ValueError('fewer valid rows than transform parameters')
    if not bool(torch.isfinite(design).all() and torch.isfinite(residual).all()):
        raise ValueError('analytic design/residual must be finite')
    current_residual = residual.clone()
    previous_error = float('inf')
    previous = None
    rows = []
    for iteration in range(1, 21):
        center = median_even(current_residual)
        sigma = float((median_even((current_residual - center).abs()) * 1.4826).item())
        if sigma < 2e-12:
            weights = torch.ones_like(current_residual)
        else:
            normalized = current_residual * (1.0 / sigma)
            ratio = normalized.double() / saturation
            weights = torch.where(normalized.abs().double() >= saturation, 0.0, 1 - ratio * ratio).float()
        if not bool((weights != 0).any()):
            raise ValueError('Tukey removed every row')
        parameters = weighted_qr(design, residual, weights)
        if design.device.type == 'cpu':
            value = try_cpu_ordered_residual(design, residual, parameters)
            if value is None:
                current_residual = residual - design @ parameters
            else:
                current_residual = value
        else:
            current_residual = residual - design @ parameters
        w2 = weights * weights
        if design.device.type == 'cpu':
            value = try_serial_weighted_error(current_residual, weights)
            if value is None:
                error = float((torch.sum(w2 * (current_residual * current_residual)) / torch.sum(w2)).item())
            else:
                error = value
        else:
            error = float((torch.sum(w2 * (current_residual * current_residual)) / torch.sum(w2)).item())
        if not np.isfinite(error):
            raise ValueError('nonfinite weighted residual error')
        rows.append({'iteration': iteration, 'sigma': sigma, 'weighted_error': error, 'zero_weights': int((weights == 0).sum())})
        if error > previous_error:
            (parameters, weights) = previous
            reason = 'increasing_error_previous_parameters_and_weights_restored'
            selected = iteration - 1
            break
        if error <= 2e-12 or error == previous_error:
            reason = 'absolute_weighted_error' if error <= 2e-12 else 'nondecreasing_error_equal'
            selected = iteration
            break
        previous = (parameters.clone(), weights.clone())
        previous_error = error
    else:
        (reason, selected) = ('max20_budget_exhausted', 20)
    return RegressionResult(parameters, weights, {'rows': int(design.shape[0]), 'columns': int(design.shape[1]), 'dtype': str(design.dtype), 'solver': 'torch_reduced_QR_triangular_solve', 'iterations': rows, 'selected_iteration': selected, 'stop_reason': reason, 'normal_equations': False, 'ridge_or_svd_fallback': False})


def robust_regression(design, residual, *, saturation=50.):
    if (design.device.type != 'cpu' or residual.device.type != 'cpu'
            or saturation != 50. or design.shape[1] not in (6, 12)):
        return _solver_torch.robust_regression(design, residual, saturation=saturation)
    result = _robust_regression_cpu(design, residual, saturation=saturation)
    result.report['baseline_solver_metadata'] = result.report['solver']
    result.report['solver'] = 'CPU_Float_LINPACK_with_original_guard_fallback'
    return result
