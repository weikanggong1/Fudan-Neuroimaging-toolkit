"""Analytic robust normal flow, not an autograd image-loss optimizer.

Modified PyTorch adaptation of FreeSurfer robust registration d932c45.
See licenses/FreeSurfer.txt. Float QR is retained; no normal-equation solve.

Original author: Martin Reuter. Copyright (c) 2021 The General Hospital
Corporation (Boston, MA), "MGH". FNIT modification: PyTorch backend, 2026.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ._sampling import blur, partials


@dataclass
class RegressionResult:
    parameters: torch.Tensor
    sqrt_weights: torch.Tensor
    report: dict


def median_even(values):
    if values.numel() == 0:
        raise ValueError("median needs a nonempty vector")
    n = values.numel()
    upper = values.kthvalue(n // 2 + 1).values
    if n % 2:
        return upper
    return (values.kthvalue(n // 2).values + upper) * .5


def weighted_qr(design, residual, sqrt_weights):
    if design.ndim != 2 or residual.shape != (design.shape[0],):
        raise ValueError("QR expects A=(rows,parameters), b=(rows,)")
    if design.dtype != torch.float32 or residual.dtype != torch.float32:
        raise ValueError("the source-default QR operates on Float A and b")
    matrix = design * sqrt_weights[:, None]
    rhs = residual * sqrt_weights
    q, r = torch.linalg.qr(matrix, mode="reduced")
    if r.shape[0] != r.shape[1] or bool((r.diagonal() == 0).any()):
        raise ValueError("weighted design is rank deficient; no ridge or SVD fallback")
    solution = torch.linalg.solve_triangular(r, (q.T @ rhs)[:, None], upper=True)[:, 0]
    if not bool(torch.isfinite(solution).all()):
        raise ValueError("weighted QR returned a nonfinite step")
    return solution


def robust_regression(design, residual, *, saturation=50.):
    """MAD/Tukey IRLS, max20, absolute weighted error2e-12, rollback.

    QR and reductions use PyTorch's backend rather than VNL/LINPACK. Real
    native arithmetic equivalence is an explicit, pending benchmark gate.
    An increasing error restores both previous parameters and weights.
    """
    if not np.isfinite(saturation) or saturation <= 0:
        raise ValueError("saturation must be finite and positive")
    if design.shape[0] < design.shape[1]:
        raise ValueError("fewer valid rows than transform parameters")
    if not bool(torch.isfinite(design).all() and torch.isfinite(residual).all()):
        raise ValueError("analytic design/residual must be finite")
    current_residual = residual.clone()
    previous_error = float("inf")
    previous = None
    rows = []
    for iteration in range(1, 21):
        center = median_even(current_residual)
        sigma = float((median_even((current_residual - center).abs()) * 1.4826).item())
        if sigma < 2e-12:
            weights = torch.ones_like(current_residual)
        else:
            normalized = current_residual * (1. / sigma)
            # Source getSqrtTukeyDiaWeights uses a Double ratio/product
            # even when Regression<T> is Float, then casts w back to T.
            ratio = normalized.double() / saturation
            weights = torch.where(normalized.abs().double() >= saturation, 0.,
                                  1 - ratio * ratio).float()
        if not bool((weights != 0).any()):
            raise ValueError("Tukey removed every row")
        parameters = weighted_qr(design, residual, weights)
        current_residual = residual - design @ parameters
        w2 = weights * weights
        error = float((torch.sum(w2 * (current_residual * current_residual)) / torch.sum(w2)).item())
        if not np.isfinite(error):
            raise ValueError("nonfinite weighted residual error")
        rows.append({"iteration": iteration, "sigma": sigma, "weighted_error": error,
                     "zero_weights": int((weights == 0).sum())})
        if error > previous_error:
            parameters, weights = previous
            reason = "increasing_error_previous_parameters_and_weights_restored"
            selected = iteration - 1
            break
        if error <= 2e-12 or error == previous_error:
            reason = "absolute_weighted_error" if error <= 2e-12 else "nondecreasing_error_equal"
            selected = iteration
            break
        previous = (parameters.clone(), weights.clone())
        previous_error = error
    else:
        reason, selected = "max20_budget_exhausted", 20
    return RegressionResult(parameters, weights, {
        "rows": int(design.shape[0]), "columns": int(design.shape[1]),
        "dtype": str(design.dtype), "solver": "torch_reduced_QR_triangular_solve",
        "iterations": rows, "selected_iteration": selected, "stop_reason": reason,
        "normal_equations": False, "ridge_or_svd_fallback": False,
    })


def analytic_system(source, target, *, mode):
    """Native z -> x -> y row order, raw outside-value=0 exclusion.

    Only nonnegative, one-frame 3D source-default mode is supported here;
    random subsampling/intensity scaling are deliberately not inferred.
    """
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("halfway inputs must be matching 3D grids")
    fx, fy, fz, ft = partials((source + target) * .5)
    b = blur(source - target)
    valid = ((source.abs() > 1e-5) & (target.abs() > 1e-5)
             & torch.isfinite(fx) & torch.isfinite(fy) & torch.isfinite(fz)
             & torch.isfinite(ft)
             & ((fx.abs() >= 1e-5) | (fy.abs() >= 1e-5) | (fz.abs() >= 1e-5)))
    # Avoid an XYZ coordinate volume; compact positions in source row order.
    selected = valid.permute(2, 0, 1).nonzero()
    if selected.shape[0] == 0:
        raise ValueError("no valid gradient rows in overlapping image support")
    z, x, y = (selected[:, i] for i in range(3))
    gx, gy, gz = fx[x, y, z], fy[x, y, z], fz[x, y, z]
    xf, yf, zf = x.float(), y.float(), z.float()
    if mode == "rigid":
        columns = (gx, gy, gz, gz * yf - gy * zf,
                   gx * zf - gz * xf, gy * xf - gx * yf)
    elif mode == "affine":
        columns = (gx * xf, gx * yf, gx * zf, gx,
                   gy * xf, gy * yf, gy * zf, gy,
                   gz * xf, gz * yf, gz * zf, gz)
    else:
        raise ValueError("mode must be rigid or affine")
    return torch.stack(columns, dim=1), b[x, y, z], {
        "valid_rows": int(selected.shape[0]), "row_order": "z,x,y",
        "outside_value": 0, "gradient_and_outside_threshold": 1e-5,
    }


def parameters_to_matrix(parameters, *, mode):
    p = parameters.detach().double().cpu().numpy()
    matrix = np.eye(4)
    if mode == "affine":
        if p.shape != (12,):
            raise ValueError("affine step requires12 parameters")
        matrix[:3] += p.reshape(3, 4)
    elif mode == "rigid":
        if p.shape != (6,):
            raise ValueError("rigid step requires6 parameters")
        length = float(np.linalg.norm(p[3:]))
        if length != 0:
            axis = p[3:] / length
            # Source Quaternion.importRotVec/getRotMatrix3d Double state.
            q0 = np.cos(length * .5)
            q1, q2, q3 = np.sin(length * .5) * axis
            matrix[:3, :3] = (
                (1 - 2 * (q2*q2 + q3*q3), 2*(q1*q2 - q3*q0), 2*(q1*q3 + q2*q0)),
                (2*(q1*q2 + q3*q0), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q1*q0)),
                (2*(q1*q3 - q2*q0), 2*(q2*q3 + q1*q0), 1 - 2*(q1*q1 + q2*q2)),
            )
        matrix[:3, 3] = p[:3]
    else:
        raise ValueError("mode must be rigid or affine")
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError("step produced reflection, projection or nonfinite state")
    return matrix


def transform_distance(new, previous):
    delta = np.asarray(new, float) - np.asarray(previous, float)
    return float(np.sqrt(2000 * np.sum(delta[:3, :3] ** 2)
                         + np.sum(delta[:3, 3] ** 2)))
