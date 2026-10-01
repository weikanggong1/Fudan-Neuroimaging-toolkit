"""Matrix-free linear algebra for FNIRT's Gauss-Newton/LM updates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PCGReport:
    iterations: int
    converged: bool
    relative_residual: float


@dataclass(frozen=True)
class SCGReport:
    iterations: int
    accepted_iterations: int
    converged: bool
    cost: float
    lambda_final: float
    history: tuple[dict, ...]


def preconditioned_conjugate_gradient(
    matvec,
    rhs: torch.Tensor,
    *,
    diagonal: torch.Tensor | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 500,
    execution: str = "optimized",
):
    """Solve a symmetric positive-definite system using matrix-free PCG.

    FSL configures the FNIRT equation solver with a relative tolerance of
    ``1e-3`` and at most 500 iterations.  The stopping rule here uses the
    unpreconditioned residual norm relative to the right-hand-side norm.  This
    is the stopping rule in the IML++ ``CG`` routine used by FSL 6.0.7.4.
    """
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid PCG stopping parameters")
    if execution not in ("reference", "optimized"):
        raise ValueError("execution must be reference or optimized")
    if rhs.ndim != 1:
        raise ValueError("rhs must be a vector")
    if diagonal is None:
        inverse_diagonal = torch.ones_like(rhs)
    else:
        if diagonal.shape != rhs.shape:
            raise ValueError("diagonal and rhs must have the same shape")
        floor = torch.finfo(rhs.dtype).eps * diagonal.abs().mean().clamp_min(1)
        inverse_diagonal = diagonal.clamp_min(floor).reciprocal()

    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    if float(rhs_norm) == 0:
        return solution, PCGReport(0, True, 0.0)
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    rz = torch.dot(residual, preconditioned)

    relative = 1.0
    for iteration in range(1, max_iterations + 1):
        product = matvec(direction)
        denominator = torch.dot(direction, product)
        if rhs.is_cuda and execution == "optimized":
            # The denominator and convergence test both need host control.
            # Speculate the vector update, then transfer their two scalars once.
            # A failed denominator discards the candidate, exactly as above.
            alpha = rz / denominator
            candidate_solution = solution + alpha * direction
            candidate_residual = residual - alpha * product
            denominator_value, candidate_relative = torch.stack((
                denominator,
                torch.linalg.vector_norm(candidate_residual) / rhs_norm,
            )).detach().cpu().tolist()
            if not math.isfinite(denominator_value) or denominator_value <= 0:
                return solution, PCGReport(iteration - 1, False, relative)
            solution, residual = candidate_solution, candidate_residual
            relative = candidate_relative
        else:
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
                return solution, PCGReport(iteration - 1, False, relative)
            alpha = rz / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * product
            relative = float(torch.linalg.vector_norm(residual) / rhs_norm)
        if relative <= tolerance:
            return solution, PCGReport(iteration, True, relative)
        preconditioned = inverse_diagonal * residual
        new_rz = torch.dot(residual, preconditioned)
        beta = new_rz / rz.clamp_min(torch.finfo(rhs.dtype).tiny)
        direction = preconditioned + beta * direction
        rz = new_rz
    return solution, PCGReport(max_iterations, False, relative)


def scaled_conjugate_gradient(
    cost_function,
    gradient_function,
    initial: torch.Tensor,
    *,
    max_iterations: int,
    initial_lambda: float = 0.1,
    sigma: float = 1.0e-2,
    gradient_tolerance: float = 1.0e-8,
):
    """Port FSL MISCMATHS ``sccngr`` (Moller scaled CG).

    ``cost_function`` returns a scalar and ``gradient_function`` returns a
    vector with the same shape as ``initial``. The update order intentionally
    follows ``miscmaths/nonlin.cpp`` because rejected steps retain the previous
    finite-difference Hessian-vector product.
    """
    if initial.ndim != 1:
        raise ValueError("initial must be a vector")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if initial_lambda <= 0 or sigma <= 0 or gradient_tolerance <= 0:
        raise ValueError("SCG parameters must be positive")

    parameters = initial.clone()
    cost = torch.as_tensor(
        cost_function(parameters), device=parameters.device, dtype=parameters.dtype
    )
    residual = -gradient_function(parameters)
    direction = residual.clone()
    damping = parameters.new_tensor(initial_lambda)
    damping_bar = parameters.new_zeros(())
    second = torch.zeros_like(parameters)
    delta = parameters.new_zeros(())
    success = True
    accepted = 0
    converged = False
    history = []

    for iteration in range(1, max_iterations + 1):
        direction_norm2 = torch.dot(direction, direction)
        if not bool(torch.isfinite(direction_norm2)) or float(direction_norm2) == 0.0:
            converged = True
            break
        if success:
            sigma_k = parameters.new_tensor(sigma) / torch.sqrt(direction_norm2)
            second = (
                gradient_function(parameters + sigma_k * direction) + residual
            ) / sigma_k
            delta = torch.dot(direction, second)
        second = second + (damping - damping_bar) * direction
        delta = delta + (damping - damping_bar) * direction_norm2
        if float(delta) <= 0.0:
            second = second + (
                damping - 2.0 * (delta / direction_norm2)
            ) * direction
            damping_bar = 2.0 * (damping - delta / direction_norm2)
            delta = damping * direction_norm2 - delta
            damping = damping_bar.clone()

        mu = torch.dot(direction, residual)
        alpha = mu / delta
        candidate_parameters = parameters + alpha * direction
        candidate_cost = torch.as_tensor(
            cost_function(candidate_parameters),
            device=parameters.device,
            dtype=parameters.dtype,
        )
        comparison = (
            2.0 * delta * (cost - candidate_cost) / (mu * mu)
        )
        accepted_step = bool(torch.isfinite(comparison)) and float(comparison) >= 0.0
        if accepted_step:
            parameters = candidate_parameters
            cost = candidate_cost
            damping_bar = parameters.new_zeros(())
            success = True
            accepted += 1
            old_residual = residual
            residual = -gradient_function(parameters)
            if iteration % parameters.numel() == 0:
                direction = residual.clone()
            else:
                beta = (
                    torch.dot(residual, residual)
                    - torch.dot(old_residual, residual)
                ) / mu
                direction = residual + beta * direction
            if float(comparison) > 0.75:
                damping = damping / 2.0
        else:
            damping_bar = damping.clone()
            success = False
        if float(comparison) < 0.25:
            damping = 4.0 * damping

        relative_gradient = (
            residual.abs() * parameters.abs().clamp_min(1.0)
        ).max() / cost.clamp_min(1.0)
        history.append(
            {
                "iteration": iteration,
                "accepted": accepted_step,
                "cost": float(cost),
                "lambda": float(damping),
                "comparison": float(comparison),
                "relative_gradient": float(relative_gradient),
            }
        )
        if float(relative_gradient) < gradient_tolerance:
            converged = True
            break

    return parameters, SCGReport(
        iterations=len(history),
        accepted_iterations=accepted,
        converged=converged,
        cost=float(cost),
        lambda_final=float(damping),
        history=tuple(history),
    )


__all__ = [
    "PCGReport",
    "SCGReport",
    "preconditioned_conjugate_gradient",
    "scaled_conjugate_gradient",
]
