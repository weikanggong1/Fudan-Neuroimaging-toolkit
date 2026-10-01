"""Reuse accepted L-BFGS evaluations while the GEMS objective stays fixed."""

from __future__ import annotations

import torch
from math import isfinite


class PrecisionLBFGS(torch.optim.LBFGS):
    """L-BFGS with optional FP64 arithmetic for its internal optimization state.

    ``double_precision_state`` promotes the flattened working gradient, so the
    inherited curvature products, search direction and Wolfe scalar arithmetic
    retain cost differences without changing parameter or parameter-grad dtype.
    The default follows PyTorch's parameter precision.
    """

    def __init__(self, params, *, double_precision_state: bool = False, **kwargs):
        if not isinstance(double_precision_state, bool):
            raise ValueError("double_precision_state must be a bool")
        self.double_precision_state = double_precision_state
        super().__init__(params, **kwargs)

    def _gather_flat_grad(self):
        gradient = super()._gather_flat_grad()
        return gradient.to(torch.float64) if self.double_precision_state else gradient

    def reset_history(self):
        """Restart the next step from normalized steepest descent.

        Keep parameters, their gradients and accumulated evaluation counts.
        A subclass's accepted cost/gradient cache remains available subject to
        its usual parameter-version and objective-key checks.
        """
        state = self.state.get(self._params[0])
        if state is not None:
            evaluations = state.get("func_evals", 0)
            state.clear()
            state["func_evals"] = evaluations


class CachedLBFGS(PrecisionLBFGS):
    """PyTorch strong-Wolfe L-BFGS with one iteration per ``step``.

    The accepted objective and closure-supplied gradient (including any boundary
    projection) replace the next step's initial evaluation. ``step`` retains
    PyTorch's return value: the objective before the update. Use
    ``accepted_objective`` for the cost after the update.

    Construct a new optimizer for each EM outer iteration. For other changes to
    the image, likelihood, alphas, reference or gradient projection, call
    ``invalidate_cache`` or change an immutable Python ``cache_key`` (for
    example, an integer objective revision) passed to ``step``.
    In-place parameter changes invalidate the cache automatically. The cache
    assumes a deterministic objective and does not retain its autograd graph.

    ``closure_evaluations`` and ``last_step_evaluations`` count actual closure
    executions; PyTorch's ``state['func_evals']`` also counts cache hits.
    """

    def __init__(self, params, **kwargs):
        kwargs.setdefault("max_iter", 1)
        kwargs.setdefault("line_search_fn", "strong_wolfe")
        if kwargs["max_iter"] != 1 or kwargs["line_search_fn"] != "strong_wolfe":
            raise ValueError("CachedLBFGS requires max_iter=1 and strong_wolfe")
        super().__init__(params, **kwargs)
        self.closure_evaluations = 0
        self.last_step_evaluations = 0
        self.cache_hits = 0
        self.accepted_objective: torch.Tensor | None = None
        self._cached = None
        self._versions = None
        self._cache_key = None
        self._trials = {}
        self._last_objective = None

    def invalidate_cache(self):
        """Discard evaluated cost/gradient; leave the L-BFGS history unchanged."""
        self._cached = None
        self._versions = None
        self.accepted_objective = None

    def _directional_evaluate(self, closure, x, t, d):
        loss, gradient = super()._directional_evaluate(closure, x, t, d)
        # Wolfe returns the selected trial's step object. Keep it alive and use
        # identity so a CUDA scalar step does not add another host read per trial.
        self._trials[id(t)] = (t, self._last_objective, gradient.detach())
        return loss, gradient

    @torch.no_grad()
    def step(self, closure, *, cache_key=None):
        versions = tuple(parameter._version for parameter in self._params)
        if cache_key != self._cache_key or versions != self._versions:
            self.invalidate_cache()
        self._cache_key = cache_key
        self._trials = {}
        first = True
        initial = None
        before = self.closure_evaluations

        def evaluate():
            nonlocal first, initial
            if first and self._cached is not None:
                objective, gradient = self._cached
                offset = 0
                for parameter in self._params:
                    count = parameter.numel()
                    parameter.grad = (gradient[offset:offset + count].view_as(parameter)
                                      .to(dtype=parameter.dtype).clone())
                    offset += count
                initial = self._cached
                self.cache_hits += 1
            else:
                self.closure_evaluations += 1
                objective = closure()
                if first:
                    initial = (objective.detach(), self._gather_flat_grad().detach())
            first = False
            self._last_objective = objective.detach()
            return objective

        try:
            result = super().step(evaluate)
            if self._trials:
                accepted_step = self.state[self._params[0]]["t"]
                selected = self._trials.get(id(accepted_step))
                self._cached = (selected[1:] if selected is not None else
                                initial if float(accepted_step) == 0. else None)
            else:
                self._cached = initial
            self.accepted_objective = None if self._cached is None else self._cached[0]
            self._versions = tuple(parameter._version for parameter in self._params)
            return result
        except BaseException:
            self.invalidate_cache()
            raise
        finally:
            self.last_step_evaluations = self.closure_evaluations - before
            self._trials = {}
            self._last_objective = None


class CachedArmijoLBFGS(CachedLBFGS):
    """Bounded L-BFGS updates with FP64 state and accepted-point caching.

    The standard two-loop recursion uses at most ``history_size`` curvature
    pairs (12 by default). Each trial is written to the native parameter dtype;
    its actual displacement supplies both the Armijo directional product and
    the accepted curvature pair. Backtracking accepts only finite cost/gradient
    evaluations with strict cost decrease and sufficient Armijo decrease.

    ``max_vertex_displacement`` bounds the Euclidean displacement of each row
    for a single N-by-3 mesh parameter. Other parameter shapes use a component
    bound. This class retains ``CachedLBFGS``'s cache/counter API, while its
    ``step`` implements Armijo search rather than the inherited Wolfe search.
    """

    def __init__(self, params, *, max_vertex_displacement: float = .5,
                 max_backtracking_steps: int = 20, armijo_constant: float = 1e-4,
                 **kwargs):
        if (not isfinite(max_vertex_displacement)
                or max_vertex_displacement <= 0):
            raise ValueError("max_vertex_displacement must be positive and finite")
        if (isinstance(max_backtracking_steps, bool)
                or not isinstance(max_backtracking_steps, int)
                or max_backtracking_steps < 1):
            raise ValueError("max_backtracking_steps must be a positive integer")
        if not isfinite(armijo_constant) or not 0 < armijo_constant < 1:
            raise ValueError("armijo_constant must be between zero and one")
        if not kwargs.get("double_precision_state", True):
            raise ValueError("CachedArmijoLBFGS requires double_precision_state=True")
        kwargs["double_precision_state"] = True
        kwargs.setdefault("history_size", 12)
        super().__init__(params, **kwargs)
        self.max_vertex_displacement = float(max_vertex_displacement)
        self.max_backtracking_steps = max_backtracking_steps
        self.armijo_constant = float(armijo_constant)

    def invalidate_cache(self):
        # Curvature from a different likelihood or externally changed position
        # does not describe the next objective.
        super().invalidate_cache()
        self.reset_history()

    def _flat_parameters(self):
        return torch.cat([parameter.detach().reshape(-1).double()
                          for parameter in self._params])

    def _write_parameters(self, position):
        offset = 0
        for parameter in self._params:
            count = parameter.numel()
            parameter.copy_(position[offset:offset + count].view_as(parameter))
            offset += count

    def _write_gradient(self, gradient):
        offset = 0
        for parameter in self._params:
            count = parameter.numel()
            parameter.grad = (gradient[offset:offset + count].view_as(parameter)
                              .to(dtype=parameter.dtype).clone())
            offset += count

    def _max_displacement(self, displacement):
        if (len(self._params) == 1 and self._params[0].ndim == 2
                and self._params[0].shape[1] == 3):
            return displacement.view(-1, 3).norm(dim=1).amax()
        return displacement.abs().amax()

    @torch.no_grad()
    def step(self, closure, *, cache_key=None):
        versions = tuple(parameter._version for parameter in self._params)
        if cache_key != self._cache_key or versions != self._versions:
            self.invalidate_cache()
        self._cache_key = cache_key
        before = self.closure_evaluations
        state = self.state[self._params[0]]
        initial_position = self._flat_parameters()
        initial = self._cached
        closure = torch.enable_grad()(closure)

        def evaluate():
            self.closure_evaluations += 1
            state["func_evals"] = state.get("func_evals", 0) + 1
            objective = closure().detach().double().clone()
            gradient = self._gather_flat_grad().detach().clone()
            return objective, gradient

        def retain(position, evaluation):
            self._write_parameters(position)
            self._write_gradient(evaluation[1])
            self._cached = evaluation
            self.accepted_objective = evaluation[0]
            self._versions = tuple(parameter._version for parameter in self._params)

        try:
            if initial is None:
                initial = evaluate()
            else:
                self.cache_hits += 1
                state["func_evals"] = state.get("func_evals", 0) + 1
                self._write_gradient(initial[1])
            objective, gradient = initial
            state["n_iter"] = state.get("n_iter", 0) + 1
            state["prev_flat_grad"] = gradient.clone()
            state["prev_loss"] = float(objective)
            state["t"] = 0.
            if not bool(torch.isfinite(objective) & torch.isfinite(gradient).all()):
                self.invalidate_cache()
                return objective

            displacements = state.setdefault("old_stps", [])
            gradient_changes = state.setdefault("old_dirs", [])
            inverse_curvatures = state.setdefault("ro", [])
            working = gradient.clone()
            coefficients = []
            for s, y, rho in zip(reversed(displacements), reversed(gradient_changes),
                                 reversed(inverse_curvatures)):
                coefficient = rho * s.dot(working)
                coefficients.append(coefficient)
                working.sub_(coefficient * y)
            direction = working * state.get("H_diag", 1.)
            for s, y, rho, coefficient in zip(displacements, gradient_changes,
                                              inverse_curvatures, reversed(coefficients)):
                direction.add_(s * (coefficient - rho * y.dot(direction)))
            direction.neg_()
            if not bool(torch.isfinite(direction).all() & (gradient.dot(direction) < 0)):
                displacements.clear()
                gradient_changes.clear()
                inverse_curvatures.clear()
                state["H_diag"] = 1.
                direction = -gradient
            state["d"] = direction.clone()
            if gradient.abs().max() <= self.param_groups[0]["tolerance_grad"]:
                retain(initial_position, initial)
                return objective

            largest = float(self._max_displacement(direction))
            trial_step = min(float(self.param_groups[0]["lr"]),
                             self.max_vertex_displacement / largest)
            for _ in range(self.max_backtracking_steps):
                self._write_parameters(initial_position + trial_step * direction)
                actual_delta = self._flat_parameters() - initial_position
                directional_change = gradient.dot(actual_delta)
                if not bool(directional_change < 0):
                    break
                if self._max_displacement(actual_delta) > self.max_vertex_displacement:
                    trial_step *= .5
                    continue
                trial = evaluate()
                trial_cost, trial_gradient = trial
                acceptable = (torch.isfinite(trial_cost) & torch.isfinite(trial_gradient).all()
                              & (trial_cost < objective)
                              & (trial_cost <= objective
                                 + self.armijo_constant * directional_change))
                if bool(acceptable):
                    gradient_change = trial_gradient - gradient
                    curvature = actual_delta.dot(gradient_change)
                    if curvature > 1e-12 * actual_delta.norm() * gradient_change.norm():
                        displacements.append(actual_delta.clone())
                        gradient_changes.append(gradient_change.clone())
                        inverse_curvatures.append(curvature.reciprocal())
                        if len(displacements) > self.param_groups[0]["history_size"]:
                            displacements.pop(0)
                            gradient_changes.pop(0)
                            inverse_curvatures.pop(0)
                        state["H_diag"] = curvature / gradient_change.dot(gradient_change)
                    state["t"] = trial_step
                    # Do not rewrite an accepted point: cache versions describe
                    # exactly the native-dtype parameters just evaluated.
                    self._write_gradient(trial_gradient)
                    self._cached = trial
                    self.accepted_objective = trial_cost
                    self._versions = tuple(parameter._version for parameter in self._params)
                    return objective
                trial_step *= .5
            retain(initial_position, initial)
            return objective
        except BaseException:
            self._write_parameters(initial_position)
            if initial is not None:
                retain(initial_position, initial)
            else:
                self.invalidate_cache()
            raise
        finally:
            self.last_step_evaluations = self.closure_evaluations - before
