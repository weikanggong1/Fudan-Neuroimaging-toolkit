"""Reuse accepted L-BFGS evaluations while the GEMS objective stays fixed."""

from __future__ import annotations

import torch


class CachedLBFGS(torch.optim.LBFGS):
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
                    parameter.grad = gradient[offset:offset + count].view_as(parameter).clone()
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
