"""Private CPU first-step diagnostic; no recipe or production integration.

Pass a pure ``closure(points) -> (cost, gradient)`` using a saved shared state.
A stateful closure must supply checkpoint()/restore(snapshot), covering ALL its
mutable state. Both search runners restore that snapshot and return the proposed
point as data. Evaluations/trial arrays stay in the explicitly supplied private
directory. This is a source-informed reference prototype, not a native oracle.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat

import numpy as np
import torch


class EvaluationBudget(Exception):
    pass


def max_node_norm(value):
    return float(value.double().reshape(-1, 3).norm(dim=1).amax())


def finite_number(value):
    return value if math.isfinite(value) else None


def float32_abs(value):
    # Pinned upstream Optimizer.cxx:433 (bracket), :775 (zoom interval).
    with np.errstate(over='ignore'):
        return float(abs(np.float32(value)))


@contextmanager
def preserve_closure_state(closure):
    checkpoint = getattr(closure, 'checkpoint', None)
    restore = getattr(closure, 'restore', None)
    if bool(checkpoint) != bool(restore):
        raise ValueError('stateful closure requires both checkpoint and restore')
    snapshot = checkpoint() if checkpoint else None
    try:
        yield
    finally:
        if restore:
            restore(snapshot)


@dataclass
class Evaluation:
    point: torch.Tensor
    cost: float
    gradient: torch.Tensor
    alpha: float
    directional: float


class Recorder:
    def __init__(self, start, closure, directory, budget, c1, c2, applied_gradient_dtype=None):
        if start.device.type != 'cpu' or start.dtype not in (torch.float32, torch.float64):
            raise ValueError('CPU FP32/FP64 points required')
        if start.ndim != 2 or start.shape[1] != 3 or not torch.isfinite(start).all():
            raise ValueError('finite N-by-3 mesh points required')
        if isinstance(budget, bool) or budget < 1:
            raise ValueError('budget includes the initial evaluation and must be >=1')
        self.start = start.detach().clone()
        self.closure = closure
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.directory.chmod(0o700)
        if stat.S_IMODE(self.directory.stat().st_mode) != 0o700:
            raise PermissionError('trial directory must actually have mode0700; choose a filesystem supporting Unix permissions')
        self.budget = int(budget)
        self.c1, self.c2 = c1, c2
        self.records = []
        self.direction = None
        self.initial = None
        self.calls = 0
        self.applied_gradient_dtype = applied_gradient_dtype

    def evaluate(self, proposed, alpha, phase, alpha_is_estimated=False):
        if self.calls >= self.budget:
            raise EvaluationBudget()
        actual = proposed.detach().to(dtype=self.start.dtype).clone()
        supplied = actual.clone()
        self.calls += 1
        path = self.directory / f'evaluation-{self.calls:03d}.npz'
        np.savez_compressed(path, points=actual.numpy())
        try:
            cost, gradient = self.closure(supplied)
        except BaseException as error:
            failed = {'evaluation': self.calls, 'phase': phase, 'alpha': float(alpha),
                      'point_dtype': str(actual.dtype), 'private_arrays': path.name,
                      'closure_exception_type': type(error).__name__}
            self.records.append(failed)
            (self.directory / 'failure.public.json').write_text(json.dumps(failed, indent=2) + '\n')
            raise
        if not torch.equal(supplied.detach(), actual):
            raise ValueError('closure mutated the supplied point values')
        if torch.is_tensor(cost) and (cost.device.type != 'cpu' or cost.numel() != 1):
            raise ValueError('CPU scalar cost required; device transfers are not allowed in this diagnostic')
        if not torch.is_tensor(gradient) or gradient.device.type != 'cpu':
            raise ValueError('CPU gradient required; device transfers are not allowed in this diagnostic')
        gradient = gradient.detach().clone()
        closure_gradient_dtype = str(gradient.dtype)
        if gradient.shape != actual.shape or gradient.dtype not in (torch.float32, torch.float64):
            raise ValueError('closure gradient must match point shape and be FP32/FP64')
        if self.applied_gradient_dtype is not None:
            gradient = gradient.to(dtype=self.applied_gradient_dtype)
        cost = float(cost.detach()) if torch.is_tensor(cost) else float(cost)
        directional = (float((gradient.double() * self.direction).sum())
                       if self.direction is not None else 0.)
        value = Evaluation(actual, cost, gradient, float(alpha), directional)
        if self.initial is None:
            self.initial = value
        initial_derivative = (float((self.initial.gradient.double() * self.direction).sum())
                              if self.direction is not None else None)
        actual_displacement = actual.double() - self.start.double()
        actual_product = float((self.initial.gradient.double() * actual_displacement).sum())
        source_rhs = (self.initial.cost + self.c1 * alpha * initial_derivative
                      if initial_derivative is not None else None)
        actual_rhs = self.initial.cost + self.c1 * actual_product
        bound = -self.c2 * initial_derivative if initial_derivative is not None else None
        grad_finite = bool(torch.isfinite(gradient).all())
        record = {'evaluation': self.calls, 'phase': phase, 'alpha': float(alpha),
                  'alpha_is_estimated_from_actual_fp32_displacement': alpha_is_estimated,
                  'point_dtype': str(actual.dtype), 'gradient_dtype': str(gradient.dtype),
                  'closure_gradient_dtype': closure_gradient_dtype,
                  'cost': finite_number(cost), 'cost_nonfinite': not math.isfinite(cost),
                  'gradient_finite': grad_finite,
                  'directional_derivative': finite_number(directional),
                  'initial_directional_derivative': finite_number(initial_derivative) if initial_derivative is not None else None,
                  'actual_initial_gradient_dot_displacement': finite_number(actual_product),
                  'source_armijo_rhs': finite_number(source_rhs) if source_rhs is not None else None,
                  'actual_displacement_armijo_rhs': finite_number(actual_rhs),
                  'strong_wolfe_curvature_bound': finite_number(bound) if bound is not None else None,
                  'max_actual_node_displacement': max_node_norm(actual_displacement),
                  'point_sha256': hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
                  'point_shape': list(actual.shape)}
        np.savez_compressed(path, points=actual.numpy(), gradient=gradient.numpy())
        record['private_arrays'] = path.name
        self.records.append(record)
        return value

    def finalize(self, method, reason, chosen, extra=None):
        initial = self.initial
        if initial is None:
            raise RuntimeError('no initial evaluation available')
        selected = chosen or initial
        displacement = selected.point.double() - self.start.double()
        moved = bool(torch.any(displacement != 0))
        source_d0 = (float((initial.gradient.double() * self.direction).sum())
                     if self.direction is not None else None)
        finite = math.isfinite(selected.cost) and bool(torch.isfinite(selected.gradient).all())
        source_armijo = (finite and source_d0 is not None and
                         selected.cost <= initial.cost + self.c1 * selected.alpha * source_d0)
        selected_directional = (float((selected.gradient.double() * self.direction).sum())
                                if self.direction is not None else None)
        curvature_abs = (float32_abs(selected_directional) if reason == 'strong_wolfe_bracket' else
                         abs(selected_directional)) if selected_directional is not None else None
        curvature = (finite and source_d0 is not None and curvature_abs <= -self.c2 * source_d0)
        np.savez_compressed(self.directory / 'result.private.npz', start_points=self.start.numpy(),
                            accepted_points=selected.point.numpy(), accepted_gradient=selected.gradient.numpy(),
                            direction=(self.direction.numpy() if self.direction is not None else np.zeros(self.start.shape)))
        report = {'scope': 'One CPU search step only; closure snapshot restored; no benchmark or native equivalence claim',
                  'method': method, 'reason': reason, 'evaluations': self.calls,
                  'evaluation_budget': self.budget, 'point_moved': moved,
                  'initial_cost': finite_number(initial.cost), 'selected_cost': finite_number(selected.cost),
                  'selected_alpha': selected.alpha, 'selected_max_node_displacement': max_node_norm(displacement),
                  'selected_source_armijo_satisfied': bool(source_armijo),
                  'selected_strong_wolfe_satisfied': bool(source_armijo and curvature),
                  'selected_directional_derivative': finite_number(selected_directional) if selected_directional is not None else None,
                  'selected_curvature_abs_precision': 'float32' if reason == 'strong_wolfe_bracket' else 'float64',
                  'source_initial_directional_derivative': source_d0,
                  'records': self.records, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  **(extra or {})}
        (self.directory / 'trace.public.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return selected.point.clone(), report


def cubic_candidate(left, right):
    """Cubic proposal with alpha-ascending endpoints; invalid -> None.

    The fixed source sorts zoom endpoints by alpha (Optimizer.cxx:611-629),
    then uses the nonnegative sqrt at :648. It does not sign d2 by low/high
    ordering. Bracket expansion already supplies alpha-ascending endpoints.
    """
    if left.alpha > right.alpha:
        raise ValueError('cubic endpoints must be in increasing alpha order')
    width = left.alpha - right.alpha
    if width == 0 or not all(math.isfinite(x) for x in (
            left.cost, right.cost, left.directional, right.directional)):
        return None
    d1 = left.directional + right.directional - 3 * (left.cost - right.cost) / width
    d2_squared = d1 * d1 - left.directional * right.directional
    if d2_squared <= 0 or not math.isfinite(d2_squared):
        return None
    d2 = math.sqrt(d2_squared)
    denominator = right.directional - left.directional + 2 * d2
    if denominator == 0:
        return None
    alpha = right.alpha - (right.alpha - left.alpha) * (right.directional + d2 - d1) / denominator
    return alpha if math.isfinite(alpha) else None


def reference_first_step(start, closure, directory, *, budget=64, start_distance=1.,
                         max_search_displacement=50., zoom_interval_stop=.05,
                         c1=1e-4, c2=.9):
    """Pinned-source initial scaling, expansion and strong-Wolfe/zoom diagnostic.

    Source fallback returns can fail Wolfe; report that explicitly. The added
    diagnostic evaluation budget always returns the initial point on exhaustion.
    Degenerate/nonfinite cubic proposals use bisection, a diagnostic safeguard.
    No curvature history is needed for the initial step; this is not a full
    L-BFGS optimizer. Native float/cubic rounding still needs an actual oracle.
    """
    if not (start_distance > 0 and max_search_displacement > 0 and zoom_interval_stop > 0
            and 0 < c1 < c2 < 1):
        raise ValueError('invalid reference diagnostic options')
    recorder = Recorder(start, closure, directory, budget, c1, c2)
    with preserve_closure_state(closure):
        initial = recorder.evaluate(start, 0., 'initial')
        if not math.isfinite(initial.cost) or not torch.isfinite(initial.gradient).all():
            return recorder.finalize('reference-first-step', 'nonfinite_initial', initial)
        norm = max_node_norm(initial.gradient)
        if norm == 0:
            return recorder.finalize('reference-first-step', 'zero_gradient', initial)
        direction = -initial.gradient.double() * (start_distance / norm)
        recorder.direction = direction
        d0 = float((initial.gradient.double() * direction).sum())
        initial.directional = d0
        alpha_max = max_search_displacement / max_node_norm(direction)
        alpha = min(1., alpha_max)
        previous = initial
        low = high = None
        chosen = initial
        try:
            for i in range(10):
                trial = recorder.evaluate(start.double() + alpha * direction, alpha, 'bracket')
                finite = math.isfinite(trial.cost) and bool(torch.isfinite(trial.gradient).all())
                armijo_rhs = initial.cost + c1 * alpha * d0
                if not finite or trial.cost > armijo_rhs or (i > 0 and trial.cost >= previous.cost):
                    low, high = previous, trial
                    break
                if float32_abs(trial.directional) <= -c2 * d0:
                    return recorder.finalize('reference-first-step', 'strong_wolfe_bracket', trial)
                if trial.directional > 0:
                    low, high = trial, previous
                    break
                if alpha >= alpha_max:
                    previous = trial
                    break
                lower = math.exp(math.log(alpha) + (math.log(alpha_max) - math.log(alpha)) / (10 - i))
                suggestion = cubic_candidate(previous, trial)
                next_alpha = lower if suggestion is None else min(max(suggestion, lower), alpha_max)
                previous, alpha = trial, next_alpha
            if low is None:
                return recorder.finalize('reference-first-step', 'reference_bracketing_fallback', previous)
            while True:
                # Source :611-629 sorts endpoints before the cubic even when
                # lowAlpha > highAlpha after an uphill bracket trial.
                left, right = (low, high) if low.alpha < high.alpha else (high, low)
                proposal = cubic_candidate(left, right)
                alpha = (left.alpha + right.alpha) / 2 if proposal is None else proposal
                margin = .1 * (right.alpha - left.alpha)
                alpha = min(max(alpha, left.alpha + margin), right.alpha - margin)
                trial = recorder.evaluate(start.double() + alpha * direction, alpha, 'zoom')
                finite = math.isfinite(trial.cost) and bool(torch.isfinite(trial.gradient).all())
                if not finite or trial.cost > initial.cost + c1 * alpha * d0 or trial.cost >= low.cost:
                    high = trial
                else:
                    if abs(trial.directional) <= -c2 * d0:
                        return recorder.finalize('reference-first-step', 'strong_wolfe_zoom', trial)
                    if trial.directional * (high.alpha - low.alpha) >= 0:
                        high = low
                    low = trial
                if float32_abs(high.alpha - low.alpha) * max_node_norm(direction) < zoom_interval_stop:
                    return recorder.finalize('reference-first-step', 'reference_zoom_interval_fallback', low)
        except EvaluationBudget:
            chosen = initial
            return recorder.finalize('reference-first-step', 'diagnostic_budget_exhausted_restore_initial', chosen)


def existing_armijo_first_step(start, closure, directory, optimizer_source, *, budget=64):
    """Replay exactly one existing FNIT Armijo step, with all evaluated points."""
    spec = importlib.util.spec_from_file_location('private_fnit_optimizer', optimizer_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = Recorder(start, closure, directory, budget, 1e-4, .9, applied_gradient_dtype=start.dtype)
    parameter = start.detach().clone().requires_grad_(True)
    optimizer = module.CachedArmijoLBFGS([parameter], lr=1., max_iter=1, history_size=12,
                                       tolerance_grad=1e-10, tolerance_change=1e-10,
                                       line_search_fn='strong_wolfe', double_precision_state=True)

    def wrapped():
        state = optimizer.state[parameter]
        direction = state.get('d')
        alpha = 0.
        if direction is not None:
            recorder.direction = direction.reshape_as(parameter).double().clone()
            delta = parameter.detach().double() - recorder.start.double()
            denominator = float(recorder.direction.square().sum())
            alpha = float((delta * recorder.direction).sum()) / denominator if denominator else 0.
        value = recorder.evaluate(parameter.detach(), alpha,
                                  'initial' if recorder.initial is None else 'existing-armijo',
                                  alpha_is_estimated=recorder.initial is not None)
        parameter.grad = value.gradient.to(parameter).clone()
        return parameter.new_tensor(value.cost, dtype=torch.float64)

    with preserve_closure_state(closure):
        try:
            optimizer.step(wrapped, cache_key='private-first-step')
            chosen = next((record for record in reversed(recorder.records)
                           if record['point_sha256'] == hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()), None)
            if chosen is None:
                raise RuntimeError('existing optimizer selected an unevaluated point')
            arrays = np.load(recorder.directory / chosen['private_arrays'])
            value = Evaluation(torch.from_numpy(arrays['points'].copy()),
                               float(chosen['cost']) if chosen['cost'] is not None else float('inf'),
                               torch.from_numpy(arrays['gradient'].copy()), chosen['alpha'],
                               chosen['directional_derivative'] or 0.)
            return recorder.finalize('existing-fnit-armijo', 'existing_optimizer_returned', value,
                                     {'optimizer_source_sha256': hashlib.sha256(Path(optimizer_source).read_bytes()).hexdigest(),
                                      'alpha_note': 'Estimated from recorded FP32 point displacement; existing acceptance uses the exact actual gradient dot displacement.'})
        except EvaluationBudget:
            return recorder.finalize('existing-fnit-armijo', 'diagnostic_budget_exhausted_restore_initial', recorder.initial)
