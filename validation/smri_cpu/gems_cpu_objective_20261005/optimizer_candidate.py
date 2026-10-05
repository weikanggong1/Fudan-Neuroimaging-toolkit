"""Experimental CPU GEMS L-BFGS definition; not selected by FNIT production.

The upstream recipe uses max-deformation scaling, twelve newest-first curvature
pairs and strong-Wolfe bracketing/zoom. All trial points retain caller precision.
No CUDA input or implicit device transfer is accepted. A diagnostic budget is
reported separately from algorithmic convergence and returns the previous point.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


def node_max(value):
    return float(torch.linalg.vector_norm(value.double(), dim=1).amax())


def inner(left, right):
    # CPU vector reduction is deterministic within the declared thread budget.
    # It is not asserted bitwise identical to upstream sequential accumulation.
    return float((left.double() * right.double()).sum())


def f32_abs(value):
    with np.errstate(over='ignore'):
        return float(abs(np.float32(value)))


class DiagnosticBudget(Exception):
    pass


@dataclass
class Value:
    points: torch.Tensor
    cost: float
    gradient: torch.Tensor
    alpha: float = 0.
    derivative: float = 0.


def cubic(left, right):
    if left.alpha > right.alpha:
        raise ValueError('cubic endpoints must be alpha sorted')
    width = left.alpha - right.alpha
    if width == 0 or not all(math.isfinite(x) for x in (
            left.cost, right.cost, left.derivative, right.derivative)):
        return None
    d1 = left.derivative + right.derivative - 3 * (left.cost-right.cost) / width
    d2_square = d1*d1 - left.derivative*right.derivative
    if d2_square < 0 or not math.isfinite(d2_square):
        return None
    d2 = math.sqrt(d2_square)
    denominator = right.derivative-left.derivative+2*d2
    if denominator == 0:
        return None
    result = right.alpha - (right.alpha-left.alpha)*(right.derivative+d2-d1)/denominator
    return result if math.isfinite(result) else None


class NativeDefinitionCPU:
    """Private optimization state on CPU; step returns points/cost/trace as data.

    ``closure(points)`` must return a scalar CPU cost and N×3 CPU gradient. A
    trial observer may save private points but cannot modify them. This class
    does not implement EM, change mesh masks or update caller-owned parameters.
    """
    def __init__(self, points, closure, *, memory_length=12,
                 maximal_deformation_stop=1e-10, interval_stop=1e-10,
                 max_search_displacement=50., maximum_iterations=1000,
                 evaluations_per_step=64, observer=None):
        if (not torch.is_tensor(points) or points.device.type != 'cpu'
                or points.dtype not in (torch.float32, torch.float64)
                or points.ndim != 2 or points.shape[1] != 3
                or not bool(torch.isfinite(points).all())):
            raise ValueError('finite CPU FP32/FP64 N×3 points required')
        if (isinstance(memory_length,bool) or not isinstance(memory_length,int)
                or memory_length < 1 or maximum_iterations < 1 or evaluations_per_step < 2):
            raise ValueError('positive memory/iterations and at least two evaluations required')
        if (not math.isfinite(maximal_deformation_stop) or maximal_deformation_stop < 0
                or not math.isfinite(interval_stop) or interval_stop <= 0
                or not math.isfinite(max_search_displacement) or max_search_displacement <= 0):
            raise ValueError('finite valid deformation criteria required')
        self.points = points.detach().clone()
        self.closure, self.observer = closure, observer
        self.memory_length = memory_length
        self.stop, self.interval_stop = maximal_deformation_stop, interval_stop
        self.max_search = max_search_displacement
        self.maximum_iterations, self.evaluations_per_step = maximum_iterations, evaluations_per_step
        self.history = []
        self.old_gradient = self.old_direction = None
        self.old_alpha = 0.
        self.current = None
        self.iteration = self.evaluations = 0
        self.finished = False

    def _evaluate(self, point, alpha, direction, phase, records):
        if len(records) >= self.evaluations_per_step:
            raise DiagnosticBudget()
        actual = point.detach().to(dtype=self.points.dtype).clone()
        supplied = actual.clone()
        cost, gradient = self.closure(supplied)
        if not torch.equal(actual, supplied):
            raise RuntimeError('closure modified supplied points')
        if torch.is_tensor(cost) and (cost.device.type != 'cpu' or cost.numel()!=1):
            raise ValueError('CPU scalar cost required')
        if (not torch.is_tensor(gradient) or gradient.device.type!='cpu'
                or gradient.shape!=actual.shape or gradient.dtype not in (torch.float32,torch.float64)):
            raise ValueError('matching FP32/FP64 CPU gradient required')
        cost = float(cost.detach()) if torch.is_tensor(cost) else float(cost)
        gradient = gradient.detach().double().clone()
        derivative = inner(gradient,direction) if direction is not None else 0.
        value = Value(actual,cost,gradient,float(alpha),derivative)
        record = {'alpha':float(alpha),'phase':phase,'cost':cost if math.isfinite(cost) else None,
            'gradient_finite':bool(torch.isfinite(gradient).all()),
            'directional_derivative':derivative if math.isfinite(derivative) else None}
        records.append(record)
        self.evaluations += 1
        if self.observer:
            self.observer(self.iteration,len(records),value,dict(record))
        return value

    def _direction(self):
        gradient = self.current.gradient
        gamma = 0.
        curvature = None
        if self.old_direction is None:
            norm = node_max(gradient)
            if norm == 0:
                return torch.zeros_like(gradient),gamma,curvature
            gamma = 1./norm
        else:
            # Upstream uses alpha*direction, not rounded parameter displacement.
            s = self.old_alpha*self.old_direction
            y = gradient-self.old_gradient
            curvature = inner(s,y)
            if curvature > 1e-10:
                self.history.insert(0,(s,y,curvature))
                self.history = self.history[:self.memory_length]
                gamma = curvature/inner(y,y)
            # Preserve upstream's gamma=0 branch if curvature was rejected.
        q = gradient.clone()
        coefficients = []
        for s,y,sy in self.history:
            coefficient = inner(s,q)/sy
            coefficients.append(coefficient)
            q = q-coefficient*y
        r = gamma*q
        for (s,y,sy),coefficient in zip(reversed(self.history),reversed(coefficients)):
            beta = inner(y,r)/sy
            r = r+s*(coefficient-beta)
        return -r,gamma,curvature

    def _search(self,initial,direction,records):
        initial.derivative = inner(initial.gradient,direction)
        d0 = initial.derivative
        norm = node_max(direction)
        if norm==0 or not math.isfinite(d0) or d0>=0:
            return initial,'non_descent_direction',0.
        alpha_max = self.max_search/norm
        alpha = min(1.,alpha_max)
        previous = initial
        low = high = None
        for count in range(10):
            trial = self._evaluate(initial.points.double()+alpha*direction,alpha,direction,'bracket',records)
            finite = math.isfinite(trial.cost) and bool(torch.isfinite(trial.gradient).all())
            if not finite or trial.cost > initial.cost+1e-4*alpha*d0 or (count>0 and trial.cost>=previous.cost):
                low,high = previous,trial
                break
            if f32_abs(trial.derivative)<=-.9*d0:
                return trial,'strong_wolfe_bracket',alpha
            if trial.derivative>0:
                low,high = trial,previous
                break
            if alpha>=alpha_max:
                previous=trial
                break
            minimum=math.exp(math.log(alpha)+(math.log(alpha_max)-math.log(alpha))/(10-count))
            proposal=cubic(previous,trial)
            next_alpha=minimum if proposal is None else min(max(proposal,minimum),alpha_max)
            previous,alpha=trial,next_alpha
        if low is None:
            return previous,'bracketing_fallback',previous.alpha
        while True:
            left,right=(low,high) if low.alpha<high.alpha else (high,low)
            alpha=cubic(left,right)
            if alpha is None:
                alpha=(left.alpha+right.alpha)/2
            margin=.1*(right.alpha-left.alpha)
            alpha=min(max(alpha,left.alpha+margin),right.alpha-margin)
            trial=self._evaluate(initial.points.double()+alpha*direction,alpha,direction,'zoom',records)
            finite=math.isfinite(trial.cost) and bool(torch.isfinite(trial.gradient).all())
            if not finite or trial.cost>initial.cost+1e-4*alpha*d0 or trial.cost>=low.cost:
                high=trial
            else:
                if abs(trial.derivative)<=-.9*d0:
                    return trial,'strong_wolfe_zoom',alpha
                if trial.derivative*(high.alpha-low.alpha)>=0:
                    high=low
                low=trial
            if f32_abs(high.alpha-low.alpha)*norm<self.interval_stop:
                return low,'interval_fallback',low.alpha

    def step(self):
        records=[]
        if self.finished or self.iteration>=self.maximum_iterations:
            self.finished=True
            return self.points.clone(), {'reason':'maximum_iterations_or_already_stopped','returned_maximal_deformation':0.}
        state_before=(list(self.history),self.old_gradient,self.old_direction,self.old_alpha)
        if self.current is None:
            self.current=self._evaluate(self.points,0.,None,'initial',records)
        if not math.isfinite(self.current.cost) or not torch.isfinite(self.current.gradient).all():
            self.finished=True
            return self.points.clone(),{'reason':'nonfinite_initial','returned_maximal_deformation':0.,'records':records}
        direction,gamma,curvature=self._direction()
        initial=self.current
        try:
            selected,reason,alpha=self._search(initial,direction,records)
        except DiagnosticBudget:
            self.history,self.old_gradient,self.old_direction,self.old_alpha=state_before
            return self.points.clone(), {'reason':'diagnostic_budget_exhausted','returned_maximal_deformation':0.,
                'algorithmic_convergence':False,'iteration':self.iteration,'records':records}
        self.old_gradient=initial.gradient.clone()
        self.old_direction=direction.clone()
        self.old_alpha=alpha
        self.current=selected
        self.points=selected.points.clone()
        reported_deformation=alpha*node_max(direction)
        actual=node_max(self.points.double()-initial.points.double())
        stopped=reason=='non_descent_direction' or (reported_deformation<=self.stop and (self.iteration>0 or reported_deformation==0.))
        returned=0. if stopped else reported_deformation
        self.finished=stopped
        trace={'reason':reason,'iteration':self.iteration,'initial_cost':initial.cost,'accepted_cost':selected.cost,
            'alpha':alpha,'hessian_initial_scale':gamma,'new_curvature_sy':curvature,
            'history_pairs':len(self.history),'theoretical_maximal_deformation':reported_deformation,
            'actual_maximal_deformation':actual,'returned_maximal_deformation':returned,
            'algorithmic_convergence':stopped,'records':records,'total_evaluations':self.evaluations}
        if not stopped:self.iteration+=1
        return self.points.clone(),trace
