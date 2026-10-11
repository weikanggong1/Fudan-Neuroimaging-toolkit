"""完整单通道 T1 EM：有序 CPU L-BFGS、稳定样本和全样本两次对齐。

复用已有目标/梯度和 More–Thuente 步长更新；不执行原版程序。
VXL/Netlib 协议来源和真实验证见 GCA_COMPLETE_PYTHON.md。
"""
from __future__ import annotations

import math
import numpy as np
from .mri_em_register_optimizer import _Step, _mcstep


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    # 12维 Netlib DDOT：先处理余数，再按五项展开，保持标量双精度顺序。
    size = len(left)
    result = 0.0
    prefix = size % 5
    for i in range(prefix):
        result += float(left[i]) * float(right[i])
    for i in range(prefix, size, 5):
        for j in range(i, i + 5):
            result += float(left[j]) * float(right[j])
    return result


def _relative_change(before: float, after: float) -> float:
    # 源包装层fold/fnew/pold/pcurrent为float，不提升其减法/除法为double。
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.float32(np.float32(before)-np.float32(after))/np.float32(before))


def _matrix(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3] = parameters.astype(np.float32).reshape(3, 4)
    return result


def _line_search(evaluate, origin, value, gradient, direction, initial_step):
    derivative0 = sum(float(a) * float(b) for a, b in zip(gradient, direction))
    if derivative0 >= 0 or not math.isfinite(derivative0):
        return origin, value, gradient, 0.0, 'non_descent'
    state = _Step(fx=value, dx=derivative0, fy=value, dy=derivative0, p=initial_step)
    width, previous_width = 1e20 - 1e-20, 2 * (1e20 - 1e-20)
    first_stage, update_status = True, 1
    for evaluation in range(20):
        low, high = ((min(state.x, state.y), max(state.x, state.y)) if state.bracketed
                     else (state.x, state.p + 4 * (state.p - state.x)))
        state.p = min(1e20, max(1e-20, state.p))
        if ((state.bracketed and (state.p <= low or state.p >= high)) or evaluation >= 19
                or update_status == 0 or (state.bracketed and high - low <= 1e-16 * high)):
            state.p = state.x
        trial = origin + state.p * direction
        trial_value, trial_gradient = evaluate(trial)
        derivative = sum(float(a) * float(b) for a, b in zip(trial_gradient, direction))
        test = value + state.p * 1e-4 * derivative0
        reason = None
        if (state.bracketed and (state.p <= low or state.p >= high)) or update_status == 0:
            reason = 'rounding_or_invalid_bracket'
        if state.p == 1e20 and trial_value <= test and derivative <= 1e-4 * derivative0:
            reason = 'upper_step_bound'
        if state.p == 1e-20 and (trial_value > test or derivative >= 1e-4 * derivative0):
            reason = 'lower_step_bound'
        if evaluation >= 19:
            reason = 'twenty_trial_limit'
        if state.bracketed and high - low <= 1e-16 * high:
            reason = 'interval_tolerance'
        if trial_value <= test and abs(derivative) <= .1 * (-derivative0):
            reason = 'strong_wolfe'
        if reason:
            return trial, trial_value, trial_gradient, state.p, reason
        if first_stage and trial_value <= test and derivative >= 1e-4 * derivative0:
            first_stage = False
        if first_stage and trial_value <= state.fx and trial_value > test:
            test_derivative = 1e-4 * derivative0
            state.fx -= state.x * test_derivative
            state.fy -= state.y * test_derivative
            state.dx -= test_derivative
            state.dy -= test_derivative
            update_status = _mcstep(state, trial_value - state.p * test_derivative,
                                   derivative - test_derivative, low, high)
            state.fx += state.x * test_derivative
            state.fy += state.y * test_derivative
            state.dx += test_derivative
            state.dy += test_derivative
        else:
            update_status = _mcstep(state, trial_value, derivative, low, high)
        if state.bracketed:
            if abs(state.y - state.x) >= .66 * previous_width:
                state.p = state.x + .5 * (state.y - state.x)
            previous_width, width = width, abs(state.y - state.x)
    raise AssertionError('line-search termination must be explicit')


def _minimize(objective, initial: np.ndarray, tolerance: float):
    parameters = initial[:3].astype(np.float32).ravel().astype(np.float64)
    records, updates = [], 0
    best_parameters = parameters.copy()
    best_value = math.inf

    def evaluate(vector):
        nonlocal updates, best_parameters, best_value
        matrix = _matrix(vector)
        value = float(np.float32(objective.cost(matrix)))
        gradient = objective.gradient(matrix).astype(np.float32).ravel().astype(np.float64)
        finite_gradient = bool(np.isfinite(gradient).all())
        if not math.isfinite(value) or (not records and not finite_gradient):
            raise RuntimeError('EM initial gradient or objective became nonfinite')
        improved = bool(records and value < best_value)
        if improved:
            updates += 1
            best_parameters = vector.copy()
        if value < best_value:
            best_value = value
        records.append({'cost': value,
                        'gradient_norm': math.sqrt(_dot(gradient, gradient)) if finite_gradient else None,
                        'gradient_finite': finite_gradient, 'best_updated': improved})
        if len(records) > 2001:
            raise RuntimeError('EM minimizer exceeded the source 2000 evaluation budget')
        return value, gradient

    value, gradient = evaluate(parameters)
    pairs = []
    reason = None
    while reason is None:
        if not pairs:
            direction = -gradient
            norm = math.sqrt(_dot(gradient, gradient))
            if norm == 0:
                reason = 'zero_gradient'
                break
            step = 1.0 / norm
        else:
            work = -gradient.copy()
            coefficients = []
            for displacement, difference in reversed(pairs):
                rho = 1.0 / _dot(difference, displacement)
                alpha = rho * _dot(displacement, work)
                coefficients.append((rho, alpha))
                work -= alpha * difference
            latest_s, latest_y = pairs[-1]
            work *= _dot(latest_s, latest_y) / _dot(latest_y, latest_y)
            for (displacement, difference), (rho, alpha) in zip(pairs, reversed(coefficients)):
                beta = rho * _dot(difference, work)
                work += (alpha - beta) * displacement
            direction, step = work, 1.0
        trial, new_value, new_gradient, accepted_step, line_reason = _line_search(
            evaluate, parameters, value, gradient, direction, step)
        records[-1]['line_termination'] = line_reason
        if line_reason != 'strong_wolfe':
            reason = line_reason
            break
        displacement = accepted_step * direction
        difference = new_gradient - gradient
        parameters, value, gradient = trial, new_value, new_gradient
        if math.sqrt(_dot(gradient, gradient)) / max(1.0, math.sqrt(_dot(parameters, parameters))) <= tolerance:
            reason = 'relative_gradient_tolerance'
            break
        if _dot(difference, displacement) <= 0:
            raise RuntimeError('EM L-BFGS curvature is nonpositive after Wolfe acceptance')
        pairs.append((displacement, difference))
        if len(pairs) > 7:
            pairs.pop(0)
    return _matrix(best_parameters), (records[-1]['cost'] if updates else 0.0), {'termination': reason, 'optimal_updates': updates, 'best_cost': best_value,
        'evaluations': records, 'success': reason == 'relative_gradient_tolerance'}


def align_em(objective, initial: np.ndarray, *, tolerance: float) -> tuple[np.ndarray, dict]:
    """执行完整12参数准牛顿与源EM外循环；返回FP32 voxel矩阵和终止轨迹。

    objective提供cost/gradient；initial有限4×4 affine；tolerance为源
    稳定阶段1e-5或全样本1e-7。每次L-BFGS保留7对历史、20次步长试探，
    2000次目标评估，保留最佳试探和失败终止，最多6次准牛顿pass。
    不调用SciPy优化器替换接受规则。无效参数、非有限初始梯度/目标或无进展的
    成功重启明确报错；无有效分类器试探的源Inf/NaN梯度保留供步长收缩，
    以typed标记记录，不改写成零梯度，不把缺少必要优化当完成。
    """
    values = np.asarray(initial, dtype=np.float64)
    if (values.shape != (4, 4) or not np.isfinite(values).all()
            or not np.array_equal(values[3], [0, 0, 0, 1])):
        raise ValueError('finite canonical 4x4 affine required')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        matrix = values.astype(np.float32)
        tolerance = float(np.float32(tolerance))
    if not np.isfinite(matrix).all() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('finite FP32 affine and finite positive FP32 tolerance required')
    report = {'tolerance': tolerance, 'sample_count':len(objective.samples.labels),
              'passes': [], 'source_protocol':'OpenDFPMin memory7/More-Thuente0.1/relative-gradient; MRIemAlign'}
    old_outer = float(np.float32(objective.cost(matrix)))
    while True:
        old_pass = float(np.float32(objective.cost(matrix)))
        for pass_index in range(6):
            starting_value = old_pass
            # OpenDFPMin keeps its entry value across successful minimizer restarts.
            restart_records = []
            entry_value = old_pass
            while True:
                before = matrix.copy()
                candidate, cost, record = _minimize(objective, matrix, tolerance)
                restart_records.append(record)
                if record['optimal_updates']:
                    matrix = candidate
                if not record['success'] or not (_relative_change(entry_value, cost) > tolerance):
                    break
                if np.array_equal(before, matrix):
                    raise RuntimeError('source EM successful restart made no parameter progress')
            total_updates = sum(item['optimal_updates'] for item in restart_records)
            report['passes'].append({'optimal_updates':total_updates,'outer_cost_before':old_outer,'quasi_newton_pass':pass_index,
                                      'restarts':restart_records,'cost':cost})
            if not total_updates:
                break
            old_pass = cost
            if not (_relative_change(starting_value, cost) > tolerance):
                break
        new_outer = float(np.float32(objective.cost(matrix)))
        report['outer_cost_after'] = new_outer
        # Preserve the source's signed continuation test, including worse-cost cases.
        if not (-_relative_change(old_outer, new_outer) > tolerance):
            break
        if new_outer == old_outer:
            raise RuntimeError('EM outer loop made no progress')
        old_outer = new_outer
    return matrix, report
