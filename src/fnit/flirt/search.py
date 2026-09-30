"""Cooperative FLIRT Brent searches with the reference scalar arithmetic.

Each independent search yields its next required trial. The scheduler batches
those trials, then resumes each search in its original evaluation order.
"""

import numpy as np

from .core import _extrapolated_point, _next_point, _quadratic_minimum


def _bound_trials(x1, middle, y1, y_middle, direction, point):
    f32 = np.float32
    factor = f32(1.6)
    (x1, middle) = (f32(x1), f32(middle))
    (y1, y_middle) = (f32(y1), f32(y_middle))
    if y1 == 0:
        y1 = f32((yield (float(x1) * direction + point)))
    if y_middle == 0:
        y_middle = f32((yield (float(middle) * direction + point)))
    if y1 < y_middle:
        (x1, middle) = (middle, x1)
        (y1, y_middle) = (y_middle, y1)
    sign = f32(-1.0 if middle < x1 else 1.0)
    x2 = f32(middle + f32(factor * f32(middle - x1)))
    y2 = f32((yield (float(x2) * direction + point)))
    for _ in range(256):
        if y_middle <= y2:
            return (x1, middle, x2, y1, y_middle, y2)
        maximum = f32(middle + f32(f32(factor * f32(2.0)) * f32(x2 - middle)))
        new = _quadratic_minimum(x1, middle, x2, y1, y_middle, y2)
        if new is None or (new - x1) * sign < 0 or (new - maximum) * sign > 0:
            new = f32(middle + f32(factor * f32(x2 - x1)))
        else:
            new = f32(new)
        y_new = f32((yield (float(new) * direction + point)))
        if (new - middle) * (new - x1) < 0:
            if y_new < y_middle:
                return (x1, new, middle, y1, y_new, y_middle)
            (x1, y1) = (new, y_new)
        elif y_new > y_middle:
            return (x1, middle, new, y1, y_middle, y_new)
        elif (new - x2) * sign < 0:
            (x1, y1, middle, y_middle) = (middle, y_middle, new, y_new)
        else:
            (x1, y1, middle, y_middle, x2, y2) = (middle, y_middle, x2, y2, new, y_new)
    raise RuntimeError('FLIRT line search did not bracket a minimum')

def _dimension_trials(point, direction, tolerance, maximum_iterations, initial_value, bound_guess):
    f32 = np.float32
    unit = direction / np.linalg.norm(direction)
    direction_tolerance = f32(0.0)
    for index in range(len(tolerance)):
        if abs(tolerance[index]) > 1e-15:
            direction_tolerance = f32(direction_tolerance + f32(abs(unit[index] / tolerance[index])))
    unit_tolerance = f32(abs(f32(1.0) / direction_tolerance))
    (middle, x1) = (f32(0.0), f32(f32(bound_guess) * unit_tolerance))
    y_middle = f32(initial_value if initial_value != 0 else (yield point))
    y1 = f32((yield (float(x1) * unit + point)))
    (x1, middle, x2, y1, y_middle, y2) = (yield from _bound_trials(x1, middle, y1, y_middle, unit, point))
    minimum_distance = f32(f32(0.1) * unit_tolerance)
    iteration = 0
    while iteration < maximum_iterations and abs((x2 - x1) / unit_tolerance) > 1.0:
        iteration += 1
        new = _next_point(x1, middle, x2, y1, y_middle, y2)
        sign = -1.0 if x2 < x1 else 1.0
        if abs(new - x1) < minimum_distance:
            new = f32(x1 + f32(sign * minimum_distance))
        if abs(new - x2) < minimum_distance:
            new = f32(x2 - f32(sign * minimum_distance))
        if abs(new - middle) < minimum_distance:
            new = _extrapolated_point(x1, middle, x2)
        if abs(middle - x1) < 0.4 * unit_tolerance:
            new = f32(middle + f32(f32(sign * f32(0.5)) * unit_tolerance))
        if abs(middle - x2) < 0.4 * unit_tolerance:
            new = f32(middle - f32(f32(sign * f32(0.5)) * unit_tolerance))
        new = f32(new)
        y_new = f32((yield (float(new) * unit + point)))
        if (new - middle) * (x2 - middle) > 0:
            (x1, x2, y1, y2) = (x2, x1, y2, y1)
        if y_new < y_middle:
            (x2, y2, middle, y_middle) = (middle, y_middle, new, y_new)
        else:
            (x1, y1) = (new, y_new)
    return (float(middle) * unit + point, float(y_middle))

def coordinate_trials(point, tolerance, *, maximum_iterations=4, bound_guess=(10.0, 1.0), numopt=None):
    """Port of ``MISCMATHS::optimise`` with FLIRT's default Brent mode."""
    point = np.asarray(point, dtype=np.float64).copy()
    tolerance = np.asarray(tolerance, dtype=np.float64)
    if numopt is None:
        numopt = len(point)
    if not 0 < numopt <= len(point):
        raise ValueError('numopt must be between one and the point dimension')
    inverse_tolerance = np.zeros_like(tolerance)
    nonzero = np.abs(tolerance) > 1e-15
    inverse_tolerance[nonzero] = np.abs(1 / tolerance[nonzero])
    inverse_tolerance /= len(tolerance)
    value = 0.0
    for major in range(maximum_iterations):
        initial = point.copy()
        guess = bound_guess[min(major, len(bound_guess) - 1)]
        for index in range(numopt):
            direction = np.zeros_like(point)
            direction[index] = 1
            (point, value) = (yield from _dimension_trials(point, direction, tolerance, 100, value, guess))
        average_tolerance = np.abs((initial - point) * inverse_tolerance).sum()
        if average_tolerance < 1.0:
            break
    return (point, float(value))


def map_trials(trials, transform):
    """Map yielded parameter vectors to affine matrices without changing state."""
    try:
        point = next(trials)
        while True:
            value = yield transform(point)
            point = trials.send(value)
    except StopIteration as completed:
        return completed.value


def evaluate_trials(trials, evaluate_many):
    """Resolve independent searches in stable waves with one host result batch."""
    trials = list(trials)
    results = [None] * len(trials)
    pending = {}
    for index, trial in enumerate(trials):
        try:
            pending[index] = next(trial)
        except StopIteration as completed:
            results[index] = completed.value
    while pending:
        values = evaluate_many(list(pending.values()))
        next_pending = {}
        for index, value in zip(pending, values):
            try:
                next_pending[index] = trials[index].send(float(value))
            except StopIteration as completed:
                results[index] = completed.value
        pending = next_pending
    return results
