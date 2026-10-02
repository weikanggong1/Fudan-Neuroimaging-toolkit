"""Isolated source-order matrix candidate for the nine-parameter search.

FreeSurfer's ``MatrixMultiply`` accumulates four float products in index order.
This candidate keeps the existing search grid and JIT score, and changes only
the matrix composition arithmetic that feeds the score and EM initializer.
"""

import numpy as np
from numba import njit

from .mri_em_register import _grid_values, _rotation
from .mri_em_register_search_jit import log_sample_probability_jit


@njit(cache=True)
def _multiply_float32(left, right):
    result = np.empty((4, 4), np.float32)
    for row in range(4):
        for col in range(4):
            value = np.float32(0.0)
            for inner in range(4):
                value = np.float32(value + np.float32(left[row, inner] * right[inner, col]))
            result[row, col] = value
    return result


def _centered_source(matrix, origin):
    forward = np.eye(4, dtype=np.float32)
    backward = np.eye(4, dtype=np.float32)
    forward[:3, 3] = origin
    backward[:3, 3] = -origin
    return _multiply_float32(forward, _multiply_float32(matrix, backward))


def search_linear_iteration_source(samples, source, base_transform, origin,
                                   search_scale, minimum_scale, maximum_scale, *, scorer=None):
    base = np.asarray(base_transform, np.float32).copy()
    origin = np.asarray(origin, np.float32)
    minimum_scale, maximum_scale = np.float32(minimum_scale), np.float32(maximum_scale)
    minimum_angle = np.float32(-np.pi / 6 * search_scale)
    maximum_angle = np.float32(np.pi / 6 * search_scale)
    minimum_translation = np.float32(-15 * search_scale)
    maximum_translation = np.float32(15 * search_scale)
    score_one = log_sample_probability_jit if scorer is None else scorer
    maximum = score_one(samples, source, base)
    reductions = []
    for _ in range(2):
        scales = _grid_values(minimum_scale, maximum_scale,
                              float(np.float32((maximum_scale - minimum_scale) / np.float32(2))))
        angles = _grid_values(minimum_angle, maximum_angle,
                              float(np.float32((maximum_angle - minimum_angle) / np.float32(4))))
        translations = _grid_values(minimum_translation, maximum_translation,
                                    float(np.float32((maximum_translation - minimum_translation) / np.float32(2))))
        best = (1., 1., 1., 0., 0., 0., 0., 0., 0.)
        def candidates():
            for sx in scales:
                for sy in scales:
                    for sz in scales:
                        scale_matrix = np.eye(4, dtype=np.float32)
                        scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = sx, sy, sz
                        centered_scale = _centered_source(scale_matrix, origin)
                        for ax in angles:
                            x_rotation = _rotation(0, ax)
                            for ay in angles:
                                yx_rotation = _multiply_float32(_rotation(1, ay), x_rotation)
                                for az in angles:
                                    rotation = _centered_source(
                                        _multiply_float32(_rotation(2, az), yx_rotation), origin)
                                    fixed = _multiply_float32(
                                        _multiply_float32(centered_scale, rotation), base)
                                    for tx in translations:
                                        for ty in translations:
                                            for tz in translations:
                                                trial = fixed.copy()
                                                trial[:3, 3] += np.asarray((tx, ty, tz), np.float32)
                                                yield (sx, sy, sz, ax, ay, az, tx, ty, tz), trial
        for parameters, score in score_candidates(samples, source, candidates(), scorer):
            if score > maximum:
                maximum, best = score, parameters
        sx, sy, sz, ax, ay, az, tx, ty, tz = best
        scale_matrix = np.eye(4, dtype=np.float32)
        scale_matrix[0, 0], scale_matrix[1, 1], scale_matrix[2, 2] = sx, sy, sz
        rotation = _multiply_float32(
            _rotation(2, az),
            _multiply_float32(_rotation(1, ay), _rotation(0, ax)))
        base = _multiply_float32(
            _multiply_float32(_centered_source(scale_matrix, origin),
                              _centered_source(rotation, origin)), base)
        base[:3, 3] += np.asarray((tx, ty, tz), np.float32)
        reductions.append((best, maximum, base.copy()))
        scale_center = np.float32((maximum_scale + minimum_scale) / 2)
        scale_half_width = np.float32((maximum_scale - minimum_scale) / 4)
        minimum_scale, maximum_scale = (np.float32(scale_center - scale_half_width),
                                        np.float32(scale_center + scale_half_width))
        angle_center = np.float32((maximum_angle + minimum_angle) / 2)
        angle_half_width = np.float32((maximum_angle - minimum_angle) / 4)
        minimum_angle, maximum_angle = (np.float32(angle_center - angle_half_width),
                                        np.float32(angle_center + angle_half_width))
        translation_center = np.float32((maximum_translation + minimum_translation) / 2)
        translation_half_width = np.float32((maximum_translation - minimum_translation) / 4)
        minimum_translation, maximum_translation = (
            np.float32(translation_center - translation_half_width),
            np.float32(translation_center + translation_half_width))
    return base, maximum, reductions


def find_optimal_linear_transform_source(samples, source, base_transform,
                                         origin, initial_score, *, scorer=None):
    matrix = np.asarray(base_transform, np.float32).copy()
    current_score = initial_score
    search_scale = 1.
    minimum_scale, maximum_scale = .85, 1.15
    scale_reductions = 0
    good_step = False
    done = False
    history = []
    while True:
        old_score = current_score
        matrix, current_score, reductions = search_linear_iteration_source(
            samples, source, matrix, origin, search_scale, minimum_scale, maximum_scale, scorer=scorer)
        history.append((search_scale, current_score, matrix.copy(), reductions))
        if current_score < old_score + abs(.001 * old_score):
            search_scale *= .25
            if search_scale < .025:
                break
            half_width = (maximum_scale - minimum_scale) / 2
            minimum_scale = 1 - half_width * search_scale
            maximum_scale = 1 + half_width * search_scale
            done = not good_step
            good_step = False
            scale_reductions += 1
        else:
            good_step = True
        if scale_reductions >= 3 and done:
            break
    return matrix, history


def score_candidates(samples, source, candidates, scorer=None):
    """Yield scores in source order; batch only immutable candidate evaluations."""
    if scorer is None:
        for parameters, matrix in candidates:
            yield parameters, log_sample_probability_jit(samples, source, matrix)
        return
    batch = []
    for candidate in candidates:
        batch.append(candidate)
        if len(batch) >= scorer.candidate_chunk:
            scores = scorer.score_many(np.stack([item[1] for item in batch]))
            yield from zip((item[0] for item in batch), scores)
            batch.clear()
    if batch:
        scores = scorer.score_many(np.stack([item[1] for item in batch]))
        yield from zip((item[0] for item in batch), scores)
