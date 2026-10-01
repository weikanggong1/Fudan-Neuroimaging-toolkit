"""GPU mini-batch dictionary learning with streamed projected modalities.

Sparse codes and sequential dictionary updates follow sklearn's Lasso-LARS
online objective; projected voxels are read from disk in bounded blocks.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from .pipeline import _device


def _streaming_numpy_axis0_stats(projected, feature_block: int) -> tuple[np.ndarray, np.ndarray]:
    """Bounded two-pass statistics in NumPy's C-order float64 row order.

    Float32 storage is promoted before arithmetic: its reference is a full
    C-order float64 copy, not NumPy's default float32 mean/std accumulation.
    Constant-column std remains zero for the caller's 0.1 replacement.
    """
    feature_block = operator.index(feature_block)
    if feature_block < 1 or len(projected.shape) != 2:
        raise ValueError("Expected a two-dimensional projection and positive feature_block")
    rows, columns = projected.shape
    if rows < 1 or columns < 2:
        raise ValueError("Streamed DicL statistics require nonempty input and at least two PCs")
    if np.dtype(projected.dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Streamed DicL statistics require float32 or float64 storage")
    if isinstance(projected, np.ndarray) and not projected.flags.c_contiguous:
        raise ValueError("Streamed DicL statistics use the C-order axis-0 reference")
    scratch = np.empty((min(feature_block, rows) + 1, columns), dtype=np.float64)

    def accumulate(center=None):
        total = np.zeros(columns, dtype=np.float64)
        for start in range(0, rows, feature_block):
            end = min(start + feature_block, rows)
            block = scratch[:end - start + 1]
            block[0] = total
            block[1:] = projected[start:end]
            if not np.isfinite(block[1:]).all():
                raise ValueError("DicL projection contains non-finite values")
            if center is not None:
                np.subtract(block[1:], center, out=block[1:])
                np.multiply(block[1:], block[1:], out=block[1:])
            # Carry the prefix through every row, rather than independently
            # summing each block and changing the floating-point reduction.
            np.cumsum(block, axis=0, dtype=np.float64, out=block)
            total[:] = block[-1]
        return total

    mean = accumulate()
    np.true_divide(mean, rows, out=mean)
    std = accumulate(mean)
    np.true_divide(std, rows, out=std)
    np.sqrt(std, out=std)
    return mean, std


def _preload_standardized_projection(projected, mean: torch.Tensor,
                                     std: torch.Tensor,
                                     feature_block: int) -> torch.Tensor:
    """Preallocate the device cache and fill it from bounded CPU row windows.

    Subtraction and division retain the original float64 element operations.
    Normalizing the destination slice in place avoids full-size device
    subtraction/division temporaries and never loads projected[:] into RAM.
    """
    feature_block = operator.index(feature_block)
    if feature_block < 1:
        raise ValueError("feature_block must be positive")
    samples = torch.empty(projected.shape, device=mean.device, dtype=torch.float64)
    for start in range(0, projected.shape[0], feature_block):
        end = min(start + feature_block, projected.shape[0])
        chunk_cpu = torch.as_tensor(projected[start:end])
        destination = samples[start:end]
        # Default blocking transfer keeps the CPU chunk alive through the
        # copy; only this row window and the allocated device cache are held.
        destination.copy_(chunk_cpu)
        destination.sub_(mean).div_(std)
        del chunk_cpu, destination
    return samples


def _randomized_svd_dictionary(projected: h5py.Dataset, samples_gpu: torch.Tensor | None,
                               mean_gpu: torch.Tensor, std_gpu: torch.Tensor,
                               n_components: int, rng: np.random.RandomState,
                               feature_block: int) -> torch.Tensor:
    """Match sklearn's seeded randomized SVD with float64 CUDA arithmetic."""
    n_voxels, n_features = projected.shape
    backend = mean_gpu.device
    n_random = n_components + 10
    q_bytes = n_voxels * n_random * 8
    if (backend.type == "cuda" and
            q_bytes > min(8 * 2**30, torch.cuda.mem_get_info(backend)[0] // 3)):
        raise MemoryError("Randomized SVD basis exceeds the GPU memory budget")
    q = torch.as_tensor(rng.normal(size=(n_features, n_random)),
                        device=backend, dtype=torch.float64)

    def multiply_x(matrix: torch.Tensor) -> torch.Tensor:
        if samples_gpu is not None:
            return samples_gpu @ matrix
        output = torch.empty((n_voxels, matrix.shape[1]), device=backend,
                             dtype=torch.float64)
        for start in range(0, n_voxels, feature_block):
            block = torch.as_tensor(projected[start:start + feature_block],
                                    device=backend, dtype=torch.float64)
            output[start:start + block.shape[0]] = ((block - mean_gpu) / std_gpu) @ matrix
        return output

    def multiply_xt(matrix: torch.Tensor) -> torch.Tensor:
        if samples_gpu is not None:
            return samples_gpu.T @ matrix
        output = torch.zeros((n_features, matrix.shape[1]), device=backend,
                             dtype=torch.float64)
        for start in range(0, n_voxels, feature_block):
            block = torch.as_tensor(projected[start:start + feature_block],
                                    device=backend, dtype=torch.float64)
            output += ((block - mean_gpu) / std_gpu).T @ matrix[start:start + block.shape[0]]
        return output

    n_iter = 7 if n_components < 0.1 * min(n_voxels, n_features) else 4
    for _ in range(n_iter):
        q = torch.linalg.qr(multiply_x(q), mode="reduced")[0]
        q = torch.linalg.qr(multiply_xt(q), mode="reduced")[0]
    q = torch.linalg.qr(multiply_x(q), mode="reduced")[0]
    svd_options = {"full_matrices": False}
    if backend.type == "cuda":
        svd_options["driver"] = "gesvd"
    u_small, singular_values, right = torch.linalg.svd(multiply_xt(q).T, **svd_options)
    left = q @ u_small
    indices = left.abs().argmax(dim=0)
    signs = torch.sign(left[indices, torch.arange(left.shape[1], device=backend)])
    count = min(n_components, right.shape[0])
    dictionary = torch.zeros((n_components, n_features), device=backend,
                             dtype=torch.float64)
    dictionary[:count] = (singular_values[:count, None] * right[:count] *
                          signs[:count, None])
    return dictionary


def _sparse_codes_lars(samples: torch.Tensor, dictionary: torch.Tensor,
                       alpha: float, max_events: int | None = None) -> torch.Tensor:
    """Batched Lasso-LARS path on CUDA, stopping at the sklearn penalty alpha."""
    gram = dictionary @ dictionary.T
    response = samples @ dictionary.T
    batch, atoms = response.shape
    code = torch.zeros_like(response)
    active = torch.zeros((batch, atoms), device=samples.device, dtype=torch.bool)
    signs = torch.zeros_like(response)
    done = response.abs().amax(dim=1) <= alpha
    initial = response.abs().argmax(dim=1)
    rows = torch.arange(batch, device=samples.device)
    active[rows, initial] = ~done
    signs[rows, initial] = torch.sign(response[rows, initial]) * (~done)
    solve_failed = torch.zeros((), device=samples.device, dtype=torch.bool)
    limit = max_events or 3 * atoms
    for event in range(limit):
        if event % 4 == 0 and bool(done.all()):
            break
        correlation = response - code @ gram
        current = (correlation * active).abs().amax(dim=1)
        masked = gram[None] * active[:, :, None] * active[:, None, :]
        masked = masked + torch.diag_embed((~active).to(samples.dtype))
        # Adding a ridge changes the LASSO coefficients and can accumulate
        # through the online dictionary updates. Solve the unregularized
        # active Gram; a genuinely singular system must remain diagnosable.
        solved = torch.linalg.solve_ex(masked, signs[..., None],
                                       check_errors=False)
        solve_failed |= ((solved.info != 0).any() |
                         ~torch.isfinite(solved.result).all())
        direction = solved.result[..., 0]
        scale = torch.rsqrt((direction * signs).sum(dim=1).clamp_min(1e-20))
        direction = direction * scale[:, None]
        correlations_slope = direction @ gram
        target_step = ((current - alpha) / scale).clamp_min(0)
        upper = scale[:, None] - correlations_slope
        lower = scale[:, None] + correlations_slope
        positive = (current[:, None] - correlation) / upper.clamp_min(1e-20)
        negative = (current[:, None] + correlation) / lower.clamp_min(1e-20)
        possible = (~active) & (~done[:, None])
        positive = torch.where(possible & (upper > 1e-12) & (positive > 1e-12),
                               positive, torch.inf)
        negative = torch.where(possible & (lower > 1e-12) & (negative > 1e-12),
                               negative, torch.inf)
        enter_positive, positive_index = positive.min(dim=1)
        enter_negative, negative_index = negative.min(dim=1)
        entering_positive = enter_positive <= enter_negative
        enter_step = torch.minimum(enter_positive, enter_negative)
        enter_index = torch.where(entering_positive, positive_index, negative_index)
        drop = -code / torch.where(direction.abs() > 1e-20, direction,
                                   torch.ones_like(direction))
        drop = torch.where(active & (drop > 1e-12) & (code * direction < 0),
                           drop, torch.inf)
        drop_step, drop_index = drop.min(dim=1)
        step = torch.minimum(target_step, torch.minimum(enter_step, drop_step))
        step = torch.where(done, 0, step)
        code += step[:, None] * direction
        reached_target = (target_step <= enter_step) & (target_step <= drop_step)
        done |= reached_target
        dropping = (~done) & (drop_step < enter_step)
        entering = (~done) & (~dropping)
        drop_at = drop_index[:, None]
        drop_mask = dropping[:, None]
        active.scatter_(1, drop_at, active.gather(1, drop_at) & ~drop_mask)
        signs.scatter_(1, drop_at, torch.where(drop_mask, 0, signs.gather(1, drop_at)))
        code.scatter_(1, drop_at, torch.where(drop_mask, 0, code.gather(1, drop_at)))
        enter_at = enter_index[:, None]
        enter_mask = entering[:, None]
        active.scatter_(1, enter_at, active.gather(1, enter_at) | enter_mask)
        entering_sign = torch.where(entering_positive[:, None], 1., -1.)
        signs.scatter_(1, enter_at, torch.where(enter_mask, entering_sign,
                                               signs.gather(1, enter_at)))
    if bool(solve_failed):
        raise ValueError("LARS active Gram solve failed: singular or non-finite system")
    if not bool(done.all()):
        raise ValueError("LARS path exceeded sparse_iterations; increase the limit")
    return code


def _sparse_codes_lars_compatible(samples: torch.Tensor, dictionary: torch.Tensor,
                                  alpha: float,
                                  max_events: int | None = None) -> torch.Tensor:
    """Follow sklearn's LARS nodes and its float32-epsilon stopping rule.

    A path node within epsilon of alpha is returned without interpolation.
    Otherwise the last segment is interpolated to alpha. This differs from
    always solving the exact target LASSO, even with float64 input arithmetic.
    The implementation uses PyTorch throughout; no sklearn call is made.
    """
    gram = dictionary @ dictionary.T
    response = samples @ dictionary.T
    batch, atoms = response.shape
    rows = torch.arange(batch, device=samples.device)
    code = torch.zeros_like(response)
    covariance = response.clone()
    active = torch.zeros_like(response, dtype=torch.bool)
    signs = torch.zeros_like(response)
    blocked = torch.zeros_like(active)
    done = torch.zeros(batch, device=samples.device, dtype=torch.bool)
    add_atom = torch.ones_like(done)
    previous_code = code.clone()
    previous_c = response.abs().amax(dim=1)
    has_previous = torch.zeros_like(done)
    failed = torch.zeros((), device=samples.device, dtype=torch.bool)
    tolerance = samples.shape[1] * np.finfo(np.float32).eps
    tiny = np.finfo(np.float32).tiny
    decimals = 15 if samples.dtype == torch.float64 else 6
    limit = max_events or 3 * atoms

    # The extra pass only checks the final node; it performs no path event.
    for event in range(limit + 1):
        candidate_covariance = torch.where(active | blocked, 0., covariance)
        current, entering_index = candidate_covariance.abs().max(dim=1)
        stopping = (~done) & (current <= alpha + tolerance)
        interpolate = stopping & has_previous & ((current - alpha).abs() > tolerance)
        denominator = previous_c - current
        valid_segment = (denominator > 0) & torch.isfinite(denominator)
        failed |= (interpolate & ~valid_segment).any()
        fraction = (previous_c - alpha) / torch.where(valid_segment, denominator, 1.)
        interpolated = previous_code + fraction[:, None] * (code - previous_code)
        code = torch.where((interpolate & valid_segment)[:, None], interpolated, code)
        done |= stopping
        if event % 4 == 0 and bool(done.all()):
            break
        if event == limit:
            break

        adding = (~done) & add_atom
        selected_covariance = covariance[rows, entering_index]
        active[rows, entering_index] |= adding
        signs[rows, entering_index] = torch.where(
            adding, selected_covariance.sign(), signs[rows, entering_index])
        covariance[rows, entering_index] = torch.where(
            adding, 0., covariance[rows, entering_index])

        masked = (gram[None] * active[:, :, None] * active[:, None, :] +
                  torch.diag_embed((~active).to(samples.dtype)))
        solved = torch.linalg.solve_ex(masked, signs[..., None], check_errors=False)
        raw_direction = solved.result[..., 0]
        signed_sum = (raw_direction * signs).sum(dim=1)
        bad_system = ((solved.info != 0) | ~torch.isfinite(raw_direction).all(dim=1) |
                      ~torch.isfinite(signed_sum) | (signed_sum <= 0)) & ~done
        # A dependent newly entered atom is excluded explicitly, never
        # regularized. Other failures cannot be recovered by changing alpha.
        rejected = bad_system & adding
        active[rows, entering_index] &= ~rejected
        signs[rows, entering_index] = torch.where(
            rejected, 0., signs[rows, entering_index])
        blocked[rows, entering_index] |= rejected
        failed |= (bad_system & ~adding).any()
        processing = (~done) & ~bad_system
        raw_direction = torch.where(processing[:, None], raw_direction, 0.)
        scale = torch.rsqrt(torch.where(processing, signed_sum, 1.))
        direction = raw_direction * scale[:, None]
        slope = torch.round(direction @ gram, decimals=decimals)
        possible = (~active) & (~blocked) & processing[:, None]
        positive = (current[:, None] - covariance) / (scale[:, None] - slope + tiny)
        negative = (current[:, None] + covariance) / (scale[:, None] + slope + tiny)
        positive = torch.where(possible & (positive > 0), positive, torch.inf)
        negative = torch.where(possible & (negative > 0), negative, torch.inf)
        gamma = torch.minimum(positive.amin(dim=1), negative.amin(dim=1))
        gamma = torch.minimum(gamma, current / scale)
        crossing = -code / (direction + tiny)
        crossing = torch.where(active & processing[:, None] & (crossing > 0),
                               crossing, torch.inf)
        drop_step = crossing.amin(dim=1)
        dropping = processing & (drop_step < gamma)
        gamma = torch.where(dropping, drop_step, gamma)
        gamma = torch.where(processing, gamma, 0.)
        previous_code = torch.where(processing[:, None], code, previous_code)
        previous_c = torch.where(processing, current, previous_c)
        has_previous |= processing
        # sklearn clears inactive coefficients on each new path segment.
        updated_code = torch.where(active, code + gamma[:, None] * direction, 0.)
        code = torch.where(processing[:, None], updated_code, code)
        covariance = torch.where(possible, covariance - gamma[:, None] * slope,
                                 covariance)
        drop_mask = active & dropping[:, None] & (crossing == drop_step[:, None])
        dropped_covariance = response - code @ gram
        covariance = torch.where(drop_mask, dropped_covariance, covariance)
        active &= ~drop_mask
        signs = torch.where(drop_mask, 0., signs)
        add_atom = torch.where(processing, ~dropping, torch.ones_like(add_atom))

    if bool(failed):
        raise ValueError("Compatible LARS active Gram or interpolation failed")
    if not bool(done.all()):
        raise ValueError("Compatible LARS path exceeded sparse_iterations; increase the limit")
    return code


def _lars_inverse_event(gram, response, code, active, signs, done, inverse, invalid, rows, alpha=1.0):
    correlation = response - code @ gram
    current = (correlation * active).abs().amax(dim=1)
    raw_direction = torch.bmm(inverse, signs[..., None])[..., 0]
    scale = torch.rsqrt((raw_direction * signs).sum(dim=1).clamp_min(1e-20))
    direction = raw_direction * scale[:, None]
    slope = direction @ gram
    # Check each active system before accepting the accumulated inverse.
    residual = (slope - scale[:, None] * signs) * active
    invalid |= (~torch.isfinite(raw_direction).all(dim=1) & ~done).any()
    invalid |= (((residual.abs().amax(dim=1) / scale.clamp_min(1e-20)) > 1e-7)
                & ~done).any()
    target_step = ((current - alpha) / scale).clamp_min(0)
    upper = scale[:, None] - slope
    lower = scale[:, None] + slope
    positive = (current[:, None] - correlation) / upper.clamp_min(1e-20)
    negative = (current[:, None] + correlation) / lower.clamp_min(1e-20)
    possible = (~active) & (~done[:, None])
    positive = torch.where(possible & (upper > 1e-12) & (positive > 1e-12),
                           positive, torch.inf)
    negative = torch.where(possible & (lower > 1e-12) & (negative > 1e-12),
                           negative, torch.inf)
    positive_step, positive_index = positive.min(dim=1)
    negative_step, negative_index = negative.min(dim=1)
    entering_positive = positive_step <= negative_step
    enter_step = torch.minimum(positive_step, negative_step)
    enter_index = torch.where(entering_positive, positive_index, negative_index)
    drop = -code / torch.where(direction.abs() > 1e-20, direction,
                               torch.ones_like(direction))
    drop = torch.where(active & (drop > 1e-12) & (code * direction < 0),
                       drop, torch.inf)
    drop_step, drop_index = drop.min(dim=1)
    step = torch.minimum(target_step, torch.minimum(enter_step, drop_step))
    code += torch.where(done, 0, step)[:, None] * direction
    done |= (target_step <= enter_step) & (target_step <= drop_step)
    dropping = (~done) & (drop_step < enter_step)
    entering = (~done) & (~dropping)

    # Remove an active atom with a Schur complement downdate.
    column = inverse[rows, :, drop_index]
    pivot = inverse[rows, drop_index, drop_index]
    valid_pivot = (pivot > 0) & torch.isfinite(pivot)
    denominator = torch.where(dropping & valid_pivot, pivot, 1)
    invalid |= (dropping & ~valid_pivot).any()
    inverse -= (column[:, :, None] * column[:, None, :] *
                dropping[:, None, None] / denominator[:, None, None])
    drop_at = drop_index[:, None]
    active.scatter_(1, drop_at, active.gather(1, drop_at) & ~dropping[:, None])
    signs.scatter_(1, drop_at, torch.where(dropping[:, None], 0,
                                           signs.gather(1, drop_at)))
    code.scatter_(1, drop_at, torch.where(dropping[:, None], 0,
                                          code.gather(1, drop_at)))
    inverse *= active[:, :, None] * active[:, None, :]

    # Add an atom with a rank-one inverse update, avoiding a fresh LU solve.
    cross = gram[:, enter_index].T
    vector = torch.bmm(inverse, cross[..., None])[..., 0]
    schur = gram[enter_index, enter_index] - (cross * vector).sum(dim=1)
    valid_schur = (schur > 0) & torch.isfinite(schur)
    invalid |= (entering & ~valid_schur).any()
    # Invalid rows restart the full checked solver. Keep graph intermediates
    # finite rather than silently regularizing a degenerate Schur complement.
    denominator = torch.where(entering & valid_schur, schur, 1)
    inverse += (vector[:, :, None] * vector[:, None, :] *
                entering[:, None, None] / denominator[:, None, None])
    new_row = -vector / denominator[:, None]
    new_row.scatter_(1, enter_index[:, None], (1 / denominator)[:, None])
    old_row = inverse[rows, enter_index, :]
    selected_row = torch.where(entering[:, None], new_row, old_row)
    inverse[rows, enter_index, :] = selected_row
    inverse[rows, :, enter_index] = selected_row
    enter_at = enter_index[:, None]
    active.scatter_(1, enter_at, active.gather(1, enter_at) | entering[:, None])
    entering_sign = torch.where(entering_positive[:, None], 1., -1.)
    signs.scatter_(1, enter_at, torch.where(entering[:, None], entering_sign,
                                            signs.gather(1, enter_at)))


class _LarsInverseSolver:
    """Incremental LARS with bounded graph workspace and a checked LU fallback."""

    def __init__(self, batch, atoms, device, dtype, alpha=1.0):
        self.alpha = alpha
        self.fallback_count = 0
        self.gram = torch.eye(atoms, device=device, dtype=dtype)
        self.response = torch.zeros((batch, atoms), device=device, dtype=dtype)
        self.code = torch.zeros_like(self.response)
        self.active = torch.zeros_like(self.response, dtype=torch.bool)
        self.signs = torch.zeros_like(self.response)
        self.done = torch.ones(batch, device=device, dtype=torch.bool)
        self.inverse = torch.zeros((batch, atoms, atoms), device=device, dtype=dtype)
        self.invalid = torch.zeros((), device=device, dtype=torch.bool)
        self.rows = torch.arange(batch, device=device)
        self.state = (self.gram, self.response, self.code, self.active, self.signs,
                      self.done, self.inverse, self.invalid, self.rows)
        self.graph = None
        # Bound graph-private workspace for large user-selected batch sizes.
        if self.inverse.is_cuda and self.inverse.numel() * self.inverse.element_size() <= 32 * 2**20:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _lars_inverse_event(*self.state, alpha=self.alpha)
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                for _ in range(4):
                    _lars_inverse_event(*self.state, alpha=self.alpha)

    def __call__(self, samples, dictionary, alpha, max_events=None):
        if alpha != self.alpha:
            raise ValueError('LARS workspace alpha differs from the fitted penalty')
        self.gram.copy_(dictionary @ dictionary.T)
        self.response.copy_(samples @ dictionary.T)
        self.code.zero_()
        self.active.zero_()
        self.signs.zero_()
        self.inverse.zero_()
        self.invalid.zero_()
        self.done.copy_(self.response.abs().amax(dim=1) <= alpha)
        initial = self.response.abs().argmax(dim=1)
        self.active[self.rows, initial] = ~self.done
        self.signs[self.rows, initial] = torch.sign(self.response[self.rows, initial]) * (~self.done)
        initial_diagonal = self.gram[initial, initial]
        valid_initial = ((~self.done) & (initial_diagonal > 0) &
                         torch.isfinite(initial_diagonal))
        self.invalid |= ((~self.done) & ~valid_initial).any()
        denominator = torch.where(valid_initial, initial_diagonal, 1)
        self.inverse[self.rows, initial, initial] = valid_initial / denominator
        limit = max_events or 3 * self.response.shape[1]
        for start in range(0, limit, 4):
            if self.graph is not None and start + 4 <= limit:
                self.graph.replay()
            else:
                for _ in range(min(4, limit - start)):
                    _lars_inverse_event(*self.state, alpha=self.alpha)
            finished, invalid = torch.stack((self.done.all(), self.invalid)).tolist()
            if invalid:
                self.fallback_count += 1
                return _sparse_codes_lars_compatible(samples, dictionary, alpha, max_events)
            if finished:
                return self.code.clone()
        self.fallback_count += 1
        return _sparse_codes_lars_compatible(samples, dictionary, alpha, max_events)


def _nearby_lars_knots(gram: torch.Tensor, response: torch.Tensor,
                       code: torch.Tensor, active_direction: torch.Tensor,
                       alpha: float, tolerance: float) -> dict[str, torch.Tensor]:
    """Conservatively detect either adjacent path node near target alpha.

    active_direction is the existing active Gram solve G_AA^-1 sign(code).
    Along this segment c(alpha + delta) = c(alpha) - delta * direction.
    No coefficient is changed; a sensitive row requires compatible LARS.
    """
    active = code != 0
    product = code @ gram
    correlation = response - product
    slope = active_direction @ gram
    scale = torch.maximum(response.abs().amax(dim=1), product.abs().amax(dim=1))
    scale = scale.clamp_min(max(1., alpha))
    buffer = 64 * torch.finfo(code.dtype).eps * scale
    infinity = torch.full_like(code, torch.inf)
    active_zero = torch.where(active & (active_direction != 0),
                              code / active_direction, infinity)
    positive_bound = torch.where(~active & (slope != 1),
                                 (alpha - correlation) / (slope - 1), infinity)
    negative_bound = torch.where(~active & (slope != -1),
                                 (-alpha - correlation) / (slope + 1), infinity)
    distance = torch.stack((active_zero, positive_bound, negative_bound), dim=2).flatten(1)
    legal = torch.isfinite(distance) & (alpha + distance >= 0)
    above = torch.where(legal & (distance > 0), distance, torch.inf).amin(dim=1)
    below = torch.where(legal & (distance < 0), -distance, torch.inf).amin(dim=1)
    zero_knot = (legal & (distance == 0)).any(dim=1)
    # A bound already tied at alpha can give 0/0; it must not disappear
    # merely because an event-distance denominator is zero.
    boundary_tie = ((~active) &
                    ((correlation.abs() - alpha).abs() <= buffer[:, None])).any(dim=1)
    invalid = (~torch.isfinite(active_direction).all(dim=1) |
               ~torch.isfinite(code).all(dim=1) | ~torch.isfinite(response).all(dim=1))
    near_above = above <= tolerance + buffer
    near_below = below <= tolerance + buffer
    return dict(sensitive_rows=near_above | near_below | zero_knot | boundary_tie | invalid,
                nearest_above_distance=above, nearest_below_distance=below,
                near_above=near_above, near_below=near_below, zero_knot=zero_knot,
                boundary_tie=boundary_tie, invalid=invalid)


class _SparseCodesBPDN:
    """Compatible LARS, or optional ADMM support identification and polishing.

    ADMM equations: https://sporco.readthedocs.io/en/latest/modules/sporco.admm.bpdn.html
    The online dictionary updates and sklearn stopping rule are unchanged.
    """

    def __init__(self, batch, atoms, device, dtype, alpha=1.0,
                 *, compatibility_mode=False):
        # All-LARS remains a diagnostic reference. Default ADMM is accepted
        # only away from nodes where sklearn's stopping rule changes codes.
        self.compatibility_mode = compatibility_mode
        self.alpha = alpha
        self.calls = 0
        self.fallback_count = 0
        self.near_node_fallback_count = 0
        self.admm_accepted_count = 0
        self.polish_checks = 0
        self.graph = None
        self.identity = torch.eye(atoms, device=device, dtype=dtype)
        # Bound the temporary batched active-set matrix and graph workspace.
        if (not compatibility_mode and self.identity.is_cuda and
                batch * atoms * atoms * self.identity.element_size() <= 32 * 2**20):
            self.inverse = self.identity.clone()
            self.response = torch.zeros((batch, atoms), device=device, dtype=dtype)
            self.code = torch.zeros_like(self.response)
            self.dual = torch.zeros_like(self.code)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self.step()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                for _ in range(20):
                    self.step()

    def step(self):
        estimate = (self.response + self.code - self.dual) @ self.inverse
        relaxed = 1.8 * estimate + (1 - 1.8) * self.code
        value = relaxed + self.dual
        updated = value.sign() * (value.abs() - self.alpha).clamp_min(0)
        self.dual.add_(relaxed - updated)
        self.code.copy_(updated)

    def fallback(self, samples, dictionary, alpha, max_events):
        self.fallback_count += 1
        return _sparse_codes_lars_compatible(samples, dictionary, alpha, max_events)

    def __call__(self, samples, dictionary, alpha, max_events):
        if alpha != self.alpha:
            raise ValueError("Sparse solver alpha differs from captured alpha")
        self.calls += 1
        if self.compatibility_mode:
            return self.fallback(samples, dictionary, alpha, max_events)
        if self.graph is None:
            return self.fallback(samples, dictionary, alpha, max_events)
        gram = dictionary @ dictionary.T
        response = samples @ dictionary.T
        # Early updates contain unnormalized or unused SVD atoms. Retain the
        # original solver while the online dictionary settles into its ball.
        if self.calls <= 4 or bool(gram.diagonal().max() > 2):
            return self.fallback(samples, dictionary, alpha, max_events)
        self.inverse.copy_(torch.cholesky_inverse(torch.linalg.cholesky(gram + self.identity)))
        self.response.copy_(response)
        self.code.zero_(); self.dual.zero_()
        for _ in range(20):
            self.graph.replay()
            self.polish_checks += 1
            active = self.code.abs() > 1e-9
            signs = self.code.sign() * active
            masked = (gram[None] * active[:, :, None] * active[:, None, :] +
                      torch.diag_embed((~active).to(samples.dtype)))
            lu, pivots, info = torch.linalg.lu_factor_ex(masked)
            polished = torch.linalg.lu_solve(
                lu, pivots, ((response - alpha * signs) * active)[..., None])[..., 0] * active
            gradient = polished @ gram - response
            sign_valid = ((polished * signs > 0) | ~active).all()
            residual = torch.where(active, (gradient + alpha * signs).abs(),
                                   (gradient.abs() - alpha).clamp_min(0)).max()
            valid = ((info == 0).all() & torch.isfinite(polished).all() &
                     sign_valid & (residual <= 1e-8) &
                     (lu.diagonal(dim1=-2, dim2=-1).abs().min() > 1e-8))
            if bool(valid):
                active_direction = torch.linalg.lu_solve(
                    lu, pivots, signs[..., None])[..., 0] * active
                nearby = _nearby_lars_knots(
                    gram, response, polished, active_direction, alpha,
                    samples.shape[1] * np.finfo(np.float32).eps)
                if bool(nearby["sensitive_rows"].any()):
                    self.near_node_fallback_count += 1
                    return self.fallback(samples, dictionary, alpha, max_events)
                self.admm_accepted_count += 1
                return polished
        return self.fallback(samples, dictionary, alpha, max_events)


class _DictionaryUpdater:
    """Replay sequential atom updates; resample dead atoms with the original RNG."""

    def __init__(self, atoms, features, device, dtype):
        self.a = torch.eye(atoms, device=device, dtype=dtype)
        self.b = torch.zeros((features, atoms), device=device, dtype=dtype)
        self.dictionary = torch.zeros((atoms, features), device=device, dtype=dtype)
        self.graph = None
        if self.dictionary.is_cuda and atoms <= 512:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self.update()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.update()

    def update(self):
        for atom in range(len(self.dictionary)):
            self.dictionary[atom] += ((self.b[:, atom] - self.a[atom] @ self.dictionary) /
                                      self.a[atom, atom])
            self.dictionary[atom] /= torch.linalg.vector_norm(self.dictionary[atom]).clamp_min(1)

    def __call__(self, dictionary, a, b, samples, rng):
        all_alive = bool((torch.diagonal(a) > 1e-6).all())
        if self.graph is not None and all_alive:
            self.a.copy_(a)
            self.b.copy_(b)
            self.dictionary.copy_(dictionary)
            self.graph.replay()
            dictionary.copy_(self.dictionary)
            return
        # Keep the original resampling branch and order while atoms are unused.
        for atom in range(len(dictionary)):
            if all_alive or a[atom, atom] > 1e-6:
                dictionary[atom] += (b[:, atom] - a[atom] @ dictionary) / a[atom, atom]
            else:
                replacement = samples[int(rng.choice(len(samples)))]
                noise_level = .01 * float(replacement.std(correction=0)) or .01
                noise = torch.as_tensor(rng.normal(0, noise_level, size=samples.shape[1]),
                                        device=samples.device, dtype=samples.dtype)
                dictionary[atom] = replacement + noise
            dictionary[atom] /= torch.linalg.vector_norm(dictionary[atom]).clamp_min(1)


def fit_dicl_gpu_streaming(projected_dir: str | Path,
                           modality_names: Sequence[str], dicl_dim: int,
                           *, device: str = "cuda:0", max_iter: int = 1000,
                           batch_size: int = 32, sparse_iterations: int = 1000,
                           alpha: float = 1.0, random_state: int = 0,
                           feature_block: int = 4096) -> dict[str, np.ndarray]:
    """Fit K×R dictionaries without reading a full P×R modality into RAM."""
    backend = _device(device)
    if backend.type != "cuda":
        raise ValueError("fit_dicl_gpu_streaming requires CUDA")
    if dicl_dim < 2 or max_iter < 1 or batch_size < 1 or sparse_iterations < 1:
        raise ValueError("Invalid dictionary dimensions or iteration counts")
    output = {}
    atom_updaters = {}
    for name in modality_names:
        # The cold LARS calls belong to each seeded modality fit. Sharing
        # solver call counters changes the later modalities' encoding path.
        sparse_solvers = {}
        with h5py.File(Path(projected_dir) / f"{name}_projected.h5", "r") as file:
            projected = file["data"]
            n_voxels, n_features = projected.shape
            if n_voxels < dicl_dim:
                raise ValueError(f"Too few masked voxels for DicL: {name}")
            mean, std = _streaming_numpy_axis0_stats(projected, feature_block)
            std[std == 0] = 0.1
            mean_gpu = torch.as_tensor(mean, device=backend, dtype=torch.float64)
            std_gpu = torch.as_tensor(std, device=backend, dtype=torch.float64)

            projected_bytes = n_voxels * n_features * 8
            keep_on_gpu = projected_bytes < min(4 * 2**30, torch.cuda.mem_get_info(backend)[0] // 4)
            samples_gpu = None
            if keep_on_gpu:
                samples_gpu = _preload_standardized_projection(
                    projected, mean_gpu, std_gpu, feature_block)
            rng = np.random.RandomState(random_state)
            dictionary = _randomized_svd_dictionary(
                projected, samples_gpu, mean_gpu, std_gpu, dicl_dim, rng, feature_block)
            inner_a = torch.zeros((dicl_dim, dicl_dim), device=backend,
                                  dtype=torch.float64)
            inner_b = torch.zeros((n_features, dicl_dim), device=backend,
                                  dtype=torch.float64)
            permutation = torch.as_tensor(rng.permutation(n_voxels).copy())
            n_batches = (n_voxels + batch_size - 1) // batch_size
            atom_key = (dicl_dim, n_features)
            if atom_key not in atom_updaters:
                atom_updaters[atom_key] = _DictionaryUpdater(
                    dicl_dim, n_features, backend, torch.float64)
            atom_updater = atom_updaters[atom_key]
            step = 0
            ewa_cost = None
            best_cost = None
            no_improvement = 0
            converged = False
            for _ in range(max_iter):
                for batch_index in range(n_batches):
                    start, end = batch_index * batch_size, min((batch_index + 1) * batch_size, n_voxels)
                    indices = permutation[start:end]
                    if samples_gpu is not None:
                        samples = samples_gpu[indices.to(backend)]
                    else:
                        sorted_indices, undo = torch.sort(indices)
                        samples = torch.as_tensor(projected[sorted_indices.numpy()],
                                                  device=backend, dtype=torch.float64)
                        samples = ((samples - mean_gpu) / std_gpu)[torch.argsort(undo).to(backend)]
                    size = end - start
                    if size not in sparse_solvers:
                        sparse_solvers[size] = _SparseCodesBPDN(
                            size, dicl_dim, backend, torch.float64, alpha)
                    code = sparse_solvers[size](samples, dictionary, alpha,
                                                sparse_iterations)
                    cost = (0.5 * (samples - code @ dictionary).square().sum() +
                            alpha * code.abs().sum()) / size
                    old_dictionary = dictionary.clone()
                    theta = ((step + 1) * size if step < size - 1
                             else size ** 2 + step + 1 - size)
                    beta = (theta + 1 - size) / (theta + 1)
                    inner_a.mul_(beta).addmm_(code.T, code, alpha=1 / size)
                    inner_b.mul_(beta).addmm_(samples.T, code, alpha=1 / size)
                    # Atom order and dead-atom RNG draws follow sklearn.
                    atom_updater(dictionary, inner_a, inner_b, samples, rng)
                    step += 1
                    if step <= min(100, n_voxels / size):
                        continue
                    current = float(cost)
                    ewa_cost = current if ewa_cost is None else (
                        ewa_cost * (1 - min(size / (n_voxels + 1), 1)) +
                        current * min(size / (n_voxels + 1), 1))
                    if torch.linalg.vector_norm(dictionary - old_dictionary) / dicl_dim <= 1e-3:
                        converged = True
                        break
                    if best_cost is None or ewa_cost < best_cost:
                        best_cost = ewa_cost
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= 10:
                            converged = True
                            break
                if converged:
                    break
            dictionary -= dictionary.mean(dim=1, keepdim=True)
            scale = torch.sqrt(torch.mean(dictionary ** 2))
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError(f"Degenerate GPU dictionary: {name}")
            output[name] = (dictionary / scale).cpu().numpy().astype(np.float64)
    return output
