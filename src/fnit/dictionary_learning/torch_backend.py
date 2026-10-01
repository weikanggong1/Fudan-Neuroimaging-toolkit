"""GPU mini-batch dictionary learning with streamed projected modalities.

Sparse codes and sequential dictionary updates follow sklearn's Lasso-LARS
online objective; projected voxels are read from disk in bounded blocks.
"""

from __future__ import annotations

import operator
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

# The Conda environment includes Triton. Keep CPU and installations without
# Triton usable through the unchanged PyTorch dictionary-update operations.
try:
    import triton as _dicl_triton
    import triton.language as _dicl_tl
    from triton.language.extra.cuda import libdevice as _dicl_libdevice
except ImportError:
    _dicl_triton = None
    _dicl_tl = None
    _dicl_libdevice = None

GPU_ALGORITHM_VERSION = "rsvd5rowgraph"


def _device(device: str) -> torch.device:
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    result = torch.device(selected)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return result



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


class _CompatibleLarsGraph:
    """Replay each event around eager LU, retaining four-event host checks.

    All persistent path state has fixed storage. The post-LU graph ends at
    the next node check, so a replay cannot advance past a host checkpoint.
    """

    def __init__(self, batch: int, atoms: int, device, dtype):
        self.gram = torch.eye(atoms, device=device, dtype=dtype)
        self.response = torch.zeros((batch, atoms), device=device, dtype=dtype)
        self.rows = torch.arange(batch, device=device)
        self.code = torch.zeros_like(self.response)
        self.covariance = torch.zeros_like(self.response)
        self.active = torch.zeros_like(self.response, dtype=torch.bool)
        self.signs = torch.zeros_like(self.response)
        self.blocked = torch.zeros_like(self.active)
        self.done = torch.ones(batch, device=device, dtype=torch.bool)
        self.add_atom = torch.ones_like(self.done)
        self.previous_code = torch.zeros_like(self.response)
        self.previous_c = torch.zeros(batch, device=device, dtype=dtype)
        self.current = torch.zeros_like(self.previous_c)
        self.entering_index = torch.zeros(batch, device=device, dtype=torch.long)
        self.has_previous = torch.zeros_like(self.done)
        self.failed = torch.zeros((), device=device, dtype=torch.bool)
        self.alpha = torch.ones((), device=device, dtype=dtype)
        self.tolerance = torch.zeros((), device=device, dtype=dtype)
        self.node_threshold = torch.zeros((), device=device, dtype=dtype)
        self.tiny = np.finfo(np.float32).tiny
        self.decimals = 15 if dtype == torch.float64 else 6
        self.adding = torch.zeros_like(self.done)
        self.masked = torch.empty((batch, atoms, atoms), device=device, dtype=dtype)
        self.raw_result = torch.zeros_like(self.response)
        self.solve_info = torch.zeros(batch, device=device, dtype=torch.int32)
        # LU stays eager: batched D200 solve_ex cannot be captured on every
        # supported torch/CUDA backend. Only the surrounding arithmetic is
        # captured, without changing its expression order.
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(3):
                self._prepare()
                self._solve()
                self._finish()
                self._node()
        current.wait_stream(side)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current)
        self.pre_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.pre_graph, stream=capture_stream,
                              capture_error_mode="thread_local"):
            self._prepare()
        self.post_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.post_graph, stream=capture_stream,
                              capture_error_mode="thread_local"):
            self._finish()
            self._node()
        current.wait_stream(capture_stream)
        self.completion = torch.cuda.Event()
        self.completion.record(current)

    def _node(self):
        candidate_covariance = torch.where(self.active | self.blocked, 0.,
                                           self.covariance)
        current, entering_index = candidate_covariance.abs().max(dim=1)
        self.current.copy_(current)
        self.entering_index.copy_(entering_index)
        stopping = (~self.done) & (current <= self.node_threshold)
        interpolate = (stopping & self.has_previous &
                       ((current - self.alpha).abs() > self.tolerance))
        denominator = self.previous_c - current
        valid_segment = (denominator > 0) & torch.isfinite(denominator)
        self.failed |= (interpolate & ~valid_segment).any()
        fraction = ((self.previous_c - self.alpha) /
                    torch.where(valid_segment, denominator, 1.))
        interpolated = (self.previous_code + fraction[:, None] *
                        (self.code - self.previous_code))
        self.code.copy_(torch.where((interpolate & valid_segment)[:, None],
                                    interpolated, self.code))
        self.done |= stopping
        return current, entering_index

    def _prepare(self):
        current, entering_index = self.current, self.entering_index
        adding = (~self.done) & self.add_atom
        self.adding.copy_(adding)
        selected_covariance = self.covariance[self.rows, entering_index]
        self.active[self.rows, entering_index] |= adding
        self.signs[self.rows, entering_index] = torch.where(
            adding, selected_covariance.sign(), self.signs[self.rows, entering_index])
        self.covariance[self.rows, entering_index] = torch.where(
            adding, 0., self.covariance[self.rows, entering_index])
        masked = (self.gram[None] * self.active[:, :, None] * self.active[:, None, :] +
                  torch.diag_embed((~self.active).to(self.response.dtype)))
        self.masked.copy_(masked)

    def _solve(self):
        solved = torch.linalg.solve_ex(self.masked, self.signs[..., None],
                                       check_errors=False)
        self.raw_result.copy_(solved.result[..., 0])
        self.solve_info.copy_(solved.info)

    def _finish(self):
        current, entering_index, adding = self.current, self.entering_index, self.adding
        raw_direction = self.raw_result
        signed_sum = (raw_direction * self.signs).sum(dim=1)
        bad_system = ((self.solve_info != 0) | ~torch.isfinite(raw_direction).all(dim=1) |
                      ~torch.isfinite(signed_sum) | (signed_sum <= 0)) & ~self.done
        rejected = bad_system & adding
        self.active[self.rows, entering_index] &= ~rejected
        self.signs[self.rows, entering_index] = torch.where(
            rejected, 0., self.signs[self.rows, entering_index])
        self.blocked[self.rows, entering_index] |= rejected
        self.failed |= (bad_system & ~adding).any()
        processing = (~self.done) & ~bad_system
        raw_direction = torch.where(processing[:, None], raw_direction, 0.)
        scale = torch.rsqrt(torch.where(processing, signed_sum, 1.))
        direction = raw_direction * scale[:, None]
        slope = torch.round(direction @ self.gram, decimals=self.decimals)
        possible = (~self.active) & (~self.blocked) & processing[:, None]
        positive = ((current[:, None] - self.covariance) /
                    (scale[:, None] - slope + self.tiny))
        negative = ((current[:, None] + self.covariance) /
                    (scale[:, None] + slope + self.tiny))
        positive = torch.where(possible & (positive > 0), positive, torch.inf)
        negative = torch.where(possible & (negative > 0), negative, torch.inf)
        gamma = torch.minimum(positive.amin(dim=1), negative.amin(dim=1))
        gamma = torch.minimum(gamma, current / scale)
        crossing = -self.code / (direction + self.tiny)
        crossing = torch.where(self.active & processing[:, None] & (crossing > 0),
                               crossing, torch.inf)
        drop_step = crossing.amin(dim=1)
        dropping = processing & (drop_step < gamma)
        gamma = torch.where(dropping, drop_step, gamma)
        gamma = torch.where(processing, gamma, 0.)
        self.previous_code.copy_(torch.where(processing[:, None], self.code,
                                             self.previous_code))
        self.previous_c.copy_(torch.where(processing, current, self.previous_c))
        self.has_previous |= processing
        updated_code = torch.where(self.active, self.code + gamma[:, None] * direction, 0.)
        self.code.copy_(torch.where(processing[:, None], updated_code, self.code))
        self.covariance.copy_(torch.where(possible, self.covariance - gamma[:, None] * slope,
                                          self.covariance))
        drop_mask = self.active & dropping[:, None] & (crossing == drop_step[:, None])
        dropped_covariance = self.response - self.code @ self.gram
        self.covariance.copy_(torch.where(drop_mask, dropped_covariance, self.covariance))
        self.active &= ~drop_mask
        self.signs.copy_(torch.where(drop_mask, 0., self.signs))
        self.add_atom.copy_(torch.where(processing, ~dropping,
                                        torch.ones_like(self.add_atom)))

    def _event(self):
        self.pre_graph.replay()
        self._solve()
        self.post_graph.replay()

    def __call__(self, samples, dictionary, alpha, max_events):
        stream = torch.cuda.current_stream(samples.device)
        # A retained output clone may still be reading these buffers on the
        # preceding call's stream. Never reset them before that copy finishes.
        stream.wait_event(self.completion)
        try:
            return self._run(samples, dictionary, alpha, max_events)
        finally:
            self.completion.record(stream)

    def _run(self, samples, dictionary, alpha, max_events):
        self.gram.copy_(dictionary @ dictionary.T)
        self.response.copy_(samples @ dictionary.T)
        self.code.zero_()
        self.covariance.copy_(self.response)
        self.active.zero_()
        self.signs.zero_()
        self.blocked.zero_()
        self.done.zero_()
        self.add_atom.fill_(True)
        self.previous_code.zero_()
        self.previous_c.copy_(self.response.abs().amax(dim=1))
        self.has_previous.zero_()
        self.failed.zero_()
        self.alpha.fill_(alpha)
        self.tolerance.fill_(samples.shape[1] * np.finfo(np.float32).eps)
        # Eager computes this scalar sum before casting it to the tensor
        # dtype. Adding two already-rounded float32 buffers changes a node.
        self.node_threshold.fill_(alpha + samples.shape[1] * np.finfo(np.float32).eps)
        limit = max_events or 3 * dictionary.shape[0]
        self._node()
        event = 0
        while True:
            if event % 4 == 0 and bool(self.done.all()):
                break
            if event == limit:
                break
            if event + 4 <= limit:
                for _ in range(4):
                    self._event()
                event += 4
            else:
                # At most three events remain. Preserve the final node-only
                # check rather than replaying events beyond sparse_iterations.
                self._event()
                event += 1
        if bool(self.failed):
            raise ValueError("Compatible LARS active Gram or interpolation failed")
        if not bool(self.done.all()):
            raise ValueError("Compatible LARS path exceeded sparse_iterations; increase the limit")
        # Callers can retain a result while the same workspace is reused.
        return self.code.clone()


_COMPATIBLE_LARS_GRAPH_LOCAL = threading.local()
_COMPATIBLE_LARS_GRAPH_CAPTURE_LOCK = threading.Lock()
_COMPATIBLE_LARS_GRAPH_CACHE_LIMIT = 4


def _compatible_lars_graph_cache():
    # Separate workspaces preserve independent concurrent subject calls.
    cache = getattr(_COMPATIBLE_LARS_GRAPH_LOCAL, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _COMPATIBLE_LARS_GRAPH_LOCAL.cache = cache
    return cache


def _sparse_codes_lars_compatible(samples: torch.Tensor, dictionary: torch.Tensor,
                                  alpha: float,
                                  max_events: int | None = None) -> torch.Tensor:
    """Use unchanged LARS arithmetic, with bounded CUDA launch graphs.

    CPU and unsupported CUDA dimensions retain the original eager path.
    The dense LU solve stays eager because some batched backends cannot be
    captured. Only its preceding and following tensor operations are graphed.
    """
    if (samples.device.type != "cuda" or samples.ndim != 2 or dictionary.ndim != 2 or
            samples.device != dictionary.device or samples.dtype != dictionary.dtype or
            samples.shape[1] != dictionary.shape[1]):
        return _sparse_codes_lars_compatible_eager(samples, dictionary, alpha, max_events)
    batch, atoms = samples.shape[0], dictionary.shape[0]
    limit = operator.index(max_events or 3 * atoms)
    if (batch < 1 or batch > 32 or atoms < 1 or atoms > 256 or limit < 0 or
            samples.requires_grad or dictionary.requires_grad or
            samples.dtype not in (torch.float32, torch.float64)):
        return _sparse_codes_lars_compatible_eager(samples, dictionary, alpha, max_events)
    # Captured matmul kernels retain the precision/determinism options used
    # during capture. A later option change must select a separate graph.
    key = (samples.device.index, samples.dtype, batch, atoms,
           torch.backends.cuda.matmul.allow_tf32,
           torch.get_float32_matmul_precision(),
           torch.are_deterministic_algorithms_enabled())
    cache = _compatible_lars_graph_cache()
    with torch.cuda.device(samples.device):
        if torch.cuda.is_current_stream_capturing():
            return _sparse_codes_lars_compatible_eager(samples, dictionary, alpha, max_events)
        solver = cache.pop(key, None)
        if solver is None:
            while len(cache) >= _COMPATIBLE_LARS_GRAPH_CACHE_LIMIT:
                _, discarded = cache.popitem(last=False)
                discarded.completion.synchronize()
            # Graph construction is rare. Serializing capture protects CUDA
            # allocator/library capture state without serializing solves.
            with _COMPATIBLE_LARS_GRAPH_CAPTURE_LOCK:
                solver = _CompatibleLarsGraph(batch, atoms, samples.device, samples.dtype)
        cache[key] = solver
        return solver(samples, dictionary, alpha, max_events)


def _sparse_codes_lars_compatible_eager(samples: torch.Tensor, dictionary: torch.Tensor,
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
        self.fallback_rows = 0
        self.near_node_fallback_rows = 0
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
        self.fallback_rows += len(samples)
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
                # Each row is an independent sparse-code problem. Keep the
                # accepted polished rows and restart only sensitive paths.
                # nonzero replaces the previous host-side any check, so this
                # branch needs only one index-size synchronization.
                sensitive_indices = torch.nonzero(
                    nearby["sensitive_rows"], as_tuple=False).flatten()
                if sensitive_indices.numel():
                    self.near_node_fallback_count += 1
                    self.near_node_fallback_rows += sensitive_indices.numel()
                    compatible = self.fallback(
                        samples.index_select(0, sensitive_indices), dictionary,
                        alpha, max_events)
                    polished.index_copy_(0, sensitive_indices, compatible)
                    return polished
                self.admm_accepted_count += 1
                return polished
        return self.fallback(samples, dictionary, alpha, max_events)


if _dicl_triton is not None:
    @_dicl_triton.jit
    def _dicl_update_atom_kernel(D, A, B, PRODUCT, ATOM,
                            FEATURES: _dicl_tl.constexpr,
                            D_ROW: _dicl_tl.constexpr, D_COLUMN: _dicl_tl.constexpr,
                            A_ROW: _dicl_tl.constexpr, A_COLUMN: _dicl_tl.constexpr,
                            B_ROW: _dicl_tl.constexpr, B_COLUMN: _dicl_tl.constexpr,
                            PRODUCT_STRIDE: _dicl_tl.constexpr,
                            BLOCK: _dicl_tl.constexpr):
        offset = _dicl_tl.program_id(0) * BLOCK + _dicl_tl.arange(0, BLOCK)
        valid = offset < FEATURES
        old = _dicl_tl.load(D + ATOM * D_ROW + offset * D_COLUMN, valid, 0)
        target = _dicl_tl.load(B + offset * B_ROW + ATOM * B_COLUMN, valid, 0)
        product = _dicl_tl.load(PRODUCT + offset * PRODUCT_STRIDE, valid, 0)
        diagonal = _dicl_tl.load(A + ATOM * (A_ROW + A_COLUMN))
        # Preserve the original subtraction, division, then addition; contraction
        # is disabled at launch, and division is rounded to nearest for fp32/fp64.
        correction = _dicl_libdevice.div_rn(target - product, diagonal)
        _dicl_tl.store(D + ATOM * D_ROW + offset * D_COLUMN, old + correction, valid)


    @_dicl_triton.jit
    def _dicl_normalize_atom_kernel(D, NORM, ATOM,
                               FEATURES: _dicl_tl.constexpr,
                               D_ROW: _dicl_tl.constexpr, D_COLUMN: _dicl_tl.constexpr,
                               BLOCK: _dicl_tl.constexpr):
        offset = _dicl_tl.program_id(0) * BLOCK + _dicl_tl.arange(0, BLOCK)
        valid = offset < FEATURES
        # torch.clamp_min propagates NaN. Triton's default maximum need not.
        norm = _dicl_tl.maximum(_dicl_tl.load(NORM), 1., propagate_nan=_dicl_tl.PropagateNan.ALL)
        old = _dicl_tl.load(D + ATOM * D_ROW + offset * D_COLUMN, valid, 0)
        _dicl_tl.store(D + ATOM * D_ROW + offset * D_COLUMN,
                 _dicl_libdevice.div_rn(old, norm), valid)


def _dicl_update_atom_after_matvec(dictionary, a, b, product, atom: int) -> None:
    """In-place ``D[j] += (B[:, j] - product) / A[j, j]`` on CUDA.

    ``product`` is the caller's unchanged ``A[j] @ D``. Strides are respected;
    no full-matrix copy, reduction, RNG draw, or atom reordering is performed.
    """
    features = dictionary.shape[1]
    _dicl_update_atom_kernel[(_dicl_triton.cdiv(features, 128),)](
        dictionary, a, b, product, atom, features,
        dictionary.stride(0), dictionary.stride(1),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), product.stride(0),
        128, enable_fp_fusion=False)


def _dicl_normalize_atom_after_torch_norm(dictionary, norm, atom: int) -> None:
    """In-place ``D[j] /= norm.clamp_min(1)`` on CUDA without a reduction.

    ``norm`` is the caller's unchanged ``torch.linalg.vector_norm(D[j])``.
    """
    features = dictionary.shape[1]
    _dicl_normalize_atom_kernel[(_dicl_triton.cdiv(features, 128),)](
        dictionary, norm, atom, features,
        dictionary.stride(0), dictionary.stride(1),
        128, enable_fp_fusion=False)


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
        # Keep cuBLAS matvec, Torch norm reduction, and Gauss-Seidel atom order.
        # Only the intervening elementwise operations are fused on CUDA.
        fused = (_dicl_triton is not None and self.dictionary.is_cuda and
                 self.dictionary.dtype in (torch.float32, torch.float64))
        for atom in range(len(self.dictionary)):
            if fused:
                product = self.a[atom] @ self.dictionary
                _dicl_update_atom_after_matvec(
                    self.dictionary, self.a, self.b, product, atom)
                norm = torch.linalg.vector_norm(self.dictionary[atom])
                _dicl_normalize_atom_after_torch_norm(self.dictionary, norm, atom)
            else:
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
