"""One ordered CPU owner scan for a complete compact raster evaluation.

Static points and candidate lists are cached. Interpolation, cost and gradients
remain in the existing PyTorch implementation. CUDA never imports this module.
"""

from dataclasses import dataclass, field

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads
import torch

from ._raster_cpu import _fma32


@njit(cache=True, parallel=True, fastmath=False)
def _lookup_packed(points, point_blocks, offsets, candidates, default_owners,
                   origins, inverses, singular, tolerance, work_order):
    count = points.shape[0]
    selected = np.empty(count, dtype=np.int64)
    covered = np.zeros(count, dtype=np.bool_)
    for work in prange(count):
        row = work_order[work]
        block = point_blocks[row]
        owner = default_owners[block]
        best = -np.inf
        # The mask is static: packed candidates are exactly the original True
        # entries, in their original order. Masked entries performed no FP work.
        for column in range(offsets[block], offsets[block + 1]):
            cell = candidates[column]
            if singular[cell]:
                continue
            dx = points[row, 0] - origins[cell, 0]
            dy = points[row, 1] - origins[cell, 1]
            dz = points[row, 2] - origins[cell, 2]
            w1 = _fma32(inverses[cell, 0, 2], dz, _fma32(
                inverses[cell, 0, 1], dy, inverses[cell, 0, 0] * dx))
            w2 = _fma32(inverses[cell, 1, 2], dz, _fma32(
                inverses[cell, 1, 1], dy, inverses[cell, 1, 0] * dx))
            w3 = _fma32(inverses[cell, 2, 2], dz, _fma32(
                inverses[cell, 2, 1], dy, inverses[cell, 2, 0] * dx))
            w0 = np.float32(1) - ((w1 + w2) + w3)
            score = min(w0, w1, w2, w3)
            if np.isnan(w0) or np.isnan(w1) or np.isnan(w2) or np.isnan(w3):
                owner = cell
                best = np.nan
                break
            if score > best:
                owner = cell
                best = score
        selected[row] = owner
        covered[row] = best >= -tolerance
    return selected, covered


def _versions(batches):
    return tuple(None if torch.is_inference(value) else value._version
                 for points, ids, mask, _, rows in batches for value in (points, ids, mask, rows))


@dataclass
class _CompactPlan:
    batches: tuple
    versions: tuple
    cell_count: int
    points: torch.Tensor
    points_version: int | None
    point_blocks: np.ndarray
    offsets: np.ndarray
    candidates: np.ndarray
    default_owners: np.ndarray
    work_orders: dict = field(default_factory=dict)

    def work_order(self, threads):
        if threads not in self.work_orders:
            count = len(self.points)
            # Distribute small contiguous row chunks over thread slices. This
            # balances blocks with different candidate counts while preserving
            # the row assigned to each output and every candidate's FP order.
            quantum = 64
            chunks = np.arange((count + quantum - 1) // quantum, dtype=np.int64)
            chunks = np.concatenate([chunks[index::threads] for index in range(threads)])
            order = (chunks[:, None] * quantum + np.arange(quantum, dtype=np.int64)).reshape(-1)
            self.work_orders[threads] = order[order < count]
        return self.work_orders[threads]


def _build_plan(batches, cell_count):
    point_parts, block_parts, candidate_parts, owner_parts = [], [], [], []
    offsets = [0]
    block_offset = 0
    for points, ids, mask, _, rows in batches:
        values = (points, ids, mask, rows)
        if any(value.device.type != "cpu" for value in values):
            return None
        if (points.dtype != torch.float32 or ids.dtype != torch.long
                or mask.dtype != torch.bool or rows.dtype != torch.long):
            return None
        if (points.ndim != 3 or points.shape[-1] != 3 or ids.ndim != 2
                or ids.shape[0] != points.shape[0] or mask.shape != ids.shape
                or rows.ndim != 1 or not points.shape[1] or not ids.shape[1]):
            return None
        ids_array = ids.detach().numpy()
        mask_array = mask.detach().numpy()
        row_array = rows.detach().numpy()
        if ((ids_array.size and (ids_array.min() < 0 or ids_array.max() >= cell_count))
                or (row_array.size and (row_array.min() < 0
                                       or row_array.max() >= points.shape[0] * points.shape[1]))):
            return None
        # Advanced indexing creates private points. Source tensors remain alive
        # in the plan and version changes cause the complete plan to be rebuilt.
        point_parts.append(points.detach().numpy().reshape(-1, 3)[row_array])
        block_parts.append(row_array // points.shape[1] + block_offset)
        owner_parts.append(np.array(ids_array[:, 0], copy=True))
        for block_ids, block_mask in zip(ids_array, mask_array):
            kept = block_ids[block_mask]
            candidate_parts.append(kept)
            offsets.append(offsets[-1] + len(kept))
        block_offset += points.shape[0]
    if not point_parts:
        return None
    packed_points = torch.from_numpy(np.concatenate(point_parts, axis=0))
    return _CompactPlan(
        batches, _versions(batches), cell_count, packed_points,
        None if torch.is_inference(packed_points) else packed_points._version,
        np.concatenate(block_parts), np.asarray(offsets, dtype=np.int64),
        np.concatenate(candidate_parts), np.concatenate(owner_parts))


def lookup_compact_cpu(batches, origins, inverses, singular, cache, *, tolerance=2e-5):
    """Return selected IDs, fixed points and coverage, or None for the fallback.

    The output row order is the existing concatenated compact-batch order.
    Guarded CPU FP32 execution retains explicit FMA, first-candidate ties, NaN
    handling, singular-cell exclusion and FP32 scalar tolerance. No gradients,
    model precision or GPU backend settings are changed.
    """
    geometry = (origins, inverses, singular)
    if (any(value.device.type != "cpu" for value in geometry)
            or origins.dtype != torch.float32 or inverses.dtype != torch.float32
            or singular.dtype != torch.bool or origins.shape != (len(singular), 3)
            or inverses.shape != (len(singular), 3, 3) or not len(singular)
            or not torch.backends.mkl.is_available()):
        return None
    try:
        autocast_enabled = torch.is_autocast_enabled("cpu")
    except TypeError:
        autocast_enabled = torch.is_autocast_cpu_enabled()
    if autocast_enabled:
        return None
    key = ("cpu_compact_plan", id(batches))
    plan = cache.get(key)
    if (plan is None or plan.batches is not batches or plan.cell_count != len(singular)
            or plan.versions != _versions(batches)
            or (None if torch.is_inference(plan.points) else plan.points._version) != plan.points_version):
        plan = _build_plan(batches, len(singular))
        if plan is None:
            return None
        cache[key] = plan
    before_threads = get_num_threads()
    threads = min(torch.get_num_threads(), int(config.NUMBA_NUM_THREADS))
    try:
        if before_threads != threads:
            set_num_threads(threads)
        selected, covered = _lookup_packed(
            plan.points.detach().numpy(), plan.point_blocks, plan.offsets,
            plan.candidates, plan.default_owners, origins.detach().numpy(),
            inverses.detach().numpy(), singular.detach().numpy(), np.float32(tolerance),
            plan.work_order(threads))
    finally:
        if before_threads != threads:
            set_num_threads(before_threads)
    return torch.from_numpy(selected), plan.points, torch.from_numpy(covered)
