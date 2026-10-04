"""GPU piecewise-linear rasterization of a tetrahedral GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from ._raster_triton import (fused_data_cost, lookup_candidates as _lookup_candidates_cuda,
                            supports_fused_data_cost)
from .deformation import (CurrentGeometry, ordered_row_gather,
                          prepare_vertex_reduction)


def lookup_candidates(points, *args, **kwargs):
    """Use the CPU ordered scan or the existing CUDA lookup.

    Only the discrete owner search is compiled on CPU. Selected interpolation,
    prior values and their gradients retain the existing PyTorch operations.
    Unsupported CPU formats request the original dense Torch fallback.
    """
    if points.device.type == "cpu":
        from ._raster_cpu import lookup_candidates_cpu
        return lookup_candidates_cpu(points, *args, **kwargs)
    return _lookup_candidates_cuda(points, *args, **kwargs)


def _packed_device_buffer(parts, dtype, device):
    if not parts:
        return torch.empty((0,), device=device, dtype=dtype)
    host = torch.as_tensor(np.concatenate(parts, axis=None), dtype=dtype)
    return host.to(device=device)


@dataclass(frozen=True)
class BlockIndex:
    shape: tuple[int, int, int]
    block_size: int
    candidates: tuple[np.ndarray, ...]
    _device_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def device_blocks(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._device_cache:
            blocks = []
            for block_id, ids_np in enumerate(self.candidates):
                if len(ids_np) == 0:
                    continue
                points, _ = _block_points(block_id, self, device, dtype)
                blocks.append((points,
                               torch.as_tensor(ids_np, device=device, dtype=torch.long),
                               torch.arange(len(points), device=device),
                               points.long().unbind(-1)))
            self._device_cache[key] = tuple(blocks)
        return self._device_cache[key]

    def device_batches(self, device, dtype):
        key = ("batches", str(device), dtype)
        if key not in self._device_cache:
            groups = {}
            for block_id, ids in enumerate(self.candidates):
                if len(ids) == 0:
                    continue
                points = _block_points_array(block_id, self)
                width = 1 << (len(ids) - 1).bit_length()
                groups.setdefault((len(points), width), []).append((points, ids))
            point_parts, id_parts, mask_parts, batch_shapes = [], [], [], []
            for (n_points, width), blocks in groups.items():
                batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
                for start in range(0, len(blocks), batch_size):
                    subset = blocks[start:start + batch_size]
                    point_parts.append(np.stack([block[0] for block in subset]))
                    id_parts.append(np.stack([np.pad(block[1], (0, width - len(block[1]))) for block in subset]))
                    counts = np.asarray([len(block[1]) for block in subset])
                    mask_parts.append(np.arange(width)[None] < counts[:, None])
                    batch_shapes.append((len(subset), n_points, width))
            # The dense route never creates the old per-block CUDA tensors.
            # Coordinates are converted on CPU and transferred as two packed
            # buffers (floating interpolation points and integer write indices).
            packed_points = _packed_device_buffer(point_parts, dtype, device)
            packed_coordinates = _packed_device_buffer(point_parts, torch.long, device)
            packed_ids = _packed_device_buffer(id_parts, torch.long, device)
            packed_masks = _packed_device_buffer(mask_parts, torch.bool, device)
            batch_ids = torch.arange(max((item[0] for item in batch_shapes), default=0), device=device)[:, None]
            batches = []
            point_offset = id_offset = 0
            for count, n_points, width in batch_shapes:
                point_count, id_count = count * n_points * 3, count * width
                points = packed_points[point_offset:point_offset + point_count].view(count, n_points, 3)
                coordinates = packed_coordinates[point_offset:point_offset + point_count].view(-1, 3).unbind(-1)
                ids = packed_ids[id_offset:id_offset + id_count].view(count, width)
                candidate_mask = packed_masks[id_offset:id_offset + id_count].view(count, width)
                batches.append((points, ids, candidate_mask, batch_ids[:count], coordinates))
                point_offset += point_count
                id_offset += id_count
            self._device_cache[key] = tuple(batches)
        return self._device_cache[key]

    def device_compact_batches(self, valid_mask, device, dtype):
        key = ("compact", str(device), dtype, id(valid_mask))
        version = None if torch.is_inference(valid_mask) else valid_mask._version
        cached = self._device_cache.get(key)
        if cached is not None and cached[1] == version:
            return cached[2]
        coordinates = np.argwhere(valid_mask.detach().cpu().numpy())
        nblocks = tuple(int(math.ceil(s / self.block_size)) for s in self.shape)
        block_xyz = coordinates // self.block_size
        block_ids = (block_xyz[:, 0] * nblocks[1] + block_xyz[:, 1]) * nblocks[2] + block_xyz[:, 2]
        order = np.argsort(block_ids, kind="stable")
        sorted_ids = block_ids[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(order)]
        groups = {}
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if start == stop:
                continue
            ids = self.candidates[int(sorted_ids[start])]
            if not len(ids):
                continue
            rows = order[start:stop]
            width = 1 << (len(ids) - 1).bit_length()
            n_points = 1 << (len(rows) - 1).bit_length()
            groups.setdefault((n_points, width), []).append((coordinates[rows], ids, rows))
        batch_shapes = []
        point_parts, id_parts, mask_parts, row_parts = [], [], [], []
        reorder = np.full(len(coordinates), -1, dtype=np.int64)
        offset = 0
        for (n_points, width), blocks in groups.items():
            batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
            for start in range(0, len(blocks), batch_size):
                subset = blocks[start:start + batch_size]
                point_parts.append(np.stack([np.pad(block[0], ((0, n_points - len(block[0])), (0, 0)),
                                                    mode="edge") for block in subset]))
                id_parts.append(np.stack([np.pad(block[1], (0, width - len(block[1])))
                                          for block in subset]))
                counts = np.asarray([len(block[1]) for block in subset])
                mask_parts.append(np.arange(width)[None] < counts[:, None])
                rows = np.concatenate([block[2] for block in subset])
                # Only the lookup evaluates padding; interpolation and its
                # autograd graph retain real points using this static mapping.
                row_parts.append(np.concatenate([
                    np.arange(len(block[0])) + i * n_points for i, block in enumerate(subset)]))
                reorder[rows] = np.arange(offset, offset + len(rows))
                offset += len(rows)
                batch_shapes.append((len(subset), n_points, width, len(rows)))
        # A refreshed index can contain hundreds of batches. Transfer each
        # homogeneous buffer once, then create contiguous GPU views; per-batch
        # H2D copies and CUDA arange/compare launches used to dominate cold fits.
        packed_points = _packed_device_buffer(point_parts, dtype, device)
        packed_ids = _packed_device_buffer(id_parts, torch.long, device)
        packed_masks = _packed_device_buffer(mask_parts, torch.bool, device)
        packed_rows = _packed_device_buffer(row_parts, torch.long, device)
        batch_ids = torch.arange(max((item[0] for item in batch_shapes), default=0), device=device)[:, None]
        batches = []
        point_offset = id_offset = row_offset = 0
        for count, n_points, width, real_count in batch_shapes:
            point_count, id_count = count * n_points * 3, count * width
            points = packed_points[point_offset:point_offset + point_count].view(count, n_points, 3)
            ids = packed_ids[id_offset:id_offset + id_count].view(count, width)
            candidate_mask = packed_masks[id_offset:id_offset + id_count].view(count, width)
            point_rows = packed_rows[row_offset:row_offset + real_count]
            batches.append((points, ids, candidate_mask, batch_ids[:count], point_rows))
            point_offset += point_count
            id_offset += id_count
            row_offset += real_count
        result = (tuple(batches), torch.as_tensor(reorder, device=device))
        # Keep the mask alive so a later tensor cannot reuse its identity.
        self._device_cache[key] = (valid_mask, version, result)
        return result


def build_block_index(vertices: np.ndarray, tetrahedra: np.ndarray,
                      shape: tuple[int, int, int], block_size: int = 8,
                      margin: float = 1.0) -> BlockIndex:
    """Build a CPU bounding-box index, retaining tetrahedron order in each block."""
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    nblocks = tuple(int(math.ceil(s / block_size)) for s in shape)
    block_count = int(np.prod(nblocks))
    if len(tetrahedra) == 0:
        return BlockIndex(tuple(shape), int(block_size),
                          tuple(np.empty(0, dtype=np.int64) for _ in range(block_count)))
    xyz = vertices[tetrahedra]
    lo = np.floor(xyz.min(1) - margin).astype(int)
    hi = np.ceil(xyz.max(1) + margin).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape) - 1)
    blo, bhi = lo // block_size, hi // block_size
    widths = bhi - blo + 1
    counts = widths.prod(axis=1, dtype=np.int64)
    counts[np.any(hi < lo, axis=1)] = 0
    pair_count = int(counts.sum())
    if pair_count == 0:
        return BlockIndex(tuple(shape), int(block_size),
                          tuple(np.empty(0, dtype=np.int64) for _ in range(block_count)))
    # Bound the global sorting buffers as well as the coordinate expansion.
    # Unusually large boxes retain the original low-temporary-memory traversal.
    if pair_count > 8_388_608:
        lists: list[list[int]] = [[] for _ in range(block_count)]
        for tid in range(len(tetrahedra)):
            if np.any(hi[tid] < lo[tid]):
                continue
            for bx in range(blo[tid, 0], bhi[tid, 0] + 1):
                for by in range(blo[tid, 1], bhi[tid, 1] + 1):
                    base = (bx * nblocks[1] + by) * nblocks[2]
                    for bz in range(blo[tid, 2], bhi[tid, 2] + 1):
                        lists[base + bz].append(tid)
        return BlockIndex(tuple(shape), int(block_size),
                          tuple(np.asarray(x, dtype=np.int64) for x in lists))

    tetra_ids = np.repeat(np.arange(len(tetrahedra), dtype=np.int64), counts)
    starts = np.r_[0, np.cumsum(counts[:-1])]
    block_ids = np.empty(pair_count, dtype=np.int64)
    for start in range(0, pair_count, 1_048_576):
        stop = min(start + 1_048_576, pair_count)
        tids = tetra_ids[start:stop]
        offset = np.arange(start, stop, dtype=np.int64) - starts[tids]
        bx = blo[tids, 0] + offset // (widths[tids, 1] * widths[tids, 2])
        by = blo[tids, 1] + (offset // widths[tids, 2]) % widths[tids, 1]
        bz = blo[tids, 2] + offset % widths[tids, 2]
        block_ids[start:stop] = (bx * nblocks[1] + by) * nblocks[2] + bz
    order = np.argsort(block_ids, kind="stable")
    packed = tetra_ids[order]
    boundaries = np.cumsum(np.bincount(block_ids, minlength=block_count))
    return BlockIndex(tuple(shape), int(block_size),
                      tuple(np.split(packed, boundaries[:-1])))


def _block_points_array(block_id: int, index: BlockIndex):
    bs = index.block_size
    nb = tuple(int(math.ceil(s / bs)) for s in index.shape)
    bx = block_id // (nb[1] * nb[2])
    rem = block_id % (nb[1] * nb[2])
    by, bz = divmod(rem, nb[2])
    xyz = [np.arange(block * bs, min((block + 1) * bs, size), dtype=np.int64)
           for block, size in zip((bx, by, bz), index.shape)]
    return np.stack(np.meshgrid(*xyz, indexing="ij"), -1).reshape(-1, 3)


def _block_points(block_id: int, index: BlockIndex, device, dtype):
    sx, sy, sz = index.shape
    bs = index.block_size
    nb = tuple(int(math.ceil(s / bs)) for s in index.shape)
    bx = block_id // (nb[1] * nb[2])
    rem = block_id % (nb[1] * nb[2])
    by, bz = divmod(rem, nb[2])
    xr = torch.arange(bx * bs, min((bx + 1) * bs, sx), device=device, dtype=dtype)
    yr = torch.arange(by * bs, min((by + 1) * bs, sy), device=device, dtype=dtype)
    zr = torch.arange(bz * bs, min((bz + 1) * bs, sz), device=device, dtype=dtype)
    grid = torch.stack(torch.meshgrid(xr, yr, zr, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3), (xr.long(), yr.long(), zr.long())


def rasterize_priors(
    vertices: torch.Tensor,
    tetrahedra: torch.Tensor,
    alphas: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    block_index: BlockIndex | None = None,
    block_size: int = 8,
    tolerance: float = 2e-5,
    background_channel: int | None = 0,
    return_assignment: bool = False,
    current_geometry: CurrentGeometry | None = None,
) -> tuple[torch.Tensor, ...]:
    """Rasterize node alphas to a dense ``[K,X,Y,Z]`` prior tensor.

    Tetrahedron lookup uses a conservative block index.  Barycentric solves,
    inside tests, interpolation and output tensors execute on ``vertices.device``.
    The implementation is differentiable with respect to the selected
    tetrahedron's vertex coordinates, enabling PyTorch mesh optimization.
    ``current_geometry`` may share this evaluation's solves with the mesh prior.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [V,3]")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must be [T,4]")
    if alphas.ndim != 2 or alphas.shape[0] != vertices.shape[0]:
        raise ValueError("alphas must be [V,K]")
    shape = tuple(map(int, shape))
    if block_index is None:
        block_index = build_block_index(vertices.detach().cpu().numpy(),
                                        tetrahedra.detach().cpu().numpy(), shape, block_size)
    if block_index.shape != shape:
        raise ValueError("block index shape differs from requested raster grid")

    k = int(alphas.shape[1])
    # channel-last during indexed writes; permute at return.
    out = torch.zeros((*shape, k), device=vertices.device, dtype=alphas.dtype)
    covered = torch.zeros(shape, device=vertices.device, dtype=torch.bool)
    assigned_cells = (torch.zeros((*shape, 4), device=vertices.device, dtype=torch.long)
                      if return_assignment else None)
    assigned_weights = (torch.zeros((*shape, 4), device=vertices.device, dtype=vertices.dtype)
                        if return_assignment else None)

    # Each tetrahedron can appear in many blocks. Solve its geometry once;
    # autograd accumulates the block contributions before differentiating it.
    if current_geometry is None:
        all_tet = vertices[tetrahedra]
        all_v0 = all_tet[:, 0]
        all_matrix = torch.stack((all_tet[:, 1] - all_v0, all_tet[:, 2] - all_v0,
                                  all_tet[:, 3] - all_v0), dim=-1)
        all_inv, all_info = torch.linalg.inv_ex(all_matrix, check_errors=False)
        all_singular = (all_info != 0) | (torch.linalg.det(all_matrix).abs() <= 1e-10)
    else:
        all_v0, all_inv = current_geometry.origins, current_geometry.inverse_edges
        all_singular = current_geometry.singular
    for points, ids, candidate_mask, batch_ids, (x, y, z) in block_index.device_batches(vertices.device, vertices.dtype):
        # Lookup is discrete. Retaining every candidate's graph consumes GBs
        # although only one tetrahedron per voxel contributes to the gradient.
        with torch.no_grad():
            rows = torch.arange(points.shape[0] * points.shape[1], device=vertices.device)
            lookup = lookup_candidates(points, ids, candidate_mask, all_v0,
                                       all_inv, all_singular, rows, tolerance=tolerance)
            if lookup is None:
                rel = points[:, :, None, :] - all_v0[ids][:, None]
                w123 = torch.einsum("bcij,bpcj->bpci", all_inv[ids], rel)
                weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), dim=-1)
                singular = all_singular[ids] | ~candidate_mask
                score = weights.amin(-1).masked_fill(singular[:, None], -torch.inf)
                best_score, best = score.max(dim=2)
                selected_ids = ids[batch_ids, best]
                valid = best_score >= -float(tolerance)
            else:
                selected_ids = lookup[0].reshape(points.shape[:2])
                valid = lookup[1].reshape(points.shape[:2])
        selected_cells = tetrahedra[selected_ids]
        selected_rel = points - all_v0[selected_ids]
        selected_w123 = torch.einsum("bpij,bpj->bpi", all_inv[selected_ids], selected_rel)
        selected_weights = torch.cat((1.0 - selected_w123.sum(-1, keepdim=True), selected_w123), dim=-1)
        values = (alphas[selected_cells] * selected_weights[..., None]).sum(dim=2)
        values = values.clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)

        out[x, y, z] = torch.where(valid[..., None], values, 0).reshape(-1, k)
        covered[x, y, z] = valid.flatten()
        if return_assignment:
            assigned_cells[x, y, z] = torch.where(valid[..., None], selected_cells, 0).reshape(-1, 4)
            assigned_weights[x, y, z] = torch.where(valid[..., None], selected_weights, 0).reshape(-1, 4)

    if background_channel is not None:
        bg = int(background_channel)
        if bg < 0 or bg >= k:
            raise ValueError("background_channel is out of range")
        missing = ~covered
        out[..., bg] = torch.where(missing, torch.ones_like(out[..., bg]), out[..., bg])
    result = (out.permute(3, 0, 1, 2).contiguous(), covered)
    return (*result, assigned_cells, assigned_weights) if return_assignment else result


def _compact_lookup(vertices, tetrahedra, valid_mask, block_index,
                    current_geometry=None, tolerance=2e-5, *,
                    cache_owner_hints=False, owner_hints=False,
                    hint_tolerance=2e-4, hint_stats=None):
    batches, reorder = block_index.device_compact_batches(valid_mask, vertices.device, vertices.dtype)
    if current_geometry is None:
        all_tet = vertices[tetrahedra]
        all_v0 = all_tet[:, 0]
        all_matrix = torch.stack((all_tet[:, 1] - all_v0, all_tet[:, 2] - all_v0,
                                  all_tet[:, 3] - all_v0), dim=-1)
        all_inv, all_info = torch.linalg.inv_ex(all_matrix, check_errors=False)
        all_singular = (all_info != 0) | (torch.linalg.det(all_matrix).abs() <= 1e-10)
    else:
        all_v0, all_inv = current_geometry.origins, current_geometry.inverse_edges
        all_singular = current_geometry.singular
    selected_parts, point_parts, covered_parts, hint_parts = [], [], [], []
    for points, ids, candidate_mask, batch_ids, point_rows in batches:
        hint_key = ("owner_hints", id(valid_mask), id(tetrahedra),
                    None if torch.is_inference(tetrahedra) else tetrahedra._version,
                    id(points), id(ids), id(point_rows))
        previous = block_index._device_cache.get(hint_key) if owner_hints else None
        with torch.no_grad():
            lookup = lookup_candidates(points, ids, candidate_mask, all_v0, all_inv,
                                       all_singular, point_rows, tolerance=tolerance,
                                       previous_selected=previous, hint_tolerance=hint_tolerance,
                                       return_hint_hits=hint_stats is not None)
            if lookup is None:
                rel = points[:, :, None, :] - all_v0[ids][:, None]
                w123 = torch.einsum("bcij,bpcj->bpci", all_inv[ids], rel)
                weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), dim=-1)
                singular = all_singular[ids] | ~candidate_mask
                score = weights.amin(-1).masked_fill(singular[:, None], -torch.inf)
                best_score, best = score.max(dim=2)
                selected_ids = ids[batch_ids, best].flatten()[point_rows]
                covered = (best_score >= -float(tolerance)).flatten()[point_rows]
            else:
                selected_ids, covered = lookup[:2]
                if hint_stats is not None and lookup[2] is not None:
                    hint_parts.append(lookup[2])
            if cache_owner_hints:
                # Hints came from this block's current candidates. A replacement
                # index/mask layout or changed tetrahedron ordering starts empty.
                block_index._device_cache[hint_key] = selected_ids.detach()
        selected_parts.append(selected_ids)
        point_parts.append(points.reshape(-1, 3)[point_rows])
        covered_parts.append(covered.flatten())
    if selected_parts:
        selected_ids = torch.cat(selected_parts)
        selected_points = torch.cat(point_parts)
        covered = torch.cat(covered_parts)
    else:
        selected_ids = torch.empty((0,), device=vertices.device, dtype=torch.long)
        selected_points = torch.empty((0, 3), device=vertices.device, dtype=vertices.dtype)
        covered = torch.zeros((0,), device=vertices.device, dtype=torch.bool)
    if hint_stats is not None:
        hint_stats["reused_points"] = (torch.cat(hint_parts).sum() if hint_parts else
                                       torch.zeros((), device=vertices.device, dtype=torch.long))
        hint_stats["evaluated_points"] = selected_ids.numel()
        if hint_stats.get("include_assignments", False):
            hint_stats["selected_ids"] = selected_ids.detach()
            hint_stats["covered"] = covered.detach()
    return all_v0, all_inv, selected_ids, selected_points, covered, reorder


def compact_data_cost(vertices, tetrahedra, alphas, shape, *, valid_mask,
                      block_index, likelihood, background_channel=0,
                      current_geometry=None, tolerance=2e-5,
                      cache_owner_hints=False, owner_hints=False,
                      hint_tolerance=2e-4, hint_stats=None,
                      double_accumulation=False, deterministic_gradient=False):
    """Fused compact mixture cost, or ``None`` for the autograd fallback.

    CUDA FP32 evaluates every masked voxel with the same cell ownership as
    compact rasterization. Fixed alphas and Gaussian log likelihood are required;
    an analytic derivative is returned only for the vertex positions. This path
    avoids allocating the interpolated prior and its four-node autograd graph.
    ``deterministic_gradient`` sums packed-point corner contributions in a fixed
    vertex order with FP64 accumulation, returning the same FP32 gradient dtype.
    """
    if not supports_fused_data_cost(vertices, tetrahedra, alphas, likelihood):
        return None
    if tetrahedra.numel() == 0:
        return None
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != tuple(shape):
        raise ValueError("valid_mask must be a boolean tensor matching the raster grid")
    if block_index.shape != tuple(shape):
        raise ValueError("block index shape differs from requested raster grid")
    if background_channel is not None and not 0 <= int(background_channel) < alphas.shape[1]:
        raise ValueError("background_channel is out of range")
    all_v0, all_inv, selected_ids, points, covered, reorder = _compact_lookup(
        vertices, tetrahedra, valid_mask, block_index, current_geometry, tolerance,
        cache_owner_hints=cache_owner_hints, owner_hints=owner_hints,
        hint_tolerance=hint_tolerance, hint_stats=hint_stats)
    if likelihood.shape != (alphas.shape[1], reorder.numel()):
        raise ValueError("likelihood must be [classes, valid voxels]")
    return fused_data_cost(vertices, tetrahedra, alphas, points, selected_ids,
                           covered, reorder, all_v0, all_inv, likelihood,
                           background_channel, double_accumulation, deterministic_gradient)


def rasterize_priors_compact(
    vertices: torch.Tensor,
    tetrahedra: torch.Tensor,
    alphas: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    valid_mask: torch.Tensor,
    block_index: BlockIndex,
    tolerance: float = 2e-5,
    background_channel: int | None = 0,
    current_geometry: CurrentGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize only masked voxels, in the order of ``dense[:, valid_mask]``.

    Return ``[K,N_valid]`` priors and ``[N_valid]`` coverage. The spatial index
    caches fixed mask coordinates without retaining a mesh's autograd graph;
    replacing the index naturally rebuilds that mapping after mesh movement.
    ``current_geometry`` may share this evaluation's solves with the mesh prior.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be [V,3]")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must be [T,4]")
    if alphas.ndim != 2 or alphas.shape[0] != vertices.shape[0]:
        raise ValueError("alphas must be [V,K]")
    shape = tuple(map(int, shape))
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != shape:
        raise ValueError("valid_mask must be a boolean tensor matching the raster grid")
    if block_index.shape != shape:
        raise ValueError("block index shape differs from requested raster grid")
    k = int(alphas.shape[1])
    if background_channel is not None and not 0 <= int(background_channel) < k:
        raise ValueError("background_channel is out of range")
    all_v0, all_inv, selected_ids, selected_points, covered, reorder = _compact_lookup(
        vertices, tetrahedra, valid_mask, block_index, current_geometry, tolerance)
    if selected_ids.numel():
        selected_cells = tetrahedra[selected_ids]
        if (current_geometry is not None and current_geometry.deterministic_gradient
                and torch.is_grad_enabled() and (all_v0.requires_grad or all_inv.requires_grad)):
            # A tetrahedron owns many masked voxels. Reduce their backward
            # contributions in the same fixed point order before the existing
            # ordered tetrahedron-to-vertex reduction.
            point_reduction = prepare_vertex_reduction(selected_ids, len(all_v0))
            selected_v0 = ordered_row_gather(all_v0, selected_ids, point_reduction)
            selected_inv = ordered_row_gather(all_inv, selected_ids, point_reduction)
        else:
            selected_v0, selected_inv = all_v0[selected_ids], all_inv[selected_ids]
        selected_rel = selected_points - selected_v0
        selected_w123 = torch.einsum("pij,pj->pi", selected_inv, selected_rel)
        selected_weights = torch.cat((1.0 - selected_w123.sum(-1, keepdim=True), selected_w123), dim=-1)
        values = (alphas[selected_cells] * selected_weights[..., None]).sum(dim=1)
        values = values.clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        values = torch.where(covered[..., None], values, 0)
    else:
        values = torch.zeros((0, k), device=vertices.device, dtype=alphas.dtype)
        covered = torch.zeros((0,), device=vertices.device, dtype=torch.bool)

    # One gather restores mask order, including voxels in blocks with no cells.
    out = torch.cat((values, torch.zeros((1, k), device=vertices.device, dtype=alphas.dtype)))[reorder]
    covered = torch.cat((covered, torch.zeros((1,), device=vertices.device, dtype=torch.bool)))[reorder]
    if background_channel is not None:
        bg = int(background_channel)
        out[:, bg] = torch.where(~covered, torch.ones_like(out[:, bg]), out[:, bg])
    return out.T.contiguous(), covered
