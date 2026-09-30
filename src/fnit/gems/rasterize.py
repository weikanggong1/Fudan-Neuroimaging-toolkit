"""GPU piecewise-linear rasterization of a tetrahedral GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from ._raster_triton import lookup_candidates


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
            for points, ids, _, _ in self.device_blocks(device, dtype):
                width = 1 << (len(ids) - 1).bit_length()
                groups.setdefault((len(points), width), []).append((points, ids))
            batches = []
            for (n_points, width), blocks in groups.items():
                batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
                for start in range(0, len(blocks), batch_size):
                    subset = blocks[start:start + batch_size]
                    points = torch.stack([block[0] for block in subset])
                    ids = torch.stack([torch.nn.functional.pad(block[1], (0, width - len(block[1])))
                                       for block in subset])
                    counts = torch.as_tensor([len(block[1]) for block in subset], device=device)
                    candidate_mask = torch.arange(width, device=device)[None] < counts[:, None]
                    batch_ids = torch.arange(len(subset), device=device)[:, None]
                    coordinates = points.reshape(-1, 3).long().unbind(-1)
                    batches.append((points, ids, candidate_mask, batch_ids, coordinates))
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
        batches = []
        reorder = np.full(len(coordinates), -1, dtype=np.int64)
        offset = 0
        for (n_points, width), blocks in groups.items():
            batch_size = max(1, min(16, 8_388_608 // (n_points * width)))
            for start in range(0, len(blocks), batch_size):
                subset = blocks[start:start + batch_size]
                points = torch.as_tensor(np.stack([np.pad(block[0], ((0, n_points - len(block[0])), (0, 0)),
                                                          mode="edge") for block in subset]),
                                         device=device, dtype=dtype)
                ids = torch.as_tensor(np.stack([np.pad(block[1], (0, width - len(block[1])))
                                               for block in subset]), device=device, dtype=torch.long)
                counts = torch.as_tensor([len(block[1]) for block in subset], device=device)
                candidate_mask = torch.arange(width, device=device)[None] < counts[:, None]
                batch_ids = torch.arange(len(subset), device=device)[:, None]
                rows = np.concatenate([block[2] for block in subset])
                # Only the lookup evaluates padding; interpolation and its
                # autograd graph retain real points using this static mapping.
                point_rows = torch.as_tensor(np.concatenate([
                    np.arange(len(block[0])) + i * n_points for i, block in enumerate(subset)]), device=device)
                reorder[rows] = np.arange(offset, offset + len(rows))
                offset += len(rows)
                batches.append((points, ids, candidate_mask, batch_ids, point_rows))
        result = (tuple(batches), torch.as_tensor(reorder, device=device))
        # Keep the mask alive so a later tensor cannot reuse its identity.
        self._device_cache[key] = (valid_mask, version, result)
        return result


def build_block_index(vertices: np.ndarray, tetrahedra: np.ndarray,
                      shape: tuple[int, int, int], block_size: int = 8,
                      margin: float = 1.0) -> BlockIndex:
    """Build a conservative CPU spatial index from tetrahedron bounding boxes.

    Geometry dispatch is inexpensive and deterministic; barycentric evaluation
    and alpha interpolation remain on the selected PyTorch device.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    nblocks = tuple(int(math.ceil(s / block_size)) for s in shape)
    lists: list[list[int]] = [[] for _ in range(np.prod(nblocks))]
    if len(tetrahedra) == 0:
        return BlockIndex(tuple(shape), int(block_size), tuple(np.asarray(x, dtype=np.int64) for x in lists))
    xyz = vertices[tetrahedra]
    lo = np.floor(xyz.min(1) - margin).astype(int)
    hi = np.ceil(xyz.max(1) + margin).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(shape) - 1)
    for tid in range(len(tetrahedra)):
        if np.any(hi[tid] < lo[tid]):
            continue
        blo = lo[tid] // block_size
        bhi = hi[tid] // block_size
        for bx in range(blo[0], bhi[0] + 1):
            for by in range(blo[1], bhi[1] + 1):
                base = (bx * nblocks[1] + by) * nblocks[2]
                for bz in range(blo[2], bhi[2] + 1):
                    lists[base + bz].append(tid)
    return BlockIndex(tuple(shape), int(block_size),
                      tuple(np.asarray(x, dtype=np.int64) for x in lists))


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
) -> tuple[torch.Tensor, ...]:
    """Rasterize node alphas to a dense ``[K,X,Y,Z]`` prior tensor.

    Tetrahedron lookup uses a conservative block index.  Barycentric solves,
    inside tests, interpolation and output tensors execute on ``vertices.device``.
    The implementation is differentiable with respect to the selected
    tetrahedron's vertex coordinates, enabling PyTorch mesh optimization.
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
    all_tet = vertices[tetrahedra]
    all_v0 = all_tet[:, 0]
    all_matrix = torch.stack((all_tet[:, 1] - all_v0, all_tet[:, 2] - all_v0,
                              all_tet[:, 3] - all_v0), dim=-1)
    all_inv, all_info = torch.linalg.inv_ex(all_matrix, check_errors=False)
    all_singular = (all_info != 0) | (torch.linalg.det(all_matrix).abs() <= 1e-10)
    for points, ids, candidate_mask, batch_ids, (x, y, z) in block_index.device_batches(vertices.device, vertices.dtype):
        # Lookup is discrete. Retaining every candidate's graph consumes GBs
        # although only one tetrahedron per voxel contributes to the gradient.
        with torch.no_grad():
            rel = points[:, :, None, :] - all_v0[ids][:, None]
            w123 = torch.einsum("bcij,bpcj->bpci", all_inv[ids], rel)
            weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), dim=-1)
            singular = all_singular[ids] | ~candidate_mask
            score = weights.amin(-1).masked_fill(singular[:, None], -torch.inf)
            best_score, best = score.max(dim=2)
            selected_ids = ids[batch_ids, best]
            valid = best_score >= -float(tolerance)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize only masked voxels, in the order of ``dense[:, valid_mask]``.

    Return ``[K,N_valid]`` priors and ``[N_valid]`` coverage. The spatial index
    caches fixed mask coordinates without retaining a mesh's autograd graph;
    replacing the index naturally rebuilds that mapping after mesh movement.
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
    batches, reorder = block_index.device_compact_batches(valid_mask, vertices.device, vertices.dtype)

    all_tet = vertices[tetrahedra]
    all_v0 = all_tet[:, 0]
    all_matrix = torch.stack((all_tet[:, 1] - all_v0, all_tet[:, 2] - all_v0,
                              all_tet[:, 3] - all_v0), dim=-1)
    all_inv, all_info = torch.linalg.inv_ex(all_matrix, check_errors=False)
    all_singular = (all_info != 0) | (torch.linalg.det(all_matrix).abs() <= 1e-10)
    values_parts, covered_parts = [], []
    for points, ids, candidate_mask, batch_ids, point_rows in batches:
        with torch.no_grad():
            lookup = lookup_candidates(points, ids, candidate_mask, all_v0, all_inv,
                                       all_singular, point_rows, tolerance=tolerance)
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
                selected_ids, covered = lookup
        selected_cells = tetrahedra[selected_ids]
        selected_rel = points.reshape(-1, 3)[point_rows] - all_v0[selected_ids]
        selected_w123 = torch.einsum("pij,pj->pi", all_inv[selected_ids], selected_rel)
        selected_weights = torch.cat((1.0 - selected_w123.sum(-1, keepdim=True), selected_w123), dim=-1)
        values = (alphas[selected_cells] * selected_weights[..., None]).sum(dim=1)
        values = values.clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        values_parts.append(torch.where(covered[..., None], values, 0).reshape(-1, k))
        covered_parts.append(covered.flatten())

    # One gather restores mask order, including voxels in blocks with no cells.
    values_parts.append(torch.zeros((1, k), device=vertices.device, dtype=alphas.dtype))
    covered_parts.append(torch.zeros((1,), device=vertices.device, dtype=torch.bool))
    out = torch.cat(values_parts)[reorder]
    covered = torch.cat(covered_parts)[reorder]
    if background_channel is not None:
        bg = int(background_channel)
        out[:, bg] = torch.where(~covered, torch.ones_like(out[:, bg]), out[:, bg])
    return out.T.contiguous(), covered
