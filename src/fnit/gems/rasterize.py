"""GPU piecewise-linear rasterization of a tetrahedral GEMS atlas."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch


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

    for points, ids, point_ids, (x, y, z) in block_index.device_blocks(vertices.device, vertices.dtype):
        cells = tetrahedra[ids]
        tet = vertices[cells]  # M,4,3
        v0 = tet[:, 0]
        matrix = torch.stack((tet[:, 1] - v0, tet[:, 2] - v0, tet[:, 3] - v0), dim=-1)
        inv, info = torch.linalg.inv_ex(matrix, check_errors=False)
        rel = points[:, None, :] - v0[None, :, :]
        w123 = torch.einsum("mij,pmj->pmi", inv, rel)
        weights = torch.cat((1.0 - w123.sum(-1, keepdim=True), w123), dim=-1)
        singular = (info != 0) | (torch.linalg.det(matrix).abs() <= 1e-10)
        score = weights.amin(-1).masked_fill(singular[None], -torch.inf)
        best_score, best = score.max(dim=1)
        valid = best_score >= -float(tolerance)
        selected_cells = cells[best]
        selected_weights = weights[point_ids, best]
        values = (alphas[selected_cells] * selected_weights[..., None]).sum(dim=1)
        values = values.clamp_min(0)
        values = values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)

        out[x, y, z] = torch.where(valid[:, None], values, 0)
        covered[x, y, z] = valid
        if return_assignment:
            assigned_cells[x, y, z] = torch.where(valid[:, None], selected_cells, 0)
            assigned_weights[x, y, z] = torch.where(valid[:, None], selected_weights, 0)

    if background_channel is not None:
        bg = int(background_channel)
        if bg < 0 or bg >= k:
            raise ValueError("background_channel is out of range")
        missing = ~covered
        out[..., bg] = torch.where(missing, torch.ones_like(out[..., bg]), out[..., bg])
    result = (out.permute(3, 0, 1, 2).contiguous(), covered)
    return (*result, assigned_cells, assigned_weights) if return_assignment else result
