"""Optional FP32 fused candidate lookup for compact GEMS rasterization."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _lookup_kernel(Points, Ids, CandidateMask, Origins, Inverse, Singular,
                       PointRows, Selected, Covered,
                       NPOINTS, WIDTH: tl.constexpr,
                       V0S0: tl.constexpr, V0S1: tl.constexpr,
                       IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
                       TOLERANCE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        point_row = tl.load(PointRows + row)
        batch = point_row // NPOINTS
        x = tl.load(Points + point_row * 3)
        y = tl.load(Points + point_row * 3 + 1)
        z = tl.load(Points + point_row * 3 + 2)
        candidates = tl.arange(0, BLOCK)
        lanes = candidates < WIDTH
        ids = tl.load(Ids + batch * WIDTH + candidates, mask=lanes, other=0)
        allowed = tl.load(CandidateMask + batch * WIDTH + candidates, mask=lanes, other=0)
        singular = tl.load(Singular + ids)
        rx = x - tl.load(Origins + ids * V0S0)
        ry = y - tl.load(Origins + ids * V0S0 + V0S1)
        rz = z - tl.load(Origins + ids * V0S0 + 2 * V0S1)
        base = ids * IS0
        w1 = ((tl.load(Inverse + base) * rx
               + tl.load(Inverse + base + IS2) * ry)
              + tl.load(Inverse + base + 2 * IS2) * rz)
        w2 = ((tl.load(Inverse + base + IS1) * rx
               + tl.load(Inverse + base + IS1 + IS2) * ry)
              + tl.load(Inverse + base + IS1 + 2 * IS2) * rz)
        w3 = ((tl.load(Inverse + base + 2 * IS1) * rx
               + tl.load(Inverse + base + 2 * IS1 + IS2) * ry)
              + tl.load(Inverse + base + 2 * IS1 + 2 * IS2) * rz)
        w0 = 1.0 - ((w1 + w2) + w3)
        score = tl.minimum(tl.minimum(w0, w1), tl.minimum(w2, w3))
        nan_weight = (w0 != w0) | (w1 != w1) | (w2 != w2) | (w3 != w3)
        score = tl.where(nan_weight, float("nan"), score)
        score = tl.where(lanes & allowed & ~singular, score, -float("inf"))
        nan_score = score != score
        maximum = tl.max(tl.where(nan_score, -float("inf"), score), axis=0)
        first_maximum = tl.min(tl.where(score == maximum, candidates, BLOCK), axis=0)
        first_nan = tl.min(tl.where(nan_score, candidates, BLOCK), axis=0)
        # torch.max returns the first NaN, otherwise the first equal maximum.
        best = tl.where(first_nan < BLOCK, first_nan, first_maximum)
        selected = tl.load(Ids + batch * WIDTH + best)
        covered = (first_nan == BLOCK) & (maximum >= -TOLERANCE)
        tl.store(Selected + row, selected)
        tl.store(Covered + row, covered)


@torch.no_grad()
def lookup_candidates(points, ids, candidate_mask, all_v0, all_inv,
                      all_singular, point_rows, *, tolerance=2e-5):
    """Return selected IDs and coverage, or ``None`` for the torch fallback.

    Inputs match ``BlockIndex.device_compact_batches`` and the geometry computed
    by PyTorch. Only real ``point_rows`` are searched. Geometry may be strided;
    coordinates and index tensors must be contiguous. All floating arithmetic
    uses FP32, with compiler contraction disabled; interpolation remains outside
    this nondifferentiable lookup. CUDA numerical acceptance is tested separately.
    """
    tensors = (points, ids, candidate_mask, all_v0, all_inv, all_singular, point_rows)
    if triton is None or not points.is_cuda or torch.version.hip is not None:
        return None
    if any(t.device != points.device for t in tensors):
        return None
    if any(t.dtype != torch.float32 for t in (points, all_v0, all_inv)):
        return None
    if ids.dtype != torch.long or point_rows.dtype != torch.long:
        return None
    if candidate_mask.dtype != torch.bool or all_singular.dtype != torch.bool:
        return None
    if any(not t.is_contiguous() for t in (points, ids, candidate_mask, all_singular, point_rows)):
        return None
    if points.ndim != 3 or points.shape[-1] != 3 or ids.ndim != 2:
        return None
    if ids.shape[0] != points.shape[0] or candidate_mask.shape != ids.shape:
        return None
    if all_v0.shape != (len(all_singular), 3) or all_inv.shape != (len(all_singular), 3, 3):
        return None
    if point_rows.ndim != 1 or not all_singular.numel() or not ids.shape[1] or not points.shape[1]:
        return None
    selected = torch.empty_like(point_rows)
    covered = torch.empty(point_rows.shape, device=points.device, dtype=torch.bool)
    if not point_rows.numel():
        return selected, covered
    width = int(ids.shape[1])
    warps = 1 if width <= 32 else 4 if width <= 512 else 8
    _lookup_kernel[(point_rows.numel(),)](
        points, ids, candidate_mask, all_v0, all_inv, all_singular,
        point_rows, selected, covered, int(points.shape[1]), width,
        *all_v0.stride(), *all_inv.stride(), float(tolerance), triton.next_power_of_2(width),
        num_warps=warps, enable_fp_fusion=True)
    return selected, covered


__all__ = ["lookup_candidates"]
