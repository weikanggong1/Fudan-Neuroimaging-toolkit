"""Optional FP32 fused candidate lookup for compact GEMS rasterization."""

import torch

from .deformation import ordered_vertex_sum

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


    @triton.jit
    def _lookup_hint_kernel(Points, Ids, CandidateMask, Origins, Inverse, Singular,
                            PointRows, Hints, Selected, Covered, Hits,
                            NPOINTS, WIDTH: tl.constexpr,
                            V0S0: tl.constexpr, V0S1: tl.constexpr,
                            IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
                            TOLERANCE: tl.constexpr, HINT_TOLERANCE: tl.constexpr,
                            BLOCK: tl.constexpr):
        row = tl.program_id(0)
        hint = tl.load(Hints + row)
        has_hint = hint >= 0
        safe_hint = tl.maximum(hint, 0)
        point_row = tl.load(PointRows + row)
        x = tl.load(Points + point_row * 3)
        y = tl.load(Points + point_row * 3 + 1)
        z = tl.load(Points + point_row * 3 + 2)
        singular = tl.load(Singular + safe_hint, mask=has_hint, other=1) != 0
        rx = x - tl.load(Origins + safe_hint * V0S0)
        ry = y - tl.load(Origins + safe_hint * V0S0 + V0S1)
        rz = z - tl.load(Origins + safe_hint * V0S0 + 2 * V0S1)
        base = safe_hint * IS0
        w1 = (tl.load(Inverse + base) * rx + tl.load(Inverse + base + IS2) * ry) + tl.load(Inverse + base + 2 * IS2) * rz
        w2 = (tl.load(Inverse + base + IS1) * rx + tl.load(Inverse + base + IS1 + IS2) * ry) + tl.load(Inverse + base + IS1 + 2 * IS2) * rz
        w3 = (tl.load(Inverse + base + 2 * IS1) * rx + tl.load(Inverse + base + 2 * IS1 + IS2) * ry) + tl.load(Inverse + base + 2 * IS1 + 2 * IS2) * rz
        w0 = 1.0 - ((w1 + w2) + w3)
        score = tl.minimum(tl.minimum(w0, w1), tl.minimum(w2, w3))
        reuse = has_hint & ~singular & (score > HINT_TOLERANCE)
        if reuse:
            tl.store(Selected + row, hint)
            tl.store(Covered + row, True)
        else:
            _lookup_kernel(Points, Ids, CandidateMask, Origins, Inverse, Singular,
                           PointRows, Selected, Covered, NPOINTS, WIDTH,
                           V0S0, V0S1, IS0, IS1, IS2, TOLERANCE, BLOCK)
        tl.store(Hits + row, reuse)


    @triton.jit
    def _data_cost_kernel(Points, Selected, Covered, Reorder, Tetrahedra,
                          Alphas, Origins, Inverse, Likelihood, Costs, Gradient,
                          NCLASS: tl.constexpr, NVALID: tl.constexpr,
                          AS0: tl.constexpr, AS1: tl.constexpr,
                          V0S0: tl.constexpr, V0S1: tl.constexpr,
                          IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
                          LS0: tl.constexpr, LS1: tl.constexpr,
                          BACKGROUND: tl.constexpr, BLOCK: tl.constexpr,
                          DETERMINISTIC: tl.constexpr):
        row = tl.program_id(0)
        packed = tl.load(Reorder + row)
        present = packed >= 0
        selected = tl.load(Selected + packed, mask=present, other=0)
        covered = tl.load(Covered + packed, mask=present, other=0) != 0
        x = tl.load(Points + packed * 3, mask=present, other=0)
        y = tl.load(Points + packed * 3 + 1, mask=present, other=0)
        z = tl.load(Points + packed * 3 + 2, mask=present, other=0)
        rx = x - tl.load(Origins + selected * V0S0)
        ry = y - tl.load(Origins + selected * V0S0 + V0S1)
        rz = z - tl.load(Origins + selected * V0S0 + 2 * V0S1)
        base = selected * IS0
        i00 = tl.load(Inverse + base)
        i01 = tl.load(Inverse + base + IS2)
        i02 = tl.load(Inverse + base + 2 * IS2)
        i10 = tl.load(Inverse + base + IS1)
        i11 = tl.load(Inverse + base + IS1 + IS2)
        i12 = tl.load(Inverse + base + IS1 + 2 * IS2)
        i20 = tl.load(Inverse + base + 2 * IS1)
        i21 = tl.load(Inverse + base + 2 * IS1 + IS2)
        i22 = tl.load(Inverse + base + 2 * IS1 + 2 * IS2)
        w1 = (i00 * rx + i01 * ry) + i02 * rz
        w2 = (i10 * rx + i11 * ry) + i12 * rz
        w3 = (i20 * rx + i21 * ry) + i22 * rz
        w0 = 1.0 - ((w1 + w2) + w3)
        v0 = tl.load(Tetrahedra + selected * 4)
        v1 = tl.load(Tetrahedra + selected * 4 + 1)
        v2 = tl.load(Tetrahedra + selected * 4 + 2)
        v3 = tl.load(Tetrahedra + selected * 4 + 3)
        channels = tl.arange(0, BLOCK)
        lanes = channels < NCLASS
        a0 = tl.load(Alphas + v0 * AS0 + channels * AS1, mask=lanes, other=0)
        a1 = tl.load(Alphas + v1 * AS0 + channels * AS1, mask=lanes, other=0)
        a2 = tl.load(Alphas + v2 * AS0 + channels * AS1, mask=lanes, other=0)
        a3 = tl.load(Alphas + v3 * AS0 + channels * AS1, mask=lanes, other=0)
        # Match the four-term interpolation, clipping and normalization used
        # by rasterize_priors_compact. Geometry and ownership stay FP32.
        raw = ((a0 * w0 + a1 * w1) + a2 * w2) + a3 * w3
        values = tl.maximum(raw, 0.0)
        denominator_raw = tl.sum(values, 0)
        denominator = tl.maximum(denominator_raw, 1.1920928955078125e-7)
        priors = values / denominator
        priors = tl.where(covered, priors, tl.where(channels == BACKGROUND, 1.0, 0.0))
        safe_priors = tl.maximum(priors, 1.1754943508222875e-38)
        likelihood = tl.load(Likelihood + channels * LS0 + row * LS1,
                             mask=lanes, other=-float("inf"))
        joint = tl.log(safe_priors) + likelihood
        maximum = tl.max(joint, 0)
        exponential = tl.exp(joint - maximum)
        exponential_sum = tl.sum(exponential, 0)
        tl.store(Costs + row, -(maximum + tl.log(exponential_sum)))
        dpriors = -(exponential / exponential_sum) / safe_priors
        dpriors = tl.where(lanes & (priors >= 1.1754943508222875e-38), dpriors, 0.0)
        normalization_gradient = tl.sum(dpriors * values, 0)
        dvalues = (dpriors - tl.where(
            denominator_raw >= 1.1920928955078125e-7,
            normalization_gradient / denominator, 0.0)) / denominator
        dvalues = tl.where(lanes & covered & (raw >= 0), dvalues, 0.0)
        dw0 = tl.sum(dvalues * a0, 0)
        q1 = tl.sum(dvalues * a1, 0) - dw0
        q2 = tl.sum(dvalues * a2, 0) - dw0
        q3 = tl.sum(dvalues * a3, 0) - dw0
        gx = (i00 * q1 + i10 * q2) + i20 * q3
        gy = (i01 * q1 + i11 * q2) + i21 * q3
        gz = (i02 * q1 + i12 * q2) + i22 * q3
        # Differentiating the barycentric solve gives dL/d(vertex_i)=-w_i*h.
        # No candidate graphs or inverse-matrix backward are retained.
        if DETERMINISTIC:
            for corner in tl.static_range(4):
                weight = w0 if corner == 0 else w1 if corner == 1 else w2 if corner == 2 else w3
                output = Gradient + (packed * 4 + corner) * 3
                tl.store(output, tl.where(covered, -weight * gx, 0.0), mask=present)
                tl.store(output + 1, tl.where(covered, -weight * gy, 0.0), mask=present)
                tl.store(output + 2, tl.where(covered, -weight * gz, 0.0), mask=present)
        else:
            tl.atomic_add(Gradient + v0 * 3, -w0 * gx, mask=covered)
            tl.atomic_add(Gradient + v0 * 3 + 1, -w0 * gy, mask=covered)
            tl.atomic_add(Gradient + v0 * 3 + 2, -w0 * gz, mask=covered)
            tl.atomic_add(Gradient + v1 * 3, -w1 * gx, mask=covered)
            tl.atomic_add(Gradient + v1 * 3 + 1, -w1 * gy, mask=covered)
            tl.atomic_add(Gradient + v1 * 3 + 2, -w1 * gz, mask=covered)
            tl.atomic_add(Gradient + v2 * 3, -w2 * gx, mask=covered)
            tl.atomic_add(Gradient + v2 * 3 + 1, -w2 * gy, mask=covered)
            tl.atomic_add(Gradient + v2 * 3 + 2, -w2 * gz, mask=covered)
            tl.atomic_add(Gradient + v3 * 3, -w3 * gx, mask=covered)
            tl.atomic_add(Gradient + v3 * 3 + 1, -w3 * gy, mask=covered)
            tl.atomic_add(Gradient + v3 * 3 + 2, -w3 * gz, mask=covered)


class _CompactDataCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, tetrahedra, alphas, points, selected, covered,
                reorder, origins, inverse, likelihood, background_channel,
                double_accumulation, deterministic_gradient):
        costs = torch.empty(reorder.shape, device=vertices.device, dtype=vertices.dtype)
        gradient = (torch.zeros((selected.numel() * 4, 3), device=vertices.device, dtype=vertices.dtype)
                    if deterministic_gradient else torch.zeros_like(vertices))
        if reorder.numel():
            _data_cost_kernel[(reorder.numel(),)](
                points, selected, covered, reorder, tetrahedra, alphas,
                origins, inverse, likelihood, costs, gradient,
                int(alphas.shape[1]), reorder.numel(), *alphas.stride(),
                *origins.stride(), *inverse.stride(), *likelihood.stride(),
                -1 if background_channel is None else int(background_channel),
                triton.next_power_of_2(int(alphas.shape[1])),
                bool(deterministic_gradient),
                num_warps=1, enable_fp_fusion=True)
        if deterministic_gradient:
            # The packed lookup retains the original batch/block/point order.
            # Missing-cell rows do not contribute a geometry derivative.
            vertex_ids = tetrahedra[selected].reshape(-1)
            gradient = ordered_vertex_sum(vertex_ids, gradient, vertices.shape[0])
        ctx.save_for_backward(gradient)
        return costs.sum(dtype=torch.float64 if double_accumulation else vertices.dtype)

    @staticmethod
    def backward(ctx, cost_gradient):
        gradient, = ctx.saved_tensors
        return gradient * cost_gradient, None, None, None, None, None, None, None, None, None, None, None, None


def supports_fused_data_cost(vertices, tetrahedra, alphas, likelihood):
    """Whether the fixed-alpha, fixed-likelihood FP32 CUDA path is available."""
    return (triton is not None and vertices.is_cuda and torch.version.hip is None
            and all(t.device == vertices.device for t in (tetrahedra, alphas, likelihood))
            and all(t.dtype == torch.float32 for t in (vertices, alphas, likelihood))
            and tetrahedra.dtype == torch.long and tetrahedra.is_contiguous()
            and vertices.is_contiguous() and not alphas.requires_grad
            and not likelihood.requires_grad and alphas.shape[1] > 0)


def fused_data_cost(vertices, tetrahedra, alphas, points, selected, covered,
                    reorder, origins, inverse, likelihood, background_channel,
                    double_accumulation=False, deterministic_gradient=False):
    """Fixed-ownership mixture likelihood with an analytic vertex gradient.

    The caller performs the same discrete FP32 candidate lookup as ordinary
    compact rasterization. All valid voxels are retained; only interpolation,
    mixture likelihood and their derivative are fused. The gradient is first
    order and the alphas and Gaussian likelihood must be fixed.
    """
    return _CompactDataCost.apply(vertices, tetrahedra, alphas, points, selected,
                                  covered, reorder, origins, inverse, likelihood,
                                  background_channel, bool(double_accumulation), bool(deterministic_gradient))


@torch.no_grad()
def lookup_candidates(points, ids, candidate_mask, all_v0, all_inv,
                      all_singular, point_rows, *, tolerance=2e-5,
                      previous_selected=None, hint_tolerance=2e-4,
                      return_hint_hits=False):
    """Return selected IDs and coverage, or ``None`` for the torch fallback.

    Inputs match ``BlockIndex.device_compact_batches`` and the geometry computed
    by PyTorch. Only real ``point_rows`` are searched. Geometry may be strided;
    coordinates and index tensors must be contiguous. All floating arithmetic
    uses FP32 with fused multiply-add, matching PyTorch's CUDA accumulation;
    interpolation remains outside
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
        return (selected, covered, None) if return_hint_hits else (selected, covered)
    width = int(ids.shape[1])
    warps = 1 if width <= 32 else 4 if width <= 512 else 8
    hint_hits = None
    if previous_selected is not None:
        if (previous_selected.shape != point_rows.shape or previous_selected.device != points.device
                or previous_selected.dtype != torch.long or not previous_selected.is_contiguous()):
            return None
        hint_hits = torch.empty(point_rows.shape, device=points.device, dtype=torch.bool)
        _lookup_hint_kernel[(point_rows.numel(),)](
            points, ids, candidate_mask, all_v0, all_inv, all_singular,
            point_rows, previous_selected, selected, covered, hint_hits,
            int(points.shape[1]), width, *all_v0.stride(), *all_inv.stride(),
            float(tolerance), max(float(tolerance), float(hint_tolerance)), triton.next_power_of_2(width),
            num_warps=warps, enable_fp_fusion=True)
    else:
        _lookup_kernel[(point_rows.numel(),)](
            points, ids, candidate_mask, all_v0, all_inv, all_singular,
            point_rows, selected, covered, int(points.shape[1]), width,
            *all_v0.stride(), *all_inv.stride(), float(tolerance), triton.next_power_of_2(width),
            num_warps=warps, enable_fp_fusion=True)
    return (selected, covered, hint_hits) if return_hint_hits else (selected, covered)


__all__ = ["lookup_candidates"]
