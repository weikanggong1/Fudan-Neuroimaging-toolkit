"""PyTorch active-set solvers for the AMICO 2.0.3 NODDI fit."""

from __future__ import annotations

import numpy as np
import torch

from .kernels import direction_indices


def _masked_cg(
    gram,
    rhs,
    mask,
    *,
    ridge=0.0,
    tolerance=1e-13,
    maximum_iterations=220,
):
    x = torch.zeros_like(rhs)
    r = rhs * mask
    p = r.clone()
    squared = (r * r).sum(-1)
    scale = squared.sqrt().clamp_min(1.0)
    active = squared.sqrt() > tolerance * scale
    for _ in range(maximum_iterations):
        if not bool(active.any()):
            break
        hp = (torch.matmul(p, gram) + float(ridge) * p) * mask
        denominator = (p * hp).sum(-1)
        alpha = torch.where(
            active & (denominator.abs() > 1e-30),
            squared / denominator,
            torch.zeros_like(squared),
        )
        x = x + alpha[..., None] * p
        r = r - alpha[..., None] * hp
        updated = (r * r).sum(-1)
        active = updated.sqrt() > tolerance * scale
        beta = torch.where(
            squared > 1e-30, updated / squared, torch.zeros_like(squared)
        )
        p = (r + beta[..., None] * p) * mask
        squared = updated
    return x


def _masked_solve(gram, rhs, mask, *, ridge=0.0, tolerance=1e-13):
    """Solve the current passive-set systems as compact Cholesky batches."""
    columns = rhs.shape[-1]
    flat_mask = mask.reshape(-1, columns).bool()
    flat_rhs = rhs.reshape(-1, columns)
    counts = flat_mask.sum(-1)
    width = int(counts.max())
    if width == 0:
        return torch.zeros_like(rhs)
    order = torch.topk(flat_mask.to(torch.uint8), width, dim=-1, sorted=False).indices
    occupied = flat_mask.gather(-1, order)
    if gram.ndim == 2:
        block = gram[order[:, :, None], order[:, None, :]]
    else:
        slots = rhs.shape[-2]
        groups = torch.arange(gram.shape[0], device=rhs.device).repeat_interleave(slots)
        block = gram[groups[:, None, None], order[:, :, None], order[:, None, :]]
    active_block = occupied[:, :, None] & occupied[:, None, :]
    block = block * active_block
    diagonal = torch.diagonal(block, dim1=-2, dim2=-1)
    diagonal.add_(torch.where(occupied, float(ridge), 1.0))
    compact_rhs = flat_rhs.gather(-1, order) * occupied
    factor, info = torch.linalg.cholesky_ex(block, check_errors=False)
    failed = info != 0
    if bool(failed.any()):
        identity = torch.eye(width, dtype=block.dtype, device=block.device)
        block[failed] = identity
        compact_rhs[failed] = 0
        factor = torch.linalg.cholesky(block)
    compact = torch.cholesky_solve(compact_rhs[..., None], factor).squeeze(-1)
    result = torch.zeros_like(flat_rhs)
    result.scatter_(-1, order, compact * occupied)
    if bool(failed.any()):
        if gram.ndim == 2:
            result[failed] = _masked_cg(
                gram,
                flat_rhs[failed],
                flat_mask[failed],
                ridge=ridge,
                tolerance=tolerance,
            )
        else:
            failed_gram = gram[groups[failed]]
            result[failed] = _masked_cg(
                failed_gram,
                flat_rhs[failed, None, :],
                flat_mask[failed, None, :],
                ridge=ridge,
                tolerance=tolerance,
            )[:, 0]
    return result.reshape_as(rhs)


def _single_active_set(
    design,
    signal,
    *,
    l1=0.0,
    l2=0.0,
    allowed=None,
    kkt_tolerance=1e-11,
):
    columns = design.shape[1]
    gram = design.T @ design
    linear = design.T @ signal - float(l1)
    x = torch.zeros(columns, dtype=design.dtype, device=design.device)
    passive = torch.zeros(columns, dtype=torch.bool, device=design.device)
    if allowed is None:
        allowed = torch.ones_like(passive)
    for _ in range(3 * columns):
        violation = linear - (gram @ x + float(l2) * x)
        candidates = torch.where(
            allowed & ~passive, violation, torch.full_like(violation, -torch.inf)
        )
        best, selected = candidates.max(0)
        if float(best) <= float(kkt_tolerance):
            return x, passive
        passive[selected] = True
        for _ in range(columns):
            indices = torch.nonzero(passive, as_tuple=False).flatten()
            proposed = torch.zeros_like(x)
            if float(l2) == 0.0:
                proposed[indices] = torch.linalg.lstsq(
                    design[:, indices], signal
                ).solution
            else:
                block = gram[indices][:, indices]
                block = block + float(l2) * torch.eye(
                    len(indices), dtype=design.dtype, device=design.device
                )
                proposed[indices] = torch.linalg.solve(block, linear[indices])
            negative = (proposed <= 0) & passive
            if not bool(negative.any()):
                x = proposed
                break
            step = torch.min(x[negative] / (x[negative] - proposed[negative]))
            x = x + step.clamp(0, 1) * (proposed - x)
            dropped = (x <= 1e-12) & passive
            x[dropped] = 0
            passive[dropped] = False
        else:
            raise RuntimeError("single-voxel active-set boundary loop did not converge")
    raise RuntimeError("single-voxel active-set solver did not converge")


def nonnegative_quadratic(
    design,
    signal,
    *,
    l1=0.0,
    l2=0.0,
    allowed=None,
    kkt_tolerance=1e-11,
    cg_tolerance=1e-13,
    maximum_active_steps=40,
):
    """Solve batched non-negative quadratic problems with an active set.

    ``design`` may be a shared ``(M, C)`` matrix or one matrix per direction,
    ``(G, M, C)``. In the latter case ``signal`` is ``(G, V, M)`` and all
    directions in the current memory-bounded LUT batch share one launch sequence.
    """
    gram = design.transpose(-2, -1) @ design
    linear = torch.matmul(signal, design) - float(l1)
    columns = linear.shape[-1]
    x = torch.zeros_like(linear)
    passive = torch.zeros_like(linear, dtype=torch.bool)
    if allowed is None:
        allowed = torch.ones_like(passive)
    for outer in range(maximum_active_steps):
        violation = linear - (torch.matmul(x, gram) + float(l2) * x)
        candidates = torch.where(
            allowed & ~passive,
            violation,
            torch.full_like(violation, -torch.inf),
        )
        best, selected = candidates.max(-1)
        rows = best > float(kkt_tolerance)
        if not bool(rows.any()):
            return x, passive, outer
        flat_passive = passive.reshape(-1, columns)
        flat_rows = rows.reshape(-1)
        flat_selected = selected.reshape(-1)
        flat_passive[flat_rows, flat_selected[flat_rows]] = True
        for _ in range(columns):
            proposed = _masked_solve(
                gram,
                linear,
                passive,
                ridge=l2,
                tolerance=cg_tolerance,
            )
            negative = (proposed <= 0) & passive
            needs_boundary = negative.any(-1)
            if not bool(needs_boundary.any()):
                x = proposed
                break
            ratios = torch.where(
                negative,
                x / (x - proposed).clamp_min(1e-300),
                torch.full_like(x, torch.inf),
            )
            step = ratios.min(-1).values.clamp(0, 1)
            x = torch.where(
                needs_boundary[..., None],
                x + step[..., None] * (proposed - x),
                proposed,
            )
            dropped = (x <= 1e-12) & passive & needs_boundary[..., None]
            x[dropped] = 0
            passive[dropped] = False
        else:
            raise RuntimeError("active-set boundary loop did not converge")
    violation = linear - (torch.matmul(x, gram) + float(l2) * x)
    candidates = torch.where(
        allowed & ~passive,
        violation,
        torch.full_like(violation, -torch.inf),
    )
    unresolved = candidates.max(-1).values > float(kkt_tolerance)
    flat_x = x.reshape(-1, columns)
    flat_passive = passive.reshape(-1, columns)
    flat_allowed = allowed.reshape(-1, columns)
    flat_signal = signal.reshape(-1, signal.shape[-1])
    slots = signal.shape[-2] if design.ndim == 3 else None
    for row in torch.nonzero(unresolved.reshape(-1), as_tuple=False).flatten().tolist():
        row_design = design[row // slots] if design.ndim == 3 else design
        flat_x[row], flat_passive[row] = _single_active_set(
            row_design,
            flat_signal[row],
            l1=l1,
            l2=l2,
            allowed=flat_allowed[row],
            kkt_tolerance=kkt_tolerance,
        )
    return x, passive, maximum_active_steps


def _fit_lut_batch(
    signal,
    lut_values,
    grouped_rows,
    kernels,
    *,
    device,
    lambda1,
    lambda2,
    kkt_tolerance,
    cg_tolerance,
    maximum_active_steps,
):
    wm = kernels["wm"]
    iso = kernels["iso"]
    norms = kernels["norms"]
    b0 = kernels["b0"]
    slots = max(map(len, grouped_rows))
    valid = np.zeros((len(lut_values), slots), dtype=bool)
    grouped_signal = np.zeros(
        (len(lut_values), slots, signal.shape[1]), dtype=np.float64
    )
    for group, rows in enumerate(grouped_rows):
        valid[group, : len(rows)] = True
        grouped_signal[group, : len(rows)] = signal[rows]
    design_np = np.empty(
        (len(lut_values), signal.shape[1], wm.shape[0] + 1), dtype=np.float64
    )
    design_np[:, :, :-1] = wm[:, lut_values, :].transpose(1, 2, 0)
    design_np[:, :, -1] = iso
    design = torch.as_tensor(design_np, dtype=torch.float64, device=device)
    y = torch.as_tensor(grouped_signal, dtype=torch.float64, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    allowed_all = valid_t[..., None].expand(*valid_t.shape, design.shape[-1])
    first, _, _ = nonnegative_quadratic(
        design,
        y,
        allowed=allowed_all,
        kkt_tolerance=kkt_tolerance,
        cg_tolerance=cg_tolerance,
        maximum_active_steps=maximum_active_steps,
    )
    tissue_design = torch.as_tensor(
        design_np[:, ~b0, :-1] * norms[None],
        dtype=torch.float64,
        device=device,
    )
    b0_t = torch.as_tensor(b0, dtype=torch.bool, device=device)
    iso_dwi = torch.as_tensor(iso[~b0], dtype=torch.float64, device=device)
    tissue_signal = torch.relu(y[..., ~b0_t] - first[..., -1, None] * iso_dwi)
    tissue_allowed = valid_t[..., None].expand(*valid_t.shape, tissue_design.shape[-1])
    _, tissue_support, _ = nonnegative_quadratic(
        tissue_design,
        tissue_signal,
        l1=lambda1,
        l2=lambda2,
        allowed=tissue_allowed,
        kkt_tolerance=kkt_tolerance,
        cg_tolerance=cg_tolerance,
        maximum_active_steps=maximum_active_steps,
    )
    allowed = torch.cat((tissue_support, valid_t[..., None]), -1)
    coefficients, _, _ = nonnegative_quadratic(
        design,
        y,
        allowed=allowed,
        kkt_tolerance=kkt_tolerance,
        cg_tolerance=cg_tolerance,
        maximum_active_steps=maximum_active_steps,
    )
    total = coefficients.sum(-1) + 1e-16
    tissue_fraction = coefficients[..., :-1].sum(-1) / total + 1e-16
    vf = torch.as_tensor(kernels["icvf"], dtype=torch.float64, device=device)
    ka = torch.as_tensor(kernels["kappa"], dtype=torch.float64, device=device)
    f1 = (coefficients[..., :-1] * vf).sum(-1) / total / tissue_fraction
    f2 = (coefficients[..., :-1] * (1 - vf)).sum(-1) / total / tissue_fraction
    mean_kappa = (coefficients[..., :-1] * ka).sum(-1) / total / tissue_fraction
    result = torch.stack(
        (
            f1 / (f1 + f2 + 1e-16),
            2 / np.pi * torch.atan2(torch.ones_like(mean_kappa), mean_kappa),
            coefficients[..., -1] / total,
        ),
        -1,
    )
    predicted = torch.matmul(coefficients, design.transpose(-2, -1))
    rmse = torch.sqrt(((y - predicted) ** 2).mean(-1))
    support = tissue_support.sum(-1) + 1
    return result.cpu().numpy(), rmse.cpu().numpy(), support.cpu().numpy()


def fit_noddi(
    signal,
    directions,
    kernels,
    *,
    device,
    lambda1=0.5,
    lambda2=1e-3,
    kkt_tolerance=1e-11,
    cg_tolerance=1e-13,
    maximum_active_steps=40,
    lut_batch_size=400,
):
    """Run AMICO's three fitting stages in memory-bounded LUT batches."""
    signal = np.asarray(signal, dtype=np.float64)
    lut_indices = direction_indices(directions)
    lut_values = np.unique(lut_indices)
    grouped_rows = [np.flatnonzero(lut_indices == index) for index in lut_values]
    if int(lut_batch_size) < 1:
        raise ValueError("lut_batch_size must be positive")
    buckets = (
        np.arange(start, min(start + int(lut_batch_size), len(grouped_rows)))
        for start in range(0, len(grouped_rows), int(lut_batch_size))
    )
    estimates = np.zeros((len(signal), 3), dtype=np.float64)
    rmse = np.zeros(len(signal), dtype=np.float64)
    support_sizes = np.zeros(len(signal), dtype=np.int16)
    for bucket in buckets:
        rows = [grouped_rows[index] for index in bucket]
        result, grouped_rmse, grouped_support = _fit_lut_batch(
            signal,
            lut_values[bucket],
            rows,
            kernels,
            device=device,
            lambda1=lambda1,
            lambda2=lambda2,
            kkt_tolerance=kkt_tolerance,
            cg_tolerance=cg_tolerance,
            maximum_active_steps=maximum_active_steps,
        )
        for group, voxel_rows in enumerate(rows):
            size = len(voxel_rows)
            estimates[voxel_rows] = result[group, :size]
            rmse[voxel_rows] = grouped_rmse[group, :size]
            support_sizes[voxel_rows] = grouped_support[group, :size]
    return estimates, rmse, support_sizes, lut_indices


__all__ = ["fit_noddi", "nonnegative_quadratic"]
