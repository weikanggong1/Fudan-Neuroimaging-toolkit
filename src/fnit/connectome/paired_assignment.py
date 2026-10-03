"""Rectangular structural connectomes between two declared ROI templates."""

from __future__ import annotations

import torch

from .assignment import RadialEndpointAssigner, build_connectomes


def build_pair_connectomes(
    endpoints: torch.Tensor,
    first_atlas: torch.Tensor,
    first_affine: torch.Tensor,
    second_atlas: torch.Tensor,
    second_affine: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    fa: torch.Tensor | None = None,
    first_node_count: int | None = None,
    second_node_count: int | None = None,
    radius: float = 4.0,
    batch_size: int = 1024,
    same_template: bool = False,
) -> dict[str, torch.Tensor]:
    """Return [K_first,K_second] matrices from one whole-brain tractogram.

    Each endpoint is assigned independently to each atlas with the existing
    strict radial voxel-centre rule. Streamlines have no anatomical direction:
    both (first(start), second(end)) and (first(end), second(start)) contribute.
    If these orientations produce the same cell, that streamline contributes
    once to that cell. Distinct cells each receive one contribution, including
    overlapping templates. Cross-template matrices are not symmetrised and
    their diagonal has no special meaning. Invalid endpoints contribute zero.

    ``same_template=True`` requires identical labels, affines and node counts,
    then calls the existing square builder exactly, retaining its symmetric
    matrices and self-connections. All coordinates are scanner RAS millimetres.
    ``weights`` is per-streamline SIFT2 weight; length/FA means are weighted
    when supplied. Metrics use the same float32 input and float64 reductions
    as the existing square builder. Declared empty nodes retain zero rows.
    """
    endpoints = torch.as_tensor(endpoints)
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 3):
        raise ValueError("endpoints must have shape [N,2,3]")
    device = endpoints.device
    endpoints = endpoints.to(torch.float32)
    first_atlas = torch.as_tensor(first_atlas, device=device)
    second_atlas = torch.as_tensor(second_atlas, device=device)
    first_affine = torch.as_tensor(first_affine, device=device, dtype=torch.float32)
    second_affine = torch.as_tensor(second_affine, device=device, dtype=torch.float32)
    # Constructing these validates atlas dtype, dimensionality and radius.
    first = RadialEndpointAssigner(first_atlas, first_affine, radius=radius,
                                  batch_size=batch_size)
    second = RadialEndpointAssigner(second_atlas, second_affine, radius=radius,
                                   batch_size=batch_size)
    first_max, second_max = int(first_atlas.max()), int(second_atlas.max())
    k_first = first_max if first_node_count is None else first_node_count
    k_second = second_max if second_node_count is None else second_node_count
    if (isinstance(k_first, bool) or isinstance(k_second, bool)
            or not isinstance(k_first, int) or not isinstance(k_second, int)
            or k_first < max(1, first_max) or k_second < max(1, second_max)):
        raise ValueError("node counts must be positive integers covering the atlas labels")
    if same_template:
        if (k_first != k_second or not torch.equal(first_atlas, second_atlas)
                or not torch.equal(first_affine, second_affine)):
            raise ValueError("same_template requires identical labels, affines and node counts")
        return build_connectomes(
            endpoints, first_atlas, first_affine, weights=weights, lengths=lengths,
            fa=fa, radius=radius, batch_size=batch_size, node_count=k_first,
        )

    n_tracks = len(endpoints)

    def metric(value: torch.Tensor | None, name: str) -> torch.Tensor | None:
        if value is None:
            return None
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if value.shape != (n_tracks,):
            raise ValueError(f"{name} must have shape [N]")
        if name == "weights" and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite values")
        return value

    weights, lengths, fa = metric(weights, "weights"), metric(lengths, "lengths"), metric(fa, "fa")
    if weights is not None and bool((weights < 0).any()):
        raise ValueError("weights must be non-negative")
    if not bool(torch.isfinite(endpoints).all()):
        raise ValueError("endpoints must be finite")
    n_cells = k_first * k_second
    count = torch.zeros(n_cells, dtype=torch.int64, device=device)
    fbc = torch.zeros(n_cells, dtype=torch.float64, device=device) if weights is not None else None
    denominator = (torch.zeros(n_cells, dtype=torch.float64, device=device)
                   if lengths is not None or fa is not None else None)
    length_sum = torch.zeros_like(denominator) if lengths is not None else None
    fa_sum = torch.zeros_like(denominator) if fa is not None else None
    # Each batch assigns 2 endpoints. The existing budget is per endpoint call.
    step = max(1, min(first.batch_size, second.batch_size) // 2)
    for start in range(0, n_tracks, step):
        stop = min(start + step, n_tracks)
        positions = endpoints[start:stop].reshape(-1, 3)
        a = first(positions).reshape(-1, 2)
        b = second(positions).reshape(-1, 2)
        forward = (a[:, 0] > 0) & (b[:, 1] > 0)
        reverse = (a[:, 1] > 0) & (b[:, 0] > 0)
        forward_cells = ((a[:, 0] - 1) * k_second + b[:, 1] - 1).long()
        reverse_cells = ((a[:, 1] - 1) * k_second + b[:, 0] - 1).long()
        reverse &= ~(forward & (forward_cells == reverse_cells))
        local = torch.arange(stop - start, device=device)
        cells = torch.cat((forward_cells[forward], reverse_cells[reverse]))
        tracks = torch.cat((local[forward], local[reverse])) + start
        count.scatter_add_(0, cells, torch.ones_like(cells, dtype=torch.int64))
        edge_weights = (weights[tracks].to(torch.float64) if weights is not None
                        else torch.ones_like(cells, dtype=torch.float64))
        if fbc is not None:
            fbc.scatter_add_(0, cells, edge_weights)
        if denominator is not None:
            denominator.scatter_add_(0, cells, edge_weights)
        if length_sum is not None:
            length_sum.scatter_add_(0, cells, edge_weights * lengths[tracks].to(torch.float64))
        if fa_sum is not None:
            fa_sum.scatter_add_(0, cells, edge_weights * fa[tracks].to(torch.float64))

    shape = (k_first, k_second)
    result = {"count": count.reshape(shape)}
    if fbc is not None:
        result["sift2_fbc"] = fbc.to(torch.float32).reshape(shape)
    if denominator is not None:
        denominator = denominator.clamp_min(torch.finfo(torch.float64).tiny)
        if length_sum is not None:
            result["mean_length"] = (length_sum / denominator).to(torch.float32).reshape(shape)
        if fa_sum is not None:
            result["mean_fa"] = (fa_sum / denominator).to(torch.float32).reshape(shape)
    return result
