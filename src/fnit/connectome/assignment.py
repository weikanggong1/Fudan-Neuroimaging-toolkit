"""PyTorch construction of the four UKB structural connectome matrices.

This implements the matrix stage of the UKB-connectomics MRtrix pipeline:
``tck2connectome -symmetric -assignment_radial_search 4``. Streamline
endpoints and the atlas affine must use the same world-coordinate space (mm).
"""

from __future__ import annotations

import math

import torch


def build_connectomes(
    endpoints: torch.Tensor,
    atlas: torch.Tensor,
    affine: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    fa: torch.Tensor | None = None,
    radius: float = 4.0,
    batch_size: int = 1024,
    node_count: int | None = None,
) -> dict[str, torch.Tensor]:
    """Assign endpoints to nearest labelled voxel centres and form matrices.

    ``endpoints`` has shape ``[N, 2, 3]`` in world mm. ``atlas`` contains
    non-negative integer labels; label 1 maps to matrix row/column 0, and
    absent labels below the maximum retain zero rows/columns. The optional
    length and FA vectors are per streamline, not per endpoint. Their edge
    means are SIFT2-weighted when ``weights`` is supplied, otherwise ordinary
    means. Unassigned streamlines are dropped; self-connections are retained.

    Computation runs on ``endpoints.device``. Float32 is used for coordinates
    and metrics; sums use float64 and the streamline count matrix uses int64.
    The result contains
    ``count`` and, when their inputs are provided, ``sift2_fbc``,
    ``mean_length``, and ``mean_fa``.
    """
    endpoints = torch.as_tensor(endpoints)
    if endpoints.ndim != 3 or endpoints.shape[1:] != (2, 3):
        raise ValueError("endpoints must have shape [N, 2, 3]")
    device = endpoints.device
    endpoints = endpoints.to(dtype=torch.float32)
    atlas = torch.as_tensor(atlas, device=device)
    affine = torch.as_tensor(affine, device=device, dtype=torch.float32)
    if atlas.ndim != 3 or atlas.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("atlas must be a 3D integer tensor")
    if affine.shape != (4, 4):
        raise ValueError("affine must have shape [4, 4]")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if torch.any(atlas < 0):
        raise ValueError("atlas labels must be non-negative")

    max_label = int(atlas.max().item())
    count_nodes = max_label if node_count is None else node_count
    if count_nodes < max_label:
        raise ValueError("node_count must cover the atlas maximum label")
    if count_nodes == 0:
        raise ValueError("atlas contains no labelled voxels")
    count_tracks = endpoints.shape[0]

    def metric_vector(value: torch.Tensor | None, name: str) -> torch.Tensor | None:
        if value is None:
            return None
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
        if value.shape != (count_tracks,):
            raise ValueError(f"{name} must have shape [N]")
        return value

    weights = metric_vector(weights, "weights")
    lengths = metric_vector(lengths, "lengths")
    fa = metric_vector(fa, "fa")

    linear = affine[:3, :3]
    inverse = torch.linalg.inv(linear)
    # If a point lies within radius of a voxel centre, its voxel displacement
    # from round(point in voxel coordinates) is at most radius*||inverse row||
    # plus half a voxel. This bound also covers rotated and sheared affines.
    extent = [math.ceil(radius * inverse[i].norm().item() + 0.5) for i in range(3)]
    axes = [torch.arange(-e, e + 1, device=device) for e in extent]
    offsets = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    # A rounded voxel centre differs from the endpoint by at most half a voxel
    # on each axis. Discard offsets whose centres cannot reach the search ball.
    half_voxel_bound = (0.5 * linear.abs().sum(dim=1)).norm()
    offset_world = (offsets[:, 0:1] * linear[:, 0]
                    + offsets[:, 1:2] * linear[:, 1]
                    + offsets[:, 2:3] * linear[:, 2])
    offsets = offsets[offset_world.norm(dim=1) <= radius + half_voxel_bound]
    # Bound the temporary [endpoints x candidate voxels x 3] arrays.
    batch_size = min(batch_size, max(1, 500_000 // offsets.shape[0]))
    shape = atlas.shape
    shape_tensor = torch.tensor(shape, device=device)
    flat_atlas = atlas.reshape(-1)
    count = torch.zeros(count_nodes * count_nodes, dtype=torch.int64, device=device)
    fbc = torch.zeros_like(count, dtype=torch.float64) if weights is not None else None
    mean_denominator = (
        torch.zeros_like(count, dtype=torch.float64)
        if lengths is not None or fa is not None else None
    )
    length_sum = torch.zeros_like(mean_denominator) if lengths is not None else None
    fa_sum = torch.zeros_like(mean_denominator) if fa is not None else None

    def nearest_labels(points: torch.Tensor) -> torch.Tensor:
        shifted = points - affine[:3, 3]
        ijk = (shifted[:, 0:1] * inverse[:, 0]
               + shifted[:, 1:2] * inverse[:, 1]
               + shifted[:, 2:3] * inverse[:, 2])
        voxels = torch.round(ijk).to(torch.int64)[:, None, :] + offsets[None, :, :]
        inside = ((voxels >= 0) & (voxels < shape_tensor)).all(dim=-1)
        index = ((voxels[..., 0] * shape[1] + voxels[..., 1]) * shape[2]
                 + voxels[..., 2]).clamp(0, flat_atlas.numel() - 1)
        labels = torch.where(inside, flat_atlas[index], 0)
        delta = voxels.to(torch.float32) - ijk[:, None, :]
        world_delta = (delta[..., 0:1] * linear[:, 0]
                       + delta[..., 1:2] * linear[:, 1]
                       + delta[..., 2:3] * linear[:, 2])
        distance2 = world_delta.square().sum(dim=-1)
        distance2 = distance2.masked_fill((labels == 0) | (distance2 >= radius * radius), math.inf)
        nearest = distance2.argmin(dim=1)
        picked = labels.gather(1, nearest[:, None]).squeeze(1)
        return torch.where(distance2.gather(1, nearest[:, None]).squeeze(1).isfinite(), picked, 0)

    for start in range(0, count_tracks, batch_size):
        stop = min(start + batch_size, count_tracks)
        nodes = nearest_labels(endpoints[start:stop].reshape(-1, 3)).reshape(-1, 2)
        valid = (nodes[:, 0] > 0) & (nodes[:, 1] > 0)
        left = torch.minimum(nodes[valid, 0], nodes[valid, 1]) - 1
        right = torch.maximum(nodes[valid, 0], nodes[valid, 1]) - 1
        edge = (left * count_nodes + right).to(torch.int64)
        count.scatter_add_(0, edge, torch.ones_like(edge, dtype=torch.int64))
        edge_weights = (weights[start:stop][valid].to(torch.float64) if weights is not None
                        else torch.ones_like(edge, dtype=torch.float64))
        if fbc is not None:
            fbc.scatter_add_(0, edge, edge_weights)
        if mean_denominator is not None:
            mean_denominator.scatter_add_(0, edge, edge_weights)
        if length_sum is not None:
            length_sum.scatter_add_(0, edge, edge_weights * lengths[start:stop][valid].to(torch.float64))
        if fa_sum is not None:
            fa_sum.scatter_add_(0, edge, edge_weights * fa[start:stop][valid].to(torch.float64))

    def symmetric(flat: torch.Tensor) -> torch.Tensor:
        upper = flat.reshape(count_nodes, count_nodes)
        return upper + upper.T - torch.diag(upper.diagonal())

    result = {"count": symmetric(count)}
    if fbc is not None:
        result["sift2_fbc"] = symmetric(fbc.to(torch.float32))
    if mean_denominator is not None:
        denominator = mean_denominator.clamp_min(torch.finfo(torch.float64).tiny)
        if length_sum is not None:
            result["mean_length"] = symmetric((length_sum / denominator).to(torch.float32))
        if fa_sum is not None:
            result["mean_fa"] = symmetric((fa_sum / denominator).to(torch.float32))
    return result
