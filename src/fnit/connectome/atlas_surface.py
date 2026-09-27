"""UKB-connectomics cortical ribbon projection of FreeSurfer annotations.

This is the nearest-(pial or white)-vertex rule in the original
``scripts/python/map_surface_label_to_volume.py``. All nearest-neighbor work
runs in PyTorch on the input tensor's device. Three-dimensional bins restrict
candidate comparisons; voxels whose nearest candidate lies beyond one bin
width fall back to a complete search, preserving exact nearest-neighbor
semantics rather than silently using an approximate neighborhood.
"""

import torch


def _nearest_surface_vertices(
    query: torch.Tensor,
    vertices: torch.Tensor,
    *,
    bin_width_mm: float,
    query_batch: int,
) -> torch.Tensor:
    """Return [N] indices of Euclidean-nearest vertices for [N,3] queries."""
    if not len(query):
        return torch.empty(0, dtype=torch.long, device=query.device)
    width = float(bin_width_mm)
    origin = torch.floor(torch.minimum(query.amin(0), vertices.amin(0)) / width) * width - width
    sizes = (torch.floor((torch.maximum(query.amax(0), vertices.amax(0)) - origin) / width)
             .long() + 2)
    vertex_bins = torch.floor((vertices - origin) / width).long()
    query_bins = torch.floor((query - origin) / width).long()
    size_y, size_z = (int(sizes[1]), int(sizes[2]))
    n_bins = int(sizes.prod())
    vertex_keys = (vertex_bins[:, 0] * size_y + vertex_bins[:, 1]) * size_z + vertex_bins[:, 2]
    sorted_keys, order = vertex_keys.sort()
    counts = torch.bincount(sorted_keys, minlength=n_bins)
    starts = counts.cumsum(0) - counts
    slots = torch.arange(len(vertices), device=vertices.device) - starts[sorted_keys]
    table = torch.full((n_bins, int(counts.max())), -1, dtype=torch.long, device=vertices.device)
    table[sorted_keys, slots] = order
    offsets = torch.stack(torch.meshgrid(
        *(torch.arange(-1, 2, device=query.device) for _ in range(3)), indexing="ij"
    ), -1).reshape(27, 3)
    result = torch.empty(len(query), dtype=torch.long, device=query.device)
    for start in range(0, len(query), query_batch):
        stop = min(start + query_batch, len(query))
        neighbors = query_bins[start:stop, None, :] + offsets[None, :, :]
        valid = ((neighbors >= 0) & (neighbors < sizes)).all(-1)
        neighbors = neighbors.clamp_min(0)
        for axis in range(3):
            neighbors[:, :, axis].clamp_max_(int(sizes[axis]) - 1)
        keys = (neighbors[:, :, 0] * size_y + neighbors[:, :, 1]) * size_z + neighbors[:, :, 2]
        candidates = table[keys].masked_fill(~valid[..., None], -1)
        distances = (vertices[candidates.clamp_min(0)] - query[start:stop, None, None, :]).square().sum(-1)
        distances.masked_fill_(candidates < 0, torch.inf)
        distances = distances.flatten(1)
        best_distance, best_position = distances.min(1)
        nearest = candidates.flatten(1).gather(1, best_position[:, None]).squeeze(1)
        # Every unsearched bin is at least one full bin width away. If that
        # geometric bound does not certify the nearest vertex, search all.
        unresolved = (best_distance >= width * width).nonzero().flatten()
        for subset in unresolved.split(32):
            full_distance = (query[start + subset, None, :] - vertices[None, :, :]).square().sum(-1)
            nearest[subset] = full_distance.argmin(1)
        result[start:stop] = nearest
    return result


def surface_annotation_to_volume(
    ribbon: torch.Tensor,
    vox2ras_tkr: torch.Tensor,
    lh_pial: torch.Tensor,
    lh_white: torch.Tensor,
    lh_labels: torch.Tensor,
    rh_pial: torch.Tensor,
    rh_white: torch.Tensor,
    rh_labels: torch.Tensor,
    *,
    bin_width_mm: float = 3.0,
    query_batch: int = 8192,
) -> torch.Tensor:
    """Project FreeSurfer cortical annotations to the ribbon volume.

    ``ribbon`` is a 3D integer label map: 3 is left cortex, 42 right cortex.
    ``vox2ras_tkr`` is its 4x4 FreeSurfer surface-RAS voxel transform, *not*
    the NIfTI scanner-RAS affine. Each ``{lh,rh}_{pial,white}`` is [V,3] in
    surface RAS, and ``{lh,rh}_labels`` is [V] from nibabel ``read_annot``.
    Pial and white surfaces must have matching vertex order. The output is a
    same-shape int32 tensor, zero outside cortical ribbon, containing exactly
    those annotation indices (including -1 where present). Use the ribbon's
    scanner-RAS affine when saving the output with nibabel.

    Original equivalent: ``python scripts/python/map_surface_label_to_volume.py
    <main_dir> <subjects_dir> <subject_id> <instance> <atlas_name>``.
    """
    if ribbon.ndim != 3 or tuple(vox2ras_tkr.shape) != (4, 4):
        raise ValueError("ribbon must be 3D and vox2ras_tkr must be 4x4")
    if bin_width_mm <= 0 or query_batch < 1:
        raise ValueError("bin_width_mm and query_batch must be positive")
    device = ribbon.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = torch.zeros_like(ribbon, dtype=torch.int32)
    affine = torch.as_tensor(vox2ras_tkr, dtype=torch.float64, device=device)
    if not bool(torch.isfinite(affine).all()):
        raise ValueError("vox2ras_tkr must be finite")
    for ribbon_label, pial, white, labels in (
        (3, lh_pial, lh_white, lh_labels),
        (42, rh_pial, rh_white, rh_labels),
    ):
        if (pial.ndim != 2 or pial.shape[1] != 3 or pial.shape != white.shape
                or labels.ndim != 1 or labels.numel() != len(pial)
                or not len(pial)):
            raise ValueError("each hemisphere needs matching [V,3] pial/white and [V] labels")
        vertices = torch.cat((pial, white)).to(device=device, dtype=torch.float64)
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("surface coordinates must be finite")
        indices = torch.nonzero(ribbon == ribbon_label)
        if not len(indices):
            continue
        query = indices.to(torch.float64) @ affine[:3, :3].T + affine[:3, 3]
        nearest = _nearest_surface_vertices(
            query, vertices, bin_width_mm=bin_width_mm, query_batch=query_batch
        ) % len(pial)
        output[indices[:, 0], indices[:, 1], indices[:, 2]] = labels.to(
            device=device, dtype=torch.int32
        )[nearest]
    return output
