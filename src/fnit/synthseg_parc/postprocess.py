"""PyTorch counterpart of SynthSeg 2.0's label postprocessing."""

from __future__ import annotations

import torch


def largest_connected_component(mask: torch.Tensor) -> torch.Tensor:
    """Return the largest 6-connected component of a 3-D boolean tensor.

    Equal-size components are resolved by the first voxel in raster order,
    matching ``scipy.ndimage.label`` used by FreeSurfer.
    """
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError("mask must be a 3-D boolean tensor")
    bounds = None
    if mask.device.type == "cpu":
        # Match the official CPU routine directly. Constructing the full
        # voxel-edge graph and repeatedly scatter-reducing it costs far more
        # memory/work on CPU; scipy.label has the same six-neighbor topology
        # and assigns equal-size components in raster order.
        import numpy as np
        from scipy.ndimage import label

        components, count = label(mask.numpy())
        if count == 0:
            return mask.clone()
        sizes = np.bincount(components.ravel(), minlength=count + 1)
        selected = int(sizes[1:].argmax()) + 1
        return torch.from_numpy(components == selected)
    else:
        if not bool(mask.any()):
            return mask.clone()
        work = mask

    voxel = torch.arange(work.numel(), device=work.device).reshape(work.shape)
    starts, ends = [], []
    for axis in range(3):
        before, after = [slice(None)] * 3, [slice(None)] * 3
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        connected = work[tuple(before)] & work[tuple(after)]
        starts.append(voxel[tuple(before)][connected])
        ends.append(voxel[tuple(after)][connected])
    starts, ends = torch.cat(starts), torch.cat(ends)

    parent = torch.arange(work.numel(), device=work.device)
    while True:
        a, b = parent[starts], parent[ends]
        updated = parent.clone()
        updated.scatter_reduce_(0, torch.maximum(a, b), torch.minimum(a, b), reduce="amin")
        updated = updated[updated]
        if torch.equal(updated, parent):
            break
        parent = updated

    roots, counts = torch.unique(parent[work.flatten()], return_counts=True)
    component = ((parent == roots[counts.argmax()]).reshape(work.shape) & work)
    if bounds is None:
        return component
    result = torch.zeros_like(mask)
    result[bounds] = component
    return result


def postprocess_segmentation(
    posterior: torch.Tensor,
    labels: torch.Tensor,
    topology_classes: torch.Tensor,
    content_slices: tuple[slice, slice, slice],
    *,
    fast: bool = False,
    foreground_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply official component filters, unpad, and return hard labels/posteriors.

    Channel 0 is background. ``labels`` and ``topology_classes`` must follow
    posterior channel order. The non-fast path is the default official path.
    """
    if posterior.ndim != 4 or labels.numel() != posterior.shape[0] or topology_classes.numel() != labels.numel():
        raise ValueError("posterior, labels, and topology classes must share channel order")
    labels = labels.to(posterior.device)
    topology_classes = topology_classes.to(posterior.device)
    posterior = posterior.clone()
    if fast:
        posterior = posterior[(slice(None), *content_slices)]

    foreground = largest_connected_component(posterior[1:].sum(0) > foreground_threshold)
    posterior[1:] *= foreground
    if fast:
        posterior[1:] *= posterior[1:] > 0.2
    else:
        for group in torch.unique(topology_classes):
            if int(group) == 0:
                continue
            channels = torch.where(topology_classes == group)[0]
            mask = largest_connected_component((posterior[channels] > 0.25).any(0))
            for channel in channels:
                posterior[channel] *= mask
        posterior = posterior[(slice(None), *content_slices)]

    posterior /= posterior.sum(0, keepdim=True)
    return labels[posterior.argmax(0)], posterior
