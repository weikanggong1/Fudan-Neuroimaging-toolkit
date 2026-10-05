"""Private CPU blur trial; never imported by FNIT production.

The full 33-channel geometry determines depth slabs before channels become
independent batches. No oneDNN, thread, CUDA, precision or dtype setting is
written. Small contracts and real posterior bit gates precede integration.
"""

import torch
from torch.nn import functional as F


def eligible(posterior):
    """Limit this trial to the ordinary one-image FP32 no-grad CPU path."""
    from fnit.synthseg_parc.cpu_conv import cpu_autocast_enabled
    return (posterior.device.type == "cpu" and posterior.dtype == torch.float32
            and posterior.ndim == 5 and posterior.shape[:2] == (1, 33)
            and not torch.is_grad_enabled() and not posterior.requires_grad
            and not cpu_autocast_enabled() and not torch.backends.mkldnn.enabled)


def channel_batch_slabs(posterior, kernel, *, channel_block=11,
                        maximum_slab_bytes=256 * 1024 * 1024):
    """Apply the same 27 kernel values to independent channels as batches.

    Input/output: (1,33,D,H,W) float32 CPU tensors. Kernel: (1,1,3,3,3),
    constructed by the original blur expression. Each slab retains all halo
    values and true-boundary depth zeros. Output is a fresh contiguous buffer;
    views only reindex independent channels. This changes the BLAS call branch
    and has not yet passed the actual 2.5.1 posterior gate.
    """
    if not eligible(posterior):
        raise ValueError("trial requires eligible CPU FP32 no-grad posterior")
    if (kernel.device != posterior.device or kernel.dtype != posterior.dtype
            or tuple(kernel.shape) != (1, 1, 3, 3, 3)):
        raise ValueError("trial requires matching 3x3x3 single-channel kernel")
    if channel_block != 11:
        raise ValueError("only the declared channel_block=11 is approved")
    batch, channels, depth, height, width = posterior.shape
    # Use the ORIGINAL 33 logical channels, not the smaller batch block.
    bytes_per_plane = batch * channels * height * width * posterior.element_size()
    slab_depth = max(1, min(32, maximum_slab_bytes // bytes_per_plane - 2))
    output = posterior.new_empty(posterior.shape)
    for start in range(0, depth, slab_depth):
        stop = min(start + slab_depth, depth)
        lower, upper = max(0, start - 1), min(depth, stop + 1)
        before, after = max(0, 1 - start), max(0, stop + 1 - depth)
        for first in range(0, channels, channel_block):
            last = min(first + channel_block, channels)
            chunk = posterior[:, first:last, lower:upper]
            if before or after:
                chunk = F.pad(chunk, (0, 0, 0, 0, before, after))
            independent = chunk.reshape(batch * (last - first), 1,
                                        chunk.shape[2], height, width)
            convolved = F.conv3d(independent, kernel, padding=(0, 1, 1), groups=1)
            output[:, first:last, start:stop].copy_(
                convolved.view(batch, last - first, stop - start, height, width))
    return output


def blur(posterior, original_blur):
    """Use the trial only in the declared path, otherwise call the original."""
    if not eligible(posterior):
        return original_blur(posterior)
    # Identical to the original kernel: axis, grid, exp, sum then division.
    axis = torch.arange(-1, 2, device=posterior.device, dtype=posterior.dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
    kernel = torch.exp(-grid.square().sum(0) / (2 * 0.5 ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3, 3)
    return channel_batch_slabs(posterior, kernel)
