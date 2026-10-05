"""CPU inference copies for SynthSeg's unchanged nearest/skip concatenation."""

import torch
from torch import nn

from .cpu_conv import cpu_autocast_enabled


def cpu_join_allowed(network, skip, value):
    """Keep CUDA, observed forwards, training and caller autocast on the old path."""
    if (value.device.type != "cpu" or skip.device.type != "cpu"
            or value.dtype != torch.float32 or skip.dtype != value.dtype
            or value.ndim != 5 or skip.ndim != 5
            or value.shape[0] != skip.shape[0]
            or any(size <= 0 for size in value.shape)
            or any(size <= 0 for size in skip.shape)
            or tuple(skip.shape[2:]) != tuple(2 * size for size in value.shape[2:])
            or not value.is_contiguous() or not skip.is_contiguous()
            or torch.is_grad_enabled() or cpu_autocast_enabled()):
        return False
    hook_names = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
    if any(module.training or any(getattr(module, name, {}) for name in hook_names)
           for module in network.modules()):
        return False
    global_names = ("_global_forward_pre_hooks", "_global_forward_hooks",
                    "_global_backward_pre_hooks", "_global_backward_hooks")
    return not any(getattr(nn.modules.module, name, {}) for name in global_names)


def join_nearest_cpu(skip, value):
    """Copy skip-first NCDHW values into one independent contiguous allocation.

    The caller qualifies CPU FP32 inference. A factor-two nearest neighbour
    duplicates each source voxel without floating point arithmetic. Splitting
    each destination spatial axis into source-index and duplicate-index axes
    allows one broadcast copy rather than a complete upsample allocation.
    """
    batch, value_channels, depth, height, width = value.shape
    skip_channels = skip.shape[1]
    joined = value.new_empty((batch, skip_channels + value_channels,
                              2 * depth, 2 * height, 2 * width))
    joined[:, :skip_channels].copy_(skip)
    duplicated = joined.view(batch, skip_channels + value_channels,
                             depth, 2, height, 2, width, 2)
    duplicated[:, skip_channels:].copy_(value[:, :, :, None, :, None, :, None])
    return joined
