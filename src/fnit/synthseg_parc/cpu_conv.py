"""Bounded CPU-only convolution slabs for SynthSeg inference.

Large NCDHW convolutions have previously crashed in oneDNN on nodecw10;
disabling it for the entire U-Net uses a very large im2col workspace. Slabs
retain each kernel's complete depth halo and the guarded original backend.
CUDA and gradient-enabled/training forwards retain nn.Conv3d unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F


_ONEDNN_POINTWISE_BYTES = 2**31 - 1


def _pointwise_blocked_bytes(image, output_channels):
    """Conservatively include the AVX-512 channel padding observed on nodecw10."""
    channels = ((max(image.shape[1], output_channels) + 15) // 16) * 16
    return image.shape[0] * channels * image.shape[2] * image.shape[3] * image.shape[4] * image.element_size()


def _pointwise_onednn_slabs(image, weight, bias=None,
                          maximum_slab_bytes=256 * 1024 * 1024):
    """Bound a large CPU 1x1x1 call while retaining the enabled oneDNN backend.

    This projection has no spatial halo. Every voxel and every input channel
    remains in its original order. Only the depth extent of each independent
    call changes. No backend, precision, thread or CUDA flag is written.
    The byte limit is a conservative guard for the observed large-volume
    failure, rather than a claim about the exact native addressing defect.
    """
    batch, _, depth, height, width = image.shape
    channels = ((max(image.shape[1], weight.shape[0]) + 15) // 16) * 16
    plane_bytes = batch * channels * height * width * image.element_size()
    slab_depth = max(1, min(32, maximum_slab_bytes // plane_bytes))
    output = image.new_empty((batch, weight.shape[0], depth, height, width))
    for start in range(0, depth, slab_depth):
        stop = min(depth, start + slab_depth)
        chunk = image[:, :, start:stop].contiguous()
        output[:, :, start:stop] = F.conv3d(chunk, weight, bias)
    return output


def cpu_autocast_enabled():
    """Query caller CPU autocast without changing it; compatible with Torch 2.1."""
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


def convolution_slabs(image, weight, bias=None, *, padding=0, groups=1,
                      maximum_slab_bytes=256 * 1024 * 1024):
    """CPU stride-one, dilation-one Conv3D with exact spatial neighborhoods.

    The target bounds individual input/output slabs, not the complete model
    RSS; at least one output plane and its full halo are always retained.
    Zero padding on depth boundaries is applied only at the true image
    edges; internal chunk edges use the neighboring original input planes.
    All arithmetic stays in the input dtype. The original non-oneDNN CPU
    backend remains selected in a restoring context: switching to oneDNN
    inside a slab changed near-tie labels on a real validation T1. No model,
    precision, thread or CUDA setting is changed.
    """
    if image.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("convolution_slabs accepts CPU tensors only")
    if image.ndim != 5 or weight.ndim != 5:
        raise ValueError("Expected five-dimensional image and convolution weight")
    padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    kernels = tuple(weight.shape[2:])
    if any(k != 2 * p + 1 for k, p in zip(kernels, padding)):
        raise ValueError("Slab convolution requires odd kernels with same-size padding")
    batch, _, depth, height, width = image.shape
    channels = max(image.shape[1], weight.shape[0])
    bytes_per_plane = batch * channels * height * width * image.element_size()
    slab_depth = max(1, min(32, maximum_slab_bytes // bytes_per_plane - 2 * padding[0]))
    output = image.new_empty((batch, weight.shape[0], depth, height, width))
    halo = padding[0]
    with torch.backends.mkldnn.flags(enabled=False):
        for start in range(0, depth, slab_depth):
            stop = min(start + slab_depth, depth)
            lower, upper = max(0, start - halo), min(depth, stop + halo)
            chunk = image[:, :, lower:upper]
            pad_before = max(0, halo - start)
            pad_after = max(0, stop + halo - depth)
            if pad_before or pad_after:
                chunk = F.pad(chunk, (0, 0, 0, 0, pad_before, pad_after))
            convolved = F.conv3d(chunk, weight, bias, padding=(0, padding[1], padding[2]),
                                groups=groups)
            output[:, :, start:stop] = convolved
    return output


class CPUInferenceConv3d(nn.Conv3d):
    """Use safe CPU slabs inside the existing full-volume oneDNN guard.

    SynthSeg's ordinary entry point disables full-volume oneDNN after large
    convolutions crashed on nodecw10. Only that guarded path needs slabs.
    SynthSegPlus retains its established oneDNN backend. Large 1x1x1 output
    projections use bounded depth calls after that native path crashed on
    a real T1; smaller projections and all other oneDNN layers stay whole.
    """

    def forward(self, image):
        if (image.ndim == 5 and image.device.type == "cpu"
                and not self.training and not torch.is_grad_enabled()
                and self.padding_mode == "zeros" and self.stride == (1, 1, 1)
                and self.dilation == (1, 1, 1) and image.dtype == torch.float32
                and not cpu_autocast_enabled()):
            if not torch.backends.mkldnn.enabled:
                if getattr(self, "_fnit_columns_reuse", False):
                    # Lazy CPU-only optional path; CUDA/training never imports it.
                    from .cpu_columns import try_columns_reuse
                    reused = try_columns_reuse(self, image)
                    if reused is not None:
                        return reused
                return convolution_slabs(image, self.weight, self.bias,
                                         padding=self.padding, groups=self.groups)
            if (self.kernel_size == (1, 1, 1) and self.padding == (0, 0, 0)
                    and self.groups == 1
                    and _pointwise_blocked_bytes(image, self.out_channels) > _ONEDNN_POINTWISE_BYTES):
                return _pointwise_onednn_slabs(image, self.weight, self.bias)
        return super().forward(image)
