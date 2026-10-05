"""PyTorch implementation of SynthSeg 2.0's cortical parcellation U-Net.

Architecture and H5 layer names follow FreeSurfer ``mri_synthseg --parc``.
The input has three channels: normalized image, non-cortex, cortex.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cpu_conv import CPUInferenceConv3d
from .cpu_join import cpu_join_allowed, join_nearest_cpu


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = CPUInferenceConv3d(in_channels, out_channels, 3, padding=1)
        self.conv1 = CPUInferenceConv3d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3)

    def forward(self, x):
        x = F.elu(self.conv0(x))
        skip = F.elu(self.conv1(x))
        return self.bn(skip), skip


class ParcUNet(nn.Module):
    """Five-level U-Net with the 69 SynthSeg parcellation output channels."""

    def __init__(self):
        super().__init__()
        widths = (24, 48, 96, 192, 384)
        self.down = nn.ModuleList(_Block(3 if i == 0 else widths[i - 1], width)
                                  for i, width in enumerate(widths))
        self.up = nn.ModuleList(_Block(widths[i + 1] + widths[i], widths[i])
                                for i in (3, 2, 1, 0))
        self.likelihood = CPUInferenceConv3d(24, 69, 1)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 3 or any(size % 32 for size in x.shape[2:]):
            raise ValueError("input must have shape (B, 3, D, H, W), with spatial sizes divisible by 32")
        skips = []
        for level, block in enumerate(self.down):
            x, skip = block(x)
            skips.append(skip)  # Keras skip taps the second convolution before BN.
            if level < 4:
                x = F.max_pool3d(x, 2)
        skips.pop()  # The bottleneck has no decoder skip connection.
        del skip
        for level, block in enumerate(self.up):
            if cpu_join_allowed(self, skips[-1], x):
                x = block(join_nearest_cpu(skips.pop(), x))[0]
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = block(torch.cat((skips.pop(), x), dim=1))[0]
        logits = self.likelihood(x)
        del x
        return torch.softmax(logits, dim=1)

    def load_h5(self, path: str | Path):
        """Load official Keras H5 weights, transposing Conv3D kernels to PyTorch."""
        def load_conv(h5, name, layer):
            group = h5[name][name]
            kernel = np.asarray(group["kernel:0"][()], dtype=np.float32).transpose(4, 3, 0, 1, 2)
            bias = np.asarray(group["bias:0"][()], dtype=np.float32)
            if kernel.shape != tuple(layer.weight.shape) or bias.shape != tuple(layer.bias.shape):
                raise ValueError(f"Unexpected weight shape for {name}")
            layer.weight.copy_(torch.from_numpy(kernel.copy()))
            layer.bias.copy_(torch.from_numpy(bias.copy()))

        def load_bn(h5, name, layer):
            group = h5[name][name]
            for source, target in (("gamma:0", layer.weight), ("beta:0", layer.bias),
                                   ("moving_mean:0", layer.running_mean),
                                   ("moving_variance:0", layer.running_var)):
                values = np.asarray(group[source][()], dtype=np.float32)
                if values.shape != tuple(target.shape):
                    raise ValueError(f"Unexpected weight shape for {name}/{source}")
                target.copy_(torch.from_numpy(values.copy()))

        with torch.no_grad(), h5py.File(path, "r") as h5:
            for level, block in enumerate(self.down):
                for conv, layer in enumerate((block.conv0, block.conv1)):
                    load_conv(h5, f"unet_parc_conv_downarm_{level}_{conv}", layer)
                load_bn(h5, f"unet_parc_bn_down_{level}", block.bn)
            for level, block in enumerate(self.up):
                for conv, layer in enumerate((block.conv0, block.conv1)):
                    load_conv(h5, f"unet_parc_conv_uparm_{level + 5}_{conv}", layer)
                load_bn(h5, f"unet_parc_bn_up_{level}", block.bn)
            load_conv(h5, "unet_parc_likelihood", self.likelihood)
        return self
