"""PyTorch version of the single-input ``mri_synthsr`` U-Net."""

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SynthSRUNet(nn.Module):
    """Five-level 3D U-Net used by the official SynthSR checkpoints."""

    def __init__(self):
        super().__init__()
        channels = (24, 48, 96, 192, 384)
        self.down = nn.ModuleList()
        self.down_bn = nn.ModuleList()
        previous = 1
        for width in channels:
            self.down.append(nn.ModuleList((
                nn.Conv3d(previous, width, 3, padding=1),
                nn.Conv3d(width, width, 3, padding=1),
            )))
            self.down_bn.append(nn.BatchNorm3d(width, eps=1e-3))
            previous = width

        self.up = nn.ModuleList()
        self.up_bn = nn.ModuleList()
        for width in reversed(channels[:-1]):
            self.up.append(nn.ModuleList((
                nn.Conv3d(previous + width, width, 3, padding=1),
                nn.Conv3d(width, width, 3, padding=1),
            )))
            self.up_bn.append(nn.BatchNorm3d(width, eps=1e-3))
            previous = width
        self.likelihood = nn.Conv3d(channels[0], 1, 1)

    def forward(self, image):
        if image.device.type == "cpu" and image.dtype == torch.float32 and not torch.is_grad_enabled():
            from ._cpu_inference import cpu_forward_if_supported
            output = cpu_forward_if_supported(self, image)
            if output is not None:
                return output
        skips = []
        value = image
        for level, (convs, norm) in enumerate(zip(self.down, self.down_bn)):
            value = F.elu(convs[0](value))
            value = F.elu(convs[1](value))
            skips.append(value)  # The Keras skip taps the convolution before BN.
            value = norm(value)
            if level < len(self.down) - 1:
                value = F.max_pool3d(value, 2)
        for convs, norm, skip in zip(self.up, self.up_bn, reversed(skips[:-1])):
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = torch.cat((skip, value), dim=1)
            value = F.elu(convs[0](value))
            value = F.elu(convs[1](value))
            value = norm(value)
        return self.likelihood(value)


def _read_layer(root, name, keys):
    if name not in root or name not in root[name]:
        raise ValueError(f"Missing SynthSR layer {name} in checkpoint")
    group = root[name][name]
    return [np.asarray(group[key + ":0"]) for key in keys]


def _copy_conv(root, name, layer):
    kernel, bias = _read_layer(root, name, ("kernel", "bias"))
    weight = np.transpose(kernel, (4, 3, 0, 1, 2)).copy()
    if weight.shape != tuple(layer.weight.shape) or bias.shape != tuple(layer.bias.shape):
        raise ValueError(f"Unexpected shape for SynthSR layer {name}")
    layer.weight.copy_(torch.from_numpy(weight))
    layer.bias.copy_(torch.from_numpy(bias))


def _copy_bn(root, name, layer):
    values = _read_layer(root, name, ("gamma", "beta", "moving_mean", "moving_variance"))
    for value, target in zip(values, (layer.weight, layer.bias, layer.running_mean, layer.running_var)):
        if value.shape != tuple(target.shape):
            raise ValueError(f"Unexpected shape for SynthSR layer {name}")
        target.copy_(torch.from_numpy(value))


def load_h5_weights(model, checkpoint_path):
    """Read the named Keras layers from an official SynthSR .h5 checkpoint."""
    with h5py.File(checkpoint_path, "r") as root, torch.no_grad():
        for level, (convs, norm) in enumerate(zip(model.down, model.down_bn)):
            for index, conv in enumerate(convs):
                _copy_conv(root, f"unet_conv_downarm_{level}_{index}", conv)
            _copy_bn(root, f"unet_bn_down_{level}", norm)
        for index, (convs, norm) in enumerate(zip(model.up, model.up_bn)):
            for subindex, conv in enumerate(convs):
                _copy_conv(root, f"unet_conv_uparm_{index + 5}_{subindex}", conv)
            _copy_bn(root, f"unet_bn_up_{index}", norm)
        _copy_conv(root, "unet_likelihood", model.likelihood)
    return model
