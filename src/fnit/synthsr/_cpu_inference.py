"""Guarded CPU inference; original module state and CUDA storage stay intact."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import platform

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _supported_isa():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        return False
    try:
        flags = Path('/proc/cpuinfo').read_text().split('flags', 1)[1].split('\n', 1)[0].split()
    except (OSError, IndexError):
        return False
    return {'sse2', 'fma'}.issubset(flags)


def _has_hooks(model):
    from torch.nn.modules import module
    names = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
    global_names = ('_global_forward_pre_hooks', '_global_forward_hooks',
                    '_global_backward_pre_hooks', '_global_backward_hooks')
    return (any(getattr(module, name, {}) for name in global_names)
            or any(getattr(child, name, {}) for child in model.modules() for name in names))


def _cpu_autocast_enabled():
    try:
        return torch.is_autocast_enabled('cpu')
    except TypeError:  # Torch 2.1 uses the separate CPU accessor.
        return torch.is_autocast_cpu_enabled()


def _eligible(model, image):
    if (image.device.type != 'cpu' or image.dtype != torch.float32 or image.ndim != 5
            or image.shape[:2] != (1, 1) or image.requires_grad or torch.is_grad_enabled()
            or _cpu_autocast_enabled()
            or not torch.backends.mkldnn.enabled or not torch.backends.mkldnn.is_available()
            or not _supported_isa() or _has_hooks(model)):
        return False
    from .model import SynthSRUNet
    if type(model) is not SynthSRUNet or any(child.training for child in model.modules()):
        return False
    if (any(type(items) is not nn.ModuleList for items in (model.down, model.up, model.down_bn, model.up_bn))
            or any(type(convs) is not nn.ModuleList for convs in (*model.down, *model.up))):
        return False
    if (len(model.down) != 5 or len(model.up) != 4 or len(model.down_bn) != 5 or len(model.up_bn) != 4
            or any(len(convs) != 2 for convs in (*model.down, *model.up))
            or any(type(layer) is not nn.Conv3d for convs in (*model.down, *model.up) for layer in convs)
            or type(model.likelihood) is not nn.Conv3d
            or any(type(norm) is not nn.BatchNorm3d for norm in (*model.down_bn, *model.up_bn))):
        return False
    if any(value.device.type != 'cpu' or value.dtype != torch.float32 for value in model.parameters()):
        return False
    for child in model.modules():
        if isinstance(child, nn.Conv3d):
            if (type(child) is not nn.Conv3d or 'forward' in child.__dict__
                    or child.padding_mode != 'zeros' or child.groups != 1
                    or child.stride != (1, 1, 1) or child.dilation != (1, 1, 1)):
                return False
        elif isinstance(child, nn.BatchNorm3d):
            if (type(child) is not nn.BatchNorm3d or 'forward' in child.__dict__
                    or not child.affine or not child.track_running_stats or child.eps != 1e-3
                    or child.running_mean is None or child.running_var is None
                    or child.running_mean.device.type != 'cpu' or child.running_var.device.type != 'cpu'
                    or child.running_mean.dtype != torch.float32 or child.running_var.dtype != torch.float32):
                return False
            if (child.num_batches_tracked is None or child.num_batches_tracked.device.type != 'cpu'
                    or child.num_batches_tracked.dtype != torch.int64):
                return False
    return True


@contextmanager
def _thread_budget():
    from numba import get_num_threads, set_num_threads
    previous = get_num_threads()
    selected = min(previous, torch.get_num_threads())
    try:
        if selected != previous:
            set_num_threads(selected)
        yield
    finally:
        if selected != previous:
            set_num_threads(previous)


def _convolve(layer, value):
    # This private CPU view is rebuilt each call: edits to weights remain
    # visible, and moving the public model to CUDA keeps its original strides.
    weight = layer.weight.detach().contiguous(memory_format=torch.channels_last_3d)
    return F.conv3d(value, weight, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)


def cpu_forward_if_supported(model, image):
    """Return None for original-path calls, otherwise compute CPU inference."""
    if not _eligible(model, image):
        return None
    from ._cpu_math import _elu_tensor, _batch_norm_tensor
    with _thread_budget():
        skips = []
        value = image.contiguous(memory_format=torch.channels_last_3d)
        for level, (convs, norm) in enumerate(zip(model.down, model.down_bn)):
            value = _elu_tensor(_convolve(convs[0], value))
            value = _elu_tensor(_convolve(convs[1], value))
            skips.append(value)
            value = _batch_norm_tensor(value, norm)
            if level < len(model.down) - 1:
                value = F.max_pool3d(value, 2)
        for convs, norm, skip in zip(model.up, model.up_bn, reversed(skips[:-1])):
            value = F.interpolate(value, scale_factor=2, mode='nearest')
            value = torch.cat((skip, value), dim=1)
            value = _elu_tensor(_convolve(convs[0], value))
            value = _elu_tensor(_convolve(convs[1], value))
            value = _batch_norm_tensor(value, norm)
        return _convolve(model.likelihood, value)
