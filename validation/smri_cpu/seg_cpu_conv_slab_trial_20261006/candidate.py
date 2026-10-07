"""Private single CPU layer cap trial; production never imports this module."""
import torch
from torch import nn


def eligible(layer, image):
    from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d, cpu_autocast_enabled
    if (not isinstance(layer, CPUInferenceConv3d) or image.ndim != 5
            or image.device.type != "cpu" or image.dtype != torch.float32
            or layer.weight.device.type != "cpu" or layer.weight.dtype != image.dtype
            or layer.bias is None or layer.bias.device.type != "cpu" or layer.bias.dtype != image.dtype
            or layer.in_channels != 72 or layer.out_channels != 24 or tuple(image.shape[:2]) != (1,72)
            or layer.kernel_size != (3,3,3) or layer.padding != (1,1,1)
            or layer.stride != (1,1,1) or layer.dilation != (1,1,1)
            or layer.groups != 1 or layer.padding_mode != "zeros"
            or layer.training or torch.is_grad_enabled() or cpu_autocast_enabled()
            or torch.backends.mkldnn.enabled):
        return False
    names=("_forward_pre_hooks","_forward_hooks","_backward_pre_hooks","_backward_hooks")
    if any(getattr(layer,name,{}) for name in names):
        return False
    global_names=("_global_forward_pre_hooks","_global_forward_hooks",
                  "_global_backward_pre_hooks","_global_backward_hooks")
    return not any(getattr(nn.modules.module,name,{}) for name in global_names)


def forward(layer, image, *, maximum_slab_bytes=64 * 1024**2):
    """Lower only one eligible layer's cap; other callers use the old module."""
    if not eligible(layer,image):
        return layer(image)
    from fnit.synthseg_parc.cpu_conv import convolution_slabs
    return convolution_slabs(image,layer.weight,layer.bias,padding=layer.padding,
                             groups=layer.groups,maximum_slab_bytes=maximum_slab_bytes)
