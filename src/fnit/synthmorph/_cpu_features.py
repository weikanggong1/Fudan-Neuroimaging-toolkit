"""CPU joint inference with the declared channels-last convolution order."""
import torch
from torch.nn import functional as F


def supported_inference(detector, image):
    """Only bypass Conv3d calls for ordinary unobserved FP32 eval inference."""
    try:
        autocast = torch.is_autocast_enabled('cpu')
    except TypeError:  # Torch 2.1 retains the device-specific API.
        autocast = torch.is_autocast_cpu_enabled()
    if (image.device.type != 'cpu' or image.dtype != torch.float32 or image.ndim != 5
            or torch.is_grad_enabled() or detector.training or autocast
            or not torch.backends.mkldnn.is_available() or not torch.backends.mkldnn.enabled):
        return False
    module = torch.nn.modules.module
    if any(getattr(module, name, {}) for name in
           ('_global_forward_pre_hooks', '_global_forward_hooks',
            '_global_backward_pre_hooks', '_global_backward_hooks')):
        return False
    layers = getattr(detector, 'layers', ())
    if len(layers) != 9:
        return False
    for layer in layers:
        if (type(layer) is not torch.nn.Conv3d or layer.training or layer.padding_mode != 'zeros'
                or layer.weight.device.type != 'cpu' or layer.weight.dtype != torch.float32
                or (layer.bias is not None and (layer.bias.device.type != 'cpu' or layer.bias.dtype != torch.float32))
                or any(getattr(layer, name, {}) for name in
                       ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks'))):
            return False
    return True


def detector_features(detector, image):
    if not supported_inference(detector, image):
        # This helper can be reached inside detector.__call__. Calling that
        # entry a second time would duplicate its own local forward hooks.
        return detector.forward(image)
    result = image
    for index, layer in enumerate(detector.layers):
        result = result.contiguous(memory_format=torch.channels_last_3d)
        kernel = layer.weight.contiguous(memory_format=torch.channels_last_3d)
        # F.conv3d selects Slow3d for these small FP32 grids in Torch 2.5.1,
        # even with channels-last storage. Select the existing oneDNN kernel
        # explicitly; keep the parameter itself and caller backend policy.
        result = torch.mkldnn_convolution(result, kernel, layer.bias,
                                         list(layer.padding), list(layer.stride),
                                         list(layer.dilation), layer.groups)
        if index == 8:
            result = F.relu(result)
        else:
            result = F.leaky_relu(result, .2)
            if index < 4:
                result = F.max_pool3d(result, 2)
    return result
