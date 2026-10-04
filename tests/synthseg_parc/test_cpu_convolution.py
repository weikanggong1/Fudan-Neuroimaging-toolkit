"""CPU slabs keep kernel halos/bias/groups and leave training paths intact."""

import pytest
import torch
from torch.nn import functional as F
from unittest.mock import patch

from fnit.synthseg_parc.cpu_conv import CPUInferenceConv3d, convolution_slabs


@pytest.mark.parametrize("kernel,groups,input_channels,output_channels", [
    (3, 1, 3, 5), (3, 3, 3, 3), (1, 1, 4, 7)])
@pytest.mark.parametrize("shape", [(1, 7, 9), (9, 7, 5), (33, 5, 7)])
def test_slab_halos_match_full_convolution_at_every_boundary(kernel, groups,
                                                             input_channels,
                                                             output_channels, shape):
    torch.manual_seed(113)
    image = torch.randn(1, input_channels, *shape)
    weight = torch.randn(output_channels, input_channels // groups, kernel, kernel, kernel)
    bias = torch.randn(output_channels)
    padding = kernel // 2
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        expected = F.conv3d(image, weight, bias, padding=padding, groups=groups)
        actual = convolution_slabs(image, weight, bias, padding=padding, groups=groups,
                                   maximum_slab_bytes=4096)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=3e-5)


def test_local_backend_context_restores_disabled_setting():
    with torch.backends.mkldnn.flags(enabled=False), torch.inference_mode():
        convolution_slabs(torch.ones(1, 1, 5, 4, 3), torch.ones(1, 1, 3, 3, 3), padding=1)
        assert not torch.backends.mkldnn.enabled


def test_slab_preserves_original_backend_even_when_caller_enabled_onednn():
    original = F.conv3d
    seen = []

    def record_backend(*args, **kwargs):
        seen.append(torch.backends.mkldnn.enabled)
        return original(*args, **kwargs)

    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=True):
        with patch("fnit.synthseg_parc.cpu_conv.F.conv3d", side_effect=record_backend):
            convolution_slabs(torch.ones(1, 1, 5, 4, 3),
                              torch.ones(1, 1, 3, 3, 3), padding=1)
        assert torch.backends.mkldnn.enabled
    assert seen and not any(seen)


@pytest.mark.parametrize("full_volume_backend", [False, True])
def test_inference_preserves_full_volume_path_and_uses_slabs_only_under_guard(full_volume_backend):
    layer = CPUInferenceConv3d(2, 3, 3, padding=1).eval()
    image = torch.randn(1, 2, 5, 6, 7)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=full_volume_backend):
        expected = F.conv3d(image, layer.weight, layer.bias, padding=1)
        with patch("fnit.synthseg_parc.cpu_conv.convolution_slabs", wraps=convolution_slabs) as slabs:
            actual = layer(image)
        assert slabs.call_count == (0 if full_volume_backend else 1)
        assert torch.backends.mkldnn.enabled == full_volume_backend
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=3e-5)


def test_training_and_autograd_use_original_conv_path():
    layer = CPUInferenceConv3d(2, 3, 3, padding=1)
    image = torch.randn(1, 2, 5, 6, 7, requires_grad=True)
    expected = F.conv3d(image, layer.weight, layer.bias, padding=1)
    actual = layer(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert image.grad is not None and layer.weight.grad is not None


def test_caller_autocast_keeps_original_output_dtype_and_values():
    layer = CPUInferenceConv3d(2, 3, 3, padding=1).eval()
    image = torch.randn(1, 2, 5, 6, 7)
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = F.conv3d(image, layer.weight, layer.bias, padding=1)
        actual = layer(image)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
