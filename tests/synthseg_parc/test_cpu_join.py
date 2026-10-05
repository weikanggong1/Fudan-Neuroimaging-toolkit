"""Independent copies must preserve voxel bits, layout and observed forwards."""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fnit.synthseg_parc.cpu_join import cpu_join_allowed, join_nearest_cpu
from fnit.synthseg_parc.model import ParcUNet
from fnit.synthseg_parc.segment import SegmentUNet


@pytest.mark.parametrize("batch,skip_channels,value_channels,shape", [
    (1, 3, 5, (1, 3, 7)), (2, 7, 2, (5, 1, 3)), (1, 24, 48, (9, 11, 13))])
def test_exact_copies_preserve_channel_order_stride_bits_and_independence(
        batch, skip_channels, value_channels, shape):
    torch.manual_seed(20261005)
    value = torch.randn(batch, value_channels, *shape)
    value.flatten()[:4] = torch.tensor([-0.0, 0.0, float("inf"), float("nan")])
    skip = torch.randn(batch, skip_channels, *(2 * size for size in shape))
    before = (skip.clone(), value.clone())
    expected = torch.cat((skip, F.interpolate(value, scale_factor=2, mode="nearest")), 1)
    with torch.inference_mode():
        actual = join_nearest_cpu(skip, value)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert actual.stride() == expected.stride() and actual.is_contiguous()
    assert actual.data_ptr() not in (skip.data_ptr(), value.data_ptr())
    with torch.inference_mode():
        actual.zero_()
    assert torch.equal(skip.view(torch.int32), before[0].view(torch.int32))
    assert torch.equal(value.view(torch.int32), before[1].view(torch.int32))


def test_guards_preserve_training_grad_autocast_layout_and_wrong_scale():
    network = nn.Sequential(nn.Identity()).eval()
    value = torch.ones(1, 2, 3, 5, 7)
    skip = torch.ones(1, 4, 6, 10, 14)
    assert not cpu_join_allowed(network, skip, value)  # Default grad context.
    with torch.inference_mode():
        assert cpu_join_allowed(network, skip, value)
        network.train()
        assert not cpu_join_allowed(network, skip, value)
        network.eval()
        assert not cpu_join_allowed(network, skip.double(), value.double())
        assert not cpu_join_allowed(network, skip[:, :, :-1], value)
        assert not cpu_join_allowed(network, skip.to(memory_format=torch.channels_last_3d), value)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert not cpu_join_allowed(network, skip, value)


@pytest.mark.parametrize("hook_kind", ["forward", "forward_pre", "full_backward", "global_forward"])
def test_container_leaf_and_global_hooks_keep_original_route(hook_kind):
    network = nn.Sequential(nn.Identity()).eval()
    registrar = (nn.modules.module.register_module_forward_hook if hook_kind == "global_forward"
                 else getattr(network[0], "register_" + hook_kind + "_hook"))
    handle = registrar(lambda *args: None)
    try:
        with torch.inference_mode():
            assert not cpu_join_allowed(network, torch.ones(1, 4, 6, 10, 14),
                                        torch.ones(1, 2, 3, 5, 7))
    finally:
        handle.remove()


@pytest.mark.parametrize("network_type,channels", [(SegmentUNet, 1), (ParcUNet, 3)])
def test_two_networks_use_exact_join_and_observed_route_keeps_same_output(network_type, channels):
    torch.manual_seed(508)
    network = network_type().eval()
    image = torch.randn(1, channels, 32, 32, 32)
    module = "fnit.synthseg_parc." + ("segment" if channels == 1 else "model")
    with torch.inference_mode(), patch(module + ".join_nearest_cpu", wraps=join_nearest_cpu) as joined:
        actual = network(image)
        assert joined.call_count == 4
        with patch(module + ".cpu_join_allowed", return_value=False):
            expected = network(image)
        joined.reset_mock()
        observed_inputs = []
        handle = network.up[3].conv0.register_forward_pre_hook(
            lambda _layer, inputs: observed_inputs.append(inputs[0].shape))
        try:
            observed = network(image)
            assert joined.call_count == 0 and observed_inputs == [(1, 72, 32, 32, 32)]
        finally:
            handle.remove()
    assert torch.equal(actual, expected)
    assert torch.equal(observed, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA unavailable locally")
def test_cuda_rejects_cpu_route():
    network = nn.Sequential(nn.Identity()).eval().cuda()
    with torch.inference_mode():
        assert not cpu_join_allowed(network, torch.ones(1, 4, 6, 10, 14, device="cuda"),
                                    torch.ones(1, 2, 3, 5, 7, device="cuda"))
