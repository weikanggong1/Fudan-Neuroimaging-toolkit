"""State, dispatch and caller thread contracts for the CPU reference path.

Numerical acceptance uses the complete real server oracle, not these small
arrays. Small tensors here exercise hooks, gradients and autocast contracts.
"""
import copy
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fnit.synthsr.model import SynthSRUNet
from fnit.synthsr import _cpu_inference


@pytest.fixture(autouse=True)
def thread_limit():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def legacy(model, image):
    skips = []
    value = image
    for level, (convs, norm) in enumerate(zip(model.down, model.down_bn)):
        value = F.elu(convs[0](value)); value = F.elu(convs[1](value))
        skips.append(value); value = norm(value)
        if level < len(model.down)-1:
            value = F.max_pool3d(value, 2)
    for convs, norm, skip in zip(model.up, model.up_bn, reversed(skips[:-1])):
        value = torch.cat((skip, F.interpolate(value, scale_factor=2, mode='nearest')), dim=1)
        value = F.elu(convs[0](value)); value = F.elu(convs[1](value)); value = norm(value)
    return model.likelihood(value)


def test_non_cpu_forward_does_not_import_cpu_helpers():
    program = """
import sys, torch
from fnit.synthsr.model import SynthSRUNet
with torch.device('meta'):
    model=SynthSRUNet().eval()
    image=torch.empty(1,1,32,32,32)
with torch.inference_mode():
    result=model(image)
assert result.shape==(1,1,32,32,32)
assert 'fnit.synthsr._cpu_inference' not in sys.modules
assert 'fnit.synthsr._cpu_math' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', program], check=True, env=os.environ.copy())


def test_cpu_inference_preserves_parameters_buffers_strides_and_rng():
    if not _cpu_inference._supported_isa():
        pytest.skip('verified packet path requires Linux x86-64 SSE2/FMA')
    model = SynthSRUNet().eval()
    state = {name:value.clone() for name,value in model.state_dict().items()}
    parameters = [(id(value),value.data_ptr(),value.stride(),value._version) for value in model.parameters()]
    rng = torch.get_rng_state()
    with torch.inference_mode():
        assert _cpu_inference._eligible(model, torch.zeros(1,1,16,16,16))
        output = model(torch.zeros(1,1,16,16,16))
    assert torch.isfinite(output).all()
    assert parameters == [(id(value),value.data_ptr(),value.stride(),value._version) for value in model.parameters()]
    assert torch.equal(rng, torch.get_rng_state())
    for name,value in model.state_dict().items():
        assert torch.equal(value,state[name])
    assert all(norm.running_var.dtype == torch.float32 and norm.num_batches_tracked.dtype == torch.int64
               for norm in (*model.down_bn,*model.up_bn))


def test_training_and_gradients_match_original_module_calls():
    model = SynthSRUNet().train(); reference = copy.deepcopy(model)
    image = torch.randn(2,1,16,16,16,requires_grad=True)
    other = image.detach().clone().requires_grad_(True)
    output=model(image); expected=legacy(reference,other)
    assert torch.equal(output,expected)
    output.sum().backward(); expected.sum().backward()
    assert torch.equal(image.grad,other.grad)
    assert torch.equal(model.down[0][0].weight.grad,reference.down[0][0].weight.grad)
    assert all(norm.num_batches_tracked.item()==1 for norm in (*model.down_bn,*model.up_bn))
    for name,value in model.named_buffers():
        assert torch.equal(value,dict(reference.named_buffers())[name])


def test_cpu_autocast_preserves_original_dtype_and_values():
    model=SynthSRUNet().eval()
    image=torch.randn(1,1,16,16,16)
    with torch.inference_mode(), torch.autocast('cpu',dtype=torch.bfloat16):
        assert not _cpu_inference._eligible(model,image)
        actual=model(image); expected=legacy(model,image)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual,expected)


@pytest.mark.parametrize('scope', ('leaf','container','model','global'))
def test_hooks_use_original_module_calls(scope,monkeypatch):
    monkeypatch.setattr(_cpu_inference,'_supported_isa',lambda:True)
    model=SynthSRUNet().eval(); image=torch.zeros(1,1,16,16,16); events=[]
    def observe(module,args,output):
        events.append(type(module).__name__)
    if scope=='global':
        handle=nn.modules.module.register_module_forward_hook(observe)
    else:
        target={'leaf':model.down[0][0],'container':model.down[0],'model':model}[scope]
        handle=target.register_forward_hook(observe)
    try:
        with torch.inference_mode():
            assert not _cpu_inference._eligible(model,image)
            model(image)
        # ModuleList is iterated by the original forward, never called itself.
        assert bool(events) == (scope!='container')
    finally:
        handle.remove()


@pytest.mark.parametrize('replacement', ('conv_wrapper','norm_identity'))
def test_modified_leaf_modules_fall_back_with_original_values(replacement,monkeypatch):
    monkeypatch.setattr(_cpu_inference,'_supported_isa',lambda:True)
    model=SynthSRUNet().eval()
    if replacement=='conv_wrapper':
        model.down[0][0]=nn.Sequential(model.down[0][0],nn.Identity()).eval()
    else:
        model.down_bn[0]=nn.Identity().eval()
    image=torch.randn(1,1,16,16,16)
    with torch.inference_mode():
        assert not _cpu_inference._eligible(model,image)
        assert torch.equal(model(image),legacy(model,image))


def test_unsupported_isa_and_disabled_mkldnn_fall_back(monkeypatch):
    model=SynthSRUNet().eval(); image=torch.zeros(1,1,16,16,16)
    monkeypatch.setattr(_cpu_inference,'_supported_isa',lambda:False)
    with torch.inference_mode():
        assert not _cpu_inference._eligible(model,image)
    monkeypatch.setattr(_cpu_inference,'_supported_isa',lambda:True)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        assert not _cpu_inference._eligible(model,image)


def test_numba_mask_restored_on_success_and_exception():
    from numba import get_num_threads
    previous=get_num_threads(); torch.set_num_threads(1)
    with _cpu_inference._thread_budget():
        assert get_num_threads()==1
    assert get_num_threads()==previous
    with pytest.raises(RuntimeError,match='probe'):
        with _cpu_inference._thread_budget():
            assert get_num_threads()==1
            raise RuntimeError('probe')
    assert get_num_threads()==previous
