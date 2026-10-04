"""Release inference temporaries without changing tensors or autograd."""

import copy
import weakref

import pytest
import torch

from fnit.wmh_synthseg.model import UNet3D, _Encoder, _Decoder, _decoder_has_dispatch_hooks, _join_upsampled_skip
from fnit.wmh_synthseg.pipeline import WMHSynthSeg


@pytest.mark.parametrize('requires_grad', (False, True))
def test_decoder_preserves_original_values_and_gradients(requires_grad):
    torch.manual_seed(31)
    original = _Decoder(12, 4, num_groups=4)
    candidate = copy.deepcopy(original)
    skip = torch.randn(1, 4, 8, 8, 8, requires_grad=requires_grad)
    lower = torch.randn(1, 8, 4, 4, 4, requires_grad=requires_grad)
    candidate_skip = skip.detach().clone().requires_grad_(requires_grad)
    candidate_lower = lower.detach().clone().requires_grad_(requires_grad)
    upsampled = original.upsampling(skip, lower)
    expected = original.basic_module(torch.cat((skip, upsampled), dim=1))
    actual = candidate(candidate_skip, candidate_lower)
    assert torch.equal(expected, actual)
    if requires_grad:
        expected.square().sum().backward()
        actual.square().sum().backward()
        assert torch.equal(skip.grad, candidate_skip.grad)
        assert torch.equal(lower.grad, candidate_lower.grad)
        for first, second in zip(original.parameters(), candidate.parameters()):
            assert torch.equal(first.grad, second.grad)


def test_inference_releases_upsample_before_groupnorm():
    model = _small_network()
    decoder = model.decoders[0]
    saved = {}
    original = decoder.upsampling.forward
    def upsampled(*args):
        result = original(*args)
        saved['upsample'] = weakref.ref(result)
        return result
    def before_norm(module, args):
        assert saved['upsample']() is None
    decoder.upsampling.forward = upsampled
    decoder.basic_module.SingleConv1.groupnorm.register_forward_pre_hook(before_norm)
    with torch.no_grad():
        model(torch.ones(1, 1, 8, 8, 8))


@pytest.mark.parametrize('container', ('decoder', 'upsampling', 'double', 'single'))
@pytest.mark.parametrize('kind', ('forward', 'forward_pre', 'backward', 'backward_pre'))
def test_observed_container_dispatch_is_preserved(container, kind):
    decoder = _Decoder(12, 4, num_groups=4)
    target = {'decoder': decoder, 'upsampling': decoder.upsampling, 'double': decoder.basic_module,
              'single': decoder.basic_module.SingleConv1}[container]
    assert not _decoder_has_dispatch_hooks(decoder)
    register = getattr(target, 'register_'+('full_' if kind.startswith('backward') else '')+kind+'_hook')
    handle = register(lambda *args: None)
    assert _decoder_has_dispatch_hooks(decoder)
    handle.remove()
    assert not _decoder_has_dispatch_hooks(decoder)


def test_global_observation_preserves_container_dispatch():
    decoder = _Decoder(12, 4, num_groups=4)
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
    try:
        assert _decoder_has_dispatch_hooks(decoder)
    finally:
        handle.remove()


@pytest.mark.parametrize('shape', ((8, 8, 8), (7, 12, 9)))
def test_bounded_nearest_join_preserves_values_and_source_tensors(shape):
    skip = torch.randn(2, 4, *shape)
    value = torch.randn(2, 8, 3, 5, 4)
    saved_skip, saved_value = skip.clone(), value.clone()
    expected = torch.cat((skip, torch.nn.functional.interpolate(value, size=shape, mode='nearest')), dim=1)
    actual = _join_upsampled_skip(skip, value, max_block_bytes=64)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(skip, saved_skip)
    assert torch.equal(value, saved_value)


def _small_network():
    model = UNet3D.__new__(UNet3D)
    torch.nn.Module.__init__(model)
    model.encoders = torch.nn.ModuleList((_Encoder(1, 4, pooling=False, num_groups=2),
                                         _Encoder(4, 8, pooling=True, num_groups=2)))
    model.decoders = torch.nn.ModuleList((_Decoder(12, 4, num_groups=2),))
    model.final_conv = torch.nn.Conv3d(4, 3, 1)
    return model.eval()


def test_unet_container_forward_hooks_are_called_during_inference():
    model = _small_network()
    image = torch.randn(2, 1, 7, 12, 9)
    with torch.no_grad():
        expected = model(image)
    called = []
    decoder = model.decoders[0]
    for name, module in (('upsampling', decoder.upsampling),
                         ('single', decoder.basic_module.SingleConv1),
                         ('double', decoder.basic_module), ('decoder', decoder)):
        module.register_forward_hook(lambda module, args, output, _name=name: called.append(_name))
    with torch.no_grad():
        actual = model(image)
    assert torch.equal(actual, expected)
    assert called == ['upsampling', 'single', 'double', 'decoder']


@pytest.mark.parametrize('training', (False, True))
def test_unet_autograd_preserves_container_dispatch(training):
    model = _small_network().train(training)
    image = torch.randn(2, 1, 7, 12, 9, requires_grad=True)
    calls = []
    model.decoders[0].register_forward_hook(lambda *args: calls.append('decoder'))
    model(image).square().sum().backward()
    assert calls == ['decoder']
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize('previous', (False, True))
def test_public_mode_is_restored_across_crop_full_and_failure(previous):
    model = WMHSynthSeg.__new__(WMHSynthSeg)
    model.device = torch.device('cuda:0')
    model.model = _small_network()
    model.model._memory_efficient_inference = previous
    model._predict = lambda *args: model.model._memory_efficient_inference
    for crop in (False, True, False, True):
        assert model('unused', crop=crop) == crop
        assert model.model._memory_efficient_inference == previous
    def fail(*args):
        assert not model.model._memory_efficient_inference
        raise ValueError('inference failed')
    model._predict = fail
    with pytest.raises(ValueError, match='inference failed'):
        model('unused', crop=False)
    assert model.model._memory_efficient_inference == previous
