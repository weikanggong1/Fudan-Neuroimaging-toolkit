"""Finite helper guards and input contracts; no synthetic benchmark claim."""
import numba
import numpy as np
import pytest
import torch
from fnit.synthmorph import _cpu_preprocessing, spatial

from fnit.synthmorph import _cpu_raw_sampler as helper


@pytest.fixture(autouse=True)
def restore_state(monkeypatch):
    previous_torch, previous_numba = torch.get_num_threads(), numba.get_num_threads()
    monkeypatch.setattr(helper, '_FAILURE', None)
    yield
    torch.set_num_threads(previous_torch); numba.set_num_threads(previous_numba)


def inputs(source=(3, 4, 5), batch=2, channels=3):
    volume = torch.randn((batch, channels, *source[:-1], 2 * source[-1]),
                         generator=torch.Generator().manual_seed(817))[..., ::2]
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([-.5, .5, 0.])
    coords = spatial.grid((32,) * 3, 'cpu')
    locations = coords + spatial._dense_from_grid(matrix, coords)
    return volume, matrix, locations


@pytest.mark.parametrize('source', [(3, 4, 5), (1, 4, 5), (1, 1, 1)])
@pytest.mark.parametrize('fill', [0., None, 3.25])
def test_compiled_batches_channels_strides_singletons_and_fill_exact(source, fill, monkeypatch):
    volume, matrix, locations = inputs(source)
    before = volume.clone(); numba.set_num_threads(2); torch.set_num_threads(8)
    with torch.inference_mode():
        actual = helper.try_sample(volume, locations, fill)
        route = helper.backend_info()
        with monkeypatch.context() as oracle:
            # Keep the existing Torch loop as oracle after production integration.
            oracle.setenv('FNIT_SYNTHMORPH_CPU_RAW_NUMBA', '0')
            expected = _cpu_preprocessing.network_transform(volume, matrix, shape=(32,) * 3, fill_value=fill)
    assert route['backend'] == 'numba', route
    assert route['requested_threads'] == 2
    assert numba.get_num_threads() == 2 and torch.get_num_threads() == 8
    assert np.array_equal(actual.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    assert actual.is_contiguous() and actual.data_ptr() != volume.data_ptr()
    assert torch.equal(volume, before)


@pytest.mark.parametrize('kind', ['volume_nan', 'volume_inf', 'coordinate_nan', 'fill_nan',
                                 'disabled', 'jit_disabled', 'unavailable', 'dtype', 'grad', 'autocast'])
def test_unsupported_or_disabled_returns_torch_without_kernel(monkeypatch, kind):
    volume, matrix, locations = inputs(batch=1, channels=1); fill = 0
    if kind == 'volume_nan': volume[0, 0, 0, 0, 0] = float('nan')
    if kind == 'volume_inf': volume[0, 0, 0, 0, 0] = float('inf')
    if kind == 'coordinate_nan': locations[0, 0, 0, 0, 0] = float('nan')
    if kind == 'fill_nan': fill = float('nan')
    if kind == 'disabled': monkeypatch.setenv('FNIT_SYNTHMORPH_CPU_RAW_NUMBA', '0')
    if kind == 'jit_disabled': monkeypatch.setattr(numba.config, 'DISABLE_JIT', True)
    if kind == 'unavailable': monkeypatch.setattr(helper, 'numba', None)
    if kind == 'dtype': volume = volume.double()
    if kind == 'grad': volume.requires_grad_(True)
    monkeypatch.setattr(helper, '_kernel', lambda: pytest.fail('fallback compiled'))
    previous = numba.get_num_threads()
    with torch.inference_mode(), torch.autocast('cpu', enabled=kind == 'autocast'):
        assert helper.try_sample(volume, locations, fill) is None
    assert helper.backend_info()['backend'] == 'torch'
    assert numba.get_num_threads() == previous


def test_numba_failure_falls_back_once_and_restores_mask(monkeypatch):
    volume, matrix, locations = inputs(batch=1, channels=1); observed = []
    numba.set_num_threads(4); torch.set_num_threads(2)
    def fail(*args):
        observed.append(numba.get_num_threads()); raise RuntimeError('controlled compilation failure')
    monkeypatch.setattr(helper, '_kernel', lambda: fail)
    with torch.inference_mode():
        for _ in range(2): assert helper.try_sample(volume, locations) is None
    assert observed == [2] and numba.get_num_threads() == 4
    assert 'controlled compilation failure' in helper.backend_info()['reason']


def test_thread_initialization_failure_falls_back_once(monkeypatch):
    volume, _, locations = inputs(batch=1, channels=1)
    attempted = []
    def failed_initialization():
        attempted.append(True)
        raise RuntimeError('controlled threading-layer failure')
    monkeypatch.setattr(numba, 'get_num_threads', failed_initialization)
    monkeypatch.setattr(helper, '_kernel', lambda: pytest.fail('initialization compiled'))
    with torch.inference_mode():
        for _ in range(2):
            assert helper.try_sample(volume, locations) is None
    assert attempted == [True]
    assert 'controlled threading-layer failure' in helper.backend_info()['reason']


@pytest.mark.parametrize('kind', ['gradient', 'double', 'other_device'])
def test_network_guard_returns_before_importing_optional_helper(monkeypatch, kind):
    import builtins
    volume = torch.ones((1, 1, 3, 4, 5),
                        device='meta' if kind == 'other_device' else 'cpu',
                        dtype=torch.float64 if kind == 'double' else torch.float32)
    sentinel = object()
    monkeypatch.setattr(_cpu_preprocessing, 'transform', lambda *args, **kwargs: sentinel)
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if '_cpu_raw_sampler' in name:
            pytest.fail('non-inference route imported optional helper')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    with torch.set_grad_enabled(kind == 'gradient'):
        assert _cpu_preprocessing.network_transform(volume, torch.eye(4)) is sentinel


def test_failed_compiled_network_sampling_retains_ordered_torch_result(monkeypatch):
    volume, matrix, _ = inputs(batch=1, channels=1)
    with torch.inference_mode(), monkeypatch.context() as disabled:
        disabled.setenv('FNIT_SYNTHMORPH_CPU_RAW_NUMBA', '0')
        expected = _cpu_preprocessing.network_transform(volume, matrix, shape=(32,) * 3)
    def failed_kernel(*args):
        raise RuntimeError('controlled compile failure')
    monkeypatch.setattr(helper, '_kernel', lambda: failed_kernel)
    with torch.inference_mode():
        actual = _cpu_preprocessing.network_transform(volume, matrix, shape=(32,) * 3)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert 'controlled compile failure' in helper.backend_info()['reason']


def test_autograd_and_small_grid_guards(monkeypatch):
    volume, matrix, locations = inputs(batch=1, channels=1)
    monkeypatch.setattr(helper, '_kernel', lambda: pytest.fail('guard called kernel'))
    assert helper.try_sample(volume, locations) is None
    with torch.inference_mode(): assert helper.try_sample(volume, locations[..., :1, :1, :1]) is None


def test_default_policy_subnormal_and_signed_zero_contract(monkeypatch):
    # Leave the process floating policy untouched, including the worker pools.
    bits = np.resize(np.array([0, 0x80000000, 1, 0x80000001,
                               0x007fffff, 0x807fffff], dtype=np.uint32), (1, 1, 3, 4, 5))
    volume = torch.from_numpy(bits.view(np.float32))
    matrix = torch.eye(4); matrix[:3, 3] = .5
    coords = spatial.grid((32,) * 3, 'cpu')
    locations = coords + spatial._dense_from_grid(matrix, coords)
    before_volume = volume.numpy().view(np.uint32).copy()
    before_locations = locations.numpy().view(np.uint32).copy()
    with torch.inference_mode():
        actual = helper.try_sample(volume, locations)
        route = helper.backend_info()
        with monkeypatch.context() as oracle:
            oracle.setenv('FNIT_SYNTHMORPH_CPU_RAW_NUMBA', '0')
            expected = _cpu_preprocessing.network_transform(volume, matrix, shape=(32,) * 3)
    assert route['backend'] == 'numba', route
    assert np.array_equal(actual.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    assert np.array_equal(volume.numpy().view(np.uint32), before_volume)
    assert np.array_equal(locations.numpy().view(np.uint32), before_locations)
