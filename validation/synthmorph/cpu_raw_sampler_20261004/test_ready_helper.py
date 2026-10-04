"""Finite helper guards and input contracts; no synthetic benchmark claim."""
import sys
from pathlib import Path

import numba
import numpy as np
import pytest
import torch
from fnit.synthmorph import _cpu_preprocessing, spatial

sys.path.insert(0, str(Path(__file__).parent))
import _cpu_raw_sampler_ready as helper


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
def test_compiled_batches_channels_strides_singletons_and_fill_exact(source, fill):
    volume, matrix, locations = inputs(source)
    before = volume.clone(); numba.set_num_threads(2); torch.set_num_threads(8)
    with torch.inference_mode():
        actual = helper.try_sample(volume, locations, fill)
        expected = _cpu_preprocessing.network_transform(volume, matrix, shape=(32,) * 3, fill_value=fill)
    assert helper.backend_info()['backend'] == 'numba', helper.backend_info()
    assert helper.backend_info()['requested_threads'] == 2
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


def test_autograd_and_small_grid_guards(monkeypatch):
    volume, matrix, locations = inputs(batch=1, channels=1)
    monkeypatch.setattr(helper, '_kernel', lambda: pytest.fail('guard called kernel'))
    assert helper.try_sample(volume, locations) is None
    with torch.inference_mode(): assert helper.try_sample(volume, locations[..., :1, :1, :1]) is None
