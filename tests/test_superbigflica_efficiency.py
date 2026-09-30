"""Verify cached input order, training equivalence and bounded fallback."""

import contextlib
import json

import h5py
import numpy as np
import pytest
import torch

from fnit.bigflica.stats_torch import SpatialRegression
from fnit.superbigflica import pipeline
from test_superbigflica import _mixed_inputs


def _stores(stack, tmp_path):
    arrays = [np.arange(77, dtype=np.float32).reshape(7, 11),
              np.arange(49, dtype=np.float32).reshape(7, 7) / 3]
    files = []
    for index, array in enumerate(arrays):
        file = stack.enter_context(h5py.File(tmp_path / f'{index}.h5', 'w'))
        file.create_dataset('data', data=array, chunks=(1, 4))
        files.append(file)
    return files, arrays


def test_cpu_cache_fallback_uses_original_stores(tmp_path):
    with contextlib.ExitStack() as stack:
        files, _ = _stores(stack, tmp_path)
        assert pipeline._cache_inputs(files, torch.device('cpu'), 2**30, 0, 3) is files


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_cache_exact_rows_and_budget_or_oom_fallback(tmp_path, monkeypatch):
    device = torch.device('cuda:0')
    with contextlib.ExitStack() as stack:
        files, arrays = _stores(stack, tmp_path)
        data_bytes = sum(array.nbytes for array in arrays)
        assert pipeline._cache_inputs(files, device, data_bytes - 1, 0, 3) is files
        cache = pipeline._cache_inputs(files, device, data_bytes, 0, 3)
        rows = np.array([5, 0, 5, 6, 2, 1])
        for actual, array in zip(pipeline._batch(cache, rows, device), arrays):
            assert actual.dtype == torch.float32 and actual.device == device
            np.testing.assert_array_equal(actual.cpu().numpy(), array[rows])

        original_empty = torch.empty
        calls = 0

        def limited_empty(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.cuda.OutOfMemoryError('Test partial allocation failure')
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, 'empty', limited_empty)
        assert pipeline._cache_inputs(files, device, data_bytes, 0, 3) is files
        assert calls == 2


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_tensor_spatial_regression_matches_independent_numpy_reference(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    rng = np.random.default_rng(73)
    courses = rng.normal(size=(32, 3))
    projected = rng.normal(size=(17, 32)).astype(np.float32)
    design = np.column_stack((courses, np.ones(len(courses))))
    beta = np.linalg.lstsq(design, projected.T, rcond=None)[0]
    residual = projected.T - design @ beta
    variance = np.sum(residual**2, axis=0) / (len(courses) - design.shape[1])
    expected = (beta[:-1] / np.sqrt(
        np.diag(np.linalg.inv(design.T @ design))[:-1, None] * variance)).T
    regression = SpatialRegression(courses, device=device)
    actual = regression.t(torch.as_tensor(projected, device=device))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(actual, regression.t(projected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_cached_training_matches_hdf_training(tmp_path, monkeypatch):
    root, modalities, _, _, table, _ = _mixed_inputs(tmp_path)
    options = dict(n_components=2, split_column='split', max_epochs=3, batch_size=3,
                   learning_rate=.001, dropout=.1, random_state=42, device='cuda:0',
                   max_gpu_gb=.064, feature_block=3, top_voxels=2, make_plots=False)
    cached = pipeline.run_superbigflica(
        root, modalities, table, {'score': 'continuous', 'group': 'categorical'},
        tmp_path / 'cached', **options)
    monkeypatch.setattr(pipeline, '_cache_inputs', lambda files, *_: files)
    streamed = pipeline.run_superbigflica(
        root, modalities, table, {'score': 'continuous', 'group': 'categorical'},
        tmp_path / 'streamed', **options)
    first = torch.load(cached / 'model.pt', weights_only=True)
    second = torch.load(streamed / 'model.pt', weights_only=True)
    for section in ('model', 'objective'):
        for key in first[section]:
            torch.testing.assert_close(first[section][key], second[section][key], rtol=0, atol=0)
    for filename in ('subj_course.npy', 'vbm_zstat.npy', 'fa_zstat.npy'):
        np.testing.assert_array_equal(np.load(cached / filename), np.load(streamed / filename))
    for filename in ('predictions.csv', 'history.csv', 'metrics.json'):
        assert (cached / filename).read_bytes() == (streamed / filename).read_bytes()
    cached_metadata = json.loads((cached / 'model.json').read_text())
    streamed_metadata = json.loads((streamed / 'model.json').read_text())
    assert cached_metadata['input_cache'] == 'cuda'
    assert streamed_metadata['input_cache'] == 'hdf5'
    assert cached_metadata['input_cache_gib'] > 0
    assert cached_metadata['peak_gpu_allocated_gib'] < options['max_gpu_gb']
