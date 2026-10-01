"""Dictionary reference regressions, separated from BigFLICA integration."""

import h5py
import numpy as np
import pytest
import torch
from sklearn.decomposition import sparse_encode
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.utils.extmath import randomized_svd

import fnit.dictionary_learning.torch_backend as dicl_torch_module
from fnit.dictionary_learning.torch_backend import (
    _DictionaryUpdater, _LarsInverseSolver, _SparseCodesBPDN,
    _randomized_svd_dictionary, _sparse_codes_lars,
)


def test_lars_codes_match_sklearn_objective():
    generator = np.random.default_rng(7)
    samples = generator.normal(size=(16, 6))
    dictionary = generator.normal(size=(9, 6))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.4)
    actual = _sparse_codes_lars(torch.as_tensor(samples),
                               torch.as_tensor(dictionary), 0.4).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_lars_mixed_finished_rows_match_sklearn():
    generator = np.random.default_rng(17)
    samples = generator.normal(size=(32, 10))
    samples[::4] = 0
    dictionary = generator.normal(size=(40, 10))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.8)
    actual = _sparse_codes_lars(torch.as_tensor(samples),
                               torch.as_tensor(dictionary), 0.8).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bpdn_polish_matches_sklearn_and_reuses_workspaces():
    generator = np.random.default_rng(73)
    dictionary = generator.normal(size=(40, 20))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    solver = _SparseCodesBPDN(32, 40, "cuda", torch.float64, alpha=0.4,
                             compatibility_mode=False)
    dictionary_gpu = torch.as_tensor(dictionary, device="cuda")
    for index in range(7):
        samples = generator.normal(size=(32, 20)) if index < 6 else np.zeros((32, 20))
        samples[::4] = 0
        expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.4)
        actual = solver(torch.as_tensor(samples, device="cuda"), dictionary_gpu, 0.4, 120)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-7, rtol=1e-7)
    assert solver.polish_checks > 0
    assert solver.fallback_count == 4
    with pytest.raises(ValueError, match="alpha differs"):
        solver(torch.zeros((32, 20), device="cuda"), dictionary_gpu, 0.8, 120)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bpdn_nonunique_support_restarts_lars():
    # ADMM can split coefficients across identical atoms. KKT alone accepts
    # that solution, whereas sklearn LARS selects a single atom.
    dictionary = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], device="cuda")
    samples = torch.tensor([[2., 0.]], device="cuda")
    dictionary = dictionary.double(); samples = samples.double()
    solver = _SparseCodesBPDN(1, 3, "cuda", torch.float64,
                             compatibility_mode=False)
    solver.calls = 4
    actual = solver(samples, dictionary, 1., 120)
    expected = _sparse_codes_lars(samples, dictionary, 1., 120)
    torch.testing.assert_close(actual, expected)
    assert solver.fallback_count == 1


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_incremental_lars_workspace_reuse_matches_sklearn(backend):
    generator = np.random.default_rng(17)
    dictionary = generator.normal(size=(40, 10))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    solver = _LarsInverseSolver(32, 40, backend, torch.float64, alpha=0.8)
    for scale in (1., 0.5, 0.):
        samples = scale * generator.normal(size=(32, 10))
        samples[::4] = 0
        expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=0.8)
        actual = solver(torch.as_tensor(samples, device=backend),
                        torch.as_tensor(dictionary, device=backend), 0.8).cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert solver.fallback_count == 0


def test_incremental_lars_bad_inverse_restarts_original_solver(monkeypatch):
    generator = np.random.default_rng(7)
    samples = torch.as_tensor(generator.normal(size=(16, 6)))
    dictionary = torch.as_tensor(generator.normal(size=(9, 6)))
    dictionary /= torch.linalg.vector_norm(dictionary, dim=1)[:, None]
    expected = _sparse_codes_lars(samples, dictionary, 0.4)
    original_event = dicl_torch_module._lars_inverse_event

    def failed_event(*args, **kwargs):
        original_event(*args, **kwargs)
        args[7].fill_(True)

    solver = _LarsInverseSolver(16, 9, "cpu", torch.float64, alpha=0.4)
    with monkeypatch.context() as context:
        context.setattr(dicl_torch_module, "_lars_inverse_event", failed_event)
        torch.testing.assert_close(solver(samples, dictionary, 0.4), expected)
    assert solver.fallback_count == 1
    # The next batch must clear the previous failure and active-set state.
    torch.testing.assert_close(solver(samples * 0, dictionary, 0.4), torch.zeros_like(expected))
    assert solver.fallback_count == 1


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_dictionary_updates_match_sklearn_with_dead_atoms(backend):
    generator = np.random.default_rng(25)
    samples = generator.normal(size=(32, 10))
    dictionary = generator.normal(size=(40, 10))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    updater = _DictionaryUpdater(40, 10, backend, torch.float64)
    for dead_atoms in ((1, 7, 25), ()):
        codes = generator.normal(size=(32, 40))
        codes[:, dead_atoms] = 0
        a, b = codes.T @ codes, samples.T @ codes
        expected = dictionary.copy()
        expected_rng = np.random.RandomState(0)
        _update_dict(expected, samples, codes.copy(), A=a.copy(), B=b.copy(), random_state=expected_rng)
        actual = torch.as_tensor(dictionary.copy(), device=backend)
        actual_rng = np.random.RandomState(0)
        updater(actual, torch.as_tensor(a, device=backend), torch.as_tensor(b, device=backend),
                torch.as_tensor(samples, device=backend), actual_rng)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(actual_rng.get_state()[1], expected_rng.get_state()[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_dicl_randomized_svd_initialization_matches_sklearn(tmp_path):
    generator = np.random.default_rng(23)
    projected = generator.normal(size=(128, 48))
    mean = projected.mean(axis=0)
    std = projected.std(axis=0)
    standardized = (projected - mean) / std
    _, singular_values, right = randomized_svd(
        standardized, n_components=12, n_oversamples=10, n_iter=4,
        random_state=0, transpose=False, flip_sign=True)
    expected = singular_values[:, None] * right
    with h5py.File(tmp_path / "projected.h5", "w") as file:
        data = file.create_dataset("data", data=projected)
        actual = _randomized_svd_dictionary(
            data, torch.as_tensor(standardized, device="cuda", dtype=torch.float64),
            torch.as_tensor(mean, device="cuda", dtype=torch.float64),
            torch.as_tensor(std, device="cuda", dtype=torch.float64), 12,
            np.random.RandomState(0), 64).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
