"""Sparse-row compatibility regressions; these fixtures are not benchmarks."""

import numpy as np
import pytest
import torch
from sklearn.decomposition import sparse_encode

import fnit.dictionary_learning.torch_backend as dicl


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _observe_lars(monkeypatch):
    calls = []
    original = dicl._sparse_codes_lars_compatible

    def observed(samples, dictionary, alpha, max_events):
        calls.append(samples.detach().clone())
        return original(samples, dictionary, alpha, max_events)

    monkeypatch.setattr(dicl, "_sparse_codes_lars_compatible", observed)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("selection", ["partial", "all", "none"])
def test_near_node_restart_preserves_independent_codes(monkeypatch, dtype, selection):
    tolerance = 2 * np.finfo(np.float32).eps
    near = [[2., 1. + tolerance / 2], [2., 1. - tolerance / 2]]
    far = [[2., .5], [-2., -.5]]
    rows = near + far if selection == "partial" else near if selection == "all" else far
    samples = torch.tensor(rows, dtype=dtype, device="cuda")
    dictionary = torch.eye(2, dtype=dtype, device="cuda")
    original_samples, original_dictionary = samples.clone(), dictionary.clone()
    expected = sparse_encode(samples.cpu().numpy(), dictionary.cpu().numpy(),
                             algorithm="lasso_lars", alpha=1., max_iter=1000)
    calls = _observe_lars(monkeypatch)
    solver = dicl._SparseCodesBPDN(len(rows), 2, "cuda", dtype)
    solver.calls = 4

    actual = solver(samples, dictionary, 1., 1000)

    assert actual.shape == samples.shape
    assert actual.dtype == dtype and actual.device == samples.device
    np.testing.assert_allclose(actual.cpu().numpy(), expected,
                               atol=2e-7 if dtype == torch.float32 else 1e-14,
                               rtol=2e-7 if dtype == torch.float32 else 1e-12)
    torch.testing.assert_close(samples, original_samples, atol=0, rtol=0)
    torch.testing.assert_close(dictionary, original_dictionary, atol=0, rtol=0)
    assert solver.polish_checks > 0
    if selection == "none":
        assert not calls
        assert solver.fallback_count == solver.near_node_fallback_count == 0
        assert solver.fallback_rows == solver.near_node_fallback_rows == 0
        assert solver.admm_accepted_count == 1
    else:
        assert len(calls) == 1
        # The distant rows must not enter the compatibility solver.
        torch.testing.assert_close(calls[0], samples[:2], atol=0, rtol=0)
        assert solver.fallback_count == solver.near_node_fallback_count == 1
        assert solver.fallback_rows == solver.near_node_fallback_rows == 2
        assert solver.admm_accepted_count == 0


def test_first_four_batches_keep_whole_batch_reference(monkeypatch):
    tolerance = 2 * np.finfo(np.float32).eps
    samples = torch.tensor([[2., 1. + tolerance / 2], [2., .5], [-2., -.5]],
                           device="cuda", dtype=torch.float64)
    dictionary = torch.eye(2, device="cuda", dtype=torch.float64)
    expected = sparse_encode(samples.cpu().numpy(), np.eye(2),
                             algorithm="lasso_lars", alpha=1.)
    calls = _observe_lars(monkeypatch)
    solver = dicl._SparseCodesBPDN(3, 2, "cuda", torch.float64)

    for _ in range(4):
        np.testing.assert_allclose(solver(samples, dictionary, 1., 1000).cpu().numpy(),
                                   expected, atol=1e-14, rtol=1e-12)

    assert len(calls) == solver.calls == solver.fallback_count == 4
    assert solver.fallback_rows == 12
    assert solver.near_node_fallback_count == solver.near_node_fallback_rows == 0
    assert solver.admm_accepted_count == solver.polish_checks == 0
    for passed in calls:
        torch.testing.assert_close(passed, samples, atol=0, rtol=0)


def test_failed_polish_keeps_whole_batch_reference(monkeypatch):
    # Identical atoms make the polished active Gram singular. Preserve the
    # compatibility solver's finite vertex for every row in this failure.
    dictionary = torch.tensor([[1., 0.], [1., 0.], [0., 1.]],
                              device="cuda", dtype=torch.float64)
    samples = torch.tensor([[2., 0.], [0., 2.], [0., 0.]],
                           device="cuda", dtype=torch.float64)
    expected = sparse_encode(samples.cpu().numpy(), dictionary.cpu().numpy(),
                             algorithm="lasso_lars", alpha=1.)
    calls = _observe_lars(monkeypatch)
    solver = dicl._SparseCodesBPDN(3, 3, "cuda", torch.float64)
    solver.calls = 4

    actual = solver(samples, dictionary, 1., 1000)

    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-14, rtol=1e-12)
    assert len(calls) == solver.fallback_count == 1
    torch.testing.assert_close(calls[0], samples, atol=0, rtol=0)
    assert solver.fallback_rows == len(samples)
    assert solver.near_node_fallback_count == solver.near_node_fallback_rows == 0


def test_partial_restart_does_not_leak_into_next_batch(monkeypatch):
    tolerance = 2 * np.finfo(np.float32).eps
    samples = torch.tensor([[2., 1. + tolerance / 2], [2., .5]],
                           device="cuda", dtype=torch.float64)
    dictionary = torch.eye(2, device="cuda", dtype=torch.float64)
    calls = _observe_lars(monkeypatch)
    solver = dicl._SparseCodesBPDN(2, 2, "cuda", torch.float64)
    solver.calls = 4
    solver(samples, dictionary, 1., 1000)
    next_samples = torch.tensor([[-2., -.5], [2., .5]],
                                device="cuda", dtype=torch.float64)
    expected = torch.tensor([[-1., 0.], [1., 0.]],
                            device="cuda", dtype=torch.float64)

    torch.testing.assert_close(solver(next_samples, dictionary, 1., 1000), expected,
                               atol=1e-14, rtol=1e-12)
    assert len(calls) == solver.fallback_count == 1
    assert solver.fallback_rows == solver.near_node_fallback_rows == 1
    assert solver.admm_accepted_count == 1
