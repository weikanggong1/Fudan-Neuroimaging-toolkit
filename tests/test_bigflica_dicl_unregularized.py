"""Analytical sparse-code regressions; these fixtures are not benchmarks."""

import h5py
import numpy as np
import pytest
import torch
from sklearn.decomposition import sparse_encode

import fnit.bigflica.dicl_torch as dicl


@pytest.mark.parametrize("solver_kind", ["full", "incremental"])
def test_small_active_gram_has_no_ridge_bias(solver_kind):
    # Orthogonal atoms have an independent closed-form LASSO solution.
    # A 1e-10 ridge changes the weak atom's code by about one percent.
    dictionary = torch.diag(torch.tensor([1e-4, 1.], dtype=torch.float64))
    samples = torch.tensor([[.02, 2.], [-.02, -2.], [0., 0.]], dtype=torch.float64)
    alpha = 1e-6
    response = samples @ dictionary.T
    expected = response.sign() * (response.abs() - alpha).clamp_min(0)
    expected /= dictionary.square().sum(dim=1)
    if solver_kind == "full":
        actual = dicl._sparse_codes_lars(samples, dictionary, alpha, 1000)
    else:
        solver = dicl._LarsInverseSolver(3, 2, "cpu", torch.float64, alpha)
        actual = solver(samples, dictionary, alpha, 1000)
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-10)
    residual = actual @ dictionary - samples
    gradient = residual @ dictionary.T
    active = expected != 0
    torch.testing.assert_close(gradient[active], -alpha * expected.sign()[active],
                               atol=1e-13, rtol=1e-8)


def test_zero_dictionary_finished_rows_do_not_divide_by_zero():
    solver = dicl._LarsInverseSolver(3, 4, "cpu", torch.float64)
    actual = solver(torch.zeros((3, 2), dtype=torch.float64),
                    torch.zeros((4, 2), dtype=torch.float64), 1., 1000)
    assert torch.isfinite(solver.inverse).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    assert solver.fallback_count == 0


def test_nonunique_atoms_select_a_finite_lars_vertex_without_ridge():
    dictionary = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
    samples = torch.tensor([[2., 0.]], dtype=torch.float64)
    expected = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    actual = dicl._sparse_codes_lars(samples, dictionary, 1., 1000)
    incremental = dicl._LarsInverseSolver(1, 3, "cpu", torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(incremental(samples, dictionary, 1., 1000), expected,
                               atol=1e-14, rtol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_admm_polish_has_no_ridge_bias():
    # The weak active pivot passes the existing conditioning guard. The old
    # ridge produced a 1e-4 relative coefficient bias while still passing KKT.
    dictionary = torch.diag(torch.tensor([1e-3, 1.], device="cuda", dtype=torch.float64))
    samples = torch.tensor([[.011, 2.]], device="cuda", dtype=torch.float64)
    alpha = 1e-6
    response = samples @ dictionary.T
    expected = response.sign() * (response.abs() - alpha).clamp_min(0)
    expected /= dictionary.square().sum(dim=1)
    solver = dicl._SparseCodesBPDN(1, 2, "cuda", torch.float64, alpha,
                                 compatibility_mode=False)
    solver.calls = 4
    actual = solver(samples, dictionary, alpha, 1000)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert solver.polish_checks > 0
    assert solver.fallback_count == 0


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_compatible_lars_stops_at_nodes_and_interpolates_other_segments(backend):
    # A node just above/below alpha must be retained; a distant node must
    # interpolate to alpha. Raising the global penalty by epsilon fails this.
    tolerance = 2 * np.finfo(np.float32).eps
    samples = np.array([[2., 1. + tolerance / 2],
                        [2., 1. - tolerance / 2],
                        [2., .5],
                        [1. + tolerance / 2, 0.],
                        [1. + 2 * tolerance, 0.],
                        [0., 0.]])
    expected = np.array([[1. - tolerance / 2, 0.],
                         [1. + tolerance / 2, 0.],
                         [1., 0.],
                         [0., 0.],
                         [2 * tolerance, 0.],
                         [0., 0.]])
    np.testing.assert_allclose(sparse_encode(samples, np.eye(2),
                              algorithm="lasso_lars", alpha=1.), expected,
                              atol=1e-14, rtol=1e-12)
    actual = dicl._sparse_codes_lars_compatible(
        torch.as_tensor(samples, device=backend),
        torch.eye(2, device=backend, dtype=torch.float64), 1., 1000)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"))])
def test_compatible_lars_matches_sklearn_overcomplete_drop_paths(backend):
    generator = np.random.default_rng(943)
    samples = generator.normal(size=(16, 8))
    samples[::4] = 0
    dictionary = generator.normal(size=(24, 8))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = sparse_encode(samples, dictionary, algorithm="lasso_lars", alpha=.4,
                             max_iter=1000)
    actual = dicl._sparse_codes_lars_compatible(
        torch.as_tensor(samples, device=backend),
        torch.as_tensor(dictionary, device=backend), .4, 1000)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-10, rtol=1e-9)


def test_nearby_knot_guard_checks_both_directions_and_tied_bound_without_mutation():
    alpha, tolerance = 1., 500 * np.finfo(np.float32).eps
    gap = tolerance / 2
    gram = torch.tensor([[1., .2], [.2, 1.]], dtype=torch.float64)
    code = torch.tensor([[gap, 0.], [.5, 0.], [.5, 0.]], dtype=torch.float64)
    response = torch.tensor([[1. + gap, 0.],
                             [1.5, 1.1 - .8 * gap],
                             [1.5, .1]], dtype=torch.float64)
    direction = torch.tensor([[1., 0.]] * 3, dtype=torch.float64)
    initial = code.clone()
    result = dicl._nearby_lars_knots(gram, response, code, direction, alpha, tolerance)
    torch.testing.assert_close(result["near_above"], torch.tensor([True, False, False]))
    torch.testing.assert_close(result["near_below"], torch.tensor([False, True, False]))
    torch.testing.assert_close(result["sensitive_rows"], torch.tensor([True, True, False]))
    torch.testing.assert_close(code, initial, atol=0, rtol=0)
    tied = dicl._nearby_lars_knots(torch.ones((2, 2), dtype=torch.float64),
        torch.tensor([[1.5, 1.5]], dtype=torch.float64),
        torch.tensor([[.5, 0.]], dtype=torch.float64), direction[:1], alpha, tolerance)
    assert bool(tied["boundary_tie"][0]) and bool(tied["sensitive_rows"][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_admm_near_nodes_fall_back_to_cpu_compatible_codes():
    tolerance = 2 * np.finfo(np.float32).eps
    # The third row is far from either node: any sensitive row must restart
    # the whole batch, rather than requiring every row to be sensitive.
    samples = np.array([[2., 1. + tolerance / 2],
                        [2., 1. - tolerance / 2], [2., .5]])
    expected = sparse_encode(samples, np.eye(2), algorithm="lasso_lars", alpha=1.)
    solver = dicl._SparseCodesBPDN(3, 2, "cuda", torch.float64)
    solver.calls = 4
    actual = solver(torch.as_tensor(samples, device="cuda"),
                    torch.eye(2, device="cuda", dtype=torch.float64), 1., 1000)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-14, rtol=1e-12)
    assert solver.polish_checks > 0
    assert solver.near_node_fallback_count == solver.fallback_count == 1
    assert solver.admm_accepted_count == 0
    assert solver.fallback_count + solver.admm_accepted_count == solver.calls - 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_admm_away_from_nodes_keeps_fast_path_and_all_lars_remains_available():
    samples = torch.tensor([[2., .5]], device="cuda", dtype=torch.float64)
    dictionary = torch.eye(2, device="cuda", dtype=torch.float64)
    solver = dicl._SparseCodesBPDN(1, 2, "cuda", torch.float64)
    solver.calls = 4
    actual = solver(samples, dictionary, 1., 1000)
    torch.testing.assert_close(actual, torch.tensor([[1., 0.]], device="cuda", dtype=torch.float64),
                               atol=1e-14, rtol=1e-12)
    assert solver.admm_accepted_count == 1
    assert solver.fallback_count == solver.near_node_fallback_count == 0
    reference = dicl._SparseCodesBPDN(1, 2, "cuda", torch.float64,
                                     compatibility_mode=True)
    torch.testing.assert_close(reference(samples, dictionary, 1., 1000), actual,
                               atol=1e-14, rtol=1e-12)
    assert reference.admm_accepted_count == reference.polish_checks == 0
    assert reference.fallback_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multimodal_fit_matches_independent_seeded_fits(tmp_path, monkeypatch):
    # A regression for modality ordering and reused solver counters, not a
    # speed or accuracy benchmark on generated imaging data.
    names = ("vbm", "fa", "md")
    generator = np.random.default_rng(823)
    for name in names:
        with h5py.File(tmp_path / f"{name}_projected.h5", "w") as handle:
            handle.create_dataset("data", data=generator.normal(size=(64, 12)))
    original = dicl._SparseCodesBPDN
    observed = []

    class Observer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self)

    monkeypatch.setattr(dicl, "_SparseCodesBPDN", Observer)
    options = dict(device="cuda:0", max_iter=2, batch_size=16,
                   sparse_iterations=1000, random_state=0, feature_block=32)
    together = dicl.fit_dicl_gpu_streaming(tmp_path, names, 6, **options)
    together_solvers = tuple(observed)
    observed.clear()
    separate = {name: dicl.fit_dicl_gpu_streaming(tmp_path, [name], 6, **options)[name]
                for name in names}
    assert len(together_solvers) == len(observed) == len(names)
    for name, combined_solver, single_solver in zip(names, together_solvers, observed):
        np.testing.assert_allclose(together[name], separate[name], atol=1e-12, rtol=1e-12)
        assert combined_solver.calls == single_solver.calls
        assert combined_solver.fallback_count == single_solver.fallback_count
        assert combined_solver.fallback_count >= 4
