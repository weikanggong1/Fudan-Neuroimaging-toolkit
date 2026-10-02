"""Differential checks for work reuse, not synthetic performance benchmarks."""

import numpy as np
import pytest
import torch

from fnit.amico_noddi import solver
from fnit.amico_noddi.classic import ClassicNODDIModel, fit_classic_noddi


@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("dependent", (False, True))
@pytest.mark.parametrize("l1", (0.0, 0.3))
def test_reused_gram_preserves_constrained_active_set(grouped, dependent, l1):
    rng = np.random.default_rng(20261002)
    design_shape = (2, 11, 5) if grouped else (11, 5)
    design = torch.tensor(rng.uniform(0.1, 1.0, design_shape), dtype=torch.float64)
    if dependent:
        design[..., 4] = design[..., 2]
    signal_shape = (2, 3, 11) if grouped else (3, 11)
    signal = torch.tensor(rng.uniform(0.05, 1.0, signal_shape), dtype=torch.float64)
    terms = (
        design.transpose(-2, -1) @ design,
        None,
    )
    snapshot = terms[0].clone()
    # Debias selects a subset; reusing products must not reuse the passive set.
    allowed = torch.ones((*signal.shape[:-1], 5), dtype=torch.bool)
    allowed[..., 1] = False
    allowed[..., -1, :] = False
    expected = solver.nonnegative_quadratic(design, signal, allowed=allowed, l1=l1)
    actual = solver.nonnegative_quadratic(
        design, signal, allowed=allowed, l1=l1, _quadratic_terms=terms
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    assert torch.equal(terms[0], snapshot)


def test_only_full_dictionary_stages_share_gram(monkeypatch):
    rng = np.random.default_rng(39)
    volumes, atoms = 9, 4
    signal = rng.uniform(0.1, 1.0, (5, volumes)).astype(np.float64)
    kernels = {
        "wm": rng.uniform(0.1, 0.9, (atoms, 3, volumes)).astype(np.float32),
        "iso": rng.uniform(0.01, 0.2, volumes).astype(np.float32),
        "norms": rng.uniform(0.7, 1.2, (volumes - 2, atoms)),
        "b0": np.arange(volumes) < 2,
        "icvf": np.linspace(0.1, 0.8, atoms, dtype=np.float32),
        "kappa": np.linspace(0.3, 2.0, atoms, dtype=np.float32),
    }
    rows = [np.array([0, 2, 4]), np.array([1, 3])]
    settings = dict(
        device="cpu", lambda1=0.5, lambda2=1e-3,
        kkt_tolerance=1e-11, cg_tolerance=1e-13, maximum_active_steps=40,
    )
    expected = solver._fit_lut_batch(
        signal, np.array([0, 2]), rows, kernels,
        _reuse_gram=False, **settings,
    )
    calls = []
    original = solver.nonnegative_quadratic

    def record_products(design, observed, **kwargs):
        terms = kwargs.get("_quadratic_terms")
        snapshot = terms[0].clone() if terms is not None else None
        result = original(design, observed, **kwargs)
        if terms is not None:
            assert torch.equal(terms[0], snapshot)
        calls.append((design.shape, observed.shape, terms))
        return result

    monkeypatch.setattr(solver, "nonnegative_quadratic", record_products)
    actual = solver._fit_lut_batch(signal, np.array([0, 2]), rows, kernels, **settings)
    for candidate, baseline in zip(actual, expected):
        np.testing.assert_array_equal(candidate, baseline)
    assert len(calls) == 3
    assert calls[0][2] is calls[2][2]
    assert calls[1][2] is None
    assert calls[0][0] == calls[2][0] == torch.Size((2, volumes, atoms + 1))
    assert calls[1][0] == torch.Size((2, volumes - 2, atoms))
    gram, linear = calls[0][2]
    assert gram.shape == (2, atoms + 1, atoms + 1)
    assert linear is None
    assert gram.dtype == torch.float64


@pytest.mark.parametrize("maximum_iterations", (0, 4))
def test_deferred_classic_count_is_exact_and_read_once(monkeypatch, maximum_iterations):
    rng = np.random.default_rng(20261003)
    bvals = np.r_[np.zeros(3), np.full(8, 1000), np.full(8, 2000)]
    bvecs = rng.normal(size=(len(bvals), 3))
    bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[:3] = (1, 0, 0)
    model = ClassicNODDIModel(bvals, bvecs, d_par=1.7e-3, d_iso=3e-3, device="cpu")
    parameters = torch.tensor(
        [[0.4, 0.3, 0.1, 1.0], [0.6, 0.2, 0.05, 1.0], [0.2, 0.5, 0.4, 1.0]],
        dtype=torch.float64,
    )
    directions = torch.tensor([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
    signal = model.evaluate(parameters, directions).numpy()
    estimates = parameters[:, :3].numpy().copy()
    estimates[:, 0] += 0.04
    arguments = (signal, bvals, bvecs, bvals == 0, estimates, directions.numpy())
    options = dict(
        d_par=1.7e-3, d_iso=3e-3, device="cpu", batch_size=2,
        maximum_iterations=maximum_iterations,
    )
    reads = []
    original_item = torch.Tensor.item

    def record_item(value, *args, **kwargs):
        if value.dtype == torch.int64 and value.ndim == 0:
            reads.append(None)
        return original_item(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", record_item)
    expected = fit_classic_noddi(*arguments, _defer_qc_count=False, **options)
    assert len(reads) == 2 * maximum_iterations
    reads.clear()
    actual = fit_classic_noddi(*arguments, **options)
    assert len(reads) == 1
    for candidate, baseline in zip(actual[:3], expected[:3]):
        np.testing.assert_array_equal(candidate, baseline)
    assert actual[3] == expected[3]
    assert isinstance(actual[3]["accepted_updates"], int)
