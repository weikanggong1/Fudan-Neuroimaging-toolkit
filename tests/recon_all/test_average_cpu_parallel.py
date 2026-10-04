"""Parallel averaging must preserve ordered float32 sums and caller threads."""

import numba
import pytest
import torch

from fnit.recon_all import mris_register_average_numba as averaging


def test_parallel_averaging_is_thread_independent_and_does_not_modify_input():
    size = 12000
    vertices = torch.arange(size)
    neighbors = torch.stack(((vertices + 1) % size, (vertices - 1) % size,
                             torch.full_like(vertices, -1)), dim=1)
    degrees = torch.full((size,), 2, dtype=torch.int64)
    generator = torch.Generator().manual_seed(319)
    gradient = torch.randn((size, 3), generator=generator)
    gradient[::7] *= 1e8
    original = gradient.clone()
    previous_torch, previous_numba = torch.get_num_threads(), numba.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected = averaging.average_gradients_exact_cpu(gradient, neighbors, degrees, 7)
        torch.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        actual = averaging.average_gradients_exact_cpu(gradient, neighbors, degrees, 7)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient, original, rtol=0, atol=0)
        assert numba.get_num_threads() == previous_numba
    finally:
        torch.set_num_threads(previous_torch)
        numba.set_num_threads(previous_numba)


@pytest.mark.parametrize("fails", [False, True])
def test_averaging_restores_numba_thread_mask_on_success_and_error(monkeypatch, fails):
    previous_torch, previous_numba = torch.get_num_threads(), numba.get_num_threads()
    observed = []

    def probe(gradient, neighbors, degrees, reciprocals, iterations):
        observed.append(numba.get_num_threads())
        if fails:
            raise RuntimeError("averaging kernel probe")
        return gradient.copy()

    monkeypatch.setattr(averaging, "_average_numpy", probe)
    size = 12000
    gradient = torch.zeros((size, 3))
    neighbors = torch.zeros((size, 1), dtype=torch.int64)
    degrees = torch.zeros(size, dtype=torch.int64)
    try:
        torch.set_num_threads(3)
        if fails:
            with pytest.raises(RuntimeError, match="averaging kernel probe"):
                averaging.average_gradients_exact_cpu(gradient, neighbors, degrees, 1)
        else:
            averaging.average_gradients_exact_cpu(gradient, neighbors, degrees, 1)
        assert observed == [min(3, numba.config.NUMBA_NUM_THREADS)]
        assert numba.get_num_threads() == previous_numba
    finally:
        torch.set_num_threads(previous_torch)
        numba.set_num_threads(previous_numba)
