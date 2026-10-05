"""CPU acceleration must retain the ordered tensor objective and thread budget."""

import numpy as np
import pytest
import torch

from fnit.mcflirt.core import FSLMotionNormCorr


@pytest.mark.parametrize("threads", [1, 2])
def test_compiled_cpu_matches_tensor_rows_and_preserves_budget(threads):
    from numba import get_num_threads

    rng = np.random.default_rng(47)
    moving = torch.from_numpy(rng.uniform(1, 5000, (17, 18, 19)).astype(np.float32))
    reference = torch.from_numpy(rng.uniform(1, 5000, (10, 11, 12)).astype(np.float32))
    cost = FSLMotionNormCorr(reference, moving, (2, 2, 2), (1, 1, 1), centre=np.zeros(3))
    accelerated = cost.cpu_rows
    previous = torch.get_num_threads()
    numba_previous = get_num_threads()
    try:
        torch.set_num_threads(threads)
        matrices = [np.eye(4)]
        for _ in range(12):
            matrix = np.eye(4)
            matrix[:3, :3] += rng.normal(0, .03, (3, 3))
            matrix[:3, 3] = rng.normal(0, 4, 3)
            matrices.append(matrix)
        # No-overlap rows and a reversed sampling direction exercise bounds.
        outside = np.eye(4); outside[0, 3] = 1000
        reverse = np.eye(4); reverse[0, 0] = -1; reverse[0, 3] = 10
        matrices.extend([outside, reverse])
        for matrix in matrices:
            cost.cpu_rows = None
            original = cost(matrix)
            cost.cpu_rows = accelerated
            assert cost(matrix) == original
            assert get_num_threads() == numba_previous
    finally:
        torch.set_num_threads(previous)


def test_other_precision_retains_tensor_path():
    reference = torch.ones((4, 4, 4), dtype=torch.float64)
    cost = FSLMotionNormCorr(reference, reference, (1, 1, 1), (1, 1, 1), centre=np.zeros(3))
    assert cost.cpu_rows is None

@pytest.mark.parametrize('case', ['gradient', 'nan', 'overflow'])
def test_cpu_objective_retains_tensor_fallback_for_unsupported_inputs(case):
    reference = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4) + 1
    moving = reference.clone()
    if case == 'gradient':
        moving.requires_grad_()
    elif case == 'nan':
        moving[0, 0, 0] = float('nan')
    else:
        moving *= 1e30
    cost = FSLMotionNormCorr(reference, moving, (1, 1, 1), (1, 1, 1), centre=np.zeros(3))
    assert cost.cpu_rows is None
    assert np.isfinite(cost(np.eye(4)))


def test_nonfinite_matrix_does_not_enter_compiled_cpu_sampler():
    data = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4) + 1
    cost = FSLMotionNormCorr(data, data, (1, 1, 1), (1, 1, 1), centre=np.zeros(3))
    def unexpected(*args):
        raise AssertionError('nonfinite coordinates entered CPU sampler')
    cost.cpu_rows = unexpected
    matrix = np.eye(4)
    matrix[0, 3] = np.nan
    # The retained tensor path raises on NaN gather indices. Preserve that
    # existing failure contract rather than turning invalid input into a cost.
    with pytest.raises(RuntimeError, match="out of bounds"):
        cost(matrix)
