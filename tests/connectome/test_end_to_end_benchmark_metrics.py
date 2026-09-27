"""Keep final-matrix normalized MAE distinct from relative L1 error."""

import numpy as np
import pytest

from tools.benchmark_connectome_end_to_end import _matrix_metrics


def test_matrix_error_denominators_and_common_support():
    reference = np.array([[0, 10, 0], [10, 0, 2], [0, 2, 0]], dtype=float)
    candidate = np.array([[0, 8, 1], [8, 0, 2], [1, 2, 0]], dtype=float)
    result = _matrix_metrics(candidate, reference)
    assert result["mae_upper"] == pytest.approx(1.0)
    assert result["normalized_mae_upper"] == pytest.approx(1.0 / 6.0)
    assert result["relative_l1_upper"] == pytest.approx(3.0 / 12.0)
    assert result["common_support_edges"] == 2
    assert result["support_dice"] == pytest.approx(0.8)
    assert result["normalized_mae_common_support"] == pytest.approx(2.0 / 12.0)
