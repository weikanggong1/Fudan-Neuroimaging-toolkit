import pytest
import torch

from fnit.fnirt.optimizer import (
    preconditioned_conjugate_gradient,
    scaled_conjugate_gradient,
)


def test_pcg_matches_imlpp_iteration_and_step_oracle():
    matrix = torch.tensor(
        [
            [5.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 4.0, 0.5, 0.1, 0.0, 0.0],
            [0.2, 0.5, 3.0, 0.4, 0.2, 0.0],
            [0.0, 0.1, 0.4, 2.5, 0.6, 0.1],
            [0.0, 0.0, 0.2, 0.6, 2.0, 0.7],
            [0.0, 0.0, 0.0, 0.1, 0.7, 1.8],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype=torch.float64
    )
    expected = torch.tensor(
        [
            0.3212305059173806,
            -0.6372311124824229,
            0.15372274822013607,
            1.504279645884813,
            -1.5353160436077122,
            1.6245466424440114,
        ],
        dtype=torch.float64,
    )

    actual, report = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        diagonal=torch.diagonal(matrix),
        tolerance=1e-3,
        max_iterations=500,
    )

    assert report.iterations == 5
    assert report.converged
    assert abs(report.relative_residual - 0.0003584912872494894) < 1e-15
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)


def test_scg_matches_fsl_first_iteration_on_quadratic():
    matrix = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
    rhs = torch.tensor([2.0, -10.0], dtype=torch.float64)

    def cost(value):
        return 0.5 * torch.dot(value, matrix @ value) - torch.dot(rhs, value)

    def gradient(value):
        return matrix @ value - rhs

    actual, report = scaled_conjugate_gradient(
        cost,
        gradient,
        torch.zeros(2, dtype=torch.float64),
        max_iterations=1,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [0.4012345679012346, -2.006172839506173],
            dtype=torch.float64,
        ),
        atol=1e-14,
        rtol=1e-14,
    )
    assert report.iterations == 1
    assert report.accepted_iterations == 1
    assert report.cost == pytest.approx(-10.641384697454654)
    assert report.lambda_final == pytest.approx(0.05)
