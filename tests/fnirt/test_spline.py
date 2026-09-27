import pytest
import torch

from fnit.fnirt.optimizer import preconditioned_conjugate_gradient
from fnit.fnirt.spline import (
    BendingOperator,
    adjoint_field,
    cubic_bspline_basis,
    expand_coefficients,
    fsl_control_shape,
    spline_bases,
    zoom_coefficients,
    zoom_matrix,
)


def test_fsl_cubic_control_count_includes_retired_boundary_rule():
    assert fsl_control_shape((91, 109, 91), (5, 5, 5)) == (21, 24, 21)
    assert fsl_control_shape((24, 28, 24), (5, 5, 5)) == (7, 8, 7)
    assert fsl_control_shape((7, 8, 9), (1, 1, 1)) == (7, 8, 9)


def test_cubic_basis_partition_and_derivatives():
    positions = torch.linspace(0, 30, 121, dtype=torch.float64)
    controls = 9
    basis = cubic_bspline_basis(positions, 5, controls)
    first = cubic_bspline_basis(
        positions, 5, controls, derivative=1, voxel_size=2.0
    )
    second = cubic_bspline_basis(
        positions, 5, controls, derivative=2, voxel_size=2.0
    )
    torch.testing.assert_close(basis.sum(1), torch.ones_like(positions))
    torch.testing.assert_close(first.sum(1), torch.zeros_like(positions), atol=1e-14, rtol=0)
    torch.testing.assert_close(second.sum(1), torch.zeros_like(positions), atol=1e-14, rtol=0)


def test_separable_forward_and_adjoint_are_transposes():
    shape = (9, 8, 7)
    spacing = (3, 3, 3)
    bases = spline_bases(
        shape,
        spacing,
        (2.0, 2.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(7)
    coefficients = torch.randn(
        (2, *fsl_control_shape(shape, spacing)),
        generator=generator,
        dtype=torch.float64,
    )
    image = torch.randn((2, *shape), generator=generator, dtype=torch.float64)
    left = (expand_coefficients(coefficients, bases) * image).sum()
    right = (coefficients * adjoint_field(image, bases)).sum()
    torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_bending_operator_has_exact_quadratic_gradient():
    shape = (8, 9, 7)
    spacing = (3, 3, 3)
    operator = BendingOperator(
        shape, spacing, (2.0, 2.5, 3.0), device="cpu", dtype=torch.float64
    )
    coefficients = torch.randn(
        (1, *fsl_control_shape(shape, spacing)), dtype=torch.float64
    )
    direction = torch.randn_like(coefficients)
    epsilon = 1e-5
    finite_difference = (
        operator.energy(coefficients + epsilon * direction)
        - operator.energy(coefficients - epsilon * direction)
    ) / (2 * epsilon)
    analytic = 2 * (operator.normal(coefficients) * direction).sum()
    torch.testing.assert_close(finite_difference, analytic, atol=2e-6, rtol=2e-6)


def test_zoom_matrix_is_fsl_separable_least_squares_projection():
    new_size = 25
    old_controls = fsl_control_shape((13, 13, 13), (5, 5, 5))[0]
    new_controls = fsl_control_shape((25, 25, 25), (5, 5, 5))[0]
    transform = zoom_matrix(
        new_size,
        5,
        new_controls,
        10,
        old_controls,
        device="cpu",
        dtype=torch.float64,
    )
    positions = torch.arange(new_size, dtype=torch.float64)
    old_basis = cubic_bspline_basis(positions, 10, old_controls)
    new_basis = cubic_bspline_basis(positions, 5, new_controls)
    expected = torch.linalg.pinv(new_basis.T @ new_basis) @ new_basis.T @ old_basis
    torch.testing.assert_close(transform, expected, atol=1e-12, rtol=1e-12)


def test_zoom_coefficients_keeps_refinement_error_below_fsl_numerical_scale():
    old_shape = (7, 8, 6)
    new_shape = (13, 15, 11)
    spacing = (3, 3, 3)
    generator = torch.Generator().manual_seed(19)
    coefficients = torch.randn(
        (2, *fsl_control_shape(old_shape, spacing)),
        generator=generator,
        dtype=torch.float64,
    )
    zoomed = zoom_coefficients(
        coefficients,
        new_shape,
        spacing,
        (4.0, 4.0, 4.0),
        (2.0, 2.0, 2.0),
    )
    old_on_new = tuple(
        cubic_bspline_basis(
            torch.arange(size, dtype=torch.float64),
            2 * axis_spacing,
            old_controls,
        )
        for size, axis_spacing, old_controls in zip(
            new_shape, spacing, coefficients.shape[1:]
        )
    )
    new_bases = spline_bases(
        new_shape,
        spacing,
        (2.0, 2.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    difference = (
        expand_coefficients(zoomed, new_bases)
        - expand_coefficients(coefficients, old_on_new)
    )
    assert float(difference.abs().max()) < 6e-4



def test_zoom_coefficients_changes_only_knot_spacing_for_inwarp_handoff():
    shape = (25, 25, 25)
    old_spacing = (5, 5, 5)
    new_spacing = (1, 1, 1)
    generator = torch.Generator().manual_seed(29)
    coefficients = torch.randn(
        (1, *fsl_control_shape(shape, old_spacing)),
        generator=generator,
        dtype=torch.float64,
    )

    actual = zoom_coefficients(
        coefficients,
        shape,
        new_spacing,
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        old_knot_spacing=old_spacing,
    )
    old_bases = spline_bases(
        shape,
        old_spacing,
        (1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float64,
    )
    new_bases = spline_bases(
        shape,
        new_spacing,
        (1.0, 1.0, 1.0),
        device="cpu",
        dtype=torch.float64,
    )

    assert actual.shape == (1, *fsl_control_shape(shape, new_spacing))
    torch.testing.assert_close(
        expand_coefficients(actual, new_bases),
        expand_coefficients(coefficients, old_bases),
        atol=2e-10,
        rtol=2e-10,
    )


def test_zoom_coefficients_rejects_simultaneous_voxel_and_knot_change():
    coefficients = torch.zeros(
        (1, *fsl_control_shape((13, 13, 13), (5, 5, 5))),
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="voxel size and knot spacing"):
        zoom_coefficients(
            coefficients,
            (25, 25, 25),
            (2, 2, 2),
            (2.0, 2.0, 2.0),
            (1.0, 1.0, 1.0),
            old_knot_spacing=(5, 5, 5),
        )


def test_pcg_solves_spd_system():
    matrix = torch.tensor(
        [[5.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    solution, report = preconditioned_conjugate_gradient(
        lambda value: matrix @ value,
        rhs,
        diagonal=matrix.diagonal(),
        tolerance=1e-12,
        max_iterations=10,
    )
    assert report.converged
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs))
    expected_residual = torch.linalg.vector_norm(matrix @ solution - rhs) / (
        torch.linalg.vector_norm(rhs)
    )
    assert abs(report.relative_residual - float(expected_residual)) < 1e-15
