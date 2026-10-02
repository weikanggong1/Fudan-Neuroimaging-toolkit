"""Small CPU math regressions; these are not accuracy/performance benchmarks."""
import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fnirt.spline import fsl_control_shape
from fnit.topup.core import (
    _TOPUPLevel, _average_pool, _cubic_spline_coefficients, _fsl_frame_mask,
    _gaussian_blur, _grid, _levenberg_marquardt, _regrid_images, _render_fixed_topup,
    _sample_cubic_with_derivatives, _transfer_field,
)
from fnit.topup.io import make_topup_coefficient_image


def make_problem(*, regrid=False):
    torch.manual_seed(12)
    shape, spacing = (8, 8, 8), (3, 3, 3)
    images = 40 + 10 * torch.rand((2, *shape))
    acquisition = np.asarray(((0, -1, 0, .08), (0, 1, 0, .08)))
    source_voxels = (2, 2, 2)
    if regrid:
        images, source_voxels = _regrid_images(images, source_voxels, 2, 1)
    return _TOPUPLevel(_cubic_spline_coefficients(images), shape, spacing, (2, 2, 2),
                       acquisition, 1, 1, .001, torch.zeros((2, 6), dtype=torch.float64),
                       sampling_voxel_sizes=source_voxels)


def test_frame_preserves_pe_faces_and_excludes_other_faces():
    valid = torch.ones((7, 8, 9), dtype=torch.bool)
    mask = _fsl_frame_mask(valid, 1)
    assert bool(mask[1:-1, :, 1:-1].all())
    assert not bool(mask[(0, -1)].any())
    assert not bool(mask[:, :, (0, -1)].any())
    assert bool(valid.all())


def test_periodic_smoothing_preserves_constant_and_wraps_impulse():
    constant = torch.full((2, 16, 18, 20), 13.0)
    torch.testing.assert_close(_gaussian_blur(constant, 4, (2, 2, 2)), constant,
                               atol=3e-6, rtol=1e-6)
    impulse = torch.zeros((1, 16, 18, 20))
    impulse[0, 0, 9, 10] = 1
    blurred = _gaussian_blur(impulse, 4, (2, 2, 2))
    torch.testing.assert_close(blurred[0, 1, 9, 10], blurred[0, -1, 9, 10])
    assert float(blurred[0, -1, 9, 10]) > 0


def test_block_average_preserves_subsample_voxel_centres():
    grid = _grid((8, 10, 12), device="cpu", dtype=torch.float32)
    reduced = _average_pool(grid, 2)
    expected = _grid((4, 5, 6), device="cpu", dtype=torch.float32) * 2 + .5
    torch.testing.assert_close(reduced, expected, atol=0, rtol=0)


def test_official_sampler_derivatives_and_negative_periodic_coordinates():
    torch.manual_seed(3)
    coefficients = torch.rand((7, 8, 9))
    coordinates = torch.tensor(((1.31, 2.21, 4.19), (-1.23, -.24, 7.23),
                                (2.11, 3.23, 5.21)))
    sampled, gradient, valid = _sample_cubic_with_derivatives(coefficients, coordinates, 1)
    assert bool(valid.all())
    for axis in range(3):
        plus, minus = coordinates.clone(), coordinates.clone()
        plus[axis] += .001
        minus[axis] -= .001
        numeric = (_sample_cubic_with_derivatives(coefficients, plus, 1)[0] -
                   _sample_cubic_with_derivatives(coefficients, minus, 1)[0]) / .002
        torch.testing.assert_close(gradient[axis], numeric, atol=3e-5, rtol=1e-3)
    # FSL's trunc(coord+.5) start has a missing left support tap at -1.23;
    # this regression deliberately records the official periodic convention.
    ones = torch.ones_like(coefficients)
    value = _sample_cubic_with_derivatives(ones, coordinates, 1)[0]
    assert float(value[0]) < .999
    assert float(value[1]) == pytest.approx(1.0)
    assert sampled.dtype == gradient.dtype == torch.float32


@pytest.mark.parametrize("regrid", [False, True])
def test_analytic_gradient_with_frozen_ssqlambda(regrid):
    problem = make_problem(regrid=regrid)
    coefficients = torch.linspace(-.1, .1, problem.coefficient_count, dtype=torch.float64).reshape(problem.control_shape)
    parameters = problem.parameters(coefficients, False)
    problem.cost(parameters)
    weight = problem.regularization_weight()
    analytical = problem.gradient(parameters)
    for index in (30, 61, 100):
        eps = .05
        plus, minus = parameters.clone(), parameters.clone()
        plus[index] += eps
        minus[index] -= eps
        def fixed_cost(p):
            state = problem.state(p)
            return state["ssd"] + weight * problem.energy(state["coefficients"])
        numerical = (fixed_cost(plus) - fixed_cost(minus)) / (2 * eps)
        torch.testing.assert_close(analytical[index], numerical, atol=8e-5, rtol=2e-3)


def test_joint_gn_symmetry_psd_and_diagonal():
    problem = make_problem()
    parameters = problem.parameters(torch.zeros(problem.control_shape, dtype=torch.float64), True)
    problem.cost(parameters)
    matvec, diagonal = problem.hessian(parameters)
    torch.manual_seed(8)
    left, right = torch.randn_like(parameters), torch.randn_like(parameters)
    torch.testing.assert_close(torch.dot(left, matvec(right)), torch.dot(matvec(left), right),
                               atol=1e-8, rtol=1e-10)
    assert float(torch.dot(left, matvec(left))) >= 0
    for index in (40, problem.coefficient_count, parameters.numel() - 1):
        basis = torch.zeros_like(parameters)
        basis[index] = 1
        torch.testing.assert_close(matvec(basis)[index], diagonal[index], atol=1e-9, rtol=1e-10)


def test_lm_accepted_step_budget_and_joint_quadratic():
    class Quadratic:
        def cost(self, p):
            return ((p - 3) ** 2).sum()
        def gradient(self, p):
            return 2 * (p - 3)
        def hessian(self, p):
            return lambda v: 2 * v, torch.full_like(p, 2)
    result, report = _levenberg_marquardt(Quadratic(), torch.zeros(7, dtype=torch.float64), max_iterations=2)
    assert report["accepted_iterations"] == report["attempts"] == 2
    assert float((result - 3).abs().max()) < .003
    assert not report["converged"]


def test_fixed_coeff_render_frame_and_geometry_gate(tmp_path):
    shape, spacing = (18, 20, 22), (3, 3, 3)
    raw = np.full((*shape, 2), 42, dtype=np.float32)
    imain, acqp, coef, movpar = (tmp_path / name for name in ("pair.nii.gz", "acqp.txt", "coef.nii.gz", "movpar.txt"))
    nib.save(nib.Nifti1Image(raw, np.diag((-2, 2, 2, 1))), imain)
    np.savetxt(acqp, ((0, -1, .0, .08), (0, 1, .0, .08)))
    nib.save(make_topup_coefficient_image(np.zeros(fsl_control_shape(shape, spacing)), shape, (2, 2, 2), spacing), coef)
    np.savetxt(movpar, np.zeros((2, 6)))
    rendered = _render_fixed_topup(imain, acqp, coef, movpar)
    np.testing.assert_allclose(rendered["corrected"][1:-1, :, 1:-1], 42, atol=1e-5)
    assert not rendered["corrected"][[0, -1]].any()
    assert not rendered["corrected"][:, :, [0, -1]].any()
    assert np.all(rendered["jacobians"] == 1)
    assert rendered["valid_voxels"] == (shape[0] - 2) * shape[1] * (shape[2] - 2)
    assert rendered["ssd"] == 0
    bad = make_topup_coefficient_image(np.zeros(fsl_control_shape(shape, spacing)), shape, (3, 2, 2), spacing)
    nib.save(bad, coef)
    with pytest.raises(ValueError, match="identical field geometry"):
        _render_fixed_topup(imain, acqp, coef, movpar)


def test_regrid_source_geometry_keeps_original_field_target():
    problem = make_problem(regrid=True)
    parameters = problem.parameters(torch.zeros(problem.control_shape, dtype=torch.float64), False)
    state = problem.state(parameters)
    assert problem.source_shape == (10, 10, 10)
    assert state["field"].shape == state["mask"].shape == (8, 8, 8)
    assert len(state["coordinates"]) == 2
    expected_source_voxel = float(np.float32((14-1e-6)/10))
    assert problem.source_voxel_sizes == (expected_source_voxel,) * 3
    # A zero-motion target voxel at 6 is not source voxel 6 after regridding.
    expected = np.float32(np.float32(12) * np.float32(1/expected_source_voxel))
    assert float(state["coordinates"][0][0,6,3,3]) == float(expected)
    assert float(state["coordinates"][0][0,6,3,3]) != 6


def test_fixed_renderer_restores_neurological_storage_and_keeps_fsl_coefficients(tmp_path):
    shape, spacing = (18, 20, 22), (3, 3, 3)
    grid = np.indices(shape).astype(np.float32)
    raw = np.stack((40 + grid[0] + 2 * grid[1], 60 + 2 * grid[0] + grid[2]), axis=3)
    radiological, neurological = tmp_path / 'radio.nii.gz', tmp_path / 'neuro.nii.gz'
    nib.save(nib.Nifti1Image(raw, np.diag((-2, 2, 2, 1))), radiological)
    nib.save(nib.Nifti1Image(raw[::-1].copy(), np.diag((2, 2, 2, 1))), neurological)
    acqp, coef, movpar = (tmp_path / name for name in ('acqp.txt', 'coef.nii.gz', 'movpar.txt'))
    # The acquisition file is FSL's canonical contract even for neurological
    # NIfTI storage; it is not a BIDS storage-axis vector transformed here.
    np.savetxt(acqp, ((-1, 0, 0, .08), (1, 0, 0, .08)))
    coefficients = np.full(fsl_control_shape(shape, spacing), .2, dtype=np.float32)
    nib.save(make_topup_coefficient_image(coefficients, shape, (2, 2, 2), spacing), coef)
    np.savetxt(movpar, np.zeros((2, 6)))
    radio = _render_fixed_topup(radiological, acqp, coef, movpar)
    neuro = _render_fixed_topup(neurological, acqp, coef, movpar)
    assert not radio['canonical_x_flip'] and neuro['canonical_x_flip']
    np.testing.assert_array_equal(neuro['field_hz'][::-1], radio['field_hz'])
    np.testing.assert_array_equal(neuro['corrected'][::-1], radio['corrected'])
    np.testing.assert_array_equal(neuro['jacobians'], radio['jacobians'])
    np.testing.assert_array_equal(neuro['common_mask'][::-1], radio['common_mask'])
