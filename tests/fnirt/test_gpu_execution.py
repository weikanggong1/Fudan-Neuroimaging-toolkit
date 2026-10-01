"""Execution changes must retain smoothing bits and the strict PCG trace."""

import sys

import pytest
import torch

from fnit.fnirt.optimizer import PCGReport, preconditioned_conjugate_gradient
from fnit.fnirt.registration import (
    _fsl_displacement_coordinates,
    _fsl_gaussian_blur,
    _fsl_gaussian_blur_reference,
    _fsl_masked_gaussian_blur,
)
from fnit.fnirt.spline import BendingOperator, fsl_control_shape


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@CUDA
@pytest.mark.parametrize("shape", [(1, 1, 3, 4, 5), (2, 3, 17, 19, 23)])
@pytest.mark.parametrize("fwhm,zooms", [(8.0, (1.0, 2.0, 3.0)), (2.0, (2.0, 2.0, 2.0)), (1e-7, (1.0, 1.0, 1.0))])
def test_fused_gaussian_keeps_each_offset_assignment_bits(shape, fwhm, zooms):
    pytest.importorskip("triton")
    volume = torch.randn(shape, generator=torch.Generator().manual_seed(43)).cuda()
    volume.flatten()[:6] = torch.tensor([0.0, -0.0, 1.4e-45, -1.4e-45, 1e-35, -1e-35], device="cuda")
    expected = _fsl_gaussian_blur_reference(volume, fwhm, zooms)
    actual = _fsl_gaussian_blur(volume, fwhm, zooms)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@CUDA
def test_masked_fused_smoothing_keeps_normalisation_and_outside_zeros():
    volume = torch.randn((1, 1, 13, 15, 11), generator=torch.Generator().manual_seed(9)).cuda()
    mask = torch.zeros(volume.shape[2:], dtype=torch.bool, device="cuda")
    mask[1:11, 2:12, 3:10] = True
    mask_image = mask.to(volume.dtype)[None, None]
    expected = torch.where(
        mask_image > 0,
        _fsl_gaussian_blur_reference(volume * mask_image, 4.5, (1, 2, 1.5))
        / _fsl_gaussian_blur_reference(mask_image, 4.5, (1, 2, 1.5)).clamp_min(torch.finfo(volume.dtype).tiny),
        0,
    )
    actual = _fsl_masked_gaussian_blur(volume, 4.5, (1, 2, 1.5), mask)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@CUDA
def test_cuda_gaussian_without_triton_uses_reference(monkeypatch):
    volume = torch.arange(60.0, device="cuda").reshape(1, 1, 3, 4, 5)
    expected = _fsl_gaussian_blur_reference(volume, 3, (1, 1, 1))
    monkeypatch.setitem(sys.modules, "fnit.fnirt._smoothing_triton", None)
    actual = _fsl_gaussian_blur(volume, 3, (1, 1, 1))
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_zero_width_gaussian_keeps_alias_and_cpu_reference_bits():
    volume = torch.arange(60.0).reshape(1, 1, 3, 4, 5)
    assert _fsl_gaussian_blur(volume, 0, (1, 1, 1)) is volume
    assert torch.equal(_fsl_gaussian_blur(volume, 3, (1, 1, 1)), _fsl_gaussian_blur_reference(volume, 3, (1, 1, 1)))


def _serial_pcg(matvec, rhs, diagonal, tolerance, maximum):
    """Frozen pre-optimisation path, including its early denominator return."""
    floor = torch.finfo(rhs.dtype).eps * diagonal.abs().mean().clamp_min(1)
    inverse = diagonal.clamp_min(floor).reciprocal()
    solution, residual = torch.zeros_like(rhs), rhs.clone()
    rhs_norm = torch.linalg.vector_norm(rhs)
    if float(rhs_norm) == 0:
        return solution, PCGReport(0, True, 0.0)
    preconditioned = inverse * residual
    direction = preconditioned.clone()
    rz = torch.dot(residual, preconditioned)
    relative = 1.0
    for iteration in range(1, maximum + 1):
        product = matvec(direction)
        denominator = torch.dot(direction, product)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            return solution, PCGReport(iteration - 1, False, relative)
        alpha = rz / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        relative = float(torch.linalg.vector_norm(residual) / rhs_norm)
        if relative <= tolerance:
            return solution, PCGReport(iteration, True, relative)
        preconditioned = inverse * residual
        new_rz = torch.dot(residual, preconditioned)
        direction = preconditioned + (new_rz / rz.clamp_min(torch.finfo(rhs.dtype).tiny)) * direction
        rz = new_rz
    return solution, PCGReport(maximum, False, relative)


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("execution", ["reference", "optimized"])
@pytest.mark.parametrize("mode", ["converge", "maximum", "zero_rhs", "negative", "nonfinite_after_one"])
def test_pcg_combined_transfer_retains_solution_report_and_matvec_trace(dtype, execution, mode):
    matrix = torch.tensor([[5, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=dtype, device="cuda")
    rhs = torch.tensor([1, -2, 3], dtype=dtype, device="cuda")
    if mode == "zero_rhs":
        rhs.zero_()
    diagonal = matrix.diagonal()
    maximum = 1 if mode == "maximum" else 10
    tolerance = 1e-3
    traces = [[], []]

    def operator(trace):
        def matvec(value):
            trace.append(value.clone())
            if mode == "negative":
                return -value
            if mode == "nonfinite_after_one" and len(trace) > 1:
                return torch.full_like(value, float("nan"))
            return matrix @ value
        return matvec

    expected, expected_report = _serial_pcg(operator(traces[0]), rhs, diagonal, tolerance, maximum)
    actual, actual_report = preconditioned_conjugate_gradient(operator(traces[1]), rhs, diagonal=diagonal, tolerance=tolerance, max_iterations=maximum, execution=execution)
    assert actual_report == expected_report
    assert torch.equal(actual, expected)
    assert len(traces[0]) == len(traces[1])
    for original, candidate in zip(*traces):
        assert torch.equal(original, candidate)


def test_cached_bending_diagonal_keeps_oracle_values_and_reuses_contractions(monkeypatch):
    operator = BendingOperator((8, 9, 7), (3, 3, 3), (2, 2.5, 3), device="cpu", dtype=torch.float64)
    original = operator.diagonal().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("unchanged geometry must not repeat dense diagonal contractions")

    monkeypatch.setattr("fnit.fnirt.spline.design_diagonal", forbidden)
    assert torch.equal(operator.diagonal(), original)


def test_cached_affine_grid_preserves_scaled_mm_operation_order():
    from fnit.fnirt.registration import _fsl_affine_grid

    field = torch.linspace(-0.05, 0.05, 3 * 7 * 8 * 9).reshape(3, 7, 8, 9)
    affine = torch.tensor([[0.9, .03, .04, -.25], [.1, 1.1, .02, .3], [-.03, .07, 1.2, -.4], [0, 0, 0, 1]], dtype=torch.float64)
    mm_to_voxel = torch.diag(torch.tensor([-1.2, .7, .8, 1.0], dtype=torch.float32))
    original = _fsl_displacement_coordinates(field, affine, mm_to_voxel)
    actual = _fsl_displacement_coordinates(field, affine, mm_to_voxel, affine_grid=_fsl_affine_grid(affine, field.shape[1:]))
    assert torch.equal(actual.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("shape,spacing,zooms", [((7, 8, 9), (1, 2, 3), (1, 2, 3)), ((24, 28, 24), (5, 5, 5), (8, 8, 8))])
def test_gram_normal_matches_dense_double_and_avoids_wide_forward(device, shape, spacing, zooms, monkeypatch):
    dense = BendingOperator(shape, spacing, zooms, device=device, dtype=torch.float64)
    optimized = BendingOperator(shape, spacing, zooms, device=device, dtype=torch.float64, execution="optimized")
    coefficients = torch.randn((3, *fsl_control_shape(shape, spacing)), generator=torch.Generator().manual_seed(7), dtype=torch.float64).to(device)
    expected = dense.normal(coefficients)
    original_energy = dense.energy(coefficients)
    # Energy still uses the unchanged dense source ordering, independently of
    # the Hessian execution choice.
    assert torch.equal(optimized.energy(coefficients), original_energy)

    def forbidden(*args):
        raise AssertionError("PCG Gram normal must not materialise wide derivative fields")

    monkeypatch.setattr(optimized, "forward", forbidden)
    actual = optimized.normal(coefficients)
    torch.testing.assert_close(actual, expected, atol=5e-14, rtol=5e-14)
    cached = optimized._normal_grams
    assert torch.equal(optimized.normal(coefficients), actual)
    assert optimized._normal_grams is cached


def test_reference_masked_smoothing_does_not_dispatch_fused(monkeypatch):
    volume = torch.ones((1, 1, 3, 4, 5))
    mask = torch.ones(volume.shape[2:], dtype=torch.bool)
    expected = _fsl_masked_gaussian_blur(volume, 2, (1, 1, 1), mask, execution="reference")
    def forbidden(*args):
        raise AssertionError("reference smoothing must retain offset path")
    monkeypatch.setattr("fnit.fnirt.registration._fsl_gaussian_blur", forbidden)
    assert torch.equal(_fsl_masked_gaussian_blur(volume, 2, (1, 1, 1), mask, execution="reference"), expected)


def test_execution_options_are_explicit_and_reject_unknown_paths():
    from fnit.fnirt.registration import TorchFNIRT
    from fnit.fnirt.cli import build_parser
    assert TorchFNIRT(execution="reference").execution == "reference"
    assert TorchFNIRT().execution == "optimized"
    assert build_parser().parse_args(["--in", "moving.nii.gz", "--ref", "fixed.nii.gz", "--execution", "reference"]).execution == "reference"
    with pytest.raises(ValueError, match="execution"):
        TorchFNIRT(execution="fast")
