"""Fixed-point CPU buffer reuse retains the frozen solver, not FSL parity."""

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.applywarp.core import _fsl_voxel_matrix, _spatial_grid
from fnit.convertwarp.core import _PullField
from fnit.fnirt.io import make_fsl_coefficient_image
from fnit.fnirt.spline import fsl_control_shape
from fnit.invwarp import TorchInvWarp


def _frozen_solver(reference, warp, convention, iterations, tolerance):
    """Original allocation path, frozen independently of the CPU dispatch."""
    field = _PullField(warp, torch.device("cpu"), convention)
    axes = torch.meshgrid(*(torch.linspace(0, size - 1, min(size, 5), dtype=torch.float64)
                            for size in field.shape), indexing="ij")
    grid = torch.stack(axes).reshape(3, -1)
    scaled = torch.as_tensor(field.scaled, dtype=torch.float64)
    sample = (scaled[:3, :3] @ grid + scaled[:3, 3:4]).reshape(3, *(min(size, 5) for size in field.shape))
    mapped, _ = field.sample(sample)
    design = np.column_stack((sample.reshape(3, -1).T.numpy(), np.ones(sample.numel() // 3)))
    fit = np.linalg.lstsq(design, mapped.reshape(3, -1).T.numpy(), rcond=None)[0]
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = fit[:3].T, fit[3]
    inverse = torch.as_tensor(np.linalg.inv(affine))
    target = _spatial_grid(reference.shape, _fsl_voxel_matrix(reference), torch.device("cpu")).reshape(3, *reference.shape)
    flat = target.reshape(3, -1)
    estimate = (inverse[:3, :3] @ flat + inverse[:3, 3:4]).reshape(target.shape)
    converged = False
    for iteration in range(iterations):
        mapped, _ = field.sample(estimate)
        correction = inverse[:3, :3] @ (target - mapped).reshape(3, -1)
        estimate = estimate + correction.reshape(target.shape)
        if float(correction.abs().max()) < tolerance:
            converged = True
            break
    mapped, valid = field.sample(estimate)
    return estimate, target, iteration + 1, converged, mapped, valid


@pytest.mark.parametrize("kind", ["relative", "absolute", "coefficient"])
@pytest.mark.parametrize("output_convention", ["relative", "absolute"])
@pytest.mark.parametrize("iterations,tolerance", [(1, 1e-12), (30, 0.01)])
def test_inverse_buffers_keep_frozen_float32_output_and_stopping(kind, output_convention, iterations, tolerance):
    shape = (17, 13, 11)
    geometry = np.diag([-2.0, 2.5, 3.0, 1.0])
    reference = nib.Nifti1Image(np.zeros(shape, np.float32), geometry)
    if kind == "coefficient":
        controls = fsl_control_shape(shape, (3, 3, 3))
        values = np.random.default_rng(43).normal(0, 0.15, (*controls, 3)).astype(np.float32)
        affine = np.diag([1.03, 0.98, 1.02, 1.0])
        affine[:3, 3] = [0.2, -0.1, 0.05]
        warp = make_fsl_coefficient_image(values, shape, (2, 2.5, 3), (3, 3, 3), affine)
        convention = "auto"
    else:
        x, y, z = np.indices(shape, dtype=np.float32)
        values = np.stack((0.3 * np.sin(y / 5), 0.15 * np.sin(z / 4), 0.12 * np.sin(x / 6)), axis=-1)
        if kind == "absolute":
            values += np.moveaxis(_spatial_grid(shape, _fsl_voxel_matrix(reference), torch.device("cpu")).reshape(3, *shape).numpy(), 0, -1)
        warp = nib.Nifti1Image(values.astype(np.float32), geometry)
        if kind == "relative":
            warp.header["intent_code"] = 2006
        convention = kind
    estimate, target, used, converged, mapped, valid = _frozen_solver(reference, warp, convention, iterations, tolerance)
    expected = estimate - target if output_convention == "relative" else estimate
    expected = np.moveaxis(expected.numpy(), 0, -1).astype(np.float32)
    actual = TorchInvWarp("cpu")(
        reference, warp, warp_convention=convention, output_convention=output_convention,
        iterations=iterations, tolerance_mm=tolerance,
    )
    np.testing.assert_array_equal(np.asarray(actual.image.dataobj).view(np.int32), expected.view(np.int32))
    assert actual.qc["iterations_used"] == used
    assert actual.qc["converged"] == converged
    assert actual.qc["iterations"] == iterations
    residual = (mapped - target).square().sum(0).sqrt()
    assert actual.valid_fraction == float(valid.float().mean())
    assert actual.qc["median_residual_mm_in_field"] == (float(residual[valid].median()) if valid.any() else None)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, float("nan")])
def test_invalid_iteration_budget_is_rejected_before_loading(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        TorchInvWarp("cpu")(None, None, iterations=iterations)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf"), True])
def test_nonfinite_or_nonpositive_tolerance_is_rejected_before_loading(tolerance):
    with pytest.raises(ValueError, match="finite and positive"):
        TorchInvWarp("cpu")(None, None, tolerance_mm=tolerance)
