import numpy as np
import pytest
import torch

import fnit.fnirt.registration as registration_module
from fnit._nib import FNITNifti1Image
from fnit._transforms import AffineTransform
from fnit.fnirt.registration import (
    GMFNIRTConfig,
    TorchFNIRT,
    _LevelSystem,
    _fsl_affine_grid,
    _fsl_displacement_coordinates,
    _fsl_masked_gaussian_blur,
    _process_knot_spacing_schedule,
    _spline_jacobian,
    _subsampled_size,
    _trilinear_sample,
    spm_like_mean,
)
from fnit.fnirt.spline import (
    BendingOperator,
    fsl_control_shape,
    spline_bases,
)


def _image(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return FNITNifti1Image(np.asarray(data), affine)


def _volume(shape=(8, 8, 8)):
    coordinates = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum((axis - center[index]) ** 2 for index, axis in enumerate(coordinates))
        / 8
    ).astype(np.float32)
    return _image(data)


def _single_level_config():
    return GMFNIRTConfig(
        subsampling=(1,),
        maximum_iterations=(0,),
        input_fwhm_mm=(0.0,),
        reference_fwhm_mm=(0.0,),
        regularization=(0.0,),
        estimate_intensity=(False,),
        apply_reference_mask=(False,),
        implicit_reference_mask=True,
        implicit_input_mask=True,
        warp_resolution_mm=(4.0, 4.0, 4.0),
    )


def _failed_topology_projection(coefficients, shape, *_):
    jacobian = torch.ones(
        shape, dtype=coefficients.dtype, device=coefficients.device
    )
    qc = {
        "required": True,
        "calls": [],
        "range": [0.204623, 5.04025],
        "succeeded": False,
    }
    return coefficients, jacobian, qc


def test_spm_like_mean_uses_one_eighth_of_whole_image_mean():
    values = np.array([0.0, 1.0, 2.0, 20.0], dtype=np.float32)
    # Whole-volume mean is 5.75; values above 0.71875 are 1, 2 and 20.
    assert spm_like_mean(values) == np.mean([1.0, 2.0, 20.0])


def test_fsl_recursive_subsampling_keeps_endpoint_coverage():
    assert _subsampled_size(91, 4) == 24
    assert _subsampled_size(109, 4) == 28
    assert _subsampled_size(90, 2) == 46


def test_full_resolution_knot_spacing_uses_each_process_final_subsampling():
    schedule = _process_knot_spacing_schedule(
        (
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (10.0, 10.0, 10.0),
            (2.0, 2.0, 2.0),
            (2.0, 2.0, 2.0),
        ),
        (1, 1, 1, 1, 2, 3),
        (8, 4, 2, 2, 1, 1),
        (1.0, 1.0, 1.0),
    )

    assert schedule == (
        (5, 5, 5),
        (5, 5, 5),
        (5, 5, 5),
        (5, 5, 5),
        (2, 2, 2),
        (2, 2, 2),
    )
    assert fsl_control_shape((182, 218, 182), schedule[3]) == (39, 46, 39)


def test_process_ending_above_full_resolution_upsamples_before_cout():
    shape = (20, 20, 20)
    moving = _volume(shape)
    fixed = moving.copy()
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )
    config = GMFNIRTConfig(
        subsampling=(2,),
        maximum_iterations=(0,),
        input_fwhm_mm=(0.0,),
        reference_fwhm_mm=(0.0,),
        regularization=(0.0,),
        estimate_intensity=(False,),
        apply_reference_mask=(False,),
        process_stages=(1,),
        warp_resolution_mm=(10.0, 10.0, 10.0),
    )

    result = TorchFNIRT(device="cpu", config=config)(moving, fixed, initial)

    assert result.qc["final_output_upsampled_to_reference_grid"] is True
    assert result.qc["process_full_resolution_knot_spacing_voxels"] == [
        [5, 5, 5]
    ]
    assert result.coefficient_image.shape == (7, 7, 7, 3)
    np.testing.assert_allclose(
        result.coefficient_image.header.get_zooms()[:3],
        (5.0, 5.0, 5.0),
        rtol=0,
        atol=0,
    )


def test_trilinear_sampler_returns_piecewise_analytic_gradient():
    axes = torch.meshgrid(
        torch.arange(5.0), torch.arange(6.0), torch.arange(7.0), indexing="ij"
    )
    volume = 2 * axes[0] - 3 * axes[1] + 0.5 * axes[2] + 7
    coordinates = torch.tensor([2.25, 3.5, 1.75])[:, None, None, None]
    value, valid, gradient = _trilinear_sample(volume, coordinates)
    assert bool(valid)
    torch.testing.assert_close(value.squeeze(), torch.tensor(1.875))
    torch.testing.assert_close(
        gradient.squeeze(), torch.tensor([2.0, -3.0, 0.5])
    )


def test_trilinear_valid_mask_uses_newimage_boundary_tolerance():
    volume = torch.ones((2, 2, 2), dtype=torch.float32)
    coordinates = torch.tensor(
        [[-5e-9, -2e-8], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )

    sampled, valid, gradient = _trilinear_sample(volume, coordinates)

    assert valid.tolist() == [True, False]
    assert sampled[0] > 0.9999999
    assert sampled[1] == 0
    assert gradient[0, 0] == 1


def test_warpfns_coordinate_arithmetic_uses_scalar_float_order():
    affine = torch.tensor(
        [
            [1.0000001, 0.1000003, -0.2000002, 0.3000004],
            [-0.0500002, 0.9999998, 0.0700001, -0.4000003],
            [0.0300001, -0.0900002, 1.1000003, 0.2000001],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    field = torch.linspace(
        -0.015, 0.021, 3 * 3 * 2 * 2, dtype=torch.float32
    ).reshape(3, 3, 2, 2)
    mm_to_voxel = torch.tensor(
        [
            [0.7, 0.02, -0.01, 0.1],
            [0.01, 0.8, 0.03, -0.2],
            [-0.02, 0.01, 0.9, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    expected_mm = np.empty_like(field.numpy())
    matrix = affine.numpy().astype(np.float32)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for row in range(3):
                    value = np.float32(np.float32(i) * matrix[row, 0])
                    value = np.float32(
                        value + np.float32(j) * matrix[row, 1]
                    )
                    value = np.float32(
                        value + np.float32(k) * matrix[row, 2]
                    )
                    expected_mm[row, i, j, k] = np.float32(
                        value + matrix[row, 3]
                    )
    np.testing.assert_array_equal(
        _fsl_affine_grid(affine, (3, 2, 2)).numpy(), expected_mm
    )

    expected_voxels = np.empty_like(expected_mm)
    source_mm = (expected_mm + field.numpy()).astype(np.float32)
    matrix = mm_to_voxel.numpy().astype(np.float32)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                for row in range(3):
                    value = np.float32(source_mm[0, i, j, k] * matrix[row, 0])
                    value = np.float32(
                        value + source_mm[1, i, j, k] * matrix[row, 1]
                    )
                    value = np.float32(
                        value + source_mm[2, i, j, k] * matrix[row, 2]
                    )
                    expected_voxels[row, i, j, k] = np.float32(
                        value + matrix[row, 3]
                    )
    np.testing.assert_array_equal(
        _fsl_displacement_coordinates(
            field, affine, mm_to_voxel
        ).numpy(),
        expected_voxels,
    )



def test_masked_smoothing_renormalizes_inside_implicit_mask():
    mask = torch.zeros((9, 9, 9), dtype=torch.bool)
    mask[2:7, 2:7, 2:7] = True
    volume = torch.full((1, 1, 9, 9, 9), 99.0, dtype=torch.float32)
    volume[0, 0][mask] = 7.0

    actual = _fsl_masked_gaussian_blur(
        volume, 4.0, (1.0, 1.0, 1.0), mask
    )[0, 0]

    torch.testing.assert_close(
        actual[mask], torch.full_like(actual[mask], 7.0), atol=3e-6, rtol=0
    )
    assert torch.count_nonzero(actual[~mask]) == 0


def test_implicit_zero_masks_are_active_without_explicit_mask():
    shape = (8, 8, 8)
    data = np.zeros(shape, dtype=np.float32)
    data[0, 0, 0] = np.float32(1e-20)
    data[2:6, 2:6, 2:6] = 1.0
    moving = _image(data)
    fixed = moving.copy()
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    result = TorchFNIRT(device="cpu", config=_single_level_config())(
        moving, fixed, initial
    )

    level = result.qc["levels"][0]
    assert level["implicit_input_mask"] is True
    assert level["implicit_reference_mask"] is True
    assert level["mask_voxels"] == 4**3


def test_input_implicit_mask_is_built_after_mean_scaling():
    shape = (8, 8, 8)
    moving_data = np.zeros(shape, dtype=np.float32)
    moving_data[2:6, 2:6, 2:6] = 1.0
    moving_data[0, 0, 0] = np.float32(2e-18)
    fixed_data = moving_data.copy()
    fixed_data[0, 0, 0] = 1.0
    moving = _image(moving_data)
    fixed = _image(fixed_data)
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    result = TorchFNIRT(device="cpu", config=_single_level_config())(
        moving, fixed, initial
    )

    assert result.qc["levels"][0]["mask_voxels"] == 4**3 + 1


def test_warped_input_mask_is_truncated_to_char_before_thresholding():
    shape = (4, 4, 4)
    spacing = (2, 2, 2)
    dtype = torch.float64
    bases = spline_bases(
        shape,
        spacing,
        (1.0, 1.0, 1.0),
        device="cpu",
        dtype=dtype,
    )
    bending = BendingOperator(
        shape,
        spacing,
        (1.0, 1.0, 1.0),
        device="cpu",
        dtype=dtype,
    )
    moving_mask = torch.zeros(shape, dtype=torch.float32)
    moving_mask[1:3] = 1
    coordinate_affine = torch.eye(4, dtype=dtype)
    coordinate_affine[0, 3] = 0.25
    system = _LevelSystem(
        torch.ones(shape, dtype=torch.float32),
        torch.ones(shape, dtype=torch.float32),
        None,
        moving_mask,
        torch.eye(4, dtype=torch.float32),
        torch.zeros((3, *shape), dtype=torch.float32),
        torch.eye(4, dtype=torch.float32),
        bases,
        bending,
        0.0,
        False,
        False,
        coordinate_affine,
    )
    coefficients = torch.zeros(
        (3, *fsl_control_shape(shape, spacing)), dtype=dtype
    )

    state = system.evaluate(coefficients, torch.ones((), dtype=dtype))

    # At output x=2 the trilinear mask is 0.75. FSL writes it to a char
    # volume first, making it zero; a direct float >0.5 test would retain it.
    assert state["count"] == 4 * 4
    assert torch.count_nonzero(state["mask"][2]) == 0



def test_process_boundary_uses_scg_and_inwarp_knot_refinement():
    shape = (8, 8, 8)
    data = np.zeros(shape, dtype=np.float32)
    data[1:7, 1:7, 1:7] = 1.0
    moving = _image(data)
    fixed = moving.copy()
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )
    config = GMFNIRTConfig(
        subsampling=(1, 1),
        maximum_iterations=(0, 0),
        input_fwhm_mm=(0.0, 0.0),
        reference_fwhm_mm=(0.0, 0.0),
        regularization=(0.0, 0.0),
        estimate_intensity=(False, False),
        apply_reference_mask=(False, False),
        minimization_methods=("lm", "scg"),
        process_stages=(1, 2),
        warp_resolution_mm=(4.0, 4.0, 4.0),
        warp_resolution_schedule_mm=(
            (4.0, 4.0, 4.0),
            (2.0, 2.0, 2.0),
        ),
    )

    result = TorchFNIRT(device="cpu", config=config)(moving, fixed, initial)

    first, second = result.qc["levels"]
    assert first["process_boundary_handoff"] is False
    assert second["process_boundary_handoff"] is True
    assert second["minimization_method"] == "scg"
    assert second["knot_spacing_voxels"] == [2, 2, 2]
    assert result.qc["process_handoff_float32_coefficients"] is True


def test_gm_config_rejects_mismatched_schedules():
    try:
        GMFNIRTConfig(maximum_iterations=(1, 2))
    except ValueError as error:
        assert "same" in str(error)
    else:
        raise AssertionError("invalid schedule was accepted")


def test_failed_topology_projection_warns_and_continues_by_default(monkeypatch):
    monkeypatch.setattr(
        registration_module,
        "_force_jacobian_range",
        _failed_topology_projection,
    )
    moving = _volume()
    fixed = moving.copy()
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.warns(
        RuntimeWarning,
        match=r"Jacobian range was 0\.204623--5\.04025.*continuing as FSL FNIRT",
    ):
        result = TorchFNIRT(device="cpu", config=_single_level_config())(
            moving, fixed, initial
        )

    assert result.qc["strict_topology"] is False
    topology_qc = result.qc["levels"][0]["topology_projection"]
    assert topology_qc["succeeded"] is False
    assert topology_qc["range"] == [0.204623, 5.04025]


def test_failed_topology_projection_raises_in_explicit_strict_mode(monkeypatch):
    monkeypatch.setattr(
        registration_module,
        "_force_jacobian_range",
        _failed_topology_projection,
    )
    moving = _volume()
    fixed = moving.copy()
    initial = AffineTransform(
        np.eye(4), source=moving, target=fixed, space="world"
    )

    with pytest.raises(
        RuntimeError,
        match=r"Jacobian range was 0\.204623--5\.04025",
    ):
        TorchFNIRT(
            device="cpu",
            config=_single_level_config(),
            strict_topology=True,
        )(moving, fixed, initial)


def test_spline_jacobian_matches_reproduced_linear_field():
    shape = (12, 11, 10)
    spacing = (3, 3, 3)
    voxel_sizes = (2.0, 2.5, 3.0)
    control_shape = fsl_control_shape(shape, spacing)
    coefficients = torch.zeros((3, *control_shape), dtype=torch.float64)
    centres = tuple(
        (torch.arange(count, dtype=torch.float64) - 1)
        * knot_spacing
        * voxel_size
        for count, knot_spacing, voxel_size in zip(
            control_shape, spacing, voxel_sizes
        )
    )
    coefficients[0] = 0.1 * centres[0][:, None, None]
    coefficients[1] = -0.05 * centres[1][None, :, None]
    coefficients[2] = 0.2 * centres[2][None, None, :]
    determinant = _spline_jacobian(
        coefficients, shape, spacing, voxel_sizes
    )
    expected = torch.full_like(determinant, 1.1 * 0.95 * 1.2)
    # The retired cubic NCoef rule truncates one still-nonzero spline when
    # (size + 1) is exactly divisible by the knot spacing.  FSL therefore
    # reproduces the linear field on the interior but not on that final plane.
    torch.testing.assert_close(
        determinant[:, :-1], expected[:, :-1], atol=2e-12, rtol=2e-12
    )
    assert bool(torch.isfinite(determinant).all())


def test_full_spline_jacobian_includes_affine_pull():
    shape = (8, 7, 6)
    spacing = (3, 3, 3)
    coefficients = torch.zeros(
        (3, *fsl_control_shape(shape, spacing)), dtype=torch.float64
    )
    affine_pull = torch.tensor(
        [[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]],
        dtype=torch.float64,
    )
    determinant = _spline_jacobian(
        coefficients,
        shape,
        spacing,
        (2.0, 2.0, 2.0),
        affine_pull_linear=affine_pull,
    )
    torch.testing.assert_close(
        determinant,
        torch.full_like(determinant, torch.linalg.det(affine_pull)),
    )
