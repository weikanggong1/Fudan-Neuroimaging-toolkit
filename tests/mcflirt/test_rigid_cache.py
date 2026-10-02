"""CPU cache 的数值合同；真实影像精度和耗时在 validation 测量。"""

import numpy as np
import pytest
import torch

from fnit.flirt.core import _fsl_pull_coefficients, fsl_affine_from_parameters
from fnit.mcflirt._rigid import FSLPullCoefficients, RigidAffineComposer


def _parameters():
    return np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.])


def _assert_matrix_bits(actual, parameters, centre):
    expected = fsl_affine_from_parameters(
        torch.as_tensor(parameters, dtype=torch.float64), centre, 6
    )
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    np.testing.assert_array_equal(
        actual.numpy().view(np.uint64), expected.numpy().view(np.uint64)
    )


@pytest.mark.parametrize("axis", range(3))
def test_rotation_zero_and_float32_threshold_branches(axis):
    centre = np.array([102.32176228, 118.81900018, 95.9181730])
    compose = RigidAffineComposer(centre)
    threshold = np.float32(1e-8)
    angles = (
        0., -0., 1e-12, -1e-12,
        float(np.nextafter(threshold, np.float32(0))),
        float(threshold),
        float(np.nextafter(threshold, np.float32(np.inf))),
        -float(np.nextafter(threshold, np.float32(0))),
        -float(threshold),
        -float(np.nextafter(threshold, np.float32(np.inf))),
        .2, -.2, 3.14, -3.14,
    )
    for angle in angles:
        parameters = _parameters()
        parameters[axis] = angle
        _assert_matrix_bits(compose(parameters), parameters, centre)
        assert compose._axis_keys[axis] == np.asarray(angle).tobytes()


def test_per_coordinate_updates_and_translation_cache_are_bitwise_exact():
    generator = np.random.default_rng(41061)
    centre = np.array([78.41, 90.88, 104.997])
    compose = RigidAffineComposer(centre)
    parameters = _parameters()
    for _ in range(4):
        for axis in range(6):
            for _ in range(8):
                parameters[axis] += generator.normal(
                    scale=.02 if axis < 3 else .3
                )
                rotations_before = compose._axis_rotations.copy()
                product_before = compose._rotation
                _assert_matrix_bits(compose(parameters), parameters, centre)
                if axis >= 3:
                    assert compose._rotation is product_before
                    assert all(
                        current is previous
                        for current, previous in zip(
                            compose._axis_rotations, rotations_before
                        )
                    )
                else:
                    assert all(
                        current is previous
                        for index, (current, previous) in enumerate(zip(
                            compose._axis_rotations, rotations_before
                        ))
                        if index != axis and previous is not None
                    )
    # A caller may reuse and edit its output; cached matrices stay intact.
    first = compose(parameters)
    first[:] = 0
    _assert_matrix_bits(compose(parameters), parameters, centre)


def test_centres_are_copied_and_composer_instances_are_independent():
    first_centre = np.array([15.3, 33.8, 70.1])
    frozen_first_centre = first_centre.copy()
    second_centre = np.array([77.9, -25.12, 44.89])
    first = RigidAffineComposer(first_centre)
    second = RigidAffineComposer(torch.as_tensor(second_centre))
    parameters = _parameters()
    parameters[:6] = [.13, -.04, .01, 1.2, -1.9, .8]
    first_centre[:] = -200
    _assert_matrix_bits(first(parameters), parameters, frozen_first_centre)
    _assert_matrix_bits(second(parameters), parameters, second_centre)
    assert first._axis_rotations[0] is not second._axis_rotations[0]
    parameters[0] += .07
    _assert_matrix_bits(second(parameters), parameters, second_centre)
    parameters[0] -= .07
    _assert_matrix_bits(first(parameters), parameters, frozen_first_centre)


@pytest.mark.parametrize("reference_sizes", [(8., 8., 8.), (4., 4., 4.)])
@pytest.mark.parametrize("moving_sizes", [
    (2.3999996185302734,) * 3,
    (1.7, 1.3, 2.2),
])
def test_pull_coefficients_keep_original_matrix_products_and_float32_bits(
    moving_sizes, reference_sizes
):
    generator = np.random.default_rng(421)
    centre = np.array([83.13, 61.019, 100.456])
    compose = RigidAffineComposer(centre)
    coefficients = FSLPullCoefficients(moving_sizes, reference_sizes)
    parameters = _parameters()
    matrices = [np.eye(4)]
    for _ in range(30):
        parameters[:3] = generator.normal(size=3, scale=.13)
        parameters[3:6] = generator.normal(size=3, scale=3.)
        matrices.append(compose(parameters).numpy())
    for matrix in matrices:
        expected = _fsl_pull_coefficients(
            matrix, moving_sizes, reference_sizes, device="cpu"
        ).numpy()
        actual = coefficients(matrix)
        assert actual.shape == (3, 4)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(
            actual.view(np.uint32), expected.view(np.uint32)
        )


def test_pull_coefficients_preserve_general_affine_inverse_not_transpose():
    moving_sizes = (1.7, 1.3, 2.2)
    reference_sizes = (4., 4., 4.)
    coefficients = FSLPullCoefficients(moving_sizes, reference_sizes)
    matrix = np.array([
        [1.04, -.073, .018, 1.7],
        [.052, .98, .012, -.9],
        [.013, -.045, 1.03, 2.4],
        [0., 0., 0., 1.],
    ])
    expected = _fsl_pull_coefficients(
        matrix, moving_sizes, reference_sizes, device="cpu"
    ).numpy()
    actual = coefficients(matrix)
    np.testing.assert_array_equal(
        actual.view(np.uint32), expected.view(np.uint32)
    )
    actual[:] = 0
    np.testing.assert_array_equal(
        coefficients(matrix).view(np.uint32), expected.view(np.uint32)
    )


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="centre"):
        RigidAffineComposer(np.zeros(4))
    with pytest.raises(ValueError, match="parameters"):
        RigidAffineComposer(np.zeros(3))(np.zeros(6))
    with pytest.raises(ValueError, match="moving_to_reference"):
        FSLPullCoefficients((2.,) * 3, (4.,) * 3)(np.eye(3))
