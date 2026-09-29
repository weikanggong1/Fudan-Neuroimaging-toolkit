"""Component oracles for the FSL FLIRT 2111.2 PyTorch target."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fnit
from fnit.flirt import TorchFLIRT
from fnit.flirt.core import (
    _DefaultFLIRTEngine,
    _centre_of_gravity,
    _coordinates_from_fsl_coefficients,
    _fsl_pull_coefficients,
    _newimage_percentile,
    _resample_output,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
    fsl_parameters_from_affine,
)


def _synthetic_pair():
    shape = (16, 17, 18)
    coordinates = np.indices(shape, dtype=np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    reference = np.exp(
        -sum(
            (coordinates[axis] - centre[axis]) ** 2
            for axis in range(3)
        ) / 9
    ).astype(np.float32)
    moving = np.roll(reference, 1, axis=0)
    vox2world = np.array(
        [[-2, 0, 0, 30], [0, 2, 0, -10], [0, 0, 2, 5], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return moving, reference, vox2world


def test_search_angles_match_flirt_float_sampling_oracle():
    coarse = _DefaultFLIRTEngine._angles(delta=np.pi / 3)
    fine = _DefaultFLIRTEngine._angles(delta=np.pi / 10)
    np.testing.assert_array_equal(
        coarse.view(np.uint32),
        [3217625051, 3204844178, 1057360530, 1070141403],
    )
    np.testing.assert_array_equal(
        fine.view(np.uint32),
        [3217625051, 3214989692, 3211871802, 3206601084, 3198212476,
         0, 1050728828, 1059117438, 1064388154, 1067506043, 1070141403],
    )


def test_coarse_interpolation_matches_newimage_float_oracle():
    volume = (np.arange(8, dtype=np.float64).reshape(2, 2, 2)
              * 1.23456789).astype(np.float32)
    result = _DefaultFLIRTEngine._interpolate_coarse(volume, (.3, .6, .9))
    assert np.asarray(result, dtype=np.float32).view(np.uint32) == 1082285777


def test_rigid_search_free_optimization_does_not_fit_scale(monkeypatch):
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving, reference, vox2world, vox2world,
        (1, 1, 1), (1, 1, 1), device="cpu", angular_search=False,
    )
    fitted_dofs = []
    monkeypatch.setattr(engine, "_optimize_search_subset",
                        lambda parameters: (parameters.copy(), 0.1))
    monkeypatch.setattr(engine, "cost", lambda matrix: 0.1)
    def capture(matrix, dof, maximum_iterations=4):
        fitted_dofs.append(dof)
        return matrix, 0.1

    monkeypatch.setattr(engine, "optimize_matrix", capture)
    engine.angular_candidates(dof=6)
    assert fitted_dofs == [6]
    fitted_dofs.clear()
    engine.angular_candidates(dof=12)
    assert fitted_dofs == [7]


def test_torch_flirt_is_the_exact_target_public_api():
    assert fnit.TorchFLIRT is TorchFLIRT
    assert TorchFLIRT(device="cpu").angular_search is True


def test_newimage_percentile_uses_order_statistic_without_interpolation():
    values = np.arange(64, dtype=np.float32)

    assert _newimage_percentile(values, .5) == 32
    assert np.percentile(values, 50) == 31.5


def test_fsl_parameter_composition_matches_flirt_2111_2_fixture():
    parameters = torch.tensor(
        [.05, -.04, .03, 1.2, -2.3, .7, 1.04, 1.04, 1.04, 0, 0, 0],
        dtype=torch.float64,
    )
    centre = torch.tensor([16.98656918, 16, 17], dtype=torch.float64)
    expected = np.array(
        [
            [1.038700492, .03117036336, .04158890696, -.6631258241],
            [-.03323397592, 1.038170644, .05193676256, -3.229124043],
            [-.03995912465, -.0532008727, 1.037869511, 1.586200716],
            [0, 0, 0, 1],
        ]
    )

    matrix = fsl_affine_from_parameters(parameters, centre, dof=7).numpy()

    np.testing.assert_allclose(matrix, expected, atol=7e-7, rtol=0)
    recovered = fsl_parameters_from_affine(matrix, centre.numpy())
    np.testing.assert_allclose(recovered[:7], parameters.numpy()[:7], atol=1e-7)


def test_correlation_ratio_matches_fsl_default_schedule_cost_oracles():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    expected = {
        8: 0.446353,
        4: 0.111078,
        2: 0.118574,
        1: 0.118574,
    }

    measured = {}
    for scale in expected:
        engine.set_scale(scale)
        measured[scale] = engine.cost(np.eye(4))

    np.testing.assert_allclose(
        [measured[scale] for scale in expected],
        list(expected.values()),
        atol=1e-6,
        rtol=0,
    )


def test_nonidentity_correlation_ratio_matches_fsl_fixture():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    matrix = np.array(
        [
            [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
            [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
            [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
            [0, 0, 0, 1],
        ]
    )
    engine.set_scale(2)

    assert abs(engine.cost(matrix) - 0.356834) <= 1e-6


def test_correlation_ratio_bin_order_is_stable_within_each_bin():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    cost = engine.level.cost
    order = cost.bin_sort_order.cpu().numpy()
    bins = cost.bin_index.cpu().numpy()

    assert np.all(bins[order][1:] >= bins[order][:-1])
    for bin_index in np.unique(bins):
        positions = order[bins[order] == bin_index]
        assert np.all(positions[1:] > positions[:-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_correlation_ratio_reduction_is_bitwise_repeatable():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cuda",
        angular_search=False,
    )
    engine.set_scale(2)
    matrix = np.eye(4)

    measured = [engine.level.cost(matrix) for _ in range(3)]

    assert measured[0] == measured[1] == measured[2]


def test_angular_search_translation_accounts_for_initial_matrix(monkeypatch):
    moving, reference, vox2world = _synthetic_pair()
    initial = np.eye(4)
    initial[:3, 3] = (4.0, -3.0, 2.0)
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
        initial_matrix=initial,
    )
    moving_centre = _centre_of_gravity(
        engine.level.moving, np.diag([*engine.moving_sizes, 1.0])
    )
    reference_centre = _centre_of_gravity(
        engine.level.reference,
        np.diag([*engine.level.reference_sizes, 1.0]),
    )
    expected = reference_centre - (initial @ np.r_[moving_centre, 1])[:3]

    def capture(parameters, maximum_iterations=4):
        np.testing.assert_allclose(parameters[3:6], expected, atol=1e-12)
        raise RuntimeError("captured initial search parameters")

    monkeypatch.setattr(engine, "_optimize_search_subset", capture)
    with pytest.raises(RuntimeError, match="captured initial search parameters"):
        engine.angular_candidates()


def test_unavailable_finer_scale_refreshes_unblurred_moving_volume():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (.8, .8, .8),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    engine.set_scale(2)
    blurred = engine.level.moving.clone()
    reference_at_two = engine.level.reference
    centre_at_two = engine.level.centre.copy()

    engine.set_scale(1)

    assert engine.level.moving is engine.moving_original
    assert engine.level.cost.moving is engine.moving_original
    assert engine.level.reference is reference_at_two
    assert engine.level.cost.bins == 128
    assert engine.level.cost.smooth_size == 1
    assert not torch.equal(blurred, engine.level.moving)
    np.testing.assert_array_equal(engine.level.centre, centre_at_two)


def test_coordinate_optimizer_limits_the_requested_parameter_prefix():
    initial = np.array([4.0, -3.0, 19.0])
    target = np.array([1.25, -.75, -11.0])

    fitted, value = fsl_coordinate_optimize(
        initial,
        np.array([.01, .01, .01]),
        lambda point: float(np.square(point - target).sum()),
        maximum_iterations=4,
        bound_guess=(10.0, 1.0),
        numopt=2,
    )

    np.testing.assert_allclose(fitted[:2], target[:2], atol=.02, rtol=0)
    assert fitted[2] == initial[2]
    assert value < 1e-3 + (initial[2] - target[2]) ** 2


def test_output_resampling_includes_flirt_default_antialias_blur():
    moving = np.zeros((9, 9, 9), dtype=np.float32)
    moving[4, 4, 4] = 1
    moving_fsl = np.diag([1, 1, 1, 1])
    fixed_fsl = np.diag([2, 2, 2, 1])

    output = _resample_output(
        moving,
        (5, 5, 5),
        moving_fsl,
        fixed_fsl,
        np.eye(4),
        (1, 1, 1),
        (2, 2, 2),
        torch.device("cpu"),
    )

    sigma = .85
    side = np.exp(-1 / (sigma * sigma * 4))
    centre_weight = (1 / (1 + 2 * side)) ** 3
    assert float(output[2, 2, 2]) < 1
    assert abs(float(output[2, 2, 2]) - centre_weight) < 1e-6
    assert torch.count_nonzero(output) == 1


def test_cost_cache_does_not_alias_distinct_float64_affines():
    engine = object.__new__(_DefaultFLIRTEngine)
    engine.initial_matrix = np.eye(4, dtype=np.float64)
    engine.cost_evaluations = 0
    engine._cache = {}
    engine.level = SimpleNamespace(cost=lambda affine: float(affine[0, 3]))
    first = np.eye(4, dtype=np.float64)
    first[0, 3] = 1.0
    second = first.copy()
    second[0, 3] += 2.0**-30

    assert np.array_equal(first.astype(np.float32), second.astype(np.float32))
    assert engine.cost(first) == 1.0
    assert engine.cost(second) == 1.0 + 2.0**-30
    assert engine.cost_evaluations == 2


def test_pull_coefficients_cast_only_after_double_inverse_and_sampling():
    affine = np.array(
        [
            [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
            [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
            [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    coefficients = _fsl_pull_coefficients(
        affine,
        (.73, 1.17, 2.31),
        (2.0, 1.3, .91),
        device=torch.device("cpu"),
    ).numpy()
    # uint32 is the direct oracle for the twelve assignments from NEWMAT
    # double values to the local float a11 ... a34 variables.
    expected_bits = np.array(
        [
            [1076387784, 3184885958, 3164721319, 1062485557],
            [1028916002, 1066549451, 3176737758, 1074557794],
            [1024133320, 1021425494, 1053120071, 3208414883],
        ],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(coefficients.view(np.uint32), expected_bits)


def test_explicit_float_coordinate_generation_matches_scalar_oracle():
    coefficient_bits = np.array(
        [
            [1076387784, 3184885958, 3164721319, 1062485557],
            [1028916002, 1066549451, 3176737758, 1074557794],
            [1024133320, 1021425494, 1053120071, 3208414883],
        ],
        dtype=np.uint32,
    )
    coefficients = coefficient_bits.view(np.float32)
    grid = np.array(
        [[0, 1, 7, 19], [0, 3, 11, 2], [0, 5, 13, 17]],
        dtype=np.float32,
    )
    expected = np.empty((3, grid.shape[1]), dtype=np.float32)
    for row in range(3):
        for column in range(grid.shape[1]):
            x, y, z = grid[:, column]
            value = np.float32(y * coefficients[row, 1])
            value = np.float32(value + np.float32(z * coefficients[row, 2]))
            value = np.float32(value + coefficients[row, 3])
            value = np.float32(value + np.float32(x * coefficients[row, 0]))
            expected[row, column] = value

    measured = _coordinates_from_fsl_coefficients(
        torch.from_numpy(coefficients.copy()), torch.from_numpy(grid)
    ).numpy()

    np.testing.assert_array_equal(measured.view(np.uint32), expected.view(np.uint32))


def _newimage_cog_scalar_oracle(array, sampling):
    minimum = np.float32(array.min())
    nlim = max(int(np.sqrt(float(array.size))), 1000)
    total = vx = vy = vz = 0.0
    block_total = block_x = block_y = block_z = 0.0
    count = 0
    for z in range(array.shape[2]):
        for y in range(array.shape[1]):
            for x in range(array.shape[0]):
                value = float(np.float32(array[x, y, z] - minimum))
                block_x += value * x
                block_y += value * y
                block_z += value * z
                block_total += value
                count += 1
                if count > nlim:
                    total += block_total
                    vx += block_x
                    vy += block_y
                    vz += block_z
                    block_total = block_x = block_y = block_z = 0.0
                    count = 0
    total += block_total
    vx += block_x
    vy += block_y
    vz += block_z
    if abs(total) < 1e-5:
        total = 1.0
    voxel = np.array([vx / total, vy / total, vz / total])
    return sampling[:3, :3] @ voxel + sampling[:3, 3]


def test_cpu_centre_of_gravity_matches_newimage_scalar_block_order():
    index = np.arange(11 * 10 * 11, dtype=np.float32).reshape(11, 10, 11)
    array = np.float32(np.sin(index * np.float32(.071)))
    array += np.float32((index % 17) * np.float32(.003))
    sampling = np.array(
        [[.73, 0, 0, 1.2], [0, 1.17, 0, -3.4], [0, 0, 2.31, .8], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    expected = _newimage_cog_scalar_oracle(array, sampling)
    measured = _centre_of_gravity(torch.from_numpy(array), sampling)

    np.testing.assert_array_equal(measured.view(np.uint64), expected.view(np.uint64))


def test_affine_decomposition_uses_fsl_float_intermediates():
    affine = np.array(
        [
            [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
            [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
            [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    centre = np.array([16.98656918, 16, 17], dtype=np.float64)
    expected = np.array(
        [
            float.fromhex("0x1.99999a0000000p-5"),
            float.fromhex("-0x1.47ae140000000p-5"),
            float.fromhex("0x1.eb851c0000000p-6"),
            float.fromhex("0x1.33333333529d0p+0"),
            float.fromhex("-0x1.2666666664498p+1"),
            float.fromhex("0x1.66666666ed840p-1"),
            float.fromhex("0x1.0a3d700000000p+0"),
            float.fromhex("0x1.f0a3d80000000p-1"),
            float.fromhex("0x1.051eb80000000p+0"),
            float.fromhex("0x1.eb851e0000000p-6"),
            float.fromhex("-0x1.47ae160000000p-6"),
            float.fromhex("0x1.eb851e0000000p-7"),
        ],
        dtype=np.float64,
    )

    measured = fsl_parameters_from_affine(affine, centre)

    np.testing.assert_array_equal(measured.view(np.uint64), expected.view(np.uint64))
