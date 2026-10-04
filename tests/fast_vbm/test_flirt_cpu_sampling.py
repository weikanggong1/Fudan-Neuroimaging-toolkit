"""CPU fusion preserves float32 voxel arithmetic; fixtures are not benchmarks."""

import numpy as np
import pytest
import torch

from fnit.flirt import core
from fnit.flirt._cpu import sample_output
from fnit.flirt.batched import BatchedAffineCost


def _assert_bits(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(actual[valid].view(np.uint32), expected[valid].view(np.uint32))


def _cost(cost_class, shape, weighted, smooth):
    rng = np.random.default_rng(41004)
    reference = torch.from_numpy(rng.uniform(-30, 900, shape).astype(np.float32))
    moving = torch.from_numpy(rng.uniform(-40, 700, shape).astype(np.float32))
    weights = {}
    if weighted:
        weights = {"reference_weight": torch.from_numpy(rng.uniform(-0.1, 1, shape).astype(np.float32)),
                   "moving_weight": torch.from_numpy(rng.uniform(-0.1, 1, shape).astype(np.float32))}
    return cost_class(reference, moving, (1.1, 1.9, 2.7), (1.7, 1.3, 2.2),
                      bins=32, smooth_size=smooth, **weights)


@pytest.mark.parametrize("cost_class", (core.FSLCorrelationRatio, core.FSLNormalizedMutualInformation))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("smooth", (0.0, 1.3))
@pytest.mark.parametrize("shape", ((3, 4, 5), (17, 19, 23), (63, 65, 67)))
def test_cpu_fused_cost_keeps_scalar_cost_bits(cost_class, weighted, smooth, shape):
    cost = _cost(cost_class, shape, weighted, smooth)
    matrices = np.repeat(np.eye(4)[None], 4, axis=0)
    matrices[1, :3, 3] = (0.4, -0.6, 0.8)
    matrices[2, :3, :3] = ((1.03, 0.01, -0.02), (-0.01, 0.98, 0.01), (0, 0.02, 1.02))
    matrices[3, :3, 3] = (1000, -1000, 1000)
    cost._cpu_sampling = False
    expected = [cost(matrix) for matrix in matrices]
    cost._cpu_sampling = True
    _assert_bits([cost(matrix) for matrix in matrices], expected)


@pytest.mark.parametrize("shape", ((3, 4, 5), (17, 19, 23)))
def test_cpu_final_sampling_keeps_trilinear_and_background_bits(shape):
    rng = np.random.default_rng(3044)
    moving = torch.from_numpy(rng.normal(20, 30, shape).astype(np.float32))
    coefficients = torch.tensor(((1.03, 0.02, -0.03, 0.4),
                                  (-0.01, 0.98, 0.03, -0.6),
                                  (0.01, -0.02, 1.04, 0.8)), dtype=torch.float32)
    grid = core._voxel_grid(shape, device=torch.device("cpu"))
    coordinates = core._coordinates_from_fsl_coefficients(coefficients, grid)
    upper = torch.tensor(shape)[:, None] - 1
    valid = ((coordinates >= 0) & (coordinates <= upper)).all(0)
    background = core._edge_background(moving)
    expected = torch.where(valid, core._manual_trilinear(moving, coordinates), background).reshape(shape)
    actual = sample_output(moving.numpy(), coefficients.numpy(), shape, np.float32(background))
    _assert_bits(actual, expected.numpy())


def test_cpu_sampling_observes_refreshed_moving_and_taper():
    cost = _cost(core.FSLCorrelationRatio, (17, 19, 23), True, 2.0)
    for smooth in (2.0, 1.0):
        cost.moving = cost.moving * 1.01
        cost.moving_weight = cost.moving_weight * 0.9
        cost.smooth_size = smooth
        cost._cpu_sampling = False
        expected = cost(np.eye(4))
        cost._cpu_sampling = True
        _assert_bits(cost(np.eye(4)), expected)


def test_cpu_path_does_not_change_global_thread_or_precision_policy():
    threads = torch.get_num_threads()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    _cost(core.FSLCorrelationRatio, (7, 8, 9), False, 1.3)(np.eye(4))
    assert torch.get_num_threads() == threads
    assert torch.backends.cuda.matmul.allow_tf32 == matmul
    assert torch.backends.cudnn.allow_tf32 == cudnn


@pytest.mark.parametrize("weighted", (False, True))
def test_cpu_x_fastest_storage_keeps_filtered_voxel_and_cog_bits(weighted):
    rng = np.random.default_rng(4092)
    image = rng.uniform(0, 400, (17, 19, 23)).astype(np.float32)
    sizes = (1.1, 1.9, 2.7)
    kwargs = {}
    if weighted:
        kwargs["moving_weight"] = rng.uniform(0, 1, image.shape).astype(np.float32)
    engine = core._DefaultFLIRTEngine(
        image, image, np.diag((-1, 1, 1, 1)), np.diag((-1, 1, 1, 1)),
        sizes, sizes, device="cpu", angular_search=False, **kwargs,
    )
    reference = torch.from_numpy(core._clamp_like_fsl(image).copy())
    transform = lambda value: core._blur(value, 8.0, sizes)
    if weighted:
        weight = torch.from_numpy(kwargs["moving_weight"].copy())
        expected = core._filter_image_with_weight(reference, weight, transform)
    else:
        expected = transform(reference)
    assert engine.moving_original.stride() == (1, 17, 17 * 19)
    assert engine.level.cost.moving.stride() == engine.level.moving.stride()
    _assert_bits(engine.level.moving.numpy(), expected.numpy())
    np.testing.assert_array_equal(
        engine.level.centre, core._centre_of_gravity(expected, np.diag([*sizes, 1])),
    )


def test_cpu_scalar_preparation_avoids_unused_full_grid_and_sorted_permutation():
    cost = _cost(core.FSLCorrelationRatio, (17, 19, 23), False, 1.3)
    lazy_fields = ("grid", "bin_sort_order", "bin_lengths")
    assert all(name not in cost.__dict__ for name in lazy_fields)
    scalar = cost(np.eye(4))
    assert all(name not in cost.__dict__ for name in lazy_fields)
    batched = BatchedAffineCost(cost)(np.eye(4))
    assert all(name in cost.__dict__ for name in lazy_fields)
    assert abs(float(batched[0]) - scalar) < 2e-6


@pytest.mark.parametrize("weighted", (False, True))
def test_cpu_inside_row_proof_preserves_inclusive_float32_boundaries(weighted):
    from fnit.flirt._cpu import sample_cost
    cost = _cost(core.FSLCorrelationRatio, (7, 8, 9), weighted, 1.3)
    evaluator = BatchedAffineCost(cost)
    upper = np.float32(cost.moving.shape[0] - 1.0001)
    boundaries = (np.float32(0), np.nextafter(np.float32(0), np.float32(-np.inf)),
                  np.nextafter(upper, np.float32(-np.inf)), upper,
                  np.nextafter(upper, np.float32(np.inf)))
    for boundary in boundaries:
        coefficients = torch.zeros((3, 4), dtype=torch.float32)
        coefficients[:, 3] = torch.tensor((float(boundary), 0.75, 0.75))
        expected = evaluator._sample_tensor(coefficients[None])
        values, weights, _valid = sample_cost(*cost._cpu_sample_arguments(coefficients))
        _assert_bits(values, expected[0].numpy()[0])
        _assert_bits(weights, expected[1].numpy()[0])


def test_cpu_nmi_extreme_finite_intensities_fail_cleanly_before_native_indexing():
    image = torch.full((3, 4, 5), float(np.finfo(np.float32).max), dtype=torch.float32)
    cost = core.FSLNormalizedMutualInformation(
        image, image, (1, 1, 1), (1, 1, 1), bins=32, smooth_size=1,
    )
    cost._cpu_sampling = False
    with pytest.raises(RuntimeError, match="out of bounds"):
        cost(np.eye(4))
    cost._cpu_sampling = True
    with pytest.raises(ValueError, match="outside the finite histogram"):
        cost(np.eye(4))


@pytest.mark.parametrize("cost_class", (core.FSLCorrelationRatio, core.FSLNormalizedMutualInformation))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("smooth", (0.0, 1.3))
@pytest.mark.parametrize("shape", ((3, 4, 5), (17, 19, 23), (63, 65, 67)))
def test_cpu_parallel_rows_keep_cost_bits_and_restore_configuration(cost_class, weighted, smooth, shape):
    from numba import config, get_num_threads, set_num_threads

    if config.NUMBA_NUM_THREADS < 2:
        pytest.skip("parallel configuration gate requires a Numba pool of at least two threads")
    previous_torch, previous_numba = torch.get_num_threads(), get_num_threads()
    budget = min(4, config.NUMBA_NUM_THREADS)
    set_num_threads(config.NUMBA_NUM_THREADS)
    torch.set_num_threads(budget)
    try:
        test_cpu_fused_cost_keeps_scalar_cost_bits(cost_class, weighted, smooth, shape)
        assert get_num_threads() == config.NUMBA_NUM_THREADS
        assert torch.get_num_threads() == budget
    finally:
        torch.set_num_threads(previous_torch)
        set_num_threads(previous_numba)


def test_cpu_parallel_budget_restores_numba_mask_on_exception():
    from numba import config, get_num_threads, set_num_threads
    from fnit.flirt._cpu import _dispatch_with_cpu_budget

    if config.NUMBA_NUM_THREADS < 2:
        pytest.skip("exception restoration gate requires a Numba pool of at least two threads")
    previous_torch, previous_numba = torch.get_num_threads(), get_num_threads()
    precision = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    set_num_threads(config.NUMBA_NUM_THREADS)
    torch.set_num_threads(2)
    try:
        def failed_parallel_call():
            assert get_num_threads() == 2
            raise RuntimeError("intentional exception validates restoration")

        with pytest.raises(RuntimeError, match="validates restoration"):
            _dispatch_with_cpu_budget(failed_parallel_call, lambda: None, ())
        assert get_num_threads() == config.NUMBA_NUM_THREADS
        assert torch.get_num_threads() == 2
        assert precision == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    finally:
        torch.set_num_threads(previous_torch)
        set_num_threads(previous_numba)


@pytest.mark.parametrize("storage", ("C", "F", "strided"))
@pytest.mark.parametrize("exceptional", (None, np.nan, np.inf, -np.inf, np.finfo(np.float32).max))
@pytest.mark.parametrize("translation", ((0, 0, 0), (-0.6, 0.3, 0.8),
                                          (1000, -1000, 1000), (np.nan, 0, 0), (np.inf, 0, 0)))
def test_cpu_eight_point_sampling_keeps_exception_and_boundary_stats_bits(storage, exceptional, translation):
    from fnit.flirt import _cpu

    rng = np.random.default_rng(41004)
    image = np.array(rng.normal(20, 50, (17, 19, 23)), dtype=np.float32,
                     order="F" if storage == "F" else "C")
    if storage == "strided":
        backing = np.empty((34, 19, 46), np.float32)
        backing[::2, :, ::2] = image
        image = backing[::2, :, ::2]
    if exceptional is not None:
        image[4, 7, 8] = exceptional
    coefficients = np.eye(4, dtype=np.float32)[:3].copy()
    coefficients[:, 3] = translation
    arguments = (image, coefficients, image.shape,
                 np.asarray(image.shape, dtype=np.float32) - np.float32(1.0001),
                 np.asarray((1, 1.3, 2.2), np.float32), True,
                 np.empty((0, 0, 0), np.float32), np.empty(0, np.float32), False,
                 rng.integers(0, 32, size=image.size, dtype=np.int64), 32)
    reference = _cpu._production_serial_corratio(*arguments)
    optimized = _cpu._sample_corratio_serial(*arguments)
    for left, right in zip(reference[:3], optimized[:3]):
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
    assert reference[3] == optimized[3]


@pytest.mark.parametrize("storage", ("C", "F", "strided"))
@pytest.mark.parametrize("voxel_sizes", ((1, 1, 1), (1.1, 1.9, 2.7)))
@pytest.mark.parametrize("scale", (0.5, 2, 8))
def test_cpu_isotropic_fusion_keeps_original_tensor_backend_bits(storage, voxel_sizes, scale):
    rng = np.random.default_rng(41004)
    array = np.array(rng.normal(20, 100, (17, 19, 23)), dtype=np.float32,
                     order="F" if storage == "F" else "C")
    if storage == "strided":
        backing = np.empty((34, 38, 46), np.float32)
        backing[::2, ::2, ::2] = array
        array = backing[::2, ::2, ::2]
    data = torch.from_numpy(array)
    actual, actual_sizes = core._isotropic_resample(data, voxel_sizes, scale)
    # A gradient-bearing tensor deliberately uses the existing tensor path.
    expected, expected_sizes = core._isotropic_resample(
        data.detach().requires_grad_(), voxel_sizes, scale,
    )
    np.testing.assert_array_equal(actual.numpy().view(np.uint32),
                                  expected.detach().numpy().view(np.uint32))
    assert actual_sizes == expected_sizes
    assert not actual.requires_grad
    assert expected.requires_grad


@pytest.mark.parametrize("dtype,scale,exceptional,shape", (
    (torch.float64, 1, None, (17, 19, 23)),
    (torch.float32, -1, None, (17, 19, 23)),
    (torch.float32, float("inf"), None, (17, 19, 23)),
    (torch.float32, 1, np.nan, (17, 19, 23)),
    (torch.float32, 1, np.inf, (17, 19, 23)),
    (torch.float32, 1, None, (1, 19, 23)),
))
def test_cpu_isotropic_exceptional_inputs_keep_tensor_fallback(monkeypatch, dtype, scale, exceptional, shape):
    from fnit.flirt import _cpu

    def unexpected_fusion(*args):
        raise AssertionError("exceptional input entered fused isotropic sampling")
    monkeypatch.setattr(_cpu, "sample_output", unexpected_fusion)
    data = torch.ones(shape, dtype=dtype)
    if exceptional is not None:
        data[4, 7, 8] = exceptional
    if min(shape) < 2:
        with pytest.raises(IndexError):
            core._isotropic_resample(data, (1, 1, 1), scale)
    elif dtype != torch.float32:
        with pytest.raises(RuntimeError, match="dtypes match"):
            core._isotropic_resample(data, (1, 1, 1), scale)
    else:
        result, sizes = core._isotropic_resample(data, (1, 1, 1), scale)
        assert sizes == (float(scale),) * 3
        assert result.dtype == torch.float32


def test_cpu_isotropic_gradient_input_keeps_autograd(monkeypatch):
    from fnit.flirt import _cpu

    def unexpected_fusion(*args):
        raise AssertionError("gradient input entered NumPy sampling")
    monkeypatch.setattr(_cpu, "sample_output", unexpected_fusion)
    data = torch.arange(17 * 19 * 23, dtype=torch.float32).reshape(17, 19, 23).requires_grad_()
    result, _ = core._isotropic_resample(data, (1.1, 1.9, 2.7), 2)
    result.sum().backward()
    assert data.grad is not None and torch.isfinite(data.grad).all()


@pytest.mark.parametrize("storage", ("C", "F", "strided"))
@pytest.mark.parametrize("weight_mode", ("positive", "signed", "zero", "nan", "inf", "overflow"))
@pytest.mark.parametrize("translation", ((0, 0, 0), (-0.6, 0.3, 0.8),
                                          (1000, -1000, 1000), (np.nan, 0, 0), (np.inf, 0, 0)))
def test_cpu_weighted_eight_point_sampling_keeps_ordered_stats_bits(storage, weight_mode, translation):
    from fnit.flirt import _cpu

    rng = np.random.default_rng(51004)
    shape = (17, 19, 23)
    image = rng.normal(30, 50, shape).astype(np.float32)
    moving_weight = rng.uniform(0.1, 1, shape).astype(np.float32)
    reference_weight = rng.uniform(0.1, 1, image.size).astype(np.float32)
    if weight_mode in ("signed", "zero"):
        moving_weight[::2] *= np.float32(-1) if weight_mode == "signed" else np.float32(0)
        reference_weight[::3] *= np.float32(-1) if weight_mode == "signed" else np.float32(0)
        reference_weight[::7] = np.float32(-0.0)
    elif weight_mode in ("nan", "inf", "overflow"):
        exceptional = {"nan": np.nan, "inf": np.inf, "overflow": np.finfo(np.float32).max}[weight_mode]
        moving_weight[4, 7, 8] = exceptional
        reference_weight[::17] = exceptional
    if storage == "strided":
        backing = np.empty((34, 19, 46), np.float32)
        backing[::2, :, ::2] = image
        image = backing[::2, :, ::2]
        weight_backing = np.empty((17, 38, 46), np.float32)
        weight_backing[:, ::2, ::2] = moving_weight
        moving_weight = weight_backing[:, ::2, ::2]
        reference_backing = np.empty(reference_weight.size * 2, np.float32)
        reference_backing[::2] = reference_weight
        reference_weight = reference_backing[::2]
    else:
        image = np.array(image, order=storage)
        moving_weight = np.array(moving_weight, order=storage)
    coefficients = np.eye(4, dtype=np.float32)[:3].copy()
    coefficients[:, 3] = translation
    arguments = (image, coefficients, shape,
                 np.asarray(shape, dtype=np.float32) - np.float32(1.0001),
                 np.asarray((1, 1.3, 2.2), np.float32), True,
                 moving_weight, reference_weight, True,
                 rng.integers(0, 32, size=image.size, dtype=np.int64), 32)
    expected = _cpu._production_serial_corratio(*arguments)
    actual = _cpu._sample_corratio_serial(*arguments)
    for left, right in zip(expected[:3], actual[:3]):
        np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
    assert expected[3] == actual[3]


@pytest.mark.parametrize("size", (4096, 4097))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("taper", (False, True))
def test_cpu_large_axis_endpoint_neighbours_are_clipped_safely(size, weighted, taper):
    from fnit.flirt import _cpu

    # At these sizes float32(size - 1.0001) rounds to the last voxel.
    # A linear intensity ramp supplies an independent endpoint value; do not
    # execute the old unchecked floor+1 sampler as a reference.
    image = np.array(np.broadcast_to(np.arange(size, dtype=np.float32)[:, None, None],
                                    (size, 3, 3)), order="F")
    reference_shape = (8, 2, 2)
    upper = np.asarray(image.shape, dtype=np.float32) - np.float32(1.0001)
    assert upper[0] == size - 1
    moving_weight = np.ones_like(image) if weighted else np.empty((0, 0, 0), np.float32)
    reference_weight = np.ones(np.prod(reference_shape), np.float32) if weighted else np.empty(0, np.float32)
    for x in (np.float32(size - 1), np.float32(size - 1.25)):
        coefficients = np.zeros((3, 4), np.float32)
        coefficients[:, 3] = (x, 1, 1)
        arguments = (image, coefficients, reference_shape, upper, np.ones(3, np.float32), taper,
                     moving_weight, reference_weight, weighted)
        np.testing.assert_equal(_cpu._trilinear_inside(image, x, np.float32(1), np.float32(1)), x)
        values = np.full(np.prod(reference_shape), x, np.float32)
        if taper:
            axis_weights = [_cpu._taper(coordinates, limit, np.float32(1))
                            for coordinates, limit in zip((x, np.float32(1), np.float32(1)), upper)]
            weight = np.float32(np.float32(axis_weights[0] * axis_weights[1]) * axis_weights[2])
        else:
            weight = np.float32(1)
        weights = np.full(values.size, weight, np.float32)
        for sampler in (_cpu._sample_cost_serial, _cpu._sample_cost_parallel):
            actual_values, actual_weights, valid = sampler(*arguments)
            np.testing.assert_array_equal(actual_values.view(np.uint32), values.view(np.uint32))
            np.testing.assert_array_equal(actual_weights.view(np.uint32), weights.view(np.uint32))
            assert valid
        bins = np.zeros(values.size, np.int64)
        expected = _cpu.reduce_corratio(values, weights, bins, 1)
        for sampler in (_cpu._sample_corratio_serial, _cpu._sample_corratio_simd_taper):
            actual = sampler(*arguments, bins, 1)
            for left, right in zip(expected, actual[:3]):
                np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
            assert actual[3]
