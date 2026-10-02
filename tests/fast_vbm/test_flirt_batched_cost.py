"""Execution parity for FLIRT candidate batching; these are not benchmarks."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from fnit.flirt.batched import BatchedAffineCost
from fnit.flirt.affine_batch import fsl_affine_from_parameters_batch
from fnit.flirt.core import (
    FSLCorrelationRatio,
    FSLNormalizedMutualInformation,
    _DefaultFLIRTEngine,
    _find_cost_minima,
    _fsl_pull_coefficients,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
)
from fnit.flirt.search import coordinate_trials, evaluate_trials, map_trials


_COST_CLASSES = (FSLCorrelationRatio, FSLNormalizedMutualInformation)
_DEVICES = ("cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable")))


def _images(device, *, constant=False):
    coordinates = np.indices((11, 12, 13), dtype=np.float32)
    x, y, z = coordinates
    reference = 17 + 3 * x + 5 * y + z + 9 * np.sin(x / 2 + z / 3)
    moving = 8 + x + 2 * y + 4 * z + 12 * np.cos(y / 3 - z / 4)
    if constant:
        reference = np.ones_like(x)
        moving = np.ones_like(x)
    return (torch.as_tensor(reference, device=device),
            torch.as_tensor(moving, device=device))


def _cost(cost_class, device, *, smooth_size=1.3, weighted=False, constant=False):
    reference, moving = _images(device, constant=constant)
    weights = {}
    if weighted:
        reference_weight = torch.ones_like(reference)
        reference_weight[2:6, 3:7, :] = 0
        moving_weight = torch.linspace(0, 1, moving.numel(), device=device)
        weights = {"reference_weight": reference_weight,
                   "moving_weight": moving_weight.reshape(moving.shape)}
    return cost_class(reference, moving, (1.1, 1.9, 2.7), (1.7, 1.3, 2.2),
                      bins=12, smooth_size=smooth_size, **weights)


def _candidates():
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], 11, axis=0)
    matrices[1, :3, 3] = (0.4, -0.6, 0.8)
    matrices[2, :3, :3] = ((1.03, .01, -.02), (-.01, .98, .01), (0, .02, 1.02))
    matrices[3, 0, 3] = -1.7
    matrices[4, 0, 3] = -1.7 * .9999
    matrices[5, 0, 3] = -1.7 * 1.0001
    matrices[6, 1, 3] = 1.3 * .5
    matrices[7, :3, 3] = (1000, -1000, 1000)
    matrices[8] = matrices[1]
    matrices[9] = matrices[2]
    matrices[10, :3, 3] = (-2.3, 1.7, -.9)
    return matrices


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cost_class", _COST_CLASSES)
@pytest.mark.parametrize("smooth_size", (0.0, 1.3))
def test_batched_cost_matches_serial_with_anisotropic_and_boundary_samples(
    device, cost_class, smooth_size,
):
    reference_cost = _cost(cost_class, device, smooth_size=smooth_size)
    matrices = _candidates()
    expected = np.asarray([reference_cost(matrix) for matrix in matrices])

    result = BatchedAffineCost(reference_cost, max_batch_size=3)(matrices)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (len(matrices),)
    assert result.device.type == device
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-6, rtol=0)
    assert result[1] == result[8]
    assert result[2] == result[9]


@pytest.mark.parametrize("device", _DEVICES)
def test_batched_correlation_ratio_preserves_soft_image_weights(device):
    reference_cost = _cost(FSLCorrelationRatio, device, weighted=True)
    matrices = _candidates()
    expected = [reference_cost(matrix) for matrix in matrices]

    result = BatchedAffineCost(reference_cost, max_batch_size=4).evaluate(matrices)

    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-6, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cost_class", _COST_CLASSES)
def test_candidate_chunk_size_preserves_cost_order_and_first_minimum(device, cost_class):
    reference_cost = _cost(cost_class, device)
    matrices = _candidates()
    matrices = matrices[[7, 2, 1, 0, 1, 2, 0, 7, 3, 4, 5]]
    expected = np.asarray([reference_cost(matrix) for matrix in matrices])
    results = [BatchedAffineCost(reference_cost, max_batch_size=size)(matrices)
               for size in (1, 3, 32)]

    for result in results:
        np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-6, rtol=0)
        assert int(result.argmin()) == int(expected.argmin())
    # Equal-cost candidates must retain the original order across chunks.
    assert results[-1][2] == results[-1][4]
    assert results[-1][3] == results[-1][6]
    assert results[-1][1] == results[-1][5]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cost_class,sentinel", (
    (FSLCorrelationRatio, 1.0), (FSLNormalizedMutualInformation, -1.0),
))
def test_invalid_overlap_and_constant_images_keep_reference_sentinel(
    device, cost_class, sentinel,
):
    reference_cost = _cost(cost_class, device, constant=True)
    matrices = _candidates()[[0, 7]]

    result = BatchedAffineCost(reference_cost)(matrices)

    constant_cost = 0.0 if cost_class is FSLNormalizedMutualInformation else sentinel
    assert [reference_cost(matrix) for matrix in matrices] == [constant_cost, sentinel]
    np.testing.assert_array_equal(result.cpu().numpy(), [constant_cost, sentinel])


@pytest.mark.parametrize("device", _DEVICES)
def test_correlation_ratio_rejects_bins_with_only_two_samples(device):
    reference = torch.arange(64, dtype=torch.float32, device=device).reshape(4, 4, 4)
    reference_cost = FSLCorrelationRatio(reference, reference * 2,
                                         (1, 1, 1), (1, 1, 1),
                                         bins=32, smooth_size=0)

    result = BatchedAffineCost(reference_cost)(np.eye(4))

    assert reference_cost(np.eye(4)) == 1.
    assert float(result[0]) == 1.


@pytest.mark.parametrize("device", _DEVICES)
def test_single_and_empty_candidate_batches(device):
    reference_cost = _cost(FSLCorrelationRatio, device)
    evaluator = BatchedAffineCost(reference_cost)

    single = evaluator(np.eye(4))
    empty = evaluator(np.empty((0, 4, 4), dtype=np.float64))

    assert single.shape == (1,)
    assert empty.shape == (0,)
    assert empty.device == single.device == reference_cost.device
    assert abs(float(single[0]) - reference_cost(np.eye(4))) < 2e-6


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("cost_class", _COST_CLASSES)
def test_refreshed_moving_image_and_smoothing_do_not_reuse_stale_cache(device, cost_class):
    reference_cost = _cost(cost_class, device)
    evaluator = BatchedAffineCost(reference_cost, max_batch_size=4)
    matrices = _candidates()[:7]
    before = evaluator(matrices).clone()
    reference_cost.moving = reference_cost.moving.flip(0).contiguous()
    reference_cost.smooth_size = .4

    result = evaluator(matrices)
    expected = [reference_cost(matrix) for matrix in matrices]

    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2e-6, rtol=0)
    assert not torch.allclose(result, before)


@pytest.mark.parametrize("device", _DEVICES)
def test_refreshed_moving_weights_do_not_reuse_stale_cache(device):
    reference_cost = _cost(FSLCorrelationRatio, device, weighted=True)
    evaluator = BatchedAffineCost(reference_cost)
    matrices = _candidates()[:3]
    evaluator(matrices)
    reference_cost.moving_weight = reference_cost.moving_weight.flip(1).contiguous()

    result = evaluator(matrices)

    np.testing.assert_allclose(result.cpu().numpy(),
                               [reference_cost(matrix) for matrix in matrices],
                               atol=2e-6, rtol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_small_memory_budget_changes_chunking_without_discarding_candidates(device):
    reference_cost = _cost(FSLCorrelationRatio, device)
    matrices = _candidates()
    result = BatchedAffineCost(reference_cost, memory_budget_gb=1e-6)(matrices)

    assert len(result) == len(matrices)
    np.testing.assert_allclose(result.cpu().numpy(),
                               [reference_cost(matrix) for matrix in matrices],
                               atol=2e-6, rtol=0)


def test_invalid_cpu_candidate_matrix_is_rejected_before_cost_evaluation():
    evaluator = BatchedAffineCost(_cost(FSLCorrelationRatio, "cpu"))
    with pytest.raises(ValueError):
        evaluator(np.eye(3))
    invalid = np.eye(4)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluator(invalid)
    singular = np.eye(4)
    singular[2, 2] = 0
    with pytest.raises(np.linalg.LinAlgError):
        evaluator(singular)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("cost_class,sentinel", (
    (FSLCorrelationRatio, 1.0), (FSLNormalizedMutualInformation, -1.0),
))
def test_device_resident_singular_lane_does_not_poison_other_candidates(cost_class, sentinel):
    reference_cost = _cost(cost_class, "cuda")
    matrices = torch.eye(4, dtype=torch.float64, device="cuda").repeat(3, 1, 1)
    matrices[1, 2, 2] = 0
    expected = reference_cost(np.eye(4))

    result = BatchedAffineCost(reference_cost)(matrices)

    np.testing.assert_allclose(result[[0, 2]].cpu().numpy(), expected, atol=2e-6, rtol=0)
    assert float(result[1]) == sentinel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("cost_class", _COST_CLASSES)
def test_device_resident_affines_match_cpu_inverse_and_float_assignment(cost_class):
    reference_cost = _cost(cost_class, "cuda")
    matrices = _candidates()
    evaluator = BatchedAffineCost(reference_cost)

    expected = evaluator(matrices)
    actual = evaluator(torch.as_tensor(matrices, device="cuda", dtype=torch.float64))

    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(),
                               atol=2e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_batched_correlation_ratio_is_bitwise_serial():
    reference_cost = _cost(FSLCorrelationRatio, "cuda")
    matrices = _candidates()
    expected = np.asarray([reference_cost(matrix) for matrix in matrices], dtype=np.float32)

    actual = BatchedAffineCost(reference_cost, max_batch_size=3)(matrices)

    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compact_sum_preserves_scalar_grouping_with_and_without_fusion():
    evaluator = BatchedAffineCost(_cost(FSLCorrelationRatio, "cuda"))
    lengths_cpu = (0, 1, 2, 9, 12, 32, 64, 97, 128, 129, 239, 256)
    lengths = torch.tensor(lengths_cpu, dtype=torch.long, device="cuda")
    generator = torch.Generator().manual_seed(301)
    packed = torch.randn((len(lengths_cpu), 4, 256), generator=generator)
    packed *= torch.logspace(-3, 3, 256)[None, None]
    for row, length in enumerate(lengths_cpu):
        packed[row, :, length:] = 0
    packed = packed.cuda()
    expected = torch.stack([torch.stack([values[:length].sum() for values in row])
                            for row, length in zip(packed, lengths_cpu)])

    fused = evaluator._compact_sum(packed, lengths)
    evaluator._cuda_sum_imported = True
    evaluator._cuda_sum_kernel = None
    fallback = evaluator._compact_sum(packed, lengths)

    np.testing.assert_array_equal(fused.cpu().numpy().view(np.uint32),
                                  expected.cpu().numpy().view(np.uint32))
    np.testing.assert_array_equal(fallback.cpu().numpy().view(np.uint32),
                                  expected.cpu().numpy().view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bins", (32, 128, 256))
def test_cuda_correlation_ratio_without_triton_keeps_scalar_sum_bits(bins):
    reference, moving = _images("cuda")
    reference_cost = FSLCorrelationRatio(reference, moving, (1, 1, 1), (1, 1, 1),
                                         bins=bins)
    evaluator = BatchedAffineCost(reference_cost)
    evaluator._cuda_sum_imported = True
    evaluator._cuda_sum_kernel = None
    lengths_cpu = sorted({0, 1, 2, 3, 7, 8, 9, 12, 15, 16, 17, 31, 32,
                          63, 64, 65, 97, 127, 128, 129, 130, 131, 132,
                          239, 255, 256} & set(range(bins + 1)))
    lengths = torch.tensor(lengths_cpu, dtype=torch.long, device="cuda")
    generator = torch.Generator().manual_seed(130930 + bins)
    packed = torch.randn((len(lengths_cpu), 4, bins), generator=generator)
    packed *= torch.logspace(-3, 3, bins)[None, None]
    for row, length in enumerate(lengths_cpu):
        packed[row, :, length:] = 0
    packed = packed.cuda()
    # The reference indexes kept bins into a fresh aligned allocation before
    # sum(); packed rows must not inherit an odd-stride alignment as an oracle.
    expected = torch.stack([torch.stack([values[:length].clone().sum() for values in row])
                            for row, length in zip(packed, lengths_cpu)])

    actual = evaluator._compact_sum(packed, lengths)

    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32),
                                  expected.cpu().numpy().view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bins", (32, 128))
def test_cuda_large_odd_grid_preserves_scalar_segment_alignment(bins):
    # Each candidate has 902629 points, so unpadded batch rows move the next
    # CUB segment by one float relative to the serial aligned allocation.
    generator = torch.Generator().manual_seed(230930)
    reference = torch.rand((91, 109, 91), generator=generator).cuda() * 100
    moving = torch.rand((91, 109, 91), generator=generator).cuda() * 100
    reference_cost = FSLCorrelationRatio(reference, moving, (2, 2, 2), (2, 2, 2),
                                         bins=bins, smooth_size=1)
    matrices = _candidates()[[0, 0, 0, 0, 1, 2, 6, 10]]
    expected = np.asarray([reference_cost(matrix) for matrix in matrices], dtype=np.float32)
    evaluator = BatchedAffineCost(reference_cost, max_batch_size=8)

    for chunk_size in (1, 3, 8):
        actual = evaluator(matrices, chunk_size=chunk_size).cpu().numpy()

        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("bins", (32, 128, 256))
def test_cuda_small_odd_weighted_grid_without_triton_is_bitwise_serial(bins):
    generator = torch.Generator().manual_seed(330930)
    shape = (17, 19, 23)
    reference = torch.rand(shape, generator=generator).cuda() * 100
    moving = torch.rand(shape, generator=generator).cuda() * 100
    reference_weight = torch.rand(shape, generator=generator).cuda()
    moving_weight = torch.rand(shape, generator=generator).cuda()
    reference_weight[::3, ::2, :] = 0
    reference_cost = FSLCorrelationRatio(
        reference, moving, (1.1, 1.9, 2.7), (1.7, 1.3, 2.2), bins=bins,
        smooth_size=1.3, reference_weight=reference_weight, moving_weight=moving_weight,
    )
    matrices = _candidates()
    expected = np.asarray([reference_cost(matrix) for matrix in matrices], dtype=np.float32)
    evaluator = BatchedAffineCost(reference_cost, max_batch_size=8)
    evaluator._cuda_sum_imported = True
    evaluator._cuda_sum_kernel = None

    for chunk_size in (1, 3, 8):
        actual = evaluator(matrices, chunk_size=chunk_size).cpu().numpy()

        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("device", _DEVICES)
def test_batched_correlation_ratio_retains_independent_fsl_cost_fixture(device):
    # This affine and cost are an existing FSL 2111.2 oracle, not calculated
    # from the batching implementation or from the serial cost during test.
    shape = (16, 17, 18)
    coordinates = np.indices(shape, dtype=np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    reference = np.exp(-sum((coordinates[axis] - centre[axis]) ** 2
                           for axis in range(3)) / 9).astype(np.float32)
    moving = np.roll(reference, 1, axis=0)
    vox2world = np.array([[-2, 0, 0, 30], [0, 2, 0, -10],
                         [0, 0, 2, 5], [0, 0, 0, 1]], dtype=np.float64)
    engine = _DefaultFLIRTEngine(moving, reference, vox2world, vox2world,
                                (2, 2, 2), (2, 2, 2), device=device,
                                angular_search=False)
    engine.set_scale(2)
    matrix = np.array([
        [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
        [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
        [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
        [0, 0, 0, 1],
    ])

    result = BatchedAffineCost(engine.level.cost)(matrix)

    assert abs(float(result[0]) - 0.356834) <= 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("allow_tf32", (False, True))
def test_concurrent_evaluation_keeps_callers_precision_policy(allow_tf32):
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = not allow_tf32
        evaluators = [BatchedAffineCost(_cost(cls, "cuda"), max_batch_size=3)
                      for cls in _COST_CLASSES]
        matrices = _candidates()
        expected = [evaluator(matrices).cpu().numpy() for evaluator in evaluators]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda evaluator: evaluator(matrices), evaluators))

        for actual, baseline in zip(results, expected):
            np.testing.assert_array_equal(actual.cpu().numpy(), baseline)
        assert torch.backends.cuda.matmul.allow_tf32 is allow_tf32
        assert torch.backends.cudnn.allow_tf32 is not allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_local_minimum_plateau_uses_fsl_scan_order_and_excludes_upper_faces():
    costs = np.ones((3, 3, 3), dtype=np.float32)

    assert _find_cost_minima(costs) == [
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ]


def test_equal_cost_sort_keeps_candidate_insertion_order():
    candidates = [(0.1, "first"), (0.2, "last"), (0.1, "second")]

    assert _DefaultFLIRTEngine._sort(candidates) == [
        (0.1, "first"), (0.1, "second"), (0.2, "last"),
    ]


def test_cooperative_coordinate_search_preserves_each_reference_trace_and_result():
    starts = [np.array([4., -3., 19.]), np.zeros(3),
              np.array([-2., 1., 7.]), np.array([3., -1., .5])]
    targets = [np.array([1.25, -.75, -11.]), np.zeros(3),
               np.zeros(3), np.array([.7, 1.3, -2.])]
    optimized_dimensions = (2, 3, 1, 3)
    tolerance = np.array([.01, .02, .03])

    def objective(lane, point):
        if lane == 2:
            return .125  # Exact plateau checks strict line-search ties.
        difference = point - targets[lane]
        if lane == 3:
            return float(difference @ difference + .2 * (difference.sum()) ** 2)
        return float(difference @ difference)

    expected, reference_traces = [], [[] for _ in starts]
    for lane, start in enumerate(starts):
        def scalar(point, lane=lane):
            reference_traces[lane].append(point.copy())
            return objective(lane, point)

        expected.append(fsl_coordinate_optimize(
            start, tolerance, scalar, numopt=optimized_dimensions[lane],
            maximum_iterations=4, bound_guess=(10., 1.),
        ))

    actual_traces, waves = [[] for _ in starts], []

    def evaluate_many(candidates):
        waves.append(len(candidates))
        values = []
        for lane, point in candidates:
            actual_traces[lane].append(point.copy())
            values.append(objective(lane, point))
        return values

    trials = [map_trials(coordinate_trials(
        start, tolerance, numopt=optimized_dimensions[lane],
        maximum_iterations=4, bound_guess=(10., 1.),
    ), lambda point, lane=lane: (lane, point)) for lane, start in enumerate(starts)]
    actual = evaluate_trials(trials, evaluate_many)

    for lane, ((point, cost), (expected_point, expected_cost)) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(point, expected_point)
        assert cost == expected_cost
        np.testing.assert_array_equal(actual_traces[lane], reference_traces[lane])
    assert len({len(trace) for trace in actual_traces}) > 1
    assert max(waves) == len(starts)
    assert len(waves) < sum(len(trace) for trace in actual_traces)


def test_cooperative_search_empty_and_already_completed_lanes():
    def unexpected_evaluation(candidates):
        raise AssertionError("a completed search must not evaluate another cost")

    assert evaluate_trials([], unexpected_evaluation) == []
    results = evaluate_trials([coordinate_trials(
        [1., 2.], [.1, .1], maximum_iterations=0,
    )], unexpected_evaluation)
    np.testing.assert_array_equal(results[0][0], [1., 2.])
    assert results[0][1] == 0.


def _affine_parameters(count):
    generator = np.random.default_rng(230930)
    parameters = generator.uniform(-1.5, 1.5, (count, 12))
    parameters[:, 3:6] *= 12
    parameters[:, 6:9] = generator.uniform(.7, 1.3, (count, 3))
    parameters[:, 9:] *= .1
    if count >= 10:
        tiny = np.float32(1e-8)
        parameters[:10, :3] = [
            (0, 0, 0), (-0., 0., -0.),
            (np.pi / 2, -np.pi / 2, np.pi),
            (tiny, -tiny, tiny),
            (np.nextafter(tiny, np.float32(0)), tiny, 0),
            (np.nextafter(tiny, np.float32(np.inf)), 0, -tiny),
            (1e-12, -1e-40, -np.pi),
            (-1e-300, 1e-300, 1e-12),
            (-.72, 0, 0), (0, -.21, 0),
        ]
    return parameters


@pytest.mark.parametrize("dof", (6, 7, 9, 10, 11, 12))
@pytest.mark.parametrize("centre", ((0, 0, 0), (23.717283181, -14.315853159, 28.816265721)))
def test_cpu_batched_affine_composition_is_bitwise_reference_compatible(dof, centre):
    parameters = torch.as_tensor(_affine_parameters(64), dtype=torch.float64)
    centre = torch.as_tensor(centre, dtype=torch.float64)
    expected = torch.stack([fsl_affine_from_parameters(row, centre, dof)
                            for row in parameters])

    actual = fsl_affine_from_parameters_batch(parameters, centre, dof)

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float64
    np.testing.assert_array_equal(actual.view(torch.int64).numpy(),
                                  expected.view(torch.int64).numpy())


def test_cpu_batched_affine_composition_preserves_fsl_float_pull_coefficients():
    parameters = torch.as_tensor(_affine_parameters(1331), dtype=torch.float64)
    centre = torch.tensor([17.736745637, 23.387923258, 11.765926821], dtype=torch.float64)
    actual = fsl_affine_from_parameters_batch(parameters, centre, 12)
    reference = [fsl_affine_from_parameters(row, centre, 12) for row in parameters]
    moving_sizes, reference_sizes = (1.7, 1.3, 2.2), (1.1, 1.9, 2.7)
    expected_coefficients = torch.stack([
        _fsl_pull_coefficients(matrix, moving_sizes, reference_sizes, device="cpu")
        for matrix in reference
    ])
    actual_coefficients = torch.stack([
        _fsl_pull_coefficients(matrix, moving_sizes, reference_sizes, device="cpu")
        for matrix in actual
    ])

    np.testing.assert_array_equal(actual.view(torch.int64).numpy(),
                                  torch.stack(reference).view(torch.int64).numpy())
    np.testing.assert_array_equal(actual_coefficients.view(torch.int32).numpy(),
                                  expected_coefficients.view(torch.int32).numpy())


def test_cpu_batched_affine_empty_shape_and_invalid_inputs():
    empty = fsl_affine_from_parameters_batch(np.empty((0, 12)), [0, 0, 0])
    assert empty.shape == (0, 4, 4)
    with pytest.raises(ValueError, match="shapes"):
        fsl_affine_from_parameters_batch(np.zeros(12), [0, 0, 0])
    with pytest.raises(ValueError, match="shapes"):
        fsl_affine_from_parameters_batch(np.zeros((1, 12)), [0, 0])
    with pytest.raises(ValueError, match="dof"):
        fsl_affine_from_parameters_batch(np.zeros((1, 12)), [0, 0, 0], dof=8)
