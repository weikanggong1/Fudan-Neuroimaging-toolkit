"""Fused sampling must preserve tensor execution; these are not benchmarks."""

from importlib.util import find_spec

import numpy as np
import pytest
import torch

from fnit.flirt.batched import BatchedAffineCost
from fnit.flirt.core import FSLCorrelationRatio, FSLNormalizedMutualInformation


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec("triton") is None,
    reason="CUDA and Triton are required for fused sampling",
)
_COSTS = (FSLCorrelationRatio, FSLNormalizedMutualInformation)


def _evaluator(cost_class, shape=(17, 19, 23), *, smooth_size=1.3, weighted=False):
    generator = torch.Generator().manual_seed(41001)
    reference = (torch.rand(shape, generator=generator) * 900 - 80).cuda()
    moving = (torch.rand(shape, generator=generator) * 750 - 120).cuda()
    weights = {}
    if weighted:
        reference_weight = torch.rand(shape, generator=generator).cuda()
        reference_weight[::3, ::2, :] = 0
        moving_weight = torch.rand(shape, generator=generator).cuda()
        weights = dict(reference_weight=reference_weight, moving_weight=moving_weight)
    cost = cost_class(reference, moving, (1.1, 1.9, 2.7), (1.7, 1.3, 2.2),
                      bins=32, smooth_size=smooth_size, **weights)
    return BatchedAffineCost(cost, max_batch_size=8)


def _matrices():
    matrices = np.repeat(np.eye(4)[None], 6, axis=0)
    matrices[1, :3, 3] = (0.4, -0.6, 0.8)
    matrices[2, :3, :3] = ((1.03, 0.01, -0.02), (-0.01, 0.98, 0.01), (0, 0.02, 1.02))
    matrices[3, :3, 3] = (-2.3, 1.7, -0.9)
    matrices[4, :3, 3] = (1000, -1000, 1000)
    matrices[5] = matrices[1]
    return matrices


def _assert_sample_bits(actual, expected):
    for result, reference in zip(actual, expected):
        result = result.detach().cpu().numpy()
        reference = reference.detach().cpu().numpy()
        # Invalid candidates and zero-width tapering can carry NaNs internally.
        # Their positions must match; the payload of a NaN is not a cost value.
        np.testing.assert_array_equal(np.isnan(result), np.isnan(reference))
        selected = ~np.isnan(reference)
        np.testing.assert_array_equal(result[selected].view(np.uint32),
                                      reference[selected].view(np.uint32))


@pytest.mark.parametrize("cost_class", _COSTS)
@pytest.mark.parametrize("smooth_size", (0.0, 1.3))
@pytest.mark.parametrize("shape", ((3, 4, 5), (17, 19, 23), (91, 109, 91)))
def test_fused_sampling_keeps_tensor_bits_for_affines_and_odd_grids(
    cost_class, smooth_size, shape,
):
    evaluator = _evaluator(cost_class, shape, smooth_size=smooth_size)
    matrices = _matrices()[:3] if shape[0] > 90 else _matrices()
    coefficients, _ = evaluator._coefficients(matrices)
    expected = evaluator._sample_tensor(coefficients)

    actual = evaluator._sample(coefficients)

    assert evaluator._cuda_sample_kernel is not None
    _assert_sample_bits(actual, expected)


@pytest.mark.parametrize("cost_class", _COSTS)
@pytest.mark.parametrize("smooth_size", (0.0, 1.3))
def test_fused_sampling_matches_taper_and_inclusive_fov_boundaries(cost_class, smooth_size):
    evaluator = _evaluator(cost_class, (7, 8, 9), smooth_size=smooth_size)
    upper = evaluator.upper[0, 0, 0]
    positive = torch.tensor(float("inf"), device="cuda")
    negative = -positive
    x = torch.stack((torch.tensor(0.0, device="cuda"),
                     torch.nextafter(torch.tensor(0.0, device="cuda"), negative),
                     torch.nextafter(torch.tensor(0.0, device="cuda"), positive),
                     torch.nextafter(upper, negative), upper,
                     torch.nextafter(upper, positive), evaluator.smooth[0, 0, 0]))
    # Constant coordinates isolate exact boundary decisions from affine search.
    coefficients = torch.zeros((len(x), 3, 4), device="cuda")
    coefficients[:, 0, 3] = x
    coefficients[:, 1:, 3] = 0.75

    _assert_sample_bits(evaluator._sample(coefficients), evaluator._sample_tensor(coefficients))


@pytest.mark.parametrize("cost_class", _COSTS)
def test_weighted_cost_keeps_tensor_sampling_fallback(cost_class):
    evaluator = _evaluator(cost_class, weighted=True)
    coefficients, _ = evaluator._coefficients(_matrices())

    class ForbiddenKernel:
        def __getitem__(self, _grid):
            raise AssertionError("weighted cost must use tensor sampling")

    evaluator._cuda_sample_imported = True
    evaluator._cuda_sample_kernel = ForbiddenKernel()

    _assert_sample_bits(evaluator._sample(coefficients), evaluator._sample_tensor(coefficients))


@pytest.mark.parametrize("cost_class,sentinel", (
    (FSLCorrelationRatio, 1.0), (FSLNormalizedMutualInformation, -1.0),
))
def test_fused_sampling_preserves_singular_lane_isolation(cost_class, sentinel):
    evaluator = _evaluator(cost_class)
    matrices = torch.eye(4, dtype=torch.float64, device="cuda").repeat(3, 1, 1)
    matrices[1, 2, 2] = 0
    coefficients, invertible = evaluator._coefficients(matrices)
    assert invertible.tolist() == [True, False, True]
    _assert_sample_bits(evaluator._sample(coefficients), evaluator._sample_tensor(coefficients))

    actual = evaluator(matrices).cpu().numpy()
    expected = np.float32(evaluator.cost(np.eye(4)))
    assert actual[1] == sentinel
    np.testing.assert_array_equal(actual[[0, 2]].view(np.uint32),
                                  np.array([expected, expected]).view(np.uint32))


@pytest.mark.parametrize("cost_class", _COSTS)
@pytest.mark.parametrize("weighted", (False, True))
def test_fused_sampling_cost_and_chunks_keep_scalar_reference_bits(cost_class, weighted):
    evaluator = _evaluator(cost_class, weighted=weighted)
    matrices = _matrices()
    expected = np.array([evaluator.cost(matrix) for matrix in matrices], dtype=np.float32)
    for chunk_size in (1, 3, 8):
        actual = evaluator(matrices, chunk_size=chunk_size).cpu().numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    evaluator._cuda_sample_imported = True
    evaluator._cuda_sample_kernel = None
    fallback = evaluator(matrices, chunk_size=3).cpu().numpy()
    np.testing.assert_array_equal(fallback.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("allow_tf32", (False, True))
def test_fused_sampling_keeps_callers_tf32_policy(allow_tf32):
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = not allow_tf32
        evaluator = _evaluator(FSLCorrelationRatio)
        coefficients, _ = evaluator._coefficients(_matrices())
        _assert_sample_bits(evaluator._sample(coefficients), evaluator._sample_tensor(coefficients))
        assert torch.backends.cuda.matmul.allow_tf32 is allow_tf32
        assert torch.backends.cudnn.allow_tf32 is not allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
