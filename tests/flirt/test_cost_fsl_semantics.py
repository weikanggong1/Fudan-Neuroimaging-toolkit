"""Small source-derived cost oracles; these fixtures are not benchmarks."""

import numpy as np
import pytest
import torch

from fnit.flirt.batched import BatchedAffineCost
from fnit.flirt.core import FSLCorrelationRatio, FSLNormalizedMutualInformation


_DEVICES = ("cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable")))


def _evaluate_both(cost, matrix):
    return (np.float32(cost(matrix)),
            BatchedAffineCost(cost)(matrix).detach().cpu().numpy()[0])


@pytest.mark.parametrize("device", _DEVICES)
def test_reference_bin_assignment_uses_source_offset_rounding(device):
    # costfns.cc:3593-3594 computes b0=(-min*bins)/range separately from b1.
    # Computing min*(bins/range) instead changes this exact bin-1 boundary.
    values = torch.tensor([-88.73994445800781, -22.41620635986328,
                           2033.61962890625], device=device)
    reference = values[:, None, None].expand(3, 2, 2).contiguous()
    cost = FSLCorrelationRatio(
        reference, reference, (1, 1, 1), (1, 1, 1), bins=32,
    )
    expected = np.tile([0, 1, 31], 4)
    np.testing.assert_array_equal(cost.bin_index.cpu().numpy(), expected)


@pytest.mark.parametrize("device", _DEVICES)
def test_nmi_bin_assignment_uses_source_multiply_then_offset(device, monkeypatch):
    # Official val*b1+b0 is exactly bin 1 here; (val-min)*b1 is 0.99999982.
    # The source fuzzy histogram must therefore contain exactly 32 per bin.
    reference = torch.ones((4, 4, 4), device=device)
    moving = torch.full((5, 5, 5), -214.9066619873047, device=device)
    moving[4, 4, 4] = -239.16001892089844
    moving[4, 4, 3] = 536.9475708007812
    cost = FSLNormalizedMutualInformation(
        reference, moving, (1, 1, 1), (1, 1, 1), bins=32, smooth_size=0,
    )
    to_float = torch.Tensor.float
    histograms = []

    def observe_completed_histogram(tensor, *args, **kwargs):
        # Inspect the complete FP64 histogram at the FP32 entropy boundary.
        # CPU now accumulates with Numba, so counting scatter_add_ calls would
        # skip that backend. Kernel/observer replay may also change call counts.
        if tensor.dtype == torch.float64 and tensor.numel() == 33 * 33:
            histograms.append(tensor.detach().cpu().numpy().copy().reshape(-1))
        return to_float(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "float", observe_completed_histogram)
    expected = np.zeros(33 * 33, dtype=np.float64)
    expected[:2] = 32
    for evaluator in (cost, BatchedAffineCost(cost)):
        histograms.clear()
        evaluator(np.eye(4))
        assert histograms, "each evaluator must expose its completed histogram"
        for histogram in histograms:
            np.testing.assert_array_equal(histogram, expected)


@pytest.mark.parametrize("device", _DEVICES)
def test_nmi_maximum_intensity_stays_in_last_bin(device):
    # Official costfns.cc:3181-3193 forms neighbours before clamping centre.
    # Four reference groups have distinct moving intensities.  At the maximum,
    # raw bin 8 sends both centre and minus weights to bin 7, so NMI is 2.
    reference = torch.arange(4, dtype=torch.float32, device=device)[:, None, None]
    reference = reference.expand(4, 4, 4).contiguous()
    moving = torch.tensor([0, 1, 0.5625, 0.8125, 0], device=device)[:, None, None]
    moving = moving.expand(5, 5, 5).contiguous()
    cost = FSLNormalizedMutualInformation(
        reference, moving, (1, 1, 1), (1, 1, 1), bins=8, smooth_size=0,
    )

    for result in _evaluate_both(cost, np.eye(4)):
        assert result == np.float32(-2)


@pytest.mark.parametrize("device", _DEVICES)
def test_nmi_zero_joint_entropy_is_zero_cost_and_no_overlap_is_minus_one(device):
    # Test range is finite, but every sampled value is 1/16: bin 0.5 has centre
    # weight 1.  All 64 reference voxels overlap one joint bin (entropy 0).
    reference = torch.ones((4, 4, 4), device=device)
    moving = torch.full_like(reference, 1 / 16)
    moving[3, 3, 3] = 1
    moving[3, 3, 2] = 0
    cost = FSLNormalizedMutualInformation(
        reference, moving, (1, 1, 1), (1, 1, 1), bins=8, smooth_size=0,
    )
    # Official costfns.cc:3306-3307 returns normmi=0 at |joint entropy|<1e-9.
    for result in _evaluate_both(cost, np.diag([2, 2, 2, 1])):
        assert result == np.float32(0)

    no_overlap = np.eye(4)
    no_overlap[0, 3] = 100
    for result in _evaluate_both(cost, no_overlap):
        assert result == np.float32(-1)


@pytest.mark.parametrize("device", _DEVICES)
def test_corratio_keeps_official_double_complement_rounding(device):
    reference = torch.arange(1000, dtype=torch.float32, device=device).reshape(10, 10, 10)
    moving = reference + 0.1 * torch.sin(reference)
    cost = FSLCorrelationRatio(
        reference, moving, (1, 1, 1), (1, 1, 1), bins=32, smooth_size=0,
    )
    for result in _evaluate_both(cost, np.eye(4)):
        assert 0 < result < 0.01
        # Official float p_corr_ratio returns float(1.0-r), then float Costfn
        # returns float(1.0-p).  In this range costs lie on the 2**-24 lattice.
        rounded = np.float32(1.0 - float(np.float32(1.0 - float(result))))
        assert result == rounded


def test_weighted_corratio_matches_source_left_associated_second_moment():
    # Two occupied bins avoid any ambiguity about a parallel reduction tree.
    # FSL's second moment is (weight*value)*value, not weight*(value*value).
    moving_values = np.array([1596.894287109375, 1134.6533203125,
                              1185.6898193359375, 0], dtype=np.float32)
    weight = np.float32(0.4651230573654175)
    reference = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 4, 4).contiguous()
    moving = torch.from_numpy(moving_values)[:, None, None].expand(4, 4, 4).contiguous()
    cost = FSLCorrelationRatio(
        reference, moving, (1, 1, 1), (1, 1, 1), bins=2, smooth_size=0,
        reference_weight=torch.full_like(reference, float(weight)),
        moving_weight=torch.ones_like(moving),
    )

    f32 = np.float32
    counts, sums, second_moments = (np.zeros(2, dtype=np.float32) for _ in range(3))
    for _z in range(3):
        for _y in range(3):
            for x in range(3):
                bin_id = 0 if x < 2 else 1
                value = moving_values[x]
                counts[bin_id] = f32(counts[bin_id] + weight)
                sums[bin_id] = f32(sums[bin_id] + f32(weight * value))
                second_moments[bin_id] = f32(
                    second_moments[bin_id] + f32(f32(weight * value) * value)
                )
    total_count = f32(counts[0] + counts[1])
    total_sum = f32(sums[0] + sums[1])
    total_second_moment = f32(second_moments[0] + second_moments[1])
    numerator = f32(0)
    for count, value_sum, second_moment in zip(counts, sums, second_moments):
        variance = f32(f32(second_moment - f32(f32(value_sum * value_sum) / count))
                       / f32(count - 1))
        numerator = f32(numerator + f32(variance * count))
    total_variance = f32(
        f32(total_second_moment - f32(f32(total_sum * total_sum) / total_count))
        / f32(total_count - 1)
    )
    variance_ratio = f32(f32(numerator / total_count) / total_variance)
    expected = f32(1.0 - float(f32(1.0 - float(variance_ratio))))

    for result in _evaluate_both(cost, np.eye(4)):
        assert result.view(np.uint32) == expected.view(np.uint32)
