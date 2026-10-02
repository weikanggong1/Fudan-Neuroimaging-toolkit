"""核对极点反射、非连续帧和整数三跳可达性；不能代替真实阶段回归。"""

import pytest
import torch

from fnit.recon_all.mris_register_blur import blur_atlas_frame, _blur_atlas_frame_torch
from fnit.recon_all.mris_register_nonlinear import three_hop_avg_nbrs


@pytest.mark.parametrize("shape", [(8, 4), (32, 16), (3, 2)])
@pytest.mark.parametrize("sigma", [.5, 1., 4.])
def test_compiled_blur_keeps_each_float32_value(shape, sigma):
    generator = torch.Generator().manual_seed(71)
    frame = torch.randn((shape[0] * 2, shape[1]), generator=generator)[::2]
    original = frame.clone()
    expected = _blur_atlas_frame_torch(frame, sigma)
    actual = blur_atlas_frame(frame, sigma)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(frame, original, rtol=0, atol=0)


def test_blur_keeps_autograd_on_reference_path():
    frame = torch.arange(32., requires_grad=True).reshape(8, 4)
    result = blur_atlas_frame(frame, sigma=.5)
    assert result.requires_grad


def test_three_hop_count_includes_unique_paths_and_isolated_vertices():
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 3], [2, 0], [0, 0]], dtype=torch.int64)
    degrees = torch.tensor([2, 2, 2, 1, 0], dtype=torch.int64)
    rows = [neighbors[v, :degrees[v]].tolist() for v in range(len(degrees))]
    total = 0
    for vertex in range(len(rows)):
        seen, frontier = {vertex}, {vertex}
        for _ in range(3):
            new = {n for v in frontier for n in rows[v]} - seen
            seen.update(new)
            frontier = new
        total += len(seen) - 1
    expected = float(torch.tensor(float(torch.tensor(total, dtype=torch.float32)) /
                                  float(torch.tensor(len(rows), dtype=torch.float32)),
                                  dtype=torch.float32))
    assert three_hop_avg_nbrs(neighbors, degrees) == expected


@pytest.mark.parametrize("neighbors,degrees", [
    (torch.tensor([[0]], dtype=torch.int64), torch.tensor([2])),
    (torch.tensor([[1]], dtype=torch.int64), torch.tensor([1])),
    (torch.tensor([[-1]], dtype=torch.int64), torch.tensor([1])),
])
def test_gradient_averager_rejects_active_invalid_topology(neighbors, degrees):
    from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager
    with pytest.raises(ValueError):
        RegistrationGradientAverager(neighbors, degrees, device="cpu")


def test_zero_averages_keeps_clone_semantics_and_ignores_padding():
    from fnit.recon_all.mris_register_average_numba import RegistrationGradientAverager
    neighbors = torch.tensor([[-1]], dtype=torch.int64)
    degrees = torch.tensor([0], dtype=torch.int64)
    average = RegistrationGradientAverager(neighbors, degrees, device="cpu")
    gradient = torch.tensor([[1., -2., 3.]])
    actual = average(gradient, iterations=0)
    assert actual.data_ptr() != gradient.data_ptr()
    torch.testing.assert_close(actual, gradient, rtol=0, atol=0)
