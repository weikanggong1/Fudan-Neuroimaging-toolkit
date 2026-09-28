"""Tests for the UKB tck2connectome matrix stage."""

import pytest
import torch

from fnit.connectome.assignment import build_connectomes


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_count_fbc_and_weighted_edge_means(device):
    atlas = torch.zeros((8, 3, 3), dtype=torch.int16, device=device)
    atlas[1, 1, 1] = 1
    atlas[5, 1, 1] = 3  # Label 2 is absent but keeps row and column 1.
    endpoints = torch.tensor(
        [
            [[1.1, 1, 1], [5, 1, 1]],
            [[5, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[20, 1, 1], [5, 1, 1]],
        ], device=device,
    )
    matrices = build_connectomes(
        endpoints, atlas, torch.eye(4, device=device),
        weights=torch.tensor([1, 3, 2, 5], device=device),
        lengths=torch.tensor([10, 20, 30, 40], device=device),
        fa=torch.tensor([0.2, 0.8, 0.4, 0.9], device=device),
        batch_size=2,
    )
    assert set(matrices) == {"count", "sift2_fbc", "mean_length", "mean_fa"}
    torch.testing.assert_close(
        matrices["count"], torch.tensor([[1, 0, 2], [0, 0, 0], [2, 0, 0]], device=device)
    )
    torch.testing.assert_close(
        matrices["sift2_fbc"],
        torch.tensor([[2., 0, 4], [0, 0, 0], [4, 0, 0]], device=device),
    )
    torch.testing.assert_close(
        matrices["mean_length"],
        torch.tensor([[30., 0, 17.5], [0, 0, 0], [17.5, 0, 0]], device=device),
    )
    torch.testing.assert_close(
        matrices["mean_fa"],
        torch.tensor([[0.4, 0, 0.65], [0, 0, 0], [0.65, 0, 0]], device=device),
    )


def test_strict_radius_and_nearest_label():
    atlas = torch.zeros((12, 1, 1), dtype=torch.int16)
    atlas[1, 0, 0] = 1
    atlas[10, 0, 0] = 2
    endpoints = torch.tensor([
        [[5, 0, 0], [10, 0, 0]],     # Exactly 4 mm from label 1: omitted.
        [[4.999, 0, 0], [10, 0, 0]], # Just inside 4 mm: counted.
        [[8, 0, 0], [10, 0, 0]],    # Nearest non-zero centre is label 2.
    ])
    matrices = build_connectomes(endpoints, atlas, torch.eye(4))
    torch.testing.assert_close(matrices["count"], torch.tensor([[0, 1], [1, 1]]))


def test_affine_world_distance_and_unweighted_mean():
    atlas = torch.zeros((4, 3, 3), dtype=torch.int16)
    atlas[1, 1, 1] = 1
    atlas[2, 1, 1] = 2
    affine = torch.tensor([
        [2., 1., 0., 10.],
        [0., 3., 0., 20.],
        [0., 0., 4., 30.],
        [0., 0., 0., 1.],
    ])
    endpoints = torch.tensor([
        [[13., 23., 34.], [15., 23., 34.]],
        [[13., 23., 34.], [15., 23., 34.]],
    ])
    matrices = build_connectomes(
        endpoints, atlas, affine, lengths=torch.tensor([10., 30.]), radius=1.
    )
    assert set(matrices) == {"count", "mean_length"}
    torch.testing.assert_close(matrices["count"], torch.tensor([[0, 2], [2, 0]]))
    torch.testing.assert_close(matrices["mean_length"], torch.tensor([[0., 20.], [20., 0.]]))



@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_many_streamlines_use_precise_weighted_reduction(device):
    atlas = torch.zeros((3, 1, 1), dtype=torch.int16, device=device)
    atlas[0, 0, 0], atlas[2, 0, 0] = 1, 2
    n = 120_000
    endpoints = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], device=device).expand(n, -1, -1)
    weights = torch.tensor([1., 3.], device=device).repeat(n // 2)
    fa = torch.tensor([0.2, 0.8], device=device).repeat(n // 2)
    matrices = build_connectomes(
        endpoints, atlas, torch.eye(4, device=device), weights=weights, fa=fa, radius=0.1
    )
    expected_fa = (float(fa[0]) + 3 * float(fa[1])) / 4
    assert matrices["mean_fa"].dtype == torch.float32
    assert matrices["count"][0, 1].item() == n
    assert matrices["sift2_fbc"][0, 1].item() == 2 * n
    assert abs(matrices["mean_fa"][0, 1].item() - expected_fa) < 1e-7


def test_rejects_invalid_shapes():
    atlas = torch.ones((2, 2, 2), dtype=torch.int16)
    with pytest.raises(ValueError, match="endpoints"):
        build_connectomes(torch.zeros((2, 3)), atlas, torch.eye(4))
    with pytest.raises(ValueError, match="weights"):
        build_connectomes(torch.zeros((2, 2, 3)), atlas, torch.eye(4), weights=torch.ones(3))


def test_explicit_node_count_retains_absent_final_node():
    atlas = torch.ones((2, 2, 2), dtype=torch.int32)
    endpoints = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
    count = build_connectomes(endpoints, atlas, torch.eye(4), node_count=3)["count"]
    assert count.shape == (3, 3)
    assert count[0, 0] == 1
    assert count[2].sum() == 0
