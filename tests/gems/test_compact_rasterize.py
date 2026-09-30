import numpy as np
import pytest
import torch

from fnit.gems.rasterize import (BlockIndex, build_block_index, rasterize_priors,
                                  rasterize_priors_compact)


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _mesh(device):
    angle = 0.23
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    cell = np.asarray([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]) @ rotation.T
    cell += [1.137, 1.211, 1.317]
    vertices = torch.tensor(np.concatenate((cell, cell + [8.3, 6.2, 4.1])),
                            device=device, dtype=torch.float32)
    tetrahedra = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    alphas = torch.tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7], [.3, .5, .2],
                           [.2, .3, .5], [.6, .1, .3], [.4, .4, .2], [.1, .2, .7]],
                          device=device)
    return vertices, tetrahedra, alphas


@pytest.mark.parametrize("background", (None, 0, 2))
@pytest.mark.parametrize("mask_kind", ("full", "sparse"))
def test_compact_priors_and_gradients_match_dense(device, background, mask_kind):
    vertices, tetrahedra, alphas = _mesh(device)
    shape = (19, 17, 15)
    linear = torch.arange(np.prod(shape), device=device).reshape(shape)
    mask = torch.ones(shape, device=device, dtype=torch.bool) if mask_kind == "full" else linear % 3 != 0
    index = build_block_index(vertices.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=4)
    assert len(index.device_compact_batches(mask, device, vertices.dtype)[0]) > 1
    dense_vertices = vertices.clone().requires_grad_(True)
    compact_vertices = vertices.clone().requires_grad_(True)
    dense_alphas = alphas.clone().requires_grad_(True)
    compact_alphas = alphas.clone().requires_grad_(True)
    dense, dense_covered = rasterize_priors(dense_vertices, tetrahedra, dense_alphas, shape,
                                           block_index=index, background_channel=background)
    compact, covered = rasterize_priors_compact(compact_vertices, tetrahedra, compact_alphas, shape,
                                               valid_mask=mask, block_index=index,
                                               background_channel=background)
    expected = dense[:, mask]
    assert covered.any() and (~covered).any()
    assert compact.dtype == torch.float32
    assert torch.equal(covered, dense_covered[mask])
    torch.testing.assert_close(compact, expected, atol=1e-6, rtol=1e-6)
    coefficients = torch.linspace(.1, 1.3, expected.numel(), device=device).reshape_as(expected)
    dense_gradient = torch.autograd.grad((expected.square() * coefficients).sum(),
                                         (dense_vertices, dense_alphas))
    compact_gradient = torch.autograd.grad((compact.square() * coefficients).sum(),
                                           (compact_vertices, compact_alphas))
    for actual, reference in zip(compact_gradient, dense_gradient):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


def test_compact_cache_only_contains_masked_points_and_tracks_mask_changes(device):
    vertices, tetrahedra, alphas = _mesh(device)
    shape = (19, 17, 15)
    index = build_block_index(vertices.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=4)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    mask[2, 2, 2] = True
    mask[18, 16, 14] = True
    batches, reorder = index.device_compact_batches(mask, device, vertices.dtype)
    assert sum(points.numel() // 3 for points, *_ in batches) == 1
    assert len(reorder) == 2
    assert index.device_compact_batches(mask, device, vertices.dtype)[0] is batches
    for points, *_ in batches:
        xyz = points.long().reshape(-1, 3).unbind(-1)
        assert mask[xyz].all()
        assert not points.requires_grad
    mask[3, 2, 2] = True
    updated, updated_order = index.device_compact_batches(mask, device, vertices.dtype)
    assert updated is not batches
    assert sum(points.numel() // 3 for points, *_ in updated) == 2
    assert len(updated_order) == 3
    dense, coverage = rasterize_priors(vertices, tetrahedra, alphas, shape, block_index=index)
    compact, compact_coverage = rasterize_priors_compact(vertices, tetrahedra, alphas, shape,
                                                        valid_mask=mask, block_index=index)
    torch.testing.assert_close(compact, dense[:, mask])
    assert torch.equal(compact_coverage, coverage[mask])


def test_compact_batches_merge_point_counts_with_bounded_lookup_padding(device):
    shape = (32, 32, 8)
    mask_np = np.zeros(shape, dtype=bool)
    for block in range(16):
        x, y, z = np.unravel_index(np.arange(129 + block), (8, 8, 8))
        mask_np[x + (block // 4) * 8, y + (block % 4) * 8, z] = True
    mask = torch.as_tensor(mask_np, device=device)
    index = BlockIndex(shape, 8, tuple(np.asarray([0], dtype=np.int64) for _ in range(16)))
    batches, reorder = index.device_compact_batches(mask, device, torch.float32)
    # Sixteen distinct point counts share one batch instead of sixteen.
    assert len(batches) == 1
    points, _, _, _, point_rows = batches[0]
    assert points.shape == (16, 256, 3)
    assert points.numel() // 3 < 2 * int(mask_np.sum())
    assert len(point_rows) == int(mask_np.sum())
    assert torch.equal(reorder.sort().values, torch.arange(len(reorder), device=device))
    real_points = points.reshape(-1, 3)[point_rows]
    assert mask[real_points.long().unbind(-1)].all()


def test_refreshed_index_rebuilds_compact_mapping_after_mesh_movement(device):
    vertices, tetrahedra, alphas = _mesh(device)
    shape = (25, 23, 21)
    mask = torch.ones(shape, device=device, dtype=torch.bool)
    index = build_block_index(vertices.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=4)
    original, _ = rasterize_priors_compact(vertices, tetrahedra, alphas, shape,
                                           valid_mask=mask, block_index=index)
    moved = vertices + vertices.new_tensor([4.7, 2.6, 1.4])
    refreshed = build_block_index(moved.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=4)
    actual, coverage = rasterize_priors_compact(moved, tetrahedra, alphas, shape,
                                               valid_mask=mask, block_index=refreshed)
    expected, expected_coverage = rasterize_priors(moved, tetrahedra, alphas, shape,
                                                  block_index=refreshed)
    torch.testing.assert_close(actual, expected[:, mask])
    assert torch.equal(coverage, expected_coverage[mask])
    assert not torch.equal(actual, original)


@pytest.mark.parametrize("tolerance", (0, 2e-5))
def test_compact_preserves_tolerance_and_first_candidate_tie(device, tolerance):
    vertices = torch.tensor([[.00001, 0, 0], [4.00001, 0, 0], [.00001, 4, 0], [.00001, 0, 4]],
                            device=device)
    vertices = torch.cat((vertices, vertices))
    tetrahedra = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device=device)
    alphas = torch.cat((torch.tensor([[.8, .2]], device=device).repeat(4, 1),
                        torch.tensor([[.1, .9]], device=device).repeat(4, 1)))
    shape = (6, 6, 6)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    mask[0, 1, 1] = True
    index = build_block_index(vertices.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=2)
    dense, dense_covered = rasterize_priors(vertices, tetrahedra, alphas, shape, block_index=index,
                                           tolerance=tolerance, background_channel=None)
    actual, covered = rasterize_priors_compact(vertices, tetrahedra, alphas, shape,
                                              valid_mask=mask, block_index=index,
                                              tolerance=tolerance, background_channel=None)
    torch.testing.assert_close(actual, dense[:, mask], atol=0, rtol=0)
    assert torch.equal(covered, dense_covered[mask])
    assert bool(covered[0]) == (tolerance > 0)
    if tolerance:
        torch.testing.assert_close(actual[:, 0], alphas[0])


@pytest.mark.parametrize("outside_only", (False, True))
def test_compact_empty_and_unindexed_masks_keep_background_semantics(device, outside_only):
    vertices, tetrahedra, alphas = _mesh(device)
    shape = (19, 17, 15)
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    if outside_only:
        mask[-1, -1, -1] = True
    index = build_block_index(vertices.cpu().numpy(), tetrahedra.cpu().numpy(), shape, block_size=4)
    for background in (None, 2):
        dense, dense_covered = rasterize_priors(vertices, tetrahedra, alphas, shape,
                                               block_index=index, background_channel=background)
        actual, covered = rasterize_priors_compact(vertices, tetrahedra, alphas, shape,
                                                  valid_mask=mask, block_index=index,
                                                  background_channel=background)
        torch.testing.assert_close(actual, dense[:, mask], atol=0, rtol=0)
        assert torch.equal(covered, dense_covered[mask])
