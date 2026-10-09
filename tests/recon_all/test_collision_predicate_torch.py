"""固定三角对的源规则回归；模拟几何只用于算子排错，不是脑影像benchmark。"""
import numpy as np
import pytest
import torch

from fnit.recon_all.place_surface_collision_torch import triangle_pairs_intersect_torch, _source_pairs


def test_source_rules_random_contact_coplanar_degenerate_and_thresholds():
    random = np.random.default_rng(481)
    a = random.normal(size=(3000, 3, 3)).astype(np.float32)
    b = random.normal(size=(3000, 3, 3)).astype(np.float32)
    # 共面、顶点接触、边接触、平行分离、退化与源阈值两侧。
    plane = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    variants = [plane, plane + [0, 0, 1], plane + [2, 0, 0],
                np.zeros((3, 3), np.float32),
                np.array([[.2, .2, -1], [.2, .2, 1], [1, 1, 0]], np.float32)]
    variants += [plane + [0, 0, epsilon] for epsilon in (2.49e-7, 2.5e-7, 2.51e-7, 2.49e-6, 2.5e-6, 2.51e-6)]
    a = np.concatenate([a, np.repeat(plane[None], len(variants), axis=0)])
    b = np.concatenate([b, np.asarray(variants, np.float32)])
    expected = _source_pairs(a, b)
    raw, raw_info = triangle_pairs_intersect_torch(a, b, device="cpu", chunk_size=127, source_recheck=False)
    actual, info = triangle_pairs_intersect_torch(a, b, device="cpu", chunk_size=64)
    np.testing.assert_array_equal(raw.numpy(), expected)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert info["source_rechecked_pairs"] > 0
    assert raw_info["source_rechecked_pairs"] == 0
    assert info["changes_vertex_update_order"] is False


def test_empty_and_validation_failures():
    empty = np.empty((0, 3, 3), np.float32)
    result, info = triangle_pairs_intersect_torch(empty, empty, device="cpu")
    assert result.shape == (0,) and result.dtype == torch.bool and info["pairs"] == 0
    with pytest.raises(ValueError, match="indexed"):
        triangle_pairs_intersect_torch(empty, empty, device="cuda")
    with pytest.raises(TypeError, match="float32"):
        triangle_pairs_intersect_torch(empty.astype(np.float64), empty, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        triangle_pairs_intersect_torch(np.zeros((3, 3), np.float32), empty, device="cpu")
    invalid = np.full((1, 3, 3), np.nan, np.float32)
    with pytest.raises(ValueError, match="finite"):
        triangle_pairs_intersect_torch(invalid, invalid, device="cpu")
