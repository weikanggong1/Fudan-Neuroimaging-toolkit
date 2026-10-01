"""Stable alpha fitting keeps probability normalization and cache boundaries."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.smoothing import smooth_atlas_alphas


def _atlas():
    vertices = np.asarray([[1, 1, 1], [9, 1, 1], [1, 9, 1], [1, 1, 9]], float)
    alphas = np.asarray([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8], [.2, .3, .5]], np.float32)
    return GEMSAtlas(vertices, vertices, np.asarray([[0, 1, 2, 3]]), alphas,
                     .1, np.ones((4, 3), bool), np.asarray([0, 10, 49]),
                     ("Unknown", "Left-Thalamus", "Right-Thalamus"))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stable_alpha_probabilities_repeat_and_preserve_model(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    atlas, classes = _atlas(), np.arange(3)
    original = smooth_atlas_alphas(atlas, classes, 1., device=device)
    expected = smooth_atlas_alphas(atlas, classes, 1., device=device,
                                  stable_vertex_statistics=True)
    assert expected.dtype == np.float32 and np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(1), 1., atol=2e-7)
    np.testing.assert_allclose(expected, original, rtol=2e-6, atol=2e-7)
    for _ in range(4):
        actual = smooth_atlas_alphas(atlas, classes, 1., device=device,
                                    stable_vertex_statistics=True, cache={})
        np.testing.assert_array_equal(actual, expected)


def test_stable_alpha_layout_cache_tracks_reference_and_class_changes():
    atlas, classes, cache = _atlas(), np.arange(3), {}
    smooth_atlas_alphas(atlas, classes, 1., cache=cache, stable_vertex_statistics=True)
    first = cache["stable_vertex_statistics"]
    # Current vertices and smoothing bandwidth leave the reference layout fixed.
    smooth_atlas_alphas(atlas.with_vertices(atlas.vertices + .1), classes, 2.,
                        cache=cache, stable_vertex_statistics=True)
    assert cache["stable_vertex_statistics"] is first
    changed = replace(atlas, alphas=np.full_like(atlas.alphas, 1 / 3))
    result = smooth_atlas_alphas(changed, classes[::-1].copy(), 2., cache=cache,
                                stable_vertex_statistics=True)
    assert cache["stable_vertex_statistics"] is not first
    expected = smooth_atlas_alphas(changed, classes[::-1].copy(), 2.,
                                  stable_vertex_statistics=True)
    np.testing.assert_array_equal(result, expected)
