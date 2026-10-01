"""Numerical boundaries and cache invalidation for GEMS preparation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage
import torch
from types import SimpleNamespace
import nibabel as nib

from fnit.gems.atlas import GEMSAtlas
from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe
import fnit.gems.smoothing as smoothing


def _atlas():
    vertices = np.asarray([[1, 1, 1], [6, 1, 1], [1, 6, 1], [1, 1, 6]], float)
    alphas = np.asarray([[1, 0], [0, 1], [0, 1], [0, 1]], np.float32)
    return GEMSAtlas(vertices, vertices, np.asarray([[0, 1, 2, 3]]), alphas,
                     .1, np.ones((4, 3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 2, 3), (9, 11, 7)])
@pytest.mark.parametrize("mode", ["nearest", "reflect"])
def test_device_filter_matches_scipy_including_small_grid_edges(shape, mode):
    source = np.random.default_rng(94).uniform(size=(5, *shape)).astype(np.float32)
    kernel = smoothing._discrete_gaussian_kernel(3.)
    expected = source
    for axis in (1, 2, 3):
        expected = ndimage.convolve1d(expected, kernel, axis=axis, mode=mode)
    actual = smoothing._separable_convolve(torch.from_numpy(source), kernel,
                                           mode=mode, channel_chunk=2)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=5e-7, atol=2e-7)


@pytest.mark.parametrize("sigma", [.05, .3, 1.27, 3.])
def test_partial_volume_device_gaussian_keeps_scipy_truncation_and_reflection(sigma):
    source = np.random.default_rng(133).normal(80, 25, (13, 7, 9)).astype(np.float32)
    actual = smoothing._gaussian_filter3d(torch.from_numpy(source), sigma)
    np.testing.assert_allclose(actual.numpy(), ndimage.gaussian_filter(source, sigma),
                               rtol=5e-7, atol=3e-5)


def test_reference_raster_cache_reuses_only_matching_inputs(monkeypatch):
    original = smoothing.rasterize_priors
    calls = []
    def raster(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(smoothing, "rasterize_priors", raster)
    atlas, classes, cache = _atlas(), np.asarray([0, 1]), {}
    first = smoothing.smooth_atlas_alphas(atlas, classes, 1., cache=cache)
    # A moving current mesh leaves the reference-coordinate smoothing unchanged.
    moving = atlas.with_vertices(atlas.vertices + .1)
    cached = smoothing.smooth_atlas_alphas(moving, classes, 2., cache=cache)
    uncached = smoothing.smooth_atlas_alphas(atlas, classes, 2.)
    assert len(calls) == 2
    np.testing.assert_array_equal(cached, uncached)
    np.testing.assert_allclose(first.sum(1), 1., atol=1e-6)
    smoothing.smooth_atlas_alphas(atlas, classes[::-1].copy(), 2., cache=cache)
    assert len(calls) == 3
    transformed = atlas.transformed(np.diag([1.2, 1.2, 1.2, 1.]))
    smoothing.smooth_atlas_alphas(transformed, classes[::-1].copy(), 2., cache=cache)
    assert len(calls) == 4
    changed = replace(transformed, alphas=np.full((4, 2), .5, np.float32))
    smoothing.smooth_atlas_alphas(changed, classes[::-1].copy(), 2., cache=cache)
    assert len(calls) == 5
    assert len(cache) == 1


def test_partial_volume_hyperparameters_accept_numpy_scalar_tissue_means(tmp_path):
    base = _atlas()
    # GM, WM, CSF and alveus each win one tetrahedral corner. This exercises
    # masked tissue-intensity replacement and the thin-label KDE branch.
    ids = np.asarray([0, 2, 3, 4, 201, 215, 245])
    alphas = np.full((4, len(ids)), .01, np.float32)
    alphas[np.arange(4), [2, 1, 3, 4]] = .94
    atlas = replace(base, alphas=alphas, label_ids=ids,
                     label_names=("Unknown", "WM", "GM", "CSF", "alveus", "fissure", "molecular"))
    means = np.asarray([0, 110, 75, 30, 55, 55, 55], np.float32)
    counts = np.arange(10, 17, dtype=np.float32)
    context = SimpleNamespace(image=nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)))
    actual_means, actual_counts = HippoAmygdalaRecipe("left", tmp_path)._partial_volume_hyperparameters(
        atlas, np.arange(len(ids)), means, counts, context)
    assert actual_means.dtype == np.float32
    assert 0 < actual_means[4] < 110
    assert actual_counts[4] == 11.5
    assert np.isfinite(actual_means).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_reference_smoothing_matches_cpu_and_cache():
    atlas, classes = _atlas(), np.asarray([0, 1])
    expected = smoothing.smooth_atlas_alphas(atlas, classes, 1.)
    with torch.backends.cudnn.flags(allow_tf32=True):
        actual = smoothing.smooth_atlas_alphas(atlas, classes, 1., device="cuda:0", cache={})
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_partial_volume_filter_tf32_preserves_intensity_scale():
    source = np.random.default_rng(133).uniform(0, 120, (17, 13, 11)).astype(np.float32)
    with torch.backends.cudnn.flags(allow_tf32=True):
        actual = smoothing._gaussian_filter3d(torch.from_numpy(source).cuda(), 1.27)
    # TF32 convolutions round mantissas; retain a small absolute tolerance in
    # T1 intensity units rather than requiring identical voxel arithmetic.
    np.testing.assert_allclose(actual.cpu().numpy(), ndimage.gaussian_filter(source, 1.27),
                               rtol=2e-3, atol=.03)
