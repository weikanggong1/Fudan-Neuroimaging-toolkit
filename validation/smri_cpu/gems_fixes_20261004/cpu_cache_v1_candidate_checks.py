"""Fixed EM image preparation preserves arithmetic and invalidates mutations."""

import pytest
import torch

from fnit.gems.gaussian import _CPUEMImageCache, initialise_gaussians, update_gaussians


@pytest.mark.parametrize("modalities", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("hyper", [False, True])
def test_cached_m_step_is_exact_with_invalid_and_zero_samples(modalities, dtype, hyper):
    generator = torch.Generator().manual_seed(25)
    image = torch.randn((modalities, 3, 4, 5), generator=generator, dtype=dtype)
    image[:, 0, 0, 0] = 0
    image[0, 1, 0, 0] = float("nan")
    image[0, 2, 0, 0] = float("inf")
    image = image[0] if modalities == 1 else image
    responsibilities = torch.rand((3, 3, 4, 5), generator=generator, dtype=dtype)
    options = (dict(mean_hyper=torch.ones((3, modalities), dtype=dtype),
                    n_hyper=torch.arange(1, 4, dtype=dtype)) if hyper else {})
    cache = _CPUEMImageCache()
    for iteration in range(3):
        responsibilities = responsibilities.roll(1, 0)
        old = update_gaussians(image, responsibilities, **options)
        new = update_gaussians(image, responsibilities, _image_cache=cache, **options)
        assert torch.equal(old.means, new.means)
        assert torch.equal(old.covariances, new.covariances)


def test_mutation_alias_dtype_and_layout_invalidate_image_cache():
    image = torch.arange(24.).reshape(2, 3, 4)
    cache = _CPUEMImageCache()
    first = cache.prepare(image, torch.float32)
    assert first is cache.prepare(image, torch.float32)
    image.view(-1)[0] = 5
    changed = cache.prepare(image, torch.float32)
    assert changed is not first
    assert changed[1].sum() == image.numel()
    other_dtype = cache.prepare(image, torch.float64)
    assert other_dtype is not changed
    assert other_dtype[0].dtype == torch.float64
    transposed = image.transpose(0, 1)
    layout = cache.prepare(transposed, torch.float32)
    assert torch.equal(layout[0][:, 0], transposed.reshape(-1))
    image.set_(torch.arange(6.).reshape(1, 2, 3))
    assert cache.prepare(image, torch.float32)[0].shape == (5, 1)


def test_autograd_inference_and_autocast_use_original_path():
    cache = _CPUEMImageCache()
    assert cache.prepare(torch.ones((2, 2, 2), requires_grad=True), torch.float32) is None
    with torch.inference_mode():
        image = torch.ones((2, 2, 2))
    assert cache.prepare(image, torch.float32) is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert cache.prepare(torch.ones((2, 2, 2)), torch.float32) is None


def test_initialisation_and_updates_reuse_same_original_image():
    image = torch.arange(1., 25.).reshape(2, 3, 4)
    priors = torch.ones((2, 2, 3, 4)) / 2
    cache = _CPUEMImageCache()
    old = initialise_gaussians(image, priors, torch.arange(2))
    new = initialise_gaussians(image, priors, torch.arange(2), _image_cache=cache)
    prepared = cache._prepared
    update_gaussians(image, priors, _image_cache=cache)
    assert cache._prepared is prepared
    assert torch.equal(old.means, new.means)
    assert torch.equal(old.covariances, new.covariances)


def test_no_hyperprior_mean_uses_each_class_mass_in_all_modalities():
    image = torch.tensor([[[[2., 4., 8.]]], [[[20., 40., 80.]]]])
    responsibilities = torch.tensor([[[[3., 1., 0.]]], [[[0., 0., 2.]]], [[[1., 1., 1.]]]])
    result = update_gaussians(image, responsibilities)
    assert result.means.shape == (3, 2)
    assert torch.equal(result.means, torch.tensor([[2.5, 25.], [8., 80.], [14. / 3., 140. / 3.]]))


def test_image_cache_matches_complete_mesh_fit(monkeypatch):
    import numpy as np
    from fnit.gems import GEMSAtlas, TorchGEMS, core
    points = np.array([[1., 1., 1.], [6., 1., 1.], [1., 6., 1.], [1., 1., 6.]])
    alphas = np.array([[.8, .2], [.2, .8], [.3, .7], [.7, .3]], np.float32)
    atlas = GEMSAtlas(points, points.copy(), np.array([[0, 1, 2, 3]]), alphas, .1,
                      np.ones((4, 3), bool), np.array([0, 10]), ("Unknown", "ROI"))
    image = torch.zeros((8, 8, 8))
    image[1:6, 1:6, 1:6] = 1
    image[2:4, 2:4, 2:4] = 2
    options = dict(em_iterations=6, outer_iterations=2, deform_iterations=3,
                   deform_optimizer="lbfgs", deform_lr=.02,
                   mean_hyper=torch.tensor([1., 2.]), n_hyper=torch.tensor([10., 10.]),
                   em_relative_cost_stop=1e-5, stable_mesh_fitting=True)
    cached = TorchGEMS(atlas)(image, **options)
    monkeypatch.setattr(core, "_CPUEMImageCache", lambda: None)
    uncached = TorchGEMS(atlas)(image, **options)
    for field in ["vertices", "priors", "posterior", "labels"]:
        assert torch.equal(getattr(cached, field), getattr(uncached, field))
    assert cached.objective_history == uncached.objective_history
    assert cached.min_jacobian == uncached.min_jacobian
    assert cached.optimization_stats == uncached.optimization_stats
    assert torch.equal(cached.gaussian_parameters.means, uncached.gaussian_parameters.means)
    assert torch.equal(cached.gaussian_parameters.covariances, uncached.gaussian_parameters.covariances)
