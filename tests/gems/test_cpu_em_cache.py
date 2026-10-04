"""Fixed priors are cached only within one CPU inference, never across meshes."""

import numpy as np
import pytest
import torch

from fnit.gems import GEMSAtlas, TorchGEMS
from fnit.gems import core
from fnit.gems.gaussian import GaussianParameters, label_posterior


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_identity_classes_preserve_posterior_cost(masked, dtype):
    priors = torch.tensor([[0., .8, .5, .2], [1., .2, .5, .8]], dtype=dtype).reshape(2, 4, 1, 1)
    likelihood = torch.tensor([[-20., -1., -4., -2.], [-2., -5., -4., -1.]], dtype=dtype).reshape_as(priors)
    mask = torch.tensor([False, True, True, True]).reshape(4, 1, 1) if masked else None
    log_prior = torch.log(priors.clamp_min(torch.finfo(dtype).tiny))
    expected = label_posterior(priors, likelihood, torch.arange(2), mask)
    actual = core._class_posterior_from_log_prior(log_prior, likelihood, mask)
    for first, second in zip(actual, expected):
        assert torch.equal(first, second)


@pytest.mark.parametrize("grad_on", ["priors", "image", "means", "covariances"])
def test_differentiable_calls_keep_original_path(grad_on):
    priors = torch.ones((2, 8, 1, 1), requires_grad=grad_on == "priors")
    image = torch.ones((8, 1, 1), requires_grad=grad_on == "image")
    params = GaussianParameters(torch.ones((2, 1), requires_grad=grad_on == "means"),
                                torch.ones((2, 1, 1), requires_grad=grad_on == "covariances"))
    assert core._cpu_em_log_prior(priors, image, params) is None


@pytest.mark.parametrize("compact", [False, True])
def test_mesh_and_alpha_changes_refresh_cache_and_match_uncached_fit(monkeypatch, compact):
    points = np.asarray([[1., 1., 1.], [6., 1., 1.], [1., 6., 1.], [1., 1., 6.]])
    alphas = np.asarray([[.8, .2], [.2, .8], [.3, .7], [.7, .3]], np.float32)
    atlas = GEMSAtlas(points, points.copy(), np.asarray([[0, 1, 2, 3]]), alphas, .1,
                      np.ones((4, 3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))
    image = torch.zeros((8, 8, 8))
    image[1:6, 1:6, 1:6] = 1
    image[2:4, 2:4, 2:4] = 2
    options = dict(compact=compact, em_iterations=6, outer_iterations=2,
                   deform_lr=.02, fit_alpha_stages=[(torch.from_numpy(alphas), 2),
                                                    (torch.from_numpy(alphas[:, ::-1].copy()), 2)],
                   em_relative_cost_stop=1e-5, mean_hyper=torch.tensor([1., 2.]),
                   n_hyper=torch.tensor([10., 10.]))
    original_cache = core._cpu_em_log_prior
    cached_priors = []

    def observe(priors, image, params):
        result = original_cache(priors, image, params)
        if result is not None:
            cached_priors.append(result.clone())
        return result

    monkeypatch.setattr(core, "_cpu_em_log_prior", observe)
    cached = TorchGEMS(atlas)(image, **options)
    monkeypatch.setattr(core, "_cpu_em_log_prior", lambda *args: None)
    uncached = TorchGEMS(atlas)(image, **options)
    assert len(cached_priors) >= 5
    assert any(not torch.equal(cached_priors[0], value) for value in cached_priors[1:])
    for field in ["vertices", "priors", "posterior", "labels"]:
        assert torch.equal(getattr(cached, field), getattr(uncached, field))
    assert cached.objective_history == uncached.objective_history
    assert cached.optimization_stats == uncached.optimization_stats
    assert torch.equal(cached.gaussian_parameters.means, uncached.gaussian_parameters.means)
    assert torch.equal(cached.gaussian_parameters.covariances, uncached.gaussian_parameters.covariances)
