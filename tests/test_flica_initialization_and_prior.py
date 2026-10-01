"""Closed-form CPU FLICA checks; these small arrays are not benchmarks."""

import copy

import numpy as np
import pytest
from scipy.special import digamma

from fnit.bigflica import flica_vb


def _data():
    random = np.random.default_rng(51)
    return [random.normal(size=(rows, 8)) for rows in (11, 13, 9)]


def _options(initial_h='PCA', noise='R', iterations=2):
    return dict(num_components=2, maxits=iterations, lambda_dims=noise,
                initH=initial_h, dof_per_voxel=np.array([.3, .7, 1.4]), computeF=1)


def test_pca_initializer_has_matlab_svd_scaling_and_reconstruction():
    data, options = _data(), _options()
    priors, posterior, constants = flica_vb.flica_init_params(copy.deepcopy(data), options)
    weighted = np.vstack([np.sqrt(dd) * y for dd, y in zip(constants['DD'], data)])
    left, singular, right = np.linalg.svd(weighted, full_matrices=False)
    rows = len(weighted)
    expected_h = singular[:2, None] * right[:2] / np.sqrt(rows * constants['DD'].mean())
    signs = np.sign(np.sum(expected_h * posterior['H'], axis=1))
    np.testing.assert_allclose(posterior['H'], signs[:, None] * expected_h, rtol=1e-12, atol=1e-12)
    spatial = np.vstack(posterior['X'])
    np.testing.assert_allclose(np.mean(spatial ** 2, axis=0), np.ones(2), rtol=1e-12)
    np.testing.assert_allclose(spatial, left[:, :2] * signs * np.sqrt(rows), rtol=1e-12, atol=1e-12)
    expected_projection = (left[:, :2] * singular[:2]) @ right[:2]
    actual_projection = np.vstack([
        np.sqrt(dd) * (x * w) @ posterior['H']
        for dd, x, w in zip(constants['DD'], posterior['X'], posterior['W'])])
    np.testing.assert_allclose(actual_projection, expected_projection, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(priors['prior_W_var'], 1 / constants['DD'])


@pytest.mark.parametrize('noise', ['R', 'o'])
def test_w_gaussian_prior_uses_each_modalities_decimation(noise):
    data = _data()
    options = _options(np.array([[1., 0., 1., -1., .2, .3, .4, .1],
                                 [0., 1., .5, .2, -.4, .1, .8, .3]]), noise)
    priors, posterior, constants = flica_vb.flica_init_params(data, options)
    np.testing.assert_allclose(priors['prior_W_var'], 1 / constants['DD'])
    state = {**priors, **posterior, **constants, 'Y': data, 'opts': options}
    before = copy.deepcopy(state)
    updated = flica_vb.update_HlambdaHt_and_W(state)
    h = before['H']
    for k, y in enumerate(data):
        noise_precision = np.asarray(before['lambda_R'][k]).ravel()
        # E[H_i H_j] from the independent Gaussian column posteriors.
        h_second = sum(noise_precision[r] * (np.outer(h[:, r], h[:, r]) + before['H_colcov'])
                       for r in range(h.shape[1]))
        precision = before['XtDX'][k] * h_second + np.eye(2) * constants['DD'][k]
        expected_covariance = np.linalg.inv(precision)
        linear = np.array([constants['DD'][k] * np.sum(
            y * before['X'][k][:, i, None] * h[i] * noise_precision)
            for i in range(2)])
        np.testing.assert_allclose(updated['W_rowcov'][k], expected_covariance, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(updated['W'][k]).ravel(), expected_covariance @ linear,
                                   rtol=1e-12, atol=1e-12)
    state.update(updated)
    for key in ('sumN_Dq', 'sumN_DqXq', 'sumN_DqXq2', 'sumN_Dqlogq'):
        state[key].fill(0.)
    _, terms = flica_vb.compute_F(state)
    for k in range(3):
        variance = 1 / constants['DD'][k]
        expected = -np.log(2 * np.pi * variance) - np.trace(updated['WtW'][k]) / (2 * variance)
        assert float(np.asarray(terms['Wprior'][k]).item()) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('noise', ['R', 'o'])
@pytest.mark.parametrize('initialization', ['PCA', 'numeric'])
def test_modalities_permutation_preserves_posterior_and_free_energy(noise, initialization, tmp_path):
    data, order = _data(), np.array([2, 0, 1])
    initial_h = ('PCA' if initialization == 'PCA' else
                 np.array([[1., 0., 1., -1., .2, .3, .4, .1],
                           [0., 1., .5, .2, -.4, .1, .8, .3]]))
    options = _options(initial_h=initial_h, noise=noise)
    (tmp_path / 'original').mkdir()
    (tmp_path / 'permuted').mkdir()
    options['output_dir'] = str(tmp_path / 'original')
    initial = flica_vb.flica_init_params(copy.deepcopy(data), options)
    original = flica_vb.flica_iterate(data, options, *copy.deepcopy(initial))
    permuted_options = _options(initial_h=initial_h, noise=noise)
    permuted_options['dof_per_voxel'] = options['dof_per_voxel'][order]
    permuted_options['output_dir'] = str(tmp_path / 'permuted')
    permuted_data = [data[k] for k in order]
    permuted_initial = flica_vb.flica_init_params(copy.deepcopy(permuted_data), permuted_options)
    permuted = flica_vb.flica_iterate(permuted_data, permuted_options, *copy.deepcopy(permuted_initial))
    signs = np.sign(np.sum(original['H'] * permuted['H'], axis=1))
    np.testing.assert_allclose(original['H'], signs[:, None] * permuted['H'], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(original['F_history'], permuted['F_history'], rtol=1e-12, atol=1e-10)
    for new_k, old_k in enumerate(order):
        np.testing.assert_allclose(original['X'][old_k], permuted['X'][new_k] * signs,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(original['W'][old_k], permuted['W'][new_k], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(original['lambda'][old_k], permuted['lambda'][new_k], rtol=1e-10, atol=1e-10)


def test_eta_gamma_update_matches_conjugate_sufficient_statistics():
    second_moment = np.array([4., 7.])
    actual = flica_vb.update_eta(dict(prior_eta_b=np.array([2.]), prior_eta_c=np.array([.6]),
                                    H2Gmat=second_moment, Gmat=np.ones(8), L=2))
    shape, rate = np.full((2, 1), 4.6), .5 + second_moment[:, None] / 2
    np.testing.assert_allclose(actual['eta_c'], shape)
    np.testing.assert_allclose(actual['eta_binv'], rate)
    np.testing.assert_allclose(actual['eta'], shape / rate)
    np.testing.assert_allclose(actual['eta_log'], digamma(shape) - np.log(rate))


def test_beta_and_mu_updates_match_weighted_gaussian_observations():
    data, options = _data(), _options()
    priors, posterior, constants = flica_vb.flica_init_params(data, options)
    state = {**priors, **posterior, **constants}
    observations = np.array([[.2, -.5], [.7, .4], [-.1, .8]])
    observation_variances = np.array([[.1, .2], [.3, .1], [.2, .4]])
    responsibilities = np.array([[.2, .5, .3], [.6, .1, .3], [.1, .2, .7]])
    weights = constants['DD'][0] * responsibilities
    counts = weights.sum(axis=0)[:, None] * np.ones((1, 2))
    state['sumN_Dq'][0] = counts
    state['sumN_DqXq'][0] = weights.T @ observations
    state['sumN_DqXq2'][0] = weights.T @ (observations ** 2 + observation_variances)
    state['mu'][0] = np.array([[.1, -.2], [.3, .4], [-.1, .2]])
    state['mu_var'][0] = np.full((3, 2), .15)
    state['mu2'][0] = state['mu'][0] ** 2 + state['mu_var'][0]
    state['prior_beta_c'][0] = np.full((3, 2), 1.7)
    state['prior_beta_b'][0] = np.full((3, 2), 2.3)
    state['prior_mu_mean'], state['prior_mu_var'] = np.array([.2]), np.array([2.])
    squared_errors = sum(weights[n, :, None] * (
        (observations[n] - state['mu'][0]) ** 2 + observation_variances[n] + state['mu_var'][0])
        for n in range(len(observations)))
    shape, rate = 1.7 + counts / 2, 1 / 2.3 + squared_errors / 2
    beta = shape / rate
    mu_precision = .5 + beta * counts
    mu_mean = (.1 + beta * (weights.T @ observations)) / mu_precision
    # Only the first modality is supplied to this conjugate update check.
    state['K'] = 1
    actual = flica_vb.update_mixmod(state)
    np.testing.assert_allclose(actual['beta_c'][0], shape)
    np.testing.assert_allclose(actual['beta_binv'][0], rate)
    np.testing.assert_allclose(actual['beta'][0], beta)
    np.testing.assert_allclose(actual['beta_log'][0], digamma(shape) - np.log(rate))
    np.testing.assert_allclose(actual['mu'][0], mu_mean)
    np.testing.assert_allclose(actual['mu_var'][0], 1 / mu_precision)
    np.testing.assert_allclose(actual['mu2'][0], mu_mean ** 2 + 1 / mu_precision)


@pytest.mark.parametrize('noise', ['R', 'o'])
def test_lambda_gamma_update_matches_expected_gaussian_squared_errors(noise):
    y = _data()[0][:3]
    options = dict(num_components=2, maxits=1, lambda_dims=noise,
                   initH=np.array([[1., -.2, .3, .4, -.1, .5, .2, -.3],
                                   [.2, .6, -.4, .1, .3, -.2, .5, .7]]),
                   dof_per_voxel=np.array([.7]), computeF=1)
    priors, posterior, constants = flica_vb.flica_init_params([y], options)
    state = {**priors, **posterior, **constants, 'opts': options, 'Y': [y]}
    x, x_variance = np.array([[.2, -.4], [.7, .1], [-.1, .5]]), np.array([[.3, .2], [.4, .1], [.2, .5]])
    w, w_covariance = np.array([.6, -.2]), np.array([[.2, .03], [.03, .3]])
    h = state['H']
    h_covariance = np.array([[.3, .02], [.02, .4]])
    state['H_colcov'] = (np.stack([(r + 1) * h_covariance / 8 for r in range(8)], axis=2)
                         if noise == 'R' else h_covariance)
    state['X'][0], state['X2'][0] = x, x ** 2 + x_variance
    state['W'][0] = np.matrix(w[None])
    state['WtW'][0] = np.matrix(np.outer(w, w) + w_covariance)
    state['XtDX'][0] = .7 * (x.T @ x + np.diag(x_variance.sum(axis=0)))
    expected_errors = np.zeros(8)
    for r in range(8):
        covariance = state['H_colcov'][:, :, r] if noise == 'R' else h_covariance
        h_second = np.outer(h[:, r], h[:, r]) + covariance
        for n in range(3):
            x_second = np.outer(x[n], x[n]) + np.diag(x_variance[n])
            expected_errors[r] += (y[n, r] ** 2 - 2 * y[n, r] * np.sum(x[n] * w * h[:, r])
                                   + np.sum(x_second * (np.outer(w, w) + w_covariance) * h_second))
    if noise == 'R':
        shape = np.full(8, .7 * 3 / 2) + state['prior_lambda_c'][0].ravel()
        rate = .7 * expected_errors / 2 + 1 / state['prior_lambda_b'][0].ravel()
    else:
        shape = np.array([.7 * 3 * 8 / 2]) + state['prior_lambda_c'][0].ravel()
        rate = np.array([.7 * expected_errors.sum() / 2]) + 1 / state['prior_lambda_b'][0].ravel()
    actual = flica_vb.update_lambda(state)
    np.testing.assert_allclose(np.asarray(actual['lambda_c'][0]).ravel(), shape, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(actual['lambda_binv'][0]).ravel(), rate, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(actual['Lambda'][0]).ravel(), shape / rate, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(actual['lambda_log'][0]).ravel(), digamma(shape) - np.log(rate), rtol=1e-12)


def test_explicit_legacy_pca_preserves_previous_initialization_scale():
    data, options = _data(), _options('PCA_legacy')
    _, posterior, constants = flica_vb.flica_init_params(data, options)
    gram = sum(dd * (y.T @ y) for dd, y in zip(constants['DD'], data))
    eigenvalues = np.linalg.eigvalsh(gram)[::-1][:2]
    np.testing.assert_allclose(np.linalg.norm(posterior['H'], axis=1), eigenvalues, rtol=1e-12)


@pytest.mark.parametrize('variance', [[1., 2.], [1., 0., 2.], [np.inf], [np.nan], [-1.]])
def test_w_prior_variance_rejects_wrong_shape_or_nonpositive_values(variance):
    with pytest.raises(ValueError, match='prior_W_var'):
        flica_vb._W_prior_variances(variance, 3)


def test_old_scalar_w_prior_is_explicitly_broadcast():
    np.testing.assert_array_equal(flica_vb._W_prior_variances(np.array([2.]), 3), [2., 2., 2.])
