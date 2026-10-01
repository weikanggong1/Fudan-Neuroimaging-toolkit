"""Mathematical regression checks; these are not scientific benchmarks."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma
from scipy.stats import dirichlet, gamma, multivariate_normal, norm

from fnit.bigflica import flica_vb


@pytest.mark.parametrize("dimensions, expected", [
    ([3, 0], [[12., 15., 18., 21.]]),
    ([0, 4], [[6.], [22.], [38.]]),
    ([3, 4], [[66.]]),
])
def test_sum_dims_preserves_unsummed_columns_and_axes(dimensions, expected):
    actual = flica_vb.sum_dims(np.arange(12.).reshape(3, 4), dimensions)
    np.testing.assert_array_equal(actual, expected)


def test_sum_dims_virtual_singleton_dimension():
    column = np.array([[1.], [2.], [4.]])
    np.testing.assert_array_equal(flica_vb.sum_dims(column, [0, 5]), column * 5)
    np.testing.assert_array_equal(flica_vb.sum_dims(column, [3, 5]), [[35.]])


def test_sum_dims_three_dimensions_do_not_shift_after_reduction():
    values = np.arange(6., dtype=np.float64).reshape(2, 1, 3)
    np.testing.assert_array_equal(flica_vb.sum_dims(values, [2, 0, 0]),
                                  [[[3., 5., 7.]]])
    np.testing.assert_array_equal(flica_vb.sum_dims(values, [2, 4, 3]), [[[60.]]])


def _variational_state(modalities=2, subjectwise=False):
    """A finite, small posterior with known Gaussian/Gamma/Dirichlet moments."""
    data = [np.array([[1., 2., -1., 0.], [2., -1., 1., 3.],
                      [0., 2., 4., 1.]]) for _ in range(modalities)]
    initial_h = np.array([[1., 0., 2., 1.], [0., 1., 1., -1.]])
    options = {"num_components": 2, "maxits": 3,
               "lambda_dims": "R" if subjectwise else "o",
               "initH": initial_h, "dof_per_voxel": np.ones(modalities)}
    priors, posterior, constants = flica_vb.flica_init_params(data, options)
    state = {**priors, **posterior, **constants, "Y": data, "G": 1}
    covariance = np.array([[.7, .2], [.2, 1.1]])
    state["H_colcov"] = (np.stack([(r + 1) * covariance / 3 for r in range(4)],
                                       axis=2) if subjectwise else covariance)
    h_cov_sum = (np.diagonal(state["H_colcov"], axis1=0, axis2=1).sum(axis=0)
                 if subjectwise else 4 * np.diag(covariance))
    state["H2Gmat"] = np.square(state["H"]).sum(axis=1) + h_cov_sum
    state.update(prior_eta_b=np.array([2.5]), prior_eta_c=np.array([1.2]),
                 eta_c=np.full((2, 1), 3.), eta_binv=np.full((2, 1), 1.5),
                 eta=np.full((2, 1), 2.),
                 eta_log=np.full((2, 1), digamma(3.) - np.log(1.5)),
                 prior_W_var=np.array([2.4]), prior_mu_var=np.array([2.5]))
    for k in range(modalities):
        state["W"][k] = np.matrix([[.5, -.2]])
        state["W_rowcov"][k] = np.array([[.3, .05], [.05, .8]])
        state["WtW"][k] = np.asarray(state["W"][k].T @ state["W"][k]) + state["W_rowcov"][k]
        state["mu"][k] = np.array([[.2, -.3], [.1, .4], [-.2, .5]])
        state["mu_var"][k] = np.array([[.2, .4], [.3, .6], [.7, .9]])
        state["mu2"][k] = state["mu"][k] ** 2 + state["mu_var"][k]
        state["prior_beta_c"][k] = np.full((3, 2), 2.)
        state["prior_beta_b"][k] = np.full((3, 2), 1.5)
        state["beta_c"][k] = np.full((3, 2), 3.)
        state["beta_binv"][k] = np.full((3, 2), 2.)
        state["beta"][k] = np.full((3, 2), 1.5)
        state["beta_log"][k] = np.full((3, 2), digamma(3.) - np.log(2.))
        state["pi_weights"][k] = np.array([[2., 3.], [4., 5.], [6., 8.]])
        state["pi_log"][k] = (digamma(state["pi_weights"][k]) -
                                  digamma(state["pi_weights"][k].sum(axis=0)))
        state["sumN_Dq"][k] = 1
        state["sumN_DqXq"][k] = 0
        state["sumN_DqXq2"][k] = .4
        state["sumN_Dqlogq"][k] = -np.log(3.)
        state["Xq_var"][k] = np.full((2, 3), .4)
        shape = np.arange(2., 6.)[:, None] if subjectwise else np.array([[3.]])
        rate = shape / 2
        state["lambda_c"][k] = shape
        state["lambda_binv"][k] = rate
        state["Lambda"][k] = shape / rate
        state["lambda_log"][k] = digamma(shape) - np.log(rate)
        state["lambda_R"][k] = np.matrix(np.broadcast_to(shape / rate, (4, 1)).copy())
        state["lambda_log_R"][k] = np.matrix(np.broadcast_to(state["lambda_log"][k], (4, 1)).copy())
        state["prior_lambda_c"][k] = np.full_like(shape, 1.7)
        state["prior_lambda_b"][k] = np.full_like(shape, 2.3)
        state["HlambdaHt"][k] = np.eye(2)
    return state


@pytest.mark.parametrize("subjectwise", [False, True])
def test_gaussian_entropy_matches_distribution_entropy(subjectwise):
    state = _variational_state(subjectwise=subjectwise)
    _, terms = flica_vb.compute_F(state)
    covariance = state["H_colcov"]
    expected_h = (math.fsum(multivariate_normal(cov=covariance[:, :, r]).entropy()
                            for r in range(state["R"])) if subjectwise else
                  state["R"] * multivariate_normal(cov=covariance).entropy())
    assert float(np.asarray(terms["Hpost"]).item()) == pytest.approx(expected_h, rel=1e-12)
    expected_w = multivariate_normal(cov=state["W_rowcov"][0]).entropy()
    np.testing.assert_allclose(terms["Wpost"], expected_w, rtol=1e-12)


def test_mu_gaussian_prior_contains_negative_log_normalizer():
    state = _variational_state()
    _, terms = flica_vb.compute_F(state)
    variance = float(state["prior_mu_var"][0])
    expected = (norm.logpdf(state["mu"][0], scale=np.sqrt(variance)) -
                state["mu_var"][0] / (2 * variance)).sum()
    np.testing.assert_allclose(terms["muPrior"], expected, rtol=1e-12)


@pytest.mark.parametrize("subjectwise", [False, True])
def test_lambda_gamma_prior_matches_expected_log_density(subjectwise):
    state = _variational_state(subjectwise=subjectwise)
    _, terms = flica_vb.compute_F(state)
    expected = 0.
    for shape, rate, prior_shape, prior_scale in zip(
            state["lambda_c"][0].ravel(), state["lambda_binv"][0].ravel(),
            state["prior_lambda_c"][0].ravel(), state["prior_lambda_b"][0].ravel()):
        posterior = gamma(a=shape, scale=1 / rate)
        prior = gamma(a=prior_shape, scale=prior_scale)
        expected += quad(lambda value: posterior.pdf(value) * prior.logpdf(value),
                         0, np.inf, epsabs=1e-10, epsrel=1e-10)[0]
    np.testing.assert_allclose(terms["lambdaPrior"], expected, rtol=1e-10)


@pytest.mark.parametrize("matrix_noise", [False, True], ids=["ndarray", "matrix"])
def test_subjectwise_gaussian_likelihood_matches_independent_expected_sse(matrix_noise):
    state = _variational_state(1, subjectwise=True)
    data, h = state["Y"][0], state["H"]
    x = np.array([[.2, -.4], [.7, .1], [-.1, .5]])
    x_variance = np.array([[.3, .2], [.4, .1], [.2, .5]])
    w = np.asarray(state["W"][0]).ravel()
    h_covariance, w_moment = state["H_colcov"], state["WtW"][0]
    decimation = .7
    shape = state["lambda_c"][0]
    rate = np.array([[1.], [2.], [4.], [5.]])
    noise_mean = (shape / rate).ravel()
    noise_log_mean = (digamma(shape) - np.log(rate)).ravel()
    state["DD"][0] = decimation
    state["X"][0], state["X2"][0] = x, x ** 2 + x_variance
    state["XtDX"][0] = decimation * (x.T @ x + np.diag(x_variance.sum(axis=0)))
    state["Y2D_sumN"][0] = decimation * np.square(data).sum(axis=0)
    state["lambda_binv"][0], state["Lambda"][0] = rate, shape / rate
    state["lambda_log"][0] = noise_log_mean[:, None]
    noise_array_type = np.matrix if matrix_noise else np.asarray
    state["lambda_R"][0] = noise_array_type(noise_mean[:, None])
    state["lambda_log_R"][0] = noise_array_type(noise_log_mean[:, None])
    state["HlambdaHt"][0] = (h * noise_mean) @ h.T + sum(
        noise_mean[r] * h_covariance[:, :, r] for r in range(data.shape[1]))

    # E[(Y - sum_i X_i W_i H_i)^2] uses independent factor second moments.
    expected_sse = np.zeros(data.shape[1])
    for r in range(data.shape[1]):
        h_moment = np.outer(h[:, r], h[:, r]) + h_covariance[:, :, r]
        for n in range(data.shape[0]):
            x_moment = np.outer(x[n], x[n]) + np.diag(x_variance[n])
            prediction_mean = np.sum(x[n] * w * h[:, r])
            prediction_second = np.einsum("ij,ij,ij", x_moment, w_moment, h_moment)
            expected_sse[r] += (data[n, r] ** 2 - 2 * data[n, r] * prediction_mean +
                                prediction_second)
    expected = decimation / 2 * np.sum(
        data.shape[0] * (noise_log_mean - np.log(2 * np.pi)) - noise_mean * expected_sse)
    _, terms = flica_vb.compute_F(state)
    actual = math.fsum(float(np.asarray(terms[f"Ylike{i}"][0]).sum())
                       for i in range(1, 5))
    assert actual == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("large_shape", [False, True])
def test_gamma_posterior_entropies_match_scipy_including_large_shape(large_shape):
    state = _variational_state(1, subjectwise=True)
    if large_shape:
        for prefix, dimensions in (("eta", (2, 1)), ("beta", (3, 2)),
                                   ("lambda", (4, 1))):
            shape, rate = np.full(dimensions, 1e6), np.full(dimensions, 123.4)
            if prefix == "eta":
                state.update(eta_c=shape, eta_binv=rate, eta=shape / rate,
                             eta_log=digamma(shape) - np.log(rate))
            else:
                state[prefix + "_c"][0], state[prefix + "_binv"][0] = shape, rate
                state["Lambda" if prefix == "lambda" else prefix][0] = shape / rate
                state[prefix + "_log"][0] = digamma(shape) - np.log(rate)
    _, terms = flica_vb.compute_F(state)
    for prefix in ("eta", "beta", "lambda"):
        shape = state[prefix + "_c"] if prefix == "eta" else state[prefix + "_c"][0]
        rate = state[prefix + "_binv"] if prefix == "eta" else state[prefix + "_binv"][0]
        expected = gamma.entropy(a=shape, scale=1 / rate).sum()
        actual = np.asarray(terms[prefix + "Post"]).sum()
        # At shape 1e6, subtracting O(1e7) terms loses a few float64 digits.
        np.testing.assert_allclose(actual, expected, rtol=1e-12,
                                   atol=2e-8 if large_shape else 1e-12)


def test_dirichlet_entropy_includes_every_component():
    state = _variational_state()
    _, terms = flica_vb.compute_F(state)
    expected = math.fsum(dirichlet(column).entropy()
                         for column in state["pi_weights"][0].T)
    np.testing.assert_allclose(terms["piPost"], expected, rtol=1e-12)


@pytest.mark.parametrize("modalities", [1, 3])
def test_free_energy_counts_each_global_and_modality_term_once(modalities):
    total, terms = flica_vb.compute_F(_variational_state(modalities))
    expected = math.fsum(float(value) for term in terms.values()
                         for value in np.asarray(term).ravel())
    assert np.asarray(total).ndim == 0
    assert float(total) == pytest.approx(expected, rel=1e-12)


def test_duplicating_modalities_does_not_duplicate_global_entropy_and_priors():
    one, terms = flica_vb.compute_F(_variational_state(1))
    three, _ = flica_vb.compute_F(_variational_state(3))
    global_keys = {"Hprior", "Hpost", "etaPrior", "etaPost"}
    modality_terms = math.fsum(float(value) for key, term in terms.items()
                               if key not in global_keys
                               for value in np.asarray(term).ravel())
    assert float(three) == pytest.approx(float(one) + 2 * modality_terms, rel=1e-12)


def test_cpu_covariance_helpers_preserve_float64_precision():
    diagonals = np.array([[[1.1, 1.3]], [[2.2, 2.7]]], dtype=np.float64)
    matrices = flica_vb.apply3_diag(diagonals)
    extracted = flica_vb.apply3_diag2(matrices)
    inverse = flica_vb.apply3_inv_prescale(matrices)
    determinants = flica_vb.apply3_logdet(matrices, "chol")
    for value in (matrices, extracted, inverse, determinants):
        assert value.dtype == np.float64
    np.testing.assert_array_equal(extracted, diagonals[:, 0, :])
    for layer in range(matrices.shape[2]):
        np.testing.assert_allclose(inverse[:, :, layer],
                                   np.linalg.inv(matrices[:, :, layer]), rtol=1e-12)
        np.testing.assert_allclose(determinants[0, 0, layer],
                                   np.linalg.slogdet(matrices[:, :, layer])[1], rtol=1e-12)


@pytest.mark.parametrize("compute_energy", [0, 1])
def test_free_energy_history_has_one_entry_per_posterior_update(
        tmp_path, monkeypatch, compute_energy):
    data = [np.array([[1., 2., -1., 0.], [2., -1., 1., 3.],
                      [0., 2., 4., 1.]])]
    options = {"num_components": 2, "maxits": 3, "lambda_dims": "R",
               "initH": np.array([[1., 0., 2., 1.], [0., 1., 1., -1.]]),
               "dof_per_voxel": np.ones(1), "computeF": compute_energy,
               "output_dir": str(tmp_path)}
    priors, posterior, constants = flica_vb.flica_init_params(data, options)
    updates, energies = [], []
    original_update = flica_vb.update_H

    def record_update(state):
        updates.append(len(updates) + 1)
        return original_update(state)

    def energy(state):
        value = float(len(energies) + 1)
        energies.append(value)
        return value, {"test_energy": value}

    monkeypatch.setattr(flica_vb, "update_H", record_update)
    monkeypatch.setattr(flica_vb, "compute_F", energy)
    fitted = flica_vb.flica_iterate(data, options, priors, posterior, constants)
    history = np.asarray(fitted["F_history"])
    assert len(updates) == options["maxits"]
    assert history.shape == (len(updates),)
    assert fitted["F"] == history[-1]
    if compute_energy:
        np.testing.assert_array_equal(history, energies)
    else:
        assert len(energies) == 1
        assert np.isnan(history[:-1]).all()
        assert history[-1] == energies[0]
