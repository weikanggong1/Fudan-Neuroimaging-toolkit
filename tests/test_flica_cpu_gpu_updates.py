"""FLICA update regressions; generated tensors are not benchmark evidence."""

import contextlib
import copy
import io

import h5py
import numpy as np
import pytest
import torch

from fnit.bigflica import flica_vb
from fnit.bigflica.flica_torch import (RawVoxelMatrix, initialize_flica_raw,
                                      initialize_flica_torch,
                                      iterate_flica_torch)


@pytest.fixture
def flica_r_state(tmp_path):
    generator = np.random.default_rng(917)
    data = [generator.normal(size=(width, 10)) for width in (15, 18, 21)]
    options = {"num_components": 3, "maxits": 1, "lambda_dims": "R",
               "initH": generator.normal(size=(3, 10)),
               "dof_per_voxel": np.array([1., 1.5, .75]),
               "computeF": 1, "output_dir": str(tmp_path)}
    with contextlib.redirect_stdout(io.StringIO()):
        priors, posteriors, constants = flica_vb.flica_init_params(
            copy.deepcopy(data), options)
    # Different column energies exercise subjectwise noise.
    assert all(np.ptp(np.asarray(value)) > .1
               for value in posteriors["Lambda"])
    return data, options, priors, posteriors, constants


def test_flica_subjectwise_covariance_preserves_float64(flica_r_state):
    data, options, priors, posteriors, constants = flica_r_state
    state = copy.deepcopy({**priors, **posteriors, **constants,
                           "opts": options, "Y": data})
    state.update(flica_vb.update_eta(state))
    result = flica_vb.update_H(state)
    assert result["H_colcov"].dtype == np.dtype("float64")
    assert flica_vb.apply3_diag2(result["H_colcov"]).dtype == np.dtype("float64")
    diagonal = np.asarray(state["eta"])[:, None, :]
    assert flica_vb.apply3_diag(diagonal).dtype == np.dtype("float64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("max_iter", [2, 6])
@pytest.mark.parametrize("prior_format", ["per_modality", "legacy_scalar"])
def test_flica_r_complete_updates_from_same_state(flica_r_state, max_iter,
                                                 prior_format):
    data, options, priors, posteriors, constants = flica_r_state
    if prior_format == "legacy_scalar":
        priors["prior_W_var"] = 1 / constants["DD"][-1]
    options["maxits"] = max_iter
    with contextlib.redirect_stdout(io.StringIO()):
        expected = flica_vb.flica_iterate(
            copy.deepcopy(data), options, copy.deepcopy(priors),
            copy.deepcopy(posteriors), copy.deepcopy(constants))
    actual = iterate_flica_torch(
        data, copy.deepcopy(priors), copy.deepcopy(posteriors),
        dict(constants, lambda_dims="R"), max_iter, device="cuda:0")
    assert len(expected["F_history"]) == max_iter
    assert np.isfinite(expected["F_history"]).all()
    assert np.linalg.norm(expected["H"], axis=1).min() > 1e-8
    # Identical initial states allow comparison without component matching.
    for key in ("H", "H_PCs"):
        np.testing.assert_allclose(actual[key], expected[key],
                                   rtol=3e-10, atol=3e-12, equal_nan=False,
                                   err_msg=key)
    for key in ("X", "W", "lambda"):
        assert len(actual[key]) == len(expected[key]) == len(data)
        for modality, (actual_value, expected_value) in enumerate(
                zip(actual[key], expected[key])):
            expected_value = np.asarray(expected_value)
            if key in ("W", "lambda"):
                expected_value = expected_value.reshape(-1)
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=3e-10, atol=3e-12,
                equal_nan=False, err_msg=f"{key}, modality {modality}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("lambda_dims", ["o", "R"])
def test_flica_gpu_pca_matches_matlab_rms(tmp_path, raw, lambda_dims):
    generator = np.random.default_rng(123)
    data = [generator.normal(size=(width, 10)) * scale
            for width, scale in zip((15, 18, 21), (1., 2., .5))]
    with contextlib.ExitStack() as stack:
        if raw:
            matrices = []
            for index, value in enumerate(data):
                handle = stack.enter_context(h5py.File(tmp_path / f"{index}.h5", "w"))
                matrices.append(RawVoxelMatrix(handle.create_dataset("data", data=value.T), 4))
            initialize = initialize_flica_raw
        else:
            matrices = data
            initialize = initialize_flica_torch
        priors, state, constants = initialize(matrices, 3, lambda_dims=lambda_dims)
        dd = constants["DD"].cpu().numpy()
        weighted = np.vstack([value * np.sqrt(dd[k]) for k, value in enumerate(data)])
        u, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
        h_pre = singular_values[:3, None] * vt[:3] / np.sqrt(len(weighted))
        expected_h = h_pre / np.sqrt(dd.mean())
        actual_h = state["H"].cpu().numpy()
        signs = np.sign(np.sum(actual_h * expected_h, axis=1))
        np.testing.assert_allclose(actual_h * signs[:, None], expected_h,
                                   rtol=3e-10, atol=3e-12)
        np.testing.assert_allclose(priors["prior_W_var"].cpu(), 1 / dd)
        spatial = np.vstack([value.cpu().numpy() for value in state["X"]])
        np.testing.assert_allclose(spatial * signs, u[:, :3] * np.sqrt(len(weighted)),
                                   rtol=3e-10, atol=3e-12)
        np.testing.assert_allclose(np.mean(spatial ** 2, axis=0), np.ones(3), atol=3e-12)
        for k, value in enumerate(data):
            weights = state["W"][k].cpu().numpy()
            np.testing.assert_allclose(weights, np.sqrt(dd.mean() / dd[k]))
            reconstruction = (state["X"][k].cpu().numpy() * weights) @ actual_h
            np.testing.assert_allclose(reconstruction, value @ vt[:3].T @ vt[:3],
                                       rtol=3e-10, atol=3e-12)
        # The default and explicit pre-scaled H paths must create the same state.
        _, numeric_state, _ = initialize(
            matrices, 3, lambda_dims=lambda_dims,
            init_h=actual_h * np.sqrt(dd.mean()))
        for key in ("H", "H2Gmat"):
            torch.testing.assert_close(numeric_state[key], state[key], rtol=3e-10, atol=3e-12)
        for key in ("X", "W", "XtDX", "Lambda"):
            for numeric, default in zip(numeric_state[key], state[key]):
                torch.testing.assert_close(numeric, default, rtol=3e-10, atol=3e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flica_gpu_w_update_uses_each_modality_prior(flica_r_state):
    data, _, priors, state, constants = flica_r_state
    dd = np.asarray(constants["DD"])
    components, subjects = state["H"].shape
    h2 = np.asarray(state["H2Gmat"]).reshape(-1)
    eta = (priors["prior_eta_c"] + subjects / 2) / (1 / priors["prior_eta_b"] + h2 / 2)
    precision = np.broadcast_to(np.diag(eta), (subjects, components, components)).copy()
    mean_term = np.zeros((components, subjects))
    crosses = [np.asarray(state["X"][k]).T @ value for k, value in enumerate(data)]
    for k in range(len(data)):
        lam = np.asarray(state["Lambda"][k]).reshape(-1)
        precision += lam[:, None, None] * (state["WtW"][k] * state["XtDX"][k].T)[None]
        mean_term += dd[k] * np.asarray(state["W"][k]).reshape(-1, 1) * crosses[k] * lam
    h_cov = np.linalg.inv(precision)
    h = np.einsum("rij,jr->ir", h_cov, mean_term)
    expected = []
    for k in range(len(data)):
        lam = np.asarray(state["Lambda"][k]).reshape(-1)
        weighted_h = (h * lam) @ h.T + np.einsum("r,rij->ij", lam, h_cov)
        w_cov = np.linalg.inv(state["XtDX"][k] * weighted_h + dd[k] * np.eye(components))
        target = dd[k] * np.diag((crosses[k] * lam) @ h.T)
        expected.append(target @ w_cov)
    actual = iterate_flica_torch(
        data, priors, state, dict(constants, lambda_dims="R"), 1)
    for k, value in enumerate(expected):
        np.testing.assert_allclose(actual["W"][k], value, rtol=3e-10, atol=3e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("prior", [np.array([1., 2.]), 0., np.nan, np.inf])
def test_flica_gpu_rejects_invalid_w_prior(flica_r_state, prior):
    data, _, priors, state, constants = flica_r_state
    priors["prior_W_var"] = prior
    with pytest.raises(ValueError, match="prior_W_var"):
        iterate_flica_torch(data, priors, state, dict(constants, lambda_dims="R"), 1)
