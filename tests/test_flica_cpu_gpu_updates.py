"""FLICA update regressions; generated tensors are not benchmark evidence."""

import contextlib
import copy
import io

import numpy as np
import pytest
import torch

from fnit.bigflica import flica_vb
from fnit.bigflica.flica_torch import iterate_flica_torch


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
def test_flica_r_complete_updates_from_same_state(flica_r_state, max_iter):
    data, options, priors, posteriors, constants = flica_r_state
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
