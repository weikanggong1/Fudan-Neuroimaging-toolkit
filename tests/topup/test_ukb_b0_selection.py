"""Regression checks for the score used to select raw UKB b0 frames."""

import numpy as np
import pytest
import torch

from fnit.topup import ukb


@pytest.mark.parametrize("shape", [(8, 10, 12), (16, 18, 20)])
def test_registered_correlation_scores_accepted_not_last_rejected_trial(monkeypatch, shape):
    observations = {}

    class RejectedLastTrialLBFGS:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters[0]

        def zero_grad(self, set_to_none):
            self.parameters.grad = None

        def step(self, closure):
            observations["accepted_score"] = 1 - float(closure().detach())
            with torch.no_grad():
                self.parameters[0] = 4.0
            observations["last_trial_score"] = 1 - float(closure().detach())
            # Strong Wolfe may return an earlier low bracket without evaluating
            # the closure again at the accepted parameters.
            with torch.no_grad():
                self.parameters.zero_()

    monkeypatch.setattr(torch.optim, "LBFGS", RejectedLastTrialLBFGS)
    values = np.random.default_rng(1729).uniform(1, 10, size=shape).astype(np.float32)
    score = ukb._registered_correlation(values, values, (2.0, 2.0, 2.0), "cpu")
    assert observations["accepted_score"] == pytest.approx(1.0, abs=1e-6)
    assert observations["last_trial_score"] < 0.9
    assert score == pytest.approx(observations["accepted_score"], abs=1e-6)
    assert score != pytest.approx(observations["last_trial_score"], abs=1e-3)


@pytest.mark.parametrize("first_score, expected", [(0.98, 0), (0.979, 1)])
def test_best_b0_keeps_first_frame_threshold_and_float64_means(monkeypatch, first_score, expected):
    correlations = iter((first_score, first_score, 1.0))
    monkeypatch.setattr(ukb, "_registered_correlation", lambda *args: next(correlations))
    candidates = np.ones((3, 3, 3, 3), dtype=np.float32)
    chosen, scores = ukb._best_b0(candidates, (2.0, 2.0, 2.0), "cpu")
    assert chosen == expected
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, [first_score, (first_score + 1) / 2, (first_score + 1) / 2])
