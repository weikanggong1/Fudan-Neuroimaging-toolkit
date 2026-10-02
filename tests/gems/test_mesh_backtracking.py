"""Backtracking mesh contracts, separate from real imaging benchmarks."""
import pytest
import torch

from fnit.gems import core
from test_mesh_precision import _model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_backtracking_fit_decreases_objective_preserves_mesh_and_precision(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        model, image, fixed = _model(device)
        result = model(image, fixed_gaussians=fixed, em_iterations=1,
                       deform_iterations=8, deform_optimizer="lbfgs", deform_lr=1.,
                       stable_mesh_fitting=True, precise_mesh_matrices=True,
                       mesh_line_search="backtracking", materialize_outputs=False)
        stats = result.optimization_stats
        costs = torch.tensor(result.objective_history[1:], dtype=torch.float64)
        assert stats["mesh_line_search"] == "backtracking"
        assert stats["optimizer_state_precision"] == "float64"
        assert stats["line_search_restarts"] == 0
        assert result.vertices.dtype == torch.float32
        assert result.gaussian_parameters.means.dtype == torch.float32
        assert result.min_jacobian > 0
        assert torch.isfinite(result.vertices).all()
        assert len(costs) > 1 and costs[-1] < costs[0]
        assert torch.all(costs[1:] <= costs[:-1])
        assert torch.backends.cuda.matmul.allow_tf32
        if device == "cuda":
            assert stats["fused_data_evaluations"] > 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_backtracking_rejects_inverted_trials_even_with_downhill_data(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model, image, fixed = _model(device)
    original = core.ashburner_prior
    observed = []

    def downhill_invalid_prior(*args, **kwargs):
        value, jacobian = original(*args, **kwargs)
        if value.requires_grad:
            observed.append(float(jacobian.min()))
            # The first accepted point is valid; every trial is marked invalid
            # even if its data objective would decrease. This isolates the
            # independent positive-J acceptance guard from the large penalty.
            if len(observed) > 1:
                return value * 0 - 1e20, -torch.ones_like(jacobian)
        return value, jacobian

    monkeypatch.setattr(core, "ashburner_prior", downhill_invalid_prior)
    initial = torch.as_tensor(model.atlas.vertices, device=device, dtype=torch.float32)
    result = model(image, fixed_gaussians=fixed, em_iterations=1,
                   deform_iterations=2, deform_optimizer="lbfgs", deform_lr=1.,
                   stable_mesh_fitting=True, precise_mesh_matrices=True,
                   mesh_line_search="backtracking", materialize_outputs=False)
    assert len(observed) > 1
    assert torch.equal(result.vertices, initial)
    assert result.min_jacobian > 0
    assert torch.isfinite(torch.tensor(result.objective_history)).all()


def test_backtracking_requires_supported_stable_cached_configuration():
    model, image, fixed = _model("cpu")
    for options in ({"stable_mesh_fitting": False}, {"cache_mesh_evaluations": False},
                    {"deform_optimizer": "adam"}):
        kwargs = {"stable_mesh_fitting": True, "deform_optimizer": "lbfgs",
                  "mesh_line_search": "backtracking", **options}
        with pytest.raises(ValueError, match="stable cached"):
            model(image, fixed_gaussians=fixed, **kwargs)
