"""Numerical contracts for mesh matrices; these small meshes are not benchmarks."""

import numpy as np
import pytest
import torch

from fnit.gems import GEMSAtlas, TorchGEMS
from fnit.gems import core
from fnit.gems.deformation import (ashburner_prior, prepare_current_geometry,
                                  prepare_deformation_reference)
from fnit.gems.gaussian import GaussianParameters


@pytest.fixture
def cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    previous = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", previous)
    return "cuda"


def _prior_mesh(device):
    count = 256
    edges = torch.tensor([[1.0137, .2134, -.1211], [.0734, .9341, .1572],
                          [.0413, -.1921, 1.1457]], device=device)
    index = torch.arange(count, device=device)
    origins = torch.stack((index % 8, (index // 8) % 8, index // 64), 1).float() * 4 + 20
    reference = torch.cat((origins[:, None], origins[:, None] + edges.T[None]), 1).reshape(-1, 3)
    tetra = torch.arange(4 * count, device=device).reshape(count, 4)
    perturbation = reference.new_tensor([[0, 0, 0], [.001, -.0005, .0003],
                                         [-.0002, .0007, .0002], [.0002, -.0003, -.0008]])
    direction = reference.new_tensor([[0, 0, 0], [.3, -.4, .2],
                                      [-.2, .1, .3], [.1, .2, -.1]])
    return reference, tetra, perturbation.repeat(count, 1), direction.repeat(count, 1)


def _prior_value_gradient(vertices, reference, tetra, *, analytic):
    vertices = vertices.detach().clone().requires_grad_(True)
    reference_geometry = prepare_deformation_reference(reference, tetra)
    current = prepare_current_geometry(vertices, tetra, deterministic_gradient=True)
    value, _ = ashburner_prior(vertices, reference, tetra, .05,
        reference_geometry=reference_geometry, current_geometry=current,
        analytic_gradient=analytic, double_accumulation=True)
    gradient, = torch.autograd.grad(value, vertices)
    return value.detach(), gradient


@pytest.mark.parametrize("near_identity", [False, True])
def test_precise_float32_prior_gradient_agrees_with_double_oracle(cuda, near_identity):
    reference, tetra, perturbation, direction = _prior_mesh(cuda)
    initial = reference + perturbation if near_identity else reference
    torch.backends.cuda.matmul.allow_tf32 = False
    actual_cost, actual_gradient = _prior_value_gradient(initial, reference, tetra, analytic=True)
    oracle_cost, oracle_gradient = _prior_value_gradient(initial.double(), reference.double(),
                                                        tetra, analytic=False)
    assert actual_gradient.dtype == torch.float32
    torch.testing.assert_close(actual_cost, oracle_cost, atol=3e-6, rtol=0)
    torch.testing.assert_close(actual_gradient.double(), oracle_gradient, atol=2e-7, rtol=2e-3)
    actual_derivative = (actual_gradient.double() * direction.double()).sum()
    oracle_derivative = (oracle_gradient * direction.double()).sum()
    torch.testing.assert_close(actual_derivative, oracle_derivative, atol=2e-6, rtol=5e-4)

    # An independent FP64 finite difference checks the oracle at both points;
    # FP32 cost cancellation near identity makes very small finite steps noisy.
    step = 1e-5
    plus, _ = _prior_value_gradient(initial.double() + step * direction.double(),
                                    reference.double(), tetra, analytic=False)
    minus, _ = _prior_value_gradient(initial.double() - step * direction.double(),
                                     reference.double(), tetra, analytic=False)
    torch.testing.assert_close((plus - minus) / (2 * step), oracle_derivative,
                               atol=1e-7, rtol=1e-4)


def _model(device):
    vertices = np.asarray([[1.137, 1.211, 1.317], [6.137, 1.211, 1.317],
                           [1.137, 6.211, 1.317], [1.137, 1.211, 6.317]])
    alpha = np.asarray([[.7, .3], [.1, .9], [.2, .8], [.3, .7]], np.float32)
    atlas = GEMSAtlas(vertices, vertices, np.asarray([[0, 1, 2, 3]], np.int64),
                      alpha, .05, np.ones((4, 3), bool), np.asarray([0, 10]),
                      ("Unknown", "ROI"))
    model = TorchGEMS(atlas, device=device)
    image = torch.ones((8, 8, 8), device=device)
    image[2:5, 2:5, 2:5] = 2
    fixed = GaussianParameters(image.new_tensor([[1.], [2.]]),
                               image.new_full((2, 1, 1), .01))
    return model, image, fixed


@pytest.mark.parametrize("initial_tf32", [False, True])
@pytest.mark.parametrize("precise", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_mesh_precision_covers_forward_backward_and_restores_tf32(
        cuda, monkeypatch, initial_tf32, precise, cached):
    model, image, fixed = _model(cuda)
    torch.backends.cuda.matmul.allow_tf32 = initial_tf32
    forward_modes, backward_modes = [], []
    original = core.ashburner_prior

    def inspect_prior(*args, **kwargs):
        value, jacobian = original(*args, **kwargs)
        if value.requires_grad:
            forward_modes.append(torch.backends.cuda.matmul.allow_tf32)

            def inspect_backward(gradient):
                backward_modes.append(torch.backends.cuda.matmul.allow_tf32)
                return gradient

            value.register_hook(inspect_backward)
        return value, jacobian

    monkeypatch.setattr(core, "ashburner_prior", inspect_prior)
    result = model(image, fixed_gaussians=fixed, em_iterations=1, deform_iterations=2,
                   deform_optimizer="lbfgs", deform_lr=.2, materialize_outputs=False,
                   cache_mesh_evaluations=cached, stable_mesh_fitting=True,
                   precise_mesh_matrices=precise)
    expected_mode = False if precise else initial_tf32
    assert forward_modes and backward_modes
    assert all(mode == expected_mode for mode in forward_modes + backward_modes)
    assert torch.backends.cuda.matmul.allow_tf32 == initial_tf32
    assert result.vertices.dtype == torch.float32
    assert result.optimization_stats["precise_mesh_matrices"] == precise
    assert result.optimization_stats["fused_data_evaluations"] > 0


@pytest.mark.parametrize("initial_tf32", [False, True])
def test_mesh_precision_restores_tf32_after_failed_closure(cuda, monkeypatch, initial_tf32):
    model, image, fixed = _model(cuda)
    torch.backends.cuda.matmul.allow_tf32 = initial_tf32

    def fail_prior(*args, **kwargs):
        assert not torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("mesh trial failed")

    monkeypatch.setattr(core, "ashburner_prior", fail_prior)
    with pytest.raises(RuntimeError, match="mesh trial failed"):
        model(image, fixed_gaussians=fixed, em_iterations=1, deform_iterations=2,
              deform_optimizer="lbfgs", stable_mesh_fitting=True, precise_mesh_matrices=True)
    assert torch.backends.cuda.matmul.allow_tf32 == initial_tf32
