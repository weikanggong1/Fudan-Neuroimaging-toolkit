import numpy as np
import pytest
import torch

from fnit.gems import GEMSAtlas, TorchGEMS
from fnit.gems import _raster_triton
from fnit.gems.deformation import prepare_current_geometry
from fnit.gems.rasterize import (build_block_index, compact_data_cost,
                                 rasterize_priors_compact)


def _inputs(device="cpu", classes=3, sparse=False):
    vertices = torch.tensor([[1.137, 1.211, 1.317], [6.137, 1.211, 1.317],
                             [1.137, 6.211, 1.317], [1.137, 1.211, 6.317],
                             [6.137, 6.211, 6.317]], device=device)
    tetra = torch.tensor([[0, 1, 2, 3], [4, 1, 2, 3]], device=device)
    alpha = torch.arange(1, 5 * classes + 1, device=device, dtype=torch.float32).reshape(5, classes)
    if sparse:
        alpha[:, 0] = 0
        alpha[0] = 0
        alpha[1, 1:] = 0
    alpha = alpha / alpha.sum(1, keepdim=True).clamp_min(1)
    shape = (10, 9, 8)
    valid = torch.arange(np.prod(shape), device=device).reshape(shape) % 3 != 0
    # Includes outside-index voxels as well as cells and missing-cell blocks.
    index = build_block_index(vertices.cpu().numpy(), tetra.cpu().numpy(), shape, block_size=2)
    likelihood = -torch.linspace(.1, 14, classes * int(valid.sum()), device=device).reshape(classes, -1)
    return vertices, tetra, alpha, shape, valid, index, likelihood


def _evaluate(inputs, fused, background):
    initial, tetra, alpha, shape, valid, index, likelihood = inputs
    vertices = initial.detach().clone().requires_grad_(True)
    geometry = prepare_current_geometry(vertices, tetra)
    if fused:
        cost = compact_data_cost(vertices, tetra, alpha, shape, valid_mask=valid,
            block_index=index, likelihood=likelihood, background_channel=background,
            current_geometry=geometry)
        assert cost is not None
    else:
        priors, _ = rasterize_priors_compact(vertices, tetra, alpha, shape,
            valid_mask=valid, block_index=index, background_channel=background,
            current_geometry=geometry)
        cost = -(priors.clamp_min(torch.finfo(priors.dtype).tiny).log() + likelihood).logsumexp(0).sum()
    gradient, = torch.autograd.grad(cost, vertices)
    return cost.detach(), gradient.detach()


def test_cpu_fused_data_cost_requests_autograd_fallback():
    vertices, tetra, alpha, shape, valid, index, likelihood = _inputs()
    assert compact_data_cost(vertices, tetra, alpha, shape, valid_mask=valid,
        block_index=index, likelihood=likelihood) is None


@pytest.fixture
def cuda():
    if not torch.cuda.is_available() or _raster_triton.triton is None:
        pytest.skip("CUDA and Triton required")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield "cuda"
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("classes", [2, 3, 8, 19])
@pytest.mark.parametrize("background", [None, 0, 1])
@pytest.mark.parametrize("sparse", [False, True])
def test_fused_fixed_mixture_cost_and_vertex_gradient_match_autograd(cuda, classes, background, sparse):
    inputs = _inputs(cuda, classes, sparse)
    reference = _evaluate(inputs, False, background)
    actual = _evaluate(inputs, True, background)
    assert torch.isfinite(actual[0]) and torch.isfinite(actual[1]).all()
    torch.testing.assert_close(actual[0], reference[0], atol=.001, rtol=2e-6)
    reference_norm = torch.linalg.vector_norm(reference[1])
    difference_norm = torch.linalg.vector_norm(actual[1] - reference[1])
    # A single nonzero alpha class makes this objective constant in the mesh;
    # division/reduction rounding then needs an absolute zero-gradient gate.
    if float(reference_norm) < 1e-5:
        assert float(difference_norm) < 2e-7
    else:
        assert float(difference_norm / reference_norm) < 2e-4


def test_fused_data_cost_rejects_differentiable_alphas_or_likelihood(cuda):
    vertices, tetra, alpha, shape, valid, index, likelihood = _inputs(cuda)
    for a, ll in [(alpha.requires_grad_(True), likelihood), (alpha.detach(), likelihood.requires_grad_(True))]:
        assert compact_data_cost(vertices, tetra, a, shape, valid_mask=valid,
            block_index=index, likelihood=ll) is None


def _atlas():
    vertices = np.asarray([[1, 1, 1], [6, 1, 1], [1, 6, 1], [1, 1, 6]], float)
    tetra = np.asarray([[0, 1, 2, 3]], np.int64)
    alphas = np.asarray([[1, 0], [0, 1], [0, 1], [0, 1]], np.float32)
    return GEMSAtlas(vertices, vertices, tetra, alphas, .1,
                     np.ones((4, 3), bool), np.asarray([0, 10]), ("Unknown", "ROI"))


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_skipping_intermediate_outputs_preserves_fit_and_gaussians(optimizer):
    image = torch.ones((8, 8, 8))
    image[2:5, 2:5, 2:5] = 2
    options = dict(em_iterations=2, deform_iterations=3, deform_optimizer=optimizer,
                   deform_lr=.03, outer_iterations=2)
    model = TorchGEMS(_atlas())
    dense = model(image, **options)
    intermediate = model(image, materialize_outputs=False, **options)
    assert intermediate.labels is intermediate.posterior is intermediate.priors is None
    torch.testing.assert_close(intermediate.vertices, dense.vertices, rtol=0, atol=0)
    torch.testing.assert_close(intermediate.gaussian_parameters.means,
                               dense.gaussian_parameters.means, rtol=0, atol=0)
    torch.testing.assert_close(intermediate.gaussian_parameters.covariances,
                               dense.gaussian_parameters.covariances, rtol=0, atol=0)
    assert intermediate.objective_history == dense.objective_history
    assert intermediate.min_jacobian == dense.min_jacobian
    with pytest.raises(ValueError, match="not materialized"):
        intermediate.mask(10)


@pytest.mark.parametrize("stride", [0, -2, 1.5])
def test_invalid_mesh_sampling_stride_is_rejected(stride):
    with pytest.raises(ValueError, match="positive integer"):
        TorchGEMS(_atlas())(torch.ones((8, 8, 8)), mesh_sampling_stride=stride)


def test_mesh_quadrature_keeps_full_em_and_output_shape():
    image = torch.ones((8, 8, 8))
    image[2:5, 2:5, 2:5] = 2
    model = TorchGEMS(_atlas())
    full = model(image, em_iterations=2, deform_iterations=0)
    sampled = model(image, em_iterations=2, deform_iterations=0, mesh_sampling_stride=2)
    torch.testing.assert_close(sampled.priors, full.priors, rtol=0, atol=0)
    torch.testing.assert_close(sampled.posterior, full.posterior, rtol=0, atol=0)
    assert torch.equal(sampled.labels, full.labels)
    assert sampled.optimization_stats["mesh_sampling_scale"] == 2

