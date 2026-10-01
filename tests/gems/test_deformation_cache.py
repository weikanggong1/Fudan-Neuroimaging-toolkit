import pytest
import torch

from fnit.gems.deformation import ashburner_prior, prepare_deformation_reference


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _mesh(device):
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                             [2., 0., 0.], [3., 0., 0.], [2., 1., 0.], [2., 0., 1.]],
                            device=device, dtype=torch.float32)
    rotation = vertices.new_tensor([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    transform = rotation @ torch.diag(vertices.new_tensor([2., .7, 1.3]))
    reference = vertices @ transform.T + vertices.new_tensor([3., -1., 2.])
    # Opposite tetrahedron orientations exercise the absolute reference volume.
    tetrahedra = torch.tensor([[0, 1, 2, 3], [4, 6, 5, 7]], device=device)
    return reference, tetrahedra


def _value_and_gradient(vertices, reference, tetrahedra, stiffness, geometry=None):
    positions = vertices.detach().clone().requires_grad_(True)
    cost, jacobian = ashburner_prior(positions, reference, tetrahedra, stiffness,
                                    reference_geometry=geometry)
    gradient, = torch.autograd.grad(cost, positions)
    return cost.detach(), jacobian.detach(), gradient


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("stiffness", [0., .05, .8])
@pytest.mark.parametrize("invalid", [None, "inverted", "collapsed"])
def test_cached_prior_matches_value_jacobian_and_vertex_gradient(device, stiffness, invalid):
    reference, tetrahedra = _mesh(device)
    vertices = reference.clone()
    vertices[1] += vertices.new_tensor([.15, -.07, .03])
    if invalid == "inverted":
        vertices[[5, 6]] = vertices[[6, 5]]
    elif invalid == "collapsed":
        vertices[7] = vertices[4]
    geometry = prepare_deformation_reference(reference, tetrahedra)
    assert geometry.inverse_edges.dtype == reference.dtype
    assert geometry.inverse_edges.device == reference.device
    torch.testing.assert_close(geometry.volumes, reference.new_full((2,), 2. * .7 * 1.3 / 6))
    ordinary = _value_and_gradient(vertices, reference, tetrahedra, stiffness)
    cached = _value_and_gradient(vertices, reference, tetrahedra, stiffness, geometry)
    for actual, expected in zip(cached, ordinary):
        torch.testing.assert_close(actual, expected)
    if invalid is not None:
        assert cached[1][1] <= 0
        assert cached[0] >= 1e12


@pytest.mark.parametrize("device", _DEVICES)
def test_reference_cache_can_be_reused_across_backward_calls(device):
    reference, tetrahedra = _mesh(device)
    geometry = prepare_deformation_reference(reference.requires_grad_(True), tetrahedra)
    assert not geometry.inverse_edges.requires_grad
    assert not geometry.volumes.requires_grad
    for displacement in (.03, .2):
        vertices = reference.detach().clone()
        vertices[3, 2] += displacement
        ordinary = _value_and_gradient(vertices, reference.detach(), tetrahedra, .05)
        cached = _value_and_gradient(vertices, reference, tetrahedra, .05, geometry)
        for actual, expected in zip(cached, ordinary):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", _DEVICES)
def test_degenerate_reference_still_raises(device):
    reference, tetrahedra = _mesh(device)
    reference[3] = reference[0]
    with pytest.raises(torch.linalg.LinAlgError):
        ashburner_prior(reference.clone(), reference, tetrahedra, .05)
    with pytest.raises(torch.linalg.LinAlgError):
        prepare_deformation_reference(reference, tetrahedra)


@pytest.mark.parametrize("device", _DEVICES)
def test_empty_reference_cache_keeps_zero_cost_and_gradient(device):
    reference, _ = _mesh(device)
    tetrahedra = torch.empty((0, 4), device=device, dtype=torch.long)
    geometry = prepare_deformation_reference(reference, tetrahedra)
    ordinary = _value_and_gradient(reference, reference, tetrahedra, .05)
    cached = _value_and_gradient(reference, reference, tetrahedra, .05, geometry)
    for actual, expected in zip(cached, ordinary):
        torch.testing.assert_close(actual, expected)
    assert cached[0] == 0
    assert cached[1].numel() == 0
    assert torch.count_nonzero(cached[2]) == 0
