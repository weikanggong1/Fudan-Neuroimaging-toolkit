"""CPU literal geometry and thread-budget gates; not scientific benchmarks."""
import numpy as np
import pytest
import torch

from fnit.msm._execution import cpu_workers, execution_budget
from fnit.msm._sphere_map import RadialSphereMap, _projection_geometry
from fnit.msm.msmsulc import _ico


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('source_precision', [False, True])
def test_native_cpu_selection_matches_reference_at_planes_and_shared_edges(threads, source_precision):
    vertices, faces = _ico(2)
    vertices = vertices.copy()
    vertices[0] = .7*vertices[0] + .3*vertices[1]
    vertices *= 100/np.linalg.norm(vertices, axis=1, keepdims=True)
    rng = np.random.default_rng(924)
    points = rng.normal(size=(711, 3))
    points *= 100/np.linalg.norm(points, axis=1, keepdims=True)
    points = np.r_[points, vertices, vertices[faces[:31]].mean(axis=1)]
    points[-17:, 0] += 1e-13
    with execution_budget(threads):
        mapper = RadialSphereMap(vertices, faces, 'cpu', source_precision=source_precision)
        reference = RadialSphereMap(vertices, faces, 'cpu', execution='reference',
                                    source_precision=source_precision)
        expected = reference.weights(torch.as_tensor(points), batch_size=113)
        actual = mapper.weights(torch.as_tensor(points), batch_size=71)
    for first, second in zip(expected, actual):
        assert torch.equal(first, second)


def test_native_selection_failure_restores_budget_and_keeps_torch_thread_pool():
    previous = cpu_workers()
    torch_threads = torch.get_num_threads()
    vertices, faces = _ico(1)
    with pytest.raises(ValueError, match='within'):
        with execution_budget(2):
            mapper = RadialSphereMap(vertices, faces, 'cpu')
            assert mapper.cpu_threads == 2
            mapper.weights(torch.tensor([[102., 0., 100.]], dtype=torch.float64))
    assert cpu_workers() == previous
    assert torch.get_num_threads() == torch_threads


def test_differentiable_query_keeps_tensor_radial_projection_and_gradient():
    vertices = np.array([[0., 0., 100.], [10., 0., 100.], [0., 10., 100.]])
    mapper = RadialSphereMap(vertices, np.array([[0, 1, 2]]), 'cpu')
    point = torch.tensor([[2., 3., 90.]], dtype=torch.float64, requires_grad=True)
    nodes = mapper._leaf_nodes(point)
    face, projection = mapper._native_selection(point, nodes)
    assert face.item() == 0
    torch.testing.assert_close(projection, point*(100/point[:, 2])[:, None], rtol=0, atol=0)
    projection.sum().backward()
    torch.testing.assert_close(point.grad, torch.tensor([[100/90, 100/90, -500/(90*90)]],
                                                       dtype=torch.float64), rtol=0, atol=2e-16)


def test_differentiable_cached_geometry_keeps_projection_graph():
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, 'cpu')
    mapper.vertices.requires_grad_()
    mapper.triangles = mapper.vertices[mapper.faces]
    mapper.normal, mapper.normal_dot_a, mapper.edge_normals = _projection_geometry(mapper.triangles)
    point = torch.as_tensor(vertices[faces[0]].mean(0)[None], dtype=torch.float64)
    _, projection = mapper._native_selection(point, mapper._leaf_nodes(point))
    projection.sum().backward()
    assert mapper.vertices.grad is not None
    assert torch.isfinite(mapper.vertices.grad).all()
    assert torch.count_nonzero(mapper.vertices.grad) > 0


def test_float32_queries_are_promoted_to_existing_float64_lookup_contract():
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, 'cpu')
    points = torch.as_tensor(vertices[faces[:11]].mean(1), dtype=torch.float32)
    expected = mapper.weights(points.double())
    actual = mapper.weights(points)
    assert actual[1].dtype == torch.float64
    for first, second in zip(actual, expected):
        assert torch.equal(first, second)


def test_mapper_keeps_one_geometry_snapshot_after_caller_mutates_input_arrays():
    vertices, faces = _ico(1)
    expected_vertices, expected_faces = vertices.copy(), faces.copy()
    mapper = RadialSphereMap(vertices, faces, 'cpu')
    points = torch.as_tensor(expected_vertices[expected_faces[:11]].mean(1))
    expected = mapper.weights(points)
    vertices[:] = 0
    faces[:] = 0
    np.testing.assert_array_equal(mapper.vertices.numpy(), expected_vertices)
    np.testing.assert_array_equal(mapper.faces.numpy(), expected_faces)
    for actual, reference in zip(mapper.weights(points), expected):
        assert torch.equal(actual, reference)
