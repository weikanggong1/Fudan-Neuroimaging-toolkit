"""Execution gates; these geometry fixtures are not scientific benchmarks."""
import numpy as np
import pytest
import torch

from fnit.msm._sphere_map import RadialSphereMap
from fnit.msm.msmsulc import _ico


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('project', [False, True])
def test_cached_radial_lookup_is_bitwise_equal_to_reference(device, project):
    vertices, faces = _ico(2)
    rng = np.random.default_rng(313)
    points = rng.normal(size=(2001, 3)); points *= 100/np.linalg.norm(points, axis=1, keepdims=True)
    # Vertex/edge queries exercise the deterministic face-order tie break.
    points = np.r_[points, vertices[:12], vertices[faces[:12]].mean(axis=1)]
    tensor = torch.as_tensor(points, device=device)
    reference = RadialSphereMap(vertices, faces, device, execution='reference')
    optimized = RadialSphereMap(vertices, faces, device, execution='optimized')
    expected = reference.weights(tensor, batch_size=127, project=project)
    actual = optimized.weights(tensor, batch_size=911, project=project)
    for first, second in zip(expected, actual):
        assert torch.equal(first, second)
    metric = torch.as_tensor(rng.normal(size=len(vertices)), dtype=torch.float64, device=device)
    assert torch.equal(reference.sample(tensor, metric), optimized.sample(tensor, metric))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_cached_radial_lookup_keeps_fallback_points_order(device):
    vertices, faces = _ico(1)
    # A deliberately irregular but unfolded sphere forces some queries outside
    # the closest vertex's incident triangles.
    original = vertices.copy()
    vertices = vertices.copy(); vertices[0] = vertices[0]*.7 + vertices[1]*.3
    vertices *= 100/np.linalg.norm(vertices, axis=1, keepdims=True)
    def radial_sign(coordinates):
        triangles = coordinates[faces]
        return np.sum(np.cross(triangles[:, 1]-triangles[:, 0],
                               triangles[:, 2]-triangles[:, 0])*triangles.mean(1), axis=1)
    assert np.all(radial_sign(original)*radial_sign(vertices) > 0)
    rng = np.random.default_rng(611)
    points = rng.normal(size=(997, 3)); points *= 100/np.linalg.norm(points, axis=1, keepdims=True)
    tensor = torch.as_tensor(points, device=device)
    reference = RadialSphereMap(vertices, faces, device, execution='reference')
    optimized = RadialSphereMap(vertices, faces, device, execution='optimized')
    original = optimized._fallback; missing_points = []
    def capture(points, query, face, projection, missing):
        missing_points.extend(query[missing].tolist())
        return original(points, query, face, projection, missing)
    optimized._fallback = capture
    expected = reference.weights(tensor, batch_size=100)
    actual = optimized.weights(tensor, batch_size=250)
    assert missing_points
    for first, second in zip(expected, actual): assert torch.equal(first, second)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_shared_vertices_keep_source_octree_distance_selected_faces(device):
    # Offline compiled Point/Triangle/Node/Octree oracle from pinned newMSM
    # 260718953547743c028a45f8c885d163441df87a selects the closest finite
    # edge/vertex distance, rather than the smallest containing face number.
    # These generated mesh fixtures test arithmetic; they are not a benchmark.
    vertices, faces = _ico(2)
    points = torch.as_tensor(vertices[[1, 3, 5]], device=device)
    for execution in ['reference', 'optimized']:
        ids, weights, patches = RadialSphereMap(
            vertices, faces, device, execution=execution).weights(points)
        assert patches.cpu().tolist() == [123, 91, 63]
        assert torch.equal(ids, torch.as_tensor(faces[[123, 91, 63]], device=device))
        assert torch.equal(weights, torch.tensor([[0., 1., 0.]]*3, device=device))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_scalar_interpolation_uses_three_term_triangle_corner_order(device):
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, device)
    ids = torch.tensor([[2, 0, 1]], device=device)
    weights = torch.full((1, 3), 1/3, dtype=torch.float64, device=device)
    mapper.weights = lambda points, **kwargs: (ids, weights, ids[:, 0])
    metric = torch.tensor([-3e16, 3., 3e16], dtype=torch.float64, device=device)
    # Face-order terms are (1e16, -1e16, 1). Sorting vertex IDs or changing
    # the sum tree produces zero, whereas the source's (first+second)+third
    # produces one. This checks arithmetic rather than a full geometry run.
    result = mapper.sample(torch.zeros((1, 3), device=device), metric)
    assert result.item() == 1


def test_radial_lookup_handles_empty_queries():
    vertices, faces = _ico(1)
    for execution in ['optimized', 'reference']:
        result = RadialSphereMap(vertices, faces, 'cpu', execution=execution).weights(torch.empty((0, 3)))
        assert [tuple(value.shape) for value in result] == [(0, 3), (0, 3), (0,)]
