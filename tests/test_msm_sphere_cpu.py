"""CPU fusion arithmetic gates; these fixtures are not MRI benchmarks."""

import numpy as np
import pytest
import torch

from fnit.msm._sphere_cpu import select_faces
from fnit.msm._sphere_map import RadialSphereMap
from fnit.msm.msmsulc import _ico


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("source_precision", [False, True])
def test_fused_selection_retains_planes_ties_and_expanded_candidates(threads, source_precision):
    vertices, faces = _ico(2)
    # Nonuniform, unfolded geometry exercises the expanded-nearest fallback.
    vertices = vertices.copy()
    vertices[0] = vertices[0] * .7 + vertices[1] * .3
    vertices *= 100 / np.linalg.norm(vertices, axis=1, keepdims=True)
    mapper = RadialSphereMap(vertices, faces, "cpu", source_precision=source_precision)
    reference = RadialSphereMap(vertices, faces, "cpu", execution="reference",
                                source_precision=source_precision)
    rng = np.random.default_rng(924)
    points = rng.normal(size=(711, 3))
    points *= 100 / np.linalg.norm(points, axis=1, keepdims=True)
    points = np.r_[points, vertices, vertices[faces[:31]].mean(axis=1)]
    # Near-shared-vertex perturbations cover the scalar source fallback mask.
    points[-17:, 0] += 1e-13
    for neighbors in (1, 32):
        nearest = mapper.tree.query(points, k=neighbors)[1].reshape(len(points), neighbors)
        expected = reference._select(torch.as_tensor(points), torch.as_tensor(nearest))
        actual = select_faces(points, nearest, mapper.incident.numpy(),
                              mapper.triangles.numpy(), mapper.normal.numpy(),
                              mapper.normal_dot_a.numpy(),
                              *(edge.numpy() for edge in mapper.edge_normals), cpu_threads=threads)
        for index, first in enumerate(expected):
            if first is not None:
                np.testing.assert_array_equal(first.numpy(), actual[index])


def test_cpu_selection_restores_numba_thread_mask_on_failure(monkeypatch):
    import numba
    from fnit.msm import _sphere_cpu
    previous = numba.get_num_threads()
    if previous < 2:
        pytest.skip("Numba runtime limited to one worker")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    def fail(*arguments):
        assert numba.get_num_threads() == min(previous, 2)
        raise RuntimeError("controlled worker failure")
    monkeypatch.setattr(_sphere_cpu, "_parallel", fail)
    with pytest.raises(RuntimeError, match="controlled worker failure"):
        select_faces(cpu_threads=2)
    assert numba.get_num_threads() == previous


def test_differentiable_cpu_queries_keep_tensor_projection(monkeypatch):
    from fnit.msm import _sphere_cpu
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, "cpu")
    point = torch.as_tensor(vertices[faces[0]].mean(0)[None], dtype=torch.float64).requires_grad_()
    nearest = torch.as_tensor(mapper.tree.query(point.detach().numpy())[1][:, None])
    def forbidden(*arguments, **keywords):
        raise AssertionError("differentiable coordinates must retain tensor projection")
    monkeypatch.setattr(_sphere_cpu, "select_faces", forbidden)
    _, projected, _, _ = mapper._select(point, nearest)
    projected.sum().backward()
    assert point.grad is not None
    assert torch.isfinite(point.grad).all()


def test_differentiable_cached_geometry_keeps_tensor_projection(monkeypatch):
    from fnit.msm import _sphere_cpu
    from fnit.msm._sphere_map import _projection_geometry
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, "cpu")
    # Refresh the cache from trainable source coordinates, with fixed queries.
    mapper.vertices.requires_grad_()
    mapper.triangles = mapper.vertices[mapper.faces]
    mapper.normal, mapper.normal_dot_a, mapper.edge_normals = _projection_geometry(mapper.triangles)
    point = torch.as_tensor(vertices[faces[0]].mean(0)[None], dtype=torch.float64)
    nearest = torch.as_tensor(mapper.tree.query(point.numpy())[1][:, None])
    def forbidden(*arguments, **keywords):
        raise AssertionError("differentiable cached geometry must retain tensor projection")
    monkeypatch.setattr(_sphere_cpu, "select_faces", forbidden)
    _, projected, _, _ = mapper._select(point, nearest)
    projected.sum().backward()
    assert mapper.vertices.grad is not None
    assert torch.isfinite(mapper.vertices.grad).all()
    assert torch.count_nonzero(mapper.vertices.grad) > 0


def test_float32_cpu_queries_keep_original_tensor_selection(monkeypatch):
    from fnit.msm import _sphere_cpu
    vertices, faces = _ico(1)
    mapper = RadialSphereMap(vertices, faces, "cpu")
    reference = RadialSphereMap(vertices, faces, "cpu", execution="reference")
    points = torch.as_tensor(vertices[faces[:11]].mean(1), dtype=torch.float32)
    nearest = torch.as_tensor(mapper.tree.query(points.numpy())[1][:, None])
    expected = reference._select(points, nearest)
    def forbidden(*arguments, **keywords):
        raise AssertionError("float32 queries must retain tensor selection")
    monkeypatch.setattr(_sphere_cpu, "select_faces", forbidden)
    actual = mapper._select(points, nearest)
    for first, second in zip(actual, expected):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, rtol=0, atol=0)
