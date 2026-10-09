"""Ordered Torch normals retain source geometry rules without atomics."""
import numpy as np
import pytest
import torch

from fnit.recon_all.place_surface_normals import (
    FaceNormalTopology, TorchFaceNormalTopology, initial_vertex_normals,
)

MAX_COMPONENT_ERROR = 5.0e-7


def _mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [2, 2, 2]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ordered_normals_match_source_and_do_not_mutate(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    vertices, faces = _mesh()
    original = vertices.copy()
    topology = TorchFaceNormalTopology(triangles=faces, nvertices=len(vertices), device=device)
    observed = topology.evaluate(vertices=vertices)
    expected = initial_vertex_normals(vertices=vertices, triangles=faces)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=MAX_COMPONENT_ERROR)
    np.testing.assert_array_equal(observed[-1], np.zeros(3, np.float32))
    np.testing.assert_array_equal(vertices, original)
    assert observed.dtype == np.float32


def test_zero_and_tiny_triangles_are_not_epsilon_clamped():
    vertices = np.array([[0, 0, 0], [1e-15, 0, 0], [0, 1e-15, 0],
                         [0, 0, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 0, 3]], dtype=np.int32)
    topology = TorchFaceNormalTopology(triangles=faces, nvertices=len(vertices), device="cpu")
    observed = topology.evaluate(vertices=vertices)
    expected = FaceNormalTopology(triangles=faces, nvertices=len(vertices)).evaluate(vertices)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed[:3, 2], np.ones(3, np.float32))
    assert np.isfinite(observed).all()


def test_cached_faces_are_frozen_and_coordinate_changes_recompute():
    vertices, faces = _mesh()
    topology = TorchFaceNormalTopology(triangles=faces, nvertices=len(vertices), device="cpu")
    before = topology.evaluate(vertices=vertices)
    changed = vertices.copy()
    changed[3] = [0.3, 0.2, 1.1]
    after = initial_vertex_normals(vertices=changed, triangles=faces, topology=topology)
    assert not np.array_equal(before, after)
    np.testing.assert_allclose(after, initial_vertex_normals(changed, faces), rtol=0,
                               atol=MAX_COMPONENT_ERROR)
    with pytest.raises(ValueError):
        topology.validate(triangles=faces[::-1], nvertices=len(vertices))


def test_tensor_device_dtype_shape_validation_and_empty_mesh():
    vertices, faces = _mesh()
    topology = TorchFaceNormalTopology(triangles=faces, nvertices=len(vertices), device="cpu")
    with pytest.raises(ValueError):
        topology.evaluate_tensor(torch.from_numpy(vertices).double())
    with pytest.raises(TypeError):
        topology.evaluate_tensor(vertices)
    empty = TorchFaceNormalTopology(triangles=np.empty((0, 3), np.int32), nvertices=0, device="cpu")
    assert empty.evaluate_tensor(torch.empty((0, 3), dtype=torch.float32)).shape == (0, 3)
