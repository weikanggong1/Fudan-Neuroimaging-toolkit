"""Ordered boundary selection and preservation of the mature Sulc path."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.msm import _fastpd_native
from fnit.msm._sphere_map import RadialSphereMap
from fnit.msm.msmsulc import _ico, _sphere_warp
from fnit.msm.msmall import _nearest_weights


def select(vertices, faces, query, candidates):
    arrays = [np.asarray(vertices, np.float64), np.asarray(faces, np.int64),
              np.asarray(query, np.float64), np.asarray(candidates, np.int64)]
    raw = _fastpd_native.source_radial_selection(*(array.tobytes() for array in arrays),
                                                len(vertices), len(faces), len(query), arrays[3].shape[1])
    return np.frombuffer(raw, np.float64).reshape(-1, 4)


def test_scalar_projection_selects_containing_plane_and_retains_candidate_ties():
    vertices = [[0, 0, 100], [10, 0, 100], [0, 10, 100], [10, 10, 100]]
    faces = [[0, 1, 2], [1, 3, 2]]
    observed = select(vertices, faces, [[5, 5, 100], [5, 5, 90]], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(observed[0], [0, 5, 5, 100])
    assert observed[1, 0] == 1
    np.testing.assert_array_equal(observed[1, 1:], np.array([5, 5, 90]) * (100 / 90))
    reverse_tie = select(vertices, faces, [[5, 5, 100]], [[1, 0]])
    assert reverse_tie[0, 0] == 1  # Strictly-less comparison, no tolerance-based merging.


def test_native_radial_buffers_and_indices_are_checked():
    with pytest.raises(ValueError, match="buffer dimensions"):
        _fastpd_native.source_radial_selection(b"", b"", b"", b"", 1, 1, 1, 1)
    with pytest.raises(ValueError, match="outside the mesh"):
        select([[0, 0, 100], [10, 0, 100], [0, 10, 100]], [[0, 1, 2]],
               [[1, 1, 100]], [[4]])


def test_scalar_warp_reprojects_and_normalizes_in_vertex_id_order():
    vertices = np.array([[0, 0, 100], [10, 0, 100], [0, 10, 100]], dtype=np.float64)
    faces = np.array([[2, 0, 1]], dtype=np.int64)
    target = vertices[:, [2, 0, 1]].copy()
    query = np.array([[2, 3, 90]], dtype=np.float64)
    raw = _fastpd_native.source_sphere_warp(vertices.tobytes(), faces.tobytes(),
            target.tobytes(), query.tobytes(), np.array([0], np.int64).tobytes(), 3, 1, 1)
    output = np.frombuffer(raw, np.float64).reshape(-1, 3)
    expected = query[:, [2, 0, 1]] / np.sqrt((query*query).sum(1))[:, None]*100
    np.testing.assert_allclose(output, expected, rtol=0, atol=3e-14)
    with pytest.raises(ValueError, match="buffer dimensions"):
        _fastpd_native.source_sphere_warp(b"", b"", b"", b"", b"", 3, 1, 1)


def test_source_precision_warp_is_device_and_execution_independent():
    vertices, faces = _ico(2)
    query = vertices[::13].copy()
    query[:, 0] += 1e-13
    destination = vertices[:, [2, 0, 1]].copy()
    outputs = []
    for device in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
        for execution in ("reference", "optimized"):
            output = _sphere_warp(torch.as_tensor(query, device=device), vertices, faces,
                                 torch.as_tensor(destination, device=device), device,
                                 execution, source_precision=True)
            outputs.append(output.cpu())
    for output in outputs[1:]:
        assert torch.equal(output, outputs[0])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_source_precision_boundary_selection_is_execution_independent(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    vertices, faces = _ico(2)
    query = vertices[::13].copy()
    query[:, 0] += 1e-13
    points = torch.as_tensor(query, device=device)
    outputs = [RadialSphereMap(vertices, faces, device, execution=execution,
                               source_precision=True).weights(points, batch_size=11)
               for execution in ("reference", "optimized")]
    for first, second in zip(*outputs):
        assert torch.equal(first, second)
    np.testing.assert_allclose(outputs[0][1].cpu().sum(1), 1, rtol=0, atol=3e-16)


def test_mature_sulc_lookup_does_not_enable_new_scalar_selection(monkeypatch):
    def forbidden(*args):
        raise AssertionError("MSMSulc reference-compatible default must remain unchanged")
    monkeypatch.setattr(_fastpd_native, "source_radial_selection", forbidden)
    vertices, faces = _ico(1)
    RadialSphereMap(vertices, faces, "cpu").weights(torch.as_tensor(vertices))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_spatial_weights_use_official_containing_triangle_not_global_nearest(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = json.loads((Path(__file__).parent / "data/msm_multivariate_nearest.json").read_text())
    vertices, faces = np.array(data["vertices_xyz"]), np.array(data["faces_ijk"])
    queries = np.array([entry["query"] for entry in data["cases"]])
    # Expected IDs were measured with the pinned official Octree on this
    # public, healthy nonuniform mesh, independently of the FNIT lookup.
    values = np.arange(len(vertices))[:, None]
    actual = _nearest_weights(vertices, faces, values, queries, device, "optimized")[:, 0]
    np.testing.assert_array_equal(actual, [entry["official_nearest_id"] for entry in data["cases"]])
    from scipy.spatial import cKDTree
    global_ids = cKDTree(vertices).query(queries)[1]
    assert np.all(actual != global_ids)


def test_triangle_nearest_ties_keep_original_corner_order():
    vertices = np.array([[0, 0, 100], [10, 0, 100], [0, 10, 100]], dtype=np.float64)
    faces = np.array([[1, 0, 2]], np.int64)
    query = np.array([[5, 0, 100]], np.float64)
    raw = _fastpd_native.source_triangle_nearest(vertices.tobytes(), faces.tobytes(),
            query.tobytes(), np.array([0], np.int64).tobytes(), 3, 1, 1)
    assert np.frombuffer(raw, np.int64)[0] == 1
    with pytest.raises(ValueError, match="buffer dimensions"):
        _fastpd_native.source_triangle_nearest(b"", b"", b"", b"", 3, 1, 1)
