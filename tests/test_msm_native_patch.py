"""Small geometric contracts; fixtures are not MRI accuracy benchmarks."""

import numpy as np
import pytest
import torch

from fnit.msm import _native_patch
from fnit.msm._native_repair import repair_native_sphere
from fnit.msm.msmsulc import _ico


def signed(coordinates, faces):
    triangles = np.asarray(coordinates, dtype=np.float64)[faces]
    return np.sum(np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0]) * triangles[:, 0], axis=1)


def folded_ico():
    reference, faces = _ico(1)
    points = reference.copy()
    first, second, third = faces[0]
    points[first] = .55 * points[second] + .55 * points[third] - .1 * points[first]
    points[first] *= 100 / np.linalg.norm(points[first])
    return points, faces, reference


@pytest.mark.parametrize("chunk_size", [1, 256])
def test_quantized_joint_repairs_float32_degenerate_face_and_preserves_good_faces(chunk_size):
    reference, faces = _ico(2)
    points = reference.copy()
    points[43] = (reference[14] + reference[44]) / 2 + 1e-10 * reference[43]
    points[43] *= 100 / np.linalg.norm(points[43])
    saved = points.astype(np.float32).astype(np.float64)
    before = signed(saved, faces)
    assert np.all(signed(points, faces) > 0)
    assert np.count_nonzero(before <= 0) == 1
    copies = points.copy(), reference.copy(), faces.copy()
    output, report = _native_patch.quantized_joint_repair(
        points, faces, reference, chunk_size=chunk_size)
    actual = output.astype(np.float32).astype(np.float64)
    after = signed(actual, faces)
    assert report["success"] and report["absolute_folded_faces_after"] == 0
    assert np.all(after > 0) and np.all(after[before > 0] > 0)
    assert report["newly_folded_previously_positive_faces"] == 0
    assert 1 <= report["moved_vertices"] <= 3 and report["evaluated_candidates"] > 0
    np.testing.assert_array_equal(output, actual)
    for current, original in zip((points, reference, faces), copies):
        np.testing.assert_array_equal(current, original)
    again, repeated = _native_patch.quantized_joint_repair(output, faces, reference)
    np.testing.assert_array_equal(again, output)
    assert repeated["success"] and repeated["accepted_joint_updates"] == 0


def test_harmonic_repair_certifies_saved_coordinates_and_preserves_inputs():
    points, faces, reference = folded_ico()
    saved = points.astype(np.float32).astype(np.float64)
    initial_good = signed(saved, faces) > 0
    copies = points.copy(), reference.copy(), faces.copy()
    output, report = _native_patch.harmonic_repair(
        points, faces, reference, maximum_rings=4, allow_nonconvex=True)
    actual = output.astype(np.float32).astype(np.float64)
    assert report["success"] and report["accepted_joint_updates"] > 0
    assert np.all(signed(actual, faces) > 0)
    assert np.all(signed(actual, faces)[initial_good] > 0)
    assert report["newly_folded_previously_positive_faces"] == 0
    np.testing.assert_array_equal(output, actual)
    for current, original in zip((points, reference, faces), copies):
        np.testing.assert_array_equal(current, original)


def test_nonconvex_harmonic_proposal_requires_explicit_option_and_actual_float32_qc():
    # A simple concave disk with one interior vertex: its averaged Dirichlet
    # solution is in the polygon kernel, but convex embedding guarantees do
    # not apply. The candidate is checked using its actual float32 vertices.
    planar = np.array([[-2., -2.], [2., -2.], [.5, 0.], [2., 2.],
                       [-2., 2.], [1.5, 0.]])
    points = np.column_stack((planar, np.full(len(planar), 100.)))
    points *= 100 / np.linalg.norm(points, axis=1, keepdims=True)
    faces = np.array([[5, vertex, (vertex + 1) % 5] for vertex in range(5)])
    arguments = (points, faces, np.arange(5), np.arange(6), np.arange(5), np.array([5]), 1)
    strict, strict_report = _native_patch.harmonic_target(*arguments)
    assert strict is None and strict_report["boundary_simple"]
    assert not strict_report["boundary_convex"]
    candidate, report = _native_patch.harmonic_target(*arguments, allow_nonconvex=True)
    assert candidate is not None and not report["convex_embedding_theorem_applies"]
    result = points.astype(np.float32)
    fixed_boundary = result[:5].copy()
    result[5] = candidate[0].astype(np.float32)
    assert np.all(signed(result, faces) > 0)
    np.testing.assert_array_equal(result[:5], fixed_boundary)


def test_native_harmonic_restart_uses_saved_solver_coordinates_not_reference(monkeypatch):
    points, faces, reference = folded_ico()
    input_saved = points.astype(np.float32).astype(np.float64)
    direct = input_saved.copy()
    vertex = faces[0, 0]
    direct[vertex] = .99 * direct[vertex] + .01 * reference[vertex]
    direct[vertex] *= 100 / np.linalg.norm(direct[vertex])
    direct = direct.astype(np.float32).astype(np.float64)
    baseline = signed(reference, faces)
    assert np.count_nonzero(signed(direct, faces) <= 0) == 1
    assert np.min(signed(direct, faces) / baseline) > np.min(signed(input_saved, faces) / baseline)
    calls = []

    def guard(coordinates, triangles, target, **kwargs):
        np.testing.assert_array_equal(coordinates, input_saved)
        return direct.copy(), {"updates": 1, "sweeps": 1, "success": False}

    def quantized(coordinates, triangles, original, **kwargs):
        np.testing.assert_array_equal(coordinates, direct)
        return coordinates.copy(), {"accepted_joint_updates": 0, "success": False}

    def harmonic(coordinates, triangles, original, **kwargs):
        calls.append(coordinates.copy())
        np.testing.assert_array_equal(coordinates, input_saved)
        np.testing.assert_array_equal(original, reference)
        assert not np.array_equal(coordinates, original)
        assert kwargs["allow_nonconvex"] and kwargs["maximum_rings"] == 16
        return original.astype(np.float32).astype(np.float64), {"success": True, "accepted_joint_updates": 1}

    def no_coarse(*args, **kwargs):
        raise AssertionError("a certified joint correction must not call source unfolding")

    monkeypatch.setattr("fnit.msm._native_repair._guarded_repair", guard)
    monkeypatch.setattr(_native_patch, "quantized_joint_repair", quantized)
    monkeypatch.setattr(_native_patch, "harmonic_repair", harmonic)
    monkeypatch.setattr("fnit.msm.msmsulc._unfold", no_coarse)
    vertices = torch.as_tensor(points.copy())
    repaired, report = repair_native_sphere(vertices, faces, reference)
    assert len(calls) == 1 and report["success"] and report["unfold_updates"] == 0
    assert [row["stage"] for row in report["attempts"]] == ["direct_guard", "direct_quantized", "raw_harmonic_restart"]
    assert np.all(signed(repaired.numpy().astype(np.float32), faces) > 0)
    np.testing.assert_array_equal(vertices.numpy(), points)
