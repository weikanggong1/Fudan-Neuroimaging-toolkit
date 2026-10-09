"""Scalar unfolding and device-clear gate regressions; no benchmark fixtures."""

import hashlib

import numpy as np
import pytest
import torch

from fnit.msm import _fastpd_native
from fnit.msm.msmsulc import _ico, _unfold


def test_scalar_unfold_matches_fixed_original_on_small_mathematical_mesh():
    # A 12-vertex mathematical icosahedron, independently run through pinned
    # newMSM CPU1. This checks multiplication/normalization order across many
    # updates; the stress fixture is not evidence of scientific accuracy.
    points, faces = _ico(0)
    points = points.copy()
    first, second, third = faces[0]
    points[first] = .55 * points[second] + .55 * points[third] - .1 * points[first]
    points[first] *= 100 / np.linalg.norm(points[first])
    before = points.copy()
    raw, updates = _fastpd_native.source_unfold(
        points.tobytes(), faces.astype(np.int64).tobytes(), len(points), len(faces), 1000
    )
    assert updates == 11684
    assert hashlib.sha256(raw).hexdigest() == "0e72c79791891b9c9d5a3d00165a83ac1f0549959de85ed17c67fea56bd2faad"
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clear_mesh_stays_on_device_without_native_copy(monkeypatch, dtype):
    points, faces = _ico(2)
    tensor = torch.tensor(points, dtype=dtype)
    def forbidden(*args):
        raise AssertionError("a clearly smooth sphere must bypass scalar unfolding")
    monkeypatch.setattr(_fastpd_native, "source_unfold", forbidden)
    result, updates = _unfold(tensor, faces)
    assert result is tensor and updates == 0


@pytest.mark.parametrize("cosine", [.5, .5 + 5e-13])
def test_near_intersection_threshold_uses_scalar_gate(monkeypatch, cosine):
    # Two incident faces meet at the literal cosine=0.5 source threshold.
    # A tiny positive rounding difference also needs the conservative gate.
    points = torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
                           [-cosine, 0., 1. + np.sqrt(1. - cosine*cosine)]], dtype=torch.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    calls = []
    def scalar(v, f, nv, nf, sweeps):
        calls.append((nv, nf, sweeps))
        return v, 0
    monkeypatch.setattr(_fastpd_native, "source_unfold", scalar)
    result, updates = _unfold(points, faces, maximum_sweeps=7)
    assert calls == [(4, 2, 7)] and result is points and updates == 0


def test_degenerate_face_cannot_pass_device_clear_gate(monkeypatch):
    points = torch.tensor([[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]], dtype=torch.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    calls = []
    def scalar(v, f, nv, nf, sweeps):
        calls.append(True)
        return v, 0
    monkeypatch.setattr(_fastpd_native, "source_unfold", scalar)
    _unfold(points, faces)
    assert calls == [True]


@pytest.mark.parametrize("geometry", ["cross_cancellation", "normalize_cutoff"])
def test_ambiguous_normal_arithmetic_uses_scalar_gate(monkeypatch, geometry):
    if geometry == "cross_cancellation":
        # Each cross subtracts products near 1e16 to get a normal near 10.
        # Device fused arithmetic may normalize a different normal, even
        # when its computed cosine is comfortably above 0.5.
        points = [[0., 0., 0.], [1e8, 1e8, 1e8],
                  [1e8, 1e8 + 1e-7, 1e8], [1e8, 1e8, 1e8 + 1e-7]]
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    else:
        # A one-ulp normal just above the scalar normalization cutoff.
        points = [[0., 0., 0.], [1., 0., 0.], [0., np.nextafter(1e-8, np.inf), 0.]]
        faces = np.array([[0, 1, 2]], dtype=np.int64)
    tensor = torch.tensor(points, dtype=torch.float64)
    calls = []
    def scalar(v, f, nv, nf, sweeps):
        calls.append(True)
        return v, 0
    monkeypatch.setattr(_fastpd_native, "source_unfold", scalar)
    _unfold(tensor, faces)
    assert calls == [True]


def test_scalar_zero_budget_keeps_coordinates_exact():
    points, faces = _ico(0)
    points[0] = points[1]
    raw, updates = _fastpd_native.source_unfold(
        points.tobytes(), faces.astype(np.int64).tobytes(), len(points), len(faces), 0
    )
    assert raw == points.tobytes() and updates == 0


@pytest.mark.parametrize("bad", ["nonfinite", "face_index", "shape", "negative_budget"])
def test_scalar_rejects_invalid_buffers_without_unsafe_access(bad):
    points, faces = _ico(0)
    points, faces = points.copy(), faces.astype(np.int64).copy()
    limit = 1000
    if bad == "nonfinite": points[0, 0] = np.nan
    if bad == "face_index": faces[0, 0] = len(points)
    if bad == "negative_budget": limit = -1
    raw = points.tobytes()[:-1] if bad == "shape" else points.tobytes()
    with pytest.raises(ValueError):
        _fastpd_native.source_unfold(raw, faces.tobytes(), len(points), len(faces), limit)
