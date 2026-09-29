import numpy as np
import pytest

from fnit.fmri import _fastpd_native


def test_hocr_fastpd_selects_joint_triangle_move():
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    costs = np.zeros((1, 8), dtype=np.float64)
    costs[0, 7] = -1.0
    choice = _fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 3)
    assert list(choice) == [1, 1, 1]


def test_hocr_fastpd_requires_sorted_faces():
    faces = np.array([[2, 0, 1]], dtype=np.int32)
    costs = np.zeros((1, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="sorted"):
        _fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 3)
