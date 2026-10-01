import numpy as np
import pytest

from fnit.msm import _fastpd_native


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


def test_hocr_aggregates_duplicate_cubic_terms_before_fusion():
    # Fixed integer fixture checked against the pinned official ELC->FastPD
    # pipeline. Separate auxiliary nodes for duplicated cliques select 0100.
    faces = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]] * 2, np.int32)
    costs = np.array([
        [1, 1, -1, 1, -3, -1, -1, -1], [3, 1, 0, -2, 0, -2, -1, 1],
        [-3, 2, 0, 0, 1, 1, 2, -3], [-1, -2, -2, -2, 2, 2, 1, -3],
        [-1, 3, -3, -3, -1, -3, -3, -1], [0, -2, -2, -1, -2, -1, 3, 1],
        [1, -3, 3, 1, 0, 0, 0, 1], [-3, 0, -2, 1, 0, -3, 0, 1],
    ], np.float64)
    choice = _fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 4)
    assert list(choice) == [0, 1, 0, 1]


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_fastpd_rejects_nonfinite_costs(invalid):
    faces = np.array([[0, 1, 2]], np.int32)
    costs = np.zeros((1, 8), np.float64)
    costs[0, 3] = invalid
    with pytest.raises(ValueError, match='finite'):
        _fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 3)


@pytest.mark.parametrize('faces', [[[0, 1, 3]], [[-1, 0, 1]], [[0, 1, 1]]])
def test_fastpd_rejects_out_of_range_or_duplicate_vertex_ids(faces):
    costs = np.zeros((1, 8), np.float64)
    with pytest.raises(ValueError, match='sorted, distinct and in range'):
        _fastpd_native.optimize(np.array(faces, np.int32).tobytes(), costs.tobytes(), 3)


def test_fastpd_accepts_unaligned_readonly_buffers():
    faces = np.array([[0, 1, 2]], np.int32)
    costs = np.zeros((1, 8), np.float64)
    costs[0, 7] = -1
    face_view = memoryview(b'x' + faces.tobytes())[1:]
    cost_view = memoryview(b'x' + costs.tobytes())[1:]
    assert list(_fastpd_native.optimize(face_view, cost_view, 3)) == [1, 1, 1]


def test_fastpd_preserves_isolated_original_vertices():
    faces = np.array([[0, 1, 2]], np.int32)
    costs = np.zeros((1, 8), np.float64)
    costs[0, 7] = -1
    assert list(_fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 5)) == [1, 1, 1, 0, 0]


@pytest.mark.parametrize('target', range(8))
def test_hocr_keeps_energy_table_bit_order(target):
    # Independent unary terms have a unique minimizer. The first vertex is
    # table bit 4 and the last is bit 1, as in newMSM's triplet likelihood.
    faces = np.array([[0, 1, 2]], np.int32)
    desired = [(target >> shift) & 1 for shift in (2, 1, 0)]
    costs = np.array([[sum(((state >> shift) & 1) != desired[corner]
                           for corner, shift in enumerate((2, 1, 0)))
                       for state in range(8)]], np.float64)
    assert list(_fastpd_native.optimize(faces.tobytes(), costs.tobytes(), 3)) == desired
