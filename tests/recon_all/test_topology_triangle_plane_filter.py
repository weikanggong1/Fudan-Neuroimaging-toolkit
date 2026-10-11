import numpy as np
import pytest

from fnit.recon_all.topology_triangle_plane_filter import separated_triangle_pairs
from fnit.recon_all.mris_remove_intersection_python import _triangles_intersect, mark_intersections


def test_certified_rejection_preserves_tolerance_and_coplanar_rules():
    a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    cases = np.array([a+[0, 0, 1], a+[0, 0, 1e-8], a,
                      [[.2, .2, -1], [.2, .2, 1], [.8, .2, 0]],
                      a+[0, 0, 1e-145]])
    rejected = separated_triangle_pairs(triangles_a=np.repeat(a[None], len(cases), axis=0), triangles_b=cases, chunk_size=2)
    assert rejected.tolist() == [True, False, False, False, False]
    assert all(not _triangles_intersect(a, b) for b, flag in zip(cases, rejected) if flag)


def test_no_false_rejections_across_scale_and_cancellation():
    rng = np.random.default_rng(43)
    a = rng.normal(size=(3000, 3, 3))
    b = rng.normal(size=(3000, 3, 3))
    for scale, shift in ((1., 0.), (1e-5, 0.), (1e5, 1e12)):
        x, y = a*scale+shift, b*scale+shift
        mask = separated_triangle_pairs(triangles_a=x, triangles_b=y)
        assert all(not _triangles_intersect(u, v) for u, v in zip(x[mask], y[mask]))


def test_marking_retains_nonzero_intersections_and_shared_vertex_exclusion():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0], [.2,.2,-1],[.2,.2,1],[.8,.2,0],
                         [0,0,1],[1,0,1],[0,1,1]], dtype=np.float64)
    faces = np.array([[0,1,2],[3,4,5],[6,7,8],[0,1,5]])
    baseline = mark_intersections(vertices, faces)
    candidate = mark_intersections(vertices, faces, separation_backend="numba", chunk_size=1)
    assert baseline[1] > 0
    assert baseline[1] == candidate[1]
    np.testing.assert_array_equal(baseline[0], candidate[0])


def test_nonfinite_and_degenerate_fall_back():
    a = np.zeros((3, 3, 3))
    b = a.copy()
    b[0, 0, 0] = np.inf
    b[1, 0, 0] = np.nan
    assert not separated_triangle_pairs(triangles_a=a, triangles_b=b).any()


@pytest.mark.parametrize("backend, chunk", [("other", 4), ("numba", 0), ("numba", True)])
def test_bad_policy_is_rejected(backend, chunk):
    with pytest.raises(ValueError):
        separated_triangle_pairs(triangles_a=np.zeros((1,3,3)), triangles_b=np.zeros((1,3,3)), backend=backend, chunk_size=chunk)
