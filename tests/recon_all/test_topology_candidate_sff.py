"""Contract/analytic checks; real candidate benchmark is recorded separately."""
import numpy as np
import pytest

from fnit.recon_all.topology_candidate_sff import (
    _eigen2, candidate_second_fundamental_form,
)


def data():
    xy = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                   [1., 1.], [-1., -1.]], np.float32)
    z = np.float32(.25) * xy[:, 0] ** 2 + np.float32(.125) * xy[:, 1] ** 2
    xyz = np.column_stack((xy, z))
    return dict(original_vertices=xyz, vertex_normals=np.tile([0., 0., 1.], (7, 1)),
                neighbor_offsets=np.array([0, 6, 6, 6, 6, 6, 6, 6], np.int32),
                neighbors=np.arange(1, 7, dtype=np.int32), patch_vertices=np.array([0], np.int32),
                initial_k1=np.full(7, 7., np.float32), initial_k2=np.full(7, -3., np.float32),
                initial_tangent_e1=np.tile([1., 0., 0.], (7, 1)),
                initial_tangent_e2=np.tile([0., 1., 0.], (7, 1)))


def test_quadratic_surface_and_unselected_fields():
    inputs = data(); before = {name: value.copy() for name, value in inputs.items()}
    result = candidate_second_fundamental_form(**inputs)
    assert np.allclose([result['k1'][0], result['k2'][0]], [.5, .25], atol=2e-6, rtol=0)
    for name, value in before.items():
        assert np.array_equal(inputs[name], value)
    for output, source in [('k1', 'initial_k1'), ('k2', 'initial_k2'),
                           ('tangent_e1', 'initial_tangent_e1'), ('tangent_e2', 'initial_tangent_e2')]:
        assert np.array_equal(result[output][1:], inputs[source][1:])
    assert result['fit_kind'].tolist() == [1]


def test_planar_fit_is_finite():
    inputs = data(); inputs['original_vertices'][:, 2] = 0.
    result = candidate_second_fundamental_form(**inputs)
    assert result['k1'][0] == result['k2'][0] == 0.
    assert np.isfinite(result['tangent_e1']).all()


def test_zero_matrix_keeps_source_zero_principal_fields():
    inputs = data(); inputs['original_vertices'].fill(0.)
    result = candidate_second_fundamental_form(**inputs)
    assert result['fit_kind'].tolist() == [3]
    assert result['k1'][0] == result['k2'][0] == 0.
    assert result['condition_number'][0] == np.float32(1e10)


def test_no_neighbors_preserves_curvature_but_builds_tangent():
    inputs = data(); inputs['patch_vertices'] = np.array([1], np.int32)
    result = candidate_second_fundamental_form(**inputs)
    assert result['fit_kind'].tolist() == [0]
    assert result['k1'][1] == 7.
    assert np.linalg.norm(result['tangent_e1'][1]) == 1.


def test_ill_conditioned_source_extreme_curvature_has_no_extra_factor_two():
    inputs = data()
    x = np.array([0., 1., -1., 2., -2., 3., -3.], np.float32)
    inputs['original_vertices'][:, 0] = x
    inputs['original_vertices'][:, 1] = 0.
    inputs['original_vertices'][:, 2] = np.float32(.25) * x * x
    result = candidate_second_fundamental_form(**inputs)
    assert result['fit_kind'].tolist() == [2]
    assert result['condition_number'][0] >= 500000.
    assert result['k1'][0] == result['k2'][0] == np.float32(.25)
    assert np.array_equal(result['tangent_e1'][0], [0., 1., 0.])


@pytest.mark.parametrize('matrix', [np.array([[3., 0.], [0., -4.]], np.float32),
    np.array([[2., 1.], [1., 2.]], np.float32), np.zeros((2, 2), np.float32),
    np.array([[1e-3, .07], [.07, -.2]], np.float32)])
def test_source_eigenvectors_and_absolute_order(matrix):
    values, vectors = _eigen2(matrix)
    assert abs(values[0]) >= abs(values[1])
    assert np.allclose(matrix.astype(float) @ vectors, vectors * values, atol=2e-7, rtol=1e-6)
    assert np.allclose(vectors.T @ vectors, np.eye(2), atol=2e-7, rtol=0)


@pytest.mark.parametrize('field,value', [('neighbors', np.array([-2147483649], np.int64)),
    ('neighbor_offsets', np.zeros(8, np.float32)), ('patch_vertices', np.array([0, 0], np.int32)),
    ('patch_vertices', np.array([7], np.int32)), ('original_vertices', np.array(1.)),
    ('vertex_normals', np.full((7, 3), np.nan))])
def test_malformed_candidate_fails(field, value):
    inputs = data(); inputs[field] = value
    with pytest.raises(ValueError):
        candidate_second_fundamental_form(**inputs)
