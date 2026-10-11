import numpy as np
import pytest

from fnit.recon_all.topology_fitness_search import (
    compose_patch_fitness, rank_patch_fitness,
)


def test_source_rank_keeps_earlier_candidate_first_inside_float_epsilon_tie():
    scores = np.asarray([-1.0, -1.0 + 5e-8, -2.0], np.float64)
    assert rank_patch_fitness(scores).tolist() == [0, 1, 2]


def test_old_one_e_minus_five_tie_changes_source_candidate_selection():
    scores = np.asarray([-1.0, -1.0 + 5e-6, -2.0], np.float64)
    assert rank_patch_fitness(scores).tolist() == [1, 0, 2]


def test_tie_is_strictly_less_than_float_epsilon():
    epsilon = float(np.finfo(np.float32).eps)
    assert rank_patch_fitness([0.0, epsilon]).tolist() == [1, 0]


def test_invalid_patch_receives_source_face_penalty():
    valid = compose_patch_fitness(-1.0, -2.0, -3.0, -4.0, -5.0, True)
    invalid = compose_patch_fitness(-1.0, -2.0, -3.0, -4.0, -5.0, False)
    assert valid == -60.0
    assert invalid - valid == -10_000_000.0


def test_components_write_float32_before_double_mri_sum():
    # Individually representable fields whose float32 sum loses the low bit.
    result = compose_patch_fitness(-16777216., -1., 0., 0., 0., True)
    assert result == -16777217.
    # A pre-write double excess must be removed by the source TP store.
    assert compose_patch_fitness(-16777216.1, -1., 0., 0., 0., True) == result


def test_numpy_float32_fields_do_not_round_the_mri_sum():
    face, vertex = np.float32(-7.894798278808594), np.float32(-7.215606212615967)
    result = compose_patch_fitness(face, vertex, np.float32(0), np.float32(0), np.float32(0), True)
    assert result == -15.11040449142456
    assert result != float(face+vertex)


def test_all_component_fields_are_float32_even_for_double_inputs():
    result = compose_patch_fitness(.100000000001, .200000000001, .300000000001,
                                   .400000000001, .500000000001, True,
                                   return_diagnostics=True)
    assert result['tp_fields'] == {
        'face_ll': .10000000149011612, 'vertex_ll': .20000000298023224,
        'curv_ll': .30000001192092896, 'qcurv_ll': .4000000059604645, 'unmri_ll': .5}
    assert result['fitness'] == 6.000000022351742


def test_weighted_composition_preserves_negative_source_weights():
    result = compose_patch_fitness(-1., -2., -3., -4., -5., True,
                                   mri_weight=-2., volume_unmri_weight=3.,
                                   quadratic_curvature_weight=-4., normal_dot_weight=5.)
    assert result == -8.


def test_double_weights_are_not_rounded_to_tp_float32():
    result = compose_patch_fitness(-1., 0., 0., 0., 0., True, mri_weight=1.00000000001)
    assert result == -1.00000000001


@pytest.mark.parametrize('weight', [0., float(np.finfo(np.float32).eps) / 2,
                                    -float(np.finfo(np.float32).eps) / 2])
def test_source_fzero_disables_small_positive_and_negative_weight(weight):
    result = compose_patch_fitness(-1., -2., -3., -4., -5., True,
                                   mri_weight=weight, volume_unmri_weight=0.,
                                   normal_dot_weight=0., quadratic_curvature_weight=0.)
    assert result == 0.


def test_source_epsilon_weight_is_active():
    epsilon = float(np.finfo(np.float32).eps)
    assert compose_patch_fitness(-1., 0., 0., 0., 0., True,
                                 mri_weight=epsilon) == -epsilon


@pytest.mark.parametrize('shape', [(5, 7, 8), (8, 5, 8), (8, 7, 5), (1, 1, 1)])
def test_source_small_local_volume_only_disables_unmri(shape):
    result = compose_patch_fitness(-1., -2., -3., -4., -5., True,
                                   defect_volume_shape=shape, return_diagnostics=True)
    assert result['fitness'] == -10.
    assert result['requested_weights']['l_unmri'] == 10.
    assert result['effective_weights']['l_unmri'] == 0.
    assert result['volume_term_enabled'] is False


def test_new_candidate_restores_requested_volume_weight_without_hidden_state():
    small = compose_patch_fitness(0., 0., 0., 0., -1., True, defect_volume_shape=(5, 6, 6))
    large = compose_patch_fitness(0., 0., 0., 0., -1., True, defect_volume_shape=(6, 6, 6))
    assert (small, large) == (0., -10.)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, [1.], 1+2j, 1e40])
def test_invalid_source_field_is_rejected(value):
    with pytest.raises(ValueError):
        compose_patch_fitness(value, 0., 0., 0., 0., True)


@pytest.mark.parametrize('shape', [(0, 6, 6), (6., 6., 6.), (6, 6), '6,6,6'])
def test_invalid_local_volume_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        compose_patch_fitness(0., 0., 0., 0., 0., True, defect_volume_shape=shape)


@pytest.mark.parametrize('values', [[np.inf], [[1., 2.]], [np.nan]])
def test_invalid_rank_vector_is_rejected(values):
    with pytest.raises(ValueError):
        rank_patch_fitness(values)


def test_empty_rank_vector_has_source_integer_type():
    result = rank_patch_fitness([])
    assert result.shape == (0,) and result.dtype == np.int32
