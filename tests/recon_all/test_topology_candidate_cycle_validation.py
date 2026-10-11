"""同次闭环复用校验；独立入口及下一候选仍检查真实状态。"""
from unittest.mock import patch

import numpy as np
import pytest

from fnit.recon_all import topology_candidate_cycle as cycle
from test_topology_candidate_cycle import candidate, fixture


def run_once():
    state, snapshot, retess = fixture()
    return cycle.evaluate_candidate_cycle(
        surface_state=state, snapshot=snapshot,
        retessellation_inputs=retess, candidate_inputs={},
        use_counts=np.zeros(1, np.int32), vertex_fitness=np.zeros(1, np.float32))


def assert_same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            if key != 'seconds':
                assert_same(a[key], b[key])
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_same(first, second)
    else:
        assert a == b


def test_same_cycle_reuses_only_current_validation_and_preserves_full_output():
    original = cycle._one_ring_state
    with patch.object(cycle, 'evaluate_candidate', side_effect=candidate), \
            patch.object(cycle, '_one_ring_state', wraps=original) as check:
        fast = run_once()
        assert check.call_count == 1
        core = cycle._candidate_state_validated
        def full_check(**kwargs):
            # 对照路径在装配时再次完整校验，使用相同几何/评分和写回。
            return core(initial=kwargs['initial'], result=kwargs['result'],
                        validated=original(kwargs['initial']))
        with patch.object(cycle, '_candidate_state_validated', side_effect=full_check):
            independently_checked = run_once()
    assert_same(fast, independently_checked)


def test_next_candidate_revalidates_mutated_row_without_publishing_feedback():
    state, snapshot, retess = fixture()
    evaluator = cycle.CandidateCycleEvaluator(
        surface_state=state, snapshot=snapshot,
        retessellation_inputs=retess, candidate_inputs={})
    args = dict(ordering=np.arange(2), discarded_defect_vertices=np.zeros(1, bool),
                vertex_rip=np.zeros(4, bool), use_counts=np.zeros(1, np.int32),
                vertex_fitness=np.zeros(1, np.float32))
    with patch.object(cycle, 'evaluate_candidate', side_effect=candidate) as score:
        first = evaluator(**args)
        previous = evaluator.last_result
        evaluator.surface_state['incident_corners'][1] = np.empty(0, np.uint8)
        args.update(use_counts=first['use_counts'], vertex_fitness=first['vertex_fitness'])
        with pytest.raises(ValueError, match='incident corners'):
            evaluator(**args)
        assert score.call_count == 1
        assert evaluator.last_result is previous
        assert args['use_counts'][0] == 1


def test_independent_public_materialization_still_rejects_invalid_capacity():
    state, _, retess = fixture()
    result = candidate(surface_state=state, retessellation_inputs=retess)
    state['neighbor_storage'][0] = []
    with pytest.raises(ValueError, match='neighbors'):
        cycle.materialize_candidate_state(initial=state, result=result)


def test_candidate_order_validation_remains_after_initial_check():
    def nonappend(**kwargs):
        result = candidate(**kwargs)
        result['geometry']['neighbors'][0] = [2, 1, 3]
        return result
    with patch.object(cycle, 'evaluate_candidate', side_effect=nonappend):
        with pytest.raises(ValueError, match='append only'):
            run_once()
