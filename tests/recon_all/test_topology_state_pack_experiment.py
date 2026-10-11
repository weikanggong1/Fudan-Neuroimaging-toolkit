"""真实benchmark前的状态别名、容量、动态反馈、全域失败合同。"""
import copy
import numpy as np
import pytest
from test_topology_candidate_state_cache import complete_candidate
from test_topology_candidate_cycle import fixture
from fnit.recon_all import topology_candidate_cycle as cycle
from recon_topology_state_pack_experiment import materialize_packed,state_pack_scope


def compare(left,right):
    assert set(left)==set(right)
    for k in left:
        if k in ('neighbor_storage','incident_faces','incident_corners'):
            assert len(left[k])==len(right[k])
            for a,b in zip(left[k],right[k]):np.testing.assert_array_equal(a,b)
        else:np.testing.assert_array_equal(left[k],right[k])


def state():
    initial,snapshot,r=fixture()
    initial['neighbor_storage'][1]=np.array([0,2,37,42],np.int32)
    initial['incident_faces'][1]=np.array([0,99],np.int32)
    initial['incident_corners'][1]=np.array([1,2],np.uint8)
    return initial,snapshot,r


def test_full_fields_capacity_tail_and_unchanged_neighbor_identity():
    initial,_,r=state();result=complete_candidate(surface_state=initial,retessellation_inputs=r)
    actual=materialize_packed(initial=initial,result=result);compare(cycle.materialize_candidate_state(initial=initial,result=result),actual)
    assert actual['neighbor_storage'][1]is initial['neighbor_storage'][1]
    np.testing.assert_array_equal(actual['neighbor_storage'][1],[0,2,37,42])


def test_new_buffers_input_isolation_and_rows_do_not_overlap():
    initial,_,r=state();result=complete_candidate(surface_state=initial,retessellation_inputs=r)
    actual=materialize_packed(initial=initial,result=result)
    result['geometry']['incident_faces'][:]=-5;result['geometry']['incident_corners'][:]=2
    np.testing.assert_array_equal(actual['incident_faces'][1],[0])
    actual['incident_faces'][0][0]=42;np.testing.assert_array_equal(actual['incident_faces'][1],[0])
    actual['neighbor_storage'][0][0]=9;assert initial['neighbor_storage'][0][0]==1


@pytest.mark.parametrize('field',['neighbor_storage','incident_faces','incident_corners'])
def test_current_mutable_base_prefix_is_checked_each_call(field):
    initial,_,r=state();result=complete_candidate(surface_state=initial,retessellation_inputs=r)
    materialize_packed(initial=initial,result=result)
    initial[field][0][0]=2
    with pytest.raises(ValueError,match='prefix'):materialize_packed(initial=initial,result=result)


@pytest.mark.parametrize('field',['neighbors','ordered_neighbors','incident_faces','incident_corners'])
def test_every_candidate_prefix_and_redundant_CSR_checked(field):
    initial,_,r=state();result=complete_candidate(surface_state=initial,retessellation_inputs=r);before=copy.deepcopy(initial)
    if field=='neighbors':result['geometry'][field][0][0]=2
    else:result['geometry'][field][0]=2
    with pytest.raises(ValueError):materialize_packed(initial=initial,result=result)
    compare(initial,before)


def test_scope_identity_restored_on_exception():
    original=cycle._candidate_state_validated
    with pytest.raises(RuntimeError):
        with state_pack_scope(True):
            assert cycle._candidate_state_validated is materialize_packed
            raise RuntimeError('caller failure')
    assert cycle._candidate_state_validated is original


def test_two_cycles_preserve_dynamic_feedback_signed_rip_and_statistics(monkeypatch):
    monkeypatch.setattr(cycle,'evaluate_candidate',complete_candidate)
    initial,snapshot,r=state();a=cycle.CandidateCycleEvaluator(surface_state=initial,snapshot=snapshot,retessellation_inputs=r,candidate_inputs={});b=cycle.CandidateCycleEvaluator(surface_state=initial,snapshot=snapshot,retessellation_inputs=r,candidate_inputs={})
    for i in range(2):
        options=dict(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),vertex_rip=np.array([-1,0,2,0],np.int32),use_counts=np.full(1,i,np.int32),vertex_fitness=np.zeros(1,np.float32))
        left=a(**options)
        with state_pack_scope(True):right=b(**options)
        compare(a.surface_state,b.surface_state)
        for k in left:np.testing.assert_array_equal(left[k],right[k])
