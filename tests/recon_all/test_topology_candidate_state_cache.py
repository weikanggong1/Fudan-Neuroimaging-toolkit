"""显式缓存完整装配的容量/动态状态/失败/别名边界。"""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from test_topology_candidate_cycle import fixture, candidate
from fnit.recon_all.topology_candidate_cycle import materialize_candidate_state, CandidateCycleEvaluator
from fnit.recon_all.topology_candidate_state_cache import prepare_candidate_state_context


def complete_candidate(**kwargs):
    output=candidate(**kwargs);g=output['geometry']
    counts=np.array([len(row) for row in g['neighbors']],np.int32)
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int32)
    g.update(neighbor_offsets=offsets,ordered_neighbors=np.array(sum(g['neighbors'],[]),np.int32))
    return output


class StateCacheTests(unittest.TestCase):
    def state(self):
        initial,snapshot,retess=fixture()
        initial['neighbor_storage'][1]=np.array([0,2,37,42],np.int32)
        initial['incident_faces'][1]=np.array([0,99],np.int32)
        initial['incident_corners'][1]=np.array([1,2],np.uint8)
        return initial,snapshot,retess

    def test_full_output_capacity_and_dynamic_attributes_match_old(self):
        initial,_,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
        old=materialize_candidate_state(initial=initial,result=result)
        new=materialize_candidate_state(initial=initial,result=result,state_context=context)
        for key in old:
            if key in ('neighbor_storage','incident_faces','incident_corners'):
                for a,b in zip(old[key],new[key]):np.testing.assert_array_equal(a,b)
            else:np.testing.assert_array_equal(old[key],new[key])
        self.assertIs(new['neighbor_storage'][1],initial['neighbor_storage'][1])
        np.testing.assert_array_equal(new['neighbor_storage'][1],[0,2,37,42])

    def test_new_buffers_independent_of_inputs_and_nonoverlapping_rows(self):
        initial,_,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
        new=materialize_candidate_state(initial=initial,result=result,state_context=context)
        result['geometry']['incident_faces'][:]=-7;result['geometry']['incident_corners'][:]=2
        np.testing.assert_array_equal(new['incident_faces'][1],[0])
        np.testing.assert_array_equal(new['incident_corners'][1],[1])
        new['incident_faces'][0][0]=42
        np.testing.assert_array_equal(new['incident_faces'][1],[0])

    def test_base_mutation_cannot_bypass_check_by_object_identity(self):
        initial,_,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        for name in ('neighbor_storage','incident_faces','incident_corners'):
            changed=copy.deepcopy(initial);changed[name][0][0]=2
            result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
            with self.subTest(name=name),self.assertRaises(ValueError):
                materialize_candidate_state(initial=changed,result=result,state_context=context)
        changed=copy.deepcopy(initial);changed['vertex_topology'][0,4]=7
        with self.assertRaises(ValueError):materialize_candidate_state(initial=changed,result=result,state_context=context)

    def test_all_candidate_rows_prefix_and_redundant_csr_are_checked(self):
        initial,_,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        for name in ('neighbors','ordered_neighbors','incident_faces','incident_corners'):
            result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
            if name=='neighbors':result['geometry'][name][0][0]=2
            else:result['geometry'][name][0]=2
            with self.subTest(name=name),self.assertRaises(ValueError):
                materialize_candidate_state(initial=initial,result=result,state_context=context)
        np.testing.assert_array_equal(initial['vertex_topology'][0],[1,1,2,2,0,0,1])

    def test_context_cannot_reopen_data_or_hide_metadata_damage(self):
        initial,_,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        with self.assertRaises(ValueError):context.neighbors.flags.writeable=True
        context.neighbor_offsets.shape=(1,len(context.neighbor_offsets))
        result=complete_candidate(surface_state=initial,retessellation_inputs=retess)
        with self.assertRaises(ValueError):materialize_candidate_state(initial=initial,result=result,state_context=context)

    def test_fractional_index_rejected_and_unsupported_topology_fails(self):
        initial,_,_=self.state();initial['neighbor_storage'][0]=[1.5,2]
        with self.assertRaises(ValueError):prepare_candidate_state_context(surface_state=initial)
        initial,_,_=self.state();initial['vertex_topology'][0,0:2]=2
        with self.assertRaises(ValueError):prepare_candidate_state_context(surface_state=initial)

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=complete_candidate)
    def test_multiple_cycles_preserve_dvs_and_dynamic_feedback(self,_):
        initial,snapshot,retess=self.state();context=prepare_candidate_state_context(surface_state=initial)
        old=CandidateCycleEvaluator(surface_state=initial,snapshot=snapshot,retessellation_inputs=retess,candidate_inputs={})
        new=CandidateCycleEvaluator(surface_state=initial,snapshot=snapshot,retessellation_inputs=retess,candidate_inputs={},state_context=context)
        for count in range(2):
            args=dict(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),
                vertex_rip=np.zeros(4,bool),use_counts=np.full(1,count,np.int32),vertex_fitness=np.zeros(1,np.float32))
            left,right=old(**args),new(**args)
            for name in left:np.testing.assert_array_equal(left[name],right[name])
            for name in ('curvbak','k1','k2','flattened_vertices','vertex_topology','original_vertices'):
                np.testing.assert_array_equal(old.surface_state[name],new.surface_state[name])


if __name__=='__main__':unittest.main()
