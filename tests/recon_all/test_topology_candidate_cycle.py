"""跨候选状态、选择性恢复及失败事务；不代替真实几何评分基准。"""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.topology_candidate_cycle import evaluate_candidate_cycle, CandidateCycleEvaluator, materialize_candidate_state
from fnit.recon_all.topology_candidate_state import record_candidate_state
from fnit.recon_all.topology_candidate_geometry import candidate_incidents


def fixture():
    faces=np.array([[0,1,2]],np.int32)
    top=np.zeros((4,7),np.int32);top[:3,:2]=1;top[:3,2:4]=2;top[:3,6]=1
    state=dict(original_vertices=np.arange(12,dtype=np.float32).reshape(4,3),
        vertex_normals=np.tile(np.array([0,0,1],np.float32),(4,1)),
        vertex_rip=np.zeros(4,bool),vertex_marks=np.zeros(4,np.int32),vertex_topology=top,
        neighbor_storage=[[1,2],[0,2],[0,1],[]],incident_faces=[[0],[0],[0],[]],
        incident_corners=[[0],[1],[2],[]],faces=faces,nfaces=1,face_area=np.zeros(1,np.float32),
        curvbak=np.array([9,9,9,2],np.float32),k1=np.zeros(4,np.float32),k2=np.zeros(4,np.float32),
        tangent_e1=np.zeros((4,3),np.float32),tangent_e2=np.zeros((4,3),np.float32),
        flattened_vertices=np.zeros((4,3),np.float32),vertex_area=np.ones(4,np.float32))
    snap=record_candidate_state(surface_state=state,defect_vertices=[3],
        convex_hull_vertices=[0,2],border_vertices=[0,2],vertex_translation=np.arange(4))
    retess=dict(base_faces=faces,candidate_edges=np.array([[0,3],[2,3]],np.int32),
        edge_used=np.zeros(2,np.int32),ordering=np.arange(2),inside_vertices=np.array([3],np.int32),
        inside_discarded=np.zeros(1,bool),border_vertices=np.array([0,2],np.int32),mode='all')
    return state,snap,retess


def candidate(*,surface_state,retessellation_inputs,**kwargs):
    state=surface_state;faces=np.concatenate((state['faces'],[[0,2,3]])).astype(np.int32)
    incidence=candidate_incidents(faces=faces,vertex_count=4)
    xyz=state['original_vertices'].copy();xyz[[0,2,3]]+=1
    normal=state['vertex_normals'].copy();normal[[0,2,3]]+=2
    sff={key:state[key].copy()+1 for key in ['k1','k2','tangent_e1','tangent_e2']}
    feedback=state['curvbak'].copy();feedback[3]+=1
    return dict(geometry=dict(original_vertices=xyz,faces=faces,neighbors=[[1,2,3],[0,2],[0,1,3],[0,2]],
        removed_vertices=np.empty(0,np.int32),marks=np.zeros(4,np.int32),edge_used=np.array([3,4],np.int32),**incidence),
        vertex_normals=normal,face_area=np.zeros(2,np.float32),
        flattened_vertices=state['flattened_vertices'].copy()+3,vertex_curvbak=feedback,
        sff=sff,fitness=dict(composition=dict(fitness=-17.)))


class CandidateCycleTests(unittest.TestCase):
    def run_cycle(self,state=None):
        original,snap,r=fixture()
        return evaluate_candidate_cycle(surface_state=original if state is None else state,snapshot=snap,
            retessellation_inputs=r,candidate_inputs={},use_counts=np.zeros(1,np.int32),
            vertex_fitness=np.zeros(1,np.float32))

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=candidate)
    def test_source_restore_fields_and_edge_reset_preserve_feedback(self,_):
        original,_,_=fixture();before=copy.deepcopy(original);out=self.run_cycle(original)
        state=out['surface_state'];self.assertEqual(state['nfaces'],1)
        np.testing.assert_array_equal(state['original_vertices'],original['original_vertices'])
        np.testing.assert_array_equal(state['vertex_normals'],original['vertex_normals'])
        np.testing.assert_array_equal(state['vertex_topology'],original['vertex_topology'])
        self.assertEqual(state['faces'].shape,(2,3))  # capacity not restored
        np.testing.assert_array_equal(out['edge_used'],[0,2])
        np.testing.assert_array_equal(state['k1'],np.ones(4,np.float32))
        np.testing.assert_array_equal(state['flattened_vertices'],np.full((4,3),3,np.float32))
        self.assertEqual(state['curvbak'][3],3);self.assertEqual(out['use_counts'][0],1)
        self.assertEqual(out['vertex_fitness'][0],3)
        for name in ['original_vertices','k1','curvbak','vertex_topology','faces']:
            np.testing.assert_array_equal(original[name],before[name])

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=candidate)
    def test_two_calls_reuse_K_fx_curvbak_RP_not_initial_reset(self,_):
        state,snap,r=fixture();callback=CandidateCycleEvaluator(surface_state=state,snapshot=snap,
            retessellation_inputs=r,candidate_inputs={})
        first=callback(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),
            vertex_rip=np.zeros(4,bool),use_counts=np.zeros(1,np.int32),vertex_fitness=np.zeros(1,np.float32))
        second=callback(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),
            vertex_rip=first['vertex_rip'],use_counts=first['use_counts'],vertex_fitness=first['vertex_fitness'])
        self.assertEqual(second['use_counts'][0],2);self.assertEqual(second['vertex_fitness'][0],3.5)
        self.assertEqual(callback.surface_state['curvbak'][3],4)
        np.testing.assert_array_equal(callback.surface_state['k1'],np.full(4,2,np.float32))
        np.testing.assert_array_equal(callback.surface_state['flattened_vertices'],np.full((4,3),6,np.float32))

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=candidate)
    def test_outer_discard_is_used_next_call_not_cached(self,mock):
        state,snap,r=fixture();callback=CandidateCycleEvaluator(surface_state=state,snapshot=snap,
            retessellation_inputs=r,candidate_inputs={})
        rip=np.array([False,False,False,True])
        out=callback(ordering=np.arange(2),discarded_defect_vertices=np.ones(1,bool),
            vertex_rip=rip,use_counts=np.zeros(1,np.int32),vertex_fitness=np.zeros(1,np.float32))
        self.assertEqual(out['use_counts'][0],0);self.assertTrue(out['vertex_rip'][3])
        np.testing.assert_array_equal(mock.call_args.kwargs['retessellation_inputs']['inside_discarded'],[True])
        self.assertFalse(r['inside_discarded'][0])

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=candidate)
    def test_signed_rip_storage_survives_bool_nonzero_engine_predicate(self,mock):
        state,snap,r=fixture();callback=CandidateCycleEvaluator(surface_state=state,snapshot=snap,
            retessellation_inputs=r,candidate_inputs={})
        rip=np.array([-1,0,2,0],np.int32)
        out=callback(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),
            vertex_rip=rip,use_counts=np.zeros(1,np.int32),vertex_fitness=np.zeros(1,np.float32))
        np.testing.assert_array_equal(out['vertex_rip'],rip)
        self.assertEqual(out['vertex_rip'].dtype,np.int32)
        np.testing.assert_array_equal(mock.call_args.kwargs['retessellation_inputs']['vertex_rip'],[True,False,True,False])
        np.testing.assert_array_equal(rip,[-1,0,2,0])

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate',side_effect=RuntimeError('stage failed'))
    def test_failure_keeps_callback_previous_state_and_statistics(self,_):
        state,snap,r=fixture();callback=CandidateCycleEvaluator(surface_state=state,snapshot=snap,
            retessellation_inputs=r,candidate_inputs={});counts=np.array([7],np.int32)
        with self.assertRaises(RuntimeError):callback(ordering=np.arange(2),discarded_defect_vertices=np.zeros(1,bool),
            vertex_rip=np.zeros(4,bool),use_counts=counts,vertex_fitness=np.zeros(1,np.float32))
        self.assertIsNone(callback.last_result);self.assertEqual(counts[0],7)
        np.testing.assert_array_equal(callback.surface_state['k1'],state['k1'])

    def test_materializer_preserves_empty_fit_scalars_and_source_signed_face_area(self):
        state,_,r=fixture()
        state.update(H=np.array([17,18,19,20],np.float32),K=np.array([21,22,23,24],np.float32),
                     face_rip=np.array([-1],np.int32),face_area=np.array([-3],np.float32))
        result=candidate(surface_state=state,retessellation_inputs=r)
        result['geometry'].update(selected_vertices=np.array([0,2,3],np.int32),
                                  face_normals=np.array([[0,0,1],[0,0,-1]],np.float32))
        result['sff'].update(fit_kind=np.array([1,0,2],np.int32),
                            k1=np.array([3,5,7,11],np.float32),k2=np.array([-2,6,8,13],np.float32))
        result['face_area']=np.array([-3,0],np.float32)
        out=materialize_candidate_state(initial=state,result=result)
        np.testing.assert_array_equal(out['H'],[.5,18,19,12])
        np.testing.assert_array_equal(out['K'],[-6,22,23,143])
        np.testing.assert_array_equal(out['face_rip'],[-1,0])
        np.testing.assert_array_equal(out['face_area'],[-3,0])
        np.testing.assert_array_equal(out['face_normals'],result['geometry']['face_normals'])
        np.testing.assert_array_equal(state['H'],[17,18,19,20])
        # Final materialization may reuse persisted K/E after a final match.
        # Without an SFF fit marker it must keep those existing H/K fields.
        del result['sff']['fit_kind']
        final=materialize_candidate_state(initial=state,result=result)
        np.testing.assert_array_equal(final['H'],state['H'])
        np.testing.assert_array_equal(final['K'],state['K'])

    def test_unverified_multiring_and_some_mode_fail_before_algorithm(self):
        state,snap,r=fixture();state['vertex_topology'][0,1]=2
        with self.assertRaisesRegex(ValueError,'nsize'):self.run_cycle(state)
        state,snap,r=fixture();r['mode']='some'
        with self.assertRaisesRegex(ValueError,'all-mode'):evaluate_candidate_cycle(surface_state=state,
            snapshot=snap,retessellation_inputs=r,candidate_inputs={},use_counts=np.zeros(1,np.int32),vertex_fitness=np.zeros(1,np.float32))

    @patch('fnit.recon_all.topology_candidate_cycle.evaluate_candidate')
    def test_nonappend_adjacency_rejects_no_partial_publication(self,mock):
        def bad(**kw):
            out=candidate(**kw);out['geometry']['neighbors'][0]=[2,1,3];return out
        mock.side_effect=bad
        with self.assertRaisesRegex(ValueError,'append only'):self.run_cycle()

    def test_materializer_preserves_unchanged_neighbor_capacity_tail_and_copies_incidence(self):
        state,_,r=fixture()
        state['neighbor_storage'][1]=np.array([0,2,37,42],np.int32)
        state['incident_faces'][1]=np.array([0,99],np.int32)
        state['incident_corners'][1]=np.array([1,2],np.uint8)
        result=candidate(surface_state=state,retessellation_inputs=r)
        out=materialize_candidate_state(initial=state,result=result)
        self.assertIs(out['neighbor_storage'][1],state['neighbor_storage'][1])
        np.testing.assert_array_equal(out['neighbor_storage'][1],[0,2,37,42])
        np.testing.assert_array_equal(out['incident_faces'][1],[0])
        np.testing.assert_array_equal(out['incident_corners'][1],[1])
        result['geometry']['incident_faces'][:]=-7
        result['geometry']['incident_corners'][:]=2
        np.testing.assert_array_equal(out['incident_faces'][1],[0])
        np.testing.assert_array_equal(out['incident_corners'][1],[1])
        np.testing.assert_array_equal(state['incident_faces'][1],[0,99])

    def test_all_rows_validate_append_only_incidence_and_corner_prefix(self):
        state,_,r=fixture()
        for key in ('incident_faces','incident_corners'):
            result=candidate(surface_state=state,retessellation_inputs=r)
            first=int(result['geometry']['incident_offsets'][1])
            result['geometry'][key][first]=2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'append-only'):
                materialize_candidate_state(initial=state,result=result)
        np.testing.assert_array_equal(state['vertex_topology'][1],[1,1,2,2,0,0,1])


if __name__=='__main__':unittest.main()
