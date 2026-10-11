"""完整单缺陷接线的域、DVS、持久状态和随机流合同；不替代真实MRI。"""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from fnit.recon_all.gcsa_permutation import VnlRandom
from fnit.recon_all.topology_genetic_search import (TopologyRandomStream,
    source_population_limits,prepare_genetic_defect_evaluator,search_and_finalize_defect)


def inputs():
    faces=np.array([[0,1,2]],np.int32);top=np.zeros((4,7),np.int32)
    top[:3,:2]=1;top[:3,2:4]=2;top[:3,6]=1
    xyz=np.arange(12,dtype=np.float32).reshape(4,3)+1
    state=dict(original_vertices=xyz.copy(),canonical_vertices=xyz.copy(),
        vertex_normals=np.tile(np.array([0,0,1],np.float32),(4,1)),
        vertex_rip=np.zeros(4,np.int32),vertex_marks=np.zeros(4,np.int32),vertex_topology=top,
        neighbor_storage=[[1,2],[0,2],[0,1],[]],incident_faces=[[0],[0],[0],[]],
        incident_corners=[[0],[1],[2],[]],faces=faces,nfaces=1,face_area=np.zeros(1,np.float32),
        curvbak=np.array([9,9,9,2],np.float32),k1=np.zeros(4,np.float32),k2=np.zeros(4,np.float32),
        tangent_e1=np.zeros((4,3),np.float32),tangent_e2=np.zeros((4,3),np.float32),
        flattened_vertices=np.zeros((4,3),np.float32),vertex_area=np.ones(4,np.float32))
    defect=dict(edge_vertices=np.array([[0,3],[0,2],[2,3]],np.int32),edge_used=np.array([0,2,0],np.int32),
        ordering=np.arange(3,dtype=np.int32),inside_vertices=np.array([3],np.int32),
        inside_discarded=np.zeros(1,bool),border_vertices=np.array([0,2],np.int32),
        hull_vertices=np.array([0,2],np.int32),use_overlap=False,
        old_edge_table_indices=np.array([1],np.int32),segment_rows=[np.array([1],np.int32),[]])
    return state,defect


class SearchConnectionTests(unittest.TestCase):
    def test_rng_reuses_primed_stream_and_double_scaling_including_zero_width(self):
        r=TopologyRandomStream(seed=1234);base=VnlRandom(1234);base.open_ran1()
        for lo,hi in ((0.,1.),(0.,4722.9),(3.,3.),(-12.,10.)):
            expected=lo+(hi-lo)*base.open_ran1()
            self.assertEqual(r(lo,hi),expected)
        self.assertEqual(r.draws,4)
        with self.assertRaises(ValueError):r(2.,1.)
        with self.assertRaises(ValueError):r(float('nan'),1.)
        self.assertEqual(r.draws,4)
        for seed in (True,1.2,2**31,[-1]):
            with self.assertRaises(ValueError):TopologyRandomStream(seed=seed)

    def test_huge_source_limits_preserve_strict_boundaries_and_integer_division(self):
        self.assertEqual(source_population_limits(edge_count=50000,population_size=11,max_unchanged=9),
            dict(population_size=11,max_unchanged=9))
        self.assertEqual(source_population_limits(edge_count=50001,population_size=11,max_unchanged=9),
            dict(population_size=5,max_unchanged=1))
        self.assertEqual(source_population_limits(edge_count=100001,population_size=11,max_unchanged=9),
            dict(population_size=10,max_unchanged=1))
        self.assertEqual(source_population_limits(edge_count=200001,population_size=11,max_unchanged=9),
            dict(population_size=3,max_unchanged=1))
        with self.assertRaises(ValueError):source_population_limits(edge_count=True,population_size=10,max_unchanged=10)

    def test_ETable_groups_map_to_ES_positions_and_keep_empty_source_slots(self):
        state,defect=inputs();defect.update(old_edge_table_indices=np.array([2,0],np.int32),
            segment_rows=[np.array([0],np.int32),[],np.array([2],np.int32)])
        out=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=defect,candidate_inputs={})
        np.testing.assert_array_equal(out['segment_edge_numbers'],[2,0])
        for actual,expected in zip(out['segment_rows'],([1],[],[0])):np.testing.assert_array_equal(actual,expected)
        self.assertEqual(out['overlap_policy'],'disabled')
        self.assertIsNone(out['evaluator'].retessellation_inputs['overlaps'])
        self.assertFalse(out['overlap_arguments']['use_overlap'])

    def test_DVS_records_current_corrected_geometry_and_immutable_cache_without_alias(self):
        state,defect=inputs();before=copy.deepcopy(state)
        out=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=defect,candidate_inputs={})
        callback=out['evaluator'];r=callback.retessellation_inputs
        np.testing.assert_array_equal(callback.snapshot.vertex_numbers,[3,0,2])
        self.assertFalse(r['base_faces'].flags.writeable)
        state['canonical_vertices'][:]+=17
        np.testing.assert_array_equal(r['canonical'],before['canonical_vertices'])
        self.assertEqual(callback.snapshot.nfaces,1)
        np.testing.assert_array_equal(callback.snapshot.original_vertices,before['original_vertices'][[3,0,2]])

    def test_true_overlap_is_not_replaced_with_disabled_empty_rows(self):
        state,defect=inputs();defect.update(use_overlap=True,overlap_offsets=np.array([0,1,1,2],np.int64),
            overlap_values=np.array([2,0],np.int32),overlap_complete=np.array([False,True,True]),
            overlaps=[np.array([2],np.int32),np.empty(0,np.int32),np.array([0],np.int32)])
        out=prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=defect,candidate_inputs={})
        self.assertEqual(out['overlap_policy'],'computed')
        np.testing.assert_array_equal(out['overlap_arguments']['overlap_list_incomplete'],[True,False,False])
        np.testing.assert_array_equal(out['evaluator'].retessellation_inputs['overlaps'][0],[2])

    def test_invalid_group_outside_old_ES_fails_before_evaluation(self):
        state,defect=inputs();defect['segment_rows']=[[0]]
        with self.assertRaisesRegex(ValueError,'outside the old ES'):
            prepare_genetic_defect_evaluator(surface_state=state,defect_inputs=defect,candidate_inputs={})

    def test_search_tail_elimination_rip_and_persistent_fields_reach_finalizer(self):
        state,defect=inputs();before=copy.deepcopy(state);received={}
        def search(**kw):
            callback=kw['evaluate_patch'];callback.surface_state=dict(callback.surface_state)
            callback.surface_state['curvbak']=callback.surface_state['curvbak']+np.float32(7)
            # Last elimination follows evaluation; source rip is not the
            # callback's older flags, and the finalizer must receive it.
            return dict(vertex_rip=np.array([-1,0,0,1],np.int32),best_ordering=np.array([2,0,1],np.int32),
                best_discarded=np.array([True]),best_fitness=-23.,use_counts=np.array([19],np.int32),
                vertex_fitness=np.array([3],np.float32),deletion_count=4,elimination_threshold=np.float32(2))
        def final(**kw):
            received.update(kw);return dict(surface_state=kw['surface_state'],fitness=-24.)
        with patch('fnit.recon_all.topology_genetic_search.run_population_search',side_effect=search), \
             patch('fnit.recon_all.topology_genetic_search.finalize_defect_patch',side_effect=final):
            out=search_and_finalize_defect(surface_state=state,defect_inputs=defect,candidate_inputs={},
                draw_random_number=TopologyRandomStream(seed=1234),population_size=10,max_unchanged=10,niters=-1,
                initial_selection=True,vertex_eliminate=True,deletion_count=1,elimination_threshold=np.float32(4),
                check_surface_intersection=False,intersection_checker=None)
        np.testing.assert_array_equal(received['surface_state']['vertex_rip'],[-1,0,0,1])
        np.testing.assert_array_equal(received['surface_state']['curvbak'],before['curvbak']+7)
        self.assertEqual(received['best_fitness'],-23.)
        np.testing.assert_array_equal(received['best_discarded'],[True])
        self.assertEqual(out['deletion_count'],4);self.assertEqual(out['finalization']['fitness'],-24.)
        self.assertFalse(out['complete_ga'])
        for k in ('vertex_rip','curvbak','faces'):np.testing.assert_array_equal(state[k],before[k])


if __name__=='__main__':unittest.main()
